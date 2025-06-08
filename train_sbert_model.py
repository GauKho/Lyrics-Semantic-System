from data_loader_multiplerankingloss import load_and_prepare_data, create_training_pairs, create_data_splits, save_examples, _normalize_text_improved
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import wandb
import hashlib
import pandas as pd
import numpy as np
import json
import glob
import os
import logging
import torch
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "base_model": "all-MiniLM-L6-v2",
    "epochs": 4,
    "learning_rate": 1e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "negative_ratio": 2.5,
    "max_segments_per_song": 4,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "min_lyric_words": 50,
    "max_segments_per_song": 4
}

class DetailedInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """Enhanced IR evaluator that captures detailed metrics for wandb logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detailed_scores = {}
        self.last_scores = {}
    
    def compute_metrices(self, model, corpus_embeddings=None):
        """Override to capture detailed metrics during computation."""
        # Call parent method to get all metrics
        results = super().compute_metrices(model, corpus_embeddings)
        
        # Store the computed metrics for later access
        self.last_scores = {}
        
        # Extract MRR metrics
        for k in self.mrr_at_k:
            metric_name = f"mrr@{k}"
            if hasattr(self, f'mrr_at_{k}'):
                self.last_scores[metric_name] = getattr(self, f'mrr_at_{k}')
        
        # Extract MAP metrics
        for k in self.map_at_k:
            metric_name = f"map@{k}"
            if hasattr(self, f'map_at_{k}'):
                self.last_scores[metric_name] = getattr(self, f'map_at_{k}')
        
        # Extract NDCG metrics
        for k in self.ndcg_at_k:
            metric_name = f"ndcg@{k}"
            if hasattr(self, f'ndcg_at_{k}'):
                self.last_scores[metric_name] = getattr(self, f'ndcg_at_{k}')
        
        # Extract Precision and Recall metrics
        for k in self.precision_recall_at_k:
            precision_name = f"precision@{k}"
            recall_name = f"recall@{k}"
            if hasattr(self, f'precision_at_{k}'):
                self.last_scores[precision_name] = getattr(self, f'precision_at_{k}')
            if hasattr(self, f'recall_at_{k}'):
                self.last_scores[recall_name] = getattr(self, f'recall_at_{k}')
        
        # Store in detailed_scores for external access
        self.detailed_scores = self.last_scores.copy()
        
        return results
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """Override to ensure detailed metrics are captured."""
        try:
            # Compute metrics first
            self.compute_metrices(model)
            
            # Get the main score using parent's method
            main_score = super(InformationRetrievalEvaluator, self).__call__(
                model, output_path, epoch, steps
            )
            
            return main_score
            
        except Exception as e:
            logger.error(f"Error in DetailedInformationRetrievalEvaluator: {e}")
            return 0.0

def build_ir_evaluator(dev_df, max_queries=1500, name_suffix=""):
    """Build IR evaluator with improved data validation."""
    if len(dev_df) == 0:
        logger.warning(f"Empty dev_df provided to build_ir_evaluator{name_suffix}")
        return None
    
    queries = {}
    corpus = {}
    relevant_docs = defaultdict(set)

    # Sample and validate data
    sample_size = min(max_queries, len(dev_df))
    dev_sample = dev_df.sample(n=sample_size, random_state=42)
    
    valid_count = 0
    for idx, (_, row) in enumerate(dev_sample.iterrows()):
        try:
            # Validate required fields
            if pd.isna(row["Lyric"]) or pd.isna(row["Title"]) or pd.isna(row["Artist"]):
                continue
                
            query_id = f"q{valid_count}"
            doc_id = f"d{valid_count}"

            query = str(row["Lyric"]).strip().lower()
            title = str(row["Title"]).strip().lower()
            album = str(row["Album"]).strip().lower() if pd.notna(row["Album"]) else "unknown"
            artist = str(row["Artist"]).strip().lower()

            # Skip if any essential field is empty
            if not query or not title or not artist:
                continue
                
            # Skip very short lyrics
            if len(query.split()) <= 50:
                continue

            meta = f"{title} {artist} {album}".strip()

            queries[query_id] = query
            corpus[doc_id] = meta
            relevant_docs[query_id].add(doc_id)
            
            valid_count += 1
            
        except Exception as e:
            logger.warning(f"Error processing row {idx} in evaluator: {e}")
            continue

    if valid_count == 0:
        logger.error(f"No valid queries created for evaluator{name_suffix}")
        return None
        
    logger.info(f"Created {name_suffix} evaluator with {valid_count} queries")
    
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        name=f"lyrics-ir-eval{name_suffix}",
        mrr_at_k=[1, 3, 5, 10],
        map_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10]
    )
    return evaluator

class WandbCallback:
    """Custom callback for comprehensive wandb logging."""
    
    def __init__(self, val_evaluator=None, test_evaluator=None, log_every_n_steps= 100, evaluate_test_every_n_steps=500):
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.best_val_score = -1
        self.best_test_score = -1
        self.step_count = 0
        self.log_every_n_steps = log_every_n_steps
        self.evaluate_test_every_n_steps = evaluate_test_every_n_steps
        self.training_losses = []
        self.current_epoch = 0
        
        self.train_history = {"scores": [], "steps": [], "epochs": []}
        self.val_history = {"scores": [], "steps": [], "epochs": []}
        self.test_history = {"scores": [], "steps": [], "epochs": []}

    def log_training_metrics(self, loss_value, epoch, steps):
        """Log training metrics including score approximation."""
        # Ensure we have valid step count
        effective_steps = max(0, steps) if steps >= 0 else self.step_count
        
        # For training, we use negative loss as a proxy for "score"
        train_score = -loss_value if loss_value is not None else 0.0
        
        # Update training history
        self.train_history["scores"].append(train_score)
        self.train_history["steps"].append(effective_steps)
        self.train_history["epochs"].append(epoch)
        
        # Log comprehensive training metrics
        train_metrics = {
            "train/score": train_score,
            "train/loss": loss_value if loss_value is not None else 0.0,
            "train/epoch": epoch,
            "train/steps": effective_steps,
            "train/progress": effective_steps / (effective_steps + 1),
            "train/score_trend": np.mean(self.train_history["scores"][-10:]) if len(self.train_history["scores"]) >= 10 else train_score
        }
        
        wandb.log(train_metrics, step=effective_steps)

    def __call__(self, score, epoch, steps):
        """Callback function called during training."""
        # Handle negative step values properly
        if steps < 0:
            # Use current step count or increment from last known step
            if self.step_count > 0:
                steps = self.step_count
            else:
                # This might be an initial evaluation, use 0
                steps = 0
                
        # Update our step counter
        self.step_count = max(self.step_count, steps)
        
        # Update current epoch
        self.current_epoch = max(self.current_epoch, epoch)
        
        # Always log validation metrics if we have a score and evaluator
        if self.val_evaluator and score is not None:
            # Update validation history
            self.val_history["scores"].append(score)
            self.val_history["steps"].append(steps)
            self.val_history["epochs"].append(epoch)
            
            # Basic validation metrics
            metrics_to_log = {
                "val/score": score,
                "val/main_score": score,
                "val/epoch": epoch,
                "val/steps": steps,
                "val/is_best": score > self.best_val_score,
                "val/score_trend": np.mean(self.val_history["scores"][-5:]) if len(self.val_history["scores"]) >= 5 else score
            }
            
            # Log detailed IR metrics if available
            if hasattr(self.val_evaluator, 'detailed_scores') and self.val_evaluator.detailed_scores:
                for metric_name, metric_value in self.val_evaluator.detailed_scores.items():
                    metrics_to_log[f"val/{metric_name}"] = metric_value
                
                # Log summary statistics
                all_values = list(self.val_evaluator.detailed_scores.values())
                if all_values:
                    metrics_to_log.update({
                        "val/metrics_mean": np.mean(all_values),
                        "val/metrics_std": np.std(all_values),
                        "val/metrics_min": np.min(all_values),
                        "val/metrics_max": np.max(all_values)
                    })
                
                logger.info(f"Validation detailed metrics at step {steps}: {self.val_evaluator.detailed_scores}")
            
            wandb.log(metrics_to_log, step=steps)
            
            # Update best validation score
            if score > self.best_val_score:
                self.best_val_score = score
                wandb.log({
                    "val/best_score": self.best_val_score,
                    "val/best_score_step": steps,
                    "val/best_score_epoch": epoch
                }, step=steps)
                logger.info(f"New best validation score: {self.best_val_score} at step {steps}")
        
        # Evaluate on test set periodically (but not when step is -1 or 0)
        if (self.test_evaluator and 
            steps > 0 and 
            steps % self.evaluate_test_every_n_steps == 0):
            self.evaluate_test_set(epoch, steps)
    
    def evaluate_test_set(self, epoch, steps, model=None):
        """Evaluate on test set and log metrics."""
        if not self.test_evaluator:
            return None
            
        # Don't evaluate on invalid steps
        if steps < 0:
            return None
            
        try:
            logger.info(f"Evaluating test set at step {steps}, epoch {epoch}")
            
            # Get model from evaluator if not provided
            if model is None and hasattr(self.test_evaluator, '_model'):
                model = self.test_evaluator._model
            
            test_score = self.test_evaluator(model) if model else 0.0
            
            # Update test history
            self.test_history["scores"].append(test_score)
            self.test_history["steps"].append(steps)
            self.test_history["epochs"].append(epoch)
            
            # Log test metrics
            test_metrics = {
                "test/score": test_score,
                "test/main_score": test_score,
                "test/epoch": epoch,
                "test/steps": steps,
                "test/is_best": test_score > self.best_test_score,
                "test/score_trend": np.mean(self.test_history["scores"][-3:]) if len(self.test_history["scores"]) >= 3 else test_score
            }
            
            # Log detailed test IR metrics if available
            if hasattr(self.test_evaluator, 'detailed_scores') and self.test_evaluator.detailed_scores:
                for metric_name, metric_value in self.test_evaluator.detailed_scores.items():
                    test_metrics[f"test/{metric_name}"] = metric_value
                
                # Log summary statistics
                all_values = list(self.test_evaluator.detailed_scores.values())
                if all_values:
                    test_metrics.update({
                        "test/metrics_mean": np.mean(all_values),
                        "test/metrics_std": np.std(all_values),
                        "test/metrics_min": np.min(all_values),
                        "test/metrics_max": np.max(all_values)
                    })
                
                logger.info(f"Test detailed metrics at step {steps}: {self.test_evaluator.detailed_scores}")
            
            wandb.log(test_metrics, step=steps)
            
            # Update best test score
            if test_score > self.best_test_score:
                self.best_test_score = test_score
                wandb.log({
                    "test/best_score": self.best_test_score,
                    "test/best_score_step": steps,
                    "test/best_score_epoch": epoch
                }, step=steps)
                logger.info(f"New best test score: {self.best_test_score} at step {steps}")
            
            return test_score
            
        except Exception as e:
            logger.error(f"Error evaluating test set: {e}")
            wandb.log({"test/evaluation_error": str(e)}, step=steps)
            return None
    
    def log_epoch_summary(self, epoch, steps):
        """Log summary metrics for the completed epoch."""
        # Ensure valid step count
        effective_steps = max(0, steps) if steps >= 0 else self.step_count
        
        summary_metrics = {
            "epoch_summary/epoch": epoch,
            "epoch_summary/steps": effective_steps,
            "epoch_summary/train_scores_count": len(self.train_history["scores"]),
            "epoch_summary/val_scores_count": len(self.val_history["scores"]),
            "epoch_summary/test_scores_count": len(self.test_history["scores"])
        }
        
        # Add latest scores if available
        if self.train_history["scores"]:
            summary_metrics["epoch_summary/latest_train_score"] = self.train_history["scores"][-1]
        if self.val_history["scores"]:
            summary_metrics["epoch_summary/latest_val_score"] = self.val_history["scores"][-1]
        if self.test_history["scores"]:
            summary_metrics["epoch_summary/latest_test_score"] = self.test_history["scores"][-1]
            
        wandb.log(summary_metrics, step=effective_steps)

def log_training_loss(loss_value, step, epoch, callback=None):
    """Log training loss to wandb with comprehensive metrics."""
    # Handle negative step values
    effective_step = max(0, step) if step >= 0 else (callback.step_count if callback else 0)
    
    train_metrics = {
        "train/loss": loss_value,
        "train/loss_step": effective_step,
        "train/epoch": epoch
    }
    
    # If callback is provided, use its method for comprehensive logging
    if callback and hasattr(callback, 'log_training_metrics'):
        callback.log_training_metrics(loss_value, epoch, effective_step)
    else:
        wandb.log(train_metrics, step=effective_step)


def evaluate_model_with_metrics(model, evaluator, step, prefix="test"):
    """Evaluate model on test set and log results."""
    if evaluator is None:
        logger.warning(f"No {prefix} evaluator available")
        return None
        
    # Handle negative step values
    effective_step = max(0, step) if step >= 0 else 0
    
    logger.info(f"Evaluating on {prefix} set at step {effective_step}...")

    try: 
        #Run evaluation
        main_score = evaluator(model)
        
        # Prepare metrics to log
        metrics_to_log = {
            f"{prefix}/main_score": main_score,
            f"{prefix}/evaluation_step": effective_step
        }
        
        # Log detailed metrics if available
        if hasattr(evaluator, 'detailed_scores') and evaluator.detailed_scores:
            for metric_name, metric_value in evaluator.detailed_scores.items():
                metrics_to_log[f"{prefix}/{metric_name}"] = metric_value
            
            # Log summary statistics
            all_values = list(evaluator.detailed_scores.values())
            if all_values:
                metrics_to_log.update({
                    f"{prefix}/metrics_mean": np.mean(all_values),
                    f"{prefix}/metrics_std": np.std(all_values),
                    f"{prefix}/metrics_min": np.min(all_values),
                    f"{prefix}/metrics_max": np.max(all_values)
                })
            
            logger.info(f"{prefix.capitalize()} detailed metrics: {evaluator.detailed_scores}")
            
            # Create a detailed metrics table for wandb
            metrics_data = []
            for metric_name, metric_value in evaluator.detailed_scores.items():
                metric_type = metric_name.split('@')[0]  # e.g., 'mrr', 'ndcg', etc.
                k_value = metric_name.split('@')[1] if '@' in metric_name else 'N/A'
                metrics_data.append([metric_type, k_value, metric_value])
            
            if metrics_data:
                metrics_table = wandb.Table(
                    columns=["Metric Type", "K", "Value"],
                    data=metrics_data
                )
                metrics_to_log[f"{prefix}/detailed_metrics_table"] = metrics_table
        
        # Log all metrics
        wandb.log(metrics_to_log, step=effective_step)
        
        logger.info(f"{prefix.capitalize()} main score: {main_score}")
        return main_score
        
    except Exception as e:
        logger.error(f"Error evaluating {prefix} set: {e}")
        wandb.log({f"{prefix}/evaluation_error": str(e)}, step=effective_step)
        return None

def train_model(train_examples, val_df, test_df, output_path="lyrics_sbert_model", run_name=None, project_name="lyric-semantic-search"):
    """Train model with comprehensive logging for train, validation, and test sets."""
    
    if not train_examples:
        logger.error("No training examples provided")
        return
    
    # Generate run name if not provide
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"lyrics_model_{timestamp}"
    
    try:
        # Initialize W&B with error handling
        wandb.init(
            project=project_name, 
            name=run_name, 
            reinit=True,
            tags=["lyrics", "semantic-search", "sentence-transformers", "detailed-metrics"],
            config=CONFIG
        )
        
        # Load model with better device handling
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # all-MiniLM-L6-v2 all-mpnet-base-v2
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # Training configuration
        batch_size = 8 if device == "cuda" else 4
        epochs = CONFIG["epochs"]
        learning_rate = CONFIG["learning_rate"]

        # Create data loader 
        dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        logger.info(f"Lengthen of dataloader: {len(dataloader)}")
        
        # Use appropriate loss function
        loss = losses.MultipleNegativesRankingLoss(model)
        
        # Build evaluators
        val_evaluator = build_ir_evaluator(val_df, name_suffix="-val") if val_df is not None and len(val_df) > 0 else None
        test_evaluator = build_ir_evaluator(test_df, name_suffix="-test") if test_df is not None and len(test_df) > 0 else None
        
        if val_evaluator is None:
            logger.warning("No valid evaluator created, training without evaluation")
        if test_evaluator is None:
            logger.warning("No valid evaluator created, training without evaluation")

        # Calculate steps
        total_steps = len(dataloader) * epochs  # Use epochs variable instead of hardcoded 4
        logger.info(f"Total steps: {total_steps}")
        warmup_steps = int(0.1 * total_steps)
        evaluation_steps = max(100, len(dataloader) // 4)  # Evaluate 4 times per epoch

        callback = WandbCallback(
            val_evaluator=val_evaluator, 
            test_evaluator=test_evaluator,
            evaluate_test_every_n_steps=evaluation_steps * 2
        )
        
        # Log config to wandb
        runtime_config = {
            **CONFIG,
            "device": device,
            "batch_size": batch_size,
            "num_train_samples": len(train_examples),
            "val_size": len(val_df) if val_df is not None else 0,
            "test_size": len(test_df) if test_df is not None else 0,
            "model_name":  CONFIG["base_model"],
            "optimizer": "AdamW",
            "scheduler": "WarmupLinear"
        }
        wandb.config.update(runtime_config)

        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Total steps: {total_steps}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
        logger.info(f"  - Evaluation steps: {evaluation_steps}")

        # Log additional training info
        wandb.log({
            "setup/total_steps": total_steps,
            "setup/warmup_steps": warmup_steps,
            "setup/evaluation_steps": evaluation_steps,
            "setup/steps_per_epoch": len(dataloader),
            "setup/training_started": True
        })
        
        initial_scores = {}

        if val_evaluator:
            logger.info("Running initial validation evaluation...")
            initial_val_score = evaluate_model_with_metrics(
                model, val_evaluator, 0, "val_initial"
            )
            initial_scores["val"] = initial_val_score
            
            # Log initial validation with comprehensive metrics
            wandb.log({
                "baseline/val_score": initial_val_score,
                "val/score": initial_val_score,
                "val/epoch": 0,
                "val/steps": 0,
                "val/initial": True
            }, step=0)

        if test_evaluator:
            logger.info("Running initial validation evaluation...")
            initial_test_score = evaluate_model_with_metrics(
                model, test_evaluator, 0, "test_initial"
            )
            initial_scores["test"] = initial_test_score

            # Log initial validation with comprehensive metrics
            wandb.log({
                    "baseline/test_score": initial_test_score,
                    "test/score": initial_val_score,
                    "test/epoch": 0,
                    "test/steps": 0,
                    "test/initial": True
            }, step=0)

        # Log initial training "score" (0 since no training yet)
        wandb.log({
            "baseline/train_score": 0.0,
            "train/score": 0.0,
            "train/epoch": 0,
            "train/steps": 0,
            "train/initial": True
        }, step=0)
    
        # Train model
        model.fit(
            train_objectives=[(dataloader, loss)],
            evaluator=val_evaluator,
            evaluation_steps=evaluation_steps,
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            weight_decay=CONFIG["weight_decay"],
            scheduler="WarmupLinear",
            output_path=output_path,
            show_progress_bar=True,
            use_amp=True,
            save_best_model=True,
            callback=callback if val_evaluator else None
        )
        
        logger.info(f"Model fine-tuned and saved to '{output_path}'")

        # Final comprehensive evaluations with complete score/step/epoch logging
        final_step = callback.step_count if callback.step_count > 0 else total_steps
        final_epoch = epochs

        # Final test evaluation
        if val_evaluator:
            final_val_score = evaluate_model_with_metrics(model, test_evaluator, final_step, "final_val")
        
        
        # Comprehensive final validation logging
            val_improvement = final_val_score - initial_scores.get("val", 0)
            wandb.log({
                "final/val_score": final_val_score,
                "final/val_improvement": val_improvement,
                "final/val_epoch": final_epoch,
                "final/val_steps": final_step,
                "val/final_score": final_val_score,
                "val/final_epoch": final_epoch,
                "val/final_steps": final_step
            }, step=final_step)
        
        # Final test evaluation
        if test_evaluator:
            logger.info("Running final test evaluation...")
            final_test_score = evaluate_model_with_metrics(
                model, test_evaluator, final_step, "final_test"
            )
            
            # Comprehensive final test logging
            test_improvement = final_test_score - initial_scores.get("test", 0)
            wandb.log({
                "final/test_score": final_test_score,
                "final/test_improvement": test_improvement,
                "final/test_epoch": final_epoch,
                "final/test_steps": final_step,
                "test/final_score": final_test_score,
                "test/final_epoch": final_epoch,
                "test/final_steps": final_step
            }, step=final_step)
        
        # Final training metrics
        final_train_score = callback.train_history["scores"][-1] if callback.train_history["scores"] else 0.0
        train_improvement = final_train_score - 0.0  # Initial train score was 0
        wandb.log({
            "final/train_score": final_train_score,
            "final/train_improvement": train_improvement,
            "final/train_epoch": final_epoch,
            "final/train_steps": final_step,
            "train/final_score": final_train_score,
            "train/final_epoch": final_epoch,
            "train/final_steps": final_step
        }, step=final_step)
        
        # Log comprehensive summary table
        summary_data = []
        sets = ["train", "val", "test"]
        for set_name in sets:
            if set_name == "train":
                initial_score = 0.0
                final_score = final_train_score
            elif set_name == "val" and "val" in initial_scores:
                initial_score = initial_scores["val"]
                final_score = final_val_score if 'final_val_score' in locals() else initial_score
            elif set_name == "test" and "test" in initial_scores:
                initial_score = initial_scores["test"]
                final_score = final_test_score if 'final_test_score' in locals() else initial_score
            else:
                continue
                
            summary_data.append([
                set_name.capitalize(),
                initial_score,
                final_score,
                final_score - initial_score,
                final_epoch,
                final_step
            ])
        
        if summary_data:
            summary_table = wandb.Table(
                columns=["Set", "Initial Score", "Final Score", "Improvement", "Final Epoch", "Final Steps"],
                data=summary_data
            )
            wandb.log({"final/scores_summary_table": summary_table}, step=final_step)
        
        # Log comprehensive final summary with all tracking info
        final_summary = {
            "final/model_saved": True,
            "final/output_path": output_path,
            "final/best_val_score": callback.best_val_score,
            "final/best_test_score": callback.best_test_score,
            "final/total_training_steps": final_step,
            "final/epochs_completed": final_epoch,
            "final/run_name": run_name
        }

        # Add callback history lengths
        if callback:
            final_summary.update({
                "final/train_evaluations": len(callback.train_history["scores"]),
                "final/val_evaluations": len(callback.val_history["scores"]),
                "final/test_evaluations": len(callback.test_history["scores"])
            })
        
        wandb.log(final_summary, step=final_step)
        
        # Log training completion message
        logger.info("Training completed successfully!")
        logger.info(f"Best validation score: {callback.best_val_score}")
        logger.info(f"Best test score: {callback.best_test_score}")
        logger.info(f"Model saved to: {output_path}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        wandb.log({"error/training_failed": str(e)})
        raise
    finally:
        # Ensure W&B run is properly finished
        try:
            wandb.finish()
        except:
            pass
def hash_file(file_path):
    """Create hash from file content with error handling."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return None
        
def get_file_hashes(folder):
    """Get hashes for all CSV files in folder."""
    hashes = {}
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    
    for f in csv_files:
        file_hash = hash_file(f)
        if file_hash:
            hashes[os.path.basename(f)] = file_hash
    
    return hashes
    
def update_and_train_if_changed(data_folder, hash_path="data_hashes.json", force_retrain=False):
    """Updated function to include classification training."""
    
    print("[INFO] Checking for data changes...")
    current_hashes = get_file_hashes(data_folder)

    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            old_hashes = json.load(f)
    else:
        old_hashes = {}

    # Chuyển dict thành chuỗi sorted để tránh lỗi so sánh
    if json.dumps(current_hashes, sort_keys=True) != json.dumps(old_hashes, sort_keys=True):
        print("[INFO] Data changed. Reloading and retraining...")

        with open(hash_path, "w") as f:
            json.dump(current_hashes, f, indent=2)

        df = load_and_prepare_data(data_folder)
        train_df, val_df, test_df = create_data_splits(df=df)

        train_df.to_csv("train_df.csv", index=False)
        val_df.to_csv("val_df.csv", index=False)
        test_df.to_csv("test_df.csv", index=False)
        print("[INFO] Saved train_df.csv and val_df.csv and test_df.csv for checking.")

        train_examples = create_training_pairs(train_df)
        save_examples(train_examples)
        
        print(f"[INFO] Total training pairs: {len(train_examples)}")

        train_model(train_examples, val_df=val_df, test_df=test_df)
    else:
        print("[INFO] No data changes detected. Skipping training.")
        

def test_embedding_similarity(model_path, df: pd.DataFrame, n_tests=10):
    """Test embedding similarity with multiple examples."""
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return
    
    if len(df) == 0:
        logger.error("Empty dataframe provided for testing")
        return
    
    try:
        model = SentenceTransformer(model_path)
        
        # Test multiple examples
        test_rows = df.sample(n=min(n_tests, len(df)))
        
        total_score = 0
        valid_tests = 0
        
        for idx, (_, row) in enumerate(test_rows.iterrows()):
            try:
                lyric = _normalize_text_improved(row["Lyric"])
                title = _normalize_text_improved(row["Title"])
                artist = _normalize_text_improved(row["Artist"])
                album = _normalize_text_improved(row["Album"]) if pd.notna(row["Album"]) else "Unknown"
                
                if not lyric or not title or not artist:
                    continue
                
                meta_text = f"The song {title} by {artist} from the album {album}"
                
                emb_lyric = model.encode(lyric, normalize_embeddings=True)
                emb_meta = model.encode(meta_text, normalize_embeddings=True)
                score = cosine_similarity([emb_lyric], [emb_meta])[0][0]
                
                logger.info(f"Test {idx+1}: {title} by {artist} - Similarity: {score:.4f}")
                total_score += score
                valid_tests += 1
                
            except Exception as e:
                logger.error(f"Error in test {idx+1}: {e}")
                continue
        
        if valid_tests > 0:
            avg_score = total_score / valid_tests
            logger.info(f"Average similarity across {valid_tests} tests: {avg_score:.4f}")
        else:
            logger.error("No valid tests completed")
            
    except Exception as e:
        logger.error(f"Error loading model or running tests: {e}")

def run_training_pipeline(data_dir, output_path="improved_lyrics_model", project_name="lyric-semantic-search", 
                         force_retrain=False, sample_size=None):
    """Run the complete training pipeline."""
    logger.info("Starting lyrics model training pipeline...")
    
    # Check if retraining is needed
    if not update_and_train_if_changed(data_dir, force_retrain=force_retrain):
        logger.info("No data changes detected. Skipping training.")
        return
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_and_prepare_data(data_dir, sample_size=sample_size, min_lyric_words=CONFIG["min_lyric_words"])
    
    if len(df) == 0:
        logger.error("No data loaded - check your data directory and CSV files")
        return
    
    # Create data splits
    train_df, val_df, test_df = create_data_splits(
        df, 
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"]
    )
    
    # Create training pairs
    logger.info("Creating training pairs...")
    train_examples = create_training_pairs(
        train_df,
        negative_ratio=CONFIG["negative_ratio"],
        max_segments_per_song=CONFIG["max_segments_per_song"],
        min_lyric_words=CONFIG["min_lyric_words"]
    )
    
    if not train_examples:
        logger.error("No training examples created")
        return
    
    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lyrics_{len(df)}songs_{len(train_examples)}pairs_{timestamp}"
    
    # Train model
    logger.info("Starting model training...")
    model = train_model(train_examples, val_df, test_df, output_path, project_name, run_name)
    
    if model:
        # Save training info
        training_info = {
            "total_songs": len(df),
            "train_songs": len(train_df),
            "val_songs": len(val_df),
            "test_songs": len(test_df),
            "train_pairs": len(train_examples),
            "unique_artists": df['Artist'].nunique(),
            "config": CONFIG,
            "timestamp": timestamp
        }
        
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        # Save file hashes to prevent unnecessary retraining
        update_and_train_if_changed(data_dir)
        
        # Test the model
        logger.info("Testing trained model...")
        test_embedding_similarity(output_path, df, n_tests=5)
        
        logger.info("Training pipeline completed successfully!")
    else:
        logger.error("Model training failed")


def main():
    """Main entry point."""
    # Configuration
    DATA_DIR = "backend\\data\\csv"  # Update this path as needed
    OUTPUT_PATH = "lyric_sbert_model"
    PROJECT_NAME = "lyric-semantic-search"
    
    # Parse command line arguments
    import sys
    force_retrain = "--force" in sys.argv
    sample_size = None
    
    # Look for sample size argument
    for arg in sys.argv:
        if arg.startswith("--sample="):
            try:
                sample_size = int(arg.split("=")[1])
            except ValueError:
                logger.warning(f"Invalid sample size: {arg}")
    
    # Run training pipeline
    run_training_pipeline(
        data_dir=DATA_DIR,
        output_path=OUTPUT_PATH,
        project_name=PROJECT_NAME,
        force_retrain=force_retrain,
        sample_size=sample_size
    )

if __name__ == "__main__":
    main()