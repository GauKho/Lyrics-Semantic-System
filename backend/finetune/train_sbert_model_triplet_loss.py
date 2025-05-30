from data_loader import load_and_prepare_data, create_triplet_example_hard, save_examples
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json
import glob
import os

def build_ir_evaluator(dev_df, max_queries=100):
    """
    Xây evaluator semantic search (lyrics → tìm title đúng)
    """
    queries = {}
    corpus = {}
    relevant_docs = defaultdict(set)

    for idx, row in enumerate(dev_df.sample(n=min(max_queries, len(dev_df)), random_state=42).iterrows()):
        i, row = idx, row[1]
        query_id = f"q{i}"
        doc_id = f"d{i}"

        query = row["Title"].strip().lower()
        title = row["Lyric"].strip().lower()

        queries[query_id] = query
        corpus[doc_id] = title
        relevant_docs[query_id].add(doc_id)

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        name="lyrics-ir-eval"
    )
    return evaluator


def train_model(train_examples, dev_df, output_path="lyrics_sbert_model_triplet"):
    model = SentenceTransformer("all-mpnet-base-v2")
    dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    loss = losses.TripletLoss(model)

    evaluator = build_ir_evaluator(dev_df)

    model.fit(
        train_objectives=[(dataloader, loss)],
        evaluator=evaluator,
        evaluation_steps=200,
        epochs=5,
        warmup_steps=int(0.1 * len(dataloader)),
        optimizer_params={"lr": 2e-5},
        weight_decay=0.01,
        scheduler="WarmupCosine",
        output_path=output_path,
        show_progress_bar=True
    )
    print(f"Model fine-tuned and saved to '{output_path}'")

def hash_file(file_path):
    """Tạo hash từ nội dung file để kiểm tra thay đổi."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
        
def get_file_hashes(folder):
    return {
        os.path.basename(f): hash_file(f)
        for f in glob.glob(os.path.join(folder, "*.csv"))
    }
    
def update_and_train_if_changed(data_folder, hash_path="data_hashes_triplet.json"):
    print("[INFO] Checking for data changes...")
    current_hashes = get_file_hashes(data_folder)

    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            old_hashes = json.load(f)
    else:
        old_hashes = {}

    if current_hashes != old_hashes:
        print("[INFO] Data changed. Reloading and retraining...")

        with open(hash_path, "w") as f:
            json.dump(current_hashes, f, indent=2)

        df = load_and_prepare_data(data_folder)
        dev_df = df.sample(frac=0.1, random_state=42)
        train_df = df.drop(dev_df.index)

        train_examples = create_triplet_example_hard(train_df)
        save_examples(train_examples)

        train_model(train_examples, dev_df)
    else:
        print("[INFO] No data changes detected. Skipping training.")


def test_embedding_similarity(model_path, df):
    model = SentenceTransformer(model_path)
    test_row = df.sample(1).iloc[0]

    emb_lyric = model.encode(test_row["Lyric"])
    emb_title = model.encode(test_row["Title"])
    score = cosine_similarity([emb_lyric], [emb_title])[0][0]

    print(f"[TEST] Similarity between lyric and title: {score:.4f}")

if __name__ == "__main__":
    data_dir = "backend\data\csv"
    update_and_train_if_changed(data_dir)

    df_loaded = load_and_prepare_data(data_dir)
    test_embedding_similarity("lyrics_sbert_model_triplet", df_loaded)