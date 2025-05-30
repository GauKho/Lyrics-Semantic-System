from typing import List, Dict
import numpy as np

class Evaluator:
    def __init__(self, model, query: List[Dict], top_k: int = 5):

        """
        model: method search(query, top_k)
        query
        top_k: tong so ket qua xem xet
        """
        self.model = model
        self.query = query
        self.top_k = top_k

    def precision_at_k(self, predicted: List[str], relevant: List[str]) -> float:
        predicted_k = predicted[:self.top_k]
        if not predicted_k:
            return 0.0
        return len(set(predicted_k) & set(relevant)) / len(predicted_k)

    def recall_at_k(self, predicted: List[str], relevant: List[str]) -> float:
        predicted_k = predicted[:self.top_k]
        if not relevant:
            return 0.0
        return len(set(predicted_k) & set(relevant)) / len(relevant)

    def reciprocal_rank(self, predicted: List[str], relevant: List[str]) -> float:
        for i, p in enumerate(predicted):
            if p in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def dcg_at_k(self, predicted: List[str], relevant: List[str]) -> float:
        dcg = 0.0
        for i, p in enumerate(predicted[:self.top_k]):
            if p in relevant:
                dcg += 1 / np.log2(i + 2)
        return dcg

    def idcg_at_k(self, relevant: List[str]) -> float:
        n = min(len(relevant), self.top_k)
        idcg = sum(1 / np.log2(i + 2) for i in range(n))
        return idcg

    def ndcg_at_k(self, predicted: List[str], relevant: List[str]) -> float:
        idcg = self.idcg_at_k(relevant)
        if idcg == 0:
            return 0.0
        dcg = self.dcg_at_k(predicted, relevant)
        return dcg / idcg

    def evaluate(self):
        precisions, recalls, mrrs, ndcgs = [], [], [], []

        for item in self.query:
            query = item['query']
            relevant = item['ground_truth']

            results = self.model.search(query, top_k=self.top_k)
            predicted_titles = [r['title'] for r in results]

            precisions.append(self.precision_at_k(predicted_titles, relevant))
            recalls.append(self.recall_at_k(predicted_titles, relevant))
            mrrs.append(self.reciprocal_rank(predicted_titles, relevant))
            ndcgs.append(self.ndcg_at_k(predicted_titles, relevant))

        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_mrr = np.mean(mrrs) if mrrs else 0.0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0

        print(f"Evaluation Results (top-{self.top_k}):")
        print(f"Precision@{self.top_k}: {avg_precision:.4f}")
        print(f"Recall@{self.top_k}: {avg_recall:.4f}")
        print(f"MRR: {avg_mrr:.4f}")
        print(f"NDCG@{self.top_k}: {avg_ndcg:.4f}")

        return {
            "Precision@k": avg_precision,
            "Recall@k": avg_recall,
            "MRR": avg_mrr,
            "NDCG@k": avg_ndcg,
        }
