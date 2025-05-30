import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.retrieval.bm25 import BM25LyricsSearch
from backend.retrieval.sbert import SBERTSearcher
import numpy as np

class HybridLyricsSearch:
    def __init__(self, data_path, sbert_model='lyrics_sbert_model', alpha= 0.5):
        self.bm25 = BM25LyricsSearch(data_path)
        self.sbert = SBERTSearcher(data_path, model_name=sbert_model)
        self.alpha = alpha

    def search(self, query, top_k=5):
        bm25_scores = self.bm25.get_score(query)
        _, sbert_raw_scores, indices = self.sbert.get_score(query, top_k=top_k * 2)  # search wider range

        # Normalize scores
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        sbert_norm_scores = (sbert_raw_scores - sbert_raw_scores.min()) / (
            sbert_raw_scores.max() - sbert_raw_scores.min() + 1e-8
        )

        hybrid_scores = self.alpha * sbert_norm_scores + (1 - self.alpha) * bm25_norm[indices]

        # Sort by hybrid score
        sorted_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for i in sorted_indices:
            idx = indices[i]
            if idx < len(self.bm25.df):
                results.append({
                    "title": self.bm25.df.iloc[idx]["Title"],
                    "artist": self.bm25.df.iloc[idx]["Artist"],
                    "lyrics": self.bm25.df.iloc[idx]["Lyric"][:300] + "...",
                    "hybrid_score": round(float(hybrid_scores[i]), 3),
                    "bm25_score": round(float(bm25_norm[idx]), 3),
                    "sbert_score": round(float(sbert_norm_scores[i]), 3)
                })

        return results