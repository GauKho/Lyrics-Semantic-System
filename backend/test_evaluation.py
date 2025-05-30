from retrieval.hybrid_lyrics_searching import HybridLyricsSearch
from retrieval.sbert import SBERTSearcher
from retrieval.bm25 import BM25LyricsSearch
from evaluation import Evaluator

ground_truth = ["Love Yourself", "Over", "3 a.m.", "Insomniac’s Lullaby", "Moody Ballad of Ed"]

test_query = [{"query": "And I've been so caught up in my job, didn't see what's going on But now I know, I'm better sleeping on my own", "ground_truth": ground_truth}]

hybrid_model = HybridLyricsSearch(data_path="backend\\data\\csv", sbert_model="lyrics_sbert_model", alpha=0.5)
evaluator_hybrid = Evaluator(model=hybrid_model, query=test_query, top_k=5)
result = evaluator_hybrid.evaluate()
print(f"Hybrid_model: {result}")

sbert_model = SBERTSearcher(data_path="backend\\data\\csv", model_name="lyrics_sbert_model", force_rebuild=False)
evaluator_sbert = Evaluator(model=sbert_model, query=test_query, top_k=5)
metrics_sbert = evaluator_sbert.evaluate()
print(metrics_sbert)

bm25_model = BM25LyricsSearch(data_path="backend\\data\\csv")   
evaluator_bm25 = Evaluator(model=bm25_model, query=test_query, top_k=5)
result = evaluator_bm25.evaluate()
print(f"BM25_model: {result}")