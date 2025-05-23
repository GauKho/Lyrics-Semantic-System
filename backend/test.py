from retrieval.hybrid_lyrics_searching import HybridLyricsSearch

data_path = "backend\data\csv"

try: 
    hybrid_lyrics = HybridLyricsSearch(data_path)
    results = hybrid_lyrics.search("You should go, love yourself", top_k=7)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} by {res['artist']} (Score: {res['hybrid_score']})")
        print(f"   {res['lyrics']}\n")

except Exception as e:
    print(f"Error occured: {e}")