from retrieval.hybrid_lyrics_searching import HybridLyricsSearch

data_path = "backend\\data\\csv"

try: 
    hybrid_lyrics = HybridLyricsSearch(data_path)
    results = hybrid_lyrics.search("And I've been so caught up in my job, didn't see what's going on But now I know, I'm better sleeping on my own", top_k=7)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} by {res['artist']} (Score: {res['hybrid_score']})")
        print(f"   {res['lyrics']}\n")

except Exception as e:
    print(f"Error occured: {e}")