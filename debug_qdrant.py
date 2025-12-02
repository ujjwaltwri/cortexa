from src.rag.vector_store import QdrantVecDB
from src.rag.embeddings import Embedder

def debug_db():
    print("--- 🕵️‍♂️ Debugging Qdrant Database ---")
    
    db = QdrantVecDB()
    embedder = Embedder()
    
    # 1. Check Count
    info = db.client.get_collection("cortexa_rag_store")
    print(f"Total Vectors in DB: {info.points_count}")
    
    vec = embedder.encode(["Google stock news"])[0]

    # 2. Test Search for 'Google' without filters
    print("\n--- Test 1: Broad Search (No Filters) ---")
    results = db.search(vec, top_k=3)
    for r in results:
        meta = r['meta']
        title = meta.get('title', 'NO TITLE (Market History)')
        print(f"  [{meta.get('event_type')}] Ticker: {meta.get('ticker')} | {title[:40]}...")

    # 3. Test Strict Filter (Any Google Data)
    print("\n--- Test 2: Strict Search (ticker='GOOGL') ---")
    results_strict = db.search(vec, top_k=3, filters={"ticker": "GOOGL"})
    if not results_strict:
        print("❌ No GOOGL data found.")
    else:
        for r in results_strict:
             meta = r['meta']
             title = meta.get('title', 'NO TITLE (Market History)')
             print(f"  [{meta.get('event_type')}] {title[:40]}...")

    # 4. CRITICAL TEST: Specific News Search
    print("\n--- Test 3: Chatbot Logic (ticker='GOOGL' AND event='news') ---")
    results_news = db.search(vec, top_k=3, filters={"ticker": "GOOGL", "event_type": "news"})
    
    if not results_news:
        print("❌ NO GOOGL NEWS FOUND.")
        print("   -> The qualitative brain will say 'Insufficient Context'.")
        print("   -> Fix: Run 'python -m flows.daily_update_flow' again.")
    else:
        print(f"✅ FOUND {len(results_news)} NEWS ARTICLES.")
        for r in results_news:
            print(f"  📰 {r['meta'].get('title')[:60]}...")
        print("\n🚀 SUCCESS: The Dashboard should now answer questions about Google.")

if __name__ == "__main__":
    debug_db()