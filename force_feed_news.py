from src.data_ingestion.rss_client import fetch_rss_news
from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB
import pandas as pd

def force_feed():
    print("--- 💉 Force Feeding News into Cortexa ---")
    
    # 1. Fetch News Directly
    print("1. Fetching News...")
    df = fetch_rss_news("config.yaml")
    
    if df is None or df.empty:
        print("❌ No news fetched. Check your internet or RSS sources.")
        return
        
    print(f"✅ Fetched {len(df)} articles.")
    
    # 2. Initialize DB
    print("2. Connecting to Database...")
    embedder = Embedder()
    db = QdrantVecDB(vector_size=384)
    
    # Check count before
    initial_count = db.client.count(db.collection_name).count
    print(f"   Count BEFORE: {initial_count}")
    
    # 3. Prepare Data
    print("3. Embedding & Indexing...")
    texts = (df['title'] + " - " + df['summary']).tolist()
    vectors = embedder.encode(texts)
    
    metadatas = df.to_dict('records')
    # Ensure tags are present
    for m in metadatas:
        m['event_type'] = 'news'
        # Ensure ticker is set (handle NaNs)
        if pd.isna(m.get('ticker')):
            m['ticker'] = 'MARKET'
            
    # 4. Upsert
    db.upsert(texts, vectors, metadatas)
    
    # 5. Verify
    final_count = db.client.count(db.collection_name).count
    print(f"   Count AFTER:  {final_count}")
    
    if final_count > initial_count:
        print("✅ SUCCESS: Database grew. News is inside.")
        
        # Rapid Test
        print("\n--- 🧪 Quick Retrieval Test ---")
        q_vec = embedder.encode(["Outlook on Apple"])[0]
        results = db.search(q_vec, top_k=3, filters={"event_type": "news"})
        if results:
            print(f"   Found: {results[0]['meta']['title']}")
            print(f"   Score: {results[0]['score']:.4f}")
        else:
            print("   ❌ Retrieval failed even after insert.")
    else:
        print("❌ FAILURE: Database count did not increase.")

if __name__ == "__main__":
    force_feed()