import yaml
from qdrant_client import QdrantClient
from src.rag.embeddings import Embedder

def debug_rag():
    print("--- 🕵️‍♂️ Cortexa RAG Debugger ---")
    
    # 1. Check Connection & Collection
    try:
        client = QdrantClient("localhost", port=6333)
        collections = client.get_collections().collections
        print(f"✅ Connected to Qdrant. Found collections: {[c.name for c in collections]}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("👉 Is Docker running? (docker-compose up -d)")
        return

    # 2. Check Data Count
    col_name = "cortexa_rag_store"  # Default name
    if not any(c.name == col_name for c in collections):
        print(f"❌ Collection '{col_name}' NOT found. You need to run the ingestion script.")
        return
    
    count = client.count(col_name).count
    print(f"📊 Total Documents in '{col_name}': {count}")
    
    if count == 0:
        print("❌ Database is EMPTY. Run 'python -m flows.daily_update_flow' to ingest data.")
        return

    # 3. Inspect Metadata (The most likely culprit)
    print("\n--- 📝 Sample Document Metadata ---")
    sample = client.scroll(col_name, limit=1)[0]
    if sample:
        payload = sample[0].payload
        print(f"Title: {payload.get('title', 'N/A')}")
        print(f"Ticker Tag: {payload.get('ticker', 'MISSING ❌')}")
        print(f"Source: {payload.get('source', 'N/A')}")
    
    # 4. Test Retrieval
    print("\n--- 🧪 Test Search: 'Outlook on Apple' ---")
    embedder = Embedder()
    vec = embedder.encode(["What is the outlook on Apple?"])[0]
    
    # Search WITHOUT filters first
    hits = client.search(col_name, query_vector=vec, limit=3)
    print(f"Found {len(hits)} raw matches (No Filter). Top score: {hits[0].score:.4f}" if hits else "❌ No matches found.")
    
    # Search WITH ticker filter (This is what fails)
    hits_filtered = client.search(
        col_name, 
        query_vector=vec, 
        query_filter={"must": [{"key": "ticker", "match": {"value": "AAPL"}}]},
        limit=3
    )
    print(f"Found {len(hits_filtered)} matches (Filter: ticker='AAPL').")

if __name__ == "__main__":
    debug_rag()