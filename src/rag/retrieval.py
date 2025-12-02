from src.rag.vector_store import QdrantVecDB
from src.rag.embeddings import Embedder
import yaml

def query_vector_db(query, ticker=None, config_path="config.yaml"):
    """
    Retrieves relevant news articles from the Qdrant vector database.
    Enforces a strict ticker filter if one is provided. No fallback to broad search.
    """
    print(f"--- Querying Vector DB with: '{query}' (Ticker Filter: {ticker or 'None'}) ---")
    
    try:
        # 1. Load Config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            top_k = config['rag'].get('retrieval_top_k', 10)
        except:
            top_k = 10

        # 2. Init Components
        embedder = Embedder()
        db = QdrantVecDB() 
        query_vec = embedder.encode([query])[0]
        
        # 3. Build Filter (Strict)
        search_filters = {"event_type": "news"}
        if ticker:
            search_filters["ticker"] = ticker.upper()
            print(f"Applying strict ticker filter: {search_filters['ticker']}")

        # 4. Search
        results = db.search(query_vec, top_k=top_k, filters=search_filters)
        
        # 5. Return Results (If results is empty, it returns clean empty lists)
        if not results:
            print("[RAG] No articles found matching strict filter.")
            return {"documents": [[]], "metadatas": [[]]}

        # 6. Format for the Agent
        documents = [r['text'] for r in results]
        metadatas = [r['meta'] for r in results]
        
        print(f"Found {len(documents)} relevant news articles.")
        
        return {
            "documents": [documents], 
            "metadatas": [metadatas]
        }
        
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return {"documents": [[]], "metadatas": [[]]}

if __name__ == "__main__":
    # Test
    res = query_vector_db("outlook on MSFT", ticker="MSFT")
    print(res)