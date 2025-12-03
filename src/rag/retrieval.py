from src.rag.vector_store import QdrantVecDB
from src.rag.embeddings import Embedder
import yaml

def query_vector_db(query, ticker=None, config_path="config.yaml"):
    """
    Retrieves relevant news articles.
    Fixes the 'missing context' bug by INJECTING the ticker into the search text
    instead of relying solely on metadata filters.
    """
    print(f"--- Querying Vector DB with: '{query}' (Ticker Hint: {ticker}) ---")
    
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
        
        # 3. THE FIX: Inject Ticker into Query
        # This forces the vector to align with the specific stock
        if ticker:
            search_text = f"News and outlook for {ticker}: {query}"
        else:
            search_text = query
            
        query_vec = embedder.encode([search_text])[0]
        
        # 4. Semantic Search (No Strict Filter)
        # We remove the strict 'ticker' filter because metadata tags can be messy.
        # We trust the 'search_text' injection to find the right articles.
        results = db.search(query_vec, top_k=top_k, filters={"event_type": "news"})
        
        if not results:
            print("No results found in Qdrant.")
            return {"documents": [[]], "metadatas": [[]]}

        # 5. Format for Agent
        documents = [r['text'] for r in results]
        metadatas = [r['meta'] for r in results]
        
        # DEBUG: Print what we actually found
        print(f"Found {len(documents)} articles. Top result: {metadatas[0].get('title', 'Unknown')}")
        
        return {
            "documents": [documents], 
            "metadatas": [metadatas]
        }
        
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return {"documents": [[]], "metadatas": [[]]}

if __name__ == "__main__":
    # Test
    res = query_vector_db("outlook", ticker="GOOGL")
    print(res)