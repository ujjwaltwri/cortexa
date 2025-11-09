import chromadb
import yaml
from src.rag.embedding_models import get_embedding_model
from pathlib import Path

def query_vector_db(query, config_path="config.yaml"):
    """
    Queries the ChromaDB vector store for relevant documents.
    """
    print(f"--- Querying Vector DB with: '{query}' ---")

    # 1. Load Config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None

    db_path = Path(config['data_paths']['vectorstore'])
    collection_name = config['rag']['vector_db_collection']
    top_k = config['rag']['retrieval_top_k']

    if not db_path.exists():
        print(f"Error: Vector store path not found at {db_path}")
        print("Please run 'python -m src.rag.vector_db' first to create it.")
        return None

    # 2. Load Embedding Model (for embedding the query)
    embed_model = get_embedding_model(config_path)
    if embed_model is None:
        print("Failed to load embedding model. Aborting.")
        return None

    # 3. Connect to Existing ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return None

    # 4. Embed the Query
    print(f"Generating embedding for query...")
    query_embedding = embed_model.encode(query).tolist()

    # 5. Search the Collection
    print(f"Searching collection for top {top_k} relevant documents...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print("--- Query Successful ---")
    return results

if __name__ == "__main__":
    # This allows you to test the retrieval directly
    test_query = "What is the market outlook or sentiment?"
    
    retrieved_docs = query_vector_db(test_query)
    
    if retrieved_docs:
        print(f"\nFound {len(retrieved_docs['documents'][0])} relevant documents for: '{test_query}'")
        
        # Print the results nicely
        for i, doc in enumerate(retrieved_docs['documents'][0]):
            metadata = retrieved_docs['metadatas'][0][i]
            distance = retrieved_docs['distances'][0][i]
            
            print(f"\n--- Result {i+1} (Distance: {distance:.4f}) ---")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Title: {metadata.get('title', 'N/A')}")
            print(f"Summary: {doc}")