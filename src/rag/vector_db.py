import pandas as pd
import yaml
import chromadb
from pathlib import Path
from src.rag.embedding_models import get_embedding_model # Import from our other file
import glob
import os

def find_latest_raw_news(raw_data_path):
    """Finds the most recent 'raw_news_*.csv' file."""
    search_pattern = str(raw_data_path / "raw_news_*.csv")
    files = glob.glob(search_pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def initialize_vector_db(config_path="config.yaml"):
    """
    Loads raw news, generates embeddings, and indexes them in ChromaDB.
    """
    print("--- Starting Vector DB Initialization ---")
    
    # 1. Load Config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return

    db_path = Path(config['data_paths']['vectorstore'])
    collection_name = config['rag']['vector_db_collection']
    raw_data_path = Path(config['data_paths']['raw'])

    # 2. Find Latest Raw News File
    latest_news_file = find_latest_raw_news(raw_data_path)
    if not latest_news_file:
        print(f"No raw news CSV files found in {raw_data_path}. Skipping.")
        return

    print(f"Loading raw news from: {latest_news_file}")
    df = pd.read_csv(latest_news_file)
    df = df.dropna(subset=['summary']) # Ensure we only index articles with summaries
    
    if df.empty:
        print("No valid articles found in the CSV. Skipping.")
        return

    # 3. Load Embedding Model
    embed_model = get_embedding_model(config_path)
    if embed_model is None:
        print("Failed to load embedding model. Aborting.")
        return

    # 4. Initialize ChromaDB
    # We use a PersistentClient to save the DB to disk
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Get or create the collection
    collection = client.get_or_create_collection(name=collection_name)
    print(f"ChromaDB collection '{collection_name}' ready at {db_path}")

    # 5. Prepare Documents for Indexing
    documents = df['summary'].tolist()
    
    # Create metadata for each document
    metadatas = df[['source', 'title', 'link', 'published']].to_dict('records')
    
    # Create unique IDs for each document
    ids = [f"news_{i}" for i in range(len(documents))]

    # 6. Generate Embeddings and Add to DB
    print(f"Generating embeddings for {len(documents)} articles...")
    # This step can take a moment
    embeddings = embed_model.encode(documents)
    
    print("Adding documents to ChromaDB...")
    # Use upsert to add new docs or update existing ones
    collection.upsert(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"\nSuccessfully indexed {collection.count()} documents.")
    print("--- Vector DB Initialization Complete ---")

if __name__ == "__main__":
    initialize_vector_db(config_path="config.yaml")