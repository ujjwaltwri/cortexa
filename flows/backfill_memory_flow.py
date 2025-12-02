from prefect import flow
import pandas as pd
import yaml
from pathlib import Path
from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB
from src.rag.strategy_memory import StrategyMemory

@flow(name="Cortexa Smart Backfill")
def backfill_memory_flow(config_path="config.yaml"):
    print("--- 🧠 Starting Real-Memory Backfill ---")
    
    # 1. Load Data
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config['data_paths']['processed'])
    data_file = processed_path / "features_and_targets.csv"
    
    if not data_file.exists():
        print("Error: features_and_targets.csv not found.")
        return
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} historical rows.")
    
    # 2. Initialize Real Components
    try:
        embedder = Embedder() # Uses HuggingFace Neural Net
        vector_db = QdrantVecDB(vector_size=384)
        
        # CRITICAL: Wipe the old random database so we don't mix bad data with good
        vector_db.recreate_collection()
        
        memory = StrategyMemory(vector_db, embedder)
    except Exception as e:
        print(f"Init Error: {e}")
        return

    # 3. Process Data
    texts = []
    metadatas = []
    
    # Columns to exclude from the text description
    skip_cols = ['target', 'ticker', 'future_close', 'future_return', 'market_regime']
    feature_cols = [c for c in df.columns if c not in skip_cols]
    
    print(f"Generating semantic descriptions for {len(df)} days...")
    
    for index, row in df.iterrows():
        # Create a "Sentence" describing the market state
        # e.g. "rsi:35.5 atr:2.1 natr:1.5 vix_change:-0.5"
        state_desc = " ".join([f"{k}:{v:.4f}" for k,v in row[feature_cols].items() if isinstance(v, (int, float))])
        
        texts.append(state_desc)
        
        meta = {
            "ticker": row['ticker'],
            "target": int(row['target']),
            "date": index.strftime("%Y-%m-%d"),
            # Ensure we use the regime if it exists, otherwise default to 0
            "regime": int(row.get('market_regime', 0)) 
        }
        metadatas.append(meta)

    # 4. Upload in Batches
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        print(f"Indexing batch {i} to {i+batch_size}...")
        memory.record_batch(texts[i:i+batch_size], metadatas[i:i+batch_size])
        
    print("✅ Backfill Complete. Your system now has Real Memory.")

if __name__ == "__main__":
    backfill_memory_flow()