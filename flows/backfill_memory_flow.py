from prefect import flow
import pandas as pd
import yaml
import json
from pathlib import Path
from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB
from src.rag.strategy_memory import StrategyMemory

@flow(name="Cortexa Smart Backfill V3")
def backfill_memory_flow(config_path="config.yaml"):
    print("--- 🧠 Starting V3 Memory Backfill ---")
    
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config['data_paths']['processed'])
    # CRITICAL: Point to V3
    data_file = processed_path / "features_v3.csv"
    
    if not data_file.exists():
        print("Error: features_v3.csv not found.")
        return
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} V3 rows.")
    
    try:
        embedder = Embedder()
        vector_db = QdrantVecDB(vector_size=384)
        # Wipe clean
        vector_db.recreate_collection()
        memory = StrategyMemory(vector_db, embedder)
    except Exception as e:
        print(f"Init Error: {e}")
        return

    texts = []
    metadatas = []
    
    # Features to describe the state in text
    # We exclude targets and metadata columns
    desc_cols = [c for c in df.columns if c not in ['target', 'ticker', 'regime_tag']]
    
    print("Generating V3 semantic descriptions...")
    for index, row in df.iterrows():
        # Create text: "rsi:30.5 natr:2.1 ..."
        state_desc = " ".join([f"{k}:{v:.4f}" for k,v in row[desc_cols].items() if isinstance(v, (int, float))])
        
        texts.append(state_desc)
        
        meta = {
            "ticker": row['ticker'],
            "target": int(row['target']),
            "date": index.strftime("%Y-%m-%d"),
            "regime": str(row['regime_tag']), # Store regime as string "1_0"
            "event_type": "market_state"
        }
        metadatas.append(meta)

    # Upload
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        print(f"Indexing batch {i}...")
        memory.record_batch(texts[i:i+batch_size], metadatas[i:i+batch_size])
        
    print("✅ V3 Backfill Complete.")

if __name__ == "__main__":
    backfill_memory_flow()