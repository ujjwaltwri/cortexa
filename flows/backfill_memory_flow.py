from prefect import flow, task
import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from typing import List, Dict, Any 

# Import all our modules
from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB
from src.rag.strategy_memory import StrategyMemory
from src.regime.hmm_regime import make_regime_features, predict_regime

def get_feature_text(row: pd.Series, feature_names: List[str]) -> str:
    """Serializes a row's features into a text string for embedding."""
    pieces = [f"{k}:{float(row[k]):.4f}" for k in feature_names if k in row and np.isfinite(row[k])]
    return " ".join(pieces)

@flow(name="Cortexa Backfill Strategy Memory")
def backfill_memory_flow(config_path="config.yaml"):
    """
    Loads all historical data and indexes it into the Qdrant
    vector database to build the RAG "strategy memory".
    """
    print("--- 🚀 Starting Strategy Memory Backfill Flow ---")
    
    # 1. Load Config and Artifacts
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config['data_paths']['processed'])
    model_dir = Path(config['ml_models']['saved_models'])
    
    # 2. Load the HMM model
    hmm_model_path = model_dir / "hmm_regime_model.pkl"
    if not hmm_model_path.exists():
        print(f"Error: HMM model not found at {hmm_model_path}")
        return
    hmm_model = joblib.load(hmm_model_path)
    print("Loaded HMM model.")
    
    # 3. Load the ML model metadata (to get feature list)
    meta_path = model_dir / "walkforward" / "lgbm_wf_metadata.json"
    if not meta_path.exists():
        print(f"Error: ML metadata not found at {meta_path}")
        return
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    ml_features = metadata['features']
    print(f"Loaded {len(ml_features)} feature names from metadata.")
    
    # 4. Load all processed feature data
    data_file = processed_path / "features_and_targets.csv"
    if not data_file.exists():
        print(f"Error: Processed data file not found at {data_file}")
        return
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} total historical feature rows.")
    
    # 5. Get Regimes for all GSPC data
    gspc_df = df[df['ticker'] == 'GSPC'].copy()
    
    # --- FIX: Data Validation and Length Check ---
    gspc_df['Close'] = pd.to_numeric(gspc_df['Close'], errors='coerce')
    gspc_df = gspc_df.dropna(subset=['Close'])
    
    if len(gspc_df) < 60: 
        print("FATAL: GSPC data is too short for HMM (requires min 60 days).")
        return 
    # --- END FIX ---
    
    gspc_features = make_regime_features(gspc_df['Close'])
    gspc_regimes = predict_regime(hmm_model, gspc_features)
    gspc_regimes.name = "regime"
    
    # 6. Map regimes to all tickers
    df = df.join(gspc_regimes, how="left")
    df['regime'] = df['regime'].ffill().bfill().astype(int)
    
    # 7. Init RAG Components
    try:
        embedder = Embedder()
        vector_db = QdrantVecDB()
        memory = StrategyMemory(vector_db, embedder)
    except Exception as e:
        print(f"Error initializing RAG components: {e}")
        print("Is Qdrant running?")
        return

    # 8. Prepare and Batch-Upsert Data
    print("Preparing data for indexing...")
    texts_to_embed = []
    metadatas = []
    
    # Columns to save in the payload
    meta_cols = ['ticker', 'regime', 'target'] + ml_features
    
    for index, row in df.iterrows():
        context_text = get_feature_text(row, ml_features)
        
        payload = row[meta_cols].to_dict()
        payload['regime'] = int(payload['regime'])
        payload['target'] = int(payload['target'])
        payload['date'] = index.isoformat()
        
        texts_to_embed.append(context_text)
        metadatas.append(payload)

    # 9. Upsert in batches
    batch_size = 256
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        print(f"Processing batch {i // batch_size + 1}...")
        memory.record_batch(batch_texts, batch_metas)
        
    print(f"\n--- ✅ Strategy Memory Backfill Complete ---")
    print(f"Successfully indexed {len(texts_to_embed)} historical market states.")

if __name__ == "__main__":
    backfill_memory_flow(config_path="config.yaml")