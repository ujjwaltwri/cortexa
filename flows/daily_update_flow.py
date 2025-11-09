from prefect import flow, task
from pathlib import Path
import pandas as pd
import yaml
import joblib 

# 1. Import all your project's functions
from src.data_ingestion.rss_client import fetch_rss_news
from src.data_ingestion.yfinance_client import fetch_market_data
from src.data_ingestion.fred_client import fetch_economic_data
from src.processing.feature_engine import create_features
from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB

# Import HMM functions
from src.regime.hmm_regime import make_regime_features, fit_hmm, predict_regime

# --- Task Definitions ---

@task(name="Fetch Market Prices (yfinance)")
def fetch_market_data_task(config_path="config.yaml"):
    fetch_market_data(config_path)

@task(name="Fetch Economic Data (FRED)")
def fetch_econ_data_task(config_path="config.yaml"):
    fetch_economic_data(config_path)

@task(name="Fetch News (RSS)", retries=3, retry_delay_seconds=10)
def fetch_news_task(config_path="config.yaml") -> pd.DataFrame | None:
    return fetch_rss_news(config_path)

@task(name="Run Feature Engineering")
def run_feature_engineering_task(config_path="config.yaml"):
    create_features(config_path)

@task(name="Get Current Market Regime (HMM)")
def get_current_regime_task(config_path="config.yaml") -> int:
    """
    Loads market data, fits the HMM, and returns the latest regime.
    """
    print("--- Starting Regime Detection Task ---")
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config['data_paths']['processed'])
    
    # Let's find the latest GSPC file instead of hardcoding
    gspc_files = list(Path(config['data_paths']['raw']).glob("raw_market_GSPC_*.csv"))
    if not gspc_files:
        print("Error: No GSPC market file found. Cannot run HMM.")
        return 0 # Default to regime 0

    raw_file = max(gspc_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading HMM base data from: {raw_file}")
    
    df = pd.read_csv(raw_file)
    
    # --- THIS IS THE FIX ---
    # Convert 'Close' column to numeric, as it's being read as a string
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    # --- END OF FIX ---
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    features = make_regime_features(df['Close'])
    hmm_model = fit_hmm(features, n_states=3)
    regimes = predict_regime(hmm_model, features)
    
    current_regime = int(regimes.iloc[-1])
    print(f"--- Current Market Regime Detected: {current_regime} ---")
    
    model_path = Path(config['ml_models']['saved_models']) / "hmm_regime_model.pkl"
    joblib.dump(hmm_model, model_path)
    print(f"Saved HMM model to {model_path}")
    
    return current_regime


@task(name="Index News to Vector DB")
def index_news_task(
    news_df: pd.DataFrame, 
    current_regime: int,  # New argument
    config_path="config.yaml"
):
    """
    Takes the news DataFrame, embeds it, and upserts to Qdrant
    with the current market regime.
    """
    if news_df is None or news_df.empty:
        print("No news to index. Skipping.")
        return

    print(f"--- Starting News Indexing Task for {len(news_df)} articles ---")
    print(f"--- Tagging all articles with REGIME: {current_regime} ---")
    
    embedder = Embedder() 
    vector_db = QdrantVecDB() 
    
    texts_to_embed = (news_df['title'] + " - " + news_df['summary']).tolist()
    vectors = embedder.encode(texts_to_embed)
    metadatas = news_df.to_dict('records')
    
    for meta in metadatas:
        meta['event_type'] = 'news'
        meta['regime'] = current_regime 
        
    vector_db.upsert(texts_to_embed, vectors, metadatas)
    print(f"--- News Indexing Complete ---")

# --- Main Flow (Unchanged) ---

@flow(name="Cortexa Daily Update")
def daily_update_flow(config_path="config.yaml"):
    """
    Main orchestration flow for Cortexa.
    Ingest -> ETL -> Get Regime -> Index Vector DB
    """
    print("--- 🚀 Starting Cortexa Daily Update Flow ---")
    
    # 1. Fetch all data sources
    news_data_future = fetch_news_task(config_path)
    market_data_future = fetch_market_data_task(config_path)
    econ_data_future = fetch_econ_data_task(config_path)
    
    # 2. Run Feature Engineering (depends on market/econ data)
    feature_future = run_feature_engineering_task(
        config_path, 
        wait_for=[market_data_future, econ_data_future]
    )
    
    # 3. Get Current Regime (depends on market data)
    regime_future = get_current_regime_task(
        config_path,
        wait_for=[market_data_future] 
    )
    
    # 4. Index News (depends on news data AND regime)
    index_future = index_news_task(
        news_df=news_data_future,
        current_regime=regime_future, 
        config_path=config_path
    )
    
    print("--- ✅ Cortexa Daily Update Flow Complete ---")

if __name__ == "__main__":
    daily_update_flow(config_path="config.yaml")