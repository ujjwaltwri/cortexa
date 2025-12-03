import pandas as pd
import joblib
import yaml
from pathlib import Path
# We import these for the regime calculation logic
from src.regime.hmm_regime import make_regime_features, predict_regime

def get_current_regime(config_path="config.yaml") -> int:
    """
    Loads the saved HMM model and latest market data to predict the current regime.
    """
    try:
        config = yaml.safe_load(open(config_path))
        model_path = Path(config['ml_models']['saved_models']) / "hmm_regime_model.pkl"
        
        if not model_path.exists():
            print("HMM model not found. Defaulting to 0.")
            return 0
            
        hmm_model = joblib.load(model_path)
        
        # 1. Find the latest GSPC raw file
        raw_path = Path(config['data_paths']['raw'])
        gspc_files = list(raw_path.glob("raw_market_GSPC_*.csv"))
        if not gspc_files:
            return 0
            
        raw_file = max(gspc_files, key=lambda f: f.stat().st_mtime)
        
        # 2. LOAD FIX: Read without auto-parsing, then convert explicitly
        gspc_df = pd.read_csv(raw_file, index_col=0)
        gspc_df.index = pd.to_datetime(gspc_df.index, utc=False, errors='coerce')
        
        # 3. Ensure 'Close' is clean and numeric
        # Handle case naming differences (Close vs close)
        if 'Close' in gspc_df.columns:
            close_col = 'Close'
        elif 'close' in gspc_df.columns:
            close_col = 'close'
        else:
            # Fallback to first column if headers are weird
            close_col = gspc_df.columns[0]

        gspc_df[close_col] = pd.to_numeric(gspc_df[close_col], errors='coerce')
        gspc_df = gspc_df.dropna(subset=[close_col])

        # 4. Run HMM feature calculation
        features = make_regime_features(gspc_df[close_col])
        regimes = predict_regime(hmm_model, features)
        
        current_regime = int(regimes.iloc[-1])
        print(f"Loaded current regime: {current_regime}")
        return current_regime
        
    except Exception as e:
        print(f"Error getting current regime: {e}. Defaulting to regime 0.")
        return 0

def get_latest_features(ticker: str, config_path="config.yaml") -> pd.Series | None:
    """
    Loads the latest feature row for a given ticker from the processed CSV.
    """
    try:
        config = yaml.safe_load(open(config_path))
        data_file = Path(config['data_paths']['processed']) / "features_v3.csv"
        
        # Load processed data
        df = pd.read_csv(data_file, index_col=0)
        df.index = pd.to_datetime(df.index, utc=False, errors='coerce')
        
        ticker_df = df[df['ticker'].str.upper() == ticker.upper()]
        if ticker_df.empty:
            print(f"No feature data found for ticker: {ticker}")
            return None
            
        return ticker_df.iloc[-1] 
        
    except Exception as e:
        print(f"Error getting latest features: {e}")
        return None