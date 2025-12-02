import pandas as pd
import joblib
import yaml
from pathlib import Path
from src.regime.hmm_regime import make_regime_features, predict_regime

def get_current_regime(config_path="config.yaml") -> int:
    try:
        config = yaml.safe_load(open(config_path))
        model_path = Path(config['ml_models']['saved_models']) / "hmm_regime_model.pkl"
        
        if not model_path.exists():
            print("HMM model not found. Defaulting to 0.")
            return 0
            
        hmm_model = joblib.load(model_path)
        
        # Load Raw GSPC
        raw_path = Path(config['data_paths']['raw'])
        gspc_files = list(raw_path.glob("raw_market_GSPC_*.csv"))
        if not gspc_files: return 0
        raw_file = max(gspc_files, key=lambda f: f.stat().st_mtime)
        
        # Robust Load
        gspc_df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
        gspc_df['Close'] = pd.to_numeric(gspc_df['Close'], errors='coerce')
        gspc_df = gspc_df.dropna(subset=['Close'])

        features = make_regime_features(gspc_df['Close'])
        regimes = predict_regime(hmm_model, features)
        
        current_regime = int(regimes.iloc[-1])
        print(f"Loaded current regime: {current_regime}")
        return current_regime
        
    except Exception as e:
        print(f"Error getting current regime: {e}. Defaulting to 0.")
        return 0

def get_latest_features(ticker: str, config_path="config.yaml") -> pd.Series | None:
    try:
        config = yaml.safe_load(open(config_path))
        data_file = Path(config['data_paths']['processed']) / "features_and_targets.csv"
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        ticker_df = df[df['ticker'].str.upper() == ticker.upper()]
        if ticker_df.empty: return None
        return ticker_df.iloc[-1] 
    except Exception as e:
        print(f"Error getting latest features: {e}")
        return None
