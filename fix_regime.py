import os
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

# --- 1. OVERWRITE src/regime/hmm_regime.py ---
print("1. Overwriting src/regime/hmm_regime.py...")
with open("src/regime/hmm_regime.py", "w") as f:
    f.write('''import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import yaml
from pathlib import Path

def make_regime_features(prices: pd.Series) -> pd.DataFrame:
    """
    Creates the feature set for the HMM (returns, volatility, momentum).
    """
    r = prices.pct_change(fill_method=None).dropna()
    vol = r.rolling(20).std()
    mom = prices.pct_change(20, fill_method=None)
    
    X = pd.concat([r, vol, mom], axis=1).dropna()
    X.columns = ["return", "vol", "momentum"]
    return X

def fit_hmm(X: pd.DataFrame, n_states=3, random_state=42) -> GaussianHMM:
    """
    Fits a Gaussian Hidden Markov Model.
    """
    print(f"--- Fitting HMM with {n_states} states ---")
    hmm = GaussianHMM(
        n_components=n_states, 
        covariance_type="full", 
        n_iter=100, 
        random_state=random_state
    )
    hmm.fit(X.values)
    return hmm

def predict_regime(hmm: GaussianHMM, X: pd.DataFrame) -> pd.Series:
    """
    Predicts the hidden regime for each data point.
    """
    states = hmm.predict(X.values)
    return pd.Series(states, index=X.index, name="regime")
''')

# --- 2. OVERWRITE src/utils.py ---
print("2. Overwriting src/utils.py...")
with open("src/utils.py", "w") as f:
    f.write('''import pandas as pd
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
''')

# --- 3. RE-TRAIN AND SAVE MODEL ---
print("3. Regenerating HMM Model File...")
# Import the functions we just wrote to file
from src.regime.hmm_regime import make_regime_features, fit_hmm

config = yaml.safe_load(open("config.yaml"))
raw_path = Path(config['data_paths']['raw'])
gspc_files = list(raw_path.glob("raw_market_GSPC_*.csv"))
raw_file = max(gspc_files, key=lambda f: f.stat().st_mtime)

df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

features = make_regime_features(df['Close'])
hmm_model = fit_hmm(features, n_states=3)

save_path = Path(config['ml_models']['saved_models']) / "hmm_regime_model.pkl"
joblib.dump(hmm_model, save_path)
print(f"✅ Clean HMM model saved to: {save_path}")
print("\n--- FIX COMPLETE. Restart server.py now. ---")