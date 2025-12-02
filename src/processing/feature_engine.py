# --- CRITICAL FIX for Python 3.10 / M1 Macs ---
import importlib
import importlib.metadata 
# ---------------------------------------------

import pandas as pd
import numpy as np
import pandas_ta as ta
import yaml
import glob
from pathlib import Path

# --- Configuration defaults ---
DEFAULT_CONFIG_PATH = "config.yaml"
FUTURE_DAYS = 1     
TARGET_BAND = 0.001 

def load_econ_data(config):
    raw_path = Path(config["data_paths"]["raw"])
    files = list(raw_path.glob("raw_econ_data_*.csv"))
    if not files: return None
    
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading Economic Data from: {latest_file.name}")
    
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    df = df.ffill()
    df.columns = [c.upper() for c in df.columns]
    
    if 'VIXCLS' in df.columns:
        df['VIX_change'] = df['VIXCLS'].diff(10)
        df['VIX_ROC'] = df['VIXCLS'].pct_change(5)

    if 'FEDFUNDS' in df.columns:
        df['Rates_Delta'] = df['FEDFUNDS'].diff(60) 
        
    return df

def process_ticker(file_path, econ_df):
    try:
        df = pd.read_csv(file_path)
        
        # Robust Date Parsing
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=False).dt.tz_localize(None)
            df = df.set_index("Date")
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=False).dt.tz_localize(None)
            df = df.set_index(df.columns[0])

        df.columns = [c.strip().lower() for c in df.columns]
        
        if econ_df is not None:
            df = df.join(econ_df, how="left").ffill()

        # Features
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['natr'] = (df['atr'] / df['close']) * 100 
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            df['trend_strength'] = adx['ADX_14']

        df['rsi'] = ta.rsi(df['close'], length=14)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['dist_from_mean'] = (df['close'] - df['sma_50']) / df['sma_50']

        # Target
        df['future_close'] = df['close'].shift(-FUTURE_DAYS)
        df['future_return'] = (df['future_close'] / df['close']) - 1
        df['target'] = (df['future_return'] > TARGET_BAND).astype(int)
        
        ticker = Path(file_path).stem.split('_')[2]
        df['ticker'] = ticker
        
        return df.dropna()

    except Exception as e:
        # print(f"Skipping {file_path.name}: {e}")
        return None

# --- REWRITTEN FUNCTION FOR EXTERNAL IMPORT ---
def create_features(config_path=DEFAULT_CONFIG_PATH):
    print("--- 🧠 Cortexa: Building Advanced Data Engine ---")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    econ_df = load_econ_data(config)
    
    raw_path = Path(config["data_paths"]["raw"])
    stock_files = list(raw_path.glob("raw_market_*.csv"))
    
    all_data = []
    for f in stock_files:
        if "GSPC" in f.name: continue
        processed_df = process_ticker(f, econ_df)
        if processed_df is not None:
            all_data.append(processed_df)
            
    if all_data:
        final_df = pd.concat(all_data)
        out_path = Path(config["data_paths"]["processed"]) / "features_and_targets.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_path)
        print(f"\n✅ Created Advanced Dataset: {len(final_df)} rows")
        print("   Added Features: VIX_change, Rates_Delta, NATR, ADX")
    else:
        print("❌ No data processed.")

# Allow running directly as a script too
if __name__ == "__main__":
    create_features()