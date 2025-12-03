# src/processing/feature_engine.py

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

# --- Cortexa 2.0 Configuration ---
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

        # --- CORTEXA 2.0: VOLUME ALPHA FEATURES ---
        
        # 1. RVOL (Relative Volume) - Detecting Breakouts
        # Is today's volume higher than the 20-day average?
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['rvol'] = df['volume'] / df['vol_ma20']
        
        # 2. Rolling VWAP Distance (Institutional Value)
        # Standard VWAP resets daily. We want a "Rolling VWAP" to see the monthly trend.
        # Formula: Sum(Price * Vol) / Sum(Vol) over N days
        vwap_window = 20
        df['pv'] = df['close'] * df['volume']
        df['rolling_vwap'] = df['pv'].rolling(vwap_window).sum() / df['volume'].rolling(vwap_window).sum()
        df['dist_vwap'] = (df['close'] / df['rolling_vwap']) - 1

        # 3. Force Index (Buying Pressure)
        # (Close - PrevClose) * Volume
        df['force_index'] = df['close'].diff(1) * df['volume']
        # Smooth it to remove noise
        df['force_index_ema'] = ta.ema(df['force_index'], length=13)

        # --- EXISTING TECHNICALS ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['natr'] = (df['atr'] / df['close']) * 100 
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            df['trend_strength'] = adx['ADX_14']

        df['rsi'] = ta.rsi(df['close'], length=14)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['dist_from_mean'] = (df['close'] - df['sma_50']) / df['sma_50']

        # --- Target ---
        df['future_close'] = df['close'].shift(-FUTURE_DAYS)
        df['future_return'] = (df['future_close'] / df['close']) - 1
        df['target'] = (df['future_return'] > TARGET_BAND).astype(int)
        
        ticker = Path(file_path).stem.split('_')[2]
        df['ticker'] = ticker
        
        return df.dropna()

    except Exception as e:
        return None

def create_features(config_path=DEFAULT_CONFIG_PATH):
    print("--- 🧠 Cortexa 2.0: Building Institutional Data Engine ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    econ_df = load_econ_data(config)
    raw_path = Path(config["data_paths"]["raw"])
    
    # Filter for unique tickers to avoid duplicates
    all_files = list(raw_path.glob("raw_market_*.csv"))
    latest_files = {}
    for f in all_files:
        try:
            parts = f.stem.split('_')
            ticker = parts[2]
            if ticker not in latest_files or f.stat().st_mtime > latest_files[ticker].stat().st_mtime:
                latest_files[ticker] = f
        except: continue

    print(f"Found {len(latest_files)} unique tickers.")
    
    all_data = []
    for ticker, file_path in latest_files.items():
        if ticker == "GSPC": continue
        processed_df = process_ticker(file_path, econ_df)
        if processed_df is not None:
            all_data.append(processed_df)
            
    if all_data:
        final_df = pd.concat(all_data)
        out_path = Path(config["data_paths"]["processed"]) / "features_and_targets.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_path)
        print(f"\n✅ Cortexa 2.0 Dataset Created: {len(final_df)} rows")
        print("   New Features: RVOL, Rolling VWAP, Force Index")
    else:
        print("❌ No data processed.")

if __name__ == "__main__":
    create_features()