# src/processing/feature_engine_v3.py
import importlib
import importlib.metadata
import pandas as pd
import numpy as np
import pandas_ta as ta
import yaml
from pathlib import Path

# --- CONFIGURATION ---
HORIZON = 5         # Look 5 days ahead
TARGET_BAND = 0.0   # Up vs Down

def load_econ_data(config):
    raw_path = Path(config["data_paths"]["raw"])
    files = list(raw_path.glob("raw_econ_data_*.csv"))
    if not files: return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file, index_col=0)
    
    # Explicit date parsing
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    df = df.sort_index().ffill()
    df.columns = [c.upper().strip() for c in df.columns]
    return df

def build_features_v3(raw_df):
    df = raw_df.copy()
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols: 
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_index()

    # --- 1. REGIMES ---
    df["sma_200"] = df["close"].rolling(200).mean()
    df["is_bull"] = (df["close"] > df["sma_200"]).astype(int)
    
    df["ret_1d"] = df["close"].pct_change()
    df["realized_vol_20d"] = df["ret_1d"].rolling(20).std()
    vol_median = df["realized_vol_20d"].median()
    if pd.isna(vol_median): vol_median = 0.01
    df["is_high_vol"] = (df["realized_vol_20d"] > vol_median).astype(int)
    
    df["regime_tag"] = df["is_bull"].astype(str) + "_" + df["is_high_vol"].astype(str)

    # --- 2. FEATURES ---
    df["vol_ratio"] = df["realized_vol_20d"] / df["ret_1d"].rolling(60).std()
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["natr"] = df["atr"] / df["close"]
    
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["dist_sma_50"] = (df["close"] / df["sma_50"]) - 1
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["rvol"] = df["volume"] / df["volume"].rolling(20).mean()

    # --- 3. NEW ALPHA FEATURES (The "Sniper" Additions) ---
    # A. Volatility Change
    df["vol_roll"] = df["ret_1d"].rolling(10).std()
    df["vol_change"] = df["vol_roll"].pct_change()

    # B. 5-Day Range
    df["range_5d"] = (df["high"].rolling(5).max() - df["low"].rolling(5).min()) / df["close"]

    # C. Trend Strength Squared
    df["trend_strength_raw"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]
    df["trend_strength_sq"] = df["trend_strength_raw"] ** 2

    # D. Interaction
    df["trend_x_vol"] = df["trend_strength_raw"] * df["vol_roll"]

    # --- 4. TARGET ---
    future_close = df["close"].shift(-HORIZON)
    df["fwd_ret"] = (future_close / df["close"]) - 1
    df["target"] = (df["fwd_ret"] > TARGET_BAND).astype(int)

    feature_cols = [
        "natr", "vol_ratio", "dist_sma_50", "rsi", "rvol", "ret_1d", "realized_vol_20d",
        "vol_change", "range_5d", "trend_strength_sq", "trend_x_vol"
    ]

    if "VIXCLS" in df.columns:
        df["vix_chg"] = df["VIXCLS"].diff(5)
        feature_cols.append("vix_chg")

    cols_to_keep = feature_cols + ["target", "regime_tag", "ticker"]
    return df.dropna()[cols_to_keep]

def create_features_v3(config_path="config.yaml"):
    print("--- 🧠 Cortexa 3.0: Regime-Based Binary Data ---")
    try:
        with open(config_path, "r") as f: config = yaml.safe_load(f)
    except: return

    econ_df = load_econ_data(config)
    raw_path = Path(config["data_paths"]["raw"])
    all_files = list(raw_path.glob("raw_market_*.csv"))
    
    latest_files = {}
    for f in all_files:
        try:
            t = f.stem.split("_")[2]
            if t not in latest_files or f.stat().st_mtime > latest_files[t].stat().st_mtime:
                latest_files[t] = f
        except: continue

    all_data = []
    for ticker, file_path in latest_files.items():
        if ticker == "GSPC": continue
        try:
            df = pd.read_csv(file_path, index_col=0)
            # Date parsing fix
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
            else:
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
                    df = df.set_index("Date")
                else:
                    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df = df.sort_index()

            df.columns = [c.strip().lower() for c in df.columns]
            df["ticker"] = ticker
            if econ_df is not None: df = df.join(econ_df, how="left").ffill()

            processed = build_features_v3(df)
            all_data.append(processed)
        except Exception: pass

    if all_data:
        final = pd.concat(all_data)
        out = Path(config["data_paths"]["processed"]) / "features_v3.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(out)
        print(f"\n✅ V3 Dataset Created: {len(final)} rows")
    else:
        print("❌ No data processed.")

if __name__ == "__main__":
    create_features_v3()