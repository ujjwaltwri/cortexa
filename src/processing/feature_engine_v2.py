# src/processing/feature_engine_v2.py

# --- CRITICAL FIX for Python 3.10 / M1 Macs ---
import importlib
import importlib.metadata
# ---------------------------------------------

import pandas as pd
import numpy as np
import pandas_ta as ta
import yaml
from pathlib import Path

# --- V2 Configuration ---
DEFAULT_CONFIG_PATH = "config.yaml"
FUTURE_DAYS = 1       # Predict 1 Day ahead
TARGET_BAND = 0.001   # 0.1% Minimum move


def load_econ_data(config):
    """
    Load latest economic data CSV and engineer macro features.
    Uses explicit datetime parsing to avoid UserWarnings and slow per-row parsing.
    """
    raw_path = Path(config["data_paths"]["raw"])
    files = list(raw_path.glob("raw_econ_data_*.csv"))
    if not files:
        return None

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading Economic Data from: {latest_file.name}")

    # Read WITHOUT automatic date parsing; handle dates explicitly
    df = pd.read_csv(latest_file, index_col=0)

    # Explicit, fast, warning-free datetime parsing
    # Assumes standard finance-style YYYY-MM-DD index
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")

    # Drop timezone if somehow present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    df = df.ffill()
    df.columns = [c.upper().strip() for c in df.columns]

    # Macro features
    if "VIXCLS" in df.columns:
        df["vix_shock"] = df["VIXCLS"].pct_change(5)
    if "FEDFUNDS" in df.columns:
        df["rate_stress"] = df["FEDFUNDS"].diff(20)

    return df


def build_features_v2(raw_df, horizon=FUTURE_DAYS, target_threshold=TARGET_BAND):
    """
    The Core V2 Logic: Converts raw prices into stationary, relative features.
    """
    df = raw_df.copy()

    # --- FIX: Force Numeric Types ---
    cols_to_numeric = ["open", "high", "low", "close", "volume"]
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])
    # --------------------------------

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # 2. Returns (Stationary)
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)

    # 3. Volatility Ratios
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_ratio"] = df["vol_5d"] / df["vol_20d"]

    # 4. Trend Distance
    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    df["dist_sma_20"] = (df["close"] / sma_20) - 1
    df["dist_sma_50"] = (df["close"] / sma_50) - 1
    df["trend_flag"] = (sma_20 > sma_50).astype(int)

    # 5. Volume
    vol_ma = df["volume"].rolling(20).mean()
    df["rvol"] = df["volume"] / vol_ma
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # 6. Target
    future_close = df["close"].shift(-horizon)
    df["fwd_ret"] = (future_close / df["close"]) - 1
    df["target"] = (df["fwd_ret"] > target_threshold).astype(int)

    # 7. Select Final Columns
    features = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "vol_5d",
        "vol_ratio",
        "hl_range_pct",
        "dist_sma_20",
        "dist_sma_50",
        "trend_flag",
        "rvol",
    ]

    if "vix_shock" in df.columns:
        features.append("vix_shock")
    if "rate_stress" in df.columns:
        features.append("rate_stress")

    cols_to_keep = features + ["target"]
    if "ticker" in df.columns:
        cols_to_keep.append("ticker")

    df = df.dropna()

    return df[cols_to_keep]


def create_features_v2(config_path=DEFAULT_CONFIG_PATH):
    print("--- 🧠 Cortexa: Building V2 (Stationary) Data Engine ---")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    econ_df = load_econ_data(config)
    raw_path = Path(config["data_paths"]["raw"])

    # Find the latest file per ticker
    all_files = list(raw_path.glob("raw_market_*.csv"))
    latest_files = {}
    for f in all_files:
        try:
            # Expected pattern: raw_market_<TICKER>_....
            ticker = f.stem.split("_")[2]
            if ticker not in latest_files or f.stat().st_mtime > latest_files[ticker].stat().st_mtime:
                latest_files[ticker] = f
        except Exception:
            continue

    all_data = []
    for ticker, file_path in latest_files.items():
        # Skip the benchmark itself if you're treating it differently
        if ticker == "GSPC":
            continue

        try:
            # Load Raw WITHOUT automatic date parsing
            df = pd.read_csv(file_path, index_col=0)

            # --- DATE PARSING FIX (clean + explicit) ---
            if isinstance(df.index, pd.DatetimeIndex):
                # Already parsed; just ensure tz-naive
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            else:
                # If a 'Date' column exists, prefer that
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(
                        df["Date"],
                        format="%Y-%m-%d",
                        errors="coerce",
                    )
                    df = df.set_index("Date")
                else:
                    # Fallback: parse the index as dates
                    df.index = pd.to_datetime(
                        df.index,
                        format="%Y-%m-%d",
                        errors="coerce",
                    )

            # Ensure sorted, tz-naive index
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            # ------------------------

            df.columns = [c.strip().lower() for c in df.columns]
            df["ticker"] = ticker

            # Join with econ data if available
            if econ_df is not None:
                df = df.join(econ_df, how="left").ffill()

            processed_df = build_features_v2(df)
            all_data.append(processed_df)

        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if all_data:
        final_df = pd.concat(all_data)
        out_path = Path(config["data_paths"]["processed"]) / "features_and_targets.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_path)
        print(f"\n✅ Cortexa V2 Dataset Created: {len(final_df)} rows")
    else:
        print("❌ No data processed.")


if __name__ == "__main__":
    create_features_v2()
