# src/processing/feature_engine_v4.py

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import warnings

# Suppress pandas performance warnings on M1
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

HORIZON = 5          # lookahead days
TARGET_THRESHOLD = 0.01  # > +2% over HORIZON days


# ---------- Econ loader (same style as v3) ----------

def load_econ_data(config):
    raw_path = Path(config["data_paths"]["raw"])
    files = list(raw_path.glob("raw_econ_data_*.csv"))
    if not files:
        return None
    
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    
    # Use engine='python' for better M1 compatibility
    df = pd.read_csv(latest_file, index_col=0, engine='python')
    
    # Robust datetime parsing for M1
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce", utc=False)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df = df.sort_index()
    # Use method parameter for compatibility
    df = df.ffill()
    df.columns = [c.upper().strip() for c in df.columns]
    
    # Ensure no metadata conflicts
    df = df.copy()
    
    return df


# ---------- Core v4 feature builder (cross-sectional) ----------

def build_features_v4(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional feature engine v4.
    Expects raw_df with:
        index: DatetimeIndex
        cols: ['ticker', 'open', 'high', 'low', 'close', 'volume', ...]
    """
    # Deep copy to avoid metadata issues
    df = raw_df.copy(deep=True)

    # Ensure basic numeric types
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"]).sort_index()

    if "ticker" not in df.columns:
        raise ValueError("raw_df must have a 'ticker' column for cross-sectional v4 features.")

    # ---- 1. Per-ticker features ----
    out_list = []

    for ticker, g in df.groupby("ticker", sort=False):
        # Deep copy to prevent metadata propagation
        g = g.sort_index().copy(deep=True)

        # Returns
        g["ret_1d"] = g["close"].pct_change(1)
        g["ret_5d"] = g["close"].pct_change(5)
        g["ret_10d"] = g["close"].pct_change(10)

        # Forward returns
        fwd_price = g["close"].shift(-HORIZON)
        g[f"fwd_ret_{HORIZON}d"] = fwd_price / g["close"] - 1.0

        # Realized vol
        g["vol_5d"] = g["ret_1d"].rolling(5, min_periods=1).std()
        g["vol_20d"] = g["ret_1d"].rolling(20, min_periods=1).std()
        g["vol_ratio_5_20"] = g["vol_5d"] / g["vol_20d"].replace(0, np.nan)

        # Intraday range
        g["hl_range_pct"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)

        # Trend
        g["sma_20"] = g["close"].rolling(20, min_periods=1).mean()
        g["sma_50"] = g["close"].rolling(50, min_periods=1).mean()
        g["dist_sma_20"] = (g["close"] - g["sma_20"]) / g["sma_20"].replace(0, np.nan)
        g["dist_sma_50"] = (g["close"] - g["sma_50"]) / g["sma_50"].replace(0, np.nan)
        g["trend_flag"] = (g["sma_20"] > g["sma_50"]).astype(int)
        g["trend_strength"] = (g["sma_20"] - g["sma_50"]) / g["sma_50"].replace(0, np.nan)
        g["trend_strength_sq"] = g["trend_strength"] ** 2

        # Volume / liquidity
        g["vol_ma_20"] = g["volume"].rolling(20, min_periods=1).mean()
        g["vol_ratio"] = g["volume"] / g["vol_ma_20"].replace(0, np.nan)

        # Optional macro interactions if present
        if "VIXCLS" in g.columns:
            g["vix_chg_1d"] = g["VIXCLS"].pct_change(1)
            g["vix_chg_5d"] = g["VIXCLS"].pct_change(5)
        if "T10Y2Y" in g.columns:
            g["curve_slope"] = g["T10Y2Y"]
            g["curve_chg_5d"] = g["T10Y2Y"].diff(5)
        if "FEDFUNDS" in g.columns:
            g["fedfunds_chg_20d"] = g["FEDFUNDS"].diff(20)

        # Target: binary, big move up
        g["target"] = (g[f"fwd_ret_{HORIZON}d"] > TARGET_THRESHOLD).astype(int)

        out_list.append(g)

    # Use ignore_index=False and make a clean copy
    df_feat = pd.concat(out_list, axis=0, ignore_index=False, copy=True).sort_index()
    
    # Reset metadata by creating new DataFrame
    df_feat = pd.DataFrame(df_feat)

    # ---- 2. Cross-sectional ranks per date ----

    def _cs_rank(s: pd.Series) -> pd.Series:
        # Ensure clean series for ranking
        return s.rank(pct=True, method='average')

    # Use observed=True for groupby on M1 to avoid warnings
    grouped = df_feat.groupby(df_feat.index, observed=True, sort=False)

    df_feat["cs_ret_5d_rank"] = grouped["ret_5d"].transform(_cs_rank)
    df_feat["cs_ret_10d_rank"] = grouped["ret_10d"].transform(_cs_rank)
    df_feat["cs_trend_rank"] = grouped["trend_strength"].transform(_cs_rank)
    df_feat["cs_vol_20d_rank"] = grouped["vol_20d"].transform(_cs_rank)
    df_feat["cs_vol_ratio_rank"] = grouped["vol_ratio"].transform(_cs_rank)
    df_feat["cs_close_rank"] = grouped["close"].transform(_cs_rank)

    # ---- 3. Clean & select columns ----

    df_feat = df_feat.dropna(subset=[f"fwd_ret_{HORIZON}d", "target"])

    base_cols = [
        "ticker",
        "ret_1d", "ret_5d", "ret_10d",
        "vol_5d", "vol_20d", "vol_ratio_5_20",
        "hl_range_pct",
        "sma_20", "sma_50",
        "dist_sma_20", "dist_sma_50",
        "trend_flag", "trend_strength", "trend_strength_sq",
        "vol_ma_20", "vol_ratio",
        "cs_close_rank", "cs_ret_5d_rank", "cs_ret_10d_rank",
        "cs_trend_rank", "cs_vol_20d_rank", "cs_vol_ratio_rank",
    ]

    macro_cols = []
    for col in ["vix_chg_1d", "vix_chg_5d", "curve_slope", "curve_chg_5d", "fedfunds_chg_20d"]:
        if col in df_feat.columns:
            macro_cols.append(col)

    feature_cols = base_cols + macro_cols

    cols_to_keep = feature_cols + [f"fwd_ret_{HORIZON}d", "target"]
    df_out = df_feat[cols_to_keep].copy(deep=True)

    return df_out


# ---------- Orchestrator using your per-ticker raw files ----------

def create_features_v4(config_path: str = "config.yaml"):
    print("--- 🧠 Cortexa v4: Cross-Sectional Features from Individual Ticker CSVs ---")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    econ_df = load_econ_data(config)
    raw_path = Path(config["data_paths"]["raw"])

    all_files = list(raw_path.glob("raw_market_*.csv"))
    if not all_files:
        print("❌ No raw_market_*.csv files found.")
        return

    latest_files = {}
    for f in all_files:
        try:
            # Example filename: raw_market_AAPL.csv -> ticker = 'AAPL'
            parts = f.stem.split("_")
            ticker = parts[2] if len(parts) >= 3 else None
            if not ticker:
                continue
            if ticker not in latest_files or f.stat().st_mtime > latest_files[ticker].stat().st_mtime:
                latest_files[ticker] = f
        except Exception:
            continue

    if not latest_files:
        print("❌ No valid ticker files detected from raw_market_*.csv")
        return

    panel_rows = []

    for ticker, file_path in latest_files.items():
        try:
            # Use engine='python' for M1 compatibility
            df = pd.read_csv(file_path, index_col=0, engine='python')

            # Date handling - robust for M1
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            else:
                if "Date" in df.columns:
                    # Try common date formats
                    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", None]:
                        try:
                            df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors="coerce")
                            break
                        except:
                            continue
                    df = df.set_index("Date")
                else:
                    # Try multiple date formats to avoid dateutil fallback
                    idx_parsed = None
                    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"]:
                        try:
                            idx_parsed = pd.to_datetime(df.index, format=fmt, errors="coerce")
                            if idx_parsed.notna().sum() > len(df) * 0.9:  # If >90% parsed successfully
                                break
                        except:
                            continue
                    
                    if idx_parsed is None or idx_parsed.notna().sum() < len(df) * 0.9:
                        # Fallback to infer_datetime_format
                        idx_parsed = pd.to_datetime(df.index, errors="coerce", infer_datetime_format=True)
                    
                    df.index = idx_parsed

            # Clean timezone info
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df.sort_index()

            # normalize column names
            df.columns = [c.strip().lower() for c in df.columns]

            core_cols = ["open", "high", "low", "close", "volume"]
            missing = [c for c in core_cols if c not in df.columns]
            if missing:
                print(f"Skipping {ticker}: missing columns {missing}")
                continue

            df["ticker"] = ticker

            if econ_df is not None:
                df = df.join(econ_df, how="left")
                df = df.ffill()

            # Deep copy before appending
            panel_rows.append(df.copy(deep=True))

        except Exception as e:
            print(f"Error processing {file_path} for ticker {ticker}: {e}")
            continue

    if not panel_rows:
        print("❌ No usable raw data to build v4 features.")
        return

    # Use copy=True and ignore_index=False
    raw_panel = pd.concat(panel_rows, axis=0, ignore_index=False, copy=True).sort_index()
    # Create fresh DataFrame to clear metadata
    raw_panel = pd.DataFrame(raw_panel)
    
    print(f"Loaded raw panel from individual CSVs: {raw_panel.shape[0]} rows")

    # Build v4 features
    df_v4 = build_features_v4(raw_panel)
    print(f"Built v4 cross-sectional dataset: {df_v4.shape[0]} rows, {df_v4.shape[1]} columns")

    # Save with explicit parameters for M1
    processed_root = Path(config["data_paths"]["processed"])
    processed_root.mkdir(parents=True, exist_ok=True)
    out_file = processed_root / "features_v4.csv"
    
    # Save without index name to avoid metadata issues
    df_v4.to_csv(out_file, date_format='%Y-%m-%d')
    print(f"✅ Saved v4 features to: {out_file}")


if __name__ == "__main__":
    create_features_v4()