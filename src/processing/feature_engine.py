# --- Ensure importlib.metadata is loaded on Python 3.10 before pandas_ta ---
import importlib
import importlib.metadata  # important: must come before pandas_ta on 3.10

import glob
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import pandas_ta as ta  # uses pandas-ta-openbb

pd.options.mode.copy_on_write = True

# --- (Helper functions _find_cols, _coerce_numeric, _read_one_csv are unchanged) ---
def _find_cols(df: pd.DataFrame):
    """
    Map common OHLCV column variants to a standard form.
    Returns a dict like {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}.
    """
    candidates = {
        "open":   ["Open", "open", "OPEN", "o"],
        "high":   ["High", "high", "HIGH", "h"],
        "low":    ["Low", "low", "LOW", "l"],
        "close":  ["Close", "close", "CLOSE", "c", "Adj Close", "adj_close", "adj close"],
        "volume": ["Volume", "volume", "VOLUME", "v", "Vol"],
    }
    mapping = {}
    for k, opts in candidates.items():
        for c in opts:
            if c in df.columns:
                mapping[k] = c
                break
    if "close" not in mapping:
        raise ValueError("Could not find a 'Close' column (tried Close/Adj Close/etc.).")
    return mapping

def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
        df = df.set_index("Date")
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=False)
        df = df.set_index("Datetime")
    else:
        first_col = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first_col], errors="coerce", utc=False)
            if parsed.notna().mean() > 0.8:
                df[first_col] = parsed
                df = df.set_index(first_col)
        except Exception:
            pass

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")].sort_index()

    df.columns = [c.strip() for c in df.columns]
    return df
# --- (End of unchanged helper functions) ---


# --- UPDATED _add_indicators function ---
def _add_indicators(df: pd.DataFrame, cols, market_returns: pd.Series = None):
    """Rich indicator set with robust extras."""
    close = df[cols["close"]]
    hi = cols.get("high", cols["close"])
    lo = cols.get("low", cols["close"])
    vol = cols.get("volume")
    
    # Get stock returns
    r = close.pct_change(fill_method=None)
    df["ret_1"]  = r

    # --- NEW: Market-Relative Features ---
    if market_returns is not None:
        # Align market returns to this stock's index
        market_r = market_returns.reindex(df.index).ffill()
        
        # 1. Rolling Correlation
        df["corr_60d"] = r.rolling(60).corr(market_r)
        
        # 2. Rolling Beta
        # Beta = Cov(Stock_r, Market_r) / Var(Market_r)
        rolling_cov = r.rolling(60).cov(market_r)
        rolling_var = market_r.rolling(60).var()
        df["beta_60d"] = rolling_cov / rolling_var
    # --- END OF NEW FEATURES ---

    # Classic trend/momentum/volatility
    df["SMA_20"] = ta.sma(close, length=20)
    df["SMA_50"] = ta.sma(close, length=50)
    df["RSI_14"] = ta.rsi(close, length=14)
    df.ta.macd(close=cols["close"], fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(close=cols["close"], length=20, std=2, append=True)
    df["EMA_10"] = ta.ema(close, length=10)
    df["EMA_50"] = ta.ema(close, length=50)
    df["EMA_GAP"] = df["EMA_10"] / df["EMA_50"] - 1
    df.ta.adx(high=hi, low=lo, close=cols["close"], length=14, append=True)
    df["ATR_14"] = ta.atr(high=hi, low=lo, close=cols["close"], length=14)
    df.ta.stoch(high=hi, low=lo, close=cols["close"], k=14, d=3, smooth_k=3, append=True)

    # Returns/momentum/vol/range
    df["ret_5"]  = close.pct_change(5, fill_method=None)
    df["mom_10"] = close / close.shift(10) - 1
    df["vol_10"] = r.rolling(10).std()
    df["rng_1"]  = (df[hi] - df[lo]) / close

    # Extra alpha features
    df["zret_20"] = (r - r.rolling(20).mean()) / r.rolling(20).std()
    df["dist_20h"] = close / close.rolling(20).max() - 1
    df["dist_20l"] = close / close.rolling(20).min() - 1
    df["skew_20"] = r.rolling(20).skew()
    df["kurt_20"] = r.rolling(20).kurt()

    if vol in df.columns:
        v = pd.to_numeric(df[vol], errors="coerce")
        df["OBV"] = ta.obv(close=cols["close"], volume=vol)
        df["MFI_14"] = ta.mfi(high=hi, low=lo, close=cols["close"], volume=vol, length=14)
        df["vol_ratio_20"] = v / v.rolling(20).mean() - 1

    return df

# --- UPDATED create_features function ---
def create_features(config_path="config.yaml", future_days=10, band=0.005):
    """
    Loads raw market data, engineers features (including market-relative),
    and saves to processed folder.
    """
    print("--- Starting Feature Engineering (with Beta/Corr) ---")

    # 1) Load config
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        raw_path = Path(config["data_paths"]["raw"])
        processed_path = Path(config["data_paths"]["processed"])
    except Exception as e:
        print(f"Error reading config or paths: {e}")
        return

    processed_path.mkdir(parents=True, exist_ok=True)

    # 2) Gather files
    market_files = sorted(glob.glob(str(raw_path / "raw_market_*.csv")))
    if not market_files:
        print(f"No raw market files found in {raw_path}. Exiting.")
        return
    print(f"Found {len(market_files)} market data files to process.")
    
    # --- NEW LOGIC: Load all data to get GSPC first ---
    all_data = {}
    for file_path in market_files:
        try:
            ticker = Path(file_path).stem.split("_")[2]
        except Exception:
            ticker = Path(file_path).stem
        
        df = _read_one_csv(file_path)
        if not df.empty:
            all_data[ticker] = df
    
    # Get GSPC (market) returns
    if "GSPC" not in all_data:
        print("Warning: GSPC data not found. Cannot calculate market-relative features.")
        market_returns = None
    else:
        gspc_cols = _find_cols(all_data["GSPC"])
        _coerce_numeric(all_data["GSPC"], [gspc_cols['close']])
        market_returns = all_data["GSPC"][gspc_cols['close']].pct_change(fill_method=None)
        market_returns.name = "market_ret"
        print("Market (GSPC) returns loaded for Beta/Corr calculation.")
    # --- END OF NEW LOGIC ---

    all_processed, error_count = [], 0

    # 3) Process each file
    for ticker, df in all_data.items():
        if ticker == "GSPC": # We don't need to process the market against itself
            continue 
            
        print(f"  - Processing {ticker}...")
        try:
            cols = _find_cols(df)
            _coerce_numeric(df, [
                cols.get("open", ""), cols.get("high", ""), cols.get("low", ""),
                cols["close"], cols.get("volume", "")
            ])

            # Add indicators, now passing in market returns
            df = _add_indicators(df, cols, market_returns)

            # Target: banded future return
            close = df[cols["close"]]
            df["future_price"] = close.shift(-future_days)
            df["future_ret"]   = df["future_price"] / close - 1.0
            df["target"]       = (df["future_ret"] > band).astype("Int8")

            df["ticker"] = ticker

            # Robust cleanup
            all_nan_cols = [c for c in df.columns if df[c].isna().all()]
            if all_nan_cols:
                df = df.drop(columns=all_nan_cols)
            df = df.dropna(subset=["target"])
            df = df.dropna()

            if df.empty:
                raise ValueError("All rows dropped after indicators/target (too short?)")

            # Keep a tidy subset
            keep_cols = [c for k, c in cols.items()
                         if k in {"open", "high", "low", "close", "volume"} and c in df.columns]
            feature_cols = [c for c in df.columns if c not in keep_cols]
            df = df[keep_cols + feature_cols]

            all_processed.append(df)

        except Exception as e:
            error_count += 1
            print(f"    Error processing {ticker}: {e}")

    if all_processed:
        combined = pd.concat(all_processed, axis=0)
        out_path = processed_path / "features_and_targets.csv"
        combined.to_csv(out_path)
        print(f"\nSuccessfully processed {len(all_processed)} tickers, errors: {error_count}")
        print(f"Saved combined feature data to: {out_path}")
        print("--- Feature Engineering Complete ---\n")
        print("Data Head:")
        print(combined.head(10))
    else:
        print(f"No data was processed. Errors: {error_count}")


if __name__ == "__main__":
    # Run: python -m src.processing.feature_engine
    create_features(config_path="config.yaml")