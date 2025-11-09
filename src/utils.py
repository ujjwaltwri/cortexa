# src/utils.py
from __future__ import annotations
import pandas as pd
import joblib
import yaml
from pathlib import Path
from src.regime.hmm_regime import make_regime_features, predict_regime, RegimeModel


# ============================================================
# 1️⃣ Date Parsing Helpers (no deprecated arguments)
# ============================================================

_KNOWN_DT_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M:%S",
)

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    """Try common formats first, then flexible parsing."""
    for fmt in _KNOWN_DT_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=False)
        if parsed.notna().mean() >= 0.8:
            return parsed.dt.tz_localize(None)
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    return parsed.dt.tz_localize(None)


# ============================================================
# 2️⃣ Robust Market Data Loader (used by HMM + RAG)
# ============================================================

DATE_NAME_HINTS = {"date", "datetime", "timestamp", "time"}

def _find_and_parse_datetime_col(df: pd.DataFrame) -> str:
    """Find a real datetime column and parse it safely."""
    if df.empty:
        raise ValueError("Empty DataFrame—no columns to detect date from.")

    candidates = [c for c in df.columns if c.lower() in DATE_NAME_HINTS]
    first_col = df.columns[0]
    if first_col not in candidates:
        candidates = [first_col] + candidates

    tried = set()
    for col in candidates + [c for c in df.columns if c not in candidates]:
        if col in tried:
            continue
        tried.add(col)

        if df[col].dtype.kind in {"i", "u", "f", "b"}:
            continue

        parsed = _parse_datetime_series(df[col])
        if parsed.notna().mean() >= 0.8:
            df[col] = parsed
            return col

    raise ValueError(
        f"Could not find a parseable datetime column. Columns were: {list(df.columns)}"
    )


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_latest_gspc_df(config_path: str | Path) -> pd.DataFrame:
    """
    Loads the most recently modified raw_market_GSPC_*.csv file
    and returns a clean, indexed DataFrame.
    """
    config = _load_yaml(config_path)
    raw_dir = Path(config["data_paths"]["raw"])
    gspc_files = sorted(raw_dir.glob("raw_market_GSPC_*.csv"), key=lambda p: p.stat().st_mtime)
    if not gspc_files:
        raise FileNotFoundError(f"No raw_market_GSPC_*.csv found in {raw_dir}")

    raw_file = gspc_files[-1]
    df = pd.read_csv(raw_file, engine="python")

    # Find close-like column
    close_col = None
    for cand in ["Close", "close", "Adj Close", "Adj_Close", "adj_close"]:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        raise KeyError(f"No 'Close' column found in {raw_file}. Columns: {list(df.columns)}")

    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    # Parse date column
    if "Date" in df.columns:
        date_col = "Date"
        df[date_col] = _parse_datetime_series(df[date_col])
    elif "Datetime" in df.columns:
        date_col = "Datetime"
        df[date_col] = _parse_datetime_series(df[date_col])
    else:
        date_col = _find_and_parse_datetime_col(df)

    df = df[df[date_col].notna() & df[close_col].notna()].copy()
    df.sort_values(date_col, inplace=True)
    df.drop_duplicates(subset=[date_col], keep="last", inplace=True)
    df.set_index(date_col, inplace=True)
    df.index.name = "Date"

    if close_col != "Close":
        df.rename(columns={close_col: "Close"}, inplace=True)

    return df


# ============================================================
# 3️⃣ Regime Detection (uses saved HMM + scaler)
# ============================================================

def get_current_regime(config_path="config.yaml") -> int:
    """
    Loads the saved HMM RegimeModel and latest GSPC data,
    predicts the current market regime (0/1/2).
    """
    try:
        config = _load_yaml(config_path)
        model_path = Path(config["ml_models"]["saved_models"]) / "hmm_regime_model.pkl"

        # Load RegimeModel bundle (HMM + scaler)
        model = RegimeModel.load(model_path)

        # Load market data
        gspc_df = _load_latest_gspc_df(config_path)
        if len(gspc_df) < 50:
            print(f"Not enough market rows ({len(gspc_df)}) — defaulting to regime 0.")
            return 0

        # Compute features & predict
        features = make_regime_features(gspc_df["Close"])
        regimes = predict_regime(model, features)

        current_regime = int(regimes.iloc[-1])
        print(f"Loaded current regime: {current_regime}")
        return current_regime

    except Exception as e:
        print(f"Error getting current regime: {e}. Defaulting to regime 0.")
        return 0


# ============================================================
# 4️⃣ Latest Ticker Feature Loader (used by RAG Signal)
# ============================================================

def get_latest_features(ticker: str, config_path="config.yaml") -> pd.Series | None:
    """
    Loads the most recent feature row for a given ticker
    from processed/features_and_targets.csv.
    """
    try:
        config = _load_yaml(config_path)
        data_file = Path(config["data_paths"]["processed"]) / "features_and_targets.csv"
        df = pd.read_csv(data_file, index_col=0)

        # Parse index if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            parsed_idx = pd.to_datetime(df.index, errors="coerce", utc=False)
            if parsed_idx.notna().any():
                df.index = parsed_idx.tz_localize(None)
            else:
                date_col = None
                for c in df.columns:
                    if c.lower() in DATE_NAME_HINTS:
                        date_col = c
                        break
                if date_col is None:
                    date_col = _find_and_parse_datetime_col(df)
                df[date_col] = _parse_datetime_series(df[date_col])
                df = df[df[date_col].notna()].copy()
                df.set_index(date_col, inplace=True)

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        if "ticker" not in df.columns:
            raise KeyError(f"'ticker' column not found in {data_file}. Columns: {list(df.columns)}")

        ticker_df = df[df["ticker"].str.upper() == ticker.upper()]
        if ticker_df.empty:
            print(f"No feature data found for ticker: {ticker}")
            return None

        return ticker_df.iloc[-1]

    except Exception as e:
        print(f"Error getting latest features: {e}")
        return None
