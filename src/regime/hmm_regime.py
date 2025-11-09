# src/regime/hmm_regime.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import yaml


# --------------------------- Config knobs --------------------------- #

MIN_ROWS = 300            # minimum rows required for stable features (adjust if needed)
RET_WIN_VOL = 20          # rolling window for volatility
MOM_WIN = 20              # momentum lookback
HMM_STATES = 3
RANDOM_STATE = 42

DATE_CANDIDATES = ("Date", "date", "Datetime", "datetime", "timestamp", "time")
MARKET_TICKER_CANDIDATES = ("GSPC", "^GSPC", "SPY")  # add your own if needed


# --------------------------- Utilities --------------------------- #

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a clean, ascending DatetimeIndex with unique stamps."""
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse existing index
        maybe = pd.to_datetime(df.index, errors="coerce", utc=False)
        if maybe.notna().any():
            df = df.copy()
            df.index = maybe.tz_localize(None)
        else:
            # Fall back to an obvious date-like column if present
            for c in df.columns:
                lc = c.lower()
                if lc in {"date", "datetime", "timestamp", "time"}:
                    s = pd.to_datetime(df[c], errors="coerce", utc=False)
                    df = df[s.notna()].copy()
                    df.index = s[s.notna()].dt.tz_localize(None)
                    break
            else:
                raise ValueError("Could not establish a DatetimeIndex.")
    # sort + dedupe
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _pick_close_column(df: pd.DataFrame) -> str:
    for cand in ["Close", "close", "Adj Close", "Adj_Close", "adj_close"]:
        if cand in df.columns:
            return cand
    raise KeyError(f"No close-like column in columns: {list(df.columns)}")


def _find_ticker_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() == "ticker":
            return c
    raise KeyError(f"No 'ticker' column in processed CSV. Columns: {list(df.columns)}")


def _slice_market_rows(df: pd.DataFrame, candidates=MARKET_TICKER_CANDIDATES) -> pd.DataFrame:
    """Return a copy of rows matching any of the candidate market tickers."""
    tcol = _find_ticker_column(df)
    tickers = df[tcol].astype(str).str.upper()
    for sym in candidates:
        sub = df[tickers == sym.upper()]
        if not sub.empty:
            return sub.copy()
    return pd.DataFrame()  # none matched


def _attach_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a DatetimeIndex using common date columns or parse the current index."""
    # 1) try common date-like columns
    for c in DATE_CANDIDATES:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=False)
            mask = s.notna()
            if mask.any():
                out = df.loc[mask].copy()
                out.index = s[mask].dt.tz_localize(None)
                out = out.sort_index()
                out = out[~out.index.duplicated(keep="last")]
                return out
    # 2) try parsing existing index
    maybe = pd.to_datetime(df.index, errors="coerce", utc=False)
    if maybe.notna().any():
        out = df.copy()
        out.index = maybe.dt.tz_localize(None)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out
    raise ValueError("Could not establish a DatetimeIndex from any known columns or index.")


# --------------------------- Features --------------------------- #

def make_regime_features(prices: pd.Series) -> pd.DataFrame:
    """
    Feature set for HMM:
      - return:        daily log return
      - volatility:    20-day rolling std of returns
      - momentum:      20-day log return
    Returns unscaled features indexed by date.
    """
    prices = prices.astype(float)
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices.loc[~prices.index.duplicated(keep="last")]  # safety

    # log returns (no fill_method, no deprecations)
    r = np.log(prices).diff().dropna()

    vol = r.rolling(RET_WIN_VOL, min_periods=RET_WIN_VOL).std()
    mom = (np.log(prices) - np.log(prices.shift(MOM_WIN))).reindex(r.index)

    X = pd.concat([r, vol, mom], axis=1)
    X.columns = ["return", "volatility", "momentum"]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return X


# --------------------------- Model bundle --------------------------- #

@dataclass
class RegimeModel:
    """Bundle HMM + scaler + feature schema for consistent inference."""
    hmm: GaussianHMM
    scaler: StandardScaler
    feature_names: list

    def save(self, path: Path) -> None:
        joblib.dump(
            {"hmm": self.hmm, "scaler": self.scaler, "feature_names": self.feature_names},
            path
        )

    @staticmethod
    def load(path: Path) -> "RegimeModel":
        blob = joblib.load(path)
        return RegimeModel(
            hmm=blob["hmm"],
            scaler=blob["scaler"],
            feature_names=blob.get("feature_names", ["return", "volatility", "momentum"]),
        )


# --------------------------- Training --------------------------- #

def _relabel_states_by_mean_return(hmm: GaussianHMM, X_scaled: pd.DataFrame) -> GaussianHMM:
    """
    Make state labels deterministic: sort states by mean of 'return' (ascending),
    so 0=Bear, 1=Neutral, 2=Bull.
    """
    # decode states on training set
    states = hmm.predict(X_scaled.values)
    df = X_scaled.copy()
    df["__state"] = states

    means = df.groupby("__state")["return"].mean().sort_values()
    mapping = {old: new for new, old in enumerate(means.index.tolist())}

    # remap transition/emission params to new order
    def order_states():
        return [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    order = order_states()

    hmm.startprob_ = hmm.startprob_[order]
    hmm.transmat_ = hmm.transmat_[np.ix_(order, order)]
    hmm.means_ = hmm.means_[order]
    # full covariances: reorder along state axis
    cov = hmm.covars_
    hmm.covars_ = cov[order]
    return hmm


def fit_hmm(X: pd.DataFrame, n_states: int = HMM_STATES, random_state: int = RANDOM_STATE) -> RegimeModel:
    """
    Fit HMM on features X (unscaled), return a RegimeModel (HMM + scaler).
    """
    if len(X) < MIN_ROWS:
        raise ValueError(f"Not enough rows to fit HMM: {len(X)} < {MIN_ROWS}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X.values),
        index=X.index,
        columns=X.columns,
    )

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=random_state,
        verbose=False,
    )
    hmm.fit(X_scaled.values)

    # Deterministic state labeling
    hmm = _relabel_states_by_mean_return(hmm, X_scaled)

    return RegimeModel(hmm=hmm, scaler=scaler, feature_names=list(X.columns))


# --------------------------- Inference --------------------------- #

def predict_regime(model_or_hmm, X: pd.DataFrame) -> pd.Series:
    """
    Predict hidden regimes for feature frame X.

    Accepts either:
      - RegimeModel (preferred; contains scaler), or
      - raw GaussianHMM (assumes X already scaled exactly like training).
    """
    if isinstance(model_or_hmm, RegimeModel):
        scaler = model_or_hmm.scaler
        hmm = model_or_hmm.hmm
        # Align columns in the same order as training
        X = X[model_or_hmm.feature_names]
        Xs = pd.DataFrame(
            scaler.transform(X.values),
            index=X.index,
            columns=X.columns,
        )
        states = hmm.predict(Xs.values)
    else:
        # Backward compatibility if you previously saved only the HMM
        states = model_or_hmm.predict(X.values)

    return pd.Series(states, index=X.index, name="regime")


# --------------------------- Train & Save --------------------------- #

def train_and_save_hmm(config_path: str | Path = "config.yaml",
                       ticker: str = "GSPC",
                       out_name: str = "hmm_regime_model.pkl") -> Path:
    """
    Train on processed features_and_targets.csv for a market ticker (e.g., GSPC)
    and save a RegimeModel under ml_models/saved_models/out_name.
    Robust to missing processed rows; falls back to raw GSPC CSV via utils loader.
    """
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    out_dir = Path(config["ml_models"]["saved_models"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = processed_path / "features_and_targets.csv"

    prices = None
    if data_file.exists():
        df = pd.read_csv(data_file)
        try:
            close_col = _pick_close_column(df)
            mkt = _slice_market_rows(df, candidates=(ticker, "^GSPC", "SPY"))
            if not mkt.empty:
                mkt = _attach_datetime_index(mkt)
                mkt[close_col] = pd.to_numeric(mkt[close_col], errors="coerce")
                prices = mkt[close_col].dropna()
        except Exception as e:
            print(f"[WARN][train] Processed CSV path had issues: {e}")

    if prices is None or len(prices) == 0:
        # fallback to raw
        try:
            from src.utils import _load_latest_gspc_df
            gspc_df = _load_latest_gspc_df(config_path)
            prices = gspc_df["Close"].dropna()
            print("[train] Using raw GSPC fallback.")
        except Exception as e:
            raise RuntimeError(f"[train] Could not obtain prices from processed or raw sources: {e}")

    X = make_regime_features(prices)
    if len(X) < MIN_ROWS:
        raise ValueError(f"[train] Not enough rows to fit HMM: {len(X)} < {MIN_ROWS}. "
                         f"Add more history or reduce MIN_ROWS in hmm_regime.py")

    model = fit_hmm(X, n_states=HMM_STATES, random_state=RANDOM_STATE)
    out_path = out_dir / out_name
    model.save(out_path)
    print(f"[HMM] Saved model -> {out_path}")
    return out_path


# --------------------------- CLI / Test --------------------------- #

if __name__ == "__main__":
    # Simple test run on processed CSV for GSPC (with robust fallbacks)
    print("--- Testing Regime Detection Module ---")
    cfg = "config.yaml"

    config = yaml.safe_load(open(cfg))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_and_targets.csv"

    prices = None
    used_source = None

    if not data_file.exists():
        print(f"[WARN] {data_file} not found. Will try raw GSPC fallback.")
    else:
        df = pd.read_csv(data_file)
        try:
            close_col = _pick_close_column(df)
        except KeyError as e:
            print(f"[WARN] {e} Falling back to raw GSPC.")
            df = None

        if df is not None:
            # Try to slice a known market proxy from processed CSV
            mkt = _slice_market_rows(df)
            if mkt.empty:
                # Print a quick inventory of available tickers to help debugging
                if "ticker" in df.columns:
                    sample = df["ticker"].dropna().astype(str).str.upper().value_counts().head(10)
                    print(f"[WARN] No market rows found for {MARKET_TICKER_CANDIDATES}. "
                          f"Top tickers in processed CSV: {dict(sample)}")
                else:
                    print("[WARN] 'ticker' column missing in processed CSV.")
            else:
                try:
                    mkt = _attach_datetime_index(mkt)
                    mkt[close_col] = pd.to_numeric(mkt[close_col], errors="coerce")
                    prices = mkt[close_col].dropna()
                    used_source = f"processed CSV ({data_file.name})"
                except Exception as e:
                    print(f"[WARN] Failed to build datetime index from processed CSV: {e}")

    # Fallback: load directly from the raw GSPC file (robust loader from utils)
    if prices is None or len(prices) == 0:
        try:
            from src.utils import _load_latest_gspc_df  # robust raw loader
            gspc_df = _load_latest_gspc_df(cfg)
            prices = gspc_df["Close"].dropna()
            used_source = "raw GSPC CSV (utils loader)"
        except Exception as e:
            print(f"[ERROR] Raw GSPC fallback failed: {e}")

    if prices is None or len(prices) == 0:
        print("[ERROR] Could not obtain any market prices (processed or raw). "
              "Ensure your pipeline produced GSPC/^GSPC/SPY rows or raw GSPC CSV exists.")
        raise SystemExit(1)

    print(f"Loaded {len(prices)} price rows from {used_source}: "
          f"{prices.index.min().date()} → {prices.index.max().date()}")

    # Build features
    X = make_regime_features(prices)

    if len(X) < MIN_ROWS:
        print(f"[ERROR] Not enough rows to fit HMM: {len(X)} < {MIN_ROWS}. "
              "Add more history or reduce MIN_ROWS.")
        raise SystemExit(1)

    # Fit + relabel + report
    model = fit_hmm(X, n_states=HMM_STATES, random_state=RANDOM_STATE)
    regimes = predict_regime(model, X)

    feat_with_regimes = X.join(regimes)

    print("\n--- Regime Characteristics (Bear=0, Neutral=1, Bull=2) ---")
    for i in range(model.hmm.n_components):
        rdata = feat_with_regimes[feat_with_regimes["regime"] == i]
        share = len(rdata) / len(X) if len(X) else 0.0
        print(f"\nRegime {i} ({len(rdata)} days, {share:.1%} of total):")
        print("  - Mean Return:     ", rdata["return"].mean())
        print("  - Mean Volatility: ", rdata["volatility"].mean())
        print("  - Mean Momentum:   ", rdata["momentum"].mean())

    print("\nLatest detected regime (last 5 days):")
    print(regimes.tail())

    # Save the model for get_current_regime()
    out_path = train_and_save_hmm(cfg, ticker="GSPC")
