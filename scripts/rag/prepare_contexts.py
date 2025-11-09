# scripts/rag/prepare_contexts.py
from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd

# Columns we should NOT turn into text tokens
DROP_COLS = {
    "ticker", "target", "future_price", "future_ret",
    "Open","High","Low","Close","Volume",
    "open","high","low","close","volume","Adj Close",
    "ticker_id"
}

KNOWN_DT_COLS = ("Date", "Datetime", "date", "datetime", "timestamp", "time")

def _read_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _load_processed_csv(processed_dir: Path) -> pd.DataFrame:
    fp = processed_dir / "features_and_targets.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Run feature_engine first.")
    df = pd.read_csv(fp)

    # Try to set datetime index
    for col in KNOWN_DT_COLS:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=False)
            if s.notna().mean() >= 0.8:
                df[col] = s
                df = df[s.notna()].set_index(col)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        first = df.columns[0]
        s = pd.to_datetime(df[first], errors="coerce", utc=False)
        if s.notna().mean() >= 0.8:
            df[first] = s
            df = df[s.notna()].set_index(first)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Could not establish a DatetimeIndex for processed CSV.")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.columns = [c.strip() for c in df.columns]
    return df

def _try_join_regime(df: pd.DataFrame, artifacts_dir: Path) -> pd.Series:
    """
    Try to attach a per-date 'regime' series.
    Order of attempts:
      1) use df['regime'] if present
      2) join artifacts/daily_regime.csv by date (normalize to date)
      3) else return a default of -1
    """
    if "regime" in df.columns:
        return df["regime"].astype("int16")

    daily_path = artifacts_dir / "daily_regime.csv"
    if daily_path.exists():
        reg = pd.read_csv(daily_path, parse_dates=[0], index_col=0).squeeze("columns")
        reg.index = reg.index.normalize()
        out = reg.reindex(df.index.normalize()).ffill().bfill()
        if out.notna().any():
            return out.astype("int16")

    return pd.Series(index=df.index, data=-1, dtype="int16", name="regime")

def _build_text_row(row: pd.Series, max_fields: int = 40) -> str:
    """
    Turn numeric feature row into a compact 'k:v' space-separated string,
    taking up to max_fields numeric columns (excluding DROP_COLS).
    """
    parts = []
    count = 0
    for k, v in row.items():
        if k in DROP_COLS:
            continue
        if isinstance(v, (int, float, np.number)):
            # guard finite
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue
            parts.append(f"{k}:{fv:.4f}")
            count += 1
            if count >= max_fields:
                break
    return " ".join(parts)

def run(config_path: str, out_csv: str, max_fields: int = 40, limit_rows: int | None = None):
    cfg = _read_cfg(config_path)
    processed_dir = Path(cfg["data_paths"]["processed"])
    artifacts_dir = Path(cfg.get("data_paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = _load_processed_csv(processed_dir)

    # limit to recent rows if requested (for quick dev)
    if limit_rows is not None and limit_rows > 0:
        df = df.tail(limit_rows)

    # attach helper columns
    regime = _try_join_regime(df, artifacts_dir)
    ticker = df["ticker"] if "ticker" in df.columns else pd.Series(index=df.index, data="", dtype="object")
    target = df["target"] if "target" in df.columns else pd.Series(index=df.index, data=np.nan, dtype="float32")

    # Build 'text' column from numeric features
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # Keep the numeric feature view aligned to df
    numeric_df = numeric_df.reindex(df.index)

    texts = []
    for idx in df.index:
        texts.append(_build_text_row(numeric_df.loc[idx], max_fields=max_fields))

    out = pd.DataFrame({
        "date": df.index.astype("datetime64[ns]"),
        "ticker": ticker.values,
        "regime": regime.values,
        "target": target.values,
        "text": texts,
        "event_type": "market_state",
    })
    # Drop empty text rows
    out = out[out["text"].str.len() > 0].copy()
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote RAG contexts: {out_path} (rows={len(out)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out", type=str, default="artifacts/rag_contexts.csv")
    ap.add_argument("--max-fields", type=int, default=40, help="max numeric features per context row")
    ap.add_argument("--limit-rows", type=int, default=None, help="tail N rows (debug speedup)")
    args = ap.parse_args()
    run(args.config, args.out, max_fields=args.max_fields, limit_rows=args.limit_rows)
