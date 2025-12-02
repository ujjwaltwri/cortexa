# scripts/backtest/naive_backtest.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main(preds_full_csv="data/03_predictions/wf_preds_full.csv"):
    df = pd.read_csv(preds_full_csv, parse_dates=[0], index_col=0)
    # Expect columns: pred_up, future_ret (or future_price to compute)
    if "future_ret" not in df.columns:
        raise FileNotFoundError("future_ret column missing in preds file.")
    if "pred_up" not in df.columns:
        raise FileNotFoundError("pred_up column missing in preds file.")

    # Per-row single-day PnL: go long when pred_up==1
    df["pnl_1d"] = df["pred_up"].astype(int) * df["future_ret"].astype(float)

    # Aggregate per day (average across tickers)
    daily = df.groupby(df.index.normalize())["pnl_1d"].mean().rename("avg_pnl")
    equity = (1.0 + daily.fillna(0.0)).cumprod()

    print("\n--- Naive Backtest (Topline) ---")
    print(f"Days: {len(daily)}")
    print(f"CAGR-ish (naive): {(equity.iloc[-1] ** (252/len(daily)) - 1.0):.2%}" if len(daily) > 0 else "N/A")
    print(f"Total return     : {equity.iloc[-1]-1.0:.2%}" if len(equity) else "N/A")
    print("\nLast 10 days (avg pnl):")
    print(daily.tail(10).to_string())

    # Save outputs
    out_dir = Path("data/04_backtests"); out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "naive_daily_avg_pnl.csv")
    equity.rename("equity").to_csv(out_dir / "naive_equity_curve.csv")
    print(f"\nSaved: {out_dir}/naive_daily_avg_pnl.csv, {out_dir}/naive_equity_curve.csv")


if __name__ == "__main__":
    main()
