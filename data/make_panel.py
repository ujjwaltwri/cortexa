import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "SPY", "QQQ", "^GSPC"]

def make_panel():
    print("📡 Downloading multi-ticker price panel...")
    
    dfs = []
    for t in TICKERS:
        print(f" - {t}")
        df = yf.download(t, start="2000-01-01", progress=False)
        df["ticker"] = t
        dfs.append(df)

    panel = pd.concat(dfs)
    panel.index.name = "date"

    panel = panel.reset_index()[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]]
    panel.columns = ["date", "ticker", "open", "high", "low", "close", "volume"]

    # save
    out_path = Path("data/01_raw/panel_prices.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    print(f"\n✅ Saved: {out_path} ({len(panel):,} rows)")

if __name__ == "__main__":
    make_panel()
