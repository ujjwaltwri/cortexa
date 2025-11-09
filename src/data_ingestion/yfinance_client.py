import yfinance as yf
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path

def fetch_market_data(config_path="config.yaml", period="5y"):
    """
    Fetches historical market data for all tickers in the config.
    """
    print("--- Starting yfinance Market Data Fetch ---")
    
    # Load configuration
    config = yaml.safe_load(open(config_path))
    tickers = config.get('tickers', [])
    raw_data_path = Path(config['data_paths']['raw'])
    
    if not tickers:
        print("No tickers found in config.yaml. Skipping.")
        return
        
    print(f"Fetching data for tickers: {tickers}")
    
    try:
        data = yf.download(tickers, period=period, group_by='ticker')
        
        # Save each ticker's data to its own file
        for ticker in tickers:
            ticker_data = data[ticker] if len(tickers) > 1 else data
            if ticker_data.empty:
                print(f"  - No data found for {ticker}")
                continue

            # Clean up column names for single-ticker downloads
            if '^' in ticker: # yfinance has trouble with index tickers
                ticker_data = yf.download(ticker, period=period)
                
            filename = f"raw_market_{ticker.replace('^', '')}_{datetime.now().strftime('%Y%m%d')}.csv"
            output_file = raw_data_path / filename
            
            ticker_data.to_csv(output_file)
            print(f"  - Saved data for {ticker} to {output_file}")

        print("\nSuccessfully fetched all market data.")
        
    except Exception as e:
        print(f"Error fetching yfinance data: {e}")

    print("--- yfinance Market Data Fetch Complete ---")

if __name__ == "__main__":
    fetch_market_data(config_path="config.yaml")