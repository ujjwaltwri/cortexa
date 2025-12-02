import pandas as pd
from pathlib import Path
import yaml
import glob
import os

def check_tags():
    print("--- 🕵️‍♂️ Checking Latest News CSV ---")
    cfg = yaml.safe_load(open("config.yaml"))
    raw_path = Path(cfg["data_paths"]["raw"])
    
    # Find the latest news CSV file
    files = list(raw_path.glob("raw_news_*.csv"))
    if not files:
        print("Error: No raw news files found.")
        return
        
    latest_file = max(files, key=os.path.getctime)
    print(f"Loaded: {latest_file.name}")
    
    # Load and check the data integrity
    df = pd.read_csv(latest_file)
    
    # Check GOOGL tag count (case-insensitive to be safe)
    if 'ticker' in df.columns:
        googl_rows = df[df['ticker'].astype(str).str.upper() == 'GOOGL']
        tsla_rows = df[df['ticker'].astype(str).str.upper() == 'TSLA']
        
        print(f"Total Rows Fetched: {len(df)}")
        print(f"GOOGL Ticker Rows: {len(googl_rows)}")
        print(f"TSLA Ticker Rows: {len(tsla_rows)}")
        
        if not googl_rows.empty:
            print(f"✅ First GOOGL Headline: {googl_rows['title'].iloc[0][:60]}...")
        else:
            print("❌ GOOGL Ticker Tag Missing in CSV.")
    else:
        print("❌ 'ticker' column missing from CSV. Ingestion error.")

check_tags()