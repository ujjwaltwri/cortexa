from fredapi import Fred
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path

def fetch_economic_data(config_path="config.yaml"):
    """
    Fetches all economic data series from FRED listed in the config.
    """
    print("--- Starting FRED Economic Data Fetch ---")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The configuration file was not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading YAML configuration: {e}")
        return

    fred_config = config.get('fred', {})
    api_key = fred_config.get('api_key', 'YOUR_FREE_FRED_API_KEY_HERE')
    series_ids = fred_config.get('series_ids', [])
    raw_data_path = Path(config['data_paths']['raw'])
    
    # 1. Check for API Key
    if api_key == 'YOUR_FREE_FRED_API_KEY_HERE' or not api_key:
        print("Error: FRED API key not set in config.yaml.")
        print("Please get your free key from https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Then paste it into your 'config.yaml' file.")
        print("Skipping FRED data fetch.")
        return
        
    # 2. Check for Series IDs
    if not series_ids:
        print("No FRED series IDs found in config.yaml under 'fred: series_ids:'. Skipping.")
        return

    # 3. Proceed with fetching data
    try:
        fred = Fred(api_key=api_key)
        all_series_df = []
        
        print(f"Fetching {len(series_ids)} series from FRED...")
        for series_id in series_ids:
            print(f"  - Fetching series: {series_id}")
            try:
                data = fred.get_series(series_id)
                data.name = series_id # Rename the pandas Series
                all_series_df.append(data)
            except Exception as e:
                print(f"    Error fetching {series_id}: {e}")
        
        if all_series_df:
            # Combine all series into a single DataFrame
            combined_df = pd.concat(all_series_df, axis=1)
            
            # Forward-fill and then back-fill missing values
            combined_df = combined_df.ffill().bfill() 
            
            filename = f"raw_econ_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_file = raw_data_path / filename
            
            combined_df.to_csv(output_file)
            print(f"\nSuccessfully fetched and combined {len(all_series_df)} series.")
            print(f"Saved raw economic data to: {output_file}")
            print(f"Latest data points:\n{combined_df.tail(3)}")
            
    except Exception as e:
        print(f"Error connecting to FRED with API key: {e}")

    print("--- FRED Economic Data Fetch Complete ---")

if __name__ == "__main__":
    # This allows you to run this script directly from the root 'cortexa' folder
    # Usage: (venv) ... % python src/data_ingestion/fred_client.py
    fetch_economic_data(config_path="config.yaml")