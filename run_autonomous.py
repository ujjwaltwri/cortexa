import schedule
import time
import subprocess
import datetime
import os
from pathlib import Path

def touch_server_file():
    """
    Updates the timestamp of server.py. 
    This triggers Uvicorn (running with --reload) to restart and load the new model.
    """
    try:
        server_file = "server.py"
        if os.path.exists(server_file):
            # Update file access/modified times to NOW
            os.utime(server_file, None)
            print(f"🔄 Triggered Hot-Reload for {server_file}")
        else:
            print(f"⚠️ Warning: {server_file} not found. Cannot trigger reload.")
    except Exception as e:
        print(f"⚠️ Error triggering reload: {e}")

def morning_routine():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- 🌅 Cortexa 2.0 Autonomous Update: {now} ---")
    
    # 1. Data Ingestion & Feature Engineering (The "Body")
    # This fetches News, Prices, Economics and builds 'features_and_targets.csv'
    print("\n[1/3] Running Daily Data Pipeline (Ingest + RAG Indexing)...")
    try:
        subprocess.run(["python", "-m", "flows.daily_update_flow"], check=True)
        print("✅ Data Pipeline Complete.")
    except subprocess.CalledProcessError:
        print("❌ Data Pipeline Failed. Stopping routine.")
        return

    # 2. Model Retraining (The "Brain")
    # This retrains the Random Forest/Ensemble on the latest data
    print("\n[2/3] Retraining Alpha Model (Cortexa 2.0)...")
    try:
        subprocess.run(["python", "-m", "src.training.train"], check=True)
        print("✅ Model Retraining Complete.")
    except subprocess.CalledProcessError:
        print("❌ Training Failed. Old model retained.")
        return

    # 3. System Refresh
    print("\n[3/3] Refreshing Server...")
    touch_server_file()
    
    print(f"--- 🏁 Routine Finished. Sleeping until next cycle. ---")

# --- SCHEDULER CONFIGURATION ---
# Run every day at 6:00 AM (Before US Market Open)
schedule.every().day.at("06:00").do(morning_routine)

# OPTIONAL: Run every 4 hours to catch midday news?
# schedule.every(4).hours.do(morning_routine)

if __name__ == "__main__":
    print("--- 🤖 Cortexa 2.0 Autonomy Engine Started ---")
    print("    - Schedule: Daily @ 06:00 AM")
    print("    - Features: Auto-Train, Auto-RAG, Hot-Reload")
    
    # Run once immediately on startup to ensure everything is fresh
    morning_routine()

    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute