import schedule
import time
import subprocess
import datetime

def morning_routine():
    print(f"\n--- ⏰ Cortexa Autonomous Update: {datetime.datetime.now()} ---")
    
    # 1. Fetch Data & Update Memory (RAG)
    print("1. Running Daily Data Flow...")
    subprocess.run(["python", "-m", "flows.daily_update_flow"])
    
    # 2. Retrain the Brain (ML) - Optional but recommended
    # This teaches the model about new patterns that happened yesterday
    print("2. Retraining Model on new data...")
    subprocess.run(["python", "-m", "src.training.train"])
    
    print("--- ✅ Morning Routine Complete. Sleeping until tomorrow. ---")

# Schedule it to run every day at 6:00 AM (before market open)
# You can also use .every(1).hours() for testing
schedule.every().day.at("06:00").do(morning_routine)

print("--- 🤖 Cortexa Autonomy Engine Started ---")
print("Waiting for next scheduled run...")

# Run immediately once on startup to ensure data is fresh
morning_routine()

while True:
    schedule.run_pending()
    time.sleep(60) # Check the clock every minute