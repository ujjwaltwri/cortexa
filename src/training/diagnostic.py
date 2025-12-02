import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_diagnostic(config_path="config.yaml"):
    print("--- 🩺 Cortexa Diagnostic: The Overfit Check ---")
    
    # 1. Load Data
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_and_targets.csv"
    
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # 2. RE-CREATE TARGET (Sanity Check)
    # We ignore the existing 'target' and calculate a fresh one to be 100% sure.
    # Target: Did price go up > 0.5% in the next 1 day?
    THRESHOLD = 0.005 
    
    # Ensure we use 'close' or 'Close'
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    # Calculate future return (Shift -1 means look at tomorrow)
    df['diag_future_return'] = df[close_col].shift(-1) / df[close_col] - 1
    
    # Create Labels
    df['diag_target'] = 0
    df.loc[df['diag_future_return'] > THRESHOLD, 'diag_target'] = 1
    df.loc[df['diag_future_return'] < -THRESHOLD, 'diag_target'] = 0
    # Note: We ignore small moves (between -0.5% and +0.5%) to reduce noise
    
    # Drop rows where we can't see the future
    df = df.dropna(subset=['diag_future_return'])
    
    # 3. Select Features
    # Drop anything that looks like a target or a leak
    drop_cols = [
        'target', 'future_close', 'future_return', 'ticker', 
        'diag_target', 'diag_future_return', 'future_ret'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    print(f"\nFeatures selected ({len(feature_cols)}): {feature_cols}")
    
    # 4. Simple Time Split
    split_idx = int(len(df) * 0.8) # 80% Train, 20% Test
    
    X = df[feature_cols]
    y = df['diag_target']
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining Data: {len(X_train)} rows")
    print(f"Testing Data:  {len(X_test)} rows")
    
    # 5. The "Overfit" Model
    # We use a Random Forest with NO MAX DEPTH.
    # We WANT it to memorize the data. If it can't memorize, the data is garbage.
    print("\nTraining Unconstrained Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,     # Allow infinite complexity
        min_samples_split=2,
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # 6. The Verdict
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print("\n--- 🏁 DIAGNOSTIC RESULTS ---")
    print(f"Training Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:     {test_acc:.2%}")
    
    print("\n--- Interpretation ---")
    if train_acc < 0.60:
        print("❌ CRITICAL FAILURE: The model cannot even memorize the training data.")
        print("   This means your features have ZERO relationship to the target.")
        print("   Action: You need completely different features (Macro, Fundamental).")
    elif train_acc > 0.90 and test_acc < 0.55:
        print("⚠️ Overfitting: The model memorized the past but can't predict the future.")
        print("   This is 'normal' for finance. It means we have signal, but it's noisy.")
        print("   Action: Aggressive regularization, feature selection, or RAG filtering.")
    elif test_acc > 0.55:
        print("✅ SUCCESS: You have a predictive edge!")
    else:
        print("❓ Gray Area: Weak signal.")

if __name__ == "__main__":
    run_diagnostic()