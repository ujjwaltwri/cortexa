import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_model(config_path="config.yaml"):
    print("--- 🧠 Cortexa: Training the Winning Brain (Random Forest) ---")
    
    # 1. Load Config & Data
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_and_targets.csv"
    
    if not data_file.exists():
        print("❌ Error: processed data not found.")
        return

    print("Loading smart dataset...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # 2. FEATURE SELECTION
    # The exact list that worked in the diagnostic
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume', 
        'FEDFUNDS', 'T10Y2Y', 'VIXCLS', 'UNRATE', 'PAYEMS', 
        'VIX_change', 'market_regime', 'prev_close', 'gap', 
        'close_loc', 'natr', 'rsi', 'roc_1', 'dist_sma20'
    ]
    
    # Check if features exist
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        print(f"Warning: Missing features. Using: {available_cols}")
    
    X = df[available_cols]
    y = df['target'] # Ensure this matches your feature_engine output
    
    print(f"Training on {len(X)} rows with {len(available_cols)} features.")

    # 3. Time-Based Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Data: {len(X_train)} rows")
    print(f"Testing Data:  {len(X_test)} rows")

    # 4. Train Random Forest
    print("\nTraining Model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    print("\n--- 📊 Results ---")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print(f"Accuracy: {acc:.2%}")
    print(f"ROC-AUC:  {auc:.4f}")
    
    # 6. Save Artifacts (Correctly Named)
    save_dir = Path(config["ml_models"]["saved_models"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # SAVE AS RF_MODEL.PKL (Fixing the confusion)
    joblib.dump(model, save_dir / "rf_model.pkl")
    
    metadata = {
        "features": available_cols,
        "metrics": {"accuracy": acc, "roc_auc": auc},
        "threshold": 0.5 
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f)
        
    print(f"✅ Random Forest saved to: {save_dir}/rf_model.pkl")

if __name__ == "__main__":
    train_model()