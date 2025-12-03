import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_model(config_path="config.yaml"):
    print("--- 🧠 Cortexa: Training the V2 Brain ---")
    
    # 1. Load Config & Data
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_and_targets.csv"
    
    if not data_file.exists():
        print("❌ Error: processed data not found. Run feature_engine_v2 first.")
        return

    print("Loading V2 dataset...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # 2. DYNAMIC FEATURE SELECTION
    # CRITICAL: We exclude 'fwd_ret' because that is the answer key!
    exclude_cols = ['target', 'future_close', 'future_return', 'ticker', 'fwd_ret']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"Training on {len(X)} rows with {len(feature_cols)} features.")
    print(f"Features: {feature_cols}")

    # 3. Time-Based Split (80% Train, 20% Test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Data: {len(X_train)} rows")
    print(f"Testing Data:  {len(X_test)} rows")

    # 4. Train Random Forest (Robust Baseline)
    print("\nTraining Model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,         # Constrain depth to prevent memorizing noise
        min_samples_leaf=20,  # Require significant evidence per decision
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
    
    # 6. Save Artifacts
    save_dir = Path(config["ml_models"]["saved_models"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # We save as 'rf_model.pkl' (The current standard for the server)
    joblib.dump(model, save_dir / "rf_model.pkl")
    
    metadata = {
        "features": feature_cols,
        "metrics": {"accuracy": acc, "roc_auc": auc},
        "threshold": 0.5 
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f)
        
    print(f"✅ V2 Brain saved to: {save_dir}/rf_model.pkl")

if __name__ == "__main__":
    train_model()