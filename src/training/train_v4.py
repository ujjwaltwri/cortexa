import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

def train_v4_ensemble(config_path="config.yaml"):
    print("--- 🧠 Cortexa V4: Cross-Sectional Ensemble Training ---")
    
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_v4.csv"
    
    if not data_file.exists():
        print("❌ Error: features_v4.csv not found. Run feature_engine_v4.")
        return

    print("Loading V4 Factor Dataset...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True).sort_index()
    
    # Dynamic Feature Selection
    exclude_cols = ['target', 'ticker']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"Training on {len(X)} rows with {len(feature_cols)} features.")
    print(f"Features: {feature_cols}")

    # --- CROSS-SECTIONAL SPLIT ---
    # Instead of splitting by stock, we split by TIME.
    # We train on the past of ALL stocks, test on the future of ALL stocks.
    
    # Last 10% of data for validation
    split_idx = int(len(X) * 0.90)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Rows: {len(X_train)} | Testing Rows: {len(X_test)}")

    # --- MODEL 1: LightGBM (The Booster) ---
    print("Initializing LightGBM...")
    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=40,          # Increased complexity
        min_child_samples=100,  # Reduce noise
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # --- MODEL 2: Random Forest (The Anchor) ---
    print("Initializing Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    # --- ENSEMBLE ---
    print("\n⚔️  Training Ensemble (Voting)...")
    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm_clf), ('rf', rf_clf)],
        voting='soft',
        weights=[2, 1] # Give LightGBM slightly more weight
    )
    
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    probs = ensemble.predict_proba(X_test)[:, 1]
    preds = (probs > 0.51).astype(int) # 51% Confidence threshold
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print(f"\n--- 🏆 V4 Results ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"ROC-AUC:  {auc:.4f}")
    
    # Save to Production Path
    save_dir = Path(config["ml_models"]["saved_models"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as 'rf_model.pkl' so server.py picks it up automatically
    joblib.dump(ensemble, save_dir / "rf_model.pkl")
    
    metadata = {
        "features": feature_cols,
        "metrics": {"accuracy": acc, "roc_auc": auc},
        "threshold": 0.51,
        "version": "v4_ensemble"
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f)
        
    print(f"✅ Cortexa V4 Brain saved to: {save_dir}/rf_model.pkl")

if __name__ == "__main__":
    train_v4_ensemble()