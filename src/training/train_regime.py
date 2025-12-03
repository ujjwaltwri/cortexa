import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_by_regime(config_path="config.yaml"):
    print("--- 🧠 Cortexa: Training Regime-Specific Snipers (With Metadata) ---")
    
    config = yaml.safe_load(open(config_path))
    # CRITICAL: Point to V3 data
    data_file = Path(config["data_paths"]["processed"]) / "features_v3.csv"
    
    if not data_file.exists():
        print("Error: features_v3.csv not found. Run feature_engine_v3 first.")
        return

    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    regimes = df['regime_tag'].unique()
    
    save_dir = Path(config["ml_models"]["saved_models"]) / "regime_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Exclude non-features
    non_feature_cols = ['target', 'regime_tag', 'ticker', 'future_close', 'fwd_ret']
    all_feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    results = {}

    for regime in regimes:
        print(f"\n⚔️  Training for Regime: {regime}")
        
        regime_data = df[df['regime_tag'] == regime]
        if len(regime_data) < 100: 
            print("   Skipping (Not enough data)")
            continue
            
        X = regime_data[all_feature_cols]
        y = regime_data['target']
        
        # 80/20 Split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        clf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_leaf=10,
            random_state=42, 
            n_jobs=-1,
            class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        
        # Save Model
        joblib.dump(clf, save_dir / f"model_{regime}.pkl")
        
        # --- FIX 1: SAVE FEATURE METADATA PER REGIME ---
        # This tells the inference engine EXACTLY what columns this model needs
        meta = {
            "regime": str(regime),
            "features": list(X.columns), # The exact list used for training
            "count": len(X)
        }
        with open(save_dir / f"model_{regime}_meta.json", "w") as f:
            json.dump(meta, f)
        # -----------------------------------------------
        
        # Evaluate
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"   Accuracy: {acc:.2%}")
        results[regime] = acc

    print("\n✅ Regime Training Complete. Metadata Saved.")

if __name__ == "__main__":
    train_by_regime()