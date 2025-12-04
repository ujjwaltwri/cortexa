import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def train_v5_snipers(config_path="config.yaml"):
    print("--- 🧠 Cortexa V5: Training Regime Snipers ---")
    
    config = yaml.safe_load(open(config_path))
    data_file = Path(config["data_paths"]["processed"]) / "features_v5.csv"
    
    if not data_file.exists():
        print("Error: features_v5.csv not found.")
        return

    df = pd.read_csv(data_file, index_col=0, parse_dates=True).sort_index()
    regimes = ['1_0', '1_1', '0_0', '0_1']
    
    save_dir = Path(config["ml_models"]["saved_models"]) / "regime_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    feature_cols = [c for c in df.columns if c not in ['target', 'regime_tag', 'ticker']]
    print(f"Features: {feature_cols}")
    
    results = {}

    for regime in regimes:
        print(f"\n⚔️  Training Sniper for Regime: {regime}")
        
        regime_data = df[df['regime_tag'] == regime]
        if len(regime_data) < 200: continue
            
        X = regime_data[feature_cols]
        y = regime_data['target']
        
        # 80/20 Split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # The Proven V3 Configuration
        clf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_leaf=20,
            random_state=42, 
            n_jobs=-1,
            class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        print(f"   Accuracy: {acc:.2%}")
        print(f"   ROC-AUC:  {auc:.4f}")
        
        results[regime] = {"accuracy": acc, "auc": auc}
        
        joblib.dump(clf, save_dir / f"model_{regime}.pkl")
        with open(save_dir / f"model_{regime}_meta.json", "w") as f:
            json.dump({"features": feature_cols, "regime": regime}, f)

    print("\n--- 🏆 V5 Regime Scorecard ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    train_v5_snipers()