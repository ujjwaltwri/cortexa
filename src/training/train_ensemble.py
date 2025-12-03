import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

def get_sharpe_ratio(returns):
    if len(returns) == 0 or np.std(returns) == 0: return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

def train_ensemble(config_path="config.yaml"):
    print("--- 🧠 Cortexa: Training the Ensemble Brain (The Gauntlet) ---")
    
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_and_targets.csv"
    
    if not data_file.exists(): return

    print("Loading smart dataset...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True).sort_index()
    
    exclude_cols = ['target', 'future_close', 'future_return', 'ticker']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    returns = df['future_return']

    # Start testing from recent history (last 10 years) to see if it works NOW
    years = sorted(df.index.year.unique())
    start_test_year_index = len(years) - 10 
    
    print(f"\nStarting Ensemble Validation on the last 10 years...")

    metrics_log = []

    for i in range(start_test_year_index, len(years)):
        test_year = years[i]
        
        # Sliding Window: Train on last 5 years only (Fresh Data), Test on current year
        # This helps adapts to modern markets
        train_start_year = test_year - 5
        
        X_train = X[(X.index.year >= train_start_year) & (X.index.year < test_year)]
        y_train = y[(y.index.year >= train_start_year) & (y.index.year < test_year)]
        
        X_test = X[X.index.year == test_year]
        y_test = y[y.index.year == test_year]
        r_test = returns[returns.index.year == test_year]
        
        if len(X_test) < 50: continue

        # --- MODEL 1: Random Forest (Stability) ---
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, n_jobs=-1, random_state=42)
        
        # --- MODEL 2: LightGBM (Precision) ---
        lgbm_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, n_jobs=-1, random_state=42, verbose=-1)
        
        # --- ENSEMBLE: Voting ---
        model = VotingClassifier(estimators=[('rf', rf), ('lgbm', lgbm_clf)], voting='soft')
        model.fit(X_train, y_train)
        
        # Evaluate
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.51).astype(int) # Slightly aggressive threshold
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        sharpe = get_sharpe_ratio(r_test * preds)
        
        print(f"  📅 Year {test_year}: AUC={auc:.4f} | Acc={acc:.2%} | Sharpe={sharpe:.2f}")
        metrics_log.append({"year": test_year, "auc": auc, "acc": acc, "sharpe": sharpe})

    # Summary
    avg_auc = np.mean([m['auc'] for m in metrics_log])
    avg_sharpe = np.mean([m['sharpe'] for m in metrics_log])
    print(f"\n--- 🏁 Ensemble Summary (Last 10 Years) ---")
    print(f"Avg AUC:    {avg_auc:.4f}")
    print(f"Avg Sharpe: {avg_sharpe:.2f}")

    # Save Production Model (Trained on ALL recent data)
    print("\nTraining Final Production Ensemble...")
    # Train on last 5 years of total data for freshness
    final_start = years[-5]
    X_final = X[X.index.year >= final_start]
    y_final = y[y.index.year >= final_start]
    
    model.fit(X_final, y_final)
    
    save_dir = Path(config["ml_models"]["saved_models"])
    joblib.dump(model, save_dir / "rf_model.pkl") # Save as 'rf_model.pkl' for server compatibility
    
    metadata = {"features": feature_cols, "metrics": {"auc": avg_auc}, "threshold": 0.51}
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f)
        
    print(f"✅ Ensemble Brain saved to: {save_dir}/rf_model.pkl")

if __name__ == "__main__":
    train_ensemble()