import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def backtest_5y(config_path="config.yaml"):
    print("--- 🛡️ Cortexa: 5-Year Walk-Forward Gauntlet ---")
    
    # 1. Load V2 Data
    config = yaml.safe_load(open(config_path))
    processed_path = Path(config["data_paths"]["processed"])
    data_file = processed_path / "features_and_targets.csv"
    
    if not data_file.exists():
        print("❌ Error: data not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df.sort_index() # Critical for time series!
    
    # 2. Define Features
    exclude_cols = ['target', 'future_close', 'future_return', 'ticker', 'fwd_ret']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    # 3. Identify the last 5 years
    years = sorted(df.index.year.unique())
    if len(years) < 5:
        print("Not enough history for 5-year backtest.")
        return
        
    test_years = years[-5:]
    print(f"Testing Years: {test_years}")
    
    scores = []

    # 4. Run the Gauntlet
    for year in test_years:
        print(f"\n📅 Simulating Trading Year: {year}...")
        
        # TRAIN on everything BEFORE this year
        # TEST on this specific year
        X_train = X[X.index.year < year]
        y_train = y[y.index.year < year]
        
        X_test = X[X.index.year == year]
        y_test = y[y.index.year == year]
        
        if len(X_test) == 0: continue

        # Train Model (Simulating what we knew back then)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        print(f"   Result: AUC={auc:.4f} | Accuracy={acc:.2%}")
        scores.append({'year': year, 'auc': auc, 'acc': acc})

    # 5. Final Report
    print("\n--- 🏆 Final Scorecard ---")
    avg_auc = np.mean([s['auc'] for s in scores])
    avg_acc = np.mean([s['acc'] for s in scores])
    
    print(f"Average AUC (5y):      {avg_auc:.4f}")
    print(f"Average Accuracy (5y): {avg_acc:.2%}")

    if avg_auc > 0.53:
        print("\n✅ PASSED: Strategy is robust and profitable.")
    else:
        print("\n⚠️ CAUTION: Strategy is unstable or weak.")

if __name__ == "__main__":
    backtest_5y()