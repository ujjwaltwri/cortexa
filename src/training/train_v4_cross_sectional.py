# src/training/train_v4_cross_sectional.py

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as pd
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

HORIZON = 5
TARGET_THRESHOLD = 0.02


def train_v4_cross_sectional(config_path: str = "config.yaml"):
    print("--- 🧠 Cortexa v4: Cross-Sectional Training ---")

    # 1. Load config and v4 features
    cfg = yaml.safe_load(open(config_path))
    processed_root = Path(cfg["data_paths"]["processed"])
    data_file = processed_root / "features_v4.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"v4 features not found at: {data_file}")

    print(f"Loading v4 features from: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df.sort_index()

    if "ticker" not in df.columns:
        raise ValueError("v4 feature file must contain 'ticker' column.")

    # Treat ticker as categorical and capture categories for inference
    df["ticker"] = df["ticker"].astype("category")
    ticker_categories = df["ticker"].cat.categories.tolist()

    # 2. Define features and target
    exclude_cols = [f"fwd_ret_{HORIZON}d", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["target"]

    print(f"Total rows: {len(df)}, Features: {len(feature_cols)}")
    print(f"Tickers seen in training: {ticker_categories}")

    # 3. Walk-forward validation (year by year, cross-sectional)
    years = sorted(df.index.year.unique())
    start_test_year_idx = 3  # first 3 years = warmup for training

    if len(years) <= start_test_year_idx:
        raise RuntimeError("Not enough years of data for walk-forward validation.")

    metrics_log = []

    print(f"Starting walk-forward validation over {len(years) - start_test_year_idx} test years...\n")

    for i in range(start_test_year_idx, len(years)):
        test_year = years[i]

        train_mask = df.index.year < test_year
        test_mask = df.index.year == test_year

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_test) < 100:
            print(f"⚠️  Skipping year {test_year}: too few test samples ({len(X_test)})")
            continue

        model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=100,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5  # e.g., only one class present in y_test

        print(f"📅 Year {test_year}: AUC={auc:.4f} | Acc={acc:.2%}")
        metrics_log.append(
            {
                "year": int(test_year),
                "auc": float(auc),
                "accuracy": float(acc),
            }
        )

    if not metrics_log:
        raise RuntimeError("No valid test years found for walk-forward validation.")

    avg_auc = np.mean([m["auc"] for m in metrics_log])
    avg_acc = np.mean([m["accuracy"] for m in metrics_log])

    print("\n--- 🏁 v4 Cross-Sectional Scorecard ---")
    print(f"Average AUC:      {avg_auc:.4f}")
    print(f"Average Accuracy: {avg_acc:.2%}")

    # 5. Train final model on all data
    print("\nTraining final v4 cross-sectional model on ALL data...")

    final_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    final_model.fit(X, y)

    # 6. Save model + metadata
    save_root = Path(cfg["ml_models"]["saved_models"])
    save_root.mkdir(parents=True, exist_ok=True)

    model_path = save_root / "lgbm_v4_cross_sectional.pkl"
    joblib.dump(final_model, model_path)

    metadata = {
        "features": feature_cols,
        "categorical_features": ["ticker"],
        "ticker_categories": ticker_categories,
        "horizon": HORIZON,
        "target_threshold": TARGET_THRESHOLD,
        "metrics": {
            "avg_auc": float(avg_auc),
            "avg_accuracy": float(avg_acc),
        },
    }

    meta_path = save_root / "lgbm_v4_cross_sectional_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"\n✅ Saved v4 model to: {model_path}")
    print(f"✅ Saved metadata to: {meta_path}")


if __name__ == "__main__":
    train_v4_cross_sectional()
