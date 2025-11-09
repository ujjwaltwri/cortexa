# src/training/walkforward.py
# Walk-forward CV (time-aware) + small randomized search for LightGBM.
# Saves per-fold metrics and the best params, then refits on all pre-split data.

import json, math, random
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yaml, joblib
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score
)

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.copy_on_write = True


# ---------- Shared utils (mirrors train.py but self-contained) ----------

def _read_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _load_processed_csv(processed_path: Path) -> pd.DataFrame:
    fp = processed_path / "features_and_targets.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Processed data not found at {fp}. Run feature_engine first.")

    df = pd.read_csv(fp)

    # 1) Try the obvious headers
    for col in ("Date", "Datetime"):
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=False, infer_datetime_format=True)
            if s.notna().mean() > 0.8:
                df[col] = s
                df = df.set_index(col)
                break

    # 2) If still no DatetimeIndex, try first column (common when index was saved to CSV)
    if not isinstance(df.index, pd.DatetimeIndex):
        first = df.columns[0]
        s = pd.to_datetime(df[first], errors="coerce", utc=False, infer_datetime_format=True)
        if s.notna().mean() > 0.8:
            df[first] = s
            df = df.set_index(first)

    # 3) If still no DatetimeIndex, scan all columns and pick any datetime-like with >80% valid
    if not isinstance(df.index, pd.DatetimeIndex):
        best_col = None
        best_ratio = 0.0
        for c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
            ratio = s.notna().mean()
            if ratio > 0.8 and ratio > best_ratio:
                best_col, best_ratio = c, ratio
        if best_col is not None:
            df[best_col] = pd.to_datetime(df[best_col], errors="coerce", utc=False, infer_datetime_format=True)
            df = df.set_index(best_col)

    # 4) Final tidy-up
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")].sort_index()

    df.columns = [c.strip() for c in df.columns]
    return df

def _choose_features(df: pd.DataFrame):
    drop_cols = {"target","future_price","future_ret","ticker"}
    ohlcv = {"Open","High","Low","Close","Volume","open","high","low","close","volume","Adj Close"}
    if "ticker" in df.columns:
        df = df.copy()
        df["ticker_id"] = df["ticker"].astype("category").cat.codes.astype("int16")
    num = df.select_dtypes(include=[np.number]).copy()
    if "target" in df.columns and "target" not in num.columns:
        num["target"] = pd.to_numeric(df["target"], errors="coerce")
    feats = [c for c in num.columns if c not in drop_cols and c not in ohlcv]
    X = num[feats].copy()
    y = df["target"].astype("int8")
    return X, y, feats

def _prune_features(X: pd.DataFrame, feature_names: list[str]):
    nunq = X.nunique()
    keep = nunq[nunq > 3].index
    X = X[keep]; feature_names = [c for c in feature_names if c in keep]
    if X.shape[1] > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > 0.98).any()]
        if to_drop:
            X = X.drop(columns=to_drop)
            feature_names = [c for c in feature_names if c not in to_drop]
    return X, feature_names

def _optimal_threshold(y_true, y_proba, beta=0.7):
    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(y_proba, qs))
    best_f, best_t = -1.0, 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p,r,f,_ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0, beta=beta)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ---------- Walk-forward CV ----------

def walkforward_cv(config_path="config.yaml", n_folds=5, random_seeds=(42,)):
    print("\n--- Walk-Forward CV (LightGBM) ---")
    cfg = _read_config(config_path)
    processed = Path(cfg["data_paths"]["processed"])
    model_dir = Path(cfg["ml_models"]["saved_models"])
    model_dir.mkdir(parents=True, exist_ok=True)

    df = _load_processed_csv(processed)
    if "target" not in df.columns:
        raise ValueError("Missing 'target'.")

    X_all, y_all, features = _choose_features(df)
    mask = X_all.notna().all(axis=1) & y_all.notna()
    X_all, y_all = X_all[mask], y_all[mask]
    X_all, features = _prune_features(X_all, features)

    if not isinstance(X_all.index, pd.DatetimeIndex):
        raise ValueError("Walk-forward requires a datetime index.")

    dates = X_all.index.sort_values().unique()
    fold_size = math.floor(len(dates) / (n_folds + 1))  # leave one chunk for final test if needed
    folds = []
    for k in range(1, n_folds + 1):
        split = dates[fold_size * k]
        train_mask = X_all.index < split
        val_mask = (X_all.index >= split) & (X_all.index < dates[min(fold_size * (k + 1), len(dates)-1)])
        if val_mask.sum() < 30 or train_mask.sum() < 100:
            continue
        folds.append((split, train_mask, val_mask))

    if not folds:
        raise RuntimeError("Not enough data to form walk-forward folds.")

    # Small randomized search space (safe ranges)
    space = {
        "learning_rate": [0.015, 0.02, 0.03],
        "num_leaves": [127, 191, 255],
        "min_child_samples": [10, 20, 40],
        "reg_lambda": [0.0, 0.3, 0.8],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "max_bin": [511, 1023],
    }

    def sample_params():
        return {k: random.choice(v) for k, v in space.items()}

    results = []
    best_overall = None
    tried = set()

    for seed in random_seeds:
        random.seed(seed)
        for _ in range(10):  # 10 samples/seed → small, fast
            params = sample_params()
            key = tuple(sorted(params.items()))
            if key in tried:
                continue
            tried.add(key)

            fold_metrics = []
            for split, tr_mask, va_mask in folds:
                X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
                X_va, y_va = X_all[va_mask], y_all[va_mask]

                pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
                spw = (neg / max(1, pos)) if pos and neg else 1.0

                model = lgb.LGBMClassifier(
                    objective="binary",
                    random_state=seed,
                    n_estimators=2000,
                    learning_rate=params["learning_rate"],
                    num_leaves=params["num_leaves"],
                    max_depth=-1,
                    min_child_samples=params["min_child_samples"],
                    subsample=params["subsample"],
                    colsample_bytree=params["colsample_bytree"],
                    reg_lambda=params["reg_lambda"],
                    n_jobs=-1,
                    scale_pos_weight=spw,
                    max_bin=params["max_bin"],
                    min_data_in_bin=1,
                )
                categorical_feature = ["ticker_id"] if "ticker_id" in features else "auto"
                # Tail slice for early stopping
                n = len(X_tr); n_val_tail = max(1, int(n * 0.12))
                X_tr_in, y_tr_in = X_tr.iloc[:-n_val_tail], y_tr.iloc[:-n_val_tail]
                X_tr_val, y_tr_val = X_tr.iloc[-n_val_tail:], y_tr.iloc[-n_val_tail:]

                model.fit(
                    X_tr_in, y_tr_in,
                    eval_set=[(X_tr_val, y_tr_val)],
                    eval_metric=["auc", "binary_logloss", "average_precision"],
                    categorical_feature=categorical_feature,
                    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
                )

                proba = model.predict_proba(X_va)[:, 1]
                thr, _ = _optimal_threshold(y_va.values, proba, beta=0.7)
                pred = (proba >= thr).astype(int)

                auc = roc_auc_score(y_va, proba) if len(np.unique(y_va)) > 1 else np.nan
                ap = average_precision_score(y_va, proba) if len(np.unique(y_va)) > 1 else np.nan
                p, r, f1, _ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)

                fold_metrics.append({"auc": auc, "ap": ap, "p": p, "r": r, "f1": f1})

            # Aggregate across folds
            agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
            result = {"params": params, "metrics": agg}
            results.append(result)

            # Track best by AP first, then AUC
            score = (np.nan_to_num(agg.get("ap", 0.0), nan=0.0),
            np.nan_to_num(agg.get("auc", 0.0), nan=0.0))
            if best_overall is None or score > (
                np.nan_to_num(best_overall["metrics"].get("ap", 0.0), nan=0.0),
                np.nan_to_num(best_overall["metrics"].get("auc", 0.0), nan=0.0),
            ):

                best_overall = result
                print(f"[NEW BEST] AP={agg['ap']:.4f} AUC={agg['auc']:.4f} params={params}")

    # Save CV results
    out_dir = model_dir / "walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # Refit best-on-pre-2024 (or use cfg split_date if provided)
    split_date = cfg.get("ml_models", {}).get("split_date", "2024-01-01")
    train_mask = X_all.index < split_date
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]

    pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
    spw = (neg / max(1, pos)) if pos and neg else 1.0
    bp = best_overall["params"]

    final = lgb.LGBMClassifier(
        objective="binary",
        random_state=42,
        n_estimators=2500,
        learning_rate=bp["learning_rate"],
        num_leaves=bp["num_leaves"],
        max_depth=-1,
        min_child_samples=bp["min_child_samples"],
        subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"],
        reg_lambda=bp["reg_lambda"],
        n_jobs=-1,
        scale_pos_weight=spw,
        max_bin=bp["max_bin"],
        min_data_in_bin=1,
    )
    categorical_feature = ["ticker_id"] if "ticker_id" in features else "auto"
    # Tail slice for early stopping while training on all train data
    n = len(X_tr); n_val_tail = max(1, int(n * 0.12))
    final.fit(
        X_tr.iloc[:-n_val_tail], y_tr.iloc[:-n_val_tail],
        eval_set=[(X_tr.iloc[-n_val_tail:], y_tr.iloc[-n_val_tail:])],
        eval_metric=["auc", "binary_logloss", "average_precision"],
        categorical_feature=categorical_feature,
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=True)],
    )

    # Save model + metadata
    joblib.dump(final, out_dir / "lgbm_wf_model.pkl")
    with open(out_dir / "lgbm_wf_metadata.json", "w") as f:
        json.dump({
            "features": features,
            "best_params": best_overall["params"],
            "cv_metrics_mean": best_overall["metrics"],
            "split_date": split_date
        }, f, indent=2)

    print("\nSaved:")
    print(f"  CV results     : {out_dir/'cv_results.json'}")
    print(f"  Best WF model  : {out_dir/'lgbm_wf_model.pkl'}")
    print(f"  WF metadata    : {out_dir/'lgbm_wf_metadata.json'}")


if __name__ == "__main__":
    # Run: python -m src.training.walkforward
    walkforward_cv("config.yaml", n_folds=5, random_seeds=(42, 7))
