# src/training/train.py
from __future__ import annotations

import json
from pathlib import Path
import warnings

import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    roc_auc_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.copy_on_write = True


# ========================== Config & IO ==========================

def _read_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_processed_csv(processed_path: Path) -> pd.DataFrame:
    fp = processed_path / "features_and_targets.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Processed data not found at {fp}. "
            "Run: python -m src.processing.feature_engine"
        )

    df = pd.read_csv(fp)

    # Robust datetime index
    dt_col = None
    for col in ("Date", "Datetime", "date", "datetime", "timestamp", "time"):
        if col in df.columns:
            dt_col = col
            break
    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
        df = df[df[dt_col].notna()].set_index(dt_col)
    else:
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce", utc=False)
        if parsed.notna().mean() > 0.8:
            df[first] = parsed
            df = df[parsed.notna()].set_index(first)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Could not establish a DatetimeIndex for processed CSV.")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.columns = [c.strip() for c in df.columns]
    return df


# ========================== Optional: Regime Feature ==========================

def _try_join_regime_feature(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """Join a daily 'regime' (0/1/2) column by date across all tickers."""
    try:
        from src.regime.hmm_regime import make_regime_features, predict_regime, RegimeModel
        from src.utils import _load_latest_gspc_df

        cfg = _read_config(config_path)
        model_path = Path(cfg["ml_models"]["saved_models"]) / "hmm_regime_model.pkl"

        gspc = _load_latest_gspc_df(config_path)
        X = make_regime_features(gspc["Close"])
        model = RegimeModel.load(model_path)
        regimes = predict_regime(model, X).rename("regime")

        reg_df = regimes.to_frame()
        reg_df.index = reg_df.index.normalize()

        out = df.copy()
        idx_norm = out.index.normalize()
        out["regime"] = reg_df.reindex(idx_norm).ffill()["regime"].values
        out["regime"] = out["regime"].astype("int8")
        return out
    except Exception as e:
        print(f"[WARN] Could not append regime feature: {e}")
        return df


# ========================== Feature selection ==========================

def _choose_features(df: pd.DataFrame):
    """Pick usable numeric features, excluding target/helper/raw OHLCV, and encode ticker."""
    drop_cols = {"target", "future_price", "future_ret", "ticker"}
    ohlcv_cols = {
        "Open", "High", "Low", "Close", "Volume",
        "open", "high", "low", "close", "volume", "Adj Close"
    }

    if "ticker" in df.columns:
        df = df.copy()
        df["ticker_id"] = df["ticker"].astype("category").cat.codes.astype("int16")

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if "target" in df.columns and "target" not in numeric_df.columns:
        numeric_df["target"] = pd.to_numeric(df["target"], errors="coerce")

    candidate_cols = [c for c in numeric_df.columns
                      if c not in drop_cols and c not in ohlcv_cols]

    X = numeric_df[candidate_cols].copy()
    y = df["target"].astype("int8")
    return X, y, candidate_cols


# ========================== Splitting ==========================

def _purged_walk_forward_splits(
    X: pd.DataFrame,
    n_folds: int = 5,
    horizon_days: int = 1,
    gap_days: int = 1,
):
    """
    Yield (train_mask, valid_mask) boolean arrays for purged walk-forward validation.
    Each fold trains on all data up to a point, skips (gap + horizon),
    and validates on the next slice of dates.
    """
    dates = np.array(sorted(X.index.normalize().unique()))
    n = len(dates)

    # Fallback: single purged split (last 20% for validation)
    if n_folds < 2 or n < (n_folds + horizon_days + gap_days + 5):
        cut = int(n * 0.8)
        if cut <= 0 or cut >= n:
            yield np.ones(len(X), dtype=bool), np.zeros(len(X), dtype=bool)
            return
        train_end = dates[cut - 1]
        purge_start = min(cut + gap_days + horizon_days, n - 1)
        val_start = dates[purge_start]

        train_mask = (X.index.normalize() <= train_end)
        valid_mask = (X.index.normalize() >= val_start)

        yield np.asarray(train_mask, dtype=bool), np.asarray(valid_mask, dtype=bool)
        return

    fold_size = n // n_folds
    for f in range(n_folds):
        train_end_idx = (f + 1) * fold_size - 1
        if train_end_idx < 0 or train_end_idx >= n - 1:
            continue

        train_end = dates[train_end_idx]
        purge_start = train_end_idx + 1 + gap_days + horizon_days
        if purge_start >= n:
            continue

        val_start = dates[purge_start]
        val_end_idx = min(purge_start + fold_size, n) - 1
        val_end = dates[val_end_idx]

        tr_mask = (X.index.normalize() <= train_end)
        va_mask = (X.index.normalize() >= val_start) & (X.index.normalize() <= val_end)

        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue

        yield np.asarray(tr_mask, dtype=bool), np.asarray(va_mask, dtype=bool)


def _optimal_threshold(y_true, y_proba, beta=1.0):
    """Choose a decision threshold maximizing F-beta on the validation set."""
    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(y_proba, qs))
    best_f, best_t = -1.0, 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0, beta=beta
        )
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ========================== Calibration Helper ==========================

class FrozenEstimator:
    """
    Proxy a fitted estimator for CalibratedClassifierCV(cv='prefit') without warnings.
    Forwards attributes (e.g., classes_) and methods; fit is a no-op.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    # sklearn's check_is_fitted(..., attributes=["classes_"]) expects this:
    @property
    def classes_(self):
        return getattr(self.estimator, "classes_", None)

    def fit(self, X, y=None):
        return self  # no-op (estimator is already fitted)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    # Make it transparent:
    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

    def __getattr__(self, name):
        # Forward everything else (including n_features_in_, etc.)
        return getattr(self.estimator, name)


# ========================== Training ==========================

def train_model(config_path="config.yaml"):
    """
    Trains a LightGBM model on engineered features with:
      - purged walk-forward validation (time-aware)
      - early stopping
      - isotonic probability calibration (on validation fold) via FrozenEstimator
      - precision-leaning threshold tuning (beta configurable; default 0.7)
      - feature pruning (near-constant & highly collinear)
      - optional regime feature
      - artifacts: model, calibrated wrapper (or fallback), metadata, importances
    """
    print("\n--- Starting Model Training ---")
    try:
        cfg = _read_config(config_path)
        processed_path = Path(cfg["data_paths"]["processed"])
        model_dir = Path(cfg["ml_models"]["saved_models"])
        n_folds = int(cfg.get("ml_models", {}).get("wf_folds", 5))
        horizon_days = int(cfg.get("ml_models", {}).get("label_horizon_days", 1))
        gap_days = int(cfg.get("ml_models", {}).get("purge_gap_days", 1))
        use_regime = bool(cfg.get("ml_models", {}).get("use_regime_feature", True))
        beta = float(cfg.get("ml_models", {}).get("threshold_beta", 0.7))
        model_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Load data
    try:
        df = _load_processed_csv(processed_path)
    except Exception as e:
        print(f"Error loading processed CSV: {e}")
        return

    if "target" not in df.columns:
        print("Error: 'target' column missing. Run feature engineering first.")
        return

    if use_regime:
        df = _try_join_regime_feature(df, config_path)

    X_all, y_all, feature_names = _choose_features(df)

    mask = X_all.notna().all(axis=1) & y_all.notna()
    X_all, y_all = X_all[mask], y_all[mask]

    # Prune features
    nunq = X_all.nunique()
    keep = nunq[nunq > 3].index
    dropped_const = [c for c in X_all.columns if c not in keep]
    X_all = X_all[keep]; feature_names = [c for c in feature_names if c in keep]

    if X_all.shape[1] > 1:
        corr = X_all.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > 0.98).any()]
    else:
        to_drop = []
    if to_drop:
        X_all = X_all.drop(columns=to_drop)
        feature_names = [c for c in feature_names if c not in to_drop]

    print(f"Total usable rows: {len(X_all)} | Features: {len(feature_names)} "
          f"(dropped {len(dropped_const)} near-constant, {len(to_drop)} collinear)")

    if len(X_all) < 300:
        print("Warning: small dataset; consider more history or more tickers.")

    # Walk-forward folds
    folds = list(_purged_walk_forward_splits(
        X_all, n_folds=n_folds, horizon_days=horizon_days, gap_days=gap_days
    ))
    if not folds:
        print("Error: could not generate walk-forward folds.")
        return

    fold_metrics = []
    best_model = None
    best_importances = None
    last_thr = 0.5
    last_calibrated = False

    for i, (tr_mask, va_mask) in enumerate(folds, 1):
        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_va, y_va = X_all[va_mask], y_all[va_mask]

        pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
        scale_pos_weight = (neg / max(1, pos)) if pos and neg else 1.0
        print(f"\n[Fold {i}/{len(folds)}] Train={X_tr.shape}, Val={X_va.shape}, "
              f"class balance pos={pos}, neg={neg}, spw={scale_pos_weight:.2f}")

        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=42,
            n_estimators=4000,
            learning_rate=0.02,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=50,
            min_sum_hessian_in_leaf=1e-3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.5,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            max_bin=255,
        )

        categorical_feature = ["ticker_id"] if "ticker_id" in feature_names else "auto"

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["auc", "average_precision", "binary_logloss"],
            categorical_feature=categorical_feature,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=True),
                lgb.log_evaluation(period=100),
            ],
        )

        # Probability calibration (guard single-class)
        if len(np.unique(y_va)) < 2:
            print(f"[Fold {i}] Skipping calibration/thresholding: validation is single-class.")
            cal = None
            va_proba = model.predict_proba(X_va)[:, 1]
            thr, fbest = 0.5, np.nan
            calibrated = False
        else:
            cal = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="isotonic", cv="prefit")
            cal.fit(X_va, y_va)
            va_proba = cal.predict_proba(X_va)[:, 1]
            thr, fbest = _optimal_threshold(y_va.values, va_proba, beta=beta)
            calibrated = True

        va_pred = (va_proba >= thr).astype(int)
        acc = accuracy_score(y_va, va_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_va, va_pred, average="binary", zero_division=0)
        auc = roc_auc_score(y_va, va_proba) if len(np.unique(y_va)) > 1 else float("nan")
        ap = average_precision_score(y_va, va_proba) if len(np.unique(y_va)) > 1 else float("nan")

        print(f"[Fold {i}] AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  P/R={p:.4f}/{r:.4f}  thr={thr:.4f}")

        fold_metrics.append(dict(auc=float(auc), ap=float(ap), f1=float(f1),
                                 precision=float(p), recall=float(r), thr=float(thr)))

        best_model = model
        best_importances = pd.DataFrame({
            "feature": feature_names,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        last_thr = float(thr)
        last_calibrated = calibrated

    # ----------------- Persist artifacts from last fold -----------------
    model_dir = Path(_read_config(config_path)["ml_models"]["saved_models"])
    model_dir.mkdir(parents=True, exist_ok=True)

    tr_mask, va_mask = folds[-1]
    if len(np.unique(y_all[va_mask])) >= 2:
        cal = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="isotonic", cv="prefit")
        cal.fit(X_all[va_mask], y_all[va_mask])
        calibrated = True
    else:
        cal = best_model
        calibrated = False

    model_path = model_dir / "lgbm_model.pkl"
    cal_path = model_dir / "lgbm_calibrated.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(cal, cal_path)

    imp_csv = model_dir / "feature_importances.csv"
    best_importances.to_csv(imp_csv, index=False)

    # Aggregate fold metrics
    agg = {
        "auc": float(np.nanmean([m["auc"] for m in fold_metrics])),
        "average_precision": float(np.nanmean([m["ap"] for m in fold_metrics])),
        "f1": float(np.nanmean([m["f1"] for m in fold_metrics])),
        "precision": float(np.nanmean([m["precision"] for m in fold_metrics])),
        "recall": float(np.nanmean([m["recall"] for m in fold_metrics])),
        "threshold_mean": float(np.nanmean([m["thr"] for m in fold_metrics])),
        "threshold_last": last_thr,
        "folds": len(fold_metrics),
        "horizon_days": horizon_days,
        "gap_days": gap_days,
    }

    meta = {
        "features": feature_names,
        "metrics_cv": agg,
        "use_regime_feature": use_regime,
        "wf_folds": n_folds,
        "label_horizon_days": horizon_days,
        "purge_gap_days": gap_days,
        "threshold_beta": beta,
        "calibrated": calibrated,
        "artifacts": {
            "model": str(model_path),
            "calibrated_model": str(cal_path),
            "importances": str(imp_csv),
        },
    }
    meta_path = model_dir / "lgbm_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n--- CV Summary ---")
    print(json.dumps(agg, indent=2))
    print(f"\nSaved model         : {model_path}")
    print(f"Saved calibrated    : {cal_path}")
    print(f"Saved metadata      : {meta_path}")
    print(f"Saved importances   : {imp_csv}")
    print("--- Model Training Complete ---")


if __name__ == "__main__":
    # Run: python -m src.training.train
    train_model(config_path="config.yaml")
