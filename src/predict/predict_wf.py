# src/predict/predict_wf.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml, joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

pd.options.mode.copy_on_write = True


# ---------------------- IO / Config ----------------------

def _read_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_artifacts(cfg: dict):
    saved = Path(cfg["ml_models"]["saved_models"])
    meta_path = saved / "lgbm_metadata.json"
    cal_path  = saved / "lgbm_calibrated.pkl"
    raw_path  = saved / "lgbm_model.pkl"

    if cal_path.exists():
        model = joblib.load(cal_path)
        calibrated = True
    elif raw_path.exists():
        model = joblib.load(raw_path)
        calibrated = False
    else:
        raise FileNotFoundError("No model found (expected lgbm_calibrated.pkl or lgbm_model.pkl).")

    if not meta_path.exists():
        raise FileNotFoundError("lgbm_metadata.json missing in saved_models/.")

    meta = json.load(open(meta_path))
    features = meta["features"]
    # default threshold from CV
    cv_thr = float(meta["metrics_cv"].get("threshold_mean", meta["metrics_cv"]["threshold_last"]))
    split_date = meta.get("split_date", cfg.get("ml_models", {}).get("split_date", "2024-01-01"))
    beta = float(meta.get("threshold_beta", 0.7))
    return model, features, cv_thr, split_date, beta, meta


# ---------------------- Date parsing helpers ----------------------

_KNOWN_DT_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M:%S",
)

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    for fmt in _KNOWN_DT_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=False)
        if parsed.notna().mean() >= 0.8:
            return parsed.dt.tz_localize(None)
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    return parsed.dt.tz_localize(None)


def _load_processed_csv(processed_dir: Path) -> pd.DataFrame:
    fp = processed_dir / "features_and_targets.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Run feature_engine first.")
    df = pd.read_csv(fp)

    # Preferred date columns
    for col in ("Date", "Datetime", "date", "datetime", "timestamp", "time"):
        if col in df.columns:
            s = _parse_datetime_series(df[col])
            if s.notna().mean() > 0.8:
                df[col] = s
                df = df[s.notna()].set_index(col)
                break

    if not isinstance(df.index, pd.DatetimeIndex):
        first = df.columns[0]
        s = _parse_datetime_series(df[first])
        if s.notna().mean() > 0.8:
            df[first] = s
            df = df[s.notna()].set_index(first)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Datetime index required. Check processed CSV date parsing.")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------- Feature alignment ----------------------

def _choose_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def _align_features(numeric_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = numeric_df.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


# ---------------------- Threshold tuning ----------------------

def _optimal_threshold(y_true, y_proba, beta=0.7):
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
    return float(best_t), float(best_f)


# ---------------------- Main ----------------------

def run(config_path="config.yaml",
        use_cv_threshold: bool = True,
        beta_for_tuning: float = 0.7):

    cfg = _read_config(config_path)
    processed_dir = Path(cfg["data_paths"]["processed"])
    out_dir = Path(cfg.get("data_paths", {}).get("predictions", "data/03_predictions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, features, cv_thr, split_date, beta_train, meta = _load_artifacts(cfg)

    df = _load_processed_csv(processed_dir)
    num = _choose_numeric(df)
    X_all = _align_features(num, features)

    aux_cols = [c for c in ["ticker", "future_price", "future_ret", "target",
                            "Open","High","Low","Close","Volume",
                            "open","high","low","close","volume","Adj Close"]
                if c in df.columns]
    aux = df.loc[X_all.index, aux_cols] if aux_cols else pd.DataFrame(index=X_all.index)

    train_mask = X_all.index < pd.to_datetime(split_date)
    test_mask  = ~train_mask

    thr = cv_thr
    if not use_cv_threshold:
        X_train = X_all[train_mask]
        if "target" in df.columns and len(X_train) >= 50:
            y_train = df.loc[X_train.index, "target"].astype("int8")
            n = len(X_train)
            n_val = max(1, int(n * 0.12))
            X_val, y_val = X_train.iloc[-n_val:], y_train.iloc[-n_val:]
            val_proba = model.predict_proba(X_val)[:, 1]
            thr, fbest = _optimal_threshold(y_val.values, val_proba, beta=beta_for_tuning)
            print(f"Tuned threshold on pre-split tail: thr={thr:.4f} (F{beta_for_tuning:.1f}={fbest:.4f})")
        else:
            print("Threshold tuning skipped: no target or too few rows; using CV threshold.")

    X_test = X_all[test_mask]
    if X_test.empty:
        print("No post-split rows to score; check split_date in metadata/config.")
        return

    aux_test = aux.loc[X_test.index]
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= thr).astype(int)

    out_test = aux_test.copy()
    out_test["proba_up"] = proba_test
    out_test["pred_up"] = pred_test
    out_test["threshold_used"] = thr

    if "target" in out_test.columns:
        y_test = out_test["target"].astype(int)
        try:
            acc = accuracy_score(y_test, pred_test)
            p, r, f1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary", zero_division=0)
            auc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else float("nan")
            ap  = average_precision_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else float("nan")
            print("\n--- Post-split Test Metrics ---")
            print(f"Accuracy   : {acc*100:.2f}%")
            print(f"Precision  : {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
            print(f"ROC-AUC    : {auc:.4f}")
            print(f"AvgPrec    : {ap:.4f}")
        except Exception as e:
            print(f"Metric calc warning: {e}")

    test_path = out_dir / "wf_preds_test.csv"
    out_test.to_csv(test_path)
    print(f"Saved test predictions: {test_path}")

    if "ticker" in out_test.columns:
        ts = out_test.index.name if out_test.index.name else "index"
        latest = (
            out_test.reset_index()
                    .sort_values(["ticker", ts])
                    .groupby("ticker", as_index=False)
                    .tail(1)
                    .set_index(ts)
                    .sort_index()
        )
        latest_path = out_dir / "wf_preds_latest_per_ticker.csv"
        latest.to_csv(latest_path)
        print(f"Saved latest-per-ticker snapshot: {latest_path}")

    proba_all = model.predict_proba(X_all)[:, 1]
    pred_all = (proba_all >= thr).astype(int)
    full = aux.copy()
    full["proba_up"] = proba_all
    full["pred_up"] = pred_all
    full["threshold_used"] = thr
    full_path = out_dir / "wf_preds_full.csv"
    full.to_csv(full_path)
    print(f"Saved full predictions: {full_path}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--use-cv-threshold", type=str, default="true",
                    help="true/false (use training CV threshold or re-tune on tail)")
    ap.add_argument("--beta", type=float, default=0.7, help="F-beta for threshold tuning if retuning")
    args = ap.parse_args()

    use_cv = str(args.use_cv_threshold).lower() in ("true", "1", "yes", "y")
    run(config_path=args.config, use_cv_threshold=use_cv, beta_for_tuning=args.beta)


if __name__ == "__main__":
    cli()
