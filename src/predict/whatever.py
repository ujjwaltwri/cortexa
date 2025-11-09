# src/predict/whatever.py
import joblib, json
from pathlib import Path
cfg = yaml.safe_load(open("config.yaml"))
sav = Path(cfg["ml_models"]["saved_models"])
cal = joblib.load(sav / "lgbm_calibrated.pkl")  # <— use this
meta = json.load(open(sav / "lgbm_metadata.json"))
thr = meta["metrics_cv"]["threshold_last"]  # or "threshold_mean"

proba = cal.predict_proba(latest_features.values.reshape(1, -1))[0, 1]
pred = int(proba >= thr)
