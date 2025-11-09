# src/signals/rag_signal.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml, joblib

from src.utils import get_current_regime, get_latest_features

# Quiet down sklearn calibration warnings
warnings.filterwarnings(
    "ignore",
    message="The `cv='prefit'` option is deprecated",
    category=FutureWarning,
)


# ---------------------------- Config dataclass ----------------------------
@dataclass
class RAGConfig:
    backend: str          # "tfidf", "local", or "none"
    k: int
    alpha_ml: float
    alpha_rag: float
    index_path: Path | None
    meta_path: Path | None
    embed_model: str | None
    dim: int | None


# ---------------------------- Helpers ----------------------------
def _read_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_lgbm_for_inference(cfg):
    """
    Load a model for inference.
    Prefer the calibrated pickle, but UNWRAP to the underlying estimator
    (m.estimator or m.base_estimator). Fall back to raw model if needed.
    Returns: (model, feature_list, threshold)
    """
    saved = Path(cfg["ml_models"]["saved_models"])
    meta_path = saved / "lgbm_metadata.json"
    cal_path  = saved / "lgbm_calibrated.pkl"
    raw_path  = saved / "lgbm_model.pkl"

    if not meta_path.exists():
        raise FileNotFoundError("Missing lgbm_metadata.json in saved_models/")
    meta = json.load(open(meta_path))
    features = meta["features"]
    thr = float(meta.get("metrics_cv", {}).get("threshold_mean",
             meta.get("metrics_cv", {}).get("threshold_last", 0.5)))

    model = None
    if cal_path.exists():
        try:
            m = joblib.load(cal_path)
            # unwrap calibrated wrapper if present
            if hasattr(m, "estimator") and m.estimator is not None:
                model = m.estimator
            elif hasattr(m, "base_estimator") and m.base_estimator is not None:
                model = m.base_estimator
            else:
                model = m
        except Exception as e:
            print(f"[WARN] Calibrated model load failed ({e}). Falling back to raw model.")

    if model is None:
        if not raw_path.exists():
            raise FileNotFoundError("No trained model found (neither calibrated nor raw).")
        model = joblib.load(raw_path)

    return model, features, thr


def _rag_from_cfg(cfg) -> RAGConfig:
    rag = cfg.get("rag", {}) or {}
    backend = rag.get("embedder", {}).get("backend", "none")  # "tfidf", "local", "none"
    k = int(rag.get("k", 5))
    alpha_ml = float(rag.get("alpha_ml", 0.5))
    alpha_rag = float(rag.get("alpha_rag", 0.5))

    if backend == "tfidf":
        idx = Path(rag.get("vector_store", {}).get("path", "artifacts/tfidf_nbrs.pkl"))
        meta = Path(rag.get("vector_store", {}).get("metadata_path", "artifacts/cortexa.meta.json"))
        return RAGConfig("tfidf", k, alpha_ml, alpha_rag, idx, meta, "tfidf", None)

    if backend == "local":
        idx = Path(rag.get("vector_store", {}).get("path", "artifacts/cortexa.faiss"))
        meta = Path(rag.get("vector_store", {}).get("metadata_path", "artifacts/cortexa.meta.json"))
        model = rag.get("embedder", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        dim = int(rag.get("embedder", {}).get("dim", 384))
        return RAGConfig("local", k, alpha_ml, alpha_rag, idx, meta, model, dim)

    return RAGConfig("none", k, alpha_ml, alpha_rag, None, None, None, None)


def _align_features(latest: pd.Series, feature_list: list[str]) -> np.ndarray:
    s = latest.copy()
    for c in feature_list:
        if c not in s.index:
            s[c] = 0.0
    s = s.reindex(feature_list)
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s.values.reshape(1, -1)


def _compose_context_text(row: pd.Series, max_fields: int = 32) -> str:
    parts = []
    for k, v in row.items():
        if isinstance(v, (int, float, np.number)):
            parts.append(f"{k}:{float(v):.4f}")
            if len(parts) >= max_fields:
                break
    return " ".join(parts)


# ---------------------------- RAG backends (lazy imports) ----------------------------
def _rag_probability_tfidf(latest: pd.Series, regime: int, k: int) -> tuple[float, list[dict]]:
    from src.rag.embedder_tfidf import TfidfEmbedder
    from src.rag.store_sklearn import SklearnStore

    embedder = TfidfEmbedder("artifacts/tfidf_vectorizer.pkl").load()
    store = SklearnStore("artifacts/tfidf_nbrs.pkl", "artifacts/cortexa.meta.json").load()

    qtext = _compose_context_text(latest)
    qvec = embedder.encode([qtext])[0]
    results = store.search(qvec, k=k * 3)  # overfetch then filter by regime

    hits = [(s, m) for s, m in results if m.get("regime", -1) == regime]
    hits = hits[:k] if hits else results[:k]

    if not hits:
        return 0.5, []

    sims = np.array([max(0.0, float(s)) for s, _ in hits], dtype=float)
    tars = np.array([float(m.get("target", 0.5)) for _, m in hits], dtype=float)
    rag_p = float(np.dot(sims, tars) / sims.sum()) if sims.sum() else float(tars.mean())

    ctx = [{"score": float(s),
            "text": (m.get("text", "")[:120] + ("..." if len(m.get("text", "")) > 120 else "")),
            "meta": m} for s, m in hits]
    return rag_p, ctx


def _rag_probability_faiss(latest: pd.Series, regime: int, k: int,
                           embed_model: str, idx_path: Path, meta_path: Path, dim: int) -> tuple[float, list[dict]]:
    # FAISS branch — imported only when used
    from src.rag.embedder_local import LocalEmbedder
    from src.rag.store_faiss import FaissStore

    embedder = LocalEmbedder(embed_model)
    store = FaissStore(dim=dim, path=idx_path, meta_path=meta_path)

    qtext = _compose_context_text(latest)
    q = embedder.encode([qtext])[0]
    results = store.search(q, k=k * 3)

    hits = [(s, m) for s, m in results if m.get("regime", -1) == regime]
    hits = hits[:k] if hits else results[:k]

    if not hits:
        return 0.5, []

    sims = np.array([max(0.0, float(s)) for s, _ in hits], dtype=float)
    tars = np.array([float(m.get("target", 0.5)) for _, m in hits], dtype=float)
    rag_p = float(np.dot(sims, tars) / sims.sum()) if sims.sum() else float(tars.mean())

    ctx = [{"score": float(s),
            "text": (m.get("text", "")[:120] + ("..." if len(m.get("text", "")) > 120 else "")),
            "meta": m} for s, m in hits]
    return rag_p, ctx


# ---------------------------- Main ----------------------------
def main():
    print("\n--- Testing RAGSignalEngine ---")
    cfg = _read_cfg("config.yaml")
    ragcfg = _rag_from_cfg(cfg)

    # Load model + metadata (unwrap calibrated -> raw estimator)
    model, feature_list, thr = _load_lgbm_for_inference(cfg)

    # Compute current regime (once)
    regime = get_current_regime("config.yaml")
    print(f"Loaded current regime: {regime}")

    # Pick preview ticker
    ticker = cfg.get("ml_models", {}).get("preview_ticker", "AAPL")

    # Latest engineered row
    latest = get_latest_features(ticker=ticker, config_path="config.yaml")
    if latest is None:
        print(f"No features found for {ticker}.")
        return

    # ML probability
    X = _align_features(latest, feature_list)
    ml_p = float(model.predict_proba(X)[:, 1][0])

    # RAG probability by backend (lazy import to avoid FAISS segfaults)
    if ragcfg.backend == "tfidf":
        rag_p, contexts = _rag_probability_tfidf(latest, regime, ragcfg.k)
    elif ragcfg.backend == "local":
        rag_p, contexts = _rag_probability_faiss(
            latest, regime, ragcfg.k, ragcfg.embed_model, ragcfg.index_path, ragcfg.meta_path, int(ragcfg.dim or 384)
        )
    else:
        rag_p, contexts = 0.5, []

    # Combine
    comb = float(ragcfg.alpha_ml * ml_p + ragcfg.alpha_rag * rag_p)
    decision = int(comb >= thr)

    # Output
    print("\n--- RAG-Enhanced Signal Output ---")
    print(f"ML Probability:    {ml_p:.4f}")
    print(f"RAG Probability:   {rag_p:.4f} (from {len(contexts)} contexts)")
    print(f"Combined Score:    {comb:.4f}")
    print(f"Final Decision:    {decision} (Threshold: {thr:.2f})")

    if contexts:
        top = contexts[0]
        print("\n--- RAG Context ---")
        print(f"Top Hit (Score: {top['score']:.4f}):")
        print(f"  - Text: {top['text']}")
        print(f"  - Meta: {top['meta']}")
    else:
        print("No RAG contexts (backend='none' or empty index).")


if __name__ == "__main__":
    main()
