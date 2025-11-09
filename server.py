# server.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yaml
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Project utilities you already have ---
from src.utils import get_current_regime, get_latest_features

# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def read_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _load_lgbm_for_inference(cfg: dict):
    """
    Load model for inference.
    Prefer calibrated pickle, but UNWRAP to underlying estimator if present.
    Fallback to raw model. Returns (model, feature_list, threshold).
    """
    saved = Path(cfg["ml_models"]["saved_models"])
    meta_path = saved / "lgbm_metadata.json"
    cal_path  = saved / "lgbm_calibrated.pkl"
    raw_path  = saved / "lgbm_model.pkl"

    if not meta_path.exists():
        raise FileNotFoundError("Missing lgbm_metadata.json in saved_models/")
    meta = json.load(open(meta_path))
    features = meta["features"]

    # Threshold can come from CV (preferred) or single-run metrics
    thr = float(
        meta.get("metrics_cv", {}).get("threshold_mean",
            meta.get("metrics_cv", {}).get("threshold_last",
                meta.get("threshold", 0.5)))
    )

    model = None
    if cal_path.exists():
        try:
            m = joblib.load(cal_path)
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
        if isinstance(v, (int, float, np.number)) and np.isfinite(v):
            parts.append(f"{k}:{float(v):.4f}")
            if len(parts) >= max_fields:
                break
    return " ".join(parts)

# -----------------------------------------------------------------------------
# Server-side ticker detection (mirrors your frontend but expanded)
# -----------------------------------------------------------------------------
TICKER_LOOKUP = {
    # Apple
    "AAPL": "AAPL", "APPLE": "AAPL", "APPL": "AAPL", "$AAPL":"AAPL", "$APPLE":"AAPL",
    # Microsoft
    "MSFT": "MSFT", "MICROSOFT": "MSFT", "$MSFT":"MSFT",
    # Alphabet / Google
    "GOOGL":"GOOGL", "GOOGLE":"GOOGL", "ALPHABET":"GOOGL", "$GOOGL":"GOOGL",
    # Nvidia
    "NVDA":"NVDA", "NVIDIA":"NVDA", "$NVDA":"NVDA",
    # Tesla
    "TSLA":"TSLA", "TESLA":"TSLA", "$TSLA":"TSLA",
    # S&P 500
    "GSPC":"GSPC", "^GSPC":"GSPC", "S&P":"GSPC", "S&P500":"GSPC", "S&P 500":"GSPC", "SP500":"GSPC", "THE MARKET":"GSPC"
}
ALIASES_REGEX_SAFE = {k.replace('\\','\\\\').replace('.','\\.').replace('+','\\+')
                        .replace('*','\\*').replace('?','\\?').replace('^','\\^')
                        .replace('$','\\$').replace('|','\\|').replace('{','\\{')
                        .replace('}','\\}').replace('(','\\(').replace(')','\\)')
                        .replace('[','\\[').replace(']','\\]').replace('/','\\/'): v
                      for k,v in TICKER_LOOKUP.items()}

import re
def detect_ticker_hint(text: str) -> Optional[str]:
    cleaned = re.sub(r"[.,!?():]", " ", (text or "").upper())
    for alias, official in ALIASES_REGEX_SAFE.items():
        if re.search(rf"(^|\s|\$){alias}(\s|$)", cleaned):
            return official
    return None

# -----------------------------------------------------------------------------
# TF-IDF RAG (no native deps)
# -----------------------------------------------------------------------------
class TfidfEmbedder:
    def __init__(self, model_path: str | Path = "artifacts/tfidf_vectorizer.pkl"):
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
        self.path = Path(model_path)
        self.TfidfVectorizer = TfidfVectorizer
        self.vectorizer = None

    def fit(self, texts: list[str]):
        self.vectorizer = self.TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2))
        self.vectorizer.fit(texts)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, self.path)

    def load(self):
        self.vectorizer = joblib.load(self.path)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            self.load()
        mat = self.vectorizer.transform(texts)
        return mat.toarray().astype("float32")

class SklearnStore:
    def __init__(self, index_path: str | Path = "artifacts/tfidf_nbrs.pkl",
                 meta_path: str | Path = "artifacts/cortexa.meta.json"):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.nbrs = None
        self.meta: list[dict] = []

    def build(self, vecs: np.ndarray, metas: list[dict], n_neighbors: int = 50):
        from sklearn.neighbors import NearestNeighbors  # lazy
        self.nbrs = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
        self.nbrs.fit(vecs)
        self.meta = metas
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.nbrs, self.index_path)
        self.meta_path.write_text(json.dumps(self.meta))

    def load(self):
        self.nbrs = joblib.load(self.index_path)
        self.meta = json.loads(self.meta_path.read_text())
        return self

    def search(self, qvec: np.ndarray, k: int = 5):
        dist, idx = self.nbrs.kneighbors(qvec.reshape(1, -1), n_neighbors=k, return_distance=True)
        sims = 1.0 - dist[0]
        out = []
        for s, i in zip(sims.tolist(), idx[0].tolist()):
            if 0 <= i < len(self.meta):
                out.append((float(s), self.meta[i]))
        return out

def ensure_tfidf_index(cfg: dict):
    """
    Ensure artifacts/rag_contexts.csv exists, and TF-IDF index is built.
    """
    artifacts = Path(cfg.get("data_paths", {}).get("artifacts", "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)
    contexts_csv = artifacts / "rag_contexts.csv"
    vec_path = artifacts / "tfidf_vectorizer.pkl"
    nbrs_path = artifacts / "tfidf_nbrs.pkl"
    meta_path = artifacts / "cortexa.meta.json"

    # Create rag_contexts.csv if missing
    if not contexts_csv.exists():
        processed_dir = Path(cfg["data_paths"]["processed"])
        src = processed_dir / "features_and_targets.csv"
        if not src.exists():
            print("[RAG] features_and_targets.csv missing; skipping context build.")
            return
        df = pd.read_csv(src)

        # robust datetime index
        for col in ("Date", "Datetime", "date", "datetime", "timestamp", "time"):
            if col in df.columns:
                s = pd.to_datetime(df[col], errors="coerce", utc=False)
                if s.notna().mean() >= 0.8:
                    df[col] = s
                    df = df[s.notna()].set_index(col)
                    break
        if not isinstance(df.index, pd.DatetimeIndex):
            first = df.columns[0]
            s = pd.to_datetime(df[first], errors="coerce", utc=False)
            if s.notna().mean() > 0.8:
                df[first] = s
                df = df[s.notna()].set_index(first)
        if not isinstance(df.index, pd.DatetimeIndex):
            print("[RAG] Could not create DatetimeIndex; skipping context build.")
            return

        df = df[~df.index.duplicated(keep="last")].sort_index()

        drop = {"target","future_price","future_ret",
                "Open","High","Low","Close","Volume",
                "open","high","low","close","volume","Adj Close","ticker_id"}
        numeric = df.select_dtypes(include=[np.number]).copy()

        texts = []
        for idx in numeric.index:
            parts = []
            cnt = 0
            for k, v in numeric.loc[idx].items():
                if k in drop: continue
                if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                    parts.append(f"{k}:{float(v):.4f}")
                    cnt += 1
                    if cnt >= 40: break
            texts.append(" ".join(parts))

        out = pd.DataFrame({
            "date": df.index.astype("datetime64[ns]"),
            "ticker": (df["ticker"] if "ticker" in df.columns else ""),
            "regime": (df["regime"] if "regime" in df.columns else -1),
            "target": (df["target"] if "target" in df.columns else np.nan),
            "text": texts,
            "event_type": "market_state",
        })
        out = out[out["text"].str.len() > 0].copy()
        contexts_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(contexts_csv, index=False)
        print(f"[RAG] Wrote contexts -> {contexts_csv} (rows={len(out)})")
    else:
        print(f"[RAG] Using existing contexts: {contexts_csv}")

    # Build TF-IDF index if missing
    if not (vec_path.exists() and nbrs_path.exists() and meta_path.exists()):
        df = pd.read_csv(contexts_csv)
        texts = df["text"].astype(str).tolist()
        metas = df.to_dict(orient="records")

        emb = TfidfEmbedder(vec_path).fit(texts)
        X = emb.encode(texts)

        store = SklearnStore(nbrs_path, meta_path)
        store.build(X, metas, n_neighbors=50)
        print(f"[RAG] Built TF-IDF index -> {vec_path}, {nbrs_path}, {meta_path}")
    else:
        print(f"[RAG] TF-IDF index OK: {vec_path.name}, {nbrs_path.name}")

def tfidf_query_freeform(question: str, k: int = 8, ticker_hint: Optional[str] = None) -> dict:
    """
    Local TF-IDF retrieval for /query when no external vector DB is configured.
    Returns a context dict compatible with your LLM agent:
    { documents: [[text...]], metadatas: [[meta...]] }
    """
    artifacts = Path("artifacts")
    vec_path = artifacts / "tfidf_vectorizer.pkl"
    nbrs_path = artifacts / "tfidf_nbrs.pkl"
    meta_path = artifacts / "cortexa.meta.json"

    # If index missing, bail gracefully
    if not (vec_path.exists() and nbrs_path.exists() and meta_path.exists()):
        return {"documents": [[]], "metadatas": [[]]}

    emb = TfidfEmbedder(vec_path).load()
    store = SklearnStore(nbrs_path, meta_path).load()
    qvec = emb.encode([question])[0]
    results = store.search(qvec, k=k * 4)  # overfetch, we will filter by ticker

    if not results:
        return {"documents": [[]], "metadatas": [[]]}

    # Apply ticker filter if hinted; if filter wipes out, fall back to unfiltered
    metas_all = [m for _, m in results]
    if ticker_hint:
        filt = [m for m in metas_all if str(m.get("ticker","")).upper() == ticker_hint.upper()]
        metas = filt if filt else metas_all
    else:
        metas = metas_all

    metas = metas[:k]
    docs = [m.get("text", "") for m in metas]
    meta_out = []
    for m in metas:
        meta_out.append({
            "title": m.get("title", f"{m.get('ticker','')} {m.get('date','')}").strip(),
            "source": m.get("source", "local-tfidf"),
            "published": str(m.get("date", "")),
            "link": m.get("link", "#"),
            **m
        })
    return {"documents": [docs], "metadatas": [meta_out]}

# -----------------------------------------------------------------------------
# RAG probability for the quant signal (still TF-IDF; filtered by regime)
# -----------------------------------------------------------------------------
def rag_probability_tfidf(latest_row: pd.Series, regime: int, k: int = 5) -> tuple[float, list[dict]]:
    artifacts = Path("artifacts")
    emb = TfidfEmbedder(artifacts / "tfidf_vectorizer.pkl").load()
    store = SklearnStore(artifacts / "tfidf_nbrs.pkl", artifacts / "cortexa.meta.json").load()

    qtext = _compose_context_text(latest_row)
    qvec = emb.encode([qtext])[0]
    results = store.search(qvec, k=k * 3)

    # prefer same-regime hits
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

# -----------------------------------------------------------------------------
# RAG Signal Engine (quant)
# -----------------------------------------------------------------------------
@dataclass
class SignalOutput:
    proba_ml: float
    proba_rag: float
    final_score: float
    decision: int
    context: dict

class RAGSignalEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model, self.feature_list, self.threshold = _load_lgbm_for_inference(cfg)
        rag = cfg.get("rag", {}) or {}
        self.k = int(rag.get("k", 5))
        self.alpha_ml = float(rag.get("alpha_ml", 0.5))
        self.alpha_rag = float(rag.get("alpha_rag", 0.5))
        self.backend = rag.get("embedder", {}).get("backend", "tfidf")

    @classmethod
    def from_artifacts(cls, config_path: str = "config.yaml") -> "RAGSignalEngine":
        cfg = read_cfg(config_path)
        if cfg.get("rag", {}).get("embedder", {}).get("backend", "tfidf") == "tfidf":
            ensure_tfidf_index(cfg)
        return cls(cfg)

    def score(self, latest_row: pd.Series) -> SignalOutput:
        regime = int(latest_row.get("regime", get_current_regime("config.yaml")))
        X = _align_features(latest_row, self.feature_list)
        p_ml = float(self.model.predict_proba(X)[:, 1][0])
        if self.backend == "tfidf":
            p_rag, ctx = rag_probability_tfidf(latest_row, regime, self.k)
        else:
            p_rag, ctx = 0.5, []
        score = float(self.alpha_ml * p_ml + self.alpha_rag * p_rag)
        decision = int(score >= self.threshold)
        return SignalOutput(
            proba_ml=p_ml,
            proba_rag=p_rag,
            final_score=score,
            decision=decision,
            context={"filters": {"regime": regime}, "hits": ctx}
        )

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
cortexa_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- 🚀 Server starting up... ---")
    try:
        cortexa_models["rag_engine"] = RAGSignalEngine.from_artifacts("config.yaml")
        cortexa_models["current_regime"] = get_current_regime("config.yaml")
        print("--- ✅ Server startup complete. Models loaded. ---")
    except Exception as e:
        print(f"Startup error: {e}")
        cortexa_models.clear()
        raise
    yield
    cortexa_models.clear()
    print("--- 🔌 Server shutting down. ---")

app = FastAPI(title="Cortexa API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def read_root():
    return {"status": "Cortexa API is running"}

@app.post("/query")
def run_cortexa_pipeline(request: QueryRequest):
    """
    Qualitative RAG pipeline:
      1) Try external vector DB (if configured).
      2) Fallback to local TF-IDF contexts (filtered by detected ticker if any).
      3) If LLM agent not installed or no paid keys, `agent.py` will gracefully fall back.
    """
    q = request.text
    print(f"API received RAG query: {q}")
    cfg = read_cfg("config.yaml")

    # detect ticker hint from free text
    ticker_hint = detect_ticker_hint(q)

    # Try external vector DB if configured
    context = None
    try:
        ragcfg = cfg.get("rag", {}) or {}
        _ = ragcfg["vector_db_collection"]  # raises KeyError if missing
        from src.rag.retrieval import query_vector_db  # lazy import
        context = query_vector_db(q, config_path="config.yaml")
    except Exception as e:
        print(f"--- Querying via local TF-IDF (reason: {e}) ---")
        context = tfidf_query_freeform(q, k=int(cfg.get("rag", {}).get("k", 8)), ticker_hint=ticker_hint)

    if (not context) or ("documents" not in context) or (not context["documents"]) or (not context["documents"][0]):
        return {
            "answer": "I could not find any relevant data in my current knowledge base to answer that.",
            "sources": []
        }

    # Try LLM reasoning (your agent handles free fallback / Ollama / Gemini opt-in)
    try:
        from src.reasoning.agent import get_llm_reasoning  # lazy import
        answer = get_llm_reasoning(q, context)
    except Exception as e:
        print(f"[RAG] LLM agent not available or failed ({e}); returning contexts only.")
        # Basic extractive fallback if agent import itself is missing
        docs = context["documents"][0][:4]
        answer = "**Context-based summary (no LLM):**\n" + "\n".join(f"- {d}" for d in docs)

    # Build sources list
    sources = []
    try:
        meta_list = context.get('metadatas', [[]])[0]
        for m in meta_list[:5]:
            sources.append({
                "title": m.get('title', 'N/A'),
                "source": m.get('source', 'local-tfidf'),
                "date":  m.get('published', m.get('date', 'N/A')),
                "link":  m.get('link', '#')
            })
    except Exception:
        pass

    return {"answer": answer, "sources": sources}

@app.get("/predict/{ticker}")
def get_prediction_endpoint(ticker: str):
    """
    Returns the final RAG-enhanced quantitative signal for a ticker.
    """
    print(f"API received prediction request for: {ticker}")

    engine: RAGSignalEngine | None = cortexa_models.get("rag_engine")  # type: ignore
    if engine is None:
        raise HTTPException(status_code=503, detail="RAGSignalEngine is not loaded.")

    latest_row = get_latest_features(ticker)
    if latest_row is None:
        return {"signal": "No Data", "detail": "No feature data found for this ticker."}

    current_regime = cortexa_models.get("current_regime", 0)
    latest_row['regime'] = current_regime

    try:
        signal_output = engine.score(latest_row)
        return {
            "signal": "BUY (UP)" if signal_output.decision == 1 else "SELL (DOWN/SAME)",
            "ml_probability": f"{signal_output.proba_ml * 100:.2f}%",
            "rag_probability": f"{signal_output.proba_rag * 100:.2f}%",
            "combined_score": f"{signal_output.final_score * 100:.2f}%",
            "rag_contexts_found": len(signal_output.context['hits']),
            "filter_used": signal_output.context['filters']
        }
    except Exception as e:
        print(f"Error during signal scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Error scoring signal: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
