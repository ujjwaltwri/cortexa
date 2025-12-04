# src/signals/rag_signal.py

import numpy as np
import pandas as pd
import joblib
import json
import yaml
from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path

from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB
from src.utils import get_latest_features


@dataclass
class SignalOutput:
    proba_ml: float
    proba_rag: float
    final_score: float
    decision: int
    context: Dict[str, Any]


class RAGSignalEngine:
    def __init__(
        self,
        model: Any,
        ml_features: List[str],
        ticker_categories: List[str] | None,
        embedder: Embedder,
        vecdb: QdrantVecDB,
        w_ml: float = 0.4,
        w_rag: float = 0.6,
        final_threshold: float = 0.52,
    ):
        """
        V4 Cross-Sectional RAG + ML engine.

        model            : LightGBM cross-sectional classifier
        ml_features      : ordered list of feature names used at training
        ticker_categories: list of ticker categories from training (for pandas Categorical)
        """
        self.model = model
        self.ml_features = ml_features
        self.ticker_categories = ticker_categories or []
        self.embedder = embedder
        self.vecdb = vecdb
        self.w_ml = w_ml
        self.w_rag = w_rag
        self.final_threshold = final_threshold

        print(
            f"--- RAGSignalEngine V4 Initialized "
            f"(w_ml={self.w_ml}, w_rag={self.w_rag}, Threshold={self.final_threshold}) ---"
        )

    @classmethod
    def from_artifacts(cls, config_path: str = "config.yaml") -> "RAGSignalEngine | None":
        print("--- Initializing RAGSignalEngine from artifacts ---")
        try:
            config = yaml.safe_load(open(config_path))
            model_dir = Path(config["ml_models"]["saved_models"])
        except Exception as e:
            print(f"Error loading config: {e}")
            return None

        # 1. Load V4 cross-sectional model
        model_path = model_dir / "lgbm_v4_cross_sectional.pkl"
        if not model_path.exists():
            print(f"Error: V4 Model not found at {model_path}. Run train_v4_cross_sectional.py.")
            # Fallback to older rf_model if present
            model_path = model_dir / "rf_model.pkl"

        if not model_path.exists():
            print("Critical: No model found in saved_models.")
            return None

        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)

        # 2. Load metadata
        meta_path = model_dir / "lgbm_v4_cross_sectional_meta.json"
        if not meta_path.exists():
            meta_path = model_dir / "model_metadata.json"

        if not meta_path.exists():
            print("Error: Metadata file not found for V4 or legacy model.")
            return None

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        ml_features = metadata.get("features", [])
        ticker_categories = metadata.get("ticker_categories", [])

        if not ml_features:
            print("Error: 'features' list missing from metadata.")
            return None

        # 3. Init RAG components
        try:
            embedder = Embedder()
            vecdb = QdrantVecDB(vector_size=384)
        except Exception as e:
            print(f"Error connecting to RAG components: {e}")
            return None

        return cls(model, ml_features, ticker_categories, embedder, vecdb)

    def _current_state_text(self, row: pd.Series) -> str:
        """
        Build a compact numeric feature string for semantic search.
        (Skip non-numerics like 'ticker'). 
        """
        pieces = []
        for k in self.ml_features:
            if k == "ticker":
                continue
            v = row.get(k, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                pieces.append(f"{k}:{float(v):.4f}")
        return " ".join(pieces)

    def _build_ml_input(self, row: pd.Series) -> pd.DataFrame:
        """
        Build a single-row DataFrame with the exact feature schema used in training,
        including ticker as a proper categorical with the same categories.
        """
        data = {}
        for feat in self.ml_features:
            if feat == "ticker":
                data[feat] = [row.get("ticker")]
            else:
                val = row.get(feat, 0.0)
                try:
                    val = float(val)
                    if not np.isfinite(val):
                        val = 0.0
                except Exception:
                    val = 0.0
                data[feat] = [val]

        X = pd.DataFrame(data)

        # Ensure the right dtype for ticker
        if "ticker" in X.columns and self.ticker_categories:
            X["ticker"] = pd.Categorical(
                X["ticker"],
                categories=self.ticker_categories,
            )

        # Order columns exactly like training
        X = X[self.ml_features]

        return X

    def score(self, row: pd.Series) -> SignalOutput:
        # --- 1. ML PROBABILITY (V4) ---
        try:
            X_live = self._build_ml_input(row)
            proba_ml = float(self.model.predict_proba(X_live)[0, 1])
        except Exception as e:
            print(f"ML Prediction Warning: {e}. Defaulting to 0.5")
            proba_ml = 0.5

        # --- 2. RAG PROBABILITY (historical pattern matching) ---
        q_text = self._current_state_text(row)
        q_vec = self.embedder.encode([q_text])[0]

        hits = self.vecdb.search(q_vec, top_k=30)

        if len(hits) < 5:
            proba_rag = 0.5
        else:
            wins = sum(h["meta"].get("target", 0) for h in hits)
            proba_rag = (wins + 1) / (len(hits) + 2)

        # --- 3. COMBINE ---
        final_score = (self.w_ml * proba_ml) + (self.w_rag * proba_rag)
        decision = int(final_score >= self.final_threshold)

        return SignalOutput(
            proba_ml=proba_ml,
            proba_rag=proba_rag,
            final_score=final_score,
            decision=decision,
            context={
                "hits": hits[:5],
                "filters": {},
                "model_version": "V4 Cross-Sectional",
            },
        )


if __name__ == "__main__":
    print("--- Testing Cortexa V4 Engine ---")
    engine = RAGSignalEngine.from_artifacts("config.yaml")

    if engine:
        for t in ["AAPL", "NVDA", "MSFT", "GOOGL", "META", "TSLA"]:
            print(f"\nFetching latest V4 features for {t}...")
            latest_row = get_latest_features(ticker=t)

            if latest_row is None:
                print(f"No latest features found for {t}")
                continue

            res = engine.score(latest_row)
            print(f"ML: {res.proba_ml:.2%} | RAG: {res.proba_rag:.2%}")
            print(
                f"Final: {res.final_score:.2%} -> "
                f"{'✅ BUY' if res.decision else '❌ HOLD'}"
            )
