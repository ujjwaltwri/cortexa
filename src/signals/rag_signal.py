import numpy as np
import pandas as pd
import joblib
import json
import yaml
from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
from sklearn.base import BaseEstimator 

from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB
from src.utils import get_latest_features 

# --- Fix for loading CalibratedClassifierCV models ---
class FrozenEstimator:
    def __init__(self, estimator): self.estimator = estimator
    def predict(self, X): return self.estimator.predict(X)
    def predict_proba(self, X): return self.estimator.predict_proba(X)
# -----------------------------------------------------

@dataclass
class SignalOutput:
    proba_ml: float
    proba_rag: float
    final_score: float
    decision: int
    context: Dict[str, Any]

class RAGSignalEngine:
    def __init__(self, 
                 models: Dict[str, BaseEstimator], 
                 ml_features: List[str], 
                 embedder: Embedder, 
                 vecdb: QdrantVecDB, 
                 w_ml=0.7, 
                 w_rag=0.3, 
                 final_threshold=0.50):
        
        self.models = models
        self.ml_features = ml_features
        self.embedder = embedder
        self.vecdb = vecdb
        self.w_ml = w_ml
        self.w_rag = w_rag
        self.final_threshold = final_threshold
        print(f"--- RAGSignalEngine Initialized (Loaded {len(models)} Regime Models) ---")

    @classmethod
    def from_artifacts(cls, config_path="config.yaml"):
        print("--- Initializing RAGSignalEngine from artifacts ---")
        try:
            config = yaml.safe_load(open(config_path))
            saved_dir = Path(config['ml_models']['saved_models'])
            regime_dir = saved_dir / "regime_models"
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
        
        # 1. Load ALL Regime Models
        models = {}
        regimes = ['1_0', '1_1', '0_0', '0_1']
        
        print("Loading Regime Snipers...")
        for r in regimes:
            path = regime_dir / f"model_{r}.pkl"
            if path.exists():
                models[r] = joblib.load(path)
            else:
                print(f"  Warning: Missing model for {r}")

        if not models:
            print("CRITICAL: No regime models found. Run src/training/train_regime.py")
            return None

        # 2. Define Features (Hardcoded to match Feature Engine V3)
        ml_features = [
            'natr', 'vol_ratio', 'dist_sma_50', 'rsi', 'rvol', 
            'ret_1d', 'realized_vol_20d', 'vol_change', 'range_5d', 
            'trend_strength_sq', 'trend_x_vol', 'vix_chg'
        ]
        
        # 3. Init RAG
        try:
            embedder = Embedder() 
            vecdb = QdrantVecDB(vector_size=384) 
        except Exception as e:
            print(f"Error connecting to RAG components: {e}")
            return None
        
        return cls(models, ml_features, embedder, vecdb)

    def _current_state_text(self, row: pd.Series) -> str:
        pieces = [f"{k}:{float(row[k]):.4f}" for k in self.ml_features if k in row and np.isfinite(row[k])]
        return " ".join(pieces)

    def score(self, row: pd.Series) -> SignalOutput:
        
        # 1. Identify Regime
        regime = str(row.get('regime_tag', '1_0'))
        
        # --- REGIME ROUTING ---
        model = self.models.get(regime)
        if not model:
            # Fallback
            model = self.models.get('1_0')
            if not model: return SignalOutput(0.5, 0.5, 0.0, 0, {})

        # Dynamic Thresholds
        thresholds = {'0_1': 0.55, '1_1': 0.56, '1_0': 0.58, '0_0': 0.60}
        threshold = thresholds.get(regime, 0.55)

        # 2. ML PROBABILITY
        try:
            X_live = pd.DataFrame([row[self.ml_features].to_dict()])
            for f in self.ml_features:
                if f not in X_live.columns: X_live[f] = 0.0
            proba_ml = float(model.predict_proba(X_live)[0, 1])
        except Exception as e:
            print(f"ML Pred Error: {e}")
            proba_ml = 0.5

        # 3. RAG PROBABILITY
        q_text = self._current_state_text(row)
        q_vec = self.embedder.encode([q_text])[0]
        
        # Filter RAG by SAME regime
        # Note: We use the string 'regime_tag' from the row (e.g. '1_0')
        # But the database might store it as 'regime' (int).
        # Since V3 backfill stores 'regime' as string "1_0", we use that.
        filters = {"regime": regime} 
        
        hits = self.vecdb.search(q_vec, top_k=30, filters=filters)

        if len(hits) < 3:
            proba_rag = 0.5
        else:
            wins = sum([h['meta'].get('target', 0) for h in hits])
            proba_rag = (wins + 1) / (len(hits) + 2)

        # 4. COMBINE
        final_score = (self.w_ml * proba_ml) + (self.w_rag * proba_rag)
        decision = int(final_score >= threshold)

        return SignalOutput(
            proba_ml=proba_ml, 
            proba_rag=proba_rag, 
            final_score=final_score, 
            decision=decision,
            # --- FIX: Ensure 'filters' is included in context ---
            context={
                "regime": regime, 
                "hits": hits[:5],
                "filters": filters,  # <--- THIS WAS MISSING
                "threshold": threshold
            }
        )

if __name__ == "__main__":
    print("--- Testing Multi-Regime Engine ---")
    engine = RAGSignalEngine.from_artifacts("config.yaml")
    if engine:
        print("Engine Loaded. Ready for Server.")