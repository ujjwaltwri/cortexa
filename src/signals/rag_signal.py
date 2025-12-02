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
from src.utils import get_current_regime, get_latest_features 

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
                 model: BaseEstimator, 
                 ml_features: List[str], 
                 embedder: Embedder, 
                 vecdb: QdrantVecDB, 
                 w_ml=0.5, 
                 w_rag=0.5, 
                 final_threshold=0.51): # Changed default to 0.51
        
        self.model = model
        self.ml_features = ml_features
        self.embedder = embedder
        self.vecdb = vecdb
        self.w_ml = w_ml
        self.w_rag = w_rag
        self.final_threshold = final_threshold
        print(f"--- RAGSignalEngine Initialized (Threshold: {self.final_threshold}) ---")

    @classmethod
    def from_artifacts(cls, config_path="config.yaml"):
        print("--- Initializing RAGSignalEngine from artifacts ---")
        try:
            config = yaml.safe_load(open(config_path))
            model_dir = Path(config['ml_models']['saved_models'])
            
            # LOAD THRESHOLD FROM CONFIG
            # This allows us to tune sensitivity without changing code
            threshold = config.get('ml_models', {}).get('decision_threshold', 0.51)
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
        
        # 1. Load ML Model
        model_path = model_dir / "rf_model.pkl"
        if not model_path.exists():
            model_path = model_dir / "lgbm_model.pkl"
            
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}.")
            return None
            
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # 2. Load Metadata
        meta_path = model_dir / "model_metadata.json"
        if not meta_path.exists():
             meta_path = model_dir / "lgbm_wf_metadata.json"
             
        if not meta_path.exists():
            print(f"Error: Metadata not found.")
            # Fallback to model features if metadata missing
            if hasattr(model, "feature_names_in_"):
                ml_features = list(model.feature_names_in_)
            else:
                return None
        else:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            ml_features = metadata['features']
        
        # 3. Init RAG components
        try:
            embedder = Embedder() 
            vecdb = QdrantVecDB(vector_size=384) 
        except Exception as e:
            print(f"Error connecting to RAG components: {e}")
            return None
        
        return cls(model, ml_features, embedder, vecdb, final_threshold=threshold)

    def _current_state_text(self, row: pd.Series) -> str:
        pieces = [f"{k}:{float(row[k]):.4f}" for k in self.ml_features if k in row and isinstance(row[k], (int, float)) and np.isfinite(row[k])]
        return " ".join(pieces)

    def score(self, row: pd.Series, meta_filters: Dict[str, Any] | None = None) -> SignalOutput:
        
        # --- 1. ML PROBABILITY ---
        try:
            input_data = {k: [row[k]] for k in self.ml_features if k in row}
            X_live = pd.DataFrame(input_data)
            
            if hasattr(self.model, "feature_names_in_"):
                # Add missing cols as 0
                for col in self.model.feature_names_in_:
                    if col not in X_live.columns:
                        X_live[col] = 0.0
                X_live = X_live[self.model.feature_names_in_]
            
            proba_ml = float(self.model.predict_proba(X_live)[0, 1])
        except Exception as e:
            print(f"ML Prediction Warning: {e}. Defaulting to 0.5")
            proba_ml = 0.5

        # --- 2. RAG PROBABILITY ---
        q_text = self._current_state_text(row)
        q_vec = self.embedder.encode([q_text])[0]
        
        if not meta_filters:
            meta_filters = {}
            if "market_regime" in row:
                meta_filters["regime"] = int(row["market_regime"])
            elif "regime" in row:
                meta_filters["regime"] = int(row["regime"])
        
        hits = self.vecdb.search(q_vec, top_k=30, filters=meta_filters)

        if len(hits) < 5:
            proba_rag = 0.5
            effective_w_rag = 0.0
            effective_w_ml = 1.0
        else:
            wins = sum([h['meta'].get('target', 0) for h in hits])
            total = len(hits)
            proba_rag = (wins + 1) / (total + 2)
            effective_w_rag = self.w_rag
            effective_w_ml = self.w_ml

        # --- 3. COMBINE ---
        final_score = (effective_w_ml * proba_ml) + (effective_w_rag * proba_rag)
        
        # DECISION: Using the tuned threshold
        decision = int(final_score >= self.final_threshold)

        return SignalOutput(
            proba_ml=proba_ml, 
            proba_rag=proba_rag, 
            final_score=final_score, 
            decision=decision,
            context={"q_text": q_text, "hits": hits[:5], "filters": meta_filters}
        )

if __name__ == "__main__":
    print("--- Testing RAGSignalEngine ---")
    engine = RAGSignalEngine.from_artifacts("config.yaml")
    if engine:
        print("Fetching latest features for AAPL...")
        test_row = get_latest_features(ticker="AAPL")
        if test_row is not None:
            current_regime = get_current_regime(config_path="config.yaml")
            test_row['market_regime'] = current_regime 
            
            signal_output = engine.score(test_row)
            print("\n--- RAG-Enhanced Signal Output ---")
            print(f"ML Probability:    {signal_output.proba_ml:.4f}")
            print(f"RAG Probability:   {signal_output.proba_rag:.4f}")
            print(f"Combined Score:    {signal_output.final_score:.4f}")
            print(f"Final Decision:    {signal_output.decision} (Threshold: {engine.final_threshold})")