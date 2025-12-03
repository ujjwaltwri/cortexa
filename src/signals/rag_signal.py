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

@dataclass
class SignalOutput:
    proba_ml: float
    proba_rag: float
    final_score: float
    decision: int
    context: Dict[str, Any]

class RAGSignalEngine:
    def __init__(self, models: Dict[str, Any], embedder: Embedder, vecdb: QdrantVecDB):
        self.models = models # Dict of {regime: {model: clf, features: [cols]}}
        self.embedder = embedder
        self.vecdb = vecdb
        print(f"--- RAGSignalEngine Ready ({len(models)} Regimes) ---")

    @classmethod
    def from_artifacts(cls, config_path="config.yaml"):
        try:
            config = yaml.safe_load(open(config_path))
            saved_dir = Path(config['ml_models']['saved_models'])
            regime_dir = saved_dir / "regime_models"
        except: return None
        
        # Load Model AND Feature List per Regime
        models_data = {}
        regimes = ['1_0', '1_1', '0_0', '0_1']
        
        for r in regimes:
            model_path = regime_dir / f"model_{r}.pkl"
            meta_path = regime_dir / f"model_{r}_meta.json"
            
            if model_path.exists() and meta_path.exists():
                clf = joblib.load(model_path)
                meta = json.load(open(meta_path))
                # Store both the classifier and its specific feature list
                models_data[r] = {
                    "model": clf,
                    "features": meta["features"]
                }
        
        if not models_data:
            print("CRITICAL: No regime models/metadata found.")
            return None

        try:
            embedder = Embedder() 
            vecdb = QdrantVecDB(vector_size=384) 
        except: return None
        
        return cls(models_data, embedder, vecdb)

    def _current_state_text(self, row: pd.Series, features: List[str]) -> str:
        pieces = [f"{k}:{float(row[k]):.4f}" for k in features if k in row and isinstance(row[k], (int, float))]
        return " ".join(pieces)

    def score(self, row: pd.Series) -> SignalOutput:
        # 1. Identify Regime from the row itself (V3 data has this)
        regime = str(row.get('regime_tag', '1_0'))
        
        # Get the specific model bundle for this regime
        bundle = self.models.get(regime)
        if not bundle:
            # Fallback
            bundle = list(self.models.values())[0]
        
        model = bundle["model"]
        feature_list = bundle["features"] # The EXACT features this model wants

        # --- ML PROBABILITY (Aligned) ---
        try:
            # Create DataFrame with ONLY the required features in correct order
            input_data = {k: [row.get(k, 0.0)] for k in feature_list}
            X_live = pd.DataFrame(input_data)
            X_live = X_live[feature_list] # Enforce order
            
            proba_ml = float(model.predict_proba(X_live)[0, 1])
        except Exception as e:
            print(f"ML Pred Error: {e}")
            proba_ml = 0.5

        # --- RAG PROBABILITY (V3) ---
        q_text = self._current_state_text(row, feature_list)
        q_vec = self.embedder.encode([q_text])[0]
        
        # Filter by SAME regime tag
        filters = {"regime": regime}
        hits = self.vecdb.search(q_vec, top_k=30, filters=filters)

        if len(hits) < 3:
            proba_rag = 0.5
        else:
            wins = sum([h['meta'].get('target', 0) for h in hits])
            # Laplace Smoothing
            proba_rag = (wins + 1) / (len(hits) + 2)

        # --- Thresholds ---
        # Conservative thresholds for live trading
        threshold = 0.58 if regime == '1_0' else 0.55
        
        # Combine
        final_score = (0.6 * proba_ml) + (0.4 * proba_rag)
        decision = int(final_score >= threshold)

        return SignalOutput(
            proba_ml=proba_ml, 
            proba_rag=proba_rag, 
            final_score=final_score, 
            decision=decision,
            context={"regime": regime, "hits": hits[:5], "filters": filters}
        )

# --- UPDATED TEST BLOCK: Multi-Ticker Check ---
if __name__ == "__main__":
    print("--- Testing Cortexa 3.0 (Multi-Ticker) ---")
    engine = RAGSignalEngine.from_artifacts("config.yaml")
    
    if engine:
        # List of tickers to check
        tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]
        
        print(f"\n{'TICKER':<8} | {'REGIME':<8} | {'ML':<8} | {'RAG':<8} | {'SCORE':<8} | {'DECISION'}")
        print("-" * 65)
        
        for t in tickers:
            test_row = get_latest_features(ticker=t)
            
            if test_row is not None:
                res = engine.score(test_row)
                decision_str = "✅ BUY" if res.decision else "❌ HOLD"
                
                print(f"{t:<8} | {res.context['regime']:<8} | {res.proba_ml:.2%}   | {res.proba_rag:.2%}   | {res.final_score:.2%}   | {decision_str}")
            else:
                print(f"{t:<8} | -- NO DATA --")