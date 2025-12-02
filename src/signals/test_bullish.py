import pandas as pd
import yaml
import numpy as np
from src.signals.rag_signal import RAGSignalEngine

def test_bullish_scenario():
    print("--- 🧪 Testing a 'Steady Growth' Scenario ---")
    
    engine = RAGSignalEngine.from_artifacts("config.yaml")
    if not engine:
        print("Failed to load engine.")
        return

    # TUNE: Lower the bar slightly to catch the edge
    engine.final_threshold = 0.51 

    # Construct a "Goldilocks" Day (Strong but SAFE)
    fake_data = {
        # --- Context ---
        'market_regime': 1,       # Bull Market
        
        # --- Price Action (Steady, not explosive) ---
        'close': 152.5,
        'open': 151.0,
        'high': 153.0,
        'low': 150.5,
        'prev_close': 150.8,      # Small, healthy gap
        'volume': 35000000,       # Good volume, not crazy
        
        # --- Momentum (The Sweet Spot) ---
        'rsi': 58.0,              # Strong (50-60), NOT Overbought (>70)
        'roc_1': 0.012,           # Up 1.2% (Healthy)
        'roc_3': 0.035,           # Up 3.5% over 3 days
        
        # --- Trend ---
        'dist_sma20': 0.03,       # 3% above 20-day (Riding the trend)
        'dist_from_mean': 0.05,   # 5% above 50-day
        'trend_strength': 28.0,   # Solid trend
        
        # --- Volatility ---
        'natr': 1.8,              # Normal volatility
        'atr': 2.7,
        'gap': 0.002,             # Tiny gap (Safe)
        'close_loc': 0.75,        # Closed strong
        
        # --- Econ (Perfect) ---
        'FEDFUNDS': 5.33,
        'Rates_Delta': 0.0,
        'VIXCLS': 14.5,
        'VIX_change': -0.5,       # Fear gently falling
        'VIX_ROC': -0.05,
        'T10Y2Y': -0.3,
        'UNRATE': 3.7,
        'PAYEMS': 150000,
        
        'ticker': 'FAKE_BULL',
        'target': 0
    }

    test_row = pd.Series(fake_data)
    
    print("\nFeeding 'Steady Growth' scenario to Cortexa...")
    signal_output = engine.score(test_row)
    
    print("\n--- 🤖 Cortexa's Verdict ---")
    print(f"ML Probability:    {signal_output.proba_ml:.2%}")
    print(f"RAG Probability:   {signal_output.proba_rag:.2%}")
    print(f"Combined Score:    {signal_output.final_score:.2%}")
    print(f"Decision:          {'✅ BUY' if signal_output.decision == 1 else '❌ HOLD'}")

if __name__ == "__main__":
    test_bullish_scenario()