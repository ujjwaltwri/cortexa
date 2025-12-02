import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import yaml
from pathlib import Path

def make_regime_features(prices: pd.Series) -> pd.DataFrame:
    """
    Creates the feature set for the HMM (returns, volatility, momentum).
    """
    r = prices.pct_change(fill_method=None).dropna()
    vol = r.rolling(20).std()
    mom = prices.pct_change(20, fill_method=None)
    
    X = pd.concat([r, vol, mom], axis=1).dropna()
    X.columns = ["return", "vol", "momentum"]
    return X

def fit_hmm(X: pd.DataFrame, n_states=3, random_state=42) -> GaussianHMM:
    """
    Fits a Gaussian Hidden Markov Model.
    """
    print(f"--- Fitting HMM with {n_states} states ---")
    hmm = GaussianHMM(
        n_components=n_states, 
        covariance_type="full", 
        n_iter=100, 
        random_state=random_state
    )
    hmm.fit(X.values)
    return hmm

def predict_regime(hmm: GaussianHMM, X: pd.DataFrame) -> pd.Series:
    """
    Predicts the hidden regime for each data point.
    """
    states = hmm.predict(X.values)
    return pd.Series(states, index=X.index, name="regime")
