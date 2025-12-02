from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"--- 🧠 Loading Real AI Model: {model_name} ---")
        try:
            # This loads a real AI model locally (Free & Fast)
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            # Fallback (should not happen if installed correctly)
            print("Warning: Using fallback random embeddings!")
            return np.random.rand(len(texts), 384).astype("float32")
        
        # Generate REAL semantic vectors (384 dimensions)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")