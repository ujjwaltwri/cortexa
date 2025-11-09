# Abstraction so you can swap OpenAI/Cohere/HF later
from typing import List
import numpy as np

class Embedder:
    def __init__(self, backend="openai", model="text-embedding-3-large"):
        self.backend, self.model = backend, model
        print(f"--- Initializing Embedder (Backend: {self.backend}, Model: {self.model}) ---")

    def encode(self, texts: List[str]) -> np.ndarray:
        # TODO: wire real APIs; for now use a local stub to unblock pipelines
        print(f"Stub encoding {len(texts)} texts to {1536} dims...")
        return np.random.RandomState(42).rand(len(texts), 1536).astype("float32")