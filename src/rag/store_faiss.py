# src/rag/store_faiss.py
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    raise RuntimeError("faiss is required for local vector search. pip install faiss-cpu") from e


class FaissStore:
    """
    Simple IP (cosine if normalized) FAISS index + JSON meta.
    Each add() call appends vectors and corresponding meta dicts.
    """
    def __init__(self, dim: int, path: Path, meta_path: Path):
        self.dim = int(dim)
        self.path = Path(path)
        self.meta_path = Path(meta_path)
        self.meta = []
        if self.path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.path))
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def add(self, vecs: np.ndarray, metas: list[dict]):
        vecs = np.asarray(vecs, dtype="float32")
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Bad vecs shape {vecs.shape}, expected (*, {self.dim})")
        if len(metas) != vecs.shape[0]:
            raise ValueError("metas length must match vecs rows")
        self.index.add(vecs)
        self.meta.extend(metas)

    def save(self):
        faiss.write_index(self.index, str(self.path))
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f)

    def search(self, qvec: np.ndarray, k: int = 5):
        q = np.asarray(qvec, dtype="float32").reshape(1, -1)
        D, I = self.index.search(q, k)
        sims = D[0].tolist()
        idxs = I[0].tolist()
        out = []
        for s, i in zip(sims, idxs):
            if i < 0 or i >= len(self.meta):
                continue
            out.append((float(s), self.meta[i]))
        return out
