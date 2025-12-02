# src/rag/engine.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.rag.embedder_local import LocalEmbedder
from src.rag.store_faiss import FaissStore


def build_index_from_csv(
    csv_path: str | Path,
    text_col: str = "text",
    meta_cols: list[str] = ("ticker", "regime", "target", "date", "event_type"),
    out_index: str | Path = "artifacts/cortexa.faiss",
    out_meta: str | Path = "artifacts/cortexa.meta.json",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Build a local FAISS index from a CSV that has a 'text' column and metadata cols.
    """
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"'{text_col}' column missing in {csv_path}")

    embedder = LocalEmbedder(embed_model)
    store = FaissStore(dim=embedder.dim, path=Path(out_index), meta_path=Path(out_meta))

    texts = df[text_col].astype(str).tolist()
    vecs = embedder.encode(texts)

    metas = []
    for _, row in df.iterrows():
        m = {"text": row[text_col]}
        for c in meta_cols:
            if c in df.columns:
                m[c] = row[c]
        metas.append(m)

    store.add(vecs, metas)
    store.save()
    return out_index, out_meta
