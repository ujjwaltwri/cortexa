import numpy as np
from typing import Dict, Any, List

# Import our existing modules
from src.rag.embeddings import Embedder
from src.rag.vector_store import QdrantVecDB

class StrategyMemory:
    """
    A class to handle recording and retrieving historical strategy
    contexts and outcomes from the vector store.
    """
    def __init__(self, vecdb: QdrantVecDB, embedder: Embedder):
        self.vecdb = vecdb
        self.embedder = embedder
        print("--- StrategyMemory Initialized ---")

    def record(self, context_text: str, meta: Dict[str, Any]):
        """
        Embeds a single context string and upserts it with its metadata.
        """
        # Embed the text
        vector = self.embedder.encode([context_text])
        
        # Add a new event_type to distinguish this from news
        meta['event_type'] = 'market_state'
        
        # Upsert into Qdrant
        self.vecdb.upsert([context_text], vector, [meta])

    def record_batch(self, context_texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Embeds a batch of contexts and upserts them efficiently.
        """
        print(f"Embedding batch of {len(context_texts)} market states...")
        # Embed all texts in one batch
        vectors = self.embedder.encode(context_texts)
        
        # Add event_type to all metadata
        for meta in metadatas:
            meta['event_type'] = 'market_state'
            
        print("Upserting batch to Qdrant...")
        # Upsert all points in one batch
        self.vecdb.upsert(context_texts, vectors, metadatas)