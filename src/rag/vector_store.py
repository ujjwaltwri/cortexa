from typing import List, Dict, Any
import uuid
import numpy as np
from qdrant_client import QdrantClient, models

# Your original InMemoryVecDB (for reference)
class InMemoryVecDB:
    pass

# --- The New Production-Ready Vector Store ---

class QdrantVecDB:
    """
    A drop-in replacement for InMemoryVecDB using a persistent Qdrant backend.
    
    This class matches the `upsert` and `search` methods expected by
    your RAGSignalEngine and StrategyMemory.
    """
    def __init__(self, collection_name="cortexa_rag_store", vector_size=1536):
        print("--- Initializing Qdrant Vector Store ---")
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        try:
            # Connect to the Docker container
            self.client = QdrantClient(host="localhost", port=6333)
        except Exception as e:
            print(f"Error: Could not connect to Qdrant at localhost:6333.")
            print("Please make sure your Qdrant Docker container is running.")
            print("Run: docker-compose up -d")
            raise e
        
        # Ensure the collection exists
        self.setup_collection()

    def setup_collection(self):
        """Creates the collection in Qdrant *if it doesn't exist*."""
        
        # --- THIS IS THE FIX ---
        # We now check if the collection exists instead of recreating it.
        try:
            collection_exists = self.client.collection_exists(self.collection_name)
        except Exception as e:
            print(f"Error checking Qdrant connection: {e}")
            raise e

        if collection_exists:
            print(f"Collection '{self.collection_name}' already exists. Skipping creation.")
            return
        # --- END OF FIX ---

        # If it doesn't exist, create it
        try:
            print(f"Collection '{self.collection_name}' not found. Creating...")
            self.client.create_collection(  # Changed from recreate_collection
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully.")
            
            # Create payload indexes for filtering (as requested in your plan)
            self.client.create_payload_index(self.collection_name, "regime", "keyword", wait=True)
            self.client.create_payload_index(self.collection_name, "ticker", "keyword", wait=True)
            self.client.create_payload_index(self.collection_name, "sector", "keyword", wait=True)
            self.client.create_payload_index(self.collection_name, "event_type", "keyword", wait=True)
            print("Payload indexes for filtering created (regime, ticker, sector, event_type).")

        except Exception as e:
            print(f"Error creating Qdrant collection: {e}")


    def upsert(self, texts: List[str], vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        Upserts documents, vectors, and metadata into the Qdrant collection.
        """
        points = []
        for text, vec, meta in zip(texts, vectors, metadatas):
            payload = {"text": text, **meta}
            
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=vec.tolist(),
                    payload=payload
                )
            )
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=False 
            )

    def search(self, query_vec: np.ndarray, top_k=10, filters: Dict[str, Any] = None):
        """
        Searches the vector store with an optional metadata filter.
        """
        
        # 1. Translate the simple filter dict into a Qdrant Filter
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            qdrant_filter = models.Filter(must=must_conditions)

        # 2. Perform the search
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec.tolist(),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True 
        )
        
        # 3. Format results to match your RAGSignalEngine's expectation
        results = []
        for hit in hits:
            results.append({
                "id": hit.id,
                "text": hit.payload.get("text"),
                "meta": hit.payload, 
                "score": float(hit.score)
            })
            
        return results