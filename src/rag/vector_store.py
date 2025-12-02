from typing import List, Dict, Any
import uuid
from qdrant_client import QdrantClient, models

class QdrantVecDB:
    def __init__(self, collection_name="cortexa_rag_store", vector_size=384): 
        print(f"--- Initializing Qdrant Vector Store (Size: {vector_size}) ---")
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        try:
            # Increased timeout to prevent timeouts during heavy loads
            self.client = QdrantClient(host="localhost", port=6333, timeout=60)
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise e

    def recreate_collection(self):
        """Forces a clean slate. Deletes old random data, creates new schema."""
        print(f"♻️ Recreating collection '{self.collection_name}' for {self.vector_size}-dim vectors...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            )
        )
        # Re-create indexes for filtering
        self.client.create_payload_index(self.collection_name, "regime", "keyword")
        self.client.create_payload_index(self.collection_name, "ticker", "keyword")
        print("✅ Collection ready.")

    def upsert(self, texts: List[str], vectors, metadatas: List[Dict[str, Any]]):
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

    def search(self, query_vec, top_k=10, filters: Dict[str, Any] = None):
        """
        Standard search using the official Qdrant client API.
        """
        qdrant_filter = None
        if filters:
            conditions = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v)) 
                for k, v in filters.items()
            ]
            qdrant_filter = models.Filter(must=conditions)

        # This method is standard in qdrant-client >= 1.0.0
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec.tolist(),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True 
        )
        
        results = []
        for hit in hits:
            results.append({
                "id": hit.id,
                "text": hit.payload.get("text"),
                "meta": hit.payload, 
                "score": float(hit.score)
            })
        return results