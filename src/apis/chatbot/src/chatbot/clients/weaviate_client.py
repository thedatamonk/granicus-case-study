import weaviate
from chatbot.settings import get_settings
from loguru import logger
from typing import List, Dict, Any
import json

settings = get_settings()

class WeaviateRetrieverClient:
    def __init__(self) -> None:
        weaviate_url = settings.weaviate_url.replace("http://", "").replace("https://", "")
        host, port = weaviate_url.split(":")
        self.client = weaviate.connect_to_local(host=host, port=port)
        self.collection_name = settings.weaviate_collection

    def search(self, query_vector: List[float], limit: int = 5, distance_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic similarity search on vector db"""
        try:
            collection = self.client.collections.get(self.collection_name)

            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit * 2,    # We will retrieve double the requested limit because we will be applying filter on distance_threshold
                return_metadata=["distance"]
            )

            # Filter by distance threshold
            results = []
            for obj in response.objects:
                if obj.metadata.distance <= distance_threshold:
                    results.append({
                        "chunk_text": obj.properties.get("chunk_text"),
                        "source_id": obj.properties.get("source"),
                        "doc_type": obj.properties.get("doc_type"),
                        "metadata": json.loads(obj.properties.get("metadata", {})),
                        "distance": obj.metadata.distance
                    })

            # return top-{limit} after filtering
            results = results[:limit]

            logger.info(f"Retrieved {len(results)} chunks (threshold: {distance_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            raise Exception(f"Failed to retrieve documents: {str(e)}")

    def close(self):
        """Close Weaviate connection."""
        self.client.close()

# Singleton instance
_weaviate_client = None

def get_weaviate_client() -> WeaviateRetrieverClient:
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client = WeaviateRetrieverClient()
    return _weaviate_client
