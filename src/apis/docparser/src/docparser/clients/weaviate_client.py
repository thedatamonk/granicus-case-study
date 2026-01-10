import weaviate
from docparser.settings import get_settings
from loguru import logger
from weaviate.classes.config import Configure, Property, DataType
from typing import List, Dict, Any

settings = get_settings()

class WeaviateClient:
    def __init__(self) -> None:
        weaviate_url = settings.weaviate_url.replace("http://", "").replace("https://", "")
        host, port = weaviate_url.split(":")
        self.client = weaviate.connect_to_local(host=host, port=port)
        self._initialise_schema()

    def _initialise_schema(self):
        """Initialise weaviate collection schema"""
        collection_name = settings.weaviate_collection

        try:
            # Check if collection exists
            if self.client.collections.exists(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                return
            # Create collection with schema
            logger.info(f"Creating collection '{collection_name}'")
            self.client.collections.create(
                name=collection_name,
                vector_config=Configure.Vectors.self_provided(), # we compute our own vectors using embedder service
                properties=[
                    Property(name="chunk_text", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="job_id", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                ]
            )
            logger.info(f"Collection '{collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def insert_chunks(self, chunks_data: List[Dict[str, Any]]):
        """Insert chunks data into weaviate collection in batch mode"""
        try:
            collection = self.client.collections.get(name=settings.weaviate_collection)

            with collection.batch.dynamic() as batch:
                for chunk_data in chunks_data:
                    batch.add_object(
                        properties=chunk_data["properties"],
                        vector=chunk_data["vector"]
                    )
            logger.info(f"Inserted {len(chunks_data)} into Weaviate.")
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise

    def close(self):
        """Close Weaviate connection."""
        self.client.close()

# Singleton instance
_weaviate_client = None

def get_weaviate_client() -> WeaviateClient:
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client = WeaviateClient()
    return _weaviate_client
