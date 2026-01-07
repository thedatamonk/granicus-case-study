from chatbot.settings import get_settings
from chatbot.clients.embedding_client import get_embedder_client
from chatbot.clients.weaviate_client import get_weaviate_client
from typing import Dict, List, Any
import time
from loguru import logger

settings = get_settings()


class Retriever:
    def __init__(self) -> None:
        self.embedder = get_embedder_client()
        self.vecdb = get_weaviate_client()

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        max_sources = settings.max_sources
        threshold = settings.similarity_threshold

        time_stats = {}

        # Step 1: Embed query
        embed_start = time.time()
        try:
            query_embedding = self.embedder.embed_query(query)
            time_stats["embedding_ms"] = int((time.time() - embed_start) * 1000)
            logger.debug(f"Query embedded in {time_stats['embedding_ms']}ms")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

        # Step 2: Retrieve docs based on query embedding
        retrieval_start = time.time()
        try:
            results = self.vecdb.search(
                query_vector=query_embedding,
                limit=max_sources,
                distance_threshold=threshold
            )
            time_stats["retrieval_ms"] = int((time.time() - retrieval_start) * 1000)
            logger.debug(f"Retrieved {len(results)} sources in {time_stats['retrieval_ms']}ms")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

        # Step 3: Format sources with IDs for citation
        sources = []
        for idx, result in enumerate(results, start=1):
            sources.append({
                "source_id": idx,
                "chunk_text": result["chunk_text"],
                "filename": result["filename"],
                "chunk_index": result["chunk_index"],
                "doc_type": result["doc_type"],
                "relevance_score": result["distance"],
                "cited": False  # Will be updated after LLM responds
            })

        return sources
    
_retriever = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever