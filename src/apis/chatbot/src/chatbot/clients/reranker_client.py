from chatbot.settings import get_settings
from loguru import logger
from transformers import AutoModel
from typing import Dict, Any, List

settings = get_settings()


class RerankerClient:
    def __init__(self) -> None:
        logger.info(f"Loading reranker model: {settings.reranker_model_name}")
        self.client = AutoModel.from_pretrained(
            settings.reranker_model_name,
            dtype="auto",
            trust_remote_code=True,
        )
        self.client.eval()
        logger.info("Loaded successfully!")
    
    def rerank(self, query: str, docs: List[Dict[str, Any]]):
        reranked_results = self.client.rerank(query, list(map(lambda x: x["chunk_text"], docs)), top_n=settings.reranked_articles_max_count)
        final_results = []
        for result in reranked_results:
            doc = docs[result["index"]]     # get all the original doc fields
            doc["rerank_relevance_score"] = result["relevance_score"]    # add rerank relevance score
            final_results.append(doc)
        
        return final_results


# Singleton instance
_reranker_client = None

def get_reranker_client() -> RerankerClient:
    global _reranker_client
    if _reranker_client is None:
        _reranker_client = RerankerClient()
    return _reranker_client