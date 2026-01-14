from chatbot.settings import get_settings
from loguru import logger
from transformers import AutoModel
from typing import Dict, Any, List
from abc import ABC, abstractmethod

settings = get_settings()

class BaseRerankerClient(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]]):
        pass

class JinaLargeRerankerClient(BaseRerankerClient):
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

class JinaSmallRerankerClient(BaseRerankerClient):
    def __init__(self) -> None:
        logger.info(f"Loading reranker model: {settings.reranker_model_name}")
        from sentence_transformers import CrossEncoder

        self.client = CrossEncoder(settings.reranker_model_name, trust_remote_code=True)
        logger.info("Loaded successfully!")

    def rerank(self, query: str, docs: List[Dict[str, Any]]):
        reranked_results = self.client.rank(query, 
                                            documents=list(map(lambda x: x["chunk_text"], docs)),
                                            return_documents=False,
                                            top_k=settings.reranked_articles_max_count
                                            )
        final_results = []
        for result in reranked_results:
            doc = docs[result["corpus_id"]]     # get all the original doc fields
            doc["rerank_relevance_score"] = result["score"]    # add rerank relevance score
            final_results.append(doc)
        
        return final_results

# Singleton instance
_reranker_client = None

def get_reranker_client() -> BaseRerankerClient:
    global _reranker_client
    if _reranker_client is None:
        if settings.reranker_model_name == "jinaai/jina-reranker-v1-tiny-en":
            _reranker_client = JinaSmallRerankerClient()
        else:
            _reranker_client = JinaLargeRerankerClient()
    return _reranker_client