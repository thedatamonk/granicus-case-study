from typing import List, Union
from docparser.settings import get_settings
import httpx
from loguru import logger


settings = get_settings()

class EmbedderClient:
    def __init__(self) -> None:
        self.base_url = settings.embedder_service_url
        self.timeout = settings.embedder_timeout
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings"""
        try:

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/v1/embedding",
                    json={"content": texts}
                )

                response.raise_for_status()
                return response.json()["embedding"]
        except httpx.TimeoutException as e:
            logger.error(f"Embedder service timeout: {e}")
            raise Exception(f"Embedder service timeout after {self.timeout}s")
        except httpx.HTTPError as e:
            logger.error(f"Embedder service error: {e}")
            raise Exception(f"Embedder service failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error calling embedder: {e}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")


_embedder_client = None

def get_embedder_client() -> EmbedderClient:
    global _embedder_client
    if _embedder_client is None:
        _embedder_client = EmbedderClient()
    return _embedder_client