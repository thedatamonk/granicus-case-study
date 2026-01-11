from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_name: str = "chatbot"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    api_prefix: str = ""
    allowed_hosts: list[str] = ["*"]

    # Embedder service configuration
    embedder_service_url: str = "http://localhost:8001"
    embedder_timeout: int = 300

    # Vector database configuration
    weaviate_url: str = "http://localhost:8080"
    weaviate_collection: str = "govdocs2"

    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_timeout: int = 30

    # Retrieval module configuration
    max_sources: int = 5
    similarity_threshold: float = 0.7

    model_config = SettingsConfigDict(env_file="/Users/rohil/rohil-workspace/career/interviews/granicus/src/apis/chatbot/src/chatbot/.env", env_file_encoding="utf-8")
    

@lru_cache
def get_settings() -> BaseSettings:
    return Settings()