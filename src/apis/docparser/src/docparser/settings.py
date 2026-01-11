from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    project_name: str = "docparser"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    api_prefix: str = ""
    allowed_hosts: list[str] = ["*"]
    processed_docs_dir: str = ".processed_data"

    # Embedder service configuration
    embedder_service_url: str = "http://localhost:8001"
    embedder_timeout: int = 300

    # Vector database configuration
    weaviate_url: str = "http://localhost:8080"
    weaviate_collection: str = "default_wv_collection"
    
    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_temperature: float = 0.0        # temperature set to 0 because we want deterministic conversion of csv data to JSON string
    llm_timeout: int = 30

    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", env_file_encoding="utf-8")

@lru_cache
def get_settings() -> BaseSettings:
    return Settings()