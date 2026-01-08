from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_name: str = "docparser"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    api_prefix: str = ""
    allowed_hosts: list[str] = ["*"]
    processed_docs_dir: str = ".processed_data"
    embedder_service_url: str = "http://localhost:8001"
    embedder_timeout: int = 300
    weaviate_url: str = "http://localhost:8080"
    weaviate_collection: str = "govdocs2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


    

@lru_cache
def get_settings() -> BaseSettings:
    return Settings()