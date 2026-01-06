from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_name: str = "embedder"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    api_prefix: str = ""
    allowed_hosts: list[str] = ["*"]
    embed_dim: int = 384
    embed_model_name: str = "all-MiniLM-L6-v2"


@lru_cache
def get_settings() -> BaseSettings:
    return Settings()