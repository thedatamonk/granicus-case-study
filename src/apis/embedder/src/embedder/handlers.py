from loguru import logger
from fastapi import FastAPI
from pydantic_settings import BaseSettings
from typing import Callable
from sentence_transformers import SentenceTransformer

def create_start_app_handler(
    app: FastAPI,
    settings: BaseSettings,
) -> Callable:
    async def start_app() -> None:
        logger.info("Starting application!")

        logger.info("Loading model.")
        model_instance = SentenceTransformer(settings.embed_model_name)
        app.state.model = model_instance

    return start_app