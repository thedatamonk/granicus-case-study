from loguru import logger
from docparser.settings import get_settings
from pydantic_settings import BaseSettings
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from docparser.routes.v1 import router as v1_router
from contextlib import asynccontextmanager
from docparser.clients.weaviate_client import get_weaviate_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising weaviate client...")
    get_weaviate_client()
    logger.info("Weaviate client initialized.")

    yield

    logger.info("Shutting down...")
    weaviate_client = get_weaviate_client()
    weaviate_client.close()
    logger.info("Shutdown complete.")

def create_app(settings: BaseSettings) -> FastAPI:
    logger.info(f"Initialising application {settings.project_name}...")

    app = FastAPI(
        title=settings.project_name,
        docs_url=settings.docs_url,
        openapi_url=settings.openapi_url,
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(v1_router, prefix=settings.api_prefix)
    logger.info("Application initialisation completed...")
    
    return app


settings = get_settings()

app = create_app(settings)


