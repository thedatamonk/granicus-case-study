import traceback
from contextlib import asynccontextmanager

from docparser.clients.weaviate_client import get_weaviate_client
from docparser.routes.v1 import router as v1_router
from docparser.serialisation import HeartbeatResult
from docparser.settings import get_settings
from fastapi import FastAPI
from loguru import logger
from pydantic_settings import BaseSettings
from starlette.middleware.cors import CORSMiddleware
from commons.middleware import add_prometheus_to_app, initialise_metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialise_metrics(app, settings.project_name)
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

    # Add prometheus middleware /metrics endpoint to FastAPI app
    add_prometheus_to_app(app)

    app.include_router(v1_router, prefix=settings.api_prefix)
    logger.info("Application initialisation completed...")
    
    return app


settings = get_settings()

app = create_app(settings)

@app.get("/health", response_model=HeartbeatResult)
async def health_check():
    try:
        _ = get_weaviate_client()
        return HeartbeatResult(healthy=True)
    except Exception as e:
        logger.error(
            f"Error during health check | Error: {str(e)} | Traceback: {traceback.format_exc()}",
        )
        return HeartbeatResult(healthy=False)

