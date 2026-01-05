from loguru import logger
from docparser.settings import get_settings
from pydantic_settings import BaseSettings
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from docparser.routes.v1 import router as v1_router

def create_app(settings: BaseSettings) -> FastAPI:
    logger.info(f"Initialising application {settings.project_name}...")

    app = FastAPI(
        title=settings.project_name,
        docs_url=settings.docs_url,
        openapi_url=settings.openapi_url,
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


