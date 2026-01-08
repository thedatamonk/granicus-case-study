import traceback

from embedder.handlers import create_start_app_handler
from embedder.routes.v1 import router as v1_router
from embedder.serialisation import HeartbeatResult
from embedder.settings import get_settings
from fastapi import FastAPI
from loguru import logger
from pydantic_settings import BaseSettings
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request


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

    app.add_event_handler(
        "startup",
        create_start_app_handler(app, settings),
    )
    
    app.include_router(v1_router, prefix=settings.api_prefix)
    logger.info("Application initialisation completed...")
    
    return app


settings = get_settings()

app = create_app(settings)

@app.get("/health", response_model=HeartbeatResult)
async def health_check(request: Request):
    try:
        model = request.app.state.model # load the embedding model to see if we are able to encode
        model.encode("")
        return HeartbeatResult(healthy=True)
    except Exception as e:
        logger.error(
            f"Error during health check | Error: {str(e)} | Traceback: {traceback.format_exc()}",
        )
        return HeartbeatResult(healthy=False)
