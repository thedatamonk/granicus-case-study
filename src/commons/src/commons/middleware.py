import asyncio
import time

from fastapi import FastAPI, Request, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from commons.metrics import ServiceMetrics


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics"""
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        metrics = request.app.state.metrics
        
        metrics.track_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        )
        metrics.track_latency(
            method=request.method,
            endpoint=request.url.path,
            duration=duration
        )
        
        return response


async def initialise_metrics(app: FastAPI, service_name: str):
    logger.info("Starting metrics updater...")
    app.state.metrics = ServiceMetrics(service_name=service_name)

    # Start background task to update the metrics continuously
    async def update_system_metrics():
        while True:
            app.state.metrics.update_system_metrics()
            await asyncio.sleep(15)
    asyncio.create_task(update_system_metrics())
    
def add_prometheus_to_app(app: FastAPI):

    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics")
    async def get_metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
