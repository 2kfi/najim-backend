"""
Najim Backend - Multi-Tenant Distributed Voice Assistant
"""
import asyncio
import logging
import json
import os
import time
import uuid

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.config import get_settings, Settings
from core.redis_manager import RedisManager, get_redis
from core.app_state import get_app_state, AppState
from core.jwt_auth import get_jwt_manager, verify_jwt
from api.websocket import router as ws_router, _start_ws_listener, _active_connections
from api.sessions import router as sessions_router
from api.health import router as health_router
from pipeline.orchestrator import WorkerManager

logger = logging.getLogger("najim")

limiter = Limiter(key_func=get_remote_address)

_uptime_start = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"Starting Najim cluster node: {settings.cluster.node_id}")

    await AppState.initialize()
    redis = await get_redis()

    if not await redis.ping():
        raise RuntimeError("Redis connection failed")
    else:
        logger.info("Redis connected")

    worker_mgr = WorkerManager(redis)
    await worker_mgr.start_all()

    asyncio.create_task(_start_ws_listener(settings.cluster.node_id, redis))

    logger.info("Application initialized successfully")
    yield

    logger.info("Shutting down...")
    await worker_mgr.stop_all()
    await AppState.shutdown()
    await redis.close()
    logger.info("Shutdown complete")


settings = get_settings()

app = FastAPI(
    title="Najim Backend",
    version="3.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "error_code": "RATE_LIMITED"}
    )


@app.middleware("http")
async def api_key_fallback(request: Request, call_next):
    if request.url.path in ["/health", "/ready", "/live", "/metrics", "/openapi.json", "/docs", "/redoc"]:
        return await call_next(request)
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    settings = get_settings()

    if auth.startswith("Bearer "):
        token = auth.replace("Bearer ", "").strip()
        try:
            jwt_mgr = get_jwt_manager()
            jwt_mgr.verify_token(token)
            return await call_next(request)
        except Exception:
            if settings.auth.jwt_only:
                return JSONResponse({"error": "Invalid JWT token"}, status_code=401)

    if not settings.auth.jwt_only and settings.auth.api_keys:
        if auth.startswith("Bearer "):
            key = auth.replace("Bearer ", "").strip()
            if key in settings.auth.api_keys:
                return await call_next(request)
        return JSONResponse({"error": "Invalid API key"}, status_code=401)

    if settings.auth.jwt_only:
        return JSONResponse({"error": "Missing or invalid Authorization header"}, status_code=401)

    return await call_next(request)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(ws_router)
app.include_router(sessions_router)
app.include_router(health_router)


@app.get("/")
async def root():
    return {
        "service": "najim-backend",
        "version": "3.0.0",
        "node_id": settings.cluster.node_id,
        "status": "running",
        "uptime_seconds": time.time() - _uptime_start,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )