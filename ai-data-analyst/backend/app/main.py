from __future__ import annotations
import logging
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import routes_chat, routes_upload
from app.config import get_settings
from app.utils.errors import AppError
from app.utils.logging import configure_logging, new_request_id
settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("app")
app = FastAPI(
    title="AI Data Analyst API",
    description="Upload CSVs and interact with them via natural language.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.middleware("http")
async def request_logging(request: Request, call_next):
    new_request_id()
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    logger.warning("AppError on %s: %s", request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message, "details": exc.details})
@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error.", "details": {}})
@app.get("/api/health")
async def health():
    configured = bool(settings.anthropic_api_key) if settings.llm_provider == "anthropic" else bool(settings.groq_api_key)
    model = settings.llm_model if settings.llm_provider == "anthropic" else settings.groq_model
    return {"status": "ok", "provider": settings.llm_provider, "model": model, "llm_configured": configured}
app.include_router(routes_upload.router)
app.include_router(routes_chat.router)
