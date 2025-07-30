# src/reasoning_engine/api/main.py (UPGRADED FOR WINDOWS WITH WINLOOP)

from fastapi import FastAPI
import logging
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
import winloop  # Import winloop instead of uvloop
import asyncio

from .routes import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    # Startup: Create a Redis connection pool.
    logging.info("Application starting up...")
    app.state.redis = aioredis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)

    # Install winloop as the asyncio event loop policy.
    winloop.install()

    logging.info("Redis connection established and winloop is active.")
    yield
    # Shutdown: Close the Redis connection pool.
    logging.info("Application shutting down...")
    await app.state.redis.close()
    logging.info("Redis connection closed.")


app = FastAPI(
    title="HackRx Optimized Gemini Reasoning Engine",
    description="A high-performance RAG API with caching and optimized processing for Windows.",
    version="2.0.1",  # Version bump for the fix
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/", tags=["Health"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the Optimized Reasoning Engine API!"}