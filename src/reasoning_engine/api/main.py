# src/reasoning_engine/api/main.py (FINAL, NO REDIS)

from fastapi import FastAPI
import logging
import winloop
import asyncio

from .routes import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Install winloop for performance, but we no longer need the complex lifespan manager.
winloop.install()

app = FastAPI(
    title="HackRx Final Gemini Reasoning Engine",
    description="A high-performance, multi-format RAG API.",
    version="3.0.0", # Final version
)

app.include_router(api_router)

@app.get("/", tags=["Health"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the Final Reasoning Engine API!"}