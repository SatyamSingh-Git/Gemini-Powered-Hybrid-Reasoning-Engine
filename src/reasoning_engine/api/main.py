# src/reasoning_engine/api/main.py (SIMPLIFIED AND CORRECTED)

from fastapi import FastAPI
import logging

from .routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="HackRx Gemini Reasoning Engine API",
    description="An API for querying and making decisions on policy documents.",
    version="1.0.0",
)

# Include the API router that defines the /hackrx/run endpoint
app.include_router(api_router)

@app.get("/", tags=["Health"])
async def read_root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "Welcome to the Reasoning Engine API!"}