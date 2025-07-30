import asyncio
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader

from ..core.models import APIRequest, APIResponse, Answer
from ..core.ingestion import process_documents
from ..core.retrievel import HybridRetriever
from ..core.agent import ReasoningAgent

logger = logging.getLogger(__name__)
router = APIRouter()
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
EXPECTED_BEARER_TOKEN = "Bearer bc916bc507a9b3b680e613c91243b99771a30be1587ca8d9eb8cc4b9dfab5a55"


async def verify_token(authorization: str = Security(api_key_header)):
    """Dependency to verify the bearer token provided in the request header."""
    if authorization != EXPECTED_BEARER_TOKEN:
        logger.warning(f"Invalid authentication token received.")
        raise HTTPException(
            status_code=403,
            detail="Invalid authentication token",
        )


@router.post(
    "/hackrx/run",
    response_model=APIResponse,
    tags=["Reasoning Engine"],
    dependencies=[Depends(verify_token)]
)
async def run_reasoning_engine(request_data: APIRequest, request: Request) -> APIResponse:
    """
    The main endpoint for the optimized reasoning engine. It orchestrates the
    full workflow from document ingestion to cached, AI-powered answers.
    """
    request_start_time = time.time()
    logger.info("================== NEW REQUEST RECEIVED ==================")
    logger.info(f"Received {len(request_data.documents)} docs and {len(request_data.questions)} questions.")

    try:
        # Step 1: Ingestion
        ingestion_start_time = time.time()
        all_chunks = await process_documents([str(url) for url in request_data.documents])
        logger.info(
            f"Document ingestion complete. Found {len(all_chunks)} chunks. (Took {time.time() - ingestion_start_time:.2f}s)")
        if not all_chunks:
            raise HTTPException(status_code=400, detail="Failed to process any of the provided documents.")

        # Step 2: Indexing and Retriever Setup
        retriever_start_time = time.time()
        retriever = await HybridRetriever.create(chunks=all_chunks)
        logger.info(f"Hybrid Retriever initialized successfully. (Took {time.time() - retriever_start_time:.2f}s)")

        # Step 3: Agent Initialization with Redis Client
        # Retrieve the Redis client from the application state (initialized in main.py)
        redis_client = request.app.state.redis
        agent = ReasoningAgent(retriever=retriever, redis_client=redis_client)

        # Step 4: Multi-Step Reasoning
        structured_answers: list[Answer] = await agent.answer_all_questions(request_data.questions)

        # Step 5: Formatting the final response
        simple_answers = [ans.simple_answer for ans in structured_answers]

        total_time = time.time() - request_start_time
        logger.info(f"================== REQUEST COMPLETE (Total Time: {total_time:.2f}s) ==================")
        return APIResponse(answers=simple_answers)

    except Exception as e:
        logger.error(f"An unexpected error occurred during the request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")