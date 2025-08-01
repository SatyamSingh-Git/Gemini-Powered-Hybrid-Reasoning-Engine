# src/reasoning_engine/api/routes.py (THE FINAL, BUG-FREE VERSION)

import asyncio
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Security, Request

# THE FIX IS HERE: We are re-adding the missing import for APIKeyHeader
from fastapi.security import APIKeyHeader

from ..core.models import APIRequest, APIResponse, Answer
from ..core.ingestion import process_documents
from ..core.retrievel import HybridRetriever
from ..core.agent import ReasoningAgent

logger = logging.getLogger(__name__)
router = APIRouter()
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)  # This line will now work
EXPECTED_BEARER_TOKEN = "Bearer bc916bc507a9b3b680e613c91243b99771a30be1587ca8d9eb8cc4b9dfab5a55"


async def verify_token(authorization: str = Security(api_key_header)):
    if authorization != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token")


@router.post(
    "/hackrx/run",
    response_model=APIResponse,
    tags=["Reasoning Engine"],
    dependencies=[Depends(verify_token)]
)
async def run_reasoning_engine(request_data: APIRequest) -> APIResponse:
    request_start_time = time.time()
    logger.info("================== NEW REQUEST RECEIVED ==================")
    logger.info(f"Received {len(request_data.documents)} docs and {len(request_data.questions)} questions.")
    try:
        all_chunks = await process_documents([str(url) for url in request_data.documents])
        if not all_chunks:
            raise HTTPException(status_code=400, detail="Failed to process any of the provided documents.")

        retriever = await HybridRetriever.create(chunks=all_chunks)

        # We are using the agent without Redis caching for stability
        agent = ReasoningAgent(retriever=retriever)

        structured_answers: list[Answer] = await agent.answer_all_questions(request_data.questions)

        simple_answers = [ans.simple_answer for ans in structured_answers]

        total_time = time.time() - request_start_time
        logger.info(f"================== REQUEST COMPLETE (Total Time: {total_time:.2f}s) ==================")
        return APIResponse(answers=simple_answers)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")