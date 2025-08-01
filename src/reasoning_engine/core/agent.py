# src/reasoning_engine/core/agent.py (THE FINAL, RESILIENT, AND CORRECTED VERSION)

import google.generativeai as genai
from typing import List, Dict, Any, Callable, Coroutine
import logging
import json
import asyncio
import hashlib
import random

from ..config import get_settings
from .models import DocumentChunk, Answer
from .retrievel import HybridRetriever  # Corrected import name from retrievel to retrieval

logger = logging.getLogger(__name__)

try:
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
except Exception as e:
    logger.error(f"Failed to configure Gemini: {e}")


# CRITICAL FIX: Add the resilience decorator back in to handle rate limits.
def resilient_api_call(max_retries=5, initial_delay=1.0, backoff_factor=2.0, jitter=0.5):
    """
    A decorator that makes an async function resilient to temporary API errors,
    especially the '429 ResourceExhausted' error, using exponential backoff.
    """

    def decorator(func: Callable[..., Coroutine]):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) and "ResourceExhausted" in str(e):
                        if attempt == max_retries - 1:
                            logger.error(f"API call failed after {max_retries} retries. Final error: {e}")
                            raise e

                        wait_time = delay + random.uniform(0, jitter)
                        logger.warning(
                            f"Rate limit hit. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        delay *= backoff_factor
                    else:
                        logger.error(f"Non-retryable API error: {e}")
                        raise e
            # This line should ideally not be reached
            raise Exception("Retry logic failed unexpectedly.")

        return wrapper

    return decorator


class ReasoningAgent:
    def __init__(self, retriever: HybridRetriever, redis_client):
        self.retriever = retriever
        self.redis = redis_client
        # CRITICAL FIX: Use the powerful 'pro' model for the main reasoning task.
        # Removed the unused 'fast_model' and the 'tools' parameter.
        self.reasoning_model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def _get_cache_key(self, content: str) -> str:
        """Creates a unique cache key for a given piece of content."""
        return f"gemini-cache:{hashlib.md5(content.encode()).hexdigest()}"

    def _deterministic_query_expansion(self, query: str) -> List[str]:
        """Generates query variants without an LLM call for token and latency savings."""
        variants = {query}
        keywords = [word.lower() for word in query.split() if
                    len(word) > 3 and word.lower() not in ['what', 'is', 'are', 'the', 'for', 'a', 'an']]
        if keywords:
            variants.add(' '.join(keywords))
        if 'define' in query.lower() or 'what is' in query.lower():
            variants.add(f"definition of {' '.join(keywords)}")
        return list(variants)

    # Apply the resilience decorator to the actual API call
    @resilient_api_call()
    async def _generate_with_resilience(self, prompt: str) -> str:
        """A wrapper for the Gemini API call that is now decorated for resilience."""
        response = await self.reasoning_model.generate_content_async(prompt)
        # Add robust error handling for the response object
        try:
            return response.text.strip()
        except (AttributeError, ValueError):
            logger.warning("Gemini response did not contain valid text. Falling back.")
            return "The information could not be found in the provided documents."

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Lean & Resilient Analyst process for question: '{query}' ---")

        final_answer_key = self._get_cache_key(f"final_answer:{query}")
        cached_answer = await self.redis.get(final_answer_key)
        if cached_answer:
            logger.info("CACHE HIT: Returning final answer from cache.")
            return Answer(query=query, decision="Information Found (Cached)", payout=None, justification=[],
                          simple_answer=cached_answer)

        # Phase 1: Broad Retrieval
        search_queries = self._deterministic_query_expansion(query)
        all_retrieved_chunks = [chunk for sq in search_queries for chunk in self.retriever.retrieve(sq, top_k=7)]
        unique_chunks = list({chunk.chunk_id: chunk for chunk in all_retrieved_chunks}.values())

        # Phase 2: One-Shot Distillation and Synthesis
        context_str = "\n---\n".join([f"Evidence Chunk:\n{chunk.text}" for chunk in unique_chunks])

        synthesis_prompt = f"""
        You are an expert insurance analyst. Your task is to provide a direct, concise, and accurate answer to the user's question based ONLY on the provided evidence chunks.

        **User Question:**
        {query}

        **Evidence Chunks (separated by '---'):**
        <EVIDENCE>
        {context_str}
        </EVIDENCE>

        **Instructions:**
        1.  Carefully review all evidence chunks to find the most relevant piece of information that directly answers the user's question.
        2.  Synthesize this information into a single, clear, and professional paragraph.
        3.  If the evidence contains specific numbers, dates, or conditions, you MUST include them.
        4.  If the evidence does not contain an answer, you MUST state: "The information could not be found in the provided documents."
        5.  Your answer must be definitive. Do not apologize. Do not ask for more information. State the fact or state it is not found.
        """

        # Call the new resilient wrapper function
        final_answer_text = await self._generate_with_resilience(synthesis_prompt)

        await self.redis.set(final_answer_key, final_answer_text, ex=3600)

        logger.info(f"SYNTHESIZER: Generated final answer: {final_answer_text}")
        return Answer(query=query, decision="Information Found", payout=None, justification=[],
                      simple_answer=final_answer_text)

    async def answer_all_questions(self, questions: List[str]) -> List[Answer]:
        """Processes a list of questions sequentially."""
        answers = []
        for query in questions:
            answer = await self.answer_question(query)
            answers.append(answer)
        return answers