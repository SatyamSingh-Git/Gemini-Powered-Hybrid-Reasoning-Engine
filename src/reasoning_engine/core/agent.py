# src/reasoning_engine/core/agent.py (THE FINAL, BULLETPROOF VERSION)

import google.generativeai as genai
from typing import List, Dict, Any, Callable, Coroutine
import logging
import json
import asyncio
import random
import re

from ..config import get_settings
from .models import DocumentChunk, Answer
from .retrievel import HybridRetriever

logger = logging.getLogger(__name__)


# The resilient API call decorator is still very important!
def resilient_api_call(max_retries=5, initial_delay=1.0, backoff_factor=2.0, jitter=0.5):
    def decorator(func: Callable[..., Coroutine]):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e):
                        if attempt == max_retries - 1: raise e
                        wait_time = delay + random.uniform(0, jitter)
                        logger.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                        delay *= backoff_factor
                    else:
                        raise e
            raise Exception("Retry logic failed.")

        return wrapper

    return decorator


class ReasoningAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.reasoning_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.fast_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def _deterministic_query_expansion(self, query: str) -> List[str]:
        variants = {query};
        keywords = [word.lower() for word in query.split() if
                    len(word) > 3 and word.lower() not in ['what', 'is', 'are', 'the', 'for', 'a', 'an']];
        if keywords: variants.add(' '.join(keywords))
        if 'define' in query.lower() or 'what is' in query.lower(): variants.add(f"definition of {' '.join(keywords)}");
        return list(variants)

    @resilient_api_call()
    async def _distill_evidence_with_resilience(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Uses a fast LLM to find the single best chunk of evidence, now with
        robust validation to prevent crashes from model hallucinations.
        """
        if not chunks: return []

        context_str = "\n---\n".join([f"ID: {i}\n{chunk.text}" for i, chunk in enumerate(chunks)])

        # A clearer, more forceful prompt
        distillation_prompt = f"""
        You are a research assistant. Your task is to review the following text chunks, which are separated by '---'.
        Based on the user's question, identify the SINGLE chunk ID that contains the most direct and precise answer.
        The valid IDs are from 0 to {len(chunks) - 1}.
        Respond with ONLY the numeric ID of that single best chunk. Do not add any other text.

        User Question: "{query}"

        <EVIDENCE>
        {context_str}
        </EVIDENCE>
        """

        response = await self.fast_model.generate_content_async(distillation_prompt)

        # THE FIX IS HERE: We now validate the AI's output before using it.
        try:
            # Find the first number in the response and convert it to an integer
            best_chunk_id_str = re.search(r'\d+', response.text.strip()).group()
            best_chunk_id = int(best_chunk_id_str)

            # Check if the ID is a valid index for our list
            if 0 <= best_chunk_id < len(chunks):
                logger.info(f"DISTILLER: Identified valid best evidence chunk ID: {best_chunk_id}")
                return [chunks[best_chunk_id]]
            else:
                # The ID was a number, but it was out of bounds
                logger.warning(
                    f"DISTILLER: Model returned out-of-bounds ID: {best_chunk_id}. Falling back to the first chunk.")
                return [chunks[0]]  # A safe fallback
        except (ValueError, AttributeError, IndexError) as e:
            # The model's response was not a number, or something else went wrong
            logger.error(
                f"DISTILLER: Could not parse a valid ID from model response '{response.text}'. Falling back to all chunks. Error: {e}")
            return chunks  # The safest fallback is to use all chunks

    @resilient_api_call()
    async def _synthesize_with_resilience(self, query: str, evidence_text: str) -> str:
        prompt = f"""
        You are an expert analyst. Synthesize the provided evidence into a concise, direct, and professional answer to the user's question.

        **User Question:** "{query}"

        **Evidence:**
        {evidence_text}

        **Instructions:**
        - State the single most important fact from the evidence directly.
        - Your answer must be one to two sentences maximum.
        - If the evidence is empty, state that the information could not be found.
        """
        response = await self.reasoning_model.generate_content_async(prompt)
        return response.text.strip()

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Resilient Analyst process for question: '{query}' ---")

        search_queries = self._deterministic_query_expansion(query)
        all_retrieved_chunks = [chunk for sq in search_queries for chunk in self.retriever.retrieve(sq, top_k=5)]
        unique_chunks = list({chunk.chunk_id: chunk for chunk in all_retrieved_chunks}.values())

        distilled_evidence_chunks = await self._distill_evidence_with_resilience(query, unique_chunks)

        if not distilled_evidence_chunks:
            final_answer_text = "The information could not be found in the provided documents."
        else:
            # We now only synthesize the single best chunk of evidence
            evidence_for_synthesis = distilled_evidence_chunks[0].text
            final_answer_text = await self._synthesize_with_resilience(query, evidence_for_synthesis)

        logger.info(f"SYNTHESIZER: Generated final answer: {final_answer_text}")
        return Answer(query=query, decision="Information Found", payout=None, justification=[],
                      simple_answer=final_answer_text)

    async def answer_all_questions(self, questions: List[str]) -> List[Answer]:
        answers = []
        for query in questions:
            answer = await self.answer_question(query)
            answers.append(answer)
        return answers