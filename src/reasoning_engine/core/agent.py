# src/reasoning_engine/core/agent.py (THE FINAL, BUG-FREE, NO-CACHE VERSION)

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
                    if "429" in str(e) and "ResourceExhausted" in str(e):
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
    # We no longer accept a redis_client
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
        if not chunks: return []
        context_str = "\n---\n".join([f"ID: {i}\n{chunk.text}" for i, chunk in enumerate(chunks)])
        prompt = f"""
                You are an expert insurance analyst and a master of synthesis. Your final and only job is to synthesize the provided evidence dossier into a professional, comprehensive, and definitive answer to the user's original question.

                The user's original question was:
                <QUESTION>
                {query}
                </QUESTION>

                Here is the evidence dossier your team has gathered:
                <EVIDENCE>
                {json.dumps(context_str)}
                </EVIDENCE>

                **Your Instructions for Synthesizing the Final Answer:**
                1.  Synthesize a single, clear, and professional paragraph.
                2.  Your answer MUST be grounded *exclusively* in the provided `<EVIDENCE>`.
                3.  If the evidence contains specific numbers, dates, or percentages, you MUST include them in your answer.
                4.  If the evidence contains conflicting information, state the conflict clearly (e.g., "One clause states a 30-day period, while another specifies 24 months for this condition.").
                5.  If the evidence is empty or contains an error, you MUST state that the information could not be found in the provided policy documents.

                **CRITICAL RULES OF CONDUCT:**
                -   NEVER apologize.
                -   NEVER ask for more information or say "Please provide the document."
                -   NEVER break character. You are a professional analyst providing a definitive report.
                -   NEVER mention the tools, the dossier, or your internal process (e.g., "Based on the evidence...") in your final answer. Simply state the facts.
                """
        response = await self.fast_model.generate_content_async(prompt)
        best_chunk_id = int(re.search(r'\d+', response.text.strip()).group())
        logger.info(f"DISTILLER: Identified best evidence chunk ID: {best_chunk_id}")
        return [chunks[best_chunk_id]]

    @resilient_api_call()
    async def _synthesize_with_resilience(self, query: str, evidence: List[Dict]) -> str:
        prompt = f"You are an expert analyst. Synthesize the provided evidence into a concise, direct, and professional answer to the user's question.\n\n**User Question:** \"{query}\"\n\n**Evidence:**\n{json.dumps(evidence, indent=2)}\n\n**Instructions:**\n- State the single most important fact directly.\n- Your answer must be one to two sentences maximum.\n- If the evidence is empty or contains an error, state that the information could not be found."
        response = await self.reasoning_model.generate_content_async(prompt)
        return response.text.strip()

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Resilient Analyst process for question: '{query}' ---")

        # The cache check is now removed.

        search_queries = self._deterministic_query_expansion(query)
        all_retrieved_chunks = [chunk for sq in search_queries for chunk in self.retriever.retrieve(sq, top_k=5)]
        unique_chunks = list({chunk.chunk_id: chunk for chunk in all_retrieved_chunks}.values())

        distilled_evidence_chunks = await self._distill_evidence_with_resilience(query, unique_chunks)

        # We will simplify and make one direct synthesis call
        if not distilled_evidence_chunks:
            final_answer_text = "The information could not be found in the provided documents."
        else:
            evidence_for_synthesis = [{"evidence": chunk.text} for chunk in distilled_evidence_chunks]
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