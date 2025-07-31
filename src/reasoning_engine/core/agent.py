# Python

import hashlib
import google.generativeai as genai
from typing import List, Dict, Any
import logging
import json
import asyncio

from ..config import get_settings
from .models import DocumentChunk, Answer
from .tools import AVAILABLE_TOOLS
from .retrievel import HybridRetriever

logger = logging.getLogger(__name__)

try:
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
except Exception as e:
    logger.error(f"Failed to configure Gemini: {e}")

class ReasoningAgent:
    def __init__(self, retriever: HybridRetriever, redis_client):
        self.retriever = retriever
        self.redis = redis_client
        self.reasoning_model = genai.GenerativeModel('gemini-2.5-flash', tools=list(AVAILABLE_TOOLS.values()))
        self.fast_model = genai.GenerativeModel('gemini-2.5-flash')

    def _get_cache_key(self, content: str) -> str:
        return f"gemini-cache:{hashlib.md5(content.encode()).hexdigest()}"

    def _deterministic_query_expansion(self, query: str) -> List[str]:
        variants = {query}
        keywords = [word.lower() for word in query.split() if
                    len(word) > 3 and word.lower() not in ['what', 'is', 'are', 'the', 'for', 'a', 'an']]
        if keywords:
            variants.add(' '.join(keywords))
        if 'define' in query.lower() or 'what is' in query.lower():
            variants.add(f"definition of {' '.join(keywords)}")
        return list(variants)

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Lean & Mean Analyst process for question: '{query}' ---")

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

        final_response = await self.reasoning_model.generate_content_async(synthesis_prompt)
        # Handle Gemini function_call responses gracefully
        if hasattr(final_response, "text") and final_response.text:
            final_answer_text = final_response.text.strip()
        else:
            final_answer_text = "The information could not be found in the provided documents."

        await self.redis.set(final_answer_key, final_answer_text, ex=3600)

        logger.info(f"SYNTHESIZER: Generated final answer: {final_answer_text}")
        return Answer(query=query, decision="Information Found", payout=None, justification=[],
                      simple_answer=final_answer_text)

    async def answer_all_questions(self, questions: List[str]) -> List[Answer]:
        answers = []
        for query in questions:
            answer = await self.answer_question(query)
            answers.append(answer)
            await asyncio.sleep(0.5)
        return answers