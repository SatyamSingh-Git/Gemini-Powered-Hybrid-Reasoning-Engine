# src/reasoning_engine/core/agent.py (THE FINAL "DISTILLER-ANALYST" AGENT)
import re

import google.generativeai as genai
from typing import List, Dict, Any
import logging
import json
import asyncio
import hashlib

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
        variants = {query};
        keywords = [word.lower() for word in query.split() if
                    len(word) > 3 and word.lower() not in ['what', 'is', 'are', 'the', 'for', 'a', 'an']];
        if keywords: variants.add(' '.join(keywords))
        if 'define' in query.lower() or 'what is' in query.lower(): variants.add(f"definition of {' '.join(keywords)}");
        return list(variants)

    def _execute_tool_call(self, tool_call, chunks) -> List[Dict[str, Any]]:
        tool_name = tool_call.name;
        tool_args = dict(tool_call.args)
        if tool_name in AVAILABLE_TOOLS:
            logger.info(f"EXECUTOR: Calling tool '{tool_name}' with args: {tool_args}")
            tool_function = AVAILABLE_TOOLS[tool_name];
            tool_args['clauses'] = chunks
            try:
                return tool_function(**tool_args)
            except Exception as e:
                return [{"error": f"Error executing tool: {e}"}]
        return [{"error": f"Tool '{tool_name}' not found."}]

    async def _distill_evidence(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Uses a fast LLM to find the single best chunk of evidence."""
        if not chunks:
            return []

        context_str = "\n---\n".join([f"ID: {i}\n{chunk.text}" for i, chunk in enumerate(chunks)])

        distillation_prompt = f"""
        You are a research assistant. Your task is to review the following text chunks, which are separated by '---'.
        Based on the user's question, identify the SINGLE chunk ID that contains the most direct and precise answer.
        Respond with ONLY the numeric ID of that single best chunk.

        User Question: "{query}"

        <EVIDENCE>
        {context_str}
        </EVIDENCE>
        """
        try:
            response = await self.fast_model.generate_content_async(distillation_prompt)
            best_chunk_id_str = response.text.strip()
            best_chunk_id = int(re.search(r'\d+', best_chunk_id_str).group())
            logger.info(f"DISTILLER: Identified best evidence chunk ID: {best_chunk_id}")
            return [chunks[best_chunk_id]]
        except (ValueError, IndexError, AttributeError) as e:
            logger.warning(f"DISTILLER: Could not distill evidence, falling back to all chunks. Error: {e}")
            return chunks  # Fallback to using all chunks if distillation fails

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Distiller-Analyst process for question: '{query}' ---")

        final_answer_key = self._get_cache_key(f"final_answer:{query}")
        cached_answer = await self.redis.get(final_answer_key)
        if cached_answer:
            logger.info("CACHE HIT: Returning final answer from cache.")
            return Answer(query=query, decision="Information Found (Cached)", payout=None, justification=[],
                          simple_answer=cached_answer)

        # Phase 1: Broad Retrieval
        search_queries = self._deterministic_query_expansion(query)
        all_retrieved_chunks = [chunk for sq in search_queries for chunk in self.retriever.retrieve(sq, top_k=5)]
        unique_chunks = list({chunk.chunk_id: chunk for chunk in all_retrieved_chunks}.values())

        # Phase 2: Evidence Distillation
        distilled_evidence = await self._distill_evidence(query, unique_chunks)

        # Phase 3: Focused Analysis (Planner, Executor, Synthesizer)
        planning_prompt = f"""
                        You are a meticulous and logical research planner. Your task is to create a step-by-step plan to answer the user's question about an insurance policy. Your plan must consist of a sequence of tool calls.

                        **Instructions:**
                        1.  Analyze the user's question to understand the core intent and identify all key entities (e.g., "pre-existing diseases", "room rent", "Plan A").
                        2.  Select the single most specific tool that can answer the question.
                        3.  Construct the arguments for the tool using the entities you identified.
                        4.  Prioritize the most specific tool. Only use `find_information` for general definitions or when no other tool fits.

                        **High-Quality Examples:**

                        **User Question:** "What is the waiting period for cataract surgery?"
                        **Your Plan:**
                        1.  Call `check_waiting_period` with procedure_terms=["cataract surgery", "cataract"].

                        **User Question:** "How does the policy define a 'Hospital'?"
                        **Your Plan:**
                        1.  Call `find_information` with topic="definition of a hospital".

                        **User Question:** "Are there any sub-limits on room rent and ICU charges for Plan A?"
                        **Your Plan:**
                        1.  Call `extract_monetary_limit` with limit_terms=["room rent", "ICU charges", "Plan A"].

                        **User Question:** "{query}"

                        Create the research plan now.
                        """
        chat_session = self.reasoning_model.start_chat()
        plan_response = await chat_session.send_message_async(planning_prompt)

        gathered_evidence = []
        try:
            plan_tool_calls = plan_response.candidates[0].content.parts
            for part in plan_tool_calls:
                if part.function_call:
                    tool_results = self._execute_tool_call(part.function_call, distilled_evidence)
                    if tool_results and "error" not in tool_results[0]:
                        gathered_evidence.append({"step": part.function_call.name, "evidence": tool_results})
        except (ValueError, IndexError):
            gathered_evidence.append({"error": "Failed to create a valid plan."})

        synthesis_prompt = f"""
                       You are an expert insurance analyst and a master of synthesis. Your final and only job is to synthesize the provided evidence dossier into a professional, comprehensive, and definitive answer to the user's original question.

                       The user's original question was:
                       <QUESTION>
                       {query}
                       </QUESTION>

                       Here is the evidence dossier your team has gathered:
                       <EVIDENCE>
                       {json.dumps(gathered_evidence, indent=2)}
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
        final_response = await self.reasoning_model.generate_content_async(synthesis_prompt)
        final_answer_text = final_response.text.strip()

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