# src/reasoning_engine/core/agent.py (THE FINAL, BUG-FREE VERSION)

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
        self.model = genai.GenerativeModel('gemini-2.5-flash', tools=list(AVAILABLE_TOOLS.values()))
        self.query_gen_model = genai.GenerativeModel('gemini-2.5-flash')

    def _get_cache_key(self, content: str) -> str:
        return f"gemini-cache:{hashlib.md5(content.encode()).hexdigest()}"

    def _deterministic_query_expansion(self, query: str) -> List[str]:
        # This function is correct and remains unchanged
        variants = {query};
        keywords = [word.lower() for word in query.split() if
                    len(word) > 3 and word.lower() not in ['what', 'is', 'are', 'the', 'for', 'a', 'an']];
        if keywords: variants.add(' '.join(keywords))
        if 'define' in query.lower() or 'what is' in query.lower(): variants.add(f"definition of {' '.join(keywords)}");
        logger.info(f"Generated deterministic search queries: {list(variants)}");
        return list(variants)

    # NEW UTILITY FUNCTION TO FIX THE JSON ERROR
    def _convert_proto_to_dict(self, proto_obj: Any) -> Any:
        """Recursively converts a Google proto object to a standard Python dict."""
        if hasattr(proto_obj, 'items'):  # Works for proto maps
            return {key: self._convert_proto_to_dict(value) for key, value in proto_obj.items()}
        elif hasattr(proto_obj, '__iter__') and not isinstance(proto_obj, (str, bytes)):  # Works for proto lists
            return [self._convert_proto_to_dict(item) for item in proto_obj]
        else:  # Base case for simple types
            return proto_obj

    def _execute_tool_call(self, tool_call, chunks) -> List[Dict[str, Any]]:
        tool_name = tool_call.name
        # Use the new utility to safely convert the arguments
        tool_args = self._convert_proto_to_dict(tool_call.args)
        if tool_name in AVAILABLE_TOOLS:
            logger.info(f"EXECUTOR: Calling tool '{tool_name}' with args: {tool_args}")
            tool_function = AVAILABLE_TOOLS[tool_name];
            tool_args['clauses'] = chunks
            try:
                return tool_function(**tool_args)
            except Exception as e:
                return [{"error": f"Error executing tool: {e}"}]
        return [{"error": f"Tool '{tool_name}' not found."}]

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Optimized Super-Analyst process for question: '{query}' ---")

        final_answer_key = self._get_cache_key(f"final_answer:{query}")
        cached_answer = await self.redis.get(final_answer_key)
        if cached_answer:
            logger.info("CACHE HIT: Returning final answer from cache.")
            return Answer(query=query, decision="Information Found (Cached)", payout=None, justification=[],
                          simple_answer=cached_answer)

        # The multi-query and retrieval logic remains unchanged
        search_queries = self._deterministic_query_expansion(query)
        all_retrieved_chunks = []
        for sq in search_queries:
            chunks_for_query = self.retriever.retrieve(sq, top_k=5)
            all_retrieved_chunks.extend(chunks_for_query)
        unique_chunks = {chunk.chunk_id: chunk for chunk in all_retrieved_chunks}
        retrieved_chunks = list(unique_chunks.values())

        # The planning logic remains unchanged
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
        chat_session = self.model.start_chat()
        plan_response = await chat_session.send_message_async(planning_prompt)

        gathered_evidence = []
        try:
            plan_tool_calls = plan_response.candidates[0].content.parts
            for part in plan_tool_calls:
                if not part.function_call: continue
                tool_results = self._execute_tool_call(part.function_call, retrieved_chunks)
                if tool_results and "error" not in tool_results[0]:
                    gathered_evidence.append({
                        "step_executed": part.function_call.name,
                        # Use the utility here as well to ensure the evidence log is clean
                        "parameters": self._convert_proto_to_dict(part.function_call.args),
                        "evidence_found": tool_results
                    })
        except (ValueError, IndexError) as e:
            gathered_evidence.append({"error": "Failed to create a valid plan."})

        # The synthesis prompt now receives clean, serializable JSON
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
        final_response = await self.model.generate_content_async(synthesis_prompt)
        final_answer_text = final_response.text.strip()

        await self.redis.set(final_answer_key, final_answer_text, ex=3600)

        logger.info(f"SYNTHESIZER: Generated final answer: {final_answer_text}")
        return Answer(query=query, decision="Information Found", payout=None, justification=[],
                      simple_answer=final_answer_text)

    async def answer_all_questions(self, questions: List[str]) -> List[Answer]:
        # This function is correct and remains unchanged
        answers = []
        for query in questions:
            answer = await self.answer_question(query)
            answers.append(answer)
            await asyncio.sleep(0.5)
        return answers