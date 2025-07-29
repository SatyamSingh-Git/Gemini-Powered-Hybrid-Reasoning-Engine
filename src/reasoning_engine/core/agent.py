# src/reasoning_engine/core/agent.py (THE LEAN, EFFICIENT, WINNING VERSION)

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
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.model = genai.GenerativeModel('gemini-2.5-flash', tools=list(AVAILABLE_TOOLS.values()))
        self.query_gen_model = genai.GenerativeModel('gemini-2.5-flash')

    def _convert_proto_to_dict(self, proto_obj: Any) -> Any:
        if hasattr(proto_obj, 'items'):
            return {key: self._convert_proto_to_dict(value) for key, value in proto_obj.items()}
        elif hasattr(proto_obj, '__iter__') and not isinstance(proto_obj, (str, bytes)):
            return [self._convert_proto_to_dict(item) for item in proto_obj]
        else:
            return proto_obj

    def _execute_tool_call(self, tool_call, chunks) -> List[Dict[str, Any]]:
        tool_name = tool_call.name
        tool_args = self._convert_proto_to_dict(tool_call.args)
        if tool_name in AVAILABLE_TOOLS:
            logger.info(f"EXECUTOR: Calling tool '{tool_name}' with args: {tool_args}")
            tool_function = AVAILABLE_TOOLS[tool_name]
            tool_args['clauses'] = chunks
            try:
                return tool_function(**tool_args)
            except Exception as e:
                return [{"error": f"Error executing tool: {e}"}]
        return [{"error": f"Tool '{tool_name}' not found."}]

    async def _generate_multiple_queries(self, query: str) -> List[str]:
        prompt = f"""
        Generate 3 diverse, complementary search queries based on the user's question.
        Provide the output as a JSON list of strings.
        User question: "{query}"
        """
        try:
            response = await self.query_gen_model.generate_content_async(prompt)
            json_text = response.text.strip().lstrip("```json").rstrip("```")
            queries = json.loads(json_text)
            queries.append(query)
            logger.info(f"Generated search queries: {queries}")
            return list(set(queries))
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}. Falling back to original query.")
            return [query]

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Lean Super-Analyst process for question: '{query}' ---")

        # ==============================================================================
        # PHASE 1: MULTI-QUERY RETRIEVAL (LLM CALL #1)
        # ==============================================================================
        search_queries = await self._generate_multiple_queries(query)
        all_retrieved_chunks = []
        for sq in search_queries:
            chunks_for_query = self.retriever.retrieve(sq, top_k=5)
            all_retrieved_chunks.extend(chunks_for_query)
        unique_chunks = {chunk.chunk_id: chunk for chunk in all_retrieved_chunks}
        retrieved_chunks = list(unique_chunks.values())
        logger.info(f"Retrieved {len(retrieved_chunks)} unique chunks from {len(search_queries)} queries.")

        # ==============================================================================
        # PHASE 2: TOOL-USE AND SYNTHESIS (LLM CALL #2)
        # ==============================================================================
        system_prompt = f"""
        You are a precise, fact-based insurance policy analyst. Your SOLE purpose is to answer the user's question based ONLY on the output of functions you call.

        Your workflow is as follows:
        1.  Based on the user's question, call the single most appropriate tool to find the specific data required.
        2.  You will then receive the output from that tool.
        3.  Based ONLY on the data returned by the tool, formulate a comprehensive, professional, single-paragraph answer to the original user question.
        4.  If the tool output is empty or contains an error, you MUST state that the information could not be found in the provided policy documents.

        **CRITICAL RULES:**
        - DO NOT ask for more information.
        - DO NOT apologize.
        - DO NOT say "Please provide the document."
        - Answer ONLY from the tool's output.
        """

        chat_session = self.model.start_chat()
        initial_message = f"{system_prompt}\n\n**User Question:** {query}"

        response = await chat_session.send_message_async(initial_message)
        response_part = response.candidates[0].content.parts[0]

        final_answer_text = f"The information for '{query}' could not be determined."

        if response_part.function_call:
            tool_call = response_part.function_call
            tool_results = self._execute_tool_call(tool_call, retrieved_chunks)
            logger.info(f"Tool '{tool_call.name}' returned: {json.dumps(tool_results, indent=2)}")

            if tool_results and "error" not in tool_results[0]:
                synthesis_response = await chat_session.send_message_async(
                    genai.protos.FunctionResponse(name=tool_call.name, response={"result": tool_results})
                )
                final_answer_text = synthesis_response.text.strip()
        else:
            # If the model doesn't call a tool, it's because it thinks it can answer from the prompt (which we forbid).
            # This is a fallback to prevent it from refusing to use tools.
            final_answer_text = "The model did not select a tool to answer the question, so a definitive answer could not be determined."

        logger.info(f"Generated simple answer: {final_answer_text}")
        return Answer(query=query, decision="Information Found", payout=None, justification=[],
                      simple_answer=final_answer_text)

    async def answer_all_questions(self, questions: List[str]) -> List[Answer]:
        answers = []
        for query in questions:
            answer = await self.answer_question(query)
            answers.append(answer)
            await asyncio.sleep(1.5)
        return answers