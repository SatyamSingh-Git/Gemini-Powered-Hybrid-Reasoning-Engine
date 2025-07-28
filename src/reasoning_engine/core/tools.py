# src/reasoning_engine/core/tools.py (UPGRADED VERSION)

import re
from typing import List, Dict, Any
from .models import DocumentChunk
import logging

logger = logging.getLogger(__name__)


def extract_keywords(clauses: List[DocumentChunk], keywords: List[str]) -> List[Dict[str, Any]]:
    logger.info(f"Executing tool: extract_keywords with keywords: {keywords}")
    findings = []
    # This regex is case-insensitive and looks for whole words
    keyword_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b', re.IGNORECASE)
    for chunk in clauses:
        if any(keyword.lower() in chunk.text.lower() for keyword in keywords):
            findings.append({"found_keyword": keywords[0], "evidence": chunk.text, "source": chunk.source_document,
                             "page": chunk.page_label})
    return findings


def check_waiting_period(clauses: List[DocumentChunk], procedure_terms: List[str]) -> List[Dict[str, Any]]:
    logger.info(f"Executing tool: check_waiting_period for terms: {procedure_terms}")
    findings = []
    waiting_period_pattern = re.compile(
        r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty-four|thirty-six|forty-eight)[\s-]*'
        r'(days?|months?|years?)', re.IGNORECASE)

    # First, search for waiting periods in chunks that are highly relevant to the procedure
    for chunk in clauses:
        if any(term.lower() in chunk.text.lower() for term in procedure_terms):
            matches = waiting_period_pattern.finditer(chunk.text)
            for match in matches:
                findings.append(
                    {"waiting_period": match.group(0), "condition_context": chunk.text, "source": chunk.source_document,
                     "page": chunk.page_label})

    # If no specific match is found, fall back to the first generic waiting period found in the retrieved context
    if not findings:
        for chunk in clauses:
            matches = waiting_period_pattern.finditer(chunk.text)
            for match in matches:
                findings.append(
                    {"waiting_period": match.group(0), "condition_context": chunk.text, "source": chunk.source_document,
                     "page": chunk.page_label})
                if findings: break
            if findings: break

    return findings


def extract_monetary_limit(clauses: List[DocumentChunk], limit_terms: List[str]) -> List[Dict[str, Any]]:
    logger.info(f"Executing tool: extract_monetary_limit for terms: {limit_terms}")
    findings = []
    monetary_pattern = re.compile(r'(Rs\.?|₹)\s?[\d,]+|\d+\s?(%|percent|lakh|lakhs|crore)\s*(of Sum Insured)?',
                                  re.IGNORECASE)
    for chunk in clauses:
        if any(term.lower() in chunk.text.lower() for term in limit_terms):
            matches = monetary_pattern.finditer(chunk.text)
            for match in matches:
                findings.append(
                    {"limit": match.group(0).strip(), "context": chunk.text, "source": chunk.source_document,
                     "page": chunk.page_label})
    return findings


# NEW Tool for general queries
def find_information(clauses: List[DocumentChunk], topic: str) -> List[Dict[str, Any]]:
    """Use this for general questions, definitions (e.g., 'define hospital'), or when other tools are not specific enough."""
    logger.info(f"Executing tool: find_information for topic: {topic}")
    if clauses:
        # Just return the single most relevant chunk the retriever found
        return [{"topic": topic, "evidence": clauses[0].text, "source": clauses[0].source_document,
                 "page": clauses[0].page_label}]
    return []


AVAILABLE_TOOLS = {
    "extract_keywords": extract_keywords,
    "check_waiting_period": check_waiting_period,
    "extract_monetary_limit": extract_monetary_limit,
    "find_information": find_information,  # Add the new tool
}