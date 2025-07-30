# src/reasoning_engine/core/ingestion.py (UPGRADED)

import asyncio
import uuid
import httpx
import fitz  # PyMuPDF
from typing import List
from .models import DocumentChunk
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_text_optimized(pdf_content: bytes, url: str) -> List[DocumentChunk]:
    """Optimized, memory-efficient text extraction and chunking."""
    chunks = []
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if len(text.strip()) > 30:  # Early filtering
                # Simple paragraph chunking is robust for this task
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    cleaned_text = para.strip()
                    if len(cleaned_text) > 50:
                        chunks.append(DocumentChunk(
                            chunk_id=str(uuid.uuid4()), text=cleaned_text,
                            source_document=url, page_label=str(page_num + 1)
                        ))
        doc.close()
        logger.info(f"Optimized parsing for {url} yielded {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"PDF extraction failed for {url}: {e}")
        return []


async def process_documents(document_urls: List[str]) -> List[DocumentChunk]:
    logger.info("Starting optimized document ingestion...")
    all_chunks = []
    loop = asyncio.get_event_loop()

    async with httpx.AsyncClient() as session:
        # Download all PDFs concurrently
        download_tasks = [session.get(url, timeout=30.0) for url in document_urls]
        responses = await asyncio.gather(*download_tasks, return_exceptions=True)

        pdf_contents = [
            (document_urls[i], res.content) for i, res in enumerate(responses)
            if isinstance(res, httpx.Response) and res.status_code == 200
        ]

        # Process PDFs in parallel using a thread pool for the sync fitz library
        with ThreadPoolExecutor() as executor:
            processing_tasks = [
                loop.run_in_executor(executor, _extract_text_optimized, content, url)
                for url, content in pdf_contents
            ]
            results = await asyncio.gather(*processing_tasks)
            for chunk_list in results:
                all_chunks.extend(chunk_list)

    logger.info(f"Total chunks created from all documents: {len(all_chunks)}")
    return all_chunks