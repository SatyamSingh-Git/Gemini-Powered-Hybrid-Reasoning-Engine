# src/reasoning_engine/core/ingestion.py (THE FINAL, BUG-FREE VERSION)

import asyncio
import uuid
import httpx
import tempfile
import logging
from typing import List
from pathlib import Path

from .models import DocumentChunk
from unstructured.partition.auto import partition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increased timeout and added headers to mimic a real browser
HTTP_CONFIG = {
    "timeout": 120.0,  # Increased to 2 minutes
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
}


async def _download_and_save_temp(session: httpx.AsyncClient, url: str) -> str:
    """Downloads a file from a URL and saves it to a temporary local path."""
    logger.info(f"Attempting to download from URL: {url}")
    try:
        async with session.stream("GET", url, follow_redirects=True, **HTTP_CONFIG) as response:
            response.raise_for_status()
            suffix = Path(url.split('?')[0]).suffix or '.bin'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                async for chunk in response.aiter_bytes():
                    temp_file.write(chunk)
                logger.info(f"SUCCESS: Downloaded {url} to {temp_file.name}")
                return temp_file.name
    except httpx.RequestError as e:
        logger.error(f"DOWNLOAD FAILED: HTTP request error for {url}. Error: {e}", exc_info=True)
        return ""
    except Exception as e:
        logger.error(f"DOWNLOAD FAILED: An unexpected error occurred for {url}. Error: {e}", exc_info=True)
        return ""


def _partition_and_chunk(file_path: str, source_url: str) -> List[DocumentChunk]:
    """
    Uses the unstructured library to parse the document and then intelligently
    chunks the resulting elements, preserving correct page numbers.
    """
    logger.info(f"Attempting to partition file: {file_path}")
    try:
        elements = partition(filename=file_path, strategy="auto")
    except Exception as e:
        logger.error(f"PARTITION FAILED: Could not partition file {file_path}. Error: {e}", exc_info=True)
        return []

    # THE FIX IS HERE: A single, robust loop to create chunks correctly.
    document_chunks = []
    current_chunk_text = ""
    current_chunk_page = "1"  # Default to page 1

    for element in elements:
        page_num = str(element.metadata.page_number or current_chunk_page)
        current_chunk_page = page_num  # Keep track of the current page

        element_text = str(element.text)

        # Start a new chunk for titles to maintain semantic context
        if element.category == 'Title':
            if current_chunk_text.strip():
                document_chunks.append(DocumentChunk(
                    chunk_id=str(uuid.uuid4()), text=current_chunk_text.strip(),
                    source_document=source_url, page_label=current_chunk_page
                ))
            current_chunk_text = element_text + "\n\n"
        else:
            current_chunk_text += element_text + "\n\n"

        # Split chunks that get too long
        if len(current_chunk_text) > 1500:
            document_chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()), text=current_chunk_text.strip(),
                source_document=source_url, page_label=current_chunk_page
            ))
            current_chunk_text = ""

    # Add the final remaining chunk
    if current_chunk_text.strip():
        document_chunks.append(DocumentChunk(
            chunk_id=str(uuid.uuid4()), text=current_chunk_text.strip(),
            source_document=source_url, page_label=current_chunk_page
        ))

    # Final filter for quality
    final_chunks = [chunk for chunk in document_chunks if len(chunk.text) > 50]

    logger.info(f"SUCCESS: Partitioned and chunked {file_path} into {len(final_chunks)} chunks.")
    return final_chunks


async def process_documents(document_urls: List[str]) -> List[DocumentChunk]:
    """Main orchestration function for multi-format ingestion."""
    logger.info("Starting robust multi-format document ingestion...")
    all_chunks = []
    async with httpx.AsyncClient() as session:
        download_tasks = [_download_and_save_temp(session, url) for url in document_urls]
        local_file_paths = await asyncio.gather(*download_tasks)

        for i, file_path in enumerate(local_file_paths):
            if file_path:
                chunks = _partition_and_chunk(file_path, source_url=document_urls[i])
                all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("INGESTION FAILED: No chunks were created from any of the provided documents.")

    logger.info(f"Total chunks created from all documents: {len(all_chunks)}")
    return all_chunks