# src/reasoning_engine/core/ingestion.py (THE FINAL MULTI-FORMAT VERSION)

import asyncio
import uuid
import httpx
import tempfile
import logging
from typing import List
from pathlib import Path

from .models import DocumentChunk
# The core of our new multi-format strategy
from unstructured.partition.auto import partition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _download_and_save_temp(session: httpx.AsyncClient, url: str) -> str:
    """Downloads a file from a URL and saves it to a temporary local path."""
    try:
        response = await session.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()

        # Create a temporary file with a smart suffix to help unstructured's detection
        suffix = Path(url.split('?')[0]).suffix or '.bin'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            logger.info(f"Successfully downloaded {url} to {temp_file.name}")
            return temp_file.name
    except httpx.RequestError as e:
        logger.error(f"Error downloading {url}: {e}")
        return ""


def _partition_and_chunk(file_path: str, source_url: str) -> List[DocumentChunk]:
    """
    Uses the unstructured library to parse the document and then converts
    the resulting elements into our standardized DocumentChunk objects.
    """
    try:
        # Unstructured's magic function that handles dozens of formats automatically
        elements = partition(filename=file_path, strategy="auto")
    except Exception as e:
        logger.error(f"Failed to partition file {file_path}. Error: {e}", exc_info=True)
        return []

    # Convert unstructured's "Element" objects into our app's "DocumentChunk" objects
    chunks = []
    current_chunk_text = ""
    for element in elements:
        # For better context, we start a new chunk whenever we see a Title
        if element.category == 'Title':
            if current_chunk_text.strip():
                chunks.append(current_chunk_text.strip())
            current_chunk_text = str(element.text) + "\n\n"
        else:
            current_chunk_text += str(element.text) + "\n\n"

        # Split chunks that get too long to manage context size
        if len(current_chunk_text) > 1500:
            chunks.append(current_chunk_text.strip())
            current_chunk_text = ""

    # Add the last remaining chunk
    if current_chunk_text.strip():
        chunks.append(current_chunk_text.strip())

    # Create the final DocumentChunk objects
    document_chunks = []
    for text in chunks:
        if len(text) > 50:  # Filter out very short, noisy chunks
            document_chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=text,
                source_document=source_url,  # Use the original URL as the source
                page_label=str(element.metadata.page_number or 'N/A')
            ))

    logger.info(f"Partitioned and chunked {file_path} into {len(document_chunks)} chunks.")
    return document_chunks


async def process_documents(document_urls: List[str]) -> List[DocumentChunk]:
    """
    The main orchestration function for multi-format ingestion. It downloads
    all documents concurrently and then partitions them into standardized chunks.
    """
    logger.info("Starting multi-format document ingestion...")
    all_chunks = []
    async with httpx.AsyncClient() as session:
        # Concurrently download all documents to temporary local files
        download_tasks = [_download_and_save_temp(session, url) for url in document_urls]
        local_file_paths = await asyncio.gather(*download_tasks)

        for i, file_path in enumerate(local_file_paths):
            if file_path:
                # Process each downloaded file
                chunks = _partition_and_chunk(file_path, source_url=document_urls[i])
                all_chunks.extend(chunks)

    logger.info(f"Total chunks created from all documents: {len(all_chunks)}")
    return all_chunks