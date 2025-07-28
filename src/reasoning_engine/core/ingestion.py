import asyncio
import uuid
import httpx
import fitz  # PyMuPDF
import re
from typing import List
from .models import DocumentChunk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _download_pdf_async(
        session: httpx.AsyncClient, url: str
) -> tuple[str, bytes]:
    """
    Asynchronously downloads a single PDF from a URL.

    Args:
        session: An httpx.AsyncClient session.
        url: The URL of the PDF to download.

    Returns:
        A tuple containing the URL and the PDF content as bytes.
    """
    try:
        response = await session.get(url, timeout=30.0)
        response.raise_for_status()  # Raise an exception for bad status codes
        logger.info(f"Successfully downloaded {url}")
        return str(url), response.content
    except httpx.RequestError as e:
        logger.error(f"Error downloading {url}: {e}")
        return str(url), b""


def _parse_and_chunk_pdf(
        url: str, pdf_content: bytes
) -> List[DocumentChunk]:
    """
    Parses PDF content from memory and chunks it using a policy-aware strategy.

    Args:
        url: The source URL of the PDF.
        pdf_content: The byte content of the PDF.

    Returns:
        A list of DocumentChunk objects.
    """
    chunks = []
    if not pdf_content:
        return chunks

    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        for page_num, page in enumerate(doc):
            page_label = str(page_num + 1)
            text = page.get_text("text")

            # Policy-aware chunking strategy:
            # 1. Split by double newlines (paragraphs) as a base.
            # 2. Further split by patterns that often denote new clauses
            #    (e.g., "1.", "a)", "• ").
            # 3. Filter out very short, likely irrelevant lines.

            base_chunks = text.split('\n\n')
            for base_chunk in base_chunks:
                # Use regex to split by clause markers at the beginning of a line
                sub_chunks = re.split(r'\n(?=\d+\.|\w\)|•)', base_chunk)
                for sub_chunk in sub_chunks:
                    cleaned_text = sub_chunk.strip()
                    if len(cleaned_text) > 30:  # Heuristic to filter out noise
                        chunk = DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=cleaned_text,
                            source_document=url,
                            page_label=page_label,
                        )
                        chunks.append(chunk)

        logger.info(f"Parsed and chunked {url} into {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Failed to parse PDF from {url}: {e}")

    return chunks


async def process_documents(document_urls: List[str]) -> List[DocumentChunk]:
    """
    The main orchestration function for the ingestion phase.
    It concurrently downloads multiple PDFs and then parses them.

    Args:
        document_urls: A list of URLs pointing to the PDFs.

    Returns:
        A single list containing all document chunks from all processed PDFs.
    """
    logger.info("Starting document ingestion process...")
    all_chunks = []
    async with httpx.AsyncClient() as session:
        # Create a list of download tasks to run concurrently
        download_tasks = [
            _download_pdf_async(session, url) for url in document_urls
        ]
        # Wait for all downloads to complete
        downloaded_pdfs = await asyncio.gather(*download_tasks)

        # Now, parse the downloaded content
        for url, pdf_content in downloaded_pdfs:
            if pdf_content:
                chunks = _parse_and_chunk_pdf(url, pdf_content)
                all_chunks.extend(chunks)

    logger.info(f"Total chunks created from all documents: {len(all_chunks)}")
    return all_chunks