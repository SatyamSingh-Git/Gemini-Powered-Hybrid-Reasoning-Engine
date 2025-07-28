# src/reasoning_engine/core/retrieval.py (WITH RE-RANKER)

import asyncio
import google.generativeai as genai
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict
from .models import DocumentChunk
import logging
from tqdm.asyncio import tqdm_asyncio
from sentence_transformers import CrossEncoder  # Import the re-ranker

logger = logging.getLogger(__name__)

GEMINI_EMBEDDING_MODEL = "models/embedding-001"


class HybridRetriever:
    def __init__(self, chunks: List[DocumentChunk]):
        if not chunks:
            raise ValueError("Cannot initialize HybridRetriever with empty chunks.")

        self.chunks = chunks
        self.chunk_map: Dict[str, DocumentChunk] = {chunk.chunk_id: chunk for chunk in chunks}

        # Initialize the re-ranker model
        logger.info("Initializing CrossEncoder re-ranker model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Re-ranker initialized.")

    # ... The _embed_with_retry and _build_indices_async methods remain unchanged ...
    async def _embed_with_retry(self, content: str) -> List[float]:
        try:
            result = await genai.embed_content_async(model=GEMINI_EMBEDDING_MODEL, content=content,
                                                     task_type="RETRIEVAL_DOCUMENT")
            return result['embedding']
        except Exception as e:
            logger.error(f"API embedding failed: {e}")
            await asyncio.sleep(1)
            return []

    async def _build_indices_async(self):
        logger.info(f"Embedding {len(self.chunks)} chunks via Gemini API...")
        texts = [chunk.text for chunk in self.chunks]
        tokenized_corpus = [doc.split(" ") for doc in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus_map = {i: self.chunks[i] for i in range(len(self.chunks))}

        semaphore = asyncio.Semaphore(20)

        async def embed_task(chunk):
            async with semaphore:
                return await self._embed_with_retry(chunk.text)

        embedding_coroutines = [embed_task(chunk) for chunk in self.chunks]
        embeddings = await tqdm_asyncio.gather(*embedding_coroutines, desc="Embedding Chunks")

        valid_embeddings = [emb for emb in embeddings if emb]
        if not valid_embeddings: raise RuntimeError("Failed to generate any embeddings.")

        embedding_dim = len(valid_embeddings[0])
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.chunk_ids = np.array([i for i, emb in enumerate(embeddings) if emb])
        self.faiss_index.add(np.array(valid_embeddings).astype('float32'))

    @classmethod
    async def create(cls, chunks: List[DocumentChunk]):
        retriever = cls(chunks)
        await retriever._build_indices_async()
        return retriever

    def _search_dense(self, query: str, k: int) -> Dict[str, float]:
        query_embedding_result = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=query,
                                                     task_type="RETRIEVAL_QUERY")
        query_embedding = np.array([query_embedding_result['embedding']]).astype('float32')
        _, indices = self.faiss_index.search(query_embedding, k)
        results = {}
        for idx in indices[0]:
            if idx != -1:
                original_chunk_index = self.chunk_ids[idx]
                chunk_id = self.chunks[original_chunk_index].chunk_id
                results[chunk_id] = 1.0 / (len(results) + 1)
        return results

    def _search_sparse(self, query: str, k: int) -> Dict[str, float]:
        tokenized_query = query.split(" ")
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        return {self.bm25_corpus_map[idx].chunk_id: doc_scores[idx] for idx in top_k_indices}

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        # ... This method remains unchanged ...
        fused_scores = {}
        dense_ranks = {doc_id: i + 1 for i, doc_id in enumerate(dense_results.keys())}
        sparse_ranks = {doc_id: i + 1 for i, doc_id in enumerate(sparse_results.keys())}
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        for doc_id in all_doc_ids:
            score = 0
            if dense_rank := dense_ranks.get(doc_id): score += 1 / (k + dense_rank)
            if sparse_rank := sparse_ranks.get(doc_id): score += 1 / (k + sparse_rank)
            fused_scores[doc_id] = score
        return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 10) -> List[DocumentChunk]:
        logger.info(f"Retrieving chunks for query: '{query}'")
        # Step 1: Hybrid Search
        dense_results = self._search_dense(query, k=top_k * 3)  # Retrieve more to give the re-ranker options
        sparse_results = self._search_sparse(query, k=top_k * 3)
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Step 2: Prepare for Re-ranking
        fused_chunks = [self.chunk_map[doc_id] for doc_id, _ in fused_results[:20]]  # Take top 20 for re-ranking
        pairs = [(query, chunk.text) for chunk in fused_chunks]

        # Step 3: Re-rank
        logger.info(f"Re-ranking {len(pairs)} chunks for relevance...")
        scores = self.reranker.predict(pairs)

        # Combine chunks with their new scores and sort
        reranked_results = sorted(zip(fused_chunks, scores), key=lambda x: x[1], reverse=True)

        # Step 4: Return the top k re-ranked chunks
        retrieved_chunks = [chunk for chunk, score in reranked_results[:top_k]]

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks after re-ranking.")
        return retrieved_chunks