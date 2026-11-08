"""
retrieval.py - Hybrid Retrieval Engine for Digital Clone RAG.

Assignment Requirements (Criteria 2 & 3):
1. BM25 Retriever: Tokenize chunks and score with rank_bm25.BM25Okapi.
2. Dense Retriever: Compute embeddings with sentence-transformers (all-MiniLM-L6-v2)
   and cosine similarity with numpy. Precompute and save embeddings as a .npy file.
3. Hybrid Fusion: Reciprocal Rank Fusion (RRF) with score = sum(1 / (60 + rank))
   to visibly balance keyword and semantic search.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunking import DocumentChunk, load_and_chunk_documents


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int
    retriever: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "text": self.chunk.text,
            "word_count": self.chunk.word_count,
            "score": round(self.score, 4),
            "rank": self.rank,
            "retriever": self.retriever,
        }


def simple_tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization for BM25."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """
    Dual-engine retriever implementing BM25, Dense semantic search, and RRF fusion.
    """

    def __init__(
        self,
        data_dir: str = "data",
        embeddings_path: str = "embeddings.npy",
        model_name: str = "all-MiniLM-L6-v2",
        auto_index: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_embeddings: Optional[np.ndarray] = None

        if auto_index:
            self.index()

    def _get_model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self.model is None:
            try:
                # Try loading from local Hugging Face cache first for fast offline startup
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                # Fall back to online check if not yet cached
                self.model = SentenceTransformer(self.model_name)
        return self.model

    def index(self, force_recompute_embeddings: bool = False) -> None:
        """
        Load chunks, build the BM25 index, and compute or load dense embeddings.
        """
        print(f"[HybridRetriever] Loading and chunking documents from '{self.data_dir}'...")
        self.chunks = load_and_chunk_documents(data_dir=self.data_dir)
        print(f"[HybridRetriever] Loaded {len(self.chunks)} chunks across {len(set(c.source for c in self.chunks))} documents.")

        # 1. Build BM25 index
        tokenized_corpus = [simple_tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("[HybridRetriever] BM25 index built successfully.")

        # 2. Build or load Dense embeddings (.npy)
        if self.embeddings_path.exists() and not force_recompute_embeddings:
            print(f"[HybridRetriever] Loading precomputed embeddings from '{self.embeddings_path}'...")
            embeddings = np.load(self.embeddings_path)
            # Verify shape consistency with current chunks count
            if len(embeddings) == len(self.chunks):
                self.chunk_embeddings = embeddings
                print(f"[HybridRetriever] Successfully loaded embeddings with shape {self.chunk_embeddings.shape}.")
                return
            else:
                print(f"[HybridRetriever] Embedding count ({len(embeddings)}) mismatch with chunks ({len(self.chunks)}). Recomputing...")

        # Compute embeddings
        print(f"[HybridRetriever] Computing embeddings using '{self.model_name}'...")
        model = self._get_model()
        chunk_texts = [chunk.text for chunk in self.chunks]
        embeddings = model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
        self.chunk_embeddings = np.array(embeddings, dtype=np.float32)

        # Save embeddings as .npy as required by the assignment
        np.save(self.embeddings_path, self.chunk_embeddings)
        print(f"[HybridRetriever] Saved precomputed embeddings to '{self.embeddings_path}' (Shape: {self.chunk_embeddings.shape}).")

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        BM25 keyword search.
        """
        if self.bm25 is None or not self.chunks:
            raise RuntimeError("Retriever is not indexed. Call index() first.")

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(raw_scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(raw_scores[idx]),
                    rank=rank,
                    retriever="BM25",
                )
            )
        return results

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Dense semantic search using normalized cosine similarity in numpy.
        """
        if self.chunk_embeddings is None or not self.chunks:
            raise RuntimeError("Retriever embeddings not initialized. Call index() first.")

        model = self._get_model()
        query_embedding = model.encode(query, normalize_embeddings=True)
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity for unit-normalized vectors is the dot product
        similarities = np.dot(self.chunk_embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx]),
                    rank=rank,
                    retriever="Dense",
                )
            )
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20,
        rrf_k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Hybrid fusion using Reciprocal Rank Fusion (RRF).
        Formula: score = sum(1 / (rrf_k + rank))
        """
        pool_size = max(top_k, candidate_pool)
        bm25_results = self.retrieve_bm25(query, top_k=pool_size)
        dense_results = self.retrieve_dense(query, top_k=pool_size)

        # Map chunk_id to DocumentChunk and calculate RRF scores
        chunk_map: dict[str, DocumentChunk] = {}
        rrf_scores: dict[str, float] = {}

        for res in bm25_results:
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + res.rank))

        for res in dense_results:
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + res.rank))

        # Sort chunks by final RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=float(rrf_scores[cid]),
                    rank=rank,
                    retriever="Hybrid (RRF)",
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        mode: Literal["BM25 only", "Dense only", "Hybrid"] = "Hybrid",
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Unified dispatch method for Gradio UI and generation pipeline.
        """
        normalized_mode = mode.strip().lower()
        if "bm25" in normalized_mode:
            return self.retrieve_bm25(query, top_k=top_k)
        elif "dense" in normalized_mode:
            return self.retrieve_dense(query, top_k=top_k)
        else:
            return self.retrieve_hybrid(query, top_k=top_k)


if __name__ == "__main__":
    print("=" * 65)
    print("Initializing and Testing HybridRetriever...")
    print("=" * 65)

    retriever = HybridRetriever(data_dir="data", embeddings_path="embeddings.npy")

    # Test Queries covering assignment requirements:
    # 1. Rare keyword query (BM25 should excel)
    # 2. Paraphrased query with no shared words (Dense should excel)
    # 3. Hybrid balance test
    test_queries = [
        ("Rare Keyword", "What is schedule density blindness in DeadlineOS?"),
        ("Paraphrased (No shared words)", "How do you enjoy your downtime preparing tasty dishes?"),
        ("Technical Query", "What machine learning models were used in Verdade for Urdu news?"),
    ]

    for label, query in test_queries:
        print("\n" + "-" * 65)
        print(f"[{label}] Query: '{query}'")
        print("-" * 65)

        for mode in ["BM25 only", "Dense only", "Hybrid"]:
            res = retriever.retrieve(query, mode=mode, top_k=2)
            print(f"  Mode: {mode:<12}")
            for r in res:
                print(f"    #{r.rank} [{r.score:.4f}] ({r.chunk.source} | {r.chunk.chunk_id}): {r.chunk.text[:90]}...")
