"""
engine.py - Orchestration Layer for Digital Clone RAG.

Bridges document retrieval (HybridRetriever) and grounded generation (DigitalCloneGenerator).
Keeps app.py focused strictly on the Gradio user interface.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from src.generation import DigitalCloneGenerator, greeting_response
from src.retrieval import HybridRetriever, RetrievalResult

DEFAULT_HF_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-3B-Instruct",
]


class CloneEngine:
    """
    Central orchestration engine for Shahrina Khan's Digital Clone.
    Coordinates document retrieval, multi-mode comparison, and grounded response generation.
    """

    def __init__(
        self,
        data_dir: str = "data",
        chunks_path: str = "chunks.json",
        embeddings_path: str = "embeddings.npy",
    ) -> None:
        self.retriever = HybridRetriever(
            data_dir=data_dir,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            auto_index=True,
        )
        self.generator = DigitalCloneGenerator()

    def ask(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        mode: str = "Hybrid (RRF)",
        top_k: int = 3,
        hf_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end RAG pipeline:
        1. Retrieve top-k chunks according to selected mode
        2. Generate grounded first-person response with fallback cascade
        3. Build 3-mode comparison table
        4. Format retrieved chunks for the UI sources panel
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "Please enter a question to ask Shahrina's digital clone.",
                "generated_by": "—",
                "mode": mode,
                "chunks": [],
                "chunks_formatted": "*No query provided. Ask a question to view retrieved chunks.*",
                "comparison_table": "*Enter a question to compare retrieval modes.*",
            }

        greeting = greeting_response(clean_query)
        if greeting:
            return {
                "query": clean_query,
                "answer": greeting,
                "generated_by": "Greeting",
                "mode": mode,
                "chunks": [],
                "chunks_formatted": "*Greeting received; no retrieval needed.*",
                "comparison_table": "*No retrieval comparison for a greeting.*",
            }

        # 1. Retrieve top-k chunks using the selected retrieval mode
        chunks = self.retriever.retrieve(clean_query, mode=mode, top_k=int(top_k))
        all_chunks = [
            RetrievalResult(chunk=chunk, score=0.0, rank=i + 1, retriever="Knowledge Base")
            for i, chunk in enumerate(self.retriever.chunks)
        ]

        # 2. Generate answer
        answer, generated_by = self.generator.generate(
            query=clean_query,
            retrieved_results=chunks,
            history=history,
            hf_model=hf_model,
            all_context=all_chunks,
        )

        # 3. Format chunks for UI display
        chunks_formatted = self.format_chunks_markdown(chunks)

        # 4. Generate 3-mode comparison table
        comparison_table = self.format_comparison_table(clean_query, top_k=int(top_k))

        return {
            "query": clean_query,
            "answer": answer,
            "generated_by": generated_by,
            "mode": mode,
            "chunks": chunks,
            "chunks_formatted": chunks_formatted,
            "comparison_table": comparison_table,
        }

    def format_chunks_markdown(self, chunks: List[RetrievalResult]) -> str:
        """Format retrieved chunks into clean markdown cards with metadata and previews."""
        if not chunks:
            return "*No chunks were retrieved for this query.*"

        cards: List[str] = []
        for c in chunks:
            text_snippet = c.chunk.text.strip()
            if len(text_snippet) > 280:
                text_snippet = text_snippet[:280] + "…"

            card = (
                f"**#{c.rank}** &nbsp;`{c.chunk.source}` &nbsp;·&nbsp; "
                f"`{c.chunk.chunk_id}` &nbsp;·&nbsp; "
                f"Score: `{c.score:.4f}` &nbsp;·&nbsp; *{c.retriever}*\n\n"
                f"> {text_snippet}"
            )
            cards.append(card)

        return "\n\n---\n\n".join(cards)

    def format_comparison_table(self, query: str, top_k: int = 3) -> str:
        """
        Build a side-by-side Markdown comparison table across BM25, Dense, and Hybrid retrieval.
        Satisfies assignment criteria for retrieval mode comparison.
        """
        clean_query = query.strip()
        if not clean_query:
            return "*Enter a question to compare retrieval modes.*"

        comparison = self.retriever.compare_modes(clean_query, top_k=top_k)

        # Extract shortened chunk tags (e.g. 'faq_chunk_1' -> 'faq-1 (10.2)')
        results_map: Dict[str, List[str]] = {}
        for m, results in comparison.items():
            formatted_list = []
            for r in results:
                short_id = re.sub(r"_chunk_(\d+)$", r"-\1", r.chunk.chunk_id)
                formatted_list.append(f"`{short_id}` ({r.score:.3f})")
            results_map[m] = formatted_list

        header = "| Rank | Hybrid (RRF) | BM25 only | Dense only |\n|:----:|:-------------|:----------|:-----------|\n"
        rows: List[str] = []
        for i in range(top_k):
            hybrid_val = results_map.get("Hybrid (RRF)", [])
            bm25_val = results_map.get("BM25 only", [])
            dense_val = results_map.get("Dense only", [])

            h_str = hybrid_val[i] if i < len(hybrid_val) else "—"
            b_str = bm25_val[i] if i < len(bm25_val) else "—"
            d_str = dense_val[i] if i < len(dense_val) else "—"

            rows.append(f"| {i + 1} | {h_str} | {b_str} | {d_str} |")

        return header + "\n".join(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CloneEngine Orchestration...")
    print("=" * 60)

    engine = CloneEngine()
    test_q = "What machine learning models were used in Verdade for Urdu news?"

    result = engine.ask(test_q, mode="Hybrid (RRF)", top_k=3)
    print(f"\nQuery: {result['query']}")
    print(f"Generated By: {result['generated_by']}")
    print(f"Answer: {result['answer']}\n")
    print("Retrieved Chunks:")
    print(result["chunks_formatted"])
    print("\nComparison Table:")
    print(result["comparison_table"])
