"""
engine.py - Orchestration Layer for Digital Clone RAG.

Bridges document retrieval (HybridRetriever) and grounded generation (DigitalCloneGenerator).
Keeps app.py focused strictly on the Gradio user interface.

Optional extra implemented: Agentic routing — classifies every query into one of four
categories (personal_fact, opinion, out_of_scope, general) and selects the retrieval mode
most likely to surface a relevant answer, or politely declines out-of-scope questions.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.generation import (
    DigitalCloneGenerator,
    acknowledgment_response,
    greeting_response,
)
from src.retrieval import HybridRetriever, RetrievalResult

DEFAULT_HF_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-3B-Instruct",
]

# ── Question Type Labels ──────────────────────────────────────────────────────
QuestionType = Literal["personal_fact", "opinion", "out_of_scope", "general"]


def classify_question(query: str) -> QuestionType:
    """
    Agentic routing step: classifies the query into one of four categories
    and returns the best retrieval strategy.

    Categories:
      personal_fact  — exact biographical / project data (benefits from BM25)
      opinion        — values, preferences, feelings (benefits from Dense)
      out_of_scope   — clearly unrelated to the clone owner
      general        — mixed / uncertain (use Hybrid RRF as default)

    Rule-based classifier using keyword sets. Fast, deterministic, zero-latency.
    """
    q = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    tokens = set(re.findall(r"\w+", q))

    # ── Out-of-scope signals ──────────────────────────────────────────────────
    out_of_scope_keywords = {
        # News / current events
        "stock", "crypto", "bitcoin", "price", "weather", "news", "election",
        "president", "prime", "minister", "government", "war", "attack",
        # Medicine / legal / finance advice
        "doctor", "medicine", "diagnosis", "symptom", "lawyer", "lawsuit",
        "invest", "loan", "mortgage",
        # Completely unrelated personal topics
        "recipe", "restaurant", "movie", "actor", "singer", "sport", "football",
        "cricket", "match", "score", "celebrity",
        # Harmful / irrelevant
        "killer", "murder", "crime", "weapon", "drug", "hack", "password",
    }
    # Whitelist: these overlap with out-of-scope words but are legit for Shahrina
    whitelist = {"score", "match"}  # e.g. "accuracy score", "pattern match"

    ooc_hits = (tokens & out_of_scope_keywords) - whitelist
    if ooc_hits:
        return "out_of_scope"

    # ── Personal fact signals (exact data → BM25 advantage) ──────────────────
    personal_fact_keywords = {
        # Education
        "school", "college", "university", "umt", "kips", "matriculation",
        "marks", "semester", "degree", "bscs", "cgpa", "result",
        # Projects (rare keywords → BM25 wins)
        "deadlineos", "focusos", "verdade", "sachai", "arbitrix",
        "xlm", "roberta", "xlmroberta", "accuracy", "dataset", "samples",
        "schedule", "density", "blindness", "rescue", "center", "digital", "twin",
        "pgvector", "neon", "supabase", "fastapi", "flask", "sqlalchemy",
        # CV items
        "certification", "hec", "generative", "agentic", "n8n", "linkedin",
        "github", "daska", "sialkot", "pakistan",
    }
    if tokens & personal_fact_keywords:
        return "personal_fact"

    # ── Opinion / personality signals (paraphrase → Dense advantage) ─────────
    opinion_keywords = {
        "feel", "think", "believe", "opinion", "prefer", "like", "love",
        "enjoy", "value", "mindset", "approach", "philosophy", "advice",
        "suggest", "recommend", "favorite", "favourite", "hobby", "hobbies",
        "interest", "passion", "dream", "goal", "aspire", "aspiration",
        "future", "plan", "want", "wish", "hope",
        "personality", "character", "describe", "yourself",
        "leisure", "relax", "cook", "cooking", "travel", "travelling",
        "weekend", "free", "time", "spend",
    }
    if tokens & opinion_keywords:
        return "opinion"

    return "general"


def route_mode(question_type: QuestionType, user_mode: str) -> str:
    """
    Select the best retrieval mode given the question type.
    If the user has manually chosen a mode other than the default Hybrid,
    respect their explicit choice.
    """
    if user_mode != "Hybrid (RRF)":
        # User explicitly selected a mode — honour it
        return user_mode

    if question_type == "personal_fact":
        return "BM25 only"   # exact keywords → BM25 precision
    elif question_type == "opinion":
        return "Dense only"  # semantic paraphrase → Dense recall
    else:
        return "Hybrid (RRF)"  # mixed / general → fuse both


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
        Execute end-to-end RAG pipeline with agentic routing:
        0. Classify question type (personal_fact / opinion / out_of_scope / general)
        1. Route to best retrieval mode (or decline if out_of_scope)
        2. Retrieve top-k chunks
        3. Generate grounded first-person response with fallback cascade
        4. Build 3-mode comparison table
        5. Format retrieved chunks for the UI sources panel
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "Please enter a question to ask Shahrina's digital clone.",
                "generated_by": "—",
                "mode": mode,
                "question_type": "—",
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
                "question_type": "greeting",
                "chunks": [],
                "chunks_formatted": "*Greeting received; no retrieval needed.*",
                "comparison_table": "*No retrieval comparison for a greeting.*",
            }

        acknowledgment = acknowledgment_response(clean_query)
        if acknowledgment:
            return {
                "query": clean_query,
                "answer": acknowledgment,
                "generated_by": "Conversation",
                "mode": mode,
                "question_type": "acknowledgment",
                "chunks": [],
                "chunks_formatted": "*No retrieval needed for this conversational reply.*",
                "comparison_table": "*No retrieval comparison for this conversational reply.*",
            }

        # ── Agentic Routing Step ──────────────────────────────────────────────
        question_type = classify_question(clean_query)

        if question_type == "out_of_scope":
            return {
                "query": clean_query,
                "answer": (
                    "That question is outside the scope of what I can answer as Shahrina Khan's "
                    "digital clone. I only answer questions about Shahrina's background, education, "
                    "projects, skills, hobbies, and opinions based on her own documents. "
                    "Is there something about me you'd like to know?"
                ),
                "generated_by": "Agentic Router (Out-of-Scope Decline)",
                "mode": mode,
                "question_type": "out_of_scope",
                "chunks": [],
                "chunks_formatted": "*Out-of-scope query declined by agentic router. No retrieval performed.*",
                "comparison_table": "*No comparison for out-of-scope query.*",
            }

        # Route to the best retrieval mode for this question type
        routed_mode = route_mode(question_type, mode)

        # Normalise "tell me about X" → "what is X" so BM25 retrieves the right FAQ chunk
        retrieval_query = re.sub(
            r"^(tell me about|describe|explain|give me info on|what do you know about)\s+",
            "what is ",
            clean_query,
            flags=re.IGNORECASE,
        )

        # 1. Retrieve top-k chunks using the routed mode
        chunks = self.retriever.retrieve(retrieval_query, mode=routed_mode, top_k=int(top_k))

        # 2. Generate answer
        answer, generated_by = self.generator.generate(
            query=retrieval_query,
            retrieved_results=chunks,
            history=history,
            hf_model=hf_model,
        )

        # 3. Format chunks for UI display
        chunks_formatted = self.format_chunks_markdown(chunks)

        # 4. Generate 3-mode comparison table
        comparison_table = self.format_comparison_table(clean_query, top_k=int(top_k))

        return {
            "query": clean_query,
            "answer": answer,
            "generated_by": generated_by,
            "mode": routed_mode,
            "question_type": question_type,
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
            # Strip markdown formatting so it renders as plain text in the card
            text_snippet = re.sub(r"^#+\s*", "", text_snippet, flags=re.MULTILINE)  # headings
            text_snippet = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text_snippet)      # bold/italic
            text_snippet = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text_snippet)        # underscores
            text_snippet = re.sub(r"`{1,3}.*?`{1,3}", "", text_snippet)             # inline code
            text_snippet = " ".join(text_snippet.split())                            # collapse whitespace
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
