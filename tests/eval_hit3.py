"""
eval_hit3.py — Hand-labeled evaluation set with hit@3 per retrieval mode.

Optional extra for strong students (Assignment 2).

Hit@3 definition: a query is a "hit" if at least one of the ground-truth
relevant chunk IDs appears in the top-3 results returned by the retriever.

Run:
    python tests/eval_hit3.py
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval import HybridRetriever

# ── Hand-Labeled Evaluation Set ───────────────────────────────────────────────
# Format: (query, [relevant_chunk_ids], question_type_label)
# Chunk IDs were verified by inspecting chunks.json manually.

EVAL_SET = [
    # ── Personal Fact queries (BM25 should excel) ────────────────────────────
    (
        "What is schedule density blindness in DeadlineOS?",
        ["faq_chunk_2", "projects_chunk_0", "projects_chunk_1"],
        "personal_fact",
    ),
    (
        "What dataset and accuracy did Verdade report?",
        ["faq_chunk_3", "projects_chunk_5", "projects_chunk_6", "projects_chunk_7"],
        "personal_fact",
    ),
    (
        "Where did you complete school and college before UMT?",
        ["faq_chunk_0", "bio_chunk_1", "bio_chunk_2"],
        "personal_fact",
    ),
    (
        "What is the tech stack of DeadlineOS?",
        ["faq_chunk_2", "projects_chunk_2", "cv_chunk_0"],
        "personal_fact",
    ),
    (
        "What certifications do you have?",
        ["faq_chunk_1", "cv_chunk_0"],
        "personal_fact",
    ),
    (
        "What is Arbitrix?",
        ["faq_chunk_4", "cv_chunk_0"],
        "personal_fact",
    ),
    # ── Opinion / personality queries (Dense should excel) ───────────────────
    (
        "How do you spend your leisure time preparing tasty dishes?",
        ["faq_chunk_5", "bio_chunk_0", "opinions_chunk_0"],
        "opinion",
    ),
    (
        "What are your hobbies and how do you like to spend free time?",
        ["faq_chunk_5", "faq_chunk_7", "opinions_chunk_0", "bio_chunk_0"],
        "opinion",
    ),
    (
        "Why did you choose Computer Science?",
        ["faq_chunk_1", "bio_chunk_3", "bio_chunk_4"],
        "opinion",
    ),
    (
        "How do you handle code bugs or unexpected errors?",
        ["faq_chunk_6", "faq_chunk_7", "how_i_work_chunk_0"],
        "opinion",
    ),
    (
        "What do you value most in your work?",
        ["faq_chunk_7", "opinions_chunk_1"],
        "opinion",
    ),
    # ── Mixed / general queries (Hybrid should excel) ────────────────────────
    (
        "Tell me about your projects",
        ["projects_chunk_0", "cv_chunk_0", "faq_chunk_2", "faq_chunk_3"],
        "general",
    ),
    (
        "What is your debugging strategy?",
        ["faq_chunk_7", "how_i_work_chunk_1"],
        "general",
    ),
    (
        "Are you the real Shahrina Khan or an AI clone?",
        ["faq_chunk_0"],
        "general",
    ),
    (
        "What are your future goals in AI?",
        ["future_goals_chunk_0", "faq_chunk_7"],
        "general",
    ),
]


def hit_at_k(results, relevant_ids: list[str], k: int = 3) -> bool:
    """Return True if any relevant chunk appears in the top-k results."""
    top_k_ids = {r.chunk.chunk_id for r in results[:k]}
    return bool(top_k_ids & set(relevant_ids))


def run_evaluation(top_k: int = 3) -> None:
    print("=" * 70)
    print(f"Hit@{top_k} Evaluation — Shahrina Khan Digital Clone")
    print("=" * 70)

    retriever = HybridRetriever(
        data_dir="data",
        chunks_path="chunks.json",
        embeddings_path="embeddings.npy",
        auto_index=True,
    )

    modes = ["BM25 only", "Dense only", "Hybrid (RRF)"]
    hits: dict[str, int] = {m: 0 for m in modes}
    total = len(EVAL_SET)

    # Per-type tracking
    type_totals: dict[str, int] = {}
    type_hits: dict[str, dict[str, int]] = {}

    print(f"\n{'Query':<50} {'Type':<14} {'BM25':^6} {'Dense':^6} {'Hybrid':^6}")
    print("-" * 86)

    for query, relevant_ids, q_type in EVAL_SET:
        type_totals[q_type] = type_totals.get(q_type, 0) + 1
        if q_type not in type_hits:
            type_hits[q_type] = {m: 0 for m in modes}

        row_hits = {}
        for mode in modes:
            results = retriever.retrieve(query, mode=mode, top_k=top_k)
            h = hit_at_k(results, relevant_ids, k=top_k)
            row_hits[mode] = h
            if h:
                hits[mode] += 1
                type_hits[q_type][mode] += 1

        bm25_sym  = "✓" if row_hits["BM25 only"]    else "✗"
        dense_sym = "✓" if row_hits["Dense only"]   else "✗"
        hyb_sym   = "✓" if row_hits["Hybrid (RRF)"] else "✗"

        q_short = query[:48] + ".." if len(query) > 50 else query
        print(f"{q_short:<50} {q_type:<14} {bm25_sym:^6} {dense_sym:^6} {hyb_sym:^6}")

    print("\n" + "=" * 70)
    print(f"{'OVERALL Hit@' + str(top_k) + ' Summary':}")
    print("=" * 70)
    for mode in modes:
        pct = 100.0 * hits[mode] / total
        bar = "█" * hits[mode] + "░" * (total - hits[mode])
        print(f"  {mode:<16}: {hits[mode]:2}/{total}  ({pct:5.1f}%)  {bar}")

    print("\n── Hit@3 by Question Type ──")
    for q_type, t_total in sorted(type_totals.items()):
        print(f"\n  [{q_type}] ({t_total} queries)")
        for mode in modes:
            h = type_hits[q_type][mode]
            pct = 100.0 * h / t_total
            print(f"    {mode:<16}: {h}/{t_total}  ({pct:.0f}%)")

    print("\n" + "=" * 70)
    best_mode = max(modes, key=lambda m: hits[m])
    print(f"Best overall mode: {best_mode} ({hits[best_mode]}/{total} hits)")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation(top_k=3)
