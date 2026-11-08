"""
app.py - Shahrina Khan's AI Digital Clone
Two-column layout matching the reference design:
  LEFT  : question box + retrieval controls + LLM picker + answer
  RIGHT : same-question three-mode comparison + clickable example chips
  BOTTOM: retrieved chunks table
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
import gradio as gr

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from mistralai import Mistral
    _mistral_available = True
except ImportError:
    _mistral_available = False

# ── Persona Prompt ───────────────────────────────────────────────────────────
SYSTEM_PERSONA_PROMPT = """You are the AI clone of Shahrina Khan, a final-year BS Computer Science student at UMT Sialkot (Session 2023-2027).
Speak in the first person ("I", "my", "me"), in a natural, polite, and authentic tone.

Rules:
1. Answer ONLY using the CONTEXT provided below. It comes directly from my own documents.
2. If the context does not contain enough information to answer, say exactly: "I don't have that in my notes."
3. Never share phone numbers, home addresses, CNIC, or passwords.
4. If asked whether you are the real person, say you are an AI clone of Shahrina Khan, not the real person.
5. Keep your response direct, clear, and under 150 words."""

# Available Mistral models for the dropdown
MISTRAL_MODELS = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
]

# ── Tokenizer ─────────────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


# ── RAG Engine ────────────────────────────────────────────────────────────────
class CloneEngine:
    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.dense_model = None
        self.mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        self._mistral_clients: Dict[str, Any] = {}   # cache per model
        self._load_knowledge()

    def _get_client(self, model: str):
        if not _mistral_available or not self.mistral_api_key:
            return None
        if model not in self._mistral_clients:
            self._mistral_clients[model] = Mistral(api_key=self.mistral_api_key)
        return self._mistral_clients[model]

    def _load_knowledge(self) -> None:
        chunks_file = Path("chunks.json")
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            from src.chunking import load_and_chunk_documents
            objs = load_and_chunk_documents("data", chunk_size=250, overlap=50)
            self.chunks = [c.to_dict() for c in objs]
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # BM25 — pure Python, no CUDA, safe at import time
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Load pre-computed embeddings (numpy only — no CUDA at startup)
        emb_file = Path("embeddings.npy")
        if emb_file.exists():
            loaded = np.load(emb_file)
            if len(loaded) == len(self.chunks):
                self.embeddings = loaded

        # NOTE: dense_model is intentionally NOT loaded here.
        # It is lazy-loaded on the first query to avoid CUDA initialising
        # before HF Spaces' `spaces` package has a chance to set up ZeroGPU.

    def _get_dense_model(self):
        """Lazy-load SentenceTransformer on first use (avoids CUDA-at-import crash)."""
        if self.dense_model is not None:
            return self.dense_model
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.dense_model = SentenceTransformer(
                    "all-MiniLM-L6-v2", local_files_only=True
                )
            except Exception:
                self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
            # If embeddings weren't computed yet, compute and cache them now
            if self.embeddings is None:
                vecs = self.dense_model.encode(
                    [c["text"] for c in self.chunks],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.embeddings = np.array(vecs, dtype=np.float32)
                np.save("embeddings.npy", self.embeddings)
        except Exception as e:
            print(f"[CloneEngine] dense model unavailable: {e}")
        return self.dense_model

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def _query_vec(self, query: str) -> np.ndarray:
        model = self._get_dense_model()
        if model is not None:
            v = model.encode(query, normalize_embeddings=True)
            return np.array(v, dtype=np.float32)
        # TF-IDF-style fallback when no dense model available
        q = set(tokenize(query))
        w = np.zeros(len(self.chunks), dtype=np.float32)
        for i, c in enumerate(self.chunks):
            w[i] = len(q & set(tokenize(c["text"]))) / max(len(q), 1)
        if np.all(w == 0):
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)
        v = np.dot(w, self.embeddings)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def retrieve(self, query: str, mode: str = "Hybrid (RRF)",
                 top_k: int = 3) -> List[Dict[str, Any]]:
        q_tok = tokenize(query)
        if not q_tok:
            return []
        top_k = int(top_k)
        pool = max(top_k * 4, 20)

        bm25_scores = self.bm25.get_scores(q_tok)
        bm25_idx = np.argsort(bm25_scores)[::-1][:pool]

        if self.embeddings is not None:
            qv = self._query_vec(query)
            dense_scores = np.dot(self.embeddings, qv)
            dense_idx = np.argsort(dense_scores)[::-1][:pool]
        else:
            dense_scores = bm25_scores
            dense_idx = bm25_idx

        m = mode.strip().lower()

        if "bm25" in m:
            return [
                {**self.chunks[i], "score": float(bm25_scores[i]),
                 "rank": r, "retriever": "BM25"}
                for r, i in enumerate(bm25_idx[:top_k], 1)
            ]
        if "dense" in m:
            return [
                {**self.chunks[i], "score": float(dense_scores[i]),
                 "rank": r, "retriever": "Dense"}
                for r, i in enumerate(dense_idx[:top_k], 1)
            ]

        # Hybrid RRF
        rrf: Dict[int, float] = {}
        for r, i in enumerate(bm25_idx, 1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
        for r, i in enumerate(dense_idx, 1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
        top = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]
        return [
            {**self.chunks[i], "score": float(rrf[i]),
             "rank": r, "retriever": "Hybrid (RRF)"}
            for r, i in enumerate(top, 1)
        ]

    # ── Generation ────────────────────────────────────────────────────────────
    def generate(self, query: str, chunks: List[Dict[str, Any]],
                 model: str = "mistral-small-latest") -> Tuple[str, str]:
        """Returns (answer, generated_by_label)."""
        if not chunks:
            return "I don't have that in my notes.", "—"

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']} | {c['chunk_id']}]\n{c['text']}"
            for c in chunks
        )
        client = self._get_client(model)

        if client is not None:
            try:
                resp = client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PERSONA_PROMPT},
                        {"role": "user", "content":
                            f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\n"
                            "Answer in first person using only the context above:"},
                    ],
                    temperature=0.2,
                    max_tokens=350,
                )
                ans = resp.choices[0].message.content
                if isinstance(ans, str) and ans.strip():
                    return ans.strip(), f"Mistral · `{model}`"
            except Exception as exc:
                print(f"[CloneEngine] Mistral error: {exc}")
                return (
                    f"⚠️ Generation failed ({type(exc).__name__}). "
                    "Check your API key or try again shortly.\n\n"
                    + self._fallback_extract(query, chunks),
                    f"Fallback (error: {type(exc).__name__})"
                )

        # No key configured — extract answer from chunks
        return self._fallback_extract(query, chunks), "Fallback (no API key)"

    def _fallback_extract(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        q_tok = {"your" if t == "ur" else t for t in tokenize(query)}
        qa = []
        for ch in chunks:
            for q_text, a_text in re.findall(
                r"\*\*Q:\s*(.*?)\?\*\*\s*A:\s*(.*?)(?=\*\*Q:|\Z)",
                ch["text"], flags=re.DOTALL
            ):
                ov = len(q_tok & {"your" if t == "ur" else t
                                  for t in tokenize(q_text)})
                if ov:
                    qa.append((ov, a_text.strip()))
        if qa:
            best_ov, best_a = max(qa, key=lambda x: x[0])
            if best_ov >= min(2, len(q_tok)):
                return best_a

        sents = []
        for ch in chunks:
            for s in re.split(r"(?<=[.!?])\s+",
                              re.sub(r"^#+\s*", "", ch["text"].strip())):
                ov = len(q_tok & set(tokenize(s)))
                if s and ov:
                    sents.append((ov, s.strip()))
        if sents:
            return max(sents, key=lambda x: x[0])[1]
        return "I don't have that in my notes."


engine = CloneEngine()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _chunks_table(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "*Nothing retrieved yet.*"
    rows = []
    for c in chunks:
        rows.append(
            f"**#{c['rank']}** &nbsp;`{c['source']}` &nbsp;·&nbsp; "
            f"`{c['chunk_id']}` &nbsp;·&nbsp; "
            f"score `{c['score']:.4f}` &nbsp;·&nbsp; {c['retriever']}\n\n"
            f"> {c['text'][:220].strip()}{'…' if len(c['text']) > 220 else ''}"
        )
    return "\n\n---\n\n".join(rows)


def _compare_modes(query: str, top_k: int) -> str:
    if not query.strip():
        return "*Enter a question on the left to compare modes.*"
    out = []
    for mode in ("BM25 only", "Dense only", "Hybrid (RRF)"):
        chunks = engine.retrieve(query, mode=mode, top_k=top_k)
        ids = ", ".join(f"`{c['chunk_id']}`" for c in chunks) or "—"
        out.append(f"**{mode}:** {ids}")
    return "\n\n".join(out)


# ── Main query handler ────────────────────────────────────────────────────────
def handle_query(
    query: str, mode: str, top_k: int, model: str
) -> Tuple[str, str, str, str]:
    """Returns: answer, generated_by, chunks_md, compare_md"""
    q = query.strip()
    if not q:
        return (
            "*Please enter a question.*",
            "—",
            "*Nothing retrieved yet.*",
            "*Enter a question to compare modes.*",
        )
    chunks = engine.retrieve(q, mode=mode, top_k=top_k)
    answer, gen_by = engine.generate(q, chunks, model=model)
    chunks_md = _chunks_table(chunks)
    compare_md = _compare_modes(q, top_k)
    return answer, f"Generated by — {gen_by}", chunks_md, compare_md


def set_example(q: str) -> str:
    return q


# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Where are you from and what are you studying?",
    "What is DeadlineOS and what problem does it solve?",
    "What advice would you give to a fresher joining computer science?",
    "How do you handle bugs or unexpected errors in code?",
    "What is Verdade and what accuracy did it achieve?",
    "Do you like cooking? What do you enjoy making?",
    "What is Arbitrix and how does it work?",
    "Do you like travelling? Where do you like to go?",
    "What are your future goals?",
    "Are you the real Shahrina?",
]

css = """
/* ── Global ── */
body, .gradio-container { font-family: 'Inter', sans-serif !important; }

/* ── Banner ── */
#banner {
    background: #1c1c1e;
    color: #f5c518;
    border-left: 5px solid #f5c518;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.88rem;
    margin-bottom: 4px;
}

/* ── Title area ── */
#title-area { margin-bottom: 12px; }
#title-area h1 { font-size: 1.6rem; margin-bottom: 2px; }
#title-area p  { color: #666; font-size: 0.88rem; margin: 0; }

/* ── Left card ── */
#left-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px;
}

/* ── Answer box ── */
#answer-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px;
    min-height: 80px;
    font-size: 0.95rem;
}

/* ── Generated-by label ── */
#gen-by { color: #9ca3af; font-size: 0.78rem; margin-top: 4px; }

/* ── Right card ── */
#right-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px;
}

/* ── Example chips ── */
.example-chip button, .example-chip {
    background: #f3f4f6 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 20px !important;
    padding: 4px 14px !important;
    font-size: 0.82rem !important;
    cursor: pointer;
    color: #374151 !important;
    margin: 3px 2px !important;
    transition: background 0.15s;
}
.example-chip button:hover {
    background: #e5e7eb !important;
}

/* ── Chunks section ── */
#chunks-section {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
}

/* ── Radio inline ── */
.retrieval-radio label { font-size: 0.88rem; }

/* ── Ask button ── */
#ask-btn { margin-top: 8px; }
"""

_theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Shahrina Khan | AI Digital Clone") as demo:

    # ── Banner ────────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="banner">'
        '⚠️ &nbsp;<strong>This is an AI clone, not the real person. '
        "It answers only from Shahrina Khan's own notes and can make mistakes."
        "</strong></div>"
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="title-area">'
        "<h1>🤖 Chat with Shahrina's digital clone</h1>"
        "<p>BS Computer Science (7th semester), UMT Sialkot · web dev &amp; applied AI. "
        "Ask about my studies, projects, skills, hobbies or opinions.</p>"
        "</div>"
    )

    # ── Two-column body ───────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── LEFT column ───────────────────────────────────────────────────────
        with gr.Column(scale=6, elem_id="left-card"):
            gr.Markdown("**Your question**")
            query_box = gr.Textbox(
                label="",
                placeholder="e.g. from where did you get education",
                lines=2,
                show_label=False,
            )

            mode_radio = gr.Radio(
                choices=["Hybrid (RRF)", "BM25 only", "Dense only"],
                value="Hybrid (RRF)",
                label="Retrieval mode:",
                elem_classes=["retrieval-radio"],
            )

            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Top-k chunks:",
                    scale=4,
                )

            model_dropdown = gr.Dropdown(
                choices=MISTRAL_MODELS,
                value="mistral-small-latest",
                label="LLM:",
                interactive=True,
            )

            ask_btn = gr.Button(
                "Ask Shahrina's clone", variant="primary",
                elem_id="ask-btn",
            )

            gr.Markdown("**Answer**")
            answer_box = gr.Markdown(
                value="*Your answer will appear here.*",
                elem_id="answer-box",
            )
            gen_by_box = gr.Markdown(
                value="Generated by — ",
                elem_id="gen-by",
            )

        # ── RIGHT column ──────────────────────────────────────────────────────
        with gr.Column(scale=4, elem_id="right-card"):
            gr.Markdown(
                "**Same question, all three modes**\n\n"
                "Ask a question to compare the top-k chunk IDs from BM25, "
                "dense and hybrid retrieval."
            )
            compare_box = gr.Markdown(
                value="*Ask a question to compare modes.*"
            )

            gr.HTML("<hr style='margin:14px 0; border-color:#e5e7eb'>")
            gr.Markdown("**Try an example**")

            # Clickable example chips
            with gr.Column():
                for ex in EXAMPLES:
                    btn = gr.Button(
                        ex, size="sm", elem_classes=["example-chip"],
                        variant="secondary",
                    )
                    btn.click(
                        fn=set_example,
                        inputs=[gr.State(ex)],
                        outputs=[query_box],
                    )

    # ── Retrieved chunks (full width, below both columns) ─────────────────────
    with gr.Row():
        with gr.Column(elem_id="chunks-section"):
            gr.Markdown("### Retrieved chunks (source, chunk ID and scores)")
            chunks_box = gr.Markdown(value="*Nothing retrieved yet.*")

    # ── Wire handlers ─────────────────────────────────────────────────────────
    ask_btn.click(
        fn=handle_query,
        inputs=[query_box, mode_radio, top_k_slider, model_dropdown],
        outputs=[answer_box, gen_by_box, chunks_box, compare_box],
    )
    query_box.submit(
        fn=handle_query,
        inputs=[query_box, mode_radio, top_k_slider, model_dropdown],
        outputs=[answer_box, gen_by_box, chunks_box, compare_box],
    )

if __name__ == "__main__":
    demo.launch(theme=_theme, css=css)
