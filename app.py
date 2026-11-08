"""
app.py - Shahrina Khan's AI Digital Clone
Two-column layout matching the reference design:
  LEFT  : question box + retrieval controls + LLM picker + answer
  RIGHT : same-question three-mode comparison + clickable example chips
  BOTTOM: retrieved chunks table
"""

from __future__ import annotations

# Must be imported FIRST before any ML package (ZeroGPU requirement)
try:
    import spaces
except ImportError:
    # Local dev — create a no-op stand-in
    import types, functools
    def _noop_gpu(fn=None, *, duration=60, **kw):
        if fn is not None:
            return fn
        return lambda f: f
    spaces = types.SimpleNamespace(GPU=_noop_gpu)

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

from huggingface_hub import InferenceClient

try:
    from mistralai import Mistral as _MistralClient
    _mistral_ok = True
except ImportError:
    _mistral_ok = False

# ── Persona Prompt ───────────────────────────────────────────────────────────
SYSTEM_PERSONA_PROMPT = """You are the AI clone of Shahrina Khan, a final-year BS Computer Science student at UMT Sialkot (Session 2023-2027).
Speak in the first person ("I", "my", "me"), in a natural, polite, and authentic tone.

Rules:
1. Answer ONLY using the CONTEXT provided below. It comes directly from my own documents.
2. If the context does not contain enough information to answer, say exactly: "I don't have that in my notes."
3. Never share phone numbers, home addresses, CNIC, or passwords.
4. If asked whether you are the real person, say you are an AI clone of Shahrina Khan, not the real person.
5. Keep your response direct, clear, and under 150 words."""

# Free HF Inference Providers models (new provider-based API)
HF_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-3B-Instruct",
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
        self.hf_token = os.environ.get("HF_TOKEN")
        self.mistral_key = os.environ.get("MISTRAL_API_KEY")
        self._load_knowledge()

    def _get_hf_client(self) -> InferenceClient:
        return InferenceClient(provider="hf-inference", token=self.hf_token)

    def _get_mistral_client(self):
        if _mistral_ok and self.mistral_key:
            return _MistralClient(api_key=self.mistral_key)
        return None

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
                 model: str = "meta-llama/Llama-3.2-3B-Instruct") -> Tuple[str, str]:
        """Returns (answer, generated_by_label).
        Strategy: try Mistral first (reliable), fall back to HF, then text extraction.
        """
        if not chunks:
            return "I don't have that in my notes.", "—"

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']} | {c['chunk_id']}]\n{c['text']}"
            for c in chunks
        )
        user_msg = (
            f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\n"
            "Answer in first person using only the context above:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PERSONA_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        # ── 1. Try Mistral ─────────────────────────────────────────────────
        mistral = self._get_mistral_client()
        if mistral is not None:
            try:
                resp = mistral.chat.complete(
                    model="mistral-small-latest",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=350,
                )
                ans = resp.choices[0].message.content
                if isinstance(ans, str) and ans.strip():
                    return ans.strip(), "Mistral · `mistral-small-latest`"
            except Exception as exc:
                print(f"[Mistral error] {type(exc).__name__}: {str(exc)[:300]}")

        # ── 2. Try HF Inference API ────────────────────────────────────────
        if self.hf_token:
            try:
                client = self._get_hf_client()
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=350,
                )
                ans = resp.choices[0].message.content
                if isinstance(ans, str) and ans.strip():
                    return ans.strip(), f"HF · `{model.split('/')[-1]}`"
            except Exception as exc:
                print(f"[HF error] {type(exc).__name__}: {str(exc)[:300]}")

        # ── 3. Text extraction fallback ────────────────────────────────────
        return self._fallback_extract(query, chunks), "Extracted from notes"

    def _fallback_extract(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        q_tok = {"your" if t == "ur" else t for t in tokenize(query)}

        # ── Try to match a Q&A pair ────────────────────────────────────────
        qa = []
        for ch in chunks:
            # Handle both **Q: ...** and Q: ... formats
            pairs = re.findall(
                r"\*{0,2}Q:\s*(.*?)\?[\*]{0,2}\s*\n?\s*A:\s*(.*?)(?=\*{0,2}Q:|\Z)",
                ch["text"], flags=re.DOTALL
            )
            for q_text, a_text in pairs:
                # Clean up answer — take only up to first blank line
                a_clean = a_text.strip().split("\n\n")[0].strip()
                # Remove any trailing ** or markdown
                a_clean = re.sub(r"\*+$", "", a_clean).strip()
                q_toks = {"your" if t == "ur" else t for t in tokenize(q_text)}
                ov = len(q_tok & q_toks)
                if ov:
                    qa.append((ov, a_clean))

        if qa:
            best_ov, best_a = max(qa, key=lambda x: x[0])
            if best_ov >= min(2, len(q_tok)):
                return best_a

        # ── Fall back to best matching sentence ───────────────────────────
        sents = []
        for ch in chunks:
            # Strip markdown headers and Q: A: lines
            text = re.sub(r"^#+\s*", "", ch["text"], flags=re.MULTILINE)
            text = re.sub(r"\*{0,2}[QA]:\s*", "", text)
            for s in re.split(r"(?<=[.!?])\s+", text):
                s = s.strip()
                ov = len(q_tok & set(tokenize(s)))
                if s and ov and len(s) > 20:
                    sents.append((ov, s))
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
        return ""
    rows = []
    results = {}
    for mode in ("Hybrid (RRF)", "BM25 only", "Dense only"):
        chunks = engine.retrieve(query, mode=mode, top_k=top_k)
        results[mode] = [c["chunk_id"] for c in chunks]

    max_rows = max(len(v) for v in results.values())
    header = "| Rank | Hybrid (RRF) | BM25 only | Dense only |\n|------|-------------|-----------|------------|\n"
    for i in range(max_rows):
        h = results["Hybrid (RRF)"][i] if i < len(results["Hybrid (RRF)"]) else "—"
        b = results["BM25 only"][i]    if i < len(results["BM25 only"])    else "—"
        d = results["Dense only"][i]   if i < len(results["Dense only"])   else "—"
        rows.append(f"| {i+1} | `{h}` | `{b}` | `{d}` |")
    return header + "\n".join(rows)


# ── Main query handler ────────────────────────────────────────────────────────
@spaces.GPU(duration=60)
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
    "What is schedule density blindness in DeadlineOS?",
    "What is the digital-twin feature in DeadlineOS?",
    "What accuracy did Verdade achieve on external tests?",
    "How does Verdade handle uncertainty in predictions?",
    "What is the Rescue Center in DeadlineOS?",
    "What certification do you have from HEC?",
    "What is Arbitrix and how does it use RAG?",
    "What are your hobbies?",
    "Why do you like cooking?",
    "Do you like travelling?",
    "How do you prefer to learn new technologies?",
    "How do you handle code bugs or unexpected errors?",
    "What is your approach to continuous learning?",
    "Are you the real Shahrina?",
]

css = """
/* ─── Reset & base ─────────────────────────────────────── */
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* ─── Banner ────────────────────────────────────────────── */
#banner {
    background: #1a1a1a;
    color: #f5c518;
    border-left: 5px solid #f5c518;
    padding: 11px 18px;
    border-radius: 6px;
    font-size: 0.87rem;
    font-weight: 500;
    margin-bottom: 10px;
}

/* ─── Cards ─────────────────────────────────────────────── */
.card {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 20px;
}

/* ─── Answer output ─────────────────────────────────────── */
#answer-out {
    min-height: 90px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ─── Gen-by label ──────────────────────────────────────── */
#gen-by .prose { color: var(--body-text-color-subdued) !important; font-size: 0.78rem !important; }

/* ─── Example chip buttons — inline wrapping ────────────── */
.chips-wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    margin-top: 6px;
}
.chip button {
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 20px !important;
    padding: 5px 13px !important;
    font-size: 0.80rem !important;
    text-align: left !important;
    color: var(--body-text-color) !important;
    white-space: nowrap !important;
    width: auto !important;
    transition: border-color 0.15s;
}
.chip button:hover {
    border-color: var(--color-accent) !important;
}

/* ─── Chunks section ────────────────────────────────────── */
#chunks-out {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.85rem;
}

/* ─── Compare section ───────────────────────────────────── */
#compare-out {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    min-height: 60px;
}

/* ─── Section headings ──────────────────────────────────── */
.section-head { font-size: 0.8rem !important; font-weight: 600 !important;
                letter-spacing: 0.04em !important;
                color: var(--body-text-color-subdued) !important;
                text-transform: uppercase !important; margin-bottom: 6px !important; }
"""

_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Shahrina Khan | AI Digital Clone") as demo:

    # ── Banner ────────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="banner">⚠️ &nbsp;'
        '<strong>This is an AI clone, not the real person. '
        "It answers only from Shahrina Khan's own notes and can make mistakes."
        "</strong></div>"
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    gr.Markdown(
        "## 🤖 Chat with Shahrina's digital clone\n"
        "<p style='color:var(--body-text-color-subdued);font-size:0.88rem;margin-top:-6px'>"
        "BS Computer Science (7th semester), UMT Sialkot · web dev &amp; applied AI. "
        "Ask about my studies, projects, skills, hobbies or opinions.</p>"
    )

    # ── Two-column body ───────────────────────────────────────────────────────
    with gr.Row(equal_height=False, variant="panel"):

        # ══ LEFT ══════════════════════════════════════════════════════════════
        with gr.Column(scale=55):

            gr.Markdown("**Your question**")
            query_box = gr.Textbox(
                label="",
                placeholder="e.g. Where are you from and what are you studying?",
                lines=2,
                show_label=False,
            )

            mode_radio = gr.Radio(
                choices=["Hybrid (RRF)", "BM25 only", "Dense only"],
                value="Hybrid (RRF)",
                label="Retrieval mode:",
            )

            top_k_slider = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Top-k chunks:",
            )

            model_dropdown = gr.Dropdown(
                choices=HF_MODELS,
                value="meta-llama/Llama-3.2-3B-Instruct",
                label="LLM:",
                interactive=True,
            )

            ask_btn = gr.Button(
                "Ask Shahrina's clone", variant="primary", size="lg",
            )

            gr.HTML("<div class='section-head' style='margin-top:14px'>Answer</div>")
            answer_box = gr.Markdown(
                value="*Your answer will appear here after you ask a question.*",
                elem_id="answer-out",
            )
            gen_by_box = gr.Markdown(
                value="",
                elem_id="gen-by",
            )

        # ══ RIGHT ═════════════════════════════════════════════════════════════
        with gr.Column(scale=45):

            gr.HTML("<div class='section-head'>Same question, all three modes</div>")
            gr.Markdown(
                "<span style='font-size:0.82rem;color:var(--body-text-color-subdued)'>"
                "Ask a question to compare the top-k chunk IDs from BM25, "
                "dense and hybrid retrieval.</span>",
            )
            compare_box = gr.Markdown(
                value="",
                elem_id="compare-out",
            )

            gr.HTML("<hr style='margin:16px 0;opacity:0.2'>")
            gr.HTML("<div class='section-head'>Try an example</div>")

            gr.HTML('<div class="chips-wrap">')
            for ex in EXAMPLES:
                btn = gr.Button(ex, size="sm", elem_classes=["chip"], variant="secondary")
                btn.click(fn=lambda q=ex: q, inputs=[], outputs=[query_box])
            gr.HTML('</div>')

    # ── Retrieved chunks (full-width below) ───────────────────────────────────
    gr.HTML("<div class='section-head' style='margin-top:16px'>Retrieved chunks (source, chunk ID and scores)</div>")
    chunks_box = gr.Markdown(
        value="",
        elem_id="chunks-out",
    )

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
