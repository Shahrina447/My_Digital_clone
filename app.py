"""
app.py - Clean, Professional Gradio Web Application for Shahrina Khan's AI Digital Clone.

Design:
- Collapsible Sidebar with Retrieval Settings & Profile Info.
- Clean, focused Chat & Answer interface.
- Visible AI Clone Disclaimer Banner (Rubric Criterion 5).
- Expandable Retrieved Chunks & Score Inspector.
"""

from __future__ import annotations

try:
    import spaces as _spaces
except ModuleNotFoundError as exc:
    if exc.name != "spaces":
        raise
    _spaces = None

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
import gradio as gr

# Try loading .env locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SYSTEM_PERSONA_PROMPT = """You are the AI clone of Shahrina Khan, a final-year BS Computer Science student at UMT Sialkot (Session 2023-2027).
Speak in the first person ("I", "my", "me"), in a natural, polite, and authentic tone.

Rules:
1. Answer ONLY using the CONTEXT provided below. It comes directly from my own documents.
2. If the context does not contain enough information to answer the question, say: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, home addresses, CNIC, passwords, or sensitive details.
4. If asked whether you are the real person, clearly state that you are an AI clone of Shahrina Khan, not the real person.
5. Keep your response direct, clear, and faithful to the source context."""


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric word tokenization."""
    return re.findall(r"\w+", text.lower())


class CloneEngine:
    """Core RAG engine managing BM25, dense embeddings, RRF hybrid fusion, and generation."""

    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.dense_model = None

        self._load_knowledge()

    def _load_knowledge(self) -> None:
        # 1. Load Chunks
        chunks_file = Path("chunks.json")
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            from src.chunking import load_and_chunk_documents
            self.chunks = [c.to_dict() for c in load_and_chunk_documents("data")]

        # 2. Build BM25
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Load Embeddings
        embeddings_file = Path("embeddings.npy")
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)

        # 4. Optional local sentence-transformers loader
        try:
            from sentence_transformers import SentenceTransformer
            self.dense_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            self.dense_model = None

    def _get_query_embedding(self, query: str) -> np.ndarray:
        if self.dense_model is not None:
            vec = self.dense_model.encode(query, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)

        q_tokens = set(tokenize(query))
        weights = np.zeros(len(self.chunks), dtype=np.float32)
        for i, c in enumerate(self.chunks):
            c_tokens = set(tokenize(c["text"]))
            overlap = len(q_tokens.intersection(c_tokens))
            weights[i] = overlap / max(len(q_tokens), 1)

        if np.all(weights == 0):
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)

        pseudo_vec = np.dot(weights, self.embeddings)
        norm = np.linalg.norm(pseudo_vec)
        if norm > 0:
            pseudo_vec = pseudo_vec / norm
        return pseudo_vec

    def retrieve(self, query: str, mode: str = "Hybrid", top_k: int = 3) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        top_k = int(top_k)
        candidate_n = max(top_k * 3, 15)

        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:candidate_n]

        if self.embeddings is not None:
            q_vec = self._get_query_embedding(query)
            dense_scores = np.dot(self.embeddings, q_vec)
            dense_ranked_indices = np.argsort(dense_scores)[::-1][:candidate_n]
        else:
            dense_scores = bm25_scores
            dense_ranked_indices = bm25_ranked_indices

        norm_mode = mode.strip().lower()

        if "bm25" in norm_mode:
            results = []
            for rank, idx in enumerate(bm25_ranked_indices[:top_k], start=1):
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(bm25_scores[idx])
                chunk["rank"] = rank
                chunk["retriever"] = "BM25"
                results.append(chunk)
            return results

        if "dense" in norm_mode:
            results = []
            for rank, idx in enumerate(dense_ranked_indices[:top_k], start=1):
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(dense_scores[idx])
                chunk["rank"] = rank
                chunk["retriever"] = "Dense"
                results.append(chunk)
            return results

        # Hybrid RRF: sum of 1 / (60 + rank)
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60 + rank))

        for rank, idx in enumerate(dense_ranked_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60 + rank))

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(sorted_indices, start=1):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(rrf_scores[idx])
            chunk["rank"] = rank
            chunk["retriever"] = "Hybrid (RRF)"
            results.append(chunk)
        return results

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Call Mistral AI with grounded context and persona prompt."""
        if not chunks:
            return "I don't have that in my notes."

        context_str = "\n\n---\n\n".join(
            [f"[Source: {c['source']} | Chunk: {c['chunk_id']}]\n{c['text']}" for c in chunks]
        )

        mistral_key = os.environ.get("MISTRAL_API_KEY")
        if mistral_key:
            try:
                import json, urllib.request
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {mistral_key}",
                    "Content-Type": "application/json",
                }
                messages = [
                    {"role": "system", "content": SYSTEM_PERSONA_PROMPT},
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}\n\nAnswer in first person as Shahrina Khan using only the context above:",
                    },
                ]
                payload = {
                    "model": "open-mistral-7b",
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 400,
                }
                req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode())
                    content = data["choices"][0]["message"]["content"].strip()
                    if content:
                        return content
            except Exception as e:
                # Fallback to local extraction if offline
                pass

        # Grounded extraction fallback
        top = chunks[0]
        text = top["text"].strip()
        q_tokens = set(tokenize(query))
        qa_pairs = re.findall(r"\*\*Q:\s*(.*?)\?\*\*\s*A:\s*(.*?)(?=\*\*Q:|\Z)", text, flags=re.DOTALL)
        if qa_pairs:
            best_answer = None
            max_q_overlap = 0
            for q_text, a_text in qa_pairs:
                overlap = len(q_tokens.intersection(set(tokenize(q_text))))
                if overlap > max_q_overlap:
                    max_q_overlap = overlap
                    best_answer = a_text.strip()
            if best_answer and max_q_overlap > 0:
                return best_answer

        clean_passage = re.sub(r"^#+\s*.*?\n", "", text).strip()
        if len(clean_passage) > 450:
            clean_passage = clean_passage[:450].rsplit(".", 1)[0] + "."

        return clean_passage


engine = CloneEngine()


def format_chunks_markdown(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "*No relevant passages retrieved for this query.*"

    cards = []
    for c in chunks:
        cards.append(
            f"**Rank #{c['rank']}** — `{c['retriever']}` (Score: `{c['score']:.4f}`)\n"
            f"- **Source:** `{c['source']}` | **ID:** `{c['chunk_id']}`\n"
            f"> {c['text']}\n"
        )
    return "\n---\n".join(cards)


def handle_query(query: str, mode: str, top_k: int) -> Tuple[str, str]:
    cleaned = query.strip()
    if not cleaned:
        return "Please type a question to ask my digital clone.", "*No query entered.*"

    retrieved = engine.retrieve(cleaned, mode=mode, top_k=top_k)
    answer = engine.generate(cleaned, retrieved)
    inspector = format_chunks_markdown(retrieved)

    return answer, inspector


if _spaces is not None:
    handle_query = _spaces.GPU(handle_query)


# Custom clean & professional theme
professional_theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Shahrina Khan | AI Digital Clone") as demo:
    # ------------------ SIDEBAR (Settings & Profile) ------------------
    with gr.Sidebar(open=True):
        gr.Markdown(
            """
            ### 👤 Shahrina Khan
            **BSCS Student @ UMT Sialkot**<br>
            *Web Developer & Applied AI Enthusiast*

            ---
            """
        )

        gr.Markdown("#### ⚙️ Retrieval Settings")
        mode_choice = gr.Radio(
            choices=["Hybrid", "BM25 only", "Dense only"],
            value="Hybrid",
            label="Retrieval Mode",
            info="Choose search strategy",
        )
        top_k_slider = gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="Passages to Retrieve (Top-K)",
        )

        gr.Markdown(
            """
            ---
            #### 📊 System Overview
            - **RAG Architecture:** Hybrid (BM25 + Dense)
            - **Fusion Method:** Reciprocal Rank Fusion (RRF)
            - **Embedding Model:** `all-MiniLM-L6-v2`
            - **Knowledge Base:** 9 docs (~6,800 words)
            """
        )

        clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

    # ------------------ MAIN CONTENT AREA ------------------
    # Mandatory AI Clone Banner (Rubric Criterion 5)
    gr.Markdown(
        """
        <div style="background-color: #fefce8; border: 1px solid #fef08a; border-left: 5px solid #eab308; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
            <strong style="color: #854d0e; font-size: 0.95rem;">
                ⚠️ AI Clone Notice: This is an AI clone, not the real person.
            </strong>
            <p style="margin: 4px 0 0 0; color: #713f12; font-size: 0.88rem;">
                Answers are generated strictly from Shahrina Khan's verified academic background, projects (<strong>DeadlineOS</strong>, <strong>Verdade</strong>), and technical notes.
            </p>
        </div>
        """
    )

    gr.Markdown(
        """
        # Chat with Shahrina's AI Clone
        Ask me about my degree at UMT Sialkot, projects, developer stack, study habits, cooking recipes, or travels in northern Pakistan.
        """
    )

    query_box = gr.Textbox(
        label="",
        placeholder="Type a question here and press Enter (e.g., What is DeadlineOS and what problem does it solve?)...",
        lines=2,
    )

    with gr.Row():
        ask_btn = gr.Button("Ask Question", variant="primary", scale=2)

    gr.Markdown("#### 💬 Clone Response")
    answer_box = gr.Markdown(
        value="*Your answer will appear here.*"
    )

    with gr.Accordion("🔍 Retrieved Chunks & Score Inspector", open=False):
        gr.Markdown("*Inspect the exact source passages and similarity scores used to ground the answer:*")
        inspector_box = gr.Markdown(value="*No passages retrieved yet.*")

    gr.Markdown("#### 💡 Example Questions")
    gr.Examples(
        examples=[
            ["What is schedule density blindness in DeadlineOS?", "Hybrid", 3],
            ["What dataset and evaluation results did Verdade achieve?", "Hybrid", 3],
            ["How do you enjoy your downtime preparing tasty dishes?", "Dense only", 3],
            ["What are your main technical skills and frontend tools?", "BM25 only", 3],
            ["What advice would you give to a junior computer science student?", "Hybrid", 3],
        ],
        inputs=[query_box, mode_choice, top_k_slider],
    )

    # Wire event handlers
    ask_btn.click(
        fn=handle_query,
        inputs=[query_box, mode_choice, top_k_slider],
        outputs=[answer_box, inspector_box],
    )
    query_box.submit(
        fn=handle_query,
        inputs=[query_box, mode_choice, top_k_slider],
        outputs=[answer_box, inspector_box],
    )
    clear_btn.click(
        fn=lambda: ("", "*Your answer will appear here.*", "*No passages retrieved yet.*"),
        inputs=[],
        outputs=[query_box, answer_box, inspector_box],
    )

if __name__ == "__main__":
    demo.launch(theme=professional_theme)
