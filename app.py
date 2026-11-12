"""
app.py - Shahrina Khan's AI Digital Clone (Hybrid RAG)

Gradio Chat Interface and Presentation Layer.
Orchestrates conversation history, retrieval transparency, and grounded response display
using the clean orchestration layer in src.engine.
"""

from __future__ import annotations
import os
import types
from typing import Any, Dict, List, Tuple

# ── ZeroGPU Compatibility (Free Hugging Face CPU / ZeroGPU) ───────────────────
try:
    import spaces
except ImportError:
    # Local development or CPU-only Space — safe no-op fallback
    def _noop_gpu(fn=None, *, duration=60, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f
    spaces = types.SimpleNamespace(GPU=_noop_gpu)

import gradio as gr
from src.engine import CloneEngine, DEFAULT_HF_MODELS

# Initialize clean orchestration engine
engine = CloneEngine(
    data_dir="data",
    chunks_path="chunks.json",
    embeddings_path="embeddings.npy",
)

# Example questions showcasing rare keywords, paraphrasing, project stats, and personal background
EXAMPLE_QUESTIONS = [
    "Where are you from and what are you studying?",
    "What is DeadlineOS and what problem does it solve?",
    "What dataset and accuracy did Verdade report?",
    "What are your hobbies and how do you spend your free time?",
    "Why did you choose Computer Science?",
    "How do you handle code bugs or unexpected errors?",
    "What are your future goals in AI and software engineering?",
    "Are you the real Shahrina Khan or an AI clone?",
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

#banner {
    background: #1e1e24;
    color: #fbbf24;
    border-left: 5px solid #f59e0b;
    padding: 12px 18px;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 500;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

#chatbot {
    border-radius: 12px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-primary);
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

#gen-badge {
    font-size: 0.80rem;
    color: var(--body-text-color-subdued);
    margin-top: 4px;
    padding: 4px 8px;
}

.panel-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 14px;
    font-size: 0.88rem;
    line-height: 1.5;
}

#compare-table table {
    width: 100% !important;
    table-layout: fixed !important;
    font-size: 0.80rem !important;
    border-collapse: collapse;
}

#compare-table td, #compare-table th {
    padding: 6px 8px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.chip-btn {
    border-radius: 20px !important;
    font-size: 0.80rem !important;
    padding: 4px 12px !important;
    margin: 3px !important;
    background: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
    cursor: pointer;
}

.chip-btn:hover {
    border-color: #6366f1 !important;
}
"""

# ── Event Handlers ────────────────────────────────────────────────────────────

def user_submit(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """Appends user message to session chat history and clears the input box."""
    clean_msg = message.strip() if message else ""
    if not clean_msg:
        return "", history or []
    new_history = list(history or []) + [{"role": "user", "content": clean_msg}]
    return "", new_history


@spaces.GPU(duration=60)
def generate_bot_reply(
    history: List[Dict[str, str]],
    mode: str,
    top_k: int,
    hf_model: str,
) -> Tuple[List[Dict[str, str]], str, str, str]:
    """
    Orchestrates grounded retrieval and generation for the latest user turn.
    Returns: (updated_history, chunks_markdown, comparison_table, gen_badge)
    """
    if not history:
        return [], "*No query yet.*", "*Enter a question to compare modes.*", "—"

    latest_user_message = history[-1]["content"]
    previous_history = history[:-1]

    # Execute RAG pipeline through engine
    result = engine.ask(
        query=latest_user_message,
        history=previous_history,
        mode=mode,
        top_k=top_k,
        hf_model=hf_model,
    )

    updated_history = list(history) + [{"role": "assistant", "content": result["answer"]}]
    badge_text = f"✨ Response Source: **{result['generated_by']}** &nbsp;|&nbsp; Retrieval Mode: **{result['mode']}**"

    return (
        updated_history,
        result["chunks_formatted"],
        result["comparison_table"],
        badge_text,
    )


def clear_session() -> Tuple[List[Dict[str, str]], str, str, str, str]:
    """Clears chat history and resets panels for a fresh session."""
    return (
        [],
        "",
        "*Session cleared. Ask a question to view retrieved knowledge chunks.*",
        "*Session cleared.*",
        "—",
    )


def create_chatbot() -> gr.Chatbot:
    """Instantiate Chatbot safely across Gradio versions (4.x, 5.x, and 6.x)."""
    import inspect
    kwargs = {
        "elem_id": "chatbot",
        "height": 480,
        "placeholder": "👋 Hello! I am Shahrina Khan's digital clone. Ask me anything about my academic journey, projects, coding habits, or future aspirations!",
        "show_copy_button": True,
        "bubble_full_width": False,
        "type": "messages",
    }
    sig = inspect.signature(gr.Chatbot.__init__)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return gr.Chatbot(**valid_args)


def send_example_question(
    example_text: str,
    history: List[Dict[str, str]],
    mode: str,
    top_k: int,
    hf_model: str,
) -> Tuple[List[Dict[str, str]], str, str, str]:
    """Directly triggers retrieval and generation for a clicked example question."""
    temp_history = list(history or []) + [{"role": "user", "content": example_text}]
    return generate_bot_reply(temp_history, mode, top_k, hf_model)


# ── Gradio Blocks Application ─────────────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Shahrina Khan | AI Digital Clone (Hybrid RAG)", theme=theme, css=CUSTOM_CSS) as demo:

    # 1. Mandatory Disclaimer Banner
    gr.HTML(
        '<div id="banner">'
        '<span>⚠️</span>'
        '<span><strong>This is an AI clone, not the real person.</strong> '
        "It answers questions in the first person based strictly on Shahrina Khan's own notes and documents."
        '</span></div>'
    )

    # 2. Header & Profile Summary
    with gr.Row():
        with gr.Column(scale=85):
            gr.Markdown(
                "### 🤖 Chat with Shahrina Khan's Digital Clone\n"
                "*BS Computer Science (7th semester), UMT Sialkot · Web Development, Python & Applied AI*\n"
                "Ask me about my background, projects (**DeadlineOS**, **Verdade**, **Arbitrix**), technical skills, or opinions!"
            )
        with gr.Column(scale=15):
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    # 3. Main Interface Layout: Chat (Left) + Transparency Inspector (Right)
    with gr.Row(equal_height=False):

        # ── LEFT: Modern Chatbot & Input ──────────────────────────────────────
        with gr.Column(scale=58):
            chatbot = create_chatbot()

            # Bottom Input Bar (ChatGPT / Grok style)
            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="Ask Shahrina's clone anything (e.g., What is DeadlineOS?)...",
                    show_label=False,
                    lines=1,
                    scale=85,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=15)

            gen_badge_box = gr.Markdown("—", elem_id="gen-badge")

            # Clickable Example Chips
            gr.Markdown("<p style='font-size:0.82rem;font-weight:600;margin-top:8px;margin-bottom:4px;color:var(--body-text-color-subdued);'>TRY ASKING AN EXAMPLE:</p>")
            with gr.Row(wrap=True):
                example_buttons = []
                for ex in EXAMPLE_QUESTIONS:
                    btn = gr.Button(ex, size="sm", elem_classes=["chip-btn"])
                    example_buttons.append((btn, ex))

        # ── RIGHT: Controls & RAG Inspection Panels ───────────────────────────
        with gr.Column(scale=42):
            with gr.Group():
                gr.Markdown("#### ⚙️ Retrieval & Engine Controls")
                mode_selector = gr.Radio(
                    choices=["Hybrid (RRF)", "BM25 only", "Dense only"],
                    value="Hybrid (RRF)",
                    label="Retrieval Mode",
                    info="RRF formula: score = 1 / (60 + rank)",
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Top-k Chunks",
                    )
                    hf_dropdown = gr.Dropdown(
                        choices=DEFAULT_HF_MODELS,
                        value="meta-llama/Llama-3.2-3B-Instruct",
                        label="HF Fallback Model",
                        interactive=True,
                    )

            with gr.Tabs():
                with gr.TabItem("📄 Retrieved Chunks (Sources)"):
                    gr.Markdown("<p style='font-size:0.80rem;color:var(--body-text-color-subdued);margin-bottom:6px;'>Showing source document, chunk ID, retrieval score, and text snippet:</p>")
                    chunks_display = gr.Markdown(
                        value="*Ask a question to view the retrieved document chunks and citation metadata.*",
                        elem_classes=["panel-card"],
                    )

                with gr.TabItem("⚖️ 3-Mode Comparison"):
                    gr.Markdown("<p style='font-size:0.80rem;color:var(--body-text-color-subdued);margin-bottom:6px;'>Side-by-side comparison of top chunk IDs across BM25, Dense, and Hybrid:</p>")
                    comparison_display = gr.Markdown(
                        value="*Ask a question to compare chunk rankings across BM25, Dense, and Hybrid retrieval.*",
                        elem_id="compare-table",
                        elem_classes=["panel-card"],
                    )

                with gr.TabItem("ℹ️ About & Assignment"):
                    gr.Markdown(
                        "**Assignment 2: Build Your Digital Clone**\n"
                        "- **Student:** Shahrina Khan (23018020084)\n"
                        "- **Program:** BS Computer Science (7th Semester), UMT Sialkot\n"
                        "- **Course Outcome:** CLO-2 (Hybrid RAG Architecture)\n"
                        "- **Embeddings:** `all-MiniLM-L6-v2` precomputed in `embeddings.npy`\n"
                        "- **BM25:** Okapi BM25 on word tokens\n"
                        "- **Generation:** Mistral API → HF Inference API → Extractive Notes Fallback\n"
                        "- **Grounding:** Strict first-person persona; answers solely from documents in `data/`"
                    )

    # ── Wire Interactions ─────────────────────────────────────────────────────

    # 1. User submits question via Enter or Send button
    user_step_submit = send_btn.click(
        fn=user_submit,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot],
        queue=False,
    ).then(
        fn=generate_bot_reply,
        inputs=[chatbot, mode_selector, top_k_slider, hf_dropdown],
        outputs=[chatbot, chunks_display, comparison_display, gen_badge_box],
    )

    query_input.submit(
        fn=user_submit,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot],
        queue=False,
    ).then(
        fn=generate_bot_reply,
        inputs=[chatbot, mode_selector, top_k_slider, hf_dropdown],
        outputs=[chatbot, chunks_display, comparison_display, gen_badge_box],
    )

    # 2. Clear conversation button
    clear_btn.click(
        fn=clear_session,
        inputs=[],
        outputs=[chatbot, query_input, chunks_display, comparison_display, gen_badge_box],
        queue=False,
    )

    # 3. Clickable example buttons
    for btn, ex_text in example_buttons:
        btn.click(
            fn=send_example_question,
            inputs=[
                gr.State(ex_text),
                chatbot,
                mode_selector,
                top_k_slider,
                hf_dropdown,
            ],
            outputs=[chatbot, chunks_display, comparison_display, gen_badge_box],
        )

if __name__ == "__main__":
    demo.launch()
