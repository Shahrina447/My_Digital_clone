"""
generation.py - Grounded LLM Response Generator for Digital Clone RAG.

Assignment Requirements (Criterion 4 & Appendix):
1. First-person persona prompt representing Shahrina Khan.
2. Strict grounding: answers ONLY using the retrieved context.
3. Honest fallback: replies "I don't have that in my notes." if answer is absent.
4. Uses Hugging Face Inference API via huggingface_hub.InferenceClient.
5. Reads secret from HF_TOKEN environment variable (or .env locally).
"""

from __future__ import annotations
import os
from typing import List, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.retrieval import RetrievalResult

# Load local .env if present
load_dotenv()

# System persona prompt customized for Shahrina Khan
SYSTEM_PERSONA_PROMPT = """You are the AI clone of Shahrina Khan, a final-year BS Computer Science student at UMT Sialkot (Session 2023-2027).
Speak in the first person ("I", "my", "me"), in a natural, polite, and authentic tone.

Rules:
1. Answer ONLY using the CONTEXT provided below. It comes directly from my own documents.
2. If the context does not contain enough information to answer the question, say: "I don't have that in my notes." Do not guess, speculate, or invent facts about me.
3. Never share phone numbers, home addresses, CNIC, passwords, or sensitive details.
4. If asked whether you are the real person, clearly state that you are an AI clone of Shahrina Khan, not the real person.
5. Keep your response direct, clear, and faithful to the source context."""


def format_context(retrieved_results: List[RetrievalResult]) -> str:
    """
    Format retrieved chunks into a clean context block with metadata citations.
    """
    if not retrieved_results:
        return "No relevant context found in documents."

    context_blocks = []
    for res in retrieved_results:
        header = f"[Source: {res.chunk.source} | Chunk: {res.chunk.chunk_id} | Score: {res.score:.4f}]"
        context_blocks.append(f"{header}\n{res.chunk.text.strip()}")

    return "\n\n---\n\n".join(context_blocks)


class DigitalCloneGenerator:
    """
    Manages grounded generation using Hugging Face Inference API.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_name = os.environ.get("HF_MODEL", model_name)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.client: Optional[InferenceClient] = None

        if self.hf_token:
            self.client = InferenceClient(token=self.hf_token)
        else:
            print("[DigitalCloneGenerator] Warning: HF_TOKEN not found in environment. Set it in .env or as a Space secret.")

    def generate_response(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a first-person grounded response based strictly on retrieved chunks.
        """
        context_str = format_context(retrieved_results)

        # Check for empty retrieval
        if not retrieved_results:
            return "I don't have that in my notes."

        user_content = f"""CONTEXT:
{context_str}

QUESTION:
{query}

Please answer the question as Shahrina Khan in the first person using only the context above:"""

        messages = [
            {"role": "system", "content": SYSTEM_PERSONA_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # If client is configured, call Hugging Face Inference API
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                answer = response.choices[0].message.content.strip()
                return answer
            except Exception as e:
                print(f"[DigitalCloneGenerator] HF API Error: {e}")
                # If API call fails (rate limit, model loading, etc.), provide clear feedback
                return (
                    f"*(Hugging Face Inference API notice: {e})*\n\n"
                    f"**Extracted Notes Preview:**\n"
                    f"{retrieved_results[0].chunk.text[:300]}..."
                )

        # Fallback if no HF_TOKEN is provided
        return (
            "*(HF_TOKEN not set. To enable live LLM generation, set your HF_TOKEN in Space secrets or a local .env file.)*\n\n"
            f"**Top Retrieved Context ({retrieved_results[0].chunk.source}):**\n"
            f"> {retrieved_results[0].chunk.text[:350]}..."
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DigitalCloneGenerator...")
    print("=" * 60)

    from src.retrieval import HybridRetriever

    retriever = HybridRetriever(data_dir="data", embeddings_path="embeddings.npy")
    generator = DigitalCloneGenerator()

    sample_query = "What is DeadlineOS and what problem does it solve?"
    print(f"\nQuery: {sample_query}")

    results = retriever.retrieve(sample_query, mode="Hybrid", top_k=3)
    response = generator.generate_response(sample_query, results)

    print("\n--- Response ---")
    print(response)
