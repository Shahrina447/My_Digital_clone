"""
generation.py - Grounded LLM Response Generator for Digital Clone RAG.

Assignment Requirements (Criterion 4):
1. First-person persona prompt representing Shahrina Khan.
2. Strict grounding: answers ONLY using retrieved context.
3. If information is missing -> reply exactly: "I don't have that in my notes."
4. Generation cascade:
   - Primary: Mistral API (via mistralai client)
   - Secondary: Hugging Face Inference API (via huggingface_hub.InferenceClient)
   - Last Resort: High-precision extractive text fallback
5. API keys read from environment variables (MISTRAL_API_KEY, HF_TOKEN).
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try importing mistralai
try:
    from mistralai import Mistral as _MistralClient
    _MISTRAL_AVAILABLE = True
except Exception:
    _MistralClient = None  # type: ignore
    _MISTRAL_AVAILABLE = False

# Try importing huggingface_hub
try:
    from huggingface_hub import InferenceClient as _HFInferenceClient
    _HF_AVAILABLE = True
except Exception:
    _HFInferenceClient = None  # type: ignore
    _HF_AVAILABLE = False

from src.retrieval import RetrievalResult

# ── Persona Prompt ────────────────────────────────────────────────────────────
SYSTEM_PERSONA_PROMPT = """You are the AI clone of Shahrina Khan, a final-year BS Computer Science student at UMT Sialkot (Session 2023-2027).
Speak in the first person ("I", "my", "me"), in a natural, polite, and authentic tone.

Rules:
1. Answer ONLY using the CONTEXT provided below. It comes directly from my own documents.
2. If the context does not contain enough information to answer the question, say exactly: "I don't have that in my notes." Do not speculate, guess, or invent any facts about me.
3. Never share phone numbers, home addresses, CNIC, passwords, or private contact details.
4. If asked whether you are the real person, clearly state that you are an AI clone of Shahrina Khan, not the real person.
5. Keep your response direct, clear, and faithful to the source context."""

# Stop words for extractive fallback filtering
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "my", "me", "you", "your", "we", "our", "he", "she", "it", "they",
    "do", "did", "does", "have", "has", "had", "will", "would", "can", "could",
    "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
    "not", "no", "yes", "so", "very", "just", "also", "about", "up", "out",
    "tell", "what", "where", "when", "why", "how", "who", "which",
    "did", "do", "does", "are", "is", "am", "was", "were",
}


def _normalize_for_match(text: str) -> str:
    """Normalize text for exact or near-exact matching against FAQ questions."""
    words = re.findall(r"\w+", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    return " ".join(filtered)


def _content_tokens(text: str) -> set[str]:
    """Extract meaningful content tokens, excluding common stop words."""
    words = re.findall(r"\w+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def greeting_response(query: str) -> Optional[str]:
    """Return a brief welcome for standalone greetings without invoking retrieval."""
    normalized = re.sub(r"[\W_]+", " ", query.casefold()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in {
        "hi",
        "hello",
        "hey",
        "hiya",
        "howdy",
        "hello there",
        "hi there",
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return (
            "Hi! I’m Shahrina Khan’s AI clone. Ask me about my background, "
            "studies, projects, skills, or interests."
        )
    return None


def format_context_for_prompt(results: List[RetrievalResult]) -> str:
    """Format retrieved chunks into a clean, numbered context block."""
    if not results:
        return "No relevant context found in documents."

    blocks = []
    for res in results:
        header = f"[Source: {res.chunk.source} | Chunk: {res.chunk.chunk_id} | Score: {res.score:.4f}]"
        blocks.append(f"{header}\n{res.chunk.text.strip()}")

    return "\n\n---\n\n".join(blocks)


class DigitalCloneGenerator:
    """
    Multi-tier response generator with cascading fallbacks:
    Mistral API -> Hugging Face Inference API -> Extractive Text Fallback.
    """

    def __init__(
        self,
        mistral_model: str = "mistral-small-latest",
        hf_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    ) -> None:
        self.mistral_model = os.environ.get("MISTRAL_MODEL", mistral_model)
        self.hf_model = os.environ.get("HF_MODEL", hf_model)
        self.mistral_key = os.environ.get("MISTRAL_API_KEY")
        self.hf_token = os.environ.get("HF_TOKEN")

        self._mistral_client: Any = None
        self._hf_client: Any = None

    def _get_mistral_client(self) -> Any:
        if self._mistral_client is None and _MISTRAL_AVAILABLE and self.mistral_key:
            try:
                self._mistral_client = _MistralClient(api_key=self.mistral_key)
            except Exception as e:
                print(f"[DigitalCloneGenerator] Mistral client init failed: {e}")
        return self._mistral_client

    def _get_hf_client(self) -> Any:
        if self._hf_client is None and _HF_AVAILABLE and self.hf_token:
            try:
                self._hf_client = _HFInferenceClient(token=self.hf_token)
            except Exception as e:
                print(f"[DigitalCloneGenerator] HF client init failed: {e}")
        return self._hf_client

    def generate(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        history: Optional[List[Dict[str, str]]] = None,
        hf_model: Optional[str] = None,
        all_context: Optional[List[RetrievalResult]] = None,
        temperature: float = 0.2,
        max_tokens: int = 350,
    ) -> Tuple[str, str]:
        """
        Generate grounded response. Returns (answer_text, generated_by_label).
        """
        clean_query = query.strip()
        if not clean_query:
            return "Please enter a question.", "—"

        greeting = greeting_response(clean_query)
        if greeting:
            return greeting, "Greeting"

        if not retrieved_results:
            return "I don't have that in my notes.", "Strict Grounding (No context)"

        context_str = format_context_for_prompt(retrieved_results)

        # Build prompt messages
        user_message_content = (
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION:\n{clean_query}\n\n"
            "Answer in the first person as Shahrina Khan using only the context above:"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PERSONA_PROMPT}]

        # Append recent session conversation history if provided (last 2 user/assistant turns)
        if history:
            recent_turns = history[-4:]
            for turn in recent_turns:
                role = turn.get("role")
                content = turn.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Finally, append current grounded turn
        messages.append({"role": "user", "content": user_message_content})

        # ── Tier 1: Mistral API ───────────────────────────────────────────────
        mistral_client = self._get_mistral_client()
        if mistral_client is not None:
            try:
                resp = mistral_client.chat.complete(
                    model=self.mistral_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = resp.choices[0].message.content
                if isinstance(answer, str) and answer.strip():
                    return answer.strip(), f"Mistral API · `{self.mistral_model}`"
            except Exception as exc:
                print(f"[DigitalCloneGenerator] Mistral API error: {exc}")

        # ── Tier 2: Hugging Face Inference API ────────────────────────────────
        hf_client = self._get_hf_client()
        target_hf_model = hf_model or self.hf_model
        if hf_client is not None:
            try:
                resp = hf_client.chat.completions.create(
                    model=target_hf_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = resp.choices[0].message.content
                if isinstance(answer, str) and answer.strip():
                    short_name = target_hf_model.split("/")[-1]
                    return answer.strip(), f"HF Inference API · `{short_name}`"
            except Exception as exc:
                print(f"[DigitalCloneGenerator] HF Inference error: {exc}")

        # ── Tier 3: High-Precision Extractive Fallback ────────────────────────
        search_space = all_context or retrieved_results
        fallback_answer = self._extract_fallback(clean_query, retrieved_results, search_space)
        return fallback_answer, "Extracted from notes (Offline Fallback)"

    def _extract_fallback(
        self,
        query: str,
        results: List[RetrievalResult],
        search_space: Optional[List[RetrievalResult]] = None,
    ) -> str:
        """
        Grounded rule-based text extraction when no API connection is available.
        """
        q_norm = re.sub(r"[^a-z0-9\s]", " ", query.lower())
        q_norm = re.sub(r"\s+", " ", q_norm).strip()
        q_norm = q_norm.replace("bugs", "errors").replace("bug", "error")
        q_norm = q_norm.replace("goals", "future work aspirations").replace("goal", "future work")
        q_norm = q_norm.replace("leisure", "free time hobbies").replace("dishes", "cooking food")

        # Handle clone identity queries directly according to Rule 4
        if any(phrase in q_norm for phrase in ["are you real", "real shahrina", "real person", "ai clone", "are you an ai", "are you a clone"]):
            return "I am an AI clone of Shahrina Khan, not the real person. I answer only from Shahrina's own notes and documents. For anything important, please contact the real Shahrina directly."

        q_content = _content_tokens(q_norm)
        source_results = search_space or results

        # Off-topic or empty content words
        if not q_content:
            return "I don't have that in my notes."

        # Check if query content words exist at all in any retrieved chunk
        all_words: set[str] = set()
        for res in source_results:
            all_words.update(re.findall(r"\w+", res.chunk.text.lower()))

        if not (q_content & all_words):
            return "I don't have that in my notes."

        min_overlap = 1 if len(q_content) <= 2 else 2

        # 1. Search for matching Q&A in FAQ chunks with a strong exact-question preference.
        qa_matches: List[Tuple[int, str]] = []
        for res in source_results:
            pairs = re.findall(
                r"\*{0,2}Q:\s*(.*?)\?[\*]{0,2}\s*\n?\s*A:\s*(.*?)(?=\*{0,2}Q:|\Z)",
                res.chunk.text,
                flags=re.DOTALL,
            )
            for q_text, a_text in pairs:
                a_clean = a_text.strip().split("\n\n")[0].strip()
                a_clean = re.sub(r"\*+$", "", a_clean).strip()
                q_text_norm = _normalize_for_match(q_text)
                query_norm = _normalize_for_match(q_norm)
                q_overlap = len(q_content & _content_tokens(q_text))
                a_overlap = len(q_content & _content_tokens(a_clean))

                if q_text_norm and query_norm:
                    if q_text_norm == query_norm or query_norm in q_text_norm or q_text_norm in query_norm:
                        q_overlap += 10

                score = q_overlap + (a_overlap // 2)
                if score >= min_overlap or (
                    query_norm and q_text_norm and (
                        query_norm in q_text_norm or q_text_norm in query_norm
                    )
                ):
                    qa_matches.append((score, a_clean))

        if qa_matches:
            return max(qa_matches, key=lambda x: x[0])[1]

        # 2. Search for best matching sentence across retrieved chunks
        sentence_matches: List[Tuple[int, str]] = []
        for res in source_results:
            clean_text = re.sub(r"^#+\s*", "", res.chunk.text, flags=re.MULTILINE)
            clean_text = re.sub(r"\*{0,2}[QA]:\s*", "", clean_text)
            clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", clean_text)
            # Remove markdown bold/italic asterisks
            clean_text = re.sub(r"\*+", "", clean_text)

            for sentence in re.split(r"(?<=[.!?])\s+", clean_text):
                sentence = sentence.strip()
                if not sentence or re.match(r"^(#+|---|\|)", sentence):
                    continue
                # If question prefix remains before '?', take the answer portion
                if "?" in sentence:
                    parts = sentence.split("?", 1)
                    if len(parts) > 1 and len(parts[1].strip()) > 15:
                        sentence = parts[1].strip()

                overlap = len(q_content & _content_tokens(sentence))
                if sentence and overlap >= min_overlap and len(sentence) > 20:
                    sentence_matches.append((overlap, sentence))

        if sentence_matches:
            best_sentence = max(sentence_matches, key=lambda x: x[0])[1]
            return re.sub(r"^\W+", "", best_sentence).strip()

        return "I don't have that in my notes."


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DigitalCloneGenerator...")
    print("=" * 60)

    from src.retrieval import HybridRetriever

    retriever = HybridRetriever(data_dir="data")
    generator = DigitalCloneGenerator()

    test_questions = [
        "What is DeadlineOS and what problem does it solve?",
        "Are you the real Shahrina Khan?",
        "What is your favourite color on Mars?",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        chunks = retriever.retrieve(q, mode="Hybrid (RRF)", top_k=3)
        ans, gen_by = generator.generate(q, chunks)
        print(f"[{gen_by}]")
        print(f"Answer: {ans}")
