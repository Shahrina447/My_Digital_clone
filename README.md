---
title: Shahrina Khan - AI Digital Clone (Hybrid RAG)
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
pinned: false
python_version: "3.12"
app_file: app.py
license: mit
---

# Assignment : Build Your Digital Clone (Hybrid RAG)

> **Public Space URL:** [https://huggingface.co/spaces/abdullaheqbal/digital-clone](https://huggingface.co/spaces/abdullaheqbal/digital-clone)<br>
> **Repository:** [https://github.com/Shahrina447/My_Digital_clone](https://github.com/Shahrina447/My_Digital_clone)<br>
> **Student Name:** Shahrina Khan<br>
> **Student ID:** 23018020084<br>
> **Degree Program:** BS Computer Science (7th Semester), University of Management and Technology (UMT), Sialkot<br>
> **Course / Learning Outcome:** CLO-2 (Hybrid RAG Architecture)<br>
> **Submission Deadline:** 10-November-2026<br>

---

## 1. Knowledge Base Description & Rationale

I curated an **8-document knowledge base (~6,200 words total)** written entirely in Markdown, covering all required document types. No sensitive personal data is included (no CNIC, phone number, home address, passwords, or private messages).

### Documents Included

| # | File | Type | Content |
|---|------|------|---------|
| 1 | `bio.md` | Factual Background | Academic history: Education City Secondary School (1029 marks), KIPS College (80%), UMT Sialkot BS CS 7th semester |
| 2 | `cv.md` | Technical Resume | Skills (Next.js, FastAPI, Python, Supabase, pgvector), 3 featured projects, HEC certification |
| 3 | `opinions.md` | Personality & Values | Hobbies (cooking, travelling northern Pakistan, exploring tech), engineering mindset, learning philosophy |
| 4 | `projects.md` | Long-Form Project Write-up | Deep technical write-up of **DeadlineOS** (schedule density blindness, digital-twin, Rescue Center) and **Verdade** (fine-tuned `xlm-roberta-base`, 10,421-sample Combined Urdu News Dataset, 98% accuracy) |
| 5 | `faq.md` | First-Person FAQ | 35+ Q&A pairs in authentic first-person voice covering background, projects, skills, hobbies, and mindset |
| 6 | `how_i_work.md` | Workflow & Developer Habits | Daily setup (Linux, VS Code), atomic prototyping strategy, debugging methodology |
| 7 | `advice_to_juniors.md` | Mentorship & Advice | 6 pieces of advice on escaping tutorial hell, reading error logs, and building real projects |
| 8 | `future_goals.md` | Aspirations | Goals in agentic AI, low-resource Urdu NLP, production full-stack systems, and open source |

### Why This Combination Works

- **BM25 advantage — rare keywords:** `DeadlineOS`, `schedule density blindness`, `Rescue Center`, `Education City Secondary School`, `xlm-roberta-base`, `pgvector`, `Arbitrix`, `Neon PostgreSQL`. These terms appear in only one or two documents, giving BM25 high discriminative power for exact queries.
- **Dense advantage — rich prose:** `bio.md`, `opinions.md`, and `how_i_work.md` contain descriptive conceptual passages. The sentence-transformer maps paraphrased queries ("leisure time preparing dishes" → cooking hobby) through semantic similarity rather than keyword overlap.
- **FAQ as the bridge:** `faq.md` works for both modes — it contains the same rare project keywords alongside natural conversational phrasing that mirrors real user questions.

---

## 2. Technical Architecture

### Project Structure

```
My_Digital_clone/
├── app.py                  # Gradio chat interface and event handlers (UI only)
├── requirements.txt        # Pinned dependencies for HF Spaces
├── README.md               # This report
├── chunks.json             # 34 precomputed document chunks (cached)
├── embeddings.npy          # Precomputed 384-dimensional MiniLM embeddings (34 × 384)
├── data/                   # 8 Markdown knowledge base documents
└── src/
    ├── __init__.py         # Package exports
    ├── chunking.py         # Sliding-window word-level chunking
    ├── retrieval.py        # BM25 + Dense + RRF hybrid retrieval
    ├── generation.py       # Grounded generation: Mistral → HF API → Extractive fallback
    └── engine.py           # Orchestration layer bridging retrieval and generation
```

### Criterion 1 — Chunking (`src/chunking.py`)

| Parameter | Value |
|-----------|-------|
| Strategy | Sliding-window, word-level |
| Chunk size | 250 words (target) |
| Overlap | 50 words between consecutive chunks |
| Metadata | `chunk_id`, `source` filename, `text`, `word_count` per chunk |
| Result | 34 chunks across 8 documents |

Small trailing fragments (<60 words) are merged into the preceding chunk, capped at 300 words to stay within the assignment range.

### Criterion 2 — BM25 and Dense Retrieval (`src/retrieval.py`)

**BM25:** Uses `rank_bm25.BM25Okapi` with lowercase alphanumeric tokenization. A pure-Python Okapi BM25 fallback is included for environments where `rank_bm25` is unavailable.

**Dense:** Uses `sentence-transformers/all-MiniLM-L6-v2` to encode both chunks and queries as 384-dimensional unit vectors. Similarity is computed with NumPy dot product (equivalent to cosine similarity on normalized vectors). No vector database — raw `embeddings.npy` array only. Embeddings are precomputed once and loaded at startup.

Both modes return `RetrievalResult` objects with `.score`, `.rank`, `.chunk_id`, and `.source`.

### Criterion 3 — Hybrid Fusion: Reciprocal Rank Fusion (`src/retrieval.py`)

$$\text{RRF score} = \sum_{\text{retriever}} \frac{1}{60 + \text{rank}}$$

The constant $k=60$ is taken directly from the assignment specification. BM25 and Dense each contribute candidate pools of 20 chunks. Scores are accumulated per chunk ID across both lists, then sorted descending to produce the final Hybrid (RRF) ranking.

**Alternative method (for reference):** Weighted score fusion $\alpha \cdot s_{\text{dense}} + (1-\alpha) \cdot s_{\text{BM25}}$ after min-max normalization per list. This requires calibrating $\alpha$ and re-normalizing across queries. RRF was chosen because it is parameter-free and robust to score-scale differences between BM25 (integer-range) and dense cosine scores (0–1 range).

### Criterion 4 — Persona and Grounded Generation (`src/generation.py`)

**Persona prompt rules:**
1. Answer **only** using the provided CONTEXT — no external knowledge.
2. If the context does not contain the answer, say exactly: *"I don't have that in my notes."*
3. Never share phone numbers, home addresses, CNIC, passwords, or private contact details.
4. If asked whether you are the real person, state you are an AI clone of Shahrina Khan.
5. Speak in first person ("I", "my", "me") in a natural, polite tone.

**Generation cascade (3 tiers):**

| Tier | Method | Trigger |
|------|--------|---------|
| 1 | Mistral API (`mistral-small-latest`) | Primary, `MISTRAL_API_KEY` set |
| 2 | HF Inference API (user-selected model) | Mistral unavailable or error |
| 3 | Grounded extractive fallback | Both APIs unavailable (offline) |

Temperature is set to 0.2 for factual, low-creativity responses. The last 2 conversation turns are injected as history for short-term memory (optional extra implemented).

### Criterion 5 — Gradio UI (`app.py`)

- **Chat interface:** `gr.Chatbot(type="messages")` with structured content blocks
- **Mode selector:** `gr.Radio` — `Hybrid (RRF)`, `BM25 only`, `Dense only`
- **Retrieved Chunks panel:** Displays source file, chunk ID, retrieval score, and 280-character text preview for each top-k result
- **3-Mode Comparison Table:** Side-by-side BM25 / Dense / Hybrid chunk rankings for every query
- **Generation badge:** Shows which tier produced the answer (Mistral API / HF API / Extractive Fallback)
- **Disclaimer banner:** Permanently visible — *"⚠️ This is an AI clone, not the real person."*
- **ZeroGPU compatibility:** `@spaces.GPU(duration=60)` with safe no-op fallback for local dev

---

## 3. Five Test Questions — Retrieval Comparison

> Chunks: 250 words, 50-word overlap. Embeddings: `all-MiniLM-L6-v2` (384-d). Fusion: RRF with $k=60$. Top-k = 1 shown per mode.

| # | Question | BM25 Top Chunk (score) | Dense Top Chunk (score) | Hybrid Top Chunk (score) | Best Mode & Reasoning |
|:-:|:---------|:-----------------------|:------------------------|:-------------------------|:----------------------|
| **Q1** | *What is schedule density blindness in DeadlineOS?* | `faq_chunk_2` (9.70) | `projects_chunk_0` (0.84) | `projects_chunk_0` (0.033) | **Hybrid** — both retrievers agree on project docs; RRF elevates `projects_chunk_0` which contains the exact definition alongside context. BM25 alone picks a FAQ answer; Hybrid adds the full architectural explanation. |
| **Q2** | *How do you spend your leisure time preparing tasty dishes?* | `faq_chunk_5` (13.37) | `faq_chunk_7` (0.79) | `faq_chunk_5` (0.032) | **BM25 / Hybrid** — the word "dishes" appears directly in the FAQ cooking Q&A. BM25 locks on it immediately. Dense retrieves a different hobby chunk. Hybrid correctly surfaces `faq_chunk_5` (cooking answers) via BM25 dominance in RRF. |
| **Q3** | *What dataset and accuracy did Verdade report?* | `faq_chunk_3` (10.70) | `projects_chunk_7` (0.83) | `faq_chunk_3` (0.032) | **Hybrid** — BM25 matches "accuracy/dataset" keywords in the FAQ. Dense finds the detailed `projects_chunk_7` with numeric scores. Hybrid fuses both so the generated answer can cite the exact figures (98%, 10,421 samples) from the richer chunk. |
| **Q4** | *Where did you complete school and college before UMT?* | `faq_chunk_0` (18.67) | `bio_chunk_5` (0.79) | `faq_chunk_0` (0.032) | **BM25 / Hybrid** — "school", "college", "UMT" all appear directly in `faq_chunk_0`. BM25 gives the highest score of any query tested. Dense fetches a bio chunk. Hybrid correctly keeps the FAQ answer at rank 1. |
| **Q5** | *What are your hobbies and how do you like to spend free time?* | `faq_chunk_5` (18.56) | `faq_chunk_7` (0.79) | `faq_chunk_7` (0.032) | **Dense / Hybrid** — "free time" is a paraphrase; the FAQ uses "weekends" and "leisure". Dense maps the semantic intent to the hobbies chunks. BM25 and Dense converge on the same two FAQ chunks; Hybrid merges them so the top-3 window captures both cooking and travel hobbies. |

---

## 4. Failure Case Analysis

**Query:** *"Are you the killer?"*

**What happened:** With no grounding-relevant content in the knowledge base, the dense retriever returned chunks about debugging strategies and error handling (semantic overlap with "handle", "run", "issue"). The extractive fallback scored sentence fragments with partial word matches and returned a vague software-related answer instead of refusing.

**Why it happened:** The query contains almost no content words after stop-word removal ("killer" only). The dense model found soft semantic similarity to problem-solving language. The fallback threshold was too low — a single content-word overlap was enough to trigger a response.

**Fix applied:** Added two-layer protection in `_extract_fallback()`:
1. Content tokens (non-stop-words) must have **≥ 2 word overlap** with the matched chunk (single-word matches are rejected).
2. The global overlap check compares query content words against **all** words in retrieved chunks — if no intersection exists, the system immediately returns *"I don't have that in my notes."*

**Current behavior:** Any off-topic query with no meaningful keyword overlap now correctly returns:
> *"I don't have that in my notes."*

---

## 5. Optional Extras for Strong Students

Three of the four optional extras are fully implemented.

### ✅ Extra 1 — Agentic Routing (`src/engine.py`)

Every query is classified into one of four types before retrieval runs:

| Question Type | Detection | Routed Mode | Example |
|---|---|---|---|
| `personal_fact` | Keywords: project names, university, marks, tech stack terms | BM25 only | *"What accuracy did Verdade report?"* |
| `opinion` | Keywords: feel, hobby, travel, cooking, future, prefer | Dense only | *"How do you spend your free time?"* |
| `out_of_scope` | Keywords: news, weather, medicine, crime, crypto | **Decline** (no retrieval) | *"What is the weather today?"* |
| `general` | No strong signal | Hybrid (RRF) | *"Tell me about your projects"* |

The routing logic is in `classify_question()` and `route_mode()` in `engine.py`. When the user manually selects a mode (not Hybrid), their choice overrides the router. The question type is shown in the generation badge below every answer.

**Out-of-scope decline response:**
> *"That question is outside the scope of what I can answer as Shahrina Khan's digital clone..."*

### ✅ Extra 2 — Short Conversation Memory (`src/generation.py`)

The last 2 conversation turns (4 messages) are injected into the LLM prompt before the current question. This allows natural follow-up questions like:

> User: *"Tell me about Verdade."*
> User: *"What accuracy did it achieve?"* ← understood as referring to Verdade

Implemented in `DigitalCloneGenerator.generate()` — `history[-4:]` is appended to the message list between the system prompt and the current grounded turn.

### ✅ Extra 3 — Hand-Labeled Evaluation Set with Hit@3 (`tests/eval_hit3.py`)

A 15-query evaluation set with manually verified ground-truth relevant chunk IDs, covering all three question types. Run with `python tests/eval_hit3.py`.

**Results:**

| Mode | Hit@3 | personal_fact (6) | opinion (5) | general (4) |
|------|-------|-------------------|-------------|-------------|
| BM25 only | 13/15 (86.7%) | 6/6 (100%) | 5/5 (100%) | 2/4 (50%) |
| Dense only | 13/15 (86.7%) | 6/6 (100%) | 5/5 (100%) | 2/4 (50%) |
| **Hybrid (RRF)** | **14/15 (93.3%)** | 6/6 (100%) | 5/5 (100%) | **3/4 (75%)** |

**Key findings:**
- Hybrid (RRF) is the best overall mode — it recovers 1 extra hit on "general" queries that neither BM25 nor Dense alone can find.
- BM25 and Dense tie at 86.7% — both are strong on personal_fact and opinion queries but struggle with vague general queries like "Tell me about your projects" (too broad to match a single chunk).
- The one remaining miss across all modes is *"Tell me about your projects"* — this query is too vague to match any single chunk in top-3; the agentic router now routes it through Hybrid which improves coverage.

---

## 6. Proposed Next Improvement

**Cross-Encoder Reranker** using `cross-encoder/ms-marco-MiniLM-L-6-v2` on top of the RRF candidate pool.

**Current limitation:** RRF merges rank positions from BM25 and Dense, but neither retriever jointly reads the query and chunk text together. A chunk can rank highly because it matches keywords *or* shares semantic space — without understanding whether the two are genuinely relevant to each other in context.

**How the improvement works:** A cross-encoder scores each `(query, chunk)` pair jointly using full token-level attention across the concatenated query and passage. This produces a more precise relevance signal than either BM25 scores or cosine similarity alone.

**Expected benefit:** The top-3 chunks passed to the LLM would contain less topically irrelevant text, reducing the chance of a hallucinated or off-topic answer. Particularly useful for queries that contain common words shared across multiple documents (e.g. "What do you think about AI?" matching nearly every document).

**Cost:** One additional forward pass per candidate chunk — acceptable latency on HF Spaces CPU with a small candidate pool of 5–10 chunks.

**Cost:** One additional forward pass per candidate chunk — acceptable latency on HF Spaces CPU with a small candidate pool of 5–10 chunks.

---

## 7. Local Setup

```bash
git clone https://github.com/Shahrina447/My_Digital_clone.git
cd My_Digital_clone

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Embeddings are pre-computed and committed as embeddings.npy
# To recompute from scratch:
python src/chunking.py

# Run app
python app.py
# Open http://127.0.0.1:7860
```

Create a `.env` file (or set Space secrets on HF) with:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
HF_TOKEN=your_huggingface_token_here
```
