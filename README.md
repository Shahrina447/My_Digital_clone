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

# Assignment 2: Build Your Digital Clone (Hybrid RAG)

> **Public Space URL:** [https://huggingface.co/spaces/abdullaheqbal/digital-clone](https://huggingface.co/spaces/abdullaheqbal/digital-clone)<br>
> **Repository:** [https://github.com/Shahrina447/My_Digital_clone](https://github.com/Shahrina447/My_Digital_clone)<br>
> **Student Name:** Shahrina Khan
> **Student ID:** 23018020084
> **Degree Program:** BS Computer Science (7th Semester), University of Management and Technology (UMT), Sialkot
> **Course / Learning Outcome:** CLO-2 (Hybrid RAG Architecture)
> **Submission Deadline:** 10-November-2026

---

## 1. Knowledge Base Description & Rationale

For this digital clone, I curated an **8-document knowledge base (~6,200 words)** written strictly in Markdown (`.md`), completely avoiding sensitive personal information (CNIC, passwords, phone numbers, street addresses).

### Documents Included

| # | File | Type | Description |
|---|------|------|-------------|
| 1 | `bio.md` | Factual Background | Academic trajectory: Education City Secondary School (1029 marks), KIPS College (80%), UMT Sialkot BS CS |
| 2 | `cv.md` | Technical Resume | Skills (Next.js, FastAPI, Python, Supabase, pgvector), certifications (HEC Generative & Agentic Applications) |
| 3 | `opinions.md` | Personality & Values | Personal values, engineering mindset, hobbies (cooking, exploring tech, travelling northern Pakistan) |
| 4 | `projects.md` | Long-Form Project Doc | Deep dive into **DeadlineOS** (schedule density blindness, digital-twin simulation, Rescue Center) and **Verdade** (Urdu fake-news detection using fine-tuned `xlm-roberta-base`, 10,421-sample dataset, 98% accuracy) |
| 5 | `faq.md` | First-Person FAQ | 35+ Q&A pairs capturing authentic speaking tone on background, projects, skills, hobbies, and mindset |
| 6 | `how_i_work.md` | Workflow & Habits | Daily developer setup (Linux, VS Code), atomic prototyping, debugging methodology |
| 7 | `advice_to_juniors.md` | Mentorship | Advice on escaping tutorial hell, reading error logs, mastering fundamentals |
| 8 | `future_goals.md` | Aspirations | Goals around agentic AI, low-resource Urdu NLP, production full-stack systems |

### Why This Combination Works
- **Rare keywords** (`DeadlineOS`, `schedule density blindness`, `Rescue Center`, `Education City Secondary School`, `xlm-roberta-base`) give BM25 high discriminative power.
- **Rich conceptual prose** in `bio.md`, `how_i_work.md`, and `opinions.md` gives the dense semantic model strong context for paraphrased queries.
- The **FAQ** bridges both: exact-match keywords and natural first-person phrasing that mirrors real user questions.

---

## 2. Technical Implementation

### Chunking (Criterion 1)
- **Strategy:** Sliding-window word-level chunking
- **Chunk size:** 250 words (target), within 200–300 word range as required
- **Overlap:** 50 words between consecutive chunks
- **Metadata:** Each chunk stores `chunk_id`, `source` filename, `text`, and `word_count`
- **Result:** 33 chunks across 8 documents, average 229.9 words/chunk

```python
# src/chunking.py — core parameters
chunk_size = 250   # words
overlap    = 50    # words
```

### Retrieval (Criterion 2 & 3)
| Component | Implementation |
|-----------|---------------|
| BM25 | `rank_bm25.BM25Okapi` on word-tokenized chunks |
| Dense | `sentence-transformers/all-MiniLM-L6-v2`, embeddings cached in `embeddings.npy` |
| Fusion | Reciprocal Rank Fusion: $\text{RRF} = \sum \frac{1}{60 + \text{rank}}$ |
| Candidate pool | Top-20 from each retriever before RRF |

### Generation (Criterion 4)
- **Primary LLM:** Mistral AI (`mistral-small-latest`) via Mistral API
- **Fallback:** HF Inference API (`meta-llama/Llama-3.2-3B-Instruct`)
- **Last resort:** Text extraction from retrieved chunks
- **Persona prompt:** First-person, grounded strictly in context, "I don't have that in my notes." fallback
- **Temperature:** 0.2 (factual, low creativity)

### Deployment (Criterion 5)
- **Platform:** Hugging Face Spaces (ZeroGPU — free tier)
- **Framework:** Gradio 6.31.0
- **UI:** Two-column layout — left: question + controls + answer; right: 3-mode comparison table + example chips
- **AI clone disclaimer** banner visible on every page load

---

## 3. Five Test Questions — Retrieval Comparison

Chunks: 250 words, 50-word overlap. Embeddings: `all-MiniLM-L6-v2`. Fusion: RRF with $k=60$.

| # | Question | BM25 Top Chunk | Dense Top Chunk | Hybrid Top Chunk | Best Mode |
|:-:|:---------|:--------------|:----------------|:-----------------|:----------|
| **Q1** *Rare Keyword* | What is schedule density blindness in DeadlineOS? | `faq_chunk_1` (10.76) | `projects_chunk_0` (0.485) | `faq_chunk_1` (0.0325) | **BM25 / Hybrid** — unique term, exact keyword match wins |
| **Q2** *Paraphrased* | How do you spend your leisure time preparing tasty dishes? | `faq_chunk_5` (11.73) | `faq_chunk_5` (0.363) | `faq_chunk_5` (0.0328) | **Dense** — zero keyword overlap; semantic search maps "leisure/tasty dishes" → cooking hobby |
| **Q3** *Project Stats* | What dataset and accuracy did Verdade report? | `projects_chunk_5` (10.82) | `faq_chunk_3` (0.511) | `faq_chunk_3` (0.0325) | **Hybrid** — BM25 found raw stats, Dense found Q&A summary; RRF blends both |
| **Q4** *Background* | Where did you complete school and college before UMT? | `faq_chunk_0` (20.06) | `bio_chunk_1` (0.446) | `faq_chunk_0` (0.0325) | **Hybrid** — BM25 on "education/college"; Dense on chronological phrasing |
| **Q5** *Personal* | What are your hobbies and how do you like to spend free time? | `faq_chunk_5` (11.73) | `bio_chunk_0` (0.412) | `faq_chunk_5` (0.0326) | **Dense / Hybrid** — "free time" is a paraphrase; semantic model retrieves hobbies chunk correctly |

---

## 4. Failure Case Analysis

**Query:** *"Are you the killer?"*

**What happened:** The retriever pulled chunks about debugging, errors, and handling problems (semantic overlap with words like "run", "handle", "issue"). Without a strict fallback threshold, the system returned an unrelated software answer.

**Why it happened:** The query has no meaningful content-word overlap with any knowledge base document, but stop-word-heavy queries fool low-threshold matching.

**Fix applied:** Added stop-word filtering in `_fallback_extract()` — content tokens (non-stop words) must overlap ≥ 2 with the source Q&A. If no meaningful match is found, the system returns:
> *"I don't have that in my notes."*

**Prompt rule (Rule 2):**
> *"If the context does not contain enough information to answer, say exactly: 'I don't have that in my notes.'"*

---

## 5. Proposed Next Improvement

**Cross-Encoder Reranker** — add `cross-encoder/ms-marco-MiniLM-L-6-v2` on top of the RRF candidate pool:

- Currently RRF merges rank positions without reading query and chunk text together.
- A cross-encoder scores `(query, passage)` pairs jointly with full token-level attention, filtering out high-BM25 chunks that are topically irrelevant.
- Expected improvement: fewer hallucination-risk chunks in the prompt context window.

---

## 6. Local Setup

```bash
git clone https://github.com/Shahrina447/My_Digital_clone.git
cd My_Digital_clone

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Embeddings are pre-computed (embeddings.npy committed)
# To recompute:
python src/chunking.py

# Run app
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

Set `MISTRAL_API_KEY` in `.env` for LLM generation:
```
MISTRAL_API_KEY=your_key_here
HF_TOKEN=your_hf_token_here
```
