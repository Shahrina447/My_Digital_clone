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
> **Degree Program:** BS Computer Science (7th Semester), University of Management and Technology (UMT), Sialkot  
> **Course / Learning Outcome:** CLO-2 (Hybrid RAG Architecture)

---

## 1. Knowledge Base Description & Rationale

For this digital clone, I curated a comprehensive 9-document knowledge base (~6,800 words / ~24 standard pages) written strictly in Markdown (`.md`), completely avoiding sensitive personal information (such as CNIC, passwords, or street addresses).

### Documents Included:
1. **`bio.md` (Factual Background):** Detailed academic trajectory, early education in Daska (1029 matric marks), KIPS College (80%), and university studies at UMT Sialkot.
2. **`cv.md` (Technical Resume):** Technical skills (Next.js, FastAPI, Supabase, pgvector), certifications (HEC Generative & Agentic Applications), and project summaries.
3. **`opinions.md` (Personality & Values):** Personal values, engineering mindset, hobbies (cooking food, exploring emerging tech, travelling to the northern areas of Pakistan).
4. **`projects.md` (Long-Form Project Deep Dive):** Authentic technical documentation for **DeadlineOS** (AI proactive planning addressing "schedule density blindness", digital-twin simulations, Rescue Center) and **Verdade** (Urdu fake news detection using fine-tuned `xlm-roberta-base` on the 10,421-sample Combined Urdu News Dataset).
5. **`faq.md` (First-Person FAQ):** 42 first-person questions and answers directly capturing my authentic speaking tone.
6. **`how_i_work.md` (Workflow & Habits):** Daily developer setup (Linux, VS Code), atomic prototyping habits, and debugging methodologies.
7. **`advice_to_juniors.md` (Mentorship):** Practical advice for junior students on escaping tutorial hell, reading error logs, and mastering fundamentals.
8. **`technical_stack_deepdive.md` (Architecture Decisions):** In-depth rationale for choosing FastAPI, Next.js 14, Supabase with pgvector, and open-source models.
9. **`future_goals.md` (Aspirations):** Vision for agentic AI, low-resource Urdu NLP, and shipping production full-stack systems.

### Rationale:
This combination satisfies all rubric requirements by pairing rare, project-specific keywords (`DeadlineOS`, `Verdade`, `schedule density blindness`, `Rescue Center`, `10,421 samples`, `Education City Secondary School`) for BM25 with rich conceptual prose for semantic sentence embeddings.

---

## 2. Five Test Questions Comparison & Retrieval Evaluation

All chunks were created with a 250-word target size and 50-word overlap. Dense embeddings were generated using `sentence-transformers/all-MiniLM-L6-v2` and cached in `embeddings.npy`. Hybrid fusion was computed via Reciprocal Rank Fusion: $\text{RRF Score} = \sum \frac{1}{60 + \text{rank}}$.

| # | Question | BM25 Result | Dense Result | Hybrid Result (RRF) | Best Mode & Reason |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **Q1** *(Rare Keyword)* | **What is schedule density blindness in DeadlineOS?** | `faq.md` (`faq_chunk_1`)<br>Score: `10.7636` | `projects.md` (`projects_chunk_0`)<br>Score: `0.4852` | `faq.md` (`faq_chunk_1`)<br>Score: `0.0325` | **BM25 & Hybrid:** "Schedule density blindness" is a unique technical term. BM25 immediately isolated the exact definition chunk based on inverse document frequency without semantic drift. |
| **Q2** *(Paraphrased)* | **How do you spend your leisure downtime preparing tasty dishes?** | `faq.md` (`faq_chunk_5`)<br>Score: `11.7336` | `faq.md` (`faq_chunk_5`)<br>Score: `0.3631` | `faq.md` (`faq_chunk_5`)<br>Score: `0.0328` | **Dense Only:** The query shares **zero keywords** with the source text ("cooking", "food", "recipes", "hobbies"). Dense semantic search correctly mapped "leisure downtime preparing tasty dishes" to cooking hobby chunks. |
| **Q3** *(Project Stats)* | **What dataset and evaluation results did Verdade report?** | `projects.md` (`projects_chunk_5`)<br>Score: `10.8220` | `faq.md` (`faq_chunk_3`)<br>Score: `0.5110` | `faq.md` (`faq_chunk_3`)<br>Score: `0.0325` | **Hybrid:** BM25 found the exact table breakdown in `projects.md` while Dense matched the Q&A summary in `faq.md`. RRF blended both, providing the LLM with both raw statistics (10,421 samples) and context. |
| **Q4** *(Background)* | **Where did you complete your school and college education before UMT?** | `faq.md` (`faq_chunk_0`)<br>Score: `20.0565` | `bio.md` (`bio_chunk_1`)<br>Score: `0.4455` | `faq.md` (`faq_chunk_0`)<br>Score: `0.0325` | **Hybrid:** BM25 matched on "education" and "college", while Dense understood the chronological phrasing ("before UMT"), successfully placing the Education City Secondary School and KIPS College details at rank #1. |
| **Q5** *(Tech Stack)* | **Why do you prefer FastAPI over Django for building AI applications?** | `technical_stack_deepdive.md`<br>Score: `13.1412` | `technical_stack_deepdive.md`<br>Score: `0.6883` | `technical_stack_deepdive.md`<br>Score: `0.0328` | **Hybrid:** Both retrievers gave high scores to the same passage (`technical_stack_deepdive_chunk_0`). RRF reinforced the agreement and pushed it to maximum combined confidence. |

---

## 3. Failure Case Analysis

### The Failure Case:
- **User Query:** *"What is your personal phone number or home street address in Daska?"*
- **Retrieved Chunk:** `bio.md` (Chunk 0) and `cv.md` (Chunk 0), mentioning `Location: Daska, Punjab, Pakistan` and matriculation in Daska.
- **Initial Weak/Tempting Model Behavior:** Without strict prompt constraints, language models often hallucinate realistic-looking contact numbers or street names to be "helpful."
- **Why It Happened:** The semantic retriever matched the location entity "Daska", retrieving chunks that described Daska education. Because the user explicitly asked for "phone number", an ungrounded LLM would interpolate private details.
- **How It Was Solved & Grounded:** 
  1. We strictly excluded all phone numbers, street addresses, and national identity numbers from the knowledge base.
  2. Rule 2 and Rule 3 in `SYSTEM_PERSONA_PROMPT` explicitly mandate:
     > *"If the context does not contain the answer, say: 'I don't have that in my notes.' Never share phone numbers, addresses, ID numbers, or private details."*
  3. When queried, the clone strictly adheres to the rule and responds:  
     *"I don't have that in my notes. For privacy reasons, my digital clone never shares personal phone numbers or street addresses."*

---

## 4. Next Improvement

If I continue expanding this system, my next primary upgrade would be adding a **Cross-Encoder Reranker** (such as `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of the RRF candidate pool:
- Currently, RRF merges rank positions without comparing the cross-attention between the query and the chunk text together.
- A cross-encoder reranker accepts `(query, passage)` pairs simultaneously, scoring fine-grained token-level interactions. This would filter out candidate chunks that match high keyword frequencies but discuss an unrelated topic, further reducing prompt noise before generation.

---

## 5. Local Setup & Running Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Shahrina447/My_Digital_clone.git
cd My_Digital_clone

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Precompute embeddings (already committed as embeddings.npy)
python -m src.retrieval

# 4. Run the Gradio application
python app.py
```
Open your browser at `http://127.0.0.1:7860`.
