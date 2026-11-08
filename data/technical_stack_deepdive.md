# Technical Stack Deep Dive – Shahrina Khan

## Why I Prefer My Chosen Technologies
Over the course of my degree and project work, I have gravitated toward a modern, high-velocity stack. Below is my perspective on why I use each component and how they interact in production-style systems.

---

## 1. FastAPI for Python Backends
While Django and Flask are popular in the Python community, **FastAPI** is my preferred framework for AI and API services:
- **Asynchronous Performance:** Native `async/await` support allows handling concurrent LLM and database requests without blocking the event loop.
- **Pydantic Validation:** Type hints provide automatic request validation and clear serialized schemas, preventing subtle data type bugs.
- **Automatic Swagger/OpenAPI Docs:** Developing frontend components is much smoother when you can interactively test endpoints directly in `/docs`.
- **Seamless ML Integration:** Python is the native tongue of modern machine learning; hosting Scikit-learn, PyTorch, or Hugging Face models directly inside FastAPI routes eliminates cross-language serialization overhead.

## 2. Next.js 14 & React for Frontend Development
For web interfaces, **Next.js 14** provides the ideal balance of developer experience and end-user performance:
- **Server Components:** Heavy data fetching and token validation happen on the server, keeping bundle sizes small on mobile devices.
- **App Router:** File-system based routing and nested layouts simplify building dashboards with multi-level sidebars and panels.
- **Tailwind CSS Integration:** Rapid prototyping with utility-first CSS allows quick styling of complex components, including responsive layouts and right-to-left (RTL) Urdu interfaces.

## 3. Supabase & pgvector for Vector Search
When building RAG systems and semantic search applications:
- **Supabase** gives the power of full **PostgreSQL** with real-time subscriptions and row-level security (RLS).
- The **`pgvector`** extension makes it easy to store dense vector embeddings (from models like Gemini `text-embedding-004` or Hugging Face `all-MiniLM-L6-v2`) right next to relational user data in the same database.
- It eliminates the operational complexity of managing a separate, specialized vector database for small to medium projects.

## 4. n8n for Workflow Automation
Automation is critical for streamlining repetitive data workflows:
- I use **n8n** to connect webhooks, databases, notification systems, and generative AI APIs into automated multi-step pipelines.
- Its self-hostable, open-source nature provides complete privacy and flexibility compared to closed commercial alternatives like Zapier.

## 5. Hugging Face Ecosystem & Open-Source AI
- **Transformers & Sentence-Transformers:** The go-to libraries for loading state-of-the-art embedding models and classification backbones (`xlm-roberta-base`, `all-MiniLM-L6-v2`).
- **Hugging Face Spaces & Gradio:** Rapid deployment environments that make it possible to share working AI prototypes with users and evaluators through a single shareable URL without managing complex cloud infrastructure.
