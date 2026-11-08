# How I Work: Developer Routine, Setup, & Study Habits – Shahrina Khan

## Developer Setup & Tools
My development environment is built around fast, lightweight, and modern developer tooling:
- **Operating System:** Linux / Ubuntu environment for native bash tooling, permissions, and containerized workflows.
- **Code Editor:** Visual Studio Code with key extensions:
  - Python & Pylance (strict type checking enabled)
  - Prettier & ESLint for JavaScript and TypeScript formatting
  - GitLens for tracking commit history and file diffs
  - Thunder Client / Postman for testing FastAPI endpoints
- **Terminal:** Bash shell with aliases for quick Git commands and Python virtual environment activation.
- **Package Management:** `pip` and virtual environments (`.venv`) for Python; `npm` / `pnpm` for Next.js and frontend projects.

## Daily Workflow & Habits

### 1. Breaking Down Big Tasks into Atomic Steps
When starting a complex assignment or hackathon project, I never jump straight into writing massive blocks of code. Instead:
- I sketch out the user journey and API contract on paper or a digital note.
- I build the smallest working prototype (a "tracer bullet" implementation) first. For instance, in an AI project, I build the core retrieval function before writing the web interface.
- Once the pipeline works in the terminal or a notebook, I wrap it in clean modules with type annotations and docstrings.

### 2. Git Discipline
- I maintain clean branches for distinct features (`feature/retriever`, `feature/gradio-ui`).
- I commit frequently with descriptive messages describing *what* changed and *why*.
- I strictly ensure `.gitignore` excludes virtual environments (`.venv`), `.env` files with secret API tokens, cache files (`__pycache__`), and large binary artifacts.

### 3. Debugging Strategy
When an application behaves unexpectedly:
1. **Reproduce the bug reliably:** Find the exact minimal input that triggers the issue.
2. **Inspect the data shape:** Check if the dictionary keys, array shapes, or variable types match what the downstream function expects (a common issue in NLP and RAG pipelines).
3. **Isolate external dependencies:** Test API calls independently using curl or Python scripts to determine whether the issue is local code logic or external API rate limits / downtime.

### 4. Continuous Study & Keeping Up with AI
- AI moves fast, so I allocate time each week to read technical release notes, explore GitHub repositories, and review Hugging Face model documentation.
- When a new framework emerges (like FastAPI improvements, Next.js releases, or new embedding models), I prefer building a 100-line prototype rather than just reading articles about it.
