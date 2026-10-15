# AI Agent Guidelines & Repository Standards

This repository follows strict organizational and scientific protocols to ensure maximum reproducibility, determinism, and instant retrievability for AI agents (Antigravity, Claude, Cursor, Gemini) and human researchers.

---

## 1. Directory Structure Standards

All code, data, and artifacts must strictly adhere to the following hierarchy:

- **`src/` (Core Library):**
  - Contains pure, reusable mathematical and neural modules (`kernels.py`, `models.py`, `metrics.py`, `data_loader.py`, `tasks.py`).
  - No executable evaluation code or `main()` runners belong here.
- **`experiments/` (Runners & Entrypoints):**
  - Contains all executable benchmark scripts and Modal cloud runners (`run_basis_approx.py`, `run_convergence.py`, `modal_*.py`).
  - Scripts must import from `src` (e.g., `from src.models import ...`).
- **`results/<exp_id>/` (Structured Artifacts):**
  - Every experiment run must have its own isolated subfolder (e.g., `results/exp01_basis_approximation/`, `results/exp02_convergence_benchmark/`).
  - Every experiment subfolder must contain:
    - `metrics.json`: Exact numeric metrics (loss curves, $R^2$, Frobenius errors, wall-clock times).
    - Publication-ready plots (`.png`).
    - Optional `manifest.json`: Execution metadata (hardware, seed, commit hash).
  - **Never dump raw output files directly into the root `results/` folder.**
- **`docs/` (Theory & Literature):**
  - Markdown write-ups detailing mathematical proofs, literature reviews, and cognitive/statistical theories.
- **`paper/` (Preprint & Manuscript):**
  - Academic manuscript drafts (`manuscript.md` or LaTeX `main.tex`) and BibTeX citations (`references.bib`).

---

## 2. Mandatory Experiment Registry (`EXPERIMENTS.md`)

Whenever a new experiment is implemented or executed:
1. **Assign a Unique ID:** `EXP-XX-<slug>` (e.g., `EXP-03-SEQUENCE-ATTENTION`).
2. **Log in `EXPERIMENTS.md`:** You **must** update the master registry in `EXPERIMENTS.md` with:
   - Experiment ID & Title
   - Hypothesis & Mathematical Question
   - Exact Run Command & Compute Target (Local CPU / Modal GPU)
   - Clickable links to `metrics.json` and generated figures
   - Key quantitative takeaways (exact numbers: $R^2$, error %, GPU seconds)

---

## 3. Scientific & LLM Retrieval Principles

- **No Hallucination of Numbers:** When discussing performance or drafting manuscripts, always query `results/<exp_id>/metrics.json` to extract exact numbers. Never fabricate or round without checking the JSON.
- **Seed & Generator Consistency:** Use dedicated PyTorch generators (`torch.Generator().manual_seed(...)`) to decouple synthetic task parameters from dataset sampling.
- **Modal Cloud Discipline:**
  - The local machine has limited RAM and no GPU. Heavy training or GPU matrix decompositions must run on Modal.
  - Always enforce `timeout <= 300` on any Modal cloud function to protect the free tier budget ($30/month).
  - Verify that containers are stopped (`modal app list`) immediately following a run.
