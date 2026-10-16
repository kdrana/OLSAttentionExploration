# Research Roadmap & Strategic Milestones

This roadmap outlines the long-term vision, active implementations, and future research milestones for the **OLS-Attention Exploration** project.

---

## Completed Milestones

### Phase 1: Foundations, Approximation Bounds & Tabular Solvability
- [x] **Analytical Foundations:** Spherical harmonic (Gegenbauer-Bessel) decomposition of Softmax on $\mathbb{S}^{d-1}$ and Taylor polynomial tensor expansions.
- [x] **Empirical Approximation Benchmark (`EXP-01-BASIS`):** Verified on Modal NVIDIA T4 GPU that Degree-2 polynomials achieve $< 0.01\%$ relative Frobenius error at $\beta = 1/\sqrt{d}$, outperforming Positive Random Features (FAVOR+).
- [x] **Tabular Convergence & Solvability Benchmark (`EXP-02-CONV`):**
  - **Gram Precision Metric Warm-Start:** Demonstrated that pre-whitening $W_Q, W_K$ eliminates the cold-start penalty, boosting California Housing $R^2$ by $+0.0226$.
  - **Solvable Degree-2 Regression Block:** Demonstrated that explicit GPU Ridge regression outperforms deep multi-head attention on Kin8nm by $+4.8\%$ ($R^2 = 0.9266$ vs $0.8788$) while running $13\%$ faster.
- [x] **Theoretical Framework:** Documented top-down / bottom-up predictive coding and surprise-gated Kalman filtering ([`docs/PREDICTIVE_CODING_KALMAN_ATTENTION.md`](docs/PREDICTIVE_CODING_KALMAN_ATTENTION.md)).
- [x] **Repository Standardization:** Codified agent guidelines ([`AGENTS.md`](AGENTS.md)), workspace rules ([`GEMINI.md`](GEMINI.md)), and master registry ([`EXPERIMENTS.md`](EXPERIMENTS.md)).
- [x] **Preprint Manuscript Draft v1:** Completed academic report draft in [`paper/manuscript.md`](paper/manuscript.md) and BibTeX citations in [`paper/references.bib`](paper/references.bib).

---

## Active Milestone

### Phase 2: Sequence Modeling with the Two-Stream "Bulk + Margin" Layer (`EXP-03`)
- [x] **Dual-Stream Attention Architecture:**
  - **Stream 1 (Bulk Context):** $O(T)$ linear-time solvable polynomial / Gegenbauer recurrence capturing diffuse background interactions ($\approx 95\%$ of tokens).
  - **Stream 2 (Margin Outliers):** Local-window + sparse top-$k$ Softmax attention routing high-surprise tokens ($\beta \Delta > \ln T$).
  - **Fusion Mechanism:** Adaptive surprise-gated combination.
- [x] **Empirical Validation:**
  - Benchmark on **Multi-Query Associative Recall (MQAR)**: Documented linear attention noise saturation and static landmark masking limitations.
  - Benchmark **Sequence Length Scaling:** Proved Pure Linear Attention achieves $6.6\times$ speedup ($32.5\text{ ms}$ vs $215.5\text{ ms}$) and $11.8\times$ memory reduction ($442.6\text{ MB}$ vs $5210.3\text{ MB}$) at $T=4096$.
- [x] **Cloud Deployment:** Executed on Modal NVIDIA T4 GPU, artifacts logged in `results/exp03_sequence_bulk_margin/` and `EXPERIMENTS.md`.

---

### Phase 3: Causal Recursive Least Squares (RLS) & Kalman State-Space Attention (`EXP-04`)
- [x] **Causal Online RLS & Delta/LMS Architectures:**
  - Formulated autoregressive state updates using the innovation prediction error:
    $$\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{K}_t \left(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}^\top \boldsymbol{k}_t\right)^\top$$
  - Implemented both scalar-gain Delta/LMS and optimal matrix Kalman Gain $K_t$ with Sherman-Morrison precision updates.
- [x] **Associative Recall Breakthrough:**
  - Both Delta ($21.35\%$) and Causal RLS ($15.62\%$) crushed Standard Softmax ($3.12\%$) and Linear Attention ($2.60\%$) on MQAR.
- [x] **Zero-KV-Cache Inference:**
  - Evaluated constant $O(1)$ memory flatline ($0.0078\text{ MB}$ vs $4.0\text{ MB}$ at $T=4096$, a $512\times$ reduction) during continuous autoregressive generation.
- [x] **Cloud Deployment:** Executed on Modal NVIDIA T4 GPU, artifacts logged in `results/exp04_kalman_rls_attention/` and `EXPERIMENTS.md`.

### Phase 4: Solvable Attention Architectures Suite (`EXP-05`)
- [x] **Chunkwise Closed-Form Ridge Attention (`chunk_ridge`):**
  - Solves intra-chunk linear regression in closed form across all chunks in parallel on GPU tensor cores.
  - Achieves **$37.5\times$ training speedup** over Causal RLS ($1.50\text{ s}$ vs $56.23\text{ s}$) while maintaining $14.58\%$ MQAR recall accuracy.
- [x] **Diagonal Kalman Attention (`diag_kalman`):**
  - Formulated $O(d)$ diagonal precision tracking combined with learnable input-dependent write gating.
  - Achieves strictly constant $0.0083\text{ MB}$ memory footprint ($482\times$ smaller than KV-cache at $T=4096$).
- [x] **Chunkwise Degree-2 Polynomial Solvable Attention (`chunk_poly`):**
  - Formulated pairwise degree-2 polynomial expansion resolved via exact closed-form Dual Ridge regression ($C \times C = 16 \times 16$ system).
  - Matches Delta-Net accuracy (**$20.83\%$**, peak **$21.35\%$**) while training **$4.5\times$ faster** ($6.73\text{ s}$ vs $30.57\text{ s}$) and **$8.4\times$ faster** than RLS ($56.23\text{ s}$).
- [x] **Cloud Benchmark & Artifact Logging:**
  - Evaluated 7 architectures on Modal NVIDIA T4 GPU; synced metrics and 4 publication-ready figures to `results/exp05_solvable_attention_suite/` and logged in `EXPERIMENTS.md`.

### Phase 5: Hierarchical Solvable Networks for Large-Scale Tabular Data (`EXP-06`)
- [ ] **Deep Solvable Blocks:** Stack multiple solvable blocks hierarchically (expanding degree-2 on already solved features to capture degree-4 interactions without combinatorial explosion).
- [ ] **Benchmark vs Gradient Boosted Decision Trees:** Evaluate against XGBoost, LightGBM, and CatBoost on large tabular benchmarks (Covertype, Higgs, Adult).

### Phase 6: Academic Publication & Preprint Packaging
- [ ] **LaTeX Preprint Finalization:** Convert `paper/manuscript.md` into formal NeurIPS/ICLR-styled `paper/main.tex`.
- [ ] **Integrated Figures & Tables:** Update manuscript with empirical results from Sequence, Kalman, and Solvable Suite benchmarks.
- [ ] **Overleaf / arXiv Bundle:** Generate a self-contained archive for arXiv submission.

