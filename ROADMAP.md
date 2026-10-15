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
- [ ] **Dual-Stream Attention Architecture:**
  - **Stream 1 (Bulk Context):** $O(T)$ linear-time solvable polynomial / Gegenbauer recurrence capturing diffuse background interactions ($\approx 95\%$ of tokens).
  - **Stream 2 (Margin Outliers):** Local-window + sparse top-$k$ Softmax attention routing high-surprise tokens ($\beta \Delta > \ln T$).
  - **Fusion Mechanism:** Adaptive surprise-gated combination.
- [ ] **Empirical Validation:**
  - Benchmark on **Multi-Query Associative Recall (MQAR)**: Prove the Two-Stream layer solves the capacity saturation flaw of pure linear attention.
  - Benchmark **Sequence Length Scaling:** Measure wall-clock GPU time and peak memory footprint across $T \in [256, 512, 1024, 2048, 4096]$ against standard PyTorch Multi-Head Attention.
- [ ] **Cloud Deployment:** Execute benchmark on Modal NVIDIA T4 GPU, stream artifacts to `results/exp03_sequence_bulk_margin/`, and log in `EXPERIMENTS.md`.

---

## Upcoming Milestones

### Phase 3: Causal Recursive Least Squares (RLS) & Kalman State-Space Attention (`EXP-04`)
- [ ] **Causal Online RLS State Tracking:**
  - Formulate autoregressive generation as an online Kalman filter update:
    $$\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{K}_t \left(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}^\top \boldsymbol{k}_t\right)^\top$$
  - Maintain online precision matrix $P_t$ via Sherman-Morrison rank-1 updates.
- [ ] **Zero-KV-Cache Inference:**
  - Evaluate constant $O(1)$ memory and $O(1)$ latency per token during continuous autoregressive generation.
  - Verify if innovation error correction prevents memory drift in long sequences.

### Phase 4: Hierarchical Solvable Networks for Large-Scale Tabular Data (`EXP-05`)
- [ ] **Deep Solvable Blocks:** Stack multiple solvable blocks hierarchically (expanding degree-2 on already solved features to capture degree-4 interactions without combinatorial explosion).
- [ ] **Benchmark vs Gradient Boosted Decision Trees:** Evaluate against XGBoost, LightGBM, and CatBoost on large tabular benchmarks (Covertype, Higgs, Adult).

### Phase 5: Academic Publication & Preprint Packaging
- [ ] **LaTeX Preprint Finalization:** Convert `paper/manuscript.md` into formal NeurIPS/ICLR-styled `paper/main.tex`.
- [ ] **Integrated Figures & Tables:** Update manuscript with empirical results from Phase 2 (Two-Stream Sequence Layer).
- [ ] **Overleaf / arXiv Bundle:** Generate a self-contained archive for arXiv submission.
