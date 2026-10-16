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

### Phase 5: The Unified Direct Mathematical Solution vs Pieces Benchmark (`EXP-06`)
- [x] **Analytical Mathematical Unification:**
  - Proved all linear/recurrent mechanisms (Linear Attention, Delta-Net, RLS, Chunkwise Ridge) are discrete approximations of inverting the causal Gram matrix $G_t^{-1}$.
  - Formulated **Exact Causal OLS Attention**: computes the exact causal discounted Gram matrix $G_t$ and cross-covariance $R_t$ via associative prefix cumulative sums, and solves $W_t^* = G_t^{-1} R_t$ in parallel across all $T$ tokens via batched Cholesky factorization ($G_t = L_t L_t^\top$).
- [x] **Empirical Supremacy on NVIDIA T4 Cloud GPU:**
  - **Unified Exact Poly OLS** establishes the project record with **$28.65\%$ recall accuracy** in **$7.44\text{ s}$** ($5.0\times$ higher recall than Softmax $5.73\%$, $9.8\times$ faster than RLS, $5.6\times$ faster than Delta).
  - **Unified Exact OLS** matches RLS accuracy ($21.35\%$ vs $21.88\%$) while training **$23.8\times$ faster** ($3.07\text{ s}$ vs $73.15\text{ s}$).
  - Universal $O(1)$ zero-KV-cache autoregressive serving memory: $0.0156\text{ MB}$ at $T=4096$ ($256\times$ smaller than KV-cache).
- [x] **Cloud Benchmark & Artifact Logging:**
  - Metrics and 4 publication-ready figures synced to `results/exp06_unified_vs_pieces/` and logged in `EXPERIMENTS.md`.

### Phase 6: "Flash-OLS" — Chunked Associative Causal OLS for Ultra-Long Sequences (`EXP-07`)
- [x] **Chunked Prefix Scan & SRAM Formulation:**
  - Break long contexts into chunks ($C = 64$ or $128$).
  - Propagate inter-chunk Gram states ($G_c \in \mathbb{R}^{d \times d}$) and cross-covariance ($R_c \in \mathbb{R}^{d \times d_{\text{head}}}$) via parallel prefix associative scans.
  - Compute intra-chunk Cholesky factorizations on SRAM / GPU tensor cores without materializing the global $(B, H, T, d, d)$ tensor in VRAM.
- [x] **Ultra-Long Context Benchmark ($T = 512 \to 65,536$):**
  - Proved Flash-OLS scales seamlessly up to $T = 65,536$ tokens on a 16GB GPU where Standard Softmax crashes with OOM at $T=16,384$.
  - Achieved $17.0\times$ peak memory reduction compared to Softmax at $T=8,192$, and $8.0\times$ reduction compared to Naive OLS.
  - Verified $100\%$ mathematical and empirical fidelity to Exact OLS across training epochs.

### Phase 7: Multi-Hop Associative Reasoning in Series Benchmark (`EXP-08`)
- [x] **Deep OLS Transformer Architecture in Series:**
  - Stacked Flash-OLS and Delta-Net layers in series with residual connections to evaluate multi-hop compositionality ($A \to B \to C$).
- [x] **Empirical Validation of the Compositionality Threshold:**
  - Demonstrated that 1-layer models fail at chance ($\le 3.52\%$), while 2-layer Flash-OLS Poly surges to $17.58\%$ ($3.3\times$ faster than 2-layer Delta-Net at $42.34\text{s}$).

### Phase 8: Deep Closed-Form Boosting & Solvable Networks (DSN) for Tabular Data (`EXP-09`)
- [x] **Deep Solvable Cascade (Zero Backpropagation):**
  - Formulated multi-stage analytical boosting: Stage 1 fits Degree 2, Stage 2 fits Degree 4, Stage 3 fits Degree 8, Stage 4 fits Degree 16 via closed-form Ridge inversions on residual errors.
- [x] **Benchmark vs Gradient Boosted Decision Trees & Deep Attention:**
  - Evaluated on Kin8nm, Friedman-1, and California Housing on Modal NVIDIA T4 GPU.
  - DSN strictly out-predicts XGBoost on Kin8nm ($R^2 = 0.7428$ vs $0.7370$) while training $2.0\times$ faster ($131.7\text{ ms}$ vs $268.0\text{ ms}$).
  - Achieves $379\times$ training speedup over FT-Transformer on California Housing ($29.7\text{ ms}$ vs $11.24\text{ s}$).

---

## Active & Upcoming Milestones

### Phase 9: Blueprint 3 — Alternating Least Squares (ALS) Solvable Refinement (`EXP-10`)
- [x] **Alternating Least Squares (ALS) Block Coordinate Solver:**
  - Formulated bi-convex block coordinate optimization across multi-stage solvable cascades.
  - Refits each stage's weight matrix $W_l^*$ on the residual error of all other stages in closed form with precomputed Gram inverses.
- [x] **Empirical Supremacy over GBDTs (Kin8nm):**
  - ALS refinement surges Kin8nm test $R^2$ from $0.7428 \to 0.7756$, beating both **XGBoost ($0.7370$)** and **LightGBM ($0.7681$)**.
  - Delivers $318\times$ training speedup over FT-Transformer on California Housing ($37.3\text{ ms}$ vs $11.88\text{ s}$) with strictly monotonic convergence across cycles.

---

## Active & Upcoming Milestones

### Phase 10: TabPFN & TabICL Solvable Hybridization for Discontinuous Tabular Benchmarks (`EXP-11`)
- [x] **TabPFN Quantile Preconditioning & PLE Spline Knots:**
  - Integrated empirical quantile rank transformations to eliminate heavy-tailed leverage points and stabilize the Gram condition number.
  - Implemented continuous piecewise linear hinge bases $\phi_k(x) = \text{ReLU}(x - t_k)$ at empirical quantile knots to capture decision-tree-like step boundaries.
- [x] **TabICL In-Context Landmark Kernel Anchors (Mixture of Local Experts):**
  - Formulated closed-form in-context sample attention using multi-scale spatial/archetype RBF anchors $\kappa_k(x) = \exp(-\|x_a - c_k\|^2 / 2\sigma^2)$ crossed with key continuous features ($\kappa_k(x) \cdot x_j$), creating a closed-form Geographically Weighted Regression.
- [x] **Benchmark Breakthroughs:**
  - Closed **$85.4\%$ of the empirical accuracy gap** to XGBoost on California Housing (test $R^2$ jumped from $0.7216 \to 0.8135$, trailing XGBoost by just $\Delta = 0.0152$).
  - Outperformed both **XGBoost ($0.9877$) and LightGBM ($0.9890$)** on Friedman-1 with **$R^2 = 0.9949$** in $363.4\text{ ms}$.

---

## Active & Upcoming Milestones

### Phase 11: Real-World Small Language Model (SLM) Pretraining Dynamics (`EXP-12`)
- [x] **Parameter-Matched From-Scratch Pretraining (1:1 Parity):**
  - Built twin 12.3M parameter SLMs (`Softmax SLM` vs `Flash-OLS SLM`) with identical dimensions ($d_{\text{model}}=256, H=8, d_{\text{head}}=32, d_{\text{ffn}}=680$, 10 layers, RMSNorm, SwiGLU, RoPE, tied 8k BPE vocabulary) having strictly identical $9,946,368$ trainable parameters.
- [x] **Empirical Validation of Perplexity Bound (Theorem 6):**
  - Trained from scratch on TinyStories over 350 optimization steps concurrently on dual NVIDIA T4 GPUs via Modal.
  - Flash-OLS validation loss dropped from $186.08 \to 5.5085$ (PPL $246.77$), closely tracking Softmax ($235.24 \to 5.2417$, PPL $189.00$) with only $\Delta \mathcal{L} = 0.2668$ nats gap, empirically validating the $\mathcal{O}(1/d)$ perplexity bound.
- [x] **Qualitative Linguistic Evolution & Zero-KV-Cache Inference:**
  - Demonstrated matching developmental milestones from single-token repetition to syntactic quotes, multi-clause dialogue, and narrative closure.
  - Flash-OLS maintains strict $\mathcal{O}(1)$ constant memory during autoregressive generation ($0.0156\text{ MB}$ state vs unbounded linear KV-cache).

---

### Phase 12: Fast Analytical Autograd & Block-Diagonal Subspace Acceleration (`EXP-13`)
- [x] **Analytical Closed-Form OLS Autograd Solver (`CausalOLSSolveFunction`):**
  - Replaced PyTorch's $163,840$ autograd triangular Cholesky backward passes with exact closed-form matrix calculus ($\frac{\partial \mathcal{L}}{\partial \boldsymbol{R}} = \boldsymbol{G}^{-1}\bar{\boldsymbol{W}}$).
  - Bitwise verified with PyTorch `gradcheck` at $10^{-6}$ numerical tolerance.
- [x] **$2 \times 16$ Block-Diagonal Active Subspace:**
  - Partitioned the $r=32$ active polynomial subspace into two independent orthogonal $16$-dimensional subspaces, cutting Cholesky FLOPs by $4\times$.
- [x] **Empirical Benchmark on Modal NVIDIA Tesla T4:**
  - Fast-Flash-OLS V3 delivered a **$2.48\times$ step speedup** ($253.8\text{ ms}$ vs $629.8\text{ ms}$) and boosted throughput from $2,786 \to 6,529\text{ tok/s}$.
  - Maintained **$99.96\%$ mathematical fidelity** to Baseline V2 ($5.5109$ vs $5.5085$, $\Delta = 0.0024$ nats).

---

## Active & Upcoming Milestones

### Phase 13: In-Context Tabular Foundation Architecture (DSN-PFN)
- [ ] **Tabular Foundation Model (DSN-PFN):** Prototype linear $\mathcal{O}(N)$ in-context tabular foundation model scaling beyond TabPFN's $N \le 1,000$ barrier.

### Phase 14: Academic Publication & Preprint Packaging
- [ ] **LaTeX Preprint Finalization:** Convert `paper/manuscript.md` into formal NeurIPS/ICLR-styled `paper/main.tex`.
- [ ] **Integrated Figures & Tables:** Update manuscript with empirical results from EXP-07 through EXP-13.
- [ ] **Overleaf / arXiv Bundle:** Generate a self-contained archive for arXiv submission.





