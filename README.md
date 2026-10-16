# OLS & Softmax Attention: Statistical Foundations, Kernel Approximations, and Solvable Architectures

[![Repository Guidelines](https://img.shields.io/badge/Agent%20Standard-AGENTS.md-blue)](AGENTS.md)
[![Experiment Registry](https://img.shields.io/badge/Registry-EXPERIMENTS.md-success)](EXPERIMENTS.md)
[![Theory & Synthesis](https://img.shields.io/badge/Theory%20Docs-docs%2F-purple)](docs/SOLVABLE_ATTENTION_THEORY_AND_BENCHMARKS.md)
[![Paper Manuscript](https://img.shields.io/badge/Manuscript-paper%2Fmanuscript.md-orange)](paper/manuscript.md)

This repository explores the statistical and mathematical continuum bridging **Ordinary Least Squares (OLS)**, **Kernel Ridge Regression**, and **Softmax Attention** (prompted by Philippe Goulet Coulombe, arXiv:2504.09663). 

We investigate whether kernel approximations (Taylor polynomials, Gegenbauer spherical harmonics), online Kalman/RLS error correction, and closed-form linear algebraic solvers can replace or accelerate expensive iterative attention mechanisms.

---

## Key Empirical Findings

All experiments have been benchmarked on cloud GPUs (Modal NVIDIA T4) and logged in [`EXPERIMENTS.md`](EXPERIMENTS.md):

1. **Polynomial Domination in the Weak-Coupling Regime:**
   At standard transformer temperature ($\beta = 1/\sqrt{d}$), a **Degree-2 Taylor polynomial expansion achieves $< 0.01\%$ relative Frobenius error** against exact Softmax attention, outperforming Positive Random Features (FAVOR+) by two orders of magnitude ([EXP-01](EXPERIMENTS.md#exp-01-basis-approximation-and-harmonic-decomposition)).
2. **Gram Precision Metric Warm-Start Boosts Tabular Convergence:**
   Initializing query and key projection weights to the empirical Mahalanobis precision metric $W_Q = W_K = (\frac{1}{N} X^\top X + \lambda \boldsymbol{I})^{-1/2}$ eliminates the cold-start penalty on collinear tabular data, improving test $R^2$ on California Housing by **$+0.0226$** ([EXP-02](EXPERIMENTS.md#exp-02-end-to-end-convergence-and-solvability-benchmark)).
3. **Solvable Regression Blocks Outperform Iterative Attention:**
   On dense non-linear physics (Kin8nm robot arm kinematics), replacing iterative multi-head attention with an explicit Degree-2 cross-feature expansion solved in a single closed-form GPU Ridge step **outperformed standard FT-Transformer by $+4.8\%$ ($R^2 = 0.9266$ vs $0.8788$) while running $13\%$ faster** ([EXP-02](EXPERIMENTS.md#exp-02-end-to-end-convergence-and-solvability-benchmark)).
4. **Pure Linear Attention Suffers Signal Blur on Long Sequences:**
   While Linear Attention achieves a **$6.6\times$ speedup** and **$11.8\times$ memory reduction** at $T=4096$, additive accumulation without non-linear competitive normalization collapses on Multi-Query Associative Recall ($\le 2.66\%$ accuracy) due to distractor noise accumulation ([EXP-03](EXPERIMENTS.md#exp-03-sequence-modeling-and-scaling-benchmark)).
5. **The Innovation Error Solves Associative Recall with Zero-KV-Cache Serving:**
   Using the prediction innovation $e_t = v_t - S_{t-1}^\top k_t$ filters out predictable distractors. Delta/LMS ($21.35\%$) and Causal RLS ($15.62\%$) crush Softmax ($3.12\%$) and Linear Attention ($2.60\%$) on MQAR, while maintaining a strictly constant $O(1)$ state memory ($512\times$ smaller than KV-cache at $T=4096$) ([EXP-04](EXPERIMENTS.md#exp-04-kalman-and-rls-attention-benchmark)).
6. **Chunkwise Closed-Form Solvers Match Recurrent Accuracy at $37.5\times$ Speedup:**
   - **`Chunkwise Poly Solvable`** deploys a degree-2 cross-feature expansion resolved via an exact **$16 \times 16$ Dual Ridge solve**, matching Delta-Net recall accuracy (**$20.83\%$**, peak **$21.35\%$**) while training **$4.5\times$ faster** ($6.73\text{s}$ vs $30.57\text{s}$).
   - **`Chunkwise Ridge`** solves intra-chunk linear regression in parallel across chunks on GPU tensor cores, training **$37.5\times$ faster than Causal RLS** ($1.50\text{s}$ vs $56.23\text{s}$) while achieving $14.58\%$ recall accuracy ([EXP-05](EXPERIMENTS.md#exp-05-solvable-attention-suite-benchmark)).

---

## Repository Structure

```
OLSAttentionExploration/
├── README.md                          # Repository overview & setup
├── AGENTS.md                          # Mandatory AI agent protocols & standards
├── GEMINI.md                          # Antigravity agent configuration
├── EXPERIMENTS.md                     # Master registry of all empirical runs & exact metrics
├── ROADMAP.md                         # Strategic milestones & phases
├── requirements.txt                   # Dependencies
│
├── src/                               # Pure, reusable mathematical & neural modules
│   ├── kernels.py                     # Exact Softmax, Taylor, Gegenbauer, PRF
│   ├── models.py                      # SequenceTransformer, ChunkwiseRidge, DiagKalman, ChunkwisePoly
│   ├── data_loader.py                 # California Housing, Kin8nm, Friedman-1 loaders
│   ├── metrics.py                     # Frobenius error, entropy, effective rank, R²
│   └── tasks.py                       # Synthetic DGPs & MQAR data generators
│
├── experiments/                       # Executable benchmark scripts & cloud runners
│   ├── run_basis_approx.py            # EXP-01: Basis approximation testbed
│   ├── run_convergence.py             # EXP-02: Tabular convergence benchmark
│   ├── run_sequence_benchmark.py      # EXP-03: Sequence scaling benchmark
│   ├── run_kalman_benchmark.py        # EXP-04: Causal RLS & Delta/LMS benchmark
│   ├── run_solvable_suite.py          # EXP-05: 7-model Solvable Attention suite
│   ├── plot_*.py                      # Publication-ready matplotlib scripts
│   └── modal_*_runner.py              # Modal cloud GPU runners (NVIDIA T4)
│
├── results/                           # Self-contained experiment artifacts
│   ├── exp01_basis_approximation/     # Figures 1-4 & metrics.json
│   ├── exp02_convergence_benchmark/   # Figures 5-7 & metrics.json
│   ├── exp03_sequence_bulk_margin/    # Figures 8-10 & metrics.json
│   ├── exp04_kalman_rls_attention/    # Figures 11-13 & metrics.json
│   └── exp05_solvable_attention_suite/# Figures 14-17 & metrics.json
│
├── docs/                              # Theoretical deep-dives & literature notes
│   ├── SOLVABLE_ATTENTION_THEORY_AND_BENCHMARKS.md # Unified theory & master empirical synthesis
│   └── PREDICTIVE_CODING_KALMAN_ATTENTION.md       # Predictive coding & Kalman theory
│
└── paper/                             # Academic manuscript
    ├── manuscript.md                  # Comprehensive preprint draft
    └── references.bib                 # BibTeX references
```

---

## Quickstart

### Local Setup
```bash
# Clone and enter repo
git clone https://github.com/kdrana/OLSAttentionExploration.git
cd OLSAttentionExploration

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Cloud GPU Benchmarks on Modal
All heavy matrix decompositions and GPU sequence benchmarks are packaged as serverless Modal jobs:
```bash
# Authenticate Modal (one-time setup)
modal setup

# Run any benchmark on NVIDIA T4 GPU (auto-stops when finished):
modal run experiments/modal_basis_runner.py       # EXP-01
modal run experiments/modal_convergence_runner.py # EXP-02
modal run experiments/modal_sequence_runner.py    # EXP-03
modal run experiments/modal_kalman_runner.py      # EXP-04
modal run experiments/modal_solvable_runner.py    # EXP-05
```

---

## Master Documentation

- **Unified Theory & Benchmark Synthesis:** [`docs/SOLVABLE_ATTENTION_THEORY_AND_BENCHMARKS.md`](docs/SOLVABLE_ATTENTION_THEORY_AND_BENCHMARKS.md)
- **Predictive Coding & Kalman Filtering:** [`docs/PREDICTIVE_CODING_KALMAN_ATTENTION.md`](docs/PREDICTIVE_CODING_KALMAN_ATTENTION.md)
- **Master Experiment Registry:** [`EXPERIMENTS.md`](EXPERIMENTS.md)
- **Academic Manuscript Draft:** [`paper/manuscript.md`](paper/manuscript.md)
