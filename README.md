# OLS & Softmax Attention: Statistical Foundations, Kernel Approximations, and Solvable Architectures

[![Repository Guidelines](https://img.shields.io/badge/Agent%20Standard-AGENTS.md-blue)](AGENTS.md)
[![Experiment Registry](https://img.shields.io/badge/Registry-EXPERIMENTS.md-success)](EXPERIMENTS.md)
[![Paper Manuscript](https://img.shields.io/badge/Manuscript-paper%2Fmanuscript.md-orange)](paper/manuscript.md)

This repository explores the statistical and mathematical continuum bridging **Ordinary Least Squares (OLS)**, **Kernel Ridge Regression**, and **Softmax Attention** (prompted by Philippe Goulet Coulombe, arXiv:2504.09663). 

We investigate whether kernel approximations (Taylor polynomials, Gegenbauer spherical harmonics, Positive Random Features) and closed-form linear algebraic solvers can replace or accelerate expensive iterative attention mechanisms.

---

## Key Empirical Findings

All experiments have been benchmarked on cloud GPUs (Modal NVIDIA T4) and logged in [`EXPERIMENTS.md`](EXPERIMENTS.md):

1. **Polynomial Domination in the Weak-Coupling Regime:**
   At standard transformer temperature ($\beta = 1/\sqrt{d}$), a **Degree-2 Taylor polynomial expansion achieves $< 0.01\%$ relative Frobenius error** against exact Softmax attention, outperforming Positive Random Features (FAVOR+) by two orders of magnitude ([EXP-01](EXPERIMENTS.md#exp-01-basis-approximation-and-harmonic-decomposition)).
2. **Gram Precision Metric Warm-Start Boosts Tabular Convergence:**
   Initializing query and key projection weights to the empirical Mahalanobis precision metric $W_Q = W_K = (\frac{1}{N} X^\top X + \lambda \boldsymbol{I})^{-1/2}$ eliminates the cold-start penalty on collinear tabular data, improving test $R^2$ on California Housing by **$+0.0226$** ([EXP-02](EXPERIMENTS.md#exp-02-end-to-end-convergence-and-solvability-benchmark)).
3. **Solvable Regression Blocks Outperform Iterative Attention:**
   On dense non-linear physics (Kin8nm robot arm kinematics), replacing iterative multi-head attention with an explicit Degree-2 cross-feature expansion solved in a single closed-form GPU Ridge step **outperformed standard FT-Transformer by $+4.8\%$ ($R^2 = 0.9266$ vs $0.8788$) while running $13\%$ faster** ([EXP-02](EXPERIMENTS.md#exp-02-end-to-end-convergence-and-solvability-benchmark)).

---

## Repository Structure

```
OLSAttentionExploration/
├── README.md                          # Repository overview & setup
├── AGENTS.md                          # Mandatory AI agent protocols & standards
├── GEMINI.md                          # Antigravity agent configuration
├── EXPERIMENTS.md                     # Master index of all empirical runs & metrics
├── requirements.txt                   # Dependencies
│
├── src/                               # Pure, reusable mathematical modules
│   ├── kernels.py                     # Exact Softmax, Taylor, Gegenbauer, PRF
│   ├── models.py                      # FT-Transformer, Gram Warm-Start, Solvable RegBlock
│   ├── data_loader.py                 # California Housing, Kin8nm, Friedman-1 loaders
│   ├── metrics.py                     # Frobenius error, entropy, effective rank, R²
│   └── tasks.py                       # Synthetic DGPs
│
├── experiments/                       # Executable benchmark scripts & cloud runners
│   ├── run_basis_approx.py            # Local/CPU basis error testbed
│   ├── run_convergence.py             # Local/GPU 35-epoch convergence benchmark
│   ├── plot_basis_results.py          # Figures 1-4 generator
│   ├── plot_convergence.py            # Figures 5-7 generator
│   ├── modal_basis_runner.py          # Modal cloud runner for EXP-01
│   └── modal_convergence_runner.py    # Modal cloud runner for EXP-02
│
├── results/                           # Self-contained experiment artifacts
│   ├── exp01_basis_approximation/     # Figures 1-4 & metrics.json
│   └── exp02_convergence_benchmark/   # Figures 5-7 & metrics.json
│
├── docs/                              # Theoretical deep-dives & literature notes
│   └── PREDICTIVE_CODING_KALMAN_ATTENTION.md  # Top-down/bottom-up & Kalman theory
│
└── paper/                             # Publication-ready manuscript
    ├── manuscript.md                  # Complete technical report / preprint draft
    └── references.bib                 # BibTeX references
```

---

## Quickstart

### Local Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments Locally
```bash
# Run EXP-01: Basis approximation testbed
python experiments/run_basis_approx.py
python experiments/plot_basis_results.py

# Run EXP-02: 35-epoch convergence benchmark
python experiments/run_convergence.py
python experiments/plot_convergence.py
```

### Running on Cloud GPU (Modal)
```bash
# Authenticate Modal (one-time setup)
modal setup

# Execute EXP-01 or EXP-02 on an NVIDIA T4 GPU (auto-stops when finished)
modal run experiments/modal_basis_runner.py
modal run experiments/modal_convergence_runner.py
```

---

## Documentation & Manuscript

- **Theory:** [Predictive Coding, Kalman Filters, and Top-Down / Bottom-Up Attention](docs/PREDICTIVE_CODING_KALMAN_ATTENTION.md)
- **Experiment Registry:** [Master Experiment Registry](EXPERIMENTS.md)
- **Manuscript Draft:** [Paper Draft](paper/manuscript.md)
