# OLS & Softmax Attention: Kernel Basis Function Exploration

This project provides an experimental testbed investigating how **the kernel trick and specific basis function families** relate to **Softmax Attention** and **Ordinary Least Squares (OLS)**.

---

## Theoretical Context

1. **The OLS Connection:** 
   Linear cross-attention without softmax is algebraically equivalent to OLS predictions in a normalized factor space:
   $$A_{\text{linear}} \propto \boldsymbol{Q}\boldsymbol{K}^\top \implies \hat{\boldsymbol{y}} = \boldsymbol{X}_{\text{test}} (\boldsymbol{X}^\top \boldsymbol{X})^{-1} \boldsymbol{X}^\top \boldsymbol{y}$$
2. **The Softmax Exponential Kernel:**
   Softmax attention uses the unnormalized kernel $K(\boldsymbol{q}, \boldsymbol{k}) = \exp(\beta \, \boldsymbol{q}^\top \boldsymbol{k})$.
   On normalized tokens ($\|\boldsymbol{q}\| = \|\boldsymbol{k}\| = 1$, as enforced by LayerNorm/RMSNorm), this is an exponential dot-product kernel on the unit hypersphere $\mathbb{S}^{d-1}$.
3. **Basis Function Families Explored:**
   - **Gegenbauer / Spherical Harmonics:** The canonical orthogonal basis on $\mathbb{S}^{d-1}$ with Bessel function coefficients $I_\nu(\beta)$.
   - **Taylor / Tensor Polynomials:** Truncated polynomial cross-features (Degree-1 = linear/OLS, Degree-2 = pairwise quadratic interactions).
   - **Positive Random Features (FAVOR+ / PRF):** Non-negative Gaussian tilted random feature maps.

---

## Project Structure

```
.
├── kernels.py          # Exact Softmax, Taylor Deg-1 & 2, PRF, and Gegenbauer-Bessel expansions
├── metrics.py          # Relative Frobenius error, attention entropy, effective rank, R^2
├── tasks.py            # Synthetic DGPs (Linear, Interaction, Friedman nonlinear)
├── run_experiments.py  # Automated experiment runner (CPU / Apple Silicon MPS / CUDA)
├── plot_results.py     # Generates publication-quality figures into results/
├── results/            # Generated plots and JSON logs
├── requirements.txt    # Minimal dependencies
└── README.md
```

---

## Running on MacBook Pro M3 (Apple Silicon)

Your MacBook Pro M3 with 18GB of unified memory is ideal for this benchmark. The script automatically detects and utilizes **Apple Silicon Metal (MPS)** for fast tensor matrix multiplications.

### Quick Setup:

```bash
# 1. Clone or copy this directory to your MacBook Pro
cd OLSAttentionExploration

# 2. Create virtual environment using uv or python3 venv
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt
# (or: pip install torch numpy scipy matplotlib)

# 4. Run the experiments
python run_experiments.py

# 5. Generate the plots
python plot_results.py
```

---

## What the Experiments Test

### Experiment 1: Basis Approximation Error vs. Inverse Temperature $\beta$
* Compares the relative Frobenius norm error $\|A_{\text{approx}} - A_{\text{softmax}}\|_F / \|A_{\text{softmax}}\|_F$.
* Tests where the linear / degree-1 approximation (OLS) holds, where degree-2 holds, and how Positive Random Features scale.

### Experiment 2: Gegenbauer-Bessel Harmonic Power Spectrum
* Solves the exact analytical Bessel expansion:
  $$c_n(\beta, d) \propto \frac{I_{n + \frac{d-2}{2}}(\beta)}{\beta^{\frac{d-2}{2}}}$$
* Shows that at small $\beta$ (high temperature / early layers), only order 0 (uniform pooling) and order 1 (OLS linear) carry significant power. As $\beta$ grows, higher-order spherical harmonics activate.

### Experiment 3: Downstream In-Context Regression ($R^2$)
* Evaluates out-of-sample $R^2$ across three distinct data regimes:
  1. **Linear DGP:** Where OLS (Degree-1) is optimal.
  2. **Interaction DGP:** Where Degree-2 pairwise features bridge the gap.
  3. **Friedman DGP:** Where full softmax attention / higher-order harmonics dominate.

### Experiment 4: Attention Entropy & Output Effective Rank
* Measures the thermodynamic transition of attention: from smooth kernel averaging (high entropy) to sharp discrete routing (low entropy), alongside the effective rank of output tokens.
