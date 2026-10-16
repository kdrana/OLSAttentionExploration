# Solvable Attention Architectures: Mathematical Theory, Algorithmic Synthesis & Benchmark Findings

> **Executive Summary:** This document provides the unified mathematical foundation and empirical synthesis bridging **Ordinary Least Squares (OLS)**, **Kernel Ridge Regression**, and **Softmax Attention**. It documents the progression from spherical harmonic Taylor approximations to tabular solvability, sequence error-correction (RLS / Kalman filtering), and parallel chunkwise closed-form GPU solvers.

---

## Table of Contents
1. [The Mathematical Continuum: From OLS to Softmax Attention](#1-the-mathematical-continuum-from-ols-to-softmax-attention)
2. [Tabular Solvability & The Gram Precision Metric](#2-tabular-solvability--the-gram-precision-metric)
3. [The Sequence Frontier: Unnormalized Blur vs. Innovation Correction](#3-the-sequence-frontier-unnormalized-blur-vs-innovation-correction)
4. [The Three Solvable Attention Architectures](#4-the-three-solvable-attention-architectures)
   - [4.1 Chunkwise Closed-Form Ridge Attention (`chunk_ridge`)](#41-chunkwise-closed-form-ridge-attention-chunk_ridge)
   - [4.2 Diagonal Kalman Attention (`diag_kalman`)](#42-diagonal-kalman-attention-diag_kalman)
   - [4.3 Chunkwise Degree-2 Polynomial Solvable Attention (`chunk_poly`)](#43-chunkwise-degree-2-polynomial-solvable-attention-chunk_poly)
5. [The Dual Ridge Kernel Formulation (Representer Theorem)](#5-the-dual-ridge-kernel-formulation-representer-theorem)
6. [Master Quantitative Empirical Synthesis](#6-master-quantitative-empirical-synthesis)
7. [Engineering Principles & Production Decision Tree](#7-engineering-principles--production-decision-tree)

---

## 1. The Mathematical Continuum: From OLS to Softmax Attention

Modern deep learning treats multi-head attention as a black-box heuristic. However, as demonstrated by Philippe Goulet Coulombe (2025), attention is fundamentally a **continuous-state non-parametric kernel regression estimator**.

### 1.1 The Classical OLS Hat Matrix
In classical linear regression, given design matrix $\boldsymbol{X} \in \mathbb{R}^{N \times d}$ and targets $\boldsymbol{y} \in \mathbb{R}^N$:
$$\hat{\boldsymbol{y}} = \boldsymbol{X} \left(\boldsymbol{X}^\top \boldsymbol{X}\right)^{-1} \boldsymbol{X}^\top \boldsymbol{y} = \boldsymbol{H}_{\text{OLS}} \boldsymbol{y}$$
where $\boldsymbol{H}_{\text{OLS}} = \boldsymbol{X} (\boldsymbol{X}^\top \boldsymbol{X})^{-1} \boldsymbol{X}^\top$ is the orthogonal projection operator (the "hat matrix").

### 1.2 Kernel Ridge Regression
By replacing inner products with a Reproducing Kernel Hilbert Space (RKHS) feature map $\phi(\boldsymbol{x})$, Ridge regression solves:
$$\min_{\boldsymbol{w}} \|\Phi \boldsymbol{w} - \boldsymbol{y}\|_2^2 + \lambda \|\boldsymbol{w}\|_2^2 \implies \hat{\boldsymbol{y}} = \boldsymbol{K} \left(\boldsymbol{K} + \lambda \boldsymbol{I}\right)^{-1} \boldsymbol{y}$$
where $\boldsymbol{K}_{i, j} = k(\boldsymbol{x}_i, \boldsymbol{x}_j) = \langle \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \rangle$.

### 1.3 Softmax Attention as Kernel Nadaraya-Watson
In multi-head attention, for queries $\boldsymbol{Q} \in \mathbb{R}^{T \times d}$, keys $\boldsymbol{K} \in \mathbb{R}^{T \times d}$, and values $\boldsymbol{V} \in \mathbb{R}^{T \times d}$:
$$\boldsymbol{A} = \text{Softmax}\left(\frac{\boldsymbol{Q} \boldsymbol{K}^\top}{\sqrt{d}}\right) \boldsymbol{V}$$
When normalized on the unit sphere $\mathbb{S}^{d-1}$ ($\|\boldsymbol{q}\| = \|\boldsymbol{k}\| = 1$), the dot product $\boldsymbol{q}^\top \boldsymbol{k} = \cos \theta$. The Softmax kernel is the exponential dot-product kernel:
$$k_{\text{softmax}}(\boldsymbol{q}, \boldsymbol{k}) = \exp\left(\beta \, \boldsymbol{q}^\top \boldsymbol{k}\right), \quad \beta = \frac{1}{\sqrt{d}}$$

### 1.4 Spherical Harmonic & Taylor Decomposition ([EXP-01](file:///home/kdrana/Projects/OLSAttentionExploration/results/exp01_basis_approximation/metrics.json))
Expanding the exponential kernel via Taylor series:
$$\exp(\beta \boldsymbol{q}^\top \boldsymbol{k}) = 1 + \beta (\boldsymbol{q}^\top \boldsymbol{k}) + \frac{\beta^2}{2} (\boldsymbol{q}^\top \boldsymbol{k})^2 + \mathcal{O}(\beta^3)$$
At standard transformer temperatures ($\beta = 1/\sqrt{d} \approx 0.25$ for $d=16$):
- **Order 0 ($n=0$):** Uniform spatial average (diffuse background).
- **Order 1 ($n=1$):** Pure linear OLS projection ($\boldsymbol{q}^\top \boldsymbol{k}$).
- **Order 2 ($n=2$):** Pairwise cross-feature interactions $(q_i q_j)(k_i k_j)$.
- **Empirical Finding:** Orders 0, 1, and 2 account for **$> 99.99\%$ of the total Frobenius power** ($< 0.01\%$ relative approximation error). Higher-order spherical harmonics are completely dormant unless $\beta \gg 1$.

```
                              SPECTRAL POWER SPECTRUM (d=16, beta=0.25)
       100% ┌────────────────────────────────────────────────────────┐
            │ ██████████████████████████████████████████████████████ │ >99.99% Power
        80% │                                                        │
        60% │                                                        │
        40% │                                                        │
        20% │                                                        │
         0% └────────────────────────────────────────────────────────┘
              Order 0 (Mean) + Order 1 (Linear) + Order 2 (Pairwise Quadratic)
```

---

## 2. Tabular Solvability & The Gram Precision Metric

Collinear features in tabular data create severe training instability in standard multi-head attention initialized with isotropic Xavier/He weights.

### 2.1 The Gram Precision Metric Warm-Start ([EXP-02](file:///home/kdrana/Projects/OLSAttentionExploration/results/exp02_convergence_benchmark/metrics.json))
Coulombe demonstrated that the optimal metric space for query-key matching is given by the empirical inverse covariance (precision) matrix of the token tokens:
$$\boldsymbol{G} = \frac{1}{N} \boldsymbol{X}^\top \boldsymbol{X} + \lambda \boldsymbol{I} \in \mathbb{R}^{d \times d}$$
By setting the initialization weights of the attention projections to the matrix square-root inverse:
$$\boldsymbol{W}_Q = \boldsymbol{W}_K = \boldsymbol{G}^{-1/2} = \left(\frac{1}{N} \boldsymbol{X}^\top \boldsymbol{X} + \lambda \boldsymbol{I}\right)^{-1/2}$$
the transformed inner product becomes the **Mahalanobis precision metric**:
$$\boldsymbol{q}_i^\top \boldsymbol{k}_j = \boldsymbol{x}_i^\top \boldsymbol{W}_Q^\top \boldsymbol{W}_K \boldsymbol{x}_j = \boldsymbol{x}_i^\top \boldsymbol{G}^{-1} \boldsymbol{x}_j$$
- **Empirical Takeaway:** Pre-aligning $W_Q, W_K$ eliminates the cold-start penalty, improving test $R^2$ on California Housing by **$+0.0226$** without changing parameter count.

### 2.2 The Solvable Degree-2 Regression Block
Rather than learning query-key projections via gradient descent, the **Solvable Regression Block** explicitly materializes pairwise polynomial interactions:
$$\phi_2(\boldsymbol{x}) = \left[x_1, \dots, x_P, \, x_1^2, \dots, x_P^2, \, x_1 x_2, \dots, x_{P-1} x_P\right] \in \mathbb{R}^{D_{\text{poly}}}$$
and solves the linear mixing layer in a **single closed-form GPU Ridge regression step**:
$$\boldsymbol{W}^* = \left(\Phi^\top \Phi + \lambda \boldsymbol{I}\right)^{-1} \Phi^\top \boldsymbol{y}$$
- **Empirical Takeaway:** On Kin8nm (non-linear robot arm kinematics), the Solvable Regression Block **outperformed standard FT-Transformer by $+4.8\%$ ($R^2 = 0.9266$ vs $0.8788$) while training $13\%$ faster** ($4.66\text{s}$ vs $5.33\text{s}$).

---

## 3. The Sequence Frontier: Unnormalized Blur vs. Innovation Correction

Extending linear attention to long sequences reveals a fundamental mathematical bottleneck.

### 3.1 Why Pure Linear Attention Collapses on Long Sequences ([EXP-03](file:///home/kdrana/Projects/OLSAttentionExploration/results/exp03_sequence_bulk_margin/metrics.json))
Pure Linear Attention replaces Softmax with a kernel feature map $\phi(x) = \text{ELU}(x) + 1$:
$$\boldsymbol{S}_t = \sum_{\tau=1}^t \phi(\boldsymbol{k}_\tau) \boldsymbol{v}_\tau^\top \in \mathbb{R}^{d \times d}$$
While this enables strict $O(T)$ execution ($6.6\times$ speedup at $T=4096$), its memory accumulation is **purely additive and unnormalized**.
- In Multi-Query Associative Recall (MQAR) with 500 tokens of distractor noise, $\phi(\boldsymbol{k}_\tau)$ produces positive dot products across all background tokens.
- The signal-to-noise ratio decays as $\mathcal{O}(1/T)$, causing complete retrieval collapse (**$\le 2.66\%$ accuracy**).

### 3.2 Predictive Coding & The Innovation Error ([EXP-04](file:///home/kdrana/Projects/OLSAttentionExploration/results/exp04_kalman_rls_attention/metrics.json))
To prevent memory saturation, sequence memory must not store raw inputs. It must store only the **unexpected prediction error (innovation)**:
$$\hat{\boldsymbol{v}}_t = \boldsymbol{S}_{t-1}^\top \boldsymbol{k}_t \quad (\text{Memory Prediction})$$
$$\boldsymbol{e}_t = \boldsymbol{v}_t - \hat{\boldsymbol{v}}_t \quad (\text{Prediction Innovation Error})$$
- When an incoming token is predictable from prior context, $\hat{\boldsymbol{v}}_t \to \boldsymbol{v}_t \implies \boldsymbol{e}_t \to 0$. Memory is not updated.
- When an associative key-value pair arrives, $\boldsymbol{e}_t$ spikes, driving an immediate memory update.
- **Empirical Breakthrough:** Both Delta/LMS ($21.35\%$) and Causal RLS ($15.62\%$) crushed Standard Softmax ($3.12\%$) and Pure Linear Attention ($2.60\%$) on MQAR.

---

## 4. The Three Solvable Attention Architectures

In [EXP-05](file:///home/kdrana/Projects/OLSAttentionExploration/results/exp05_solvable_attention_suite/metrics.json), we synthesized the learnings of online Kalman filtering and closed-form solvability into three novel architectures.

```
                                  ARCHITECTURAL COMPARISON

          Architecture               Compute Loop           Mathematical Operator        Memory Inference
   ┌──────────────────────────────┬──────────────────┬─────────────────────────────────┬──────────────────┐
   │ 1. Chunkwise Ridge           │ Batched Cholesky │ W* = (K^T K + λI)^{-1} K^T V    │ O(1) Bounded     │
   │ 2. Diagonal Kalman           │ O(d) Elementwise │ Precision-weighted Delta Gate   │ O(1) Vector Stat │
   │ 3. Chunkwise Poly (Dual)     │ 16x16 Dual Solve │ α = (Φ Φ^T + λI)^{-1} V         │ O(1) Dual State  │
   └──────────────────────────────┴──────────────────┴─────────────────────────────────┴──────────────────┘
```

### 4.1 Chunkwise Closed-Form Ridge Attention (`chunk_ridge`)
Instead of updating token-by-token sequentially, `ChunkwiseRidgeAttention` partitions the sequence into chunks of size $C=16$:
1. **Intra-Chunk Closed-Form Solve:**
   Within chunk $c$, the mapping $K_c \to V_c$ is an exact linear regression. We solve it analytically across all chunks in parallel in a single GPU kernel launch:
   $$\boldsymbol{W}_c^* = \left(\boldsymbol{K}_c^\top \boldsymbol{K}_c + \lambda \boldsymbol{I}\right)^{-1} \boldsymbol{K}_c^\top \boldsymbol{V}_c \in \mathbb{R}^{d \times d}$$
2. **Causal Inter-Chunk Recurrence:**
   Memory propagates across chunks via discounted accumulation:
   $$\boldsymbol{S}_c = \gamma \boldsymbol{S}_{c-1} + \boldsymbol{W}_{c-1}^*$$
3. **Causal Readout:**
   Queries inside chunk $c$ read from history via $\boldsymbol{Y}_{\text{inter}} = \boldsymbol{Q}_c \boldsymbol{S}_c$ and from the current chunk via causal softmax:
   $$\boldsymbol{Y}_c = \boldsymbol{Q}_c \boldsymbol{S}_c + \text{CausalSoftmax}\left(\frac{\boldsymbol{Q}_c \boldsymbol{K}_c^\top}{\sqrt{d}}\right) \boldsymbol{V}_c$$
- **Benchmark Result:** Trains in **$1.50\text{ s}$** on GPU (**$37.5\times$ faster than Causal RLS**).

### 4.2 Diagonal Kalman Attention (`diag_kalman`)
Full-matrix RLS updates an expensive $d \times d$ inverse covariance matrix $\boldsymbol{P}_t$ using Sherman-Morrison subtraction. `DiagonalKalmanAttention` restricts $\boldsymbol{P}_t$ to be diagonal: $\boldsymbol{P}_t = \text{diag}(\boldsymbol{p}_t) \in \mathbb{R}^d$.
1. **$O(d)$ Elementwise Kalman Gain:**
   $$\boldsymbol{u}_t = \boldsymbol{p}_{t-1} \odot \boldsymbol{k}_t, \quad \boldsymbol{K}_t = \frac{\boldsymbol{u}_t}{\lambda + \boldsymbol{k}_t^\top \boldsymbol{u}_t + \epsilon} \in \mathbb{R}^d$$
2. **Diagonal Precision Update:**
   $$\boldsymbol{p}_t = \text{clamp}\left(\frac{\boldsymbol{p}_{t-1} - \boldsymbol{K}_t \odot \boldsymbol{k}_t \odot \boldsymbol{p}_{t-1}}{\lambda}, \, 10^{-4}, \, 10^4\right)$$
3. **Gated Innovation Memory:**
   $$\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \sigma(\boldsymbol{W}_\beta \boldsymbol{x}_t) \boldsymbol{K}_t \left(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}^\top \boldsymbol{k}_t\right)^\top$$
- **Benchmark Result:** Delivers an exact **$0.0083\text{ MB}$ constant inference memory footprint** ($482\times$ smaller than KV-cache at $T=4096$).

### 4.3 Chunkwise Degree-2 Polynomial Solvable Attention (`chunk_poly`)
Linear models cannot capture non-linear multiplicative key interactions. `ChunkwisePolySolvableAttention` applies the degree-2 cross-feature expansion inside the chunkwise solver:
$$\phi_2(\boldsymbol{k}) = \left[\boldsymbol{k}, \, \boldsymbol{k}^{\odot 2}, \, k_i k_j\right]_{i < j} \in \mathbb{R}^D \quad (D = 152 \text{ for } d=16)$$
To avoid inverting a high-dimensional $152 \times 152$ matrix, we deploy the **Dual Ridge formulation**.

---

## 5. The Dual Ridge Kernel Formulation (Representer Theorem)

When the feature dimension $D = 152$ exceeds chunk length $C = 16$, solving the primal normal equations $\left(\Phi^\top \Phi + \lambda \boldsymbol{I}_D\right) \boldsymbol{W} = \Phi^\top \boldsymbol{V}$ involves an ill-conditioned, low-rank matrix that triggers CUDA memory faults in batched GPU LU solvers (`magma_sgetrf_batched`).

### 5.1 Theorem (Primal-Dual Ridge Equivalence)
For any feature matrix $\Phi \in \mathbb{R}^{C \times D}$ and value matrix $\boldsymbol{V} \in \mathbb{R}^{C \times d}$, with regularization parameter $\lambda > 0$:
$$\left(\Phi^\top \Phi + \lambda \boldsymbol{I}_D\right)^{-1} \Phi^\top \boldsymbol{V} \equiv \Phi^\top \left(\Phi \Phi^\top + \lambda \boldsymbol{I}_C\right)^{-1} \boldsymbol{V}$$

*Proof:*
Begin with the algebraic identity:
$$\Phi^\top \left(\Phi \Phi^\top + \lambda \boldsymbol{I}_C\right) = \Phi^\top \Phi \Phi^\top + \lambda \Phi^\top = \left(\Phi^\top \Phi + \lambda \boldsymbol{I}_D\right) \Phi^\top$$
Multiply on the left by $\left(\Phi^\top \Phi + \lambda \boldsymbol{I}_D\right)^{-1}$ and on the right by $\left(\Phi \Phi^\top + \lambda \boldsymbol{I}_C\right)^{-1}$:
$$\left(\Phi^\top \Phi + \lambda \boldsymbol{I}_D\right)^{-1} \Phi^\top = \Phi^\top \left(\Phi \Phi^\top + \lambda \boldsymbol{I}_C\right)^{-1}$$
Multiplying both sides by $\boldsymbol{V}$ completes the proof. $\blacksquare$

### 5.2 Algorithmic Execution
1. Compute the $C \times C = 16 \times 16$ chunk Gram matrix:
   $$\boldsymbol{G}_{\text{dual}} = \Phi_c \Phi_c^\top + \lambda \boldsymbol{I}_C \in \mathbb{R}^{16 \times 16}$$
2. Solve the dual coefficients $\boldsymbol{\alpha}_c \in \mathbb{R}^{16 \times d}$:
   $$\boldsymbol{\alpha}_c = \text{torch.linalg.solve}(\boldsymbol{G}_{\text{dual}}, \, \boldsymbol{V}_c)$$
3. Map back to feature weights:
   $$\boldsymbol{W}_c^* = \Phi_c^\top \boldsymbol{\alpha}_c \in \mathbb{R}^{152 \times d}$$
- **Computational Impact:** Inverting a $16 \times 16$ matrix is **$100\times$ faster** than a $152 \times 152$ matrix, strictly positive-definite, and executes across hundreds of chunks in parallel in microseconds.

---

## 6. Master Quantitative Empirical Synthesis

All experiments were executed on an NVIDIA T4 GPU (16GB) under identical training settings ($10$ epochs, batch size $32$, sequence length $T=128$).

### 6.1 Multi-Query Associative Recall & Training Wall-Clock Seconds
*(Exact data from [`results/exp05_solvable_attention_suite/metrics.json`](file:///home/kdrana/Projects/OLSAttentionExploration/results/exp05_solvable_attention_suite/metrics.json))*

| Architecture | Category | Epoch 5 Acc | Epoch 10 Acc | Peak Acc | GPU Training Time (10 Epochs) | Speedup vs. RLS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard Softmax** | Quadratic Attention | $2.60\%$ | $3.12\%$ | $3.65\%$ | $1.47\text{ s}$ | $38.3\times$ faster |
| **Pure Linear** | Unnormalized Recurrence | $2.60\%$ | $2.60\%$ | $3.65\%$ | $3.74\text{ s}$ | $15.0\times$ faster |
| **Causal RLS / Kalman** | Optimal Riccati Recurrence | $2.08\%$ | $15.62\%$ | $15.62\%$ | $56.23\text{ s}$ | $1.0\times$ (Baseline) |
| **Diagonal Kalman** | $O(d)$ Vector Preconditioned | $5.73\%$ | $11.98\%$ | $11.98\%$ | $44.51\text{ s}$ | $1.26\times$ faster |
| **Chunkwise Ridge** | Linear Closed-Form Solver | $5.73\%$ | **$14.58\%$** | **$14.58\%$** | **$1.50\text{ s}$** | **$37.5\times$ faster** |
| **Chunkwise Poly Solvable** | Non-Linear Dual Ridge Solver | **$8.33\%$** | **$20.83\%$** | **$21.35\%$** | **$6.73\text{ s}$** | **$8.4\times$ faster ($4.5\times$ faster than Delta)** |
| **Delta / LMS Attention** | Gated Innovation Recurrence | $5.73\%$ | **$21.35\%$** | **$21.35\%$** | $30.57\text{ s}$ | $1.84\times$ faster |

### 6.2 Autoregressive Generation State Memory (Zero-KV-Cache Serving)

| Sequence Length $T$ | Standard Softmax KV-Cache | Chunkwise Poly Solvable | Chunkwise Ridge Attention | Causal RLS Memory | Diagonal Kalman Attention | Reduction Factor vs. KV-Cache |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T = 256$** | $0.250\text{ MB}$ | $0.2227\text{ MB}$ | $0.0234\text{ MB}$ | $0.0156\text{ MB}$ | **$0.0083\text{ MB}$** | **$30\times$ smaller** |
| **$T = 512$** | $0.500\text{ MB}$ | $0.2227\text{ MB}$ | $0.0234\text{ MB}$ | $0.0156\text{ MB}$ | **$0.0083\text{ MB}$** | **$60\times$ smaller** |
| **$T = 1,024$** | $1.000\text{ MB}$ | $0.2227\text{ MB}$ | $0.0234\text{ MB}$ | $0.0156\text{ MB}$ | **$0.0083\text{ MB}$** | **$120\times$ smaller** |
| **$T = 2,048$** | $2.000\text{ MB}$ | $0.2227\text{ MB}$ | $0.0234\text{ MB}$ | $0.0156\text{ MB}$ | **$0.0083\text{ MB}$** | **$241\times$ smaller** |
| **$T = 4,096$** | $4.000\text{ MB}$ | $0.2227\text{ MB}$ | $0.0234\text{ MB}$ | $0.0156\text{ MB}$ | **$0.0083\text{ MB}$** | **$482\times$ smaller (Constant Flatline)** |

---

## 7. Engineering Principles & Production Decision Tree

When architecting models for real-world deployment, choose the attention mechanism based on operational constraints:

```
                                PRODUCTION SELECTION GUIDE

                 Is training throughput or associative recall the priority?
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
         [Throughput & Speed]                       [Associative Recall]
                  │                                         │
     Does sequence require non-linear         Can serving accommodate linear
     multiplicative feature cross-terms?      KV-cache memory growth?
          │               │                                │               │
         YES              NO                              YES              NO
          │               │                                │               │
          ▼               ▼                                ▼               ▼
   Chunkwise Poly   Chunkwise Ridge                 Standard Softmax   Diagonal Kalman
     Solvable          Attention                       Attention          or Delta-Net
  (20.8%, 6.7s)      (14.6%, 1.5s)                     (O(T) Cache)       (O(1) Memory)
```

1. **Maximum Accuracy & Sub-Quadratic Training:** Use **`ChunkwisePolySolvableAttention`**. It matches Delta-Net accuracy ($21.4\%$) but trains in parallel via dual Cholesky solves at $4.5\times$ higher speed.
2. **Maximum Training Speed & Long Contexts:** Use **`ChunkwiseRidgeAttention`**. At $1.50\text{ s}$ per 10 epochs, it matches Softmax speed while delivering $14.58\%$ recall accuracy and $O(1)$ constant serving memory.
3. **Ultra-Low Memory Edge Inference:** Use **`DiagonalKalmanAttention`**. Maintains an elementwise state footprint of just **$8.3\text{ KB}$** across arbitrarily long contexts ($482\times$ smaller than KV-cache).
