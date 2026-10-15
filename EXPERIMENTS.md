# Master Experiment Registry

This registry tracks all empirical investigations conducted in this repository. It serves as the single source of truth for both human researchers and LLM agents to retrieve exact configurations, artifacts, and quantitative results.

---

## Experiment Summary Table

| Experiment ID | Title | Compute Target | Status | Primary Metric | Key Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| [EXP-01-BASIS](#exp-01-basis-approximation-and-harmonic-decomposition) | Basis Function Approximation of Softmax Attention | Modal NVIDIA T4 GPU | Completed | Relative Frobenius Error & $R^2$ | Degree-2 polynomial captures $>99.99\%$ of attention power at $\beta = 1/\sqrt{d}$; entropy decays logarithmically. |
| [EXP-02-CONV](#exp-02-end-to-end-convergence-and-solvability-benchmark) | Convergence & Solvability Benchmark on Real Data | Modal NVIDIA T4 GPU | Completed | Test $R^2$ & GPU Wall-Clock Sec | Gram Warm-Start improves California $R^2$ by $+0.0226$; Solvable RegBlock beats FT-T by $+0.0478$ on Kin8nm (13% faster). |

---

## Detailed Experiment Logs

### EXP-01: Basis Approximation and Harmonic Decomposition

- **Date:** 2026-10-15
- **Compute:** Modal Serverless GPU (NVIDIA T4, 16GB) — Run time ~15s (Cost: < $0.0005)
- **Execution Script:** [`experiments/run_basis_approx.py`](experiments/run_basis_approx.py) (Cloud runner: [`experiments/modal_basis_runner.py`](experiments/modal_basis_runner.py))
- **Artifacts:**
  - Metrics JSON: [`results/exp01_basis_approximation/metrics.json`](results/exp01_basis_approximation/metrics.json)
  - Figure 1 (Frobenius Error vs $\beta$): [`results/exp01_basis_approximation/fig1_basis_approximation_error.png`](results/exp01_basis_approximation/fig1_basis_approximation_error.png)
  - Figure 2 (Bessel Harmonic Spectrum): [`results/exp01_basis_approximation/fig2_bessel_harmonic_spectrum.png`](results/exp01_basis_approximation/fig2_bessel_harmonic_spectrum.png)
  - Figure 3 (Downstream $R^2$ Comparison): [`results/exp01_basis_approximation/fig3_downstream_r2_comparison.png`](results/exp01_basis_approximation/fig3_downstream_r2_comparison.png)
  - Figure 4 (Attention Entropy & Effective Rank): [`results/exp01_basis_approximation/fig4_entropy_and_rank.png`](results/exp01_basis_approximation/fig4_entropy_and_rank.png)

#### Hypothesis
At standard transformer temperature $\beta = 1/\sqrt{d}$, Softmax attention behaves predominantly as a low-degree polynomial kernel on the sphere $\mathbb{S}^{d-1}$. Higher-order spherical harmonics contribute negligible power unless $\beta \gg 1$.

#### Quantitative Results
1. **Frobenius Approximation Error at $\beta = 1/\sqrt{16} = 0.25$:**
   - Degree 1 (Linear / OLS Hat Matrix): $\approx 1.2 \times 10^{-2}$
   - Degree 2 (Pairwise Quadratic Kernel): $\mathbf{< 10^{-4}}$ ($< 0.01\%$ relative error)
   - Positive Random Features (FAVOR+, $M=512$): $\approx 4.8 \times 10^{-2}$ (substantially noisier than exact Taylor)
2. **Harmonic Power Spectrum (Gegenbauer Expansion):**
   - At $\beta = 0.25$, orders $n=0$ (uniform mean) and $n=1$ (linear OLS) account for $>98\%$ of the Gegenbauer spectral mass $c_n(\beta, d)$.
3. **Downstream In-Context $R^2$ Across DGPs:**
   - Linear DGP: Exact Softmax $R^2 = 0.999$, Degree-1 $R^2 = 0.999$, Degree-2 $R^2 = 0.999$.
   - Interaction DGP ($x_1 x_2 + x_3 x_4$): Degree-1 fails ($R^2 = -0.01$), Degree-2 achieves $R^2 = 0.985$, Exact Softmax achieves $R^2 = 0.987$.
   - Friedman-1 DGP: Degree-2 achieves $R^2 = 0.842$, matching Exact Softmax ($R^2 = 0.845$).

---

### EXP-02: End-to-End Convergence and Solvability Benchmark

- **Date:** 2026-10-16
- **Compute:** Modal Serverless GPU (NVIDIA T4, 16GB) — Run time ~45s (Cost: < $0.01)
- **Execution Script:** [`experiments/run_convergence.py`](experiments/run_convergence.py) (Cloud runner: [`experiments/modal_convergence_runner.py`](experiments/modal_convergence_runner.py))
- **Artifacts:**
  - Metrics JSON: [`results/exp02_convergence_benchmark/metrics.json`](results/exp02_convergence_benchmark/metrics.json)
  - Figure 5 (Test $R^2$ vs Training Epochs): [`results/exp02_convergence_benchmark/fig_convergence_epochs.png`](results/exp02_convergence_benchmark/fig_convergence_epochs.png)
  - Figure 6 (Test $R^2$ vs Wall-Clock GPU Seconds): [`results/exp02_convergence_benchmark/fig_convergence_wallclock.png`](results/exp02_convergence_benchmark/fig_convergence_wallclock.png)
  - Figure 7 (Final Test $R^2$ Comparison): [`results/exp02_convergence_benchmark/fig_final_r2_comparison.png`](results/exp02_convergence_benchmark/fig_final_r2_comparison.png)

#### Models Evaluated
1. **Model A:** Standard FT-Transformer with Random Xavier Initialization.
2. **Model B:** Gram Precision Metric Warm-Start FT-Transformer ($W_Q = W_K = (\frac{1}{N} X^\top X + \lambda \boldsymbol{I})^{-1/2}$).
3. **Model C:** Solvable Degree-2 Regression Block (Explicit pairwise cross-features + GPU Ridge closed-form solve).

#### Quantitative Results Table (35 Epochs on NVIDIA T4)

| Dataset | Metric | Model A (Standard FT-T) | Model B (Gram Warm-Start) | Model C (Solvable RegBlock) |
| :--- | :--- | :---: | :---: | :---: |
| **California Housing**<br>($N=20,640$, 8 features) | **Test $R^2$**<br>Wall-clock GPU Time | $0.7720$<br>$13.61\text{ s}$ | **$0.7946$** ($+0.0226$)<br>$13.13\text{ s}$ | **$0.7917$** ($+0.0197$)<br>**$11.45\text{ s}$ (16% faster)** |
| **Kin8nm Robot Kinematics**<br>($N=8,192$, 8 features) | **Test $R^2$**<br>Wall-clock GPU Time | $0.8788$<br>$5.33\text{ s}$ | $0.8370$<br>$5.14\text{ s}$ | **$0.9266$** (**$+0.0478$**)<br>**$4.66\text{ s}$ (13% faster)** |
| **Friedman-1 Benchmark**<br>($N=10,000$, 10 features) | **Test $R^2$**<br>Wall-clock GPU Time | $0.9937$<br>$6.27\text{ s}$ | **$0.9967$** ($+0.0030$)<br>$6.13\text{ s}$ | **$0.9961$** ($+0.0024$)<br>**$5.70\text{ s}$ (9% faster)** |

#### Key Findings
1. **Gram Precision Metric Warm-Start (Model B):** Pre-aligning query-key matrices with the empirical precision metric eliminates the cold-start penalty on collinear tabular data, boosting California Housing $R^2$ by $+0.0226$.
2. **Solvable Regression Block Dominance (Model C):** On non-linear physical interactions (Kin8nm), solving the explicit degree-2 polynomial expansion directly via GPU Ridge regression outperforms iterative multi-head attention by **$+4.8\%$ ($0.9266$ vs $0.8788$)** while running **$13\%$ faster**.
