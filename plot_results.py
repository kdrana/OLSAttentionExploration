"""
Generates publication-quality plots from experimental results.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_all():
    results_path = "results/experiment_results.json"
    if not os.path.exists(results_path):
        print("Results file not found. Run run_experiments.py first.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Figure 1: Basis Approximation Error vs. Beta
    # -------------------------------------------------------------
    exp1 = data["exp1_approx_error"]
    betas = exp1["betas"]

    plt.figure(figsize=(8, 5))
    plt.plot(betas, exp1["deg1_error"], "r--o", label="Degree 1 (Linear / OLS)", markersize=4)
    plt.plot(betas, exp1["deg2_error"], "b-s", label="Degree 2 (Pairwise Tensor)", markersize=4)
    plt.plot(betas, exp1["prf_128_error"], "g-^", label="Positive Random Features (M=128)", markersize=4)
    plt.plot(betas, exp1["prf_512_error"], "m-d", label="Positive Random Features (M=512)", markersize=4)

    plt.axvline(x=1.0 / np.sqrt(16), color="gray", linestyle=":", label="Standard 1/sqrt(d) (d=16)")
    plt.xlabel(r"Inverse Temperature $\beta = 1/\tau$", fontsize=12)
    plt.ylabel(r"Relative Frobenius Error $\|A_{\mathrm{approx}} - A_{\mathrm{softmax}}\|_F / \|A_{\mathrm{softmax}}\|_F$", fontsize=11)
    plt.title("Approximation Error of Basis Families vs. Softmax Attention", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("results/fig1_basis_approximation_error.png", dpi=300)
    plt.close()
    print("Saved results/fig1_basis_approximation_error.png")

    # -------------------------------------------------------------
    # Figure 2: Gegenbauer-Bessel Harmonic Power Spectrum
    # -------------------------------------------------------------
    exp2 = data["exp2_harmonic_spectrum"]
    betas_harm = exp2["betas"]
    spectrum = exp2["spectrum"]

    plt.figure(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels = [
        r"$n=0$ (Uniform Mean)",
        r"$n=1$ (Linear / OLS)",
        r"$n=2$ (Quadratic / Pairwise)",
        r"$n=3$ (Cubic Harmonics)",
        r"$n=4$ (Quartic Harmonics)",
    ]

    for n_order in range(len(spectrum)):
        plt.plot(
            betas_harm,
            spectrum[str(n_order)],
            color=colors[n_order % len(colors)],
            linewidth=2,
            label=labels[n_order] if n_order < len(labels) else f"Order n={n_order}",
        )

    plt.xlabel(r"Inverse Temperature $\beta$", fontsize=12)
    plt.ylabel(r"Normalized Harmonic Spectral Power $c_n(\beta, d)$", fontsize=12)
    plt.title(f"Spherical Harmonic Power Decomposition on $S^{{{exp2['d']-1}}}$ (Bessel Expansion)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("results/fig2_bessel_harmonic_spectrum.png", dpi=300)
    plt.close()
    print("Saved results/fig2_bessel_harmonic_spectrum.png")

    # -------------------------------------------------------------
    # Figure 3: Downstream In-Context Regression R^2 Comparison
    # -------------------------------------------------------------
    exp3 = data["exp3_downstream_r2"]
    dgps = list(exp3.keys())
    models = list(exp3[dgps[0]].keys())

    x = np.arange(len(dgps))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        r2_vals = [exp3[dgp][model] for dgp in dgps]
        ax.bar(x + (i - 1.5) * width, r2_vals, width, label=model)

    ax.set_ylabel(r"Out-of-sample $R^2$", fontsize=12)
    ax.set_title("Downstream Regression Performance Across Basis Families", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([dgp.capitalize() + " DGP" for dgp in dgps], fontsize=11)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("results/fig3_downstream_r2_comparison.png", dpi=300)
    plt.close()
    print("Saved results/fig3_downstream_r2_comparison.png")

    # -------------------------------------------------------------
    # Figure 4: Attention Entropy & Output Effective Rank
    # -------------------------------------------------------------
    exp4 = data["exp4_entropy_rank"]
    betas_ent = exp4["betas"]
    entropies = exp4["entropies"]
    eranks = exp4["eranks"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel(r"Inverse Temperature $\beta$", fontsize=12)
    ax1.set_ylabel(r"Mean Attention Entropy $H(A)$", color=color, fontsize=12)
    ax1.plot(betas_ent, entropies, color=color, linewidth=2.5, label="Attention Entropy")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Output Representation Effective Rank", color=color, fontsize=12)
    ax2.plot(betas_ent, eranks, color=color, linewidth=2.5, linestyle="--", label="Output Effective Rank")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(r"Thermodynamic Phase Shift: Entropy Decay & Rank Collapse vs. $\beta$", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig("results/fig4_entropy_and_rank.png", dpi=300)
    plt.close()
    print("Saved results/fig4_entropy_and_rank.png")


if __name__ == "__main__":
    plot_all()
