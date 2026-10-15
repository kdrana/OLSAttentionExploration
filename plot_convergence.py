"""
Plotting script for convergence benchmark:
- Fig 1: Test R^2 vs Epochs (across California, Kin8nm, Friedman-1)
- Fig 2: Test R^2 vs Wall-clock GPU Seconds
- Fig 3: Final Test R^2 comparison
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence_results(results_path: str = "results/convergence_results.json"):
    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    datasets = list(data.keys())
    models = list(data[datasets[0]].keys())
    colors = {"Standard FT-T (Random Init)": "#e74c3c", "Gram Warm-Start FT-T": "#3498db", "Solvable RegBlock (Degree-2)": "#2ecc71"}
    markers = {"Standard FT-T (Random Init)": "o", "Gram Warm-Start FT-T": "s", "Solvable RegBlock (Degree-2)": "^"}

    # -----------------------------------------------------------------
    # Figure 1: Test R^2 vs Training Epochs (3 Subplots)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for idx, d_name in enumerate(datasets):
        ax = axes[idx]
        for m_name in models:
            hist = data[d_name][m_name]
            epochs = hist["epochs"]
            r2_vals = hist["test_r2"]
            ax.plot(
                epochs,
                r2_vals,
                label=m_name,
                color=colors.get(m_name, "black"),
                marker=markers.get(m_name, "o"),
                markevery=max(1, len(epochs) // 8),
                linewidth=2.2,
                alpha=0.9,
            )

        ax.set_title(f"{d_name.capitalize()} Benchmark", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=11)
        ax.set_ylabel("Out-of-sample $R^2$" if idx == 0 else "", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=10, loc="lower right")

    plt.suptitle("Learning Acceleration: Test $R^2$ vs. Training Epochs", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/fig_convergence_epochs.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved results/fig_convergence_epochs.png")

    # -----------------------------------------------------------------
    # Figure 2: Test R^2 vs Wall-clock GPU Seconds (3 Subplots)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for idx, d_name in enumerate(datasets):
        ax = axes[idx]
        for m_name in models:
            hist = data[d_name][m_name]
            wallclock = hist["wallclock_sec"]
            r2_vals = hist["test_r2"]
            ax.plot(
                wallclock,
                r2_vals,
                label=m_name,
                color=colors.get(m_name, "black"),
                marker=markers.get(m_name, "o"),
                markevery=max(1, len(wallclock) // 8),
                linewidth=2.2,
                alpha=0.9,
            )

        ax.set_title(f"{d_name.capitalize()} Benchmark", fontsize=13, fontweight="bold")
        ax.set_xlabel("Elapsed GPU Time (Seconds)", fontsize=11)
        ax.set_ylabel("Out-of-sample $R^2$" if idx == 0 else "", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=10, loc="lower right")

    plt.suptitle("Wall-Clock Efficiency: Test $R^2$ vs. GPU Execution Time", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/fig_convergence_wallclock.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved results/fig_convergence_wallclock.png")

    # -----------------------------------------------------------------
    # Figure 3: Final Test R^2 Comparison Bar Chart
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(datasets))
    width = 0.25

    for i, m_name in enumerate(models):
        final_scores = [data[d][m_name]["test_r2"][-1] for d in datasets]
        ax.bar(
            x + (i - 1) * width,
            final_scores,
            width,
            label=m_name,
            color=colors.get(m_name, "gray"),
            edgecolor="black",
            alpha=0.85,
        )

    ax.set_ylabel("Final Out-of-Sample $R^2$", fontsize=12)
    ax.set_title("Final Generalization Performance ($R^2$)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("results/fig_final_r2_comparison.png", dpi=300)
    plt.close()
    print("Saved results/fig_final_r2_comparison.png")


if __name__ == "__main__":
    plot_convergence_results()
