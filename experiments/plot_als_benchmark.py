"""
Plotting module for EXP-10: Alternating Least Squares (ALS) Solvable Refinement Benchmark.
Generates publication-ready figures:
1. fig_als_r2_improvement.png: Direct comparison of Greedy DSN vs ALS Refinement vs GBDTs & Deep Attention.
2. fig_als_convergence_trajectory.png: Cycle-by-cycle ALS monotonic convergence vs 35 AdamW epochs.
3. fig_als_wallclock_comparison.png: Log-scale training latency comparison.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def plot_als_results(metrics_file: str, out_dir: str = "results/exp10_als_solvable_refinement"):
    os.makedirs(out_dir, exist_ok=True)
    with open(metrics_file, "r") as f:
        data = json.load(f)

    datasets = ["kin8nm", "friedman1", "california"]
    ds_titles = {
        "kin8nm": "Kin8nm (Robot Kinematics)",
        "friedman1": "Friedman-1 (Nonlinear)",
        "california": "California Housing (Real-World)",
    }

    # =========================================================================
    # Figure 1: Test R^2 Comparison (Greedy DSN vs ALS-Refined DSN vs Baselines)
    # =========================================================================
    models_to_plot = [
        ("Greedy DSN (EXP-09)", "Greedy DSN (EXP-09)", "#1E88E5"),
        ("ALS-Refined DSN (5 Cycles)", "ALS-Refined DSN (Ours)", "#00E676"),
        ("XGBoost (100 Trees)", "XGBoost (100 Trees)", "#FF7043"),
        ("LightGBM (100 Trees)", "LightGBM (100 Trees)", "#E64A19"),
        ("FT-Transformer (AdamW)", "FT-Transformer (35 ep)", "#7E57C2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, ds in zip(axes, datasets):
        ds_data = data[ds]["models"]
        names = []
        r2_vals = []
        colors = []
        for key, label, col in models_to_plot:
            if key in ds_data:
                names.append(label)
                r2_vals.append(ds_data[key]["test_r2"])
                colors.append(col)

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, r2_vals, color=colors, edgecolor="black", alpha=0.88, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Test $R^2$ Score")
        ax.set_title(ds_titles[ds], fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        min_val = min(r2_vals)
        ax.set_xlim(max(0.0, min_val - 0.08), 1.02)

        for bar, val in zip(bars, r2_vals):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
            )

    plt.suptitle("EXP-10: Impact of Alternating Least Squares (ALS) Refinement", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fig_als_r2_improvement.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # Figure 2: Convergence Trajectory (ALS Cycles vs AdamW Epochs)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, ds in zip(axes, datasets):
        als_prog = data[ds]["als_cycle_progression"]
        cycles = [p["cycle"] for p in als_prog]
        als_r2 = [p["test_r2"] for p in als_prog]

        ft_prog = data[ds]["models"]["FT-Transformer (AdamW)"]["epoch_history"]
        epochs = [p["epoch"] for p in ft_prog]
        adam_r2 = [p["test_r2"] for p in ft_prog]

        # Primary axis: ALS cycles
        line1 = ax.plot(cycles, als_r2, marker="o", color="#00C853", linewidth=2.5, markersize=8, label="ALS Solvable (5 Cycles)")
        ax.set_xlabel("ALS Block-Coordinate Cycles")
        ax.set_ylabel("Test $R^2$ Score", color="#00C853")
        ax.set_xticks(cycles)
        ax.set_xticklabels([f"C{c}" if c > 0 else "Greedy" for c in cycles])
        ax.tick_params(axis="y", labelcolor="#00C853")

        # Secondary axis: AdamW epochs
        ax2 = ax.twiny()
        line2 = ax2.plot(epochs, adam_r2, color="#7E57C2", linestyle="--", linewidth=2.0, alpha=0.8, label="FT-Transformer (AdamW)")
        ax2.set_xlabel("AdamW Training Epochs", color="#7E57C2")
        ax2.tick_params(axis="x", labelcolor="#7E57C2")

        ax.set_title(ds_titles[ds], fontweight="bold", pad=15)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="lower right" if ds != "california" else "lower right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("EXP-10: Convergence Trajectory — Instant ALS Cycles vs 35 AdamW Epochs", fontweight="bold", y=1.04)
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "fig_als_convergence_trajectory.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")

    # =========================================================================
    # Figure 3: Wall-Clock Training Latency Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cal_models = data["california"]["models"]
    bench_models = [
        ("Greedy DSN (EXP-09)", "Greedy DSN", "#1E88E5"),
        ("ALS-Refined DSN (5 Cycles)", "ALS-Refined DSN (Total)", "#00C853"),
        ("LightGBM (100 Trees)", "LightGBM", "#E64A19"),
        ("XGBoost (100 Trees)", "XGBoost", "#FF7043"),
        ("FT-Transformer (AdamW)", "FT-Transformer (35 ep)", "#7E57C2"),
    ]

    labels = []
    times_ms = []
    bar_colors = []
    for key, label, col in bench_models:
        if key in cal_models:
            labels.append(label)
            times_ms.append(cal_models[key]["fit_time_ms"])
            bar_colors.append(col)

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, times_ms, color=bar_colors, edgecolor="black", alpha=0.88, width=0.55)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontweight="medium")
    ax.set_ylabel("Training Wall-Clock Time (ms, Log Scale)")
    ax.set_title("Training Latency on California Housing ($N=20,640$, 8 features)", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar, ms in zip(bars, times_ms):
        txt = f"{ms:.1f} ms" if ms < 1000 else f"{ms/1000.0:.2f} s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            txt,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.suptitle("EXP-10: Bi-Convex ALS Delivers High Expressivity at Sub-Second Latency", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "fig_als_wallclock_comparison.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")


if __name__ == "__main__":
    res_path = "results/exp10_als_solvable_refinement/metrics.json"
    if os.path.exists(res_path):
        plot_als_results(res_path)
    else:
        print(f"File {res_path} not found.")
