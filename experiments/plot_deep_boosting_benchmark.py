"""
Plotting module for EXP-09: Deep Closed-Form Boosting & Solvable Networks Benchmark.
Generates publication-ready figures:
1. fig_dsn_r2_comparison.png: Comprehensive Test R^2 comparisons across datasets.
2. fig_dsn_depth_progression.png: R^2 climbing monotonically as polynomial degree expands (2 -> 4 -> 8 -> 16 -> 32).
3. fig_dsn_training_wallclock.png: Training wall-clock latency (ms / sec) comparing DSN vs Trees vs FT-Transformer.
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


def plot_deep_boosting_results(metrics_file: str, out_dir: str = "results/exp09_deep_boosting_solvable"):
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
    # Figure 1: Test R^2 Comparison across Models and Datasets
    # =========================================================================
    models_to_plot = [
        ("Ridge Linear (Deg 1)", "Ridge (Deg 1)", "#9E9E9E"),
        ("DSN-Poly (Stage 1, Deg 2)", "DSN (Stage 1, Deg 2)", "#42A5F5"),
        ("DSN-Poly (Stage 2, Deg 4)", "DSN (Stage 2, Deg 4)", "#1E88E5"),
        ("DSN-Poly (Stage 4, Deg 16)", "DSN (Stage 4, Deg 16)", "#1565C0"),
        ("DSN-Kernel (Stage 4, Deg 16+RFF)", "DSN-Kernel (Deg 16+RFF)", "#0D47A1"),
        ("FT-Transformer (Deep Attention)", "FT-Transformer (Attention)", "#7E57C2"),
        ("XGBoost (100 Trees)", "XGBoost (100 Trees)", "#FF7043"),
        ("LightGBM (100 Trees)", "LightGBM (100 Trees)", "#E64A19"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
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
        bars = ax.barh(y_pos, r2_vals, color=colors, edgecolor="black", alpha=0.85, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # top-down
        ax.set_xlabel("Test $R^2$ Score")
        ax.set_title(ds_titles[ds], fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        # Set appropriate xlim
        min_val = min(r2_vals)
        ax.set_xlim(max(0.0, min_val - 0.08), 1.02)

        # Annotate values
        for bar, val in zip(bars, r2_vals):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle("EXP-09: Deep Solvable Networks (DSN) vs GBDTs & Deep Attention", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fig_dsn_r2_comparison.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # Figure 2: DSN Depth Progression (Degree 2 -> 4 -> 8 -> 16 -> 32)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ds_colors = {"kin8nm": "#1976D2", "friedman1": "#388E3C", "california": "#D32F2F"}

    for ax, ds in zip(axes, datasets):
        prog = data[ds]["dsn_depth_progression"]
        stages = [p["stage"] for p in prog]
        degrees = [p["effective_degree"] for p in prog]
        test_r2 = [p["test_r2"] for p in prog]
        train_r2 = [p["train_r2"] for p in prog]

        col = ds_colors[ds]
        ax.plot(stages, test_r2, marker="o", color=col, linewidth=2.5, markersize=8, label="DSN Test $R^2$")
        ax.plot(stages, train_r2, marker="s", color=col, linestyle="--", linewidth=1.5, alpha=0.6, label="DSN Train $R^2$")

        # Baselines
        xgb_r2 = data[ds]["models"]["XGBoost (100 Trees)"]["test_r2"]
        lgb_r2 = data[ds]["models"]["LightGBM (100 Trees)"]["test_r2"]
        ft_r2 = data[ds]["models"]["FT-Transformer (Deep Attention)"]["test_r2"]

        ax.axhline(xgb_r2, color="#FF7043", linestyle=":", linewidth=2, label=f"XGBoost ({xgb_r2:.4f})")
        ax.axhline(lgb_r2, color="#E64A19", linestyle="-.", linewidth=2, label=f"LightGBM ({lgb_r2:.4f})")
        ax.axhline(ft_r2, color="#7E57C2", linestyle="--", linewidth=2, label=f"FT-Trans ({ft_r2:.4f})")

        ax.set_xticks(stages)
        ax.set_xticklabels([f"S{s}\n(Deg {d})" for s, d in zip(stages, degrees)])
        ax.set_xlabel("Cascade Depth & Equivalent Polynomial Degree")
        ax.set_ylabel("Metric $R^2$")
        ax.set_title(ds_titles[ds], fontweight="bold")
        ax.legend(loc="lower right" if ds != "california" else "upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("EXP-09: Impact of Cascade Depth on Nonlinear Representational Capacity", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "fig_dsn_depth_progression.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")

    # =========================================================================
    # Figure 3: Wall-Clock Training Latency Comparison (Log Scale)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compare on California Housing (largest dataset, N=20,640)
    cal_models = data["california"]["models"]
    bench_models = [
        ("Ridge Linear (Deg 1)", "Ridge Linear", "#9E9E9E"),
        ("DSN-Poly (Stage 1, Deg 2)", "DSN (Stage 1)", "#42A5F5"),
        ("DSN-Poly (Stage 2, Deg 4)", "DSN (Stage 2)", "#1E88E5"),
        ("DSN-Poly (Stage 4, Deg 16)", "DSN (Stage 4)", "#1565C0"),
        ("DSN-Kernel (Stage 4, Deg 16+RFF)", "DSN-Kernel", "#0D47A1"),
        ("LightGBM (100 Trees)", "LightGBM", "#E64A19"),
        ("XGBoost (100 Trees)", "XGBoost", "#FF7043"),
        ("FT-Transformer (Deep Attention)", "FT-Transformer (35 ep)", "#7E57C2"),
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
    bars = ax.bar(x_pos, times_ms, color=bar_colors, edgecolor="black", alpha=0.85, width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontweight="medium")
    ax.set_ylabel("Training Wall-Clock Time (ms, Log Scale)")
    ax.set_title("Training Latency on California Housing ($N=20,640$, 8 features)", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Annotate bar values
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

    plt.suptitle("EXP-09: Closed-Form Boosting Eliminates the Deep Learning Compute Penalty", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "fig_dsn_training_wallclock.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")


if __name__ == "__main__":
    res_path = "results/exp09_deep_boosting_solvable/metrics.json"
    if os.path.exists(res_path):
        plot_deep_boosting_results(res_path)
    else:
        print(f"File {res_path} not found yet.")
