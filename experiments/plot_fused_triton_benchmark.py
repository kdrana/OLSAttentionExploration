"""
Plotting script for EXP-14: Fused Triton Flash-OLS Kernel Acceleration Benchmark.
Generates Figures 36, 37, and 38 in results/exp14_fused_triton_ols/.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_fused_triton_results(metrics_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(metrics_path, "r") as f:
        data = json.load(f)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    })

    model_styles = {
        "Softmax SLM (Llama-Nano)": {"color": "#e74c3c", "marker": "o", "label": "Softmax SLM (FlashAttention)"},
        "Flash-OLS SLM (Baseline V2)": {"color": "#7f8c8d", "marker": "x", "label": "Flash-OLS (Baseline V2 - Dense r=32)"},
        "Fast-Flash-OLS SLM (Fast V3)": {"color": "#f39c12", "marker": "s", "label": "Fast-Flash-OLS (V3 - Block-Diag + Autograd)"},
        "Fused-Flash-OLS SLM (Triton V4)": {"color": "#27ae60", "marker": "D", "label": "Fused-Flash-OLS (V4 - In-SRAM Triton Kernel)"},
    }

    # -------------------------------------------------------------
    # Figure 36: Validation & Training Loss Curves
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), dpi=300)

    for name, res in data.items():
        style = model_styles.get(name, {"color": "#3498db", "marker": "^", "label": name})
        trace = res["trace"]
        val_steps = [e["step"] for e in trace]
        val_loss = [e["val_loss"] for e in trace]
        ax1.plot(val_steps, val_loss, f"{style['marker']}-", color=style["color"], linewidth=2.2, label=style["label"], markersize=6)

        tr_steps = [e["step"] for e in trace if e.get("train_loss") is not None]
        tr_loss = [e["train_loss"] for e in trace if e.get("train_loss") is not None]
        if tr_steps:
            ax2.plot(tr_steps, tr_loss, f"{style['marker']}--", color=style["color"], alpha=0.85, linewidth=1.8, label=style["label"])

    ax1.set_title(r"Validation Loss ($\mathcal{L}_{\mathrm{val}}$) vs. Optimization Steps")
    ax1.set_xlabel("Optimization Steps")
    ax1.set_ylabel("Cross-Entropy Loss (Nats)")
    ax1.legend(frameon=True, fontsize=9.0)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title(r"Training Loss ($\mathcal{L}_{\mathrm{train}}$) vs. Optimization Steps")
    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Cross-Entropy Loss (Nats)")
    ax2.legend(frameon=True, fontsize=9.0)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig36_path = os.path.join(out_dir, "fig36_fused_triton_loss_curves.png")
    plt.savefig(fig36_path)
    plt.close()
    print(f"[OK] Saved {fig36_path}")

    # -------------------------------------------------------------
    # Figure 37: Validation Perplexity Trajectory
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    for name, res in data.items():
        style = model_styles.get(name, {"color": "#3498db", "marker": "^", "label": name})
        trace = res["trace"]
        val_steps = [e["step"] for e in trace]
        val_ppl = [e["val_ppl"] for e in trace]
        ax.plot(val_steps, val_ppl, f"{style['marker']}-", color=style["color"], linewidth=2.4, label=style["label"], markersize=7)

        if val_ppl:
            ax.annotate(
                f"{val_ppl[-1]:.1f}",
                (val_steps[-1], val_ppl[-1]),
                textcoords="offset points",
                xytext=(-10, 8 if "Triton" in name or "Fast" in name else -12),
                ha="right",
                fontweight="bold",
                color=style["color"]
            )

    ax.set_yscale("log")
    ax.set_title(r"Validation Perplexity ($\mathrm{PPL}_{\mathrm{val}}$) Trajectory" + "\nFused Triton In-SRAM Recurrence vs Baselines", pad=12)
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Perplexity (Log Scale)")
    ax.legend(frameon=True, loc="upper right", fontsize=9.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig37_path = os.path.join(out_dir, "fig37_fused_triton_perplexity_comparison.png")
    plt.savefig(fig37_path)
    plt.close()
    print(f"[OK] Saved {fig37_path}")

    # -------------------------------------------------------------
    # Figure 38: Training Speedup & Throughput Progression
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    models_order = [
        "Flash-OLS SLM (Baseline V2)",
        "Fast-Flash-OLS SLM (Fast V3)",
        "Fused-Flash-OLS SLM (Triton V4)",
        "Softmax SLM (Llama-Nano)",
    ]

    labels = []
    step_times = []
    throughputs = []
    bar_colors = []

    for m in models_order:
        if m in data:
            labels.append(m.split("(")[0].strip())
            step_times.append(data[m]["mean_step_time_ms"])
            throughputs.append(data[m]["tokens_per_sec"])
            bar_colors.append(model_styles[m]["color"])

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, step_times, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.1)
    ax1.set_title("Step Latency (Lower is Better)")
    ax1.set_ylabel("Mean Step Latency (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    baseline_time = data["Flash-OLS SLM (Baseline V2)"]["mean_step_time_ms"]
    for bar, st in zip(bars1, step_times):
        h = bar.get_height()
        speedup = baseline_time / max(st, 1e-5)
        ax1.annotate(f"{st:.1f} ms\n({speedup:.2f}x)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, throughputs, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.1)
    ax2.set_title("Training Throughput (Higher is Better)")
    ax2.set_ylabel("Tokens / Second")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar, tp in zip(bars2, throughputs):
        h = bar.get_height()
        ax2.annotate(f"{tp:,.0f} tok/s",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Pretraining Acceleration: Baseline V2 → Analytical V3 → Fused Triton V4 vs Softmax", fontsize=13, y=1.02)
    plt.tight_layout()
    fig38_path = os.path.join(out_dir, "fig38_fused_triton_speedup_pareto.png")
    plt.savefig(fig38_path)
    plt.close()
    print(f"[OK] Saved {fig38_path}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/exp14_fused_triton_ols/metrics.json"
    plot_fused_triton_results(path, os.path.dirname(path))
