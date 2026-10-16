"""
Plotting script for EXP-13: Fast Analytical Autograd & Block-Diagonal Subspace Acceleration Benchmark.
Generates Figures 33, 34, and 35 in results/exp13_fast_ols_acceleration/.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_fast_ols_results(metrics_path: str, out_dir: str):
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

    # Available models
    model_styles = {
        "Softmax SLM (Llama-Nano)": {"color": "#e74c3c", "marker": "o", "label": "Softmax SLM (FlashAttention)"},
        "Flash-OLS SLM (Baseline V2)": {"color": "#7f8c8d", "marker": "x", "label": "Flash-OLS SLM (Baseline V2 - Dense r=32)"},
        "Fast-Flash-OLS SLM (Fast V3)": {"color": "#2ecc71", "marker": "s", "label": "Fast-Flash-OLS SLM (V3 - Block-Diag + Autograd)"},
    }

    # -------------------------------------------------------------
    # Figure 33: Validation & Training Loss Curves
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

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
    ax1.legend(frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title(r"Training Loss ($\mathcal{L}_{\mathrm{train}}$) vs. Optimization Steps")
    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Cross-Entropy Loss (Nats)")
    ax2.legend(frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig33_path = os.path.join(out_dir, "fig33_fast_ols_loss_curves.png")
    plt.savefig(fig33_path)
    plt.close()
    print(f"[OK] Saved {fig33_path}")

    # -------------------------------------------------------------
    # Figure 34: Validation Perplexity Trajectory
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    for name, res in data.items():
        style = model_styles.get(name, {"color": "#3498db", "marker": "^", "label": name})
        trace = res["trace"]
        val_steps = [e["step"] for e in trace]
        val_ppl = [e["val_ppl"] for e in trace]
        ax.plot(val_steps, val_ppl, f"{style['marker']}-", color=style["color"], linewidth=2.4, label=style["label"], markersize=7)

        # Annotate final PPL
        if val_ppl:
            ax.annotate(
                f"{val_ppl[-1]:.1f}",
                (val_steps[-1], val_ppl[-1]),
                textcoords="offset points",
                xytext=(-10, 8 if "Fast" in name else -12),
                ha="right",
                fontweight="bold",
                color=style["color"]
            )

    ax.set_yscale("log")
    ax.set_title(r"Validation Perplexity ($\mathrm{PPL}_{\mathrm{val}}$) Trajectory" + "\nExact Closed-Form Autograd & Block-Diagonal Subspace", pad=12)
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Perplexity (Log Scale)")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig34_path = os.path.join(out_dir, "fig34_fast_ols_perplexity_comparison.png")
    plt.savefig(fig34_path)
    plt.close()
    print(f"[OK] Saved {fig34_path}")

    # -------------------------------------------------------------
    # Figure 35: Step Latency & Hardware Throughput Speedup Pareto
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    model_names = list(data.keys())
    short_names = [m.split("(")[0].strip().replace(" SLM", "") for m in model_names]
    step_times = [data[m]["mean_step_time_ms"] for m in model_names]
    throughputs = [data[m]["tokens_per_sec"] for m in model_names]
    colors = [model_styles.get(m, {}).get("color", "#3498db") for m in model_names]

    # Step Latency Bar Chart
    bars1 = ax1.bar(short_names, step_times, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f} ms", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Mean Step Latency (ms) [Lower is Better]")
    ax1.set_title("Training Step Execution Latency")
    ax1.set_ylim(0, max(step_times) * 1.25)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Throughput Bar Chart
    bars2 = ax2.bar(short_names, throughputs, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:,.0f} tok/s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Throughput (Tokens / Sec) [Higher is Better]")
    ax2.set_title("Pretraining Hardware Throughput")
    ax2.set_ylim(0, max(throughputs) * 1.25)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig35_path = os.path.join(out_dir, "fig35_fast_ols_training_speedup_pareto.png")
    plt.savefig(fig35_path)
    plt.close()
    print(f"[OK] Saved {fig35_path}")


if __name__ == "__main__":
    plot_fast_ols_results("results/exp13_fast_ols_acceleration/metrics.json", "results/exp13_fast_ols_acceleration")
