"""
Plotting script for EXP-06 Unified vs Pieces Attention Benchmark:
- Fig 1: MQAR Accuracy vs Epochs (Traditional vs Pieces vs Unified)
- Fig 2: Training GPU Wall-Clock Seconds & Efficiency
- Fig 3: Autoregressive Inference Memory (O(1) Flat States vs O(T) KV-Cache Growth)
- Fig 4: Per-Token Generation Latency Scaling
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_unified_results(
    results_path: str = None,
    out_dir: str = None,
):
    if results_path is None:
        p1 = "results/exp06_unified_vs_pieces/metrics.json"
        p2 = "results/unified_results.json"
        results_path = p1 if os.path.exists(p1) else p2

    if out_dir is None:
        out_dir = "results/exp06_unified_vs_pieces" if os.path.exists("results/exp06_unified_vs_pieces") else "results"

    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Traditional (Standard Softmax)": "#e74c3c",  # Red
        "Traditional (KV-Cache)": "#e74c3c",
        "Pieces: Delta / LMS Attention": "#f39c12",   # Orange
        "Pieces: Causal RLS / Kalman": "#2ecc71",     # Green
        "Pieces: Causal RLS": "#2ecc71",
        "Pieces: Chunkwise Poly Solvable": "#16a085", # Teal
        "Pieces: Chunkwise Poly": "#16a085",
        "Unified: Exact Causal OLS": "#2980b9",       # Blue
        "Unified: Exact Poly OLS": "#8e44ad",         # Purple
    }
    markers = {
        "Traditional (Standard Softmax)": "o",
        "Traditional (KV-Cache)": "o",
        "Pieces: Delta / LMS Attention": "D",
        "Pieces: Causal RLS / Kalman": "^",
        "Pieces: Causal RLS": "^",
        "Pieces: Chunkwise Poly Solvable": "*",
        "Pieces: Chunkwise Poly": "*",
        "Unified: Exact Causal OLS": "s",
        "Unified: Exact Poly OLS": "P",
    }
    linestyles = {
        "Traditional (Standard Softmax)": ":",
        "Traditional (KV-Cache)": ":",
        "Pieces: Delta / LMS Attention": "--",
        "Pieces: Causal RLS / Kalman": "--",
        "Pieces: Causal RLS": "--",
        "Pieces: Chunkwise Poly Solvable": "--",
        "Pieces: Chunkwise Poly": "--",
        "Unified: Exact Causal OLS": "-",
        "Unified: Exact Poly OLS": "-",
    }

    # -----------------------------------------------------------------
    # Figure 1: MQAR Accuracy vs Epochs (3 Paradigms)
    # -----------------------------------------------------------------
    if "mqar" in data:
        mqar = data["mqar"]
        plt.figure(figsize=(9, 5.5))

        for model_name, hist in mqar.items():
            epochs = hist["epochs"]
            acc = [val * 100.0 for val in hist["test_acc"]]
            plt.plot(
                epochs,
                acc,
                label=model_name,
                color=colors.get(model_name, "black"),
                marker=markers.get(model_name, "o"),
                linestyle=linestyles.get(model_name, "-"),
                linewidth=2.6,
                markersize=6,
            )

        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("Multi-Query Recall Accuracy (%)", fontsize=12)
        plt.title("MQAR Recall: Traditional vs. Pieces vs. Unified Mathematical Solution", fontsize=13, fontweight="bold")
        plt.ylim(-2, 102)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_mqar_unified_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

        # -----------------------------------------------------------------
        # Figure 2: Training GPU Wall-Clock Seconds
        # -----------------------------------------------------------------
        models = list(mqar.keys())
        total_times = [hist["wallclock_sec"][-1] for hist in mqar.values()]
        bar_colors = [colors.get(m, "gray") for m in models]

        plt.figure(figsize=(9.5, 5))
        bars = plt.barh(models, total_times, color=bar_colors, edgecolor="black", alpha=0.85, height=0.6)
        plt.xlabel("Total GPU Training Time (Seconds, 10 Epochs)", fontsize=12)
        plt.title("Training Wall-Clock Efficiency: Unified Parallel Solve vs Recurrence", fontsize=13, fontweight="bold")

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}s", va="center", ha="left", fontsize=10, fontweight="bold")

        plt.xlim(0, max(total_times) * 1.18)
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", alpha=0.6, axis="x")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_unified_training_wallclock.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

    # -----------------------------------------------------------------
    # Figure 3: Autoregressive Inference Memory
    # -----------------------------------------------------------------
    if "inference" in data:
        inf = data["inference"]
        seq_lengths = inf["seq_lengths"]
        memory = inf["memory_mb"]
        latency = inf["per_token_ms"]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for model_name, mems in memory.items():
            ax.plot(
                seq_lengths,
                mems,
                label=model_name,
                color=colors.get(model_name, "black"),
                marker=markers.get(model_name, "o"),
                linestyle=linestyles.get(model_name, "-"),
                linewidth=2.8,
                markersize=7,
            )

        ax.set_xlabel("Sequence Length $T$ (Tokens)", fontsize=12)
        ax.set_ylabel("Inference State Memory Footprint (MB)", fontsize=12)
        ax.set_title("Zero-KV-Cache Inference: Unified State vs. Softmax Explosion", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=10, loc="upper left")

        # Annotations
        kv_4096 = memory["Traditional (KV-Cache)"][-1]
        ax.annotate(
            f"KV-Cache: {kv_4096:.1f} MB (O(T) linear growth)",
            xy=(4096, kv_4096),
            xytext=(2600, kv_4096 * 0.85),
            arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10,
            fontweight="bold",
            color="#c0392b",
        )
        ols_4096 = memory["Unified: Exact Causal OLS"][-1]
        ax.annotate(
            f"Unified OLS: {ols_4096:.4f} MB (Strict O(1) Flatline)",
            xy=(4096, ols_4096),
            xytext=(2000, 0.4),
            arrowprops=dict(facecolor="#2980b9", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10,
            fontweight="bold",
            color="#2980b9",
        )

        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_unified_inference_memory.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

        # -----------------------------------------------------------------
        # Figure 4: Per-Token Generation Latency
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for model_name, lats in latency.items():
            ax.plot(
                seq_lengths,
                lats,
                label=model_name,
                color=colors.get(model_name, "black"),
                marker=markers.get(model_name, "o"),
                linestyle=linestyles.get(model_name, "-"),
                linewidth=2.5,
                markersize=6,
            )

        ax.set_xlabel("Sequence Length $T$ (Tokens)", fontsize=12)
        ax.set_ylabel("Per-Token Generation Latency (ms)", fontsize=12)
        ax.set_title("Autoregressive Generation Latency vs. Sequence Length", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_unified_per_token_latency.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_unified_results()
