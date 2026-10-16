"""
Plotting script for Solvable Attention Architectures Suite Benchmark (Phase 4 / EXP-05):
- Fig 1: MQAR Accuracy vs Epochs across 7 Attention Architectures
- Fig 2: Training GPU Wall-Clock Seconds & Efficiency
- Fig 3: Autoregressive Inference Memory (O(1) Flat States vs O(T) KV-Cache Growth)
- Fig 4: Per-Token Generation Latency Scaling
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_solvable_results(
    results_path: str = None,
    out_dir: str = None,
):
    if results_path is None:
        p1 = "results/exp05_solvable_attention_suite/metrics.json"
        p2 = "results/solvable_results.json"
        results_path = p1 if os.path.exists(p1) else p2

    if out_dir is None:
        out_dir = "results/exp05_solvable_attention_suite" if os.path.exists("results/exp05_solvable_attention_suite") else "results"

    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Standard Softmax": "#e74c3c",           # Red
        "Pure Linear Attention": "#95a5a6",       # Grey
        "Delta / LMS Attention": "#f39c12",       # Orange
        "Causal RLS / Kalman": "#2ecc71",         # Emerald Green
        "Chunkwise Ridge Attention": "#2980b9",   # Strong Blue
        "Diagonal Kalman Attention": "#8e44ad",   # Purple
        "Chunkwise Poly Solvable": "#16a085",     # Teal
        "KV-Cache (Standard Softmax)": "#e74c3c",
    }
    markers = {
        "Standard Softmax": "o",
        "Pure Linear Attention": "s",
        "Delta / LMS Attention": "D",
        "Causal RLS / Kalman": "^",
        "Chunkwise Ridge Attention": "P",
        "Diagonal Kalman Attention": "X",
        "Chunkwise Poly Solvable": "*",
        "KV-Cache (Standard Softmax)": "o",
    }

    # -----------------------------------------------------------------
    # Figure 1: MQAR Accuracy vs Epochs (7 Models)
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
                linewidth=2.5,
                markersize=6,
            )

        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("Multi-Query Recall Accuracy (%)", fontsize=12)
        plt.title("MQAR Accuracy: Traditional vs. Kalman vs. Closed-Form Solvable", fontsize=13, fontweight="bold")
        plt.ylim(-2, 102)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_mqar_solvable_accuracy.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

    # -----------------------------------------------------------------
    # Figure 2: Training GPU Wall-Clock Seconds
    # -----------------------------------------------------------------
    if "mqar" in data:
        mqar = data["mqar"]
        models = list(mqar.keys())
        total_times = [hist["wallclock_sec"][-1] for hist in mqar.values()]
        bar_colors = [colors.get(m, "gray") for m in models]

        plt.figure(figsize=(9, 5))
        bars = plt.barh(models, total_times, color=bar_colors, edgecolor="black", alpha=0.85, height=0.6)
        plt.xlabel("Total GPU Training Time (Seconds, 10 Epochs)", fontsize=12)
        plt.title("Training Efficiency: Closed-Form Solvable vs. Recurrent Loops", fontsize=13, fontweight="bold")

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}s", va="center", ha="left", fontsize=10, fontweight="bold")

        plt.xlim(0, max(total_times) * 1.18)
        plt.gca().invert_yaxis()  # Highest model at top
        plt.grid(True, linestyle="--", alpha=0.6, axis="x")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_solvable_training_wallclock.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

    # -----------------------------------------------------------------
    # Figure 3: Autoregressive Inference Memory: O(1) vs O(T)
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
                linewidth=2.8,
                markersize=7,
            )

        ax.set_xlabel("Sequence Length $T$ (Tokens)", fontsize=12)
        ax.set_ylabel("Inference State Memory Footprint (MB)", fontsize=12)
        ax.set_title("Zero-KV-Cache Serving: Constant O(1) Memory vs. Softmax Explosion", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=10, loc="upper left")

        # Annotation at T=4096
        kv_4096 = memory["KV-Cache (Standard Softmax)"][-1]
        ax.annotate(
            f"KV-Cache: {kv_4096:.1f} MB (O(T) linear growth)",
            xy=(4096, kv_4096),
            xytext=(2600, kv_4096 * 0.85),
            arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10,
            fontweight="bold",
            color="#c0392b",
        )
        diag_4096 = memory["Diagonal Kalman Attention"][-1]
        ax.annotate(
            f"Diagonal Kalman: {diag_4096:.4f} MB (Strict O(1) Flatline)",
            xy=(4096, diag_4096),
            xytext=(2000, 0.4),
            arrowprops=dict(facecolor="#8e44ad", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10,
            fontweight="bold",
            color="#8e44ad",
        )

        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_solvable_inference_memory.png")
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
                linewidth=2.5,
                markersize=6,
            )

        ax.set_xlabel("Sequence Length $T$ (Tokens)", fontsize=12)
        ax.set_ylabel("Per-Token Generation Latency (ms)", fontsize=12)
        ax.set_title("Autoregressive Token Generation Latency vs. Sequence Length", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_solvable_per_token_latency.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_solvable_results()
