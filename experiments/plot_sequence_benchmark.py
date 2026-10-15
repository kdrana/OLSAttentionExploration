"""
Plotting script for Sequence Benchmark (EXP-03):
- Fig 1: Multi-Query Associative Recall (MQAR) Accuracy vs Epochs
- Fig 2: Forward+Backward Latency (ms) vs Sequence Length T
- Fig 3: Peak GPU Memory (MB) vs Sequence Length T
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_sequence_results(
    results_path: str = None,
    out_dir: str = None,
):
    if results_path is None:
        p1 = "results/exp03_sequence_bulk_margin/metrics.json"
        p2 = "results/sequence_results.json"
        results_path = p1 if os.path.exists(p1) else p2

    if out_dir is None:
        out_dir = "results/exp03_sequence_bulk_margin" if os.path.exists("results/exp03_sequence_bulk_margin") else "results"

    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Standard Softmax": "#e74c3c",
        "Pure Linear Attention": "#3498db",
        "Two-Stream Bulk+Margin": "#2ecc71",
    }
    markers = {
        "Standard Softmax": "o",
        "Pure Linear Attention": "s",
        "Two-Stream Bulk+Margin": "^",
    }

    # -----------------------------------------------------------------
    # Figure 1: MQAR Accuracy vs Epochs
    # -----------------------------------------------------------------
    if "mqar" in data:
        mqar = data["mqar"]
        plt.figure(figsize=(8, 5))

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
                markersize=5,
            )

        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("Multi-Query Recall Accuracy (%)", fontsize=12)
        plt.title("Multi-Query Associative Recall (MQAR, T=512)", fontsize=13, fontweight="bold")
        plt.ylim(-5, 105)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=11, loc="lower right")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_mqar_accuracy.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

    # -----------------------------------------------------------------
    # Figure 2 & 3: Sequence Length Scaling (Latency & Memory)
    # -----------------------------------------------------------------
    if "scaling" in data:
        scaling = data["scaling"]
        seq_lengths = scaling["seq_lengths"]
        latencies = scaling["latency_ms"]
        memory = scaling["memory_mb"]

        # Latency Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, times in latencies.items():
            ax.plot(
                seq_lengths,
                times,
                label=model_name,
                color=colors.get(model_name, "black"),
                marker=markers.get(model_name, "o"),
                linewidth=2.5,
                markersize=6,
            )

        ax.set_xlabel("Sequence Length $T$", fontsize=12)
        ax.set_ylabel("Forward + Backward Latency (ms)", fontsize=12)
        ax.set_title("Computational Scaling: Execution Time vs. Sequence Length", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_scaling_latency.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

        # Memory Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, mems in memory.items():
            ax.plot(
                seq_lengths,
                mems,
                label=model_name,
                color=colors.get(model_name, "black"),
                marker=markers.get(model_name, "o"),
                linewidth=2.5,
                markersize=6,
            )

        ax.set_xlabel("Sequence Length $T$", fontsize=12)
        ax.set_ylabel("Peak Allocated GPU Memory (MB)", fontsize=12)
        ax.set_title("Memory Scaling: Peak Allocation vs. Sequence Length", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_scaling_memory.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_sequence_results()
