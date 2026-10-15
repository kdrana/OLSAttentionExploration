"""
Plotting script for Kalman & RLS Attention Benchmark (Phase 3 / EXP-04):
- Fig 1: MQAR Accuracy vs Epochs (Softmax vs Linear vs Delta vs RLS)
- Fig 2: Autoregressive Inference Memory (O(1) Flat State vs O(T) KV-Cache Growth)
- Fig 3: Per-Token Generation Latency
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_kalman_results(
    results_path: str = None,
    out_dir: str = None,
):
    if results_path is None:
        p1 = "results/exp04_kalman_rls_attention/metrics.json"
        p2 = "results/kalman_results.json"
        results_path = p1 if os.path.exists(p1) else p2

    if out_dir is None:
        out_dir = "results/exp04_kalman_rls_attention" if os.path.exists("results/exp04_kalman_rls_attention") else "results"

    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Standard Softmax": "#e74c3c",
        "Pure Linear Attention": "#95a5a6",
        "Delta / LMS Attention": "#f39c12",
        "Causal RLS / Kalman Attention": "#2ecc71",
        "KV-Cache (Standard Softmax)": "#e74c3c",
    }
    markers = {
        "Standard Softmax": "o",
        "Pure Linear Attention": "s",
        "Delta / LMS Attention": "D",
        "Causal RLS / Kalman Attention": "^",
        "KV-Cache (Standard Softmax)": "o",
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
        plt.title("MQAR Associative Recall: Kalman Innovation vs. Unnormalized Linear", fontsize=13, fontweight="bold")
        plt.ylim(-5, 105)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=10, loc="lower right")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_mqar_kalman_accuracy.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

    # -----------------------------------------------------------------
    # Figure 2: Autoregressive Inference Memory: O(1) vs O(T)
    # -----------------------------------------------------------------
    if "inference" in data:
        inf = data["inference"]
        seq_lengths = inf["seq_lengths"]
        memory = inf["memory_mb"]
        latency = inf["per_token_ms"]

        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, mems in memory.items():
            ax.plot(
                seq_lengths,
                mems,
                label=model_name,
                color=colors.get(model_name, "black"),
                marker=markers.get(model_name, "o"),
                linewidth=2.8,
                markersize=6,
            )

        ax.set_xlabel("Generated Context Length $T$ (Tokens)", fontsize=12)
        ax.set_ylabel("Inference Memory Footprint (MB)", fontsize=12)
        ax.set_title(r"Zero-KV-Cache Inference: $\mathcal{O}(1)$ Kalman State vs. $\mathcal{O}(T)$ KV-Cache", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_inference_memory_o1.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

        # Figure 3: Per-Token Latency
        fig, ax = plt.subplots(figsize=(8, 5))
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

        ax.set_xlabel("Context Length $T$", fontsize=12)
        ax.set_ylabel("Per-Token Generation Step Latency (ms)", fontsize=12)
        ax.set_title("Autoregressive Token Generation Latency", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_generation_per_token_latency.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_kalman_results()
