"""
Plotting script for EXP-07 Flash-OLS Ultra-Long Context Benchmark:
- Fig 1: Peak GPU Memory (MB) vs Sequence Length T (Highlighting OOM Boundaries)
- Fig 2: Forward Pass Latency (ms) vs Sequence Length T
- Fig 3: MQAR Recall Accuracy Fidelity (Naive vs Flash-OLS)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_flash_ols_results(
    results_path: str = None,
    out_dir: str = None,
):
    if results_path is None:
        p1 = "results/exp07_flash_ols_scaling/metrics.json"
        p2 = "results/flash_ols_results.json"
        results_path = p1 if os.path.exists(p1) else p2

    if out_dir is None:
        out_dir = "results/exp07_flash_ols_scaling" if os.path.exists("results/exp07_flash_ols_scaling") else "results"

    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Standard Softmax": "#e74c3c",           # Red
        "Naive Exact OLS": "#e67e22",            # Orange
        "Flash-OLS (Linear)": "#2980b9",         # Blue
        "Flash-OLS (Poly)": "#8e44ad",           # Purple
        "Flash-OLS (Linear, C=32)": "#2980b9",
        "Naive Exact Poly OLS": "#d35400",
        "Flash-OLS (Poly, C=32)": "#8e44ad",
    }
    markers = {
        "Standard Softmax": "o",
        "Naive Exact OLS": "^",
        "Flash-OLS (Linear)": "s",
        "Flash-OLS (Poly)": "P",
        "Flash-OLS (Linear, C=32)": "s",
        "Naive Exact Poly OLS": "D",
        "Flash-OLS (Poly, C=32)": "P",
    }
    linestyles = {
        "Standard Softmax": ":",
        "Naive Exact OLS": "--",
        "Flash-OLS (Linear)": "-",
        "Flash-OLS (Poly)": "-",
        "Flash-OLS (Linear, C=32)": "-",
        "Naive Exact Poly OLS": "--",
        "Flash-OLS (Poly, C=32)": "-",
    }

    # -----------------------------------------------------------------
    # Figure 1: Peak GPU Memory vs Sequence Length T
    # -----------------------------------------------------------------
    if "scaling" in data:
        scaling = data["scaling"]
        seq_lengths = scaling["seq_lengths"]
        peak_mems = scaling["peak_memory_mb"]
        latencies = scaling["latency_ms"]

        fig, ax = plt.subplots(figsize=(9.5, 5.5))

        for name, mems in peak_mems.items():
            valid_x = []
            valid_y = []
            oom_x = None
            for x_val, y_val in zip(seq_lengths, mems):
                if y_val is not None:
                    valid_x.append(x_val)
                    valid_y.append(y_val)
                elif oom_x is None:
                    oom_x = x_val

            if valid_x:
                ax.plot(
                    valid_x,
                    valid_y,
                    label=name,
                    color=colors.get(name, "black"),
                    marker=markers.get(name, "o"),
                    linestyle=linestyles.get(name, "-"),
                    linewidth=2.8,
                    markersize=7,
                )

            # Mark OOM point if occurred
            if oom_x is not None:
                ax.scatter(
                    [oom_x],
                    [valid_y[-1] * 1.5 if valid_y else 500],
                    color=colors.get(name, "red"),
                    marker="X",
                    s=140,
                    zorder=5,
                )
                ax.text(
                    oom_x * 1.05,
                    (valid_y[-1] * 1.5 if valid_y else 500),
                    f"{name} OOM",
                    color=colors.get(name, "red"),
                    fontweight="bold",
                    fontsize=9.5,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence Length $T$ (Tokens, Log Scale)", fontsize=12)
        ax.set_ylabel("Peak GPU Memory Allocated (MB, Log Scale)", fontsize=12)
        ax.set_title("Peak GPU Memory Scaling: Flash-OLS vs. Quadratic & Naive OOM", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_flash_ols_memory_scaling.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

        # -----------------------------------------------------------------
        # Figure 2: Latency vs Sequence Length T
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))

        for name, lats in latencies.items():
            valid_x = []
            valid_y = []
            for x_val, y_val in zip(seq_lengths, lats):
                if y_val is not None:
                    valid_x.append(x_val)
                    valid_y.append(y_val)

            if valid_x:
                ax.plot(
                    valid_x,
                    valid_y,
                    label=name,
                    color=colors.get(name, "black"),
                    marker=markers.get(name, "o"),
                    linestyle=linestyles.get(name, "-"),
                    linewidth=2.8,
                    markersize=7,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence Length $T$ (Tokens, Log Scale)", fontsize=12)
        ax.set_ylabel("Forward Pass Latency (ms, Log Scale)", fontsize=12)
        ax.set_title("Runtime Scaling: Linear $\\mathcal{O}(T)$ Flash-OLS vs. Quadratic Attention", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_flash_ols_latency_scaling.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

    # -----------------------------------------------------------------
    # Figure 3: MQAR Recall Fidelity (Naive vs Flash-OLS)
    # -----------------------------------------------------------------
    if "fidelity" in data:
        fidelity = data["fidelity"]
        plt.figure(figsize=(9, 5.5))

        for model_name, hist in fidelity.items():
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
        plt.title("Mathematical Fidelity Verification: Flash-OLS vs. Naive Exact OLS", fontsize=13, fontweight="bold")
        plt.ylim(-2, 102)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_flash_ols_fidelity_verification.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_flash_ols_results()
