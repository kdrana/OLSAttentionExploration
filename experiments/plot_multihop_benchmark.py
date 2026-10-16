"""
Plotting script for EXP-08 Multi-Hop Associative Reasoning in Series:
- Fig 1: 2-Hop Recall Accuracy vs Epochs (1-Layer vs 2-Layers vs 4-Layers)
- Fig 2: Final 2-Hop Accuracy by Network Depth (Bar Chart highlighting compositionality threshold)
- Fig 3: Training Wall-Clock Seconds vs Network Depth
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_multihop_results(
    results_path: str = None,
    out_dir: str = None,
):
    if results_path is None:
        p1 = "results/exp08_multihop_series/metrics.json"
        p2 = "results/multihop_results.json"
        results_path = p1 if os.path.exists(p1) else p2

    if out_dir is None:
        out_dir = "results/exp08_multihop_series" if os.path.exists("results/exp08_multihop_series") else "results"

    if not os.path.exists(results_path):
        print(f"File {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Standard Softmax (1-Layer)": "#e74c3c",   # Light Red
        "Standard Softmax (2-Layers)": "#c0392b",  # Dark Red
        "Delta-Net (1-Layer)": "#f39c12",          # Orange
        "Delta-Net (2-Layers)": "#d35400",         # Dark Orange
        "Flash-OLS Linear (1-Layer)": "#3498db",   # Light Blue
        "Flash-OLS Linear (2-Layers)": "#2980b9",  # Dark Blue
        "Flash-OLS Poly (1-Layer)": "#9b59b6",     # Light Purple
        "Flash-OLS Poly (2-Layers)": "#8e44ad",    # Medium Purple
        "Flash-OLS Poly (4-Layers)": "#4a235a",    # Deep Purple
    }
    linestyles = {
        1: ":",   # Dotted for 1-layer
        2: "--",  # Dashed for 2-layers
        4: "-",   # Solid for 4-layers
    }
    markers = {
        1: "x",
        2: "o",
        4: "P",
    }

    if "multihop" in data:
        mh = data["multihop"]

        # -----------------------------------------------------------------
        # Figure 1: 2-Hop Accuracy vs Epochs
        # -----------------------------------------------------------------
        plt.figure(figsize=(10, 6))

        for name, hist in mh.items():
            epochs = hist["epochs"]
            acc = [val * 100.0 for val in hist["test_acc"]]
            n_layers = hist["n_layers"]

            plt.plot(
                epochs,
                acc,
                label=name,
                color=colors.get(name, "black"),
                marker=markers.get(n_layers, "o"),
                linestyle=linestyles.get(n_layers, "-"),
                linewidth=2.6,
                markersize=6,
            )

        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("2-Hop Associative Recall Accuracy (%)", fontsize=12)
        plt.title("Compositional Reasoning in Series: 1-Layer Failure vs. Multi-Layer Induction", fontsize=13, fontweight="bold")
        plt.ylim(-2, 102)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_multihop_accuracy_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

        # -----------------------------------------------------------------
        # Figure 2: Final 2-Hop Accuracy Comparison Bar Chart
        # -----------------------------------------------------------------
        models = list(mh.keys())
        final_accs = [hist["test_acc"][-1] * 100.0 for hist in mh.values()]
        bar_colors = [colors.get(m, "gray") for m in models]

        plt.figure(figsize=(10.5, 5.5))
        bars = plt.barh(models, final_accs, color=bar_colors, edgecolor="black", alpha=0.88, height=0.6)
        plt.xlabel("Final 2-Hop Recall Accuracy (%)", fontsize=12)
        plt.title("The Compositionality Threshold: Depth in Series Solves Multi-Hop Deductions", fontsize=13, fontweight="bold")

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")

        plt.xlim(0, max(final_accs + [10]) * 1.18)
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", alpha=0.6, axis="x")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_multihop_final_accuracy_bars.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

        # -----------------------------------------------------------------
        # Figure 3: Wall-Clock Training Time vs Depth
        # -----------------------------------------------------------------
        wall_times = [hist["wallclock_sec"][-1] for hist in mh.values()]

        plt.figure(figsize=(10.5, 5))
        bars = plt.barh(models, wall_times, color=bar_colors, edgecolor="black", alpha=0.85, height=0.6)
        plt.xlabel("Total GPU Training Time (Seconds, 12 Epochs)", fontsize=12)
        plt.title("GPU Wall-Clock Training Cost vs. Depth in Series", fontsize=13, fontweight="bold")

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}s", va="center", ha="left", fontsize=10, fontweight="bold")

        plt.xlim(0, max(wall_times) * 1.18)
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", alpha=0.6, axis="x")
        plt.tight_layout()
        save_path = os.path.join(out_dir, "fig_multihop_training_wallclock.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_multihop_results()
