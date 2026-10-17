"""
Modal Cloud Runner for EXP-23: Qwen-3.5-0.8B Native Solvable Attention Replacement & LoRA Alignment.

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300s.
Saves:
- results/exp23_qwen_nsal_alignment/metrics.json
- results/exp23_qwen_nsal_alignment/fig61_qwen_nsal_alignment_loss.png
- results/exp23_qwen_nsal_alignment/fig62_qwen_nsal_cache_reduction.png
- results/exp23_qwen_nsal_alignment/fig63_qwen_nsal_benchmark_comparison.png
"""

import os
import io
import json
import modal

app = modal.App("ols-qwen-nsal-alignment")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_exp23_worker() -> dict:
    import torch
    from experiments.run_qwen_nsal_alignment import run_qwen_nsal_alignment

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")
    return run_qwen_nsal_alignment(device=device, train_steps=150, lr=2e-4, r=32)


@app.function(image=image, timeout=60)
def generate_exp23_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    # -------------------------------------------------------------
    # Figure 61: Training Loss Convergence
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    loss_curve = metrics["loss_history"]
    steps = [i * 5 for i in range(len(loss_curve))]

    ax.plot(steps, loss_curve, linewidth=2.5, color="#1b9e77", marker="o", markersize=4, label=f"Qwen-NSAL Cross-Entropy (Final: {metrics['final_loss']:.4f})")
    ax.set_xlabel("Optimization Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_title("EXP-23: Qwen-3.5-0.8B Native Solvable Attention Alignment Loss", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig61_qwen_nsal_alignment_loss.png"] = buf.getvalue()
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 62: Attention State Cache Compression (Log Scale)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    categories = ["Softmax KV-Cache (32k)", "CDCA-V2 Adapter (32k)", "Native Qwen-NSAL (O(1))"]
    mem_kb = [
        metrics["cache_memory"]["softmax_kv_32k_kb"],
        24.0 * 1024,  # 24 MB from EXP-21
        metrics["cache_memory"]["qwen_nsal_state_kb"],  # 432 KB
    ]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    bars = ax.bar(categories, mem_kb, color=colors, width=0.55, edgecolor="#333333", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Attention Cache Memory (KB, Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("EXP-23: Generation Attention State Memory Footprint", fontsize=12, fontweight="bold", pad=12)

    # Annotate bars
    for bar, val in zip(bars, mem_kb):
        yval = bar.get_height()
        if yval >= 1024:
            txt = f"{yval / 1024:.1f} MB"
        else:
            txt = f"{yval:.0f} KB"
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.3, txt, ha="center", va="bottom", fontsize=10, fontweight="bold")

    reduction = metrics["cache_memory"]["memory_reduction_factor"]
    ax.text(2, mem_kb[2] * 4.0, f"{reduction:.0f}x\nReduction", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1b9e77")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig62_qwen_nsal_cache_reduction.png"] = buf.getvalue()
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 63: MMLU Accuracy and Benchmark Retention
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    models = ["Original Softmax", "CDCA-V2 (EXP-21)", "Native Qwen-NSAL (EXP-23)"]
    scores = [100.0, 100.0, metrics["mmlu_arc_pct"]]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    bars = ax.bar(models, scores, color=colors, width=0.5, edgecolor="#333333", linewidth=1.2)
    ax.set_ylabel("MMLU / ARC-Challenge Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title("EXP-23: Scientific Reasoning Retention Post-NSAL Alignment", fontsize=12, fontweight="bold", pad=12)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 2.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig63_qwen_nsal_benchmark_comparison.png"] = buf.getvalue()
    plt.close(fig)

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print(" Triggering Modal Cloud Run for EXP-23: Qwen-3.5-0.8B NSAL Alignment")
    print(" Target: NVIDIA A10G Cloud GPU (timeout <= 300s)")
    print("==================================================================")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp23_qwen_nsal_alignment")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Run Worker on Cloud GPU
    metrics = run_exp23_worker.remote()
    print("\nCloud benchmark finished! Generating publication figures...")

    # 2. Generate Plots on Cloud
    plots = generate_exp23_plots_cloud.remote(metrics)

    # 3. Save Metrics locally
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Saved Metrics]: {metrics_path}")

    # 4. Save Plots locally
    for fname, pbytes in plots.items():
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(pbytes)
        print(f"[Saved Plot]: {fpath}")

    # Print Summary
    print("\n==================================================================")
    print(" EXP-23 BENCHMARK SUMMARY (Qwen-3.5-0.8B + Native Solvable Attention)")
    print("==================================================================")
    print(f"  Final Cross-Entropy Loss: {metrics['final_loss']:.4f} (from {metrics['initial_loss']:.4f})")
    print(f"  MMLU / ARC Accuracy:      {metrics['mmlu_arc_pct']:.1f}%")
    print(f"  Generation Sample:        \"{metrics['sample_generation']['generated_text']}\"")
    print(f"  Attention Cache Memory:   {metrics['cache_memory']['qwen_nsal_state_kb']:.1f} KB (vs {metrics['cache_memory']['softmax_kv_32k_kb']:.1f} KB Softmax -> {metrics['cache_memory']['memory_reduction_factor']:.0f}x smaller)")
    print("==================================================================")


if __name__ == "__main__":
    main()
