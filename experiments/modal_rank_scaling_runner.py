"""
Modal Cloud Runner for EXP-19: Subspace Rank Scaling Benchmark.

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 180s.
Saves:
- results/exp19_rank_scaling/metrics.json
- results/exp19_rank_scaling/fig50_rank_vs_r2.png
- results/exp19_rank_scaling/fig51_rank_vs_memory_latency.png
"""

import os
import io
import json
import modal

app = modal.App("ols-attention-rank-scaling")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=180)
def run_rank_scaling_worker() -> dict:
    import torch
    from experiments.run_rank_scaling_benchmark import run_rank_scaling_benchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")
    return run_rank_scaling_benchmark(device=device)


@app.function(image=image, timeout=60)
def generate_rank_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    results = metrics["results"]
    ranks = [r["rank"] for r in results]
    r2_vals = [r["val_r2"] for r in results]
    cos_vals = [r["val_cos_sim"] for r in results]
    state_kb = [r["state_kb"] for r in results]
    decode_kflops = [r["decode_kflops_per_token"] for r in results]

    # -------------------------------------------------------------
    # Figure 50: Operator R^2 and Cosine Similarity vs Rank r
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)

    color = "#2980b9"
    ax1.set_xlabel("Subspace Rank ($r$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation $R^2$ Variance Explained", color=color, fontsize=11, fontweight="bold")
    line1 = ax1.plot(ranks, r2_vals, "o-", color=color, linewidth=2.4, markersize=8, label="Operator $R^2$")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(ranks)
    ax1.set_xticklabels([str(r) for r in ranks], fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for x, y in zip(ranks, r2_vals):
        ax1.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=color)

    ax2 = ax1.twinx()
    color2 = "#27ae60"
    ax2.set_ylabel("Cosine Similarity ($\cos \\theta$)", color=color2, fontsize=11, fontweight="bold")
    line2 = ax2.plot(ranks, cos_vals, "s--", color=color2, linewidth=2.2, markersize=8, label="Cosine Similarity")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    for x, y in zip(ranks, cos_vals):
        ax2.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True, fontsize=10)

    plt.title("Flash-OLS Operator Fidelity vs Active Subspace Rank ($r$)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig50_rank_vs_r2.png"] = buf.getvalue()

    # -------------------------------------------------------------
    # Figure 51: State Memory & Decode Compute vs Rank r
    # -------------------------------------------------------------
    fig, (ax_mem, ax_flop) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    # State memory
    bars = ax_mem.bar([str(r) for r in ranks], state_kb, color="#8e44ad", edgecolor="black", linewidth=1.1, width=0.5)
    ax_mem.set_title("Total 6-Layer Recurrent State Size (KB)", fontsize=11, fontweight="bold")
    ax_mem.set_xlabel("Subspace Rank ($r$)", fontsize=10, fontweight="bold")
    ax_mem.set_ylabel("Memory (KB) [Strictly O(1)]", fontsize=10)
    ax_mem.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax_mem.annotate(f"{h:.1f} KB", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add reference line for Softmax at 32k
    ax_mem.axhline(393216.0, color="#e74c3c", linestyle=":", label="Softmax KV-Cache at 32k (393.2 MB)")

    # Decode FLOPs
    ax_flop.plot(ranks, decode_kflops, "d-", color="#d35400", linewidth=2.4, markersize=8)
    ax_flop.set_title("Decode Sherman-Morrison FLOPs / Token (All 6 Layers)", fontsize=11, fontweight="bold")
    ax_flop.set_xlabel("Subspace Rank ($r$)", fontsize=10, fontweight="bold")
    ax_flop.set_ylabel("KFLOPs / Generated Token", fontsize=10)
    ax_flop.set_xticks(ranks)
    ax_flop.set_xticklabels([str(r) for r in ranks], fontsize=10, fontweight="bold")
    ax_flop.grid(True, linestyle="--", alpha=0.5)

    for x, y in zip(ranks, decode_kflops):
        ax_flop.annotate(f"{y:.1f}k", xy=(x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig51_rank_vs_memory_latency.png"] = buf.getvalue()

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print(" Launching EXP-19: Subspace Rank Scaling Benchmark on Modal GPU")
    print(" Target: NVIDIA A10G Cloud GPU (Timeout <= 180s)")
    print("==================================================================")

    # 1. Execute remote worker
    metrics = run_rank_scaling_worker.remote()

    # 2. Generate publication plots
    print("\nGenerating publication figures in Modal cloud worker...")
    plot_dict = generate_rank_plots_cloud.remote(metrics)

    # 3. Save artifacts locally
    out_dir = "results/exp19_rank_scaling"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Saved] Metrics JSON: {metrics_path}")

    for fname, data in plot_dict.items():
        p_path = os.path.join(out_dir, fname)
        with open(p_path, "wb") as f:
            f.write(data)
        print(f"[Saved] Publication Plot: {p_path}")

    print("\n==================================================================")
    print(" EXP-19 Results Summary:")
    print(" Rank |   Val R^2   |  Cos Sim  | State (KB) | Decode KFLOPs | Fit Time (s)")
    print("-----------------------------------------------------------------------")
    for r in metrics["results"]:
        print(f" {r['rank']:4d} |   {r['val_r2']:7.4f}   |  {r['val_cos_sim']:7.4f}  |  {r['state_kb']:7.1f}   |   {r['decode_kflops_per_token']:7.1f}     |   {r['fit_time_s']:6.2f}s")
    print("==================================================================")


if __name__ == "__main__":
    main()
