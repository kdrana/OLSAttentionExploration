"""
Modal Cloud Runner for EXP-24: Full Native Solvable Attention Language Model (NSAL-LM) Pretraining.

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300s.
Saves:
- results/exp24_nsal_lm_pretraining/metrics.json
- results/exp24_nsal_lm_pretraining/fig64_nsal_lm_pretraining_loss.png
- results/exp24_nsal_lm_pretraining/fig65_nsal_lm_val_perplexity.png
- results/exp24_nsal_lm_pretraining/fig66_nsal_lm_cache_scaling.png
"""

import os
import io
import json
import modal

app = modal.App("ols-nsal-lm-pretraining")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "numpy", "matplotlib")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_exp24_worker() -> dict:
    import torch
    from experiments.run_nsal_lm_pretraining import run_exp24_pretraining_benchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")
    return run_exp24_pretraining_benchmark(device=device)


@app.function(image=image, timeout=60)
def generate_exp24_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    softmax_res = metrics["softmax"]
    nsal_res = metrics["nsal"]

    # -------------------------------------------------------------
    # Figure 64: Pretraining Loss Convergence
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    steps = [i * 10 for i in range(len(softmax_res["loss_history"]))]

    ax.plot(steps, softmax_res["loss_history"], linewidth=2.2, color="#2b5c8f", label=f"Softmax Transformer (Final: {softmax_res['final_loss']:.4f})")
    ax.plot(steps, nsal_res["loss_history"], linewidth=2.5, color="#1b9e77", label=f"Full NSAL-LM (Final: {nsal_res['final_loss']:.4f})")

    ax.set_xlabel("Pretraining Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss (TinyStories)", fontsize=11, fontweight="bold")
    ax.set_title("EXP-24: Full NSAL Language Model Pretraining Loss Convergence", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig64_nsal_lm_pretraining_loss.png"] = buf.getvalue()
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 65: Validation Perplexity
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    val_steps = [v["step"] for v in softmax_res["val_history"]]
    softmax_ppl = [v["ppl"] for v in softmax_res["val_history"]]
    nsal_ppl = [v["ppl"] for v in nsal_res["val_history"]]

    ax.plot(val_steps, softmax_ppl, marker="o", linewidth=2.2, color="#2b5c8f", label=f"Softmax PPL (Best: {min(softmax_ppl):.2f})")
    ax.plot(val_steps, nsal_ppl, marker="^", linewidth=2.5, color="#1b9e77", label=f"Full NSAL-LM PPL (Best: {min(nsal_ppl):.2f})")

    ax.set_xlabel("Optimization Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Perplexity", fontsize=11, fontweight="bold")
    ax.set_title("EXP-24: Narrative Language Modeling Validation Perplexity", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig65_nsal_lm_val_perplexity.png"] = buf.getvalue()
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 66: Inference Cache Memory Scaling (O(1) vs O(T))
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    t_lengths = [128, 512, 1024, 2048, 4096]
    softmax_mem = [softmax_res["memory_scaling_kb"][f"mem_T{t}"] for t in t_lengths]
    nsal_mem = [nsal_res["memory_scaling_kb"][f"mem_T{t}"] for t in t_lengths]

    x_idx = np.arange(len(t_lengths))
    width = 0.35

    bars1 = ax.bar(x_idx - width/2, softmax_mem, width, label="Softmax KV Cache (O(T) Linear)", color="#2b5c8f", edgecolor="#333333")
    bars2 = ax.bar(x_idx + width/2, nsal_mem, width, label="Full NSAL-LM State (O(1) Constant)", color="#1b9e77", edgecolor="#333333")

    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"T={t}" for t in t_lengths], fontsize=10, fontweight="bold")
    ax.set_xlabel("Context Tokens Generated (T)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attention Cache Memory (KB)", fontsize=11, fontweight="bold")
    ax.set_title("EXP-24: Generation Cache Footprint (6 Layers, D=256, H=8)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    # Annotate final bar
    ratio = softmax_mem[-1] / nsal_mem[-1]
    ax.annotate(
        f"{ratio:.1f}x Smaller\n(384 KB Flat)",
        xy=(x_idx[-1] + width/2, nsal_mem[-1]),
        xytext=(x_idx[-1], softmax_mem[-1] * 0.6),
        arrowprops=dict(facecolor="#1b9e77", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#1b9e77",
        fontsize=9,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig66_nsal_lm_cache_scaling.png"] = buf.getvalue()
    plt.close(fig)

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print(" Triggering Modal Cloud Run for EXP-24: Full NSAL-LM Pretraining")
    print(" Target: NVIDIA A10G Cloud GPU (timeout <= 300s)")
    print("==================================================================")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp24_nsal_lm_pretraining")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Run Worker on Cloud GPU
    metrics = run_exp24_worker.remote()
    print("\nCloud benchmark finished! Generating publication figures...")

    # 2. Generate Plots on Cloud
    plots = generate_exp24_plots_cloud.remote(metrics)

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

    # Summary Print
    print("\n==================================================================")
    print(" EXP-24 BENCHMARK SUMMARY (Full NSAL-LM vs Softmax Transformer)")
    print("==================================================================")
    print(f"  Softmax Final Val Loss / PPL: {metrics['softmax']['final_val_loss']:.4f} / {metrics['softmax']['final_val_ppl']:.2f}")
    print(f"  NSAL-LM Final Val Loss / PPL: {metrics['nsal']['final_val_loss']:.4f} / {metrics['nsal']['final_val_ppl']:.2f}")
    print(f"  NSAL Story Sample:            \"{metrics['nsal']['sample_generation'][:120]}...\"")
    print(f"  NSAL Generation Speed:        {metrics['nsal']['gen_tokens_per_sec']:.1f} tokens/sec")
    print(f"  Cache Memory at T=4096:       NSAL = {metrics['nsal']['memory_scaling_kb']['mem_T4096']} KB vs Softmax = {metrics['softmax']['memory_scaling_kb']['mem_T4096']} KB")
    print("==================================================================")


if __name__ == "__main__":
    main()
