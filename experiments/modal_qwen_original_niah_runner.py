"""
Modal Cloud Runner for NIAH Retrieval on Original Pre-Trained Qwen3.5-0.8B (Softmax).

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300.
Evaluates context lengths T in [1024, 2048, 4096, 8192, 16384, 32768] at depths 10%, 50%, 90%.
Saves:
- results/exp18_qwen_full_validation/original_softmax_niah.json
- Updates results/exp18_qwen_full_validation/metrics.json
- Generates side-by-side comparative NIAH figures.
"""

import os
import io
import json
import modal

app = modal.App("ols-attention-qwen-original-niah")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_original_niah_worker() -> dict:
    import torch
    from experiments.run_qwen_original_niah import run_original_qwen_niah_benchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")

    results = run_original_qwen_niah_benchmark(device=device)
    return results


@app.function(image=image, timeout=60)
def generate_comparative_niah_plots_cloud(softmax_trials: list[dict], flash_ols_trials: list[dict], lengths: list[int], depths: list[float]) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    # -------------------------------------------------------------
    # Figure 48: Side-by-side Needle-In-A-Haystack Heatmaps
    # -------------------------------------------------------------
    grid_softmax = np.zeros((len(depths), len(lengths)))
    for item in softmax_trials:
        d_idx = depths.index(item["depth"])
        l_idx = lengths.index(item["target_len"])
        grid_softmax[d_idx, l_idx] = 1.0 if item["success"] else 0.0

    grid_ols = np.zeros((len(depths), len(lengths)))
    for item in flash_ols_trials:
        d_idx = depths.index(item["depth"])
        l_idx = lengths.index(item["target_len"])
        grid_ols[d_idx, l_idx] = 1.0 if item["success"] else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Plot Softmax
    im1 = ax1.imshow(grid_softmax, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax1.set_title("Original Pre-Trained Qwen3.5-0.8B (Softmax Attention)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Context Length (Tokens)", fontsize=10)
    ax1.set_ylabel("Needle Depth Position", fontsize=10)
    ax1.set_xticks(np.arange(len(lengths)))
    ax1.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in lengths], fontsize=9.5, fontweight="bold")
    ax1.set_yticks(np.arange(len(depths)))
    ax1.set_yticklabels([f"{int(d*100)}%" for d in depths], fontsize=9.5, fontweight="bold")

    for i in range(len(depths)):
        for j in range(len(lengths)):
            val = grid_softmax[i, j]
            text = "PASS" if val == 1.0 else "FAIL"
            color = "white" if val in [0.0, 1.0] else "black"
            ax1.text(j, i, text, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    # Plot Flash-OLS
    im2 = ax2.imshow(grid_ols, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_title("Linearized Flash-OLS (Constant O(1) Memory State)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Context Length (Tokens)", fontsize=10)
    ax2.set_xticks(np.arange(len(lengths)))
    ax2.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in lengths], fontsize=9.5, fontweight="bold")
    ax2.set_yticks(np.arange(len(depths)))
    ax2.set_yticklabels([f"{int(d*100)}%" for d in depths], fontsize=9.5, fontweight="bold")

    for i in range(len(depths)):
        for j in range(len(lengths)):
            val = grid_ols[i, j]
            text = "PASS" if val == 1.0 else "FAIL"
            color = "white" if val in [0.0, 1.0] else "black"
            ax2.text(j, i, text, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    fig.suptitle("Needle-In-A-Haystack (NIAH) Retrieval Comparison: Softmax vs Flash-OLS", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig48_niah_softmax_vs_ols_heatmap.png"] = buf.getvalue()

    # -------------------------------------------------------------
    # Figure 49: Generation Latency & VRAM Scaling Comparison
    # -------------------------------------------------------------
    fig, (ax_lat, ax_vram) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

    # Depth 50% comparisons
    sm_d50 = [t for t in softmax_trials if t["depth"] == 0.50]
    ols_d50 = [t for t in flash_ols_trials if t["depth"] == 0.50]

    ctx_sm = [t["target_len"] for t in sm_d50]
    lat_sm = [t["gen_time_s"] for t in sm_d50]
    vram_sm = [t["vram_mb"] for t in sm_d50]

    ctx_ols = [t["target_len"] for t in ols_d50]
    lat_ols = [t["gen_time_s"] for t in ols_d50]
    vram_ols = [t["vram_mb"] for t in ols_d50]

    # Latency curve
    ax_lat.plot(ctx_sm, lat_sm, "s-", color="#e74c3c", linewidth=2.2, label="Original Softmax Attention")
    ax_lat.plot(ctx_ols, lat_ols, "o-", color="#27ae60", linewidth=2.2, label="Linearized Flash-OLS")
    ax_lat.set_xscale("log", base=2)
    ax_lat.set_title("16-Token Generation Latency vs Context Length", fontsize=11, fontweight="bold")
    ax_lat.set_xlabel("Context Length (Tokens)", fontsize=10)
    ax_lat.set_ylabel("Latency (Seconds)", fontsize=10)
    ax_lat.set_xticks(lengths)
    ax_lat.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in lengths])
    ax_lat.legend(frameon=True, fontsize=9.5)
    ax_lat.grid(True, linestyle="--", alpha=0.5)

    # VRAM curve
    ax_vram.plot(ctx_sm, vram_sm, "s-", color="#e74c3c", linewidth=2.2, label="Original Softmax Attention")
    ax_vram.plot(ctx_ols, vram_ols, "o-", color="#27ae60", linewidth=2.2, label="Linearized Flash-OLS")
    ax_vram.set_xscale("log", base=2)
    ax_vram.set_title("Peak GPU VRAM Allocation vs Context Length", fontsize=11, fontweight="bold")
    ax_vram.set_xlabel("Context Length (Tokens)", fontsize=10)
    ax_vram.set_ylabel("Peak VRAM (MB)", fontsize=10)
    ax_vram.set_xticks(lengths)
    ax_vram.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in lengths])
    ax_vram.legend(frameon=True, fontsize=9.5)
    ax_vram.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig49_niah_latency_vram_comparison.png"] = buf.getvalue()

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print(" Launching NIAH Benchmark for Original Qwen3.5-0.8B on Modal GPU")
    print(" Target: NVIDIA A10G Cloud GPU (Timeout <= 300s)")
    print("==================================================================")

    # 1. Run cloud worker
    results = run_original_niah_worker.remote()

    # 2. Save individual JSON
    out_dir = "results/exp18_qwen_full_validation"
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "original_softmax_niah.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Original Softmax NIAH results: {raw_path}")

    # 3. Update master metrics.json
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        metrics["niah_stress_test"]["softmax"] = results["trials"]
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Updated] Master metrics.json with original softmax NIAH trials: {metrics_path}")

        # 4. Generate comparative publication plots
        print("\nGenerating comparative publication plots in Modal cloud worker...")
        lengths = metrics["niah_stress_test"]["lengths"]
        depths = metrics["niah_stress_test"]["depths"]
        flash_ols_trials = metrics["niah_stress_test"]["flash_ols"]
        softmax_trials = results["trials"]

        plot_dict = generate_comparative_niah_plots_cloud.remote(softmax_trials, flash_ols_trials, lengths, depths)
        for fname, data in plot_dict.items():
            p_path = os.path.join(out_dir, fname)
            with open(p_path, "wb") as f:
                f.write(data)
            print(f"[Saved] {p_path}")

    print("\n==================================================================")
    print(f" Summary of Original Qwen3.5-0.8B NIAH Retrieval:")
    print(f" Total Trials: {results['total_trials']}")
    print(f" Total Successes: {results['total_success']} ({results['success_rate_pct']:.1f}%)")
    print("==================================================================")


if __name__ == "__main__":
    main()
