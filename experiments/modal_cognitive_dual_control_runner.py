"""
Modal Cloud Runner for EXP-20: Cognitive Dual-Control Memory (CDCM) Attention Benchmark.

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300s.
Saves:
- results/exp20_cognitive_dual_control/metrics.json
- results/exp20_cognitive_dual_control/fig52_cdca_benchmark_comparison.png
- results/exp20_cognitive_dual_control/fig53_cdca_niah_heatmap.png
- results/exp20_cognitive_dual_control/fig54_cdca_memory_scaling.png
"""

import os
import io
import json
import modal

app = modal.App("ols-attention-cdca-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_cdca_worker() -> dict:
    import torch
    from experiments.run_cognitive_dual_control import run_cdca_benchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")
    return run_cdca_benchmark(device=device)


@app.function(image=image, timeout=60)
def generate_cdca_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    benchmarks = metrics["academic_benchmarks"]

    # -------------------------------------------------------------
    # Figure 52: 3-Way Benchmark Comparison (Softmax vs Flash-OLS vs CDCA)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

    categories = ["MMLU / ARC Science", "GSM8K Math Reasoning", "HumanEval Syntax", "HumanEval Tests Pass"]
    softmax_scores = [
        benchmarks["mmlu_arc_pct"]["original_softmax"],
        benchmarks["gsm8k_math_pct"]["original_softmax"],
        benchmarks["humaneval_coding"]["syntax_valid_pct"]["original_softmax"],
        benchmarks["humaneval_coding"]["tests_passed_pct"]["original_softmax"],
    ]
    ols_scores = [
        benchmarks["mmlu_arc_pct"]["pure_flash_ols"],
        benchmarks["gsm8k_math_pct"]["pure_flash_ols"],
        benchmarks["humaneval_coding"]["syntax_valid_pct"]["pure_flash_ols"],
        benchmarks["humaneval_coding"]["tests_passed_pct"]["pure_flash_ols"],
    ]
    cdca_scores = [
        benchmarks["mmlu_arc_pct"]["cdca_hybrid"],
        benchmarks["gsm8k_math_pct"]["cdca_hybrid"],
        benchmarks["humaneval_coding"]["syntax_valid_pct"]["cdca_hybrid"],
        benchmarks["humaneval_coding"]["tests_passed_pct"]["cdca_hybrid"],
    ]

    x = np.arange(len(categories))
    width = 0.25

    rects1 = ax.bar(x - width, softmax_scores, width, label="Original Softmax (Empirical)", color="#e74c3c", edgecolor="black", linewidth=1.1)
    rects2 = ax.bar(x, ols_scores, width, label="Pure Flash-OLS Linearized", color="#f39c12", edgecolor="black", linewidth=1.1)
    rects3 = ax.bar(x + width, cdca_scores, width, label="Cognitive Dual-Control Memory (CDCA)", color="#27ae60", edgecolor="black", linewidth=1.1)

    ax.set_title("Qwen 0.8B Academic Benchmarks: Softmax vs Flash-OLS vs CDCA", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy / Score (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig52_cdca_benchmark_comparison.png"] = buf.getvalue()

    # -------------------------------------------------------------
    # Figure 53: CDCA Needle-In-A-Haystack (NIAH) Heatmap
    # -------------------------------------------------------------
    niah = metrics["niah_stress_test"]
    lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    depths = [0.10, 0.50, 0.90]

    grid = np.zeros((len(depths), len(lengths)))
    for item in niah["trials"]:
        d_idx = depths.index(item["depth"])
        l_idx = lengths.index(item["target_len"])
        grid[d_idx, l_idx] = 1.0 if item["success"] else 0.0

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    cax = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_title(f"Needle-In-A-Haystack Retrieval: Cognitive Dual-Control Attention ({niah['success_rate_pct']:.1f}% Success)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Context Length (Tokens)", fontsize=11)
    ax.set_ylabel("Needle Depth Position", fontsize=11)
    ax.set_xticks(np.arange(len(lengths)))
    ax.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in lengths], fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([f"{int(d*100)}%" for d in depths], fontsize=10, fontweight="bold")

    for i in range(len(depths)):
        for j in range(len(lengths)):
            val = grid[i, j]
            text = "PASS" if val == 1.0 else "FAIL"
            color = "white" if val in [0.0, 1.0] else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig53_cdca_niah_heatmap.png"] = buf.getvalue()

    # -------------------------------------------------------------
    # Figure 54: Memory Scaling Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)

    ctx_vals = lengths
    softmax_mb = [(12_288 * c) / (1024.0 * 1024.0) for c in ctx_vals]
    ols_mb = [0.204 for _ in ctx_vals]
    cdca_mb = [(12_288 * min(c, 1024) + 240 * 1024) / (1024.0 * 1024.0) for c in ctx_vals]

    ax.plot(ctx_vals, softmax_mb, "s-", color="#e74c3c", linewidth=2.4, label="Original Softmax KV-Cache (O(T) Explosion)")
    ax.plot(ctx_vals, cdca_mb, "d-", color="#27ae60", linewidth=2.4, label="Cognitive Dual-Control Memory (Clamped at ~12.2 MB)")
    ax.plot(ctx_vals, ols_mb, "o--", color="#f39c12", linewidth=2.0, label="Pure Flash-OLS State (Flat 0.204 MB)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Attention State Memory vs Context Length (Log-Log)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Context Length (Tokens)", fontsize=11)
    ax.set_ylabel("State Memory (MB)", fontsize=11)
    ax.set_xticks(lengths)
    ax.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in lengths])
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig54_cdca_memory_scaling.png"] = buf.getvalue()

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print(" Launching EXP-20: Cognitive Dual-Control Attention on Modal GPU")
    print(" Target: NVIDIA A10G Cloud GPU (Timeout <= 300s)")
    print("==================================================================")

    # 1. Execute Benchmark Worker
    metrics = run_cdca_worker.remote()

    # 2. Generate Publication Plots
    print("\nGenerating publication figures in Modal cloud worker...")
    plot_dict = generate_cdca_plots_cloud.remote(metrics)

    # 3. Save Artifacts Locally
    out_dir = "results/exp20_cognitive_dual_control"
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
    print(" EXP-20 Summary:")
    print(f" MMLU/ARC Score: {metrics['academic_benchmarks']['mmlu_arc_pct']['cdca_hybrid']:.1f}%")
    print(f" HumanEval Syntax: {metrics['academic_benchmarks']['humaneval_coding']['syntax_valid_pct']['cdca_hybrid']:.1f}%")
    print(f" HumanEval Tests Passed: {metrics['academic_benchmarks']['humaneval_coding']['tests_passed_pct']['cdca_hybrid']:.1f}%")
    print(f" NIAH Success Rate: {metrics['niah_stress_test']['total_success']}/{metrics['niah_stress_test']['total_trials']} ({metrics['niah_stress_test']['success_rate_pct']:.1f}%)")
    print("==================================================================")


if __name__ == "__main__":
    main()
