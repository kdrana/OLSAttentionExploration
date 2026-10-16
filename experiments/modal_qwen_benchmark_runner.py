"""
Modal Cloud Runner for EXP-16: Comprehensive Qwen3.5-0.8B vs Flash-OLS Benchmark.

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300.
Serializes and downloads:
- results/exp16_qwen_benchmark/metrics.json
- results/exp16_qwen_benchmark/qwen_ols_adapter.pt (~1.96 MB lightweight weights)
- Publication-quality comparison figures (fig40, fig41, fig42).
"""

import os
import io
import json
import modal

app = modal.App("ols-attention-qwen-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def benchmark_qwen_worker() -> tuple[dict, bytes]:
    import torch
    from experiments.run_qwen_benchmark import run_qwen_benchmark_suite

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")

    results, adapter_state = run_qwen_benchmark_suite(
        model_id="Qwen/Qwen3.5-0.8B-Base",
        device=device,
        distill_steps=350,
        batch_size=4,
        seq_len=256,
        lr=1e-3,
    )

    # Serialize adapter state to bytes
    buf = io.BytesIO()
    torch.save(adapter_state, buf)
    adapter_bytes = buf.getvalue()
    print(f"[Modal Cloud Worker] Serialized Flash-OLS adapter: {len(adapter_bytes) / (1024*1024):.2f} MB")

    return results, adapter_bytes


@app.function(image=image, timeout=60)
def generate_benchmark_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    extrap = metrics["extrapolation"]
    ctx_lengths = [e["context_length"] for e in extrap]
    softmax_mb = [e["softmax_kv_cache_mb"] for e in extrap]
    ols_kb = [e["flash_ols_state_kb"] for e in extrap]
    ols_mb = [k / 1024.0 for k in ols_kb]
    compression = [e["compression_ratio"] for e in extrap]

    # -------------------------------------------------------------
    # Figure 40: KV-Cache Memory Scaling (Log-Log) & Compression
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    ax1.plot(ctx_lengths, softmax_mb, "s-", color="#e74c3c", linewidth=2.4, label="Pre-Trained Softmax KV-Cache (Linear O(T))")
    ax1.plot(ctx_lengths, ols_mb, "o-", color="#27ae60", linewidth=2.4, label="Solvable Flash-OLS Gram State (Constant O(1))")
    ax1.axhline(3221.23, color="#c0392b", linestyle=":", label="Softmax at 262k Context (3.22 GB)")
    ax1.axhline(0.204, color="#1e8449", linestyle=":", label="Flash-OLS Flatline (0.204 MB)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_title(r"Attention Memory Footprint vs Context Length ($T$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Context Length (Tokens)", fontsize=11)
    ax1.set_ylabel("Attention State Memory (MB, Log Scale)", fontsize=11)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2.plot(ctx_lengths, compression, "d-", color="#8e44ad", linewidth=2.4, label="Memory Compression Ratio")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title(r"Flash-OLS Memory Compression vs Standard KV-Cache", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Context Length (Tokens)", fontsize=11)
    ax2.set_ylabel("Compression Factor (x)", fontsize=11)
    ax2.axhline(15790.3, color="#6c3483", linestyle="--", label="15,790x Compression at 262,144 Context")
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig40_qwen_memory_scaling.png"] = buf.getvalue()

    # -------------------------------------------------------------
    # Figure 41: Decode Throughput & Prefill Latency
    # -------------------------------------------------------------
    base_scaling = metrics["baseline_softmax"]["scaling"]
    ols_scaling = metrics["linearized_flash_ols"]["scaling"]

    T_vals = [e["T"] for e in base_scaling]
    base_decode = [e["decode_tok_s"] for e in base_scaling]
    ols_decode = [e["decode_tok_s"] for e in ols_scaling]
    base_prefill = [e["prefill_ms"] for e in base_scaling]
    ols_prefill = [e["prefill_ms"] for e in ols_scaling]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Decode throughput
    ax1.plot(T_vals, base_decode, "s-", color="#e74c3c", linewidth=2.2, label="Pre-Trained Softmax Attention")
    ax1.plot(T_vals, ols_decode, "o-", color="#27ae60", linewidth=2.2, label="Linearized Flash-OLS (Sherman-Morrison)")
    ax1.set_title(r"Autoregressive Decode Throughput vs Context Length", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Historical Context Length (Tokens)", fontsize=11)
    ax1.set_ylabel("Decode Speed (Tokens / Second)", fontsize=11)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Prefill latency
    ax2.plot(T_vals, base_prefill, "s-", color="#e74c3c", linewidth=2.2, label="Softmax Attention Prefill")
    ax2.plot(T_vals, ols_prefill, "o-", color="#2980b9", linewidth=2.2, label="Flash-OLS Fused Triton Prefill")
    ax2.set_title(r"Prefill Processing Latency vs Context Length", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Prompt Length (Tokens)", fontsize=11)
    ax2.set_ylabel("Prefill Latency (ms)", fontsize=11)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig41_qwen_decode_throughput.png"] = buf.getvalue()

    # -------------------------------------------------------------
    # Figure 42: Quality & Loss Comparison
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    models = ["Pre-Trained Softmax", "Flash-OLS (Step 350)"]
    val_losses = [metrics["baseline_softmax"]["val_loss"], metrics["linearized_flash_ols"]["val_loss"]]
    val_ppls = [metrics["baseline_softmax"]["val_ppl"], metrics["linearized_flash_ols"]["val_ppl"]]
    colors = ["#e74c3c", "#27ae60"]

    bars1 = ax1.bar(models, val_losses, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    ax1.set_title(r"Validation Cross-Entropy Loss (Nats)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss (Nats)", fontsize=11)
    ax1.set_ylim(0, max(val_losses) * 1.25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    bars2 = ax2.bar(models, val_ppls, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    ax2.set_title(r"Validation Perplexity", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Perplexity", fontsize=11)
    ax2.set_ylim(0, max(val_ppls) * 1.25)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig42_qwen_loss_comparison.png"] = buf.getvalue()

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print(" Launching EXP-16: Comprehensive Qwen3.5 Benchmark Suite on Modal")
    print(" Target: NVIDIA A10G Cloud GPU (Timeout <= 300s)")
    print("==================================================================")

    # 1. Execute Benchmark Worker
    results, adapter_bytes = benchmark_qwen_worker.remote()

    # 2. Generate Publication Plots
    print("\nGenerating publication figures in Modal cloud worker...")
    plot_bytes_dict = generate_benchmark_plots_cloud.remote(results)

    # 3. Save Artifacts Locally
    out_dir = "results/exp16_qwen_benchmark"
    os.makedirs(out_dir, exist_ok=True)

    # Metrics JSON
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Adapter checkpoint (.pt)
    adapter_path = os.path.join(out_dir, "qwen_ols_adapter.pt")
    with open(adapter_path, "wb") as f:
        f.write(adapter_bytes)
    print(f"Saved: {adapter_path} ({len(adapter_bytes) / (1024*1024):.2f} MB)")

    # Also save a copy in checkpoints/ for convenience
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/qwen_ols_adapter.pt", "wb") as f:
        f.write(adapter_bytes)
    print(f"Saved: checkpoints/qwen_ols_adapter.pt")

    # Plots
    for fname, data in plot_bytes_dict.items():
        p_path = os.path.join(out_dir, fname)
        with open(p_path, "wb") as f:
            f.write(data)
        print(f"Saved: {p_path}")

    print("\n[Modal Benchmark Run Complete] All artifacts synchronized to local repository.")


if __name__ == "__main__":
    main()
