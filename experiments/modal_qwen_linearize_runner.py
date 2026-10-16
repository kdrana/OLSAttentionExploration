"""
Modal Cloud Runner for EXP-15: Qwen3.5-0.8B Post-Training Linearization (Stage 1 Sanity Run).
Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300.
"""

import json
import os
import modal

app = modal.App("ols-attention-qwen-linearize")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def train_qwen_linearization_worker() -> dict:
    import torch
    from experiments.run_qwen_linearization import run_qwen_linearization_benchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")
    
    results = run_qwen_linearization_benchmark(
        model_id="Qwen/Qwen3.5-0.8B-Base",
        device=device,
        total_steps=350,
        batch_size=4,
        seq_len=256,
        lr=1e-3,
        checkpoints=[0, 25, 50, 100, 150, 200, 250, 300, 350],
    )
    return results


@app.function(image=image, timeout=60)
def generate_plots_cloud(metrics: dict) -> dict:
    import os
    import matplotlib.pyplot as plt

    tmp_dir = "/tmp/qwen_plot_out"
    os.makedirs(tmp_dir, exist_ok=True)

    trace = metrics["linearized_ols"]["trace"]
    steps = [e["step"] for e in trace]
    val_loss = [e["val_loss"] for e in trace]
    val_ppl = [e["val_ppl"] for e in trace]
    base_val_loss = metrics["baseline_softmax"]["val_loss"]
    base_val_ppl = metrics["baseline_softmax"]["val_ppl"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Loss plot
    ax1.plot(steps, val_loss, "s-", color="#27ae60", linewidth=2.2, label="Linearized Qwen3.5 (Flash-OLS)")
    ax1.axhline(base_val_loss, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Pre-Trained Softmax Baseline ({base_val_loss:.2f})")
    ax1.set_title(r"Validation Cross-Entropy Loss vs Distillation Steps")
    ax1.set_xlabel("Distillation Steps (NVIDIA A10G)")
    ax1.set_ylabel("Loss (Nats)")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Perplexity plot
    ax2.plot(steps, val_ppl, "o-", color="#2980b9", linewidth=2.2, label="Linearized Qwen3.5 (Flash-OLS)")
    ax2.axhline(base_val_ppl, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Pre-Trained Softmax Baseline ({base_val_ppl:.1f})")
    ax2.set_title(r"Validation Perplexity vs Distillation Steps")
    ax2.set_xlabel("Distillation Steps")
    ax2.set_ylabel("Perplexity (Log Scale)")
    ax2.set_yscale("log")
    ax2.legend(frameon=True)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle("Qwen3.5-0.8B Stage 1 Post-Training Linearization Benchmark", fontsize=14, y=1.02)
    plt.tight_layout()

    out_p = os.path.join(tmp_dir, "fig39_qwen_linearization_loss.png")
    plt.savefig(out_p)
    plt.close()

    artifacts = {}
    with open(out_p, "rb") as f:
        artifacts["fig39_qwen_linearization_loss.png"] = f.read()
    return artifacts


@app.local_entrypoint()
def main():
    print("🚀 Launching EXP-15 Qwen3.5-0.8B Linearization Sanity Benchmark on Modal (NVIDIA A10G Cloud GPU)...")
    
    results = train_qwen_linearization_worker.remote()

    out_dir = "results/exp15_qwen_ols_linearization"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved metrics to {metrics_path}")

    print("\n🎨 Generating Figure 39 on cloud...")
    artifacts = generate_plots_cloud.remote(results)
    for fn, img_bytes in artifacts.items():
        local_p = os.path.join(out_dir, fn)
        with open(local_p, "wb") as f:
            f.write(img_bytes)
        print(f"  Saved figure: {local_p}")

    print("\n" + "=" * 80)
    print("🏆 EXP-15: QWEN3.5-0.8B LINEARIZATION STAGE 1 SANITY RESULTS")
    print("=" * 80)
    base = results["baseline_softmax"]
    lin = results["linearized_ols"]
    print(f"Total Parameters:           {results['total_parameters']:,}")
    print(f"Trainable Parameters:       {results['trainable_parameters']:,} ({(results['trainable_parameters']/results['total_parameters'])*100:.3f}%)")
    print(f"Full Attention Layers:      {results['full_attention_layers']}")
    print(f"Baseline Softmax Val Loss:  {base['val_loss']:.4f} (PPL: {base['val_ppl']:.2f})")
    print(f"Linearized OLS Final Loss:  {lin['final_val_loss']:.4f} (PPL: {lin['final_val_ppl']:.2f})")
    print(f"Mean Step Latency:          {lin['mean_step_time_ms']:.1f} ms")
    print(f"Total Cloud Time:           {lin['total_time_s']:.1f} s")
    print(f"Flash-OLS KV State Memory:  {lin['state_memory_kb']:.1f} KB (CONSTANT)")
    print("=" * 80)
