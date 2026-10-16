"""
Modal Cloud Runner for EXP-14: Fused Triton Flash-OLS Kernel Benchmark.
Trains Fused-Flash-OLS SLM (Triton V4) on NVIDIA T4 GPU for 350 steps.
Aggregates metrics with Softmax (EXP-12), Baseline V2 (EXP-12), and Fast V3 (EXP-13).
Enforces timeout <= 300 to protect free-tier credits.
"""

import json
import os
import modal

app = modal.App("ols-attention-fused-triton-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy", "scipy", "matplotlib", "scikit-learn", "requests", "tokenizers")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def train_fused_triton_worker() -> dict:
    import torch
    from experiments.run_fast_ols_benchmark import run_fast_ols_benchmark

    total_steps = 350
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name}) | Model: Fused-Flash-OLS SLM (Triton V4) | Steps: {total_steps}")

    results = run_fast_ols_benchmark(
        device=device,
        total_steps=total_steps,
        batch_size=8,
        seq_len=256,
        lr=1e-3,
        checkpoints=[0, 25, 50, 100, 150, 200, 250, 300, 350],
        qual_checkpoints=[50, 100, 200, 350],
        target_models=["fused_triton_ols"],
    )
    return results


@app.function(image=image, timeout=60)
def generate_plots_cloud(metrics_dict: dict) -> dict:
    import json
    import os
    from experiments.plot_fused_triton_benchmark import plot_fused_triton_results

    tmp_dir = "/tmp/plot_fused_triton_out"
    os.makedirs(tmp_dir, exist_ok=True)
    metrics_path = os.path.join(tmp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    plot_fused_triton_results(metrics_path, out_dir=tmp_dir)

    artifacts = {}
    for fn in [
        "fig36_fused_triton_loss_curves.png",
        "fig37_fused_triton_perplexity_comparison.png",
        "fig38_fused_triton_speedup_pareto.png",
    ]:
        p = os.path.join(tmp_dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                artifacts[fn] = f.read()
    return artifacts


@app.local_entrypoint()
def main():
    print("🚀 Launching EXP-14 Fused Triton Flash-OLS SLM Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    
    # Train Fused-Flash-OLS V4 on GPU
    triton_results = train_fused_triton_worker.remote()

    # Load baseline metrics from EXP-12 and EXP-13
    exp12_path = "results/exp12_slm_early_pretraining/metrics.json"
    exp13_path = "results/exp13_fast_ols_acceleration/metrics.json"

    combined_results = {}
    if os.path.exists(exp12_path):
        with open(exp12_path, "r") as f:
            exp12 = json.load(f)
            if "Softmax SLM (Llama-Nano)" in exp12:
                combined_results["Softmax SLM (Llama-Nano)"] = exp12["Softmax SLM (Llama-Nano)"]
            if "Flash-OLS SLM (FlashOLS-Nano)" in exp12:
                combined_results["Flash-OLS SLM (Baseline V2)"] = exp12["Flash-OLS SLM (FlashOLS-Nano)"]

    if os.path.exists(exp13_path):
        with open(exp13_path, "r") as f:
            exp13 = json.load(f)
            if "Fast-Flash-OLS SLM (Fast V3)" in exp13:
                combined_results["Fast-Flash-OLS SLM (Fast V3)"] = exp13["Fast-Flash-OLS SLM (Fast V3)"]

    for k, v in triton_results.items():
        combined_results[k] = v

    out_dir = "results/exp14_fused_triton_ols"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"✅ Saved aggregated metrics to {metrics_path}")

    print("\n🎨 Generating publication-quality Figures 36, 37, 38...")
    artifacts = generate_plots_cloud.remote(combined_results)
    for fn, img_bytes in artifacts.items():
        local_p = os.path.join(out_dir, fn)
        with open(local_p, "wb") as f:
            f.write(img_bytes)
        print(f"  Saved figure: {local_p}")

    print("\n" + "=" * 80)
    print("🏆 EXP-14: 4-WAY SLM BENCHMARK COMPARISON (350 STEPS, NVIDIA T4 GPU)")
    print("=" * 80)
    print(f"{'Model Architecture':<35} | {'Step Time':<12} | {'Throughput':<14} | {'Val Loss':<10} | {'Val PPL':<10}")
    print("-" * 80)
    for name, res in combined_results.items():
        st = f"{res['mean_step_time_ms']:.1f} ms"
        tp = f"{res['tokens_per_sec']:,.0f} tok/s"
        last = res['trace'][-1]
        vl = f"{last['val_loss']:.4f}"
        vppl = f"{last['val_ppl']:.2f}"
        print(f"{name:<35} | {st:<12} | {tp:<14} | {vl:<10} | {vppl:<10}")
    print("=" * 80)
