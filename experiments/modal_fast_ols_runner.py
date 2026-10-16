"""
Modal Cloud Runner for EXP-13: Fast Analytical Autograd & Block-Diagonal Subspace Acceleration Benchmark.
Trains Fast-Flash-OLS SLM (Fast V3) on NVIDIA T4 GPU for 350 steps.
Aggregates metrics with Baseline Flash-OLS (EXP-12) and Softmax SLM to produce a 3-way comparative benchmark.
Enforces timeout <= 300 to protect free-tier credits.
"""

import json
import os
import modal

app = modal.App("ols-attention-fast-slm-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn", "requests", "tokenizers")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def train_fast_model_worker(model_type: str) -> dict:
    import torch
    from experiments.run_fast_ols_benchmark import run_fast_ols_benchmark

    total_steps = 350
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name}) | Model: {model_type} | Steps: {total_steps}")

    results = run_fast_ols_benchmark(
        device=device,
        total_steps=total_steps,
        batch_size=8,
        seq_len=256,
        lr=1e-3,
        checkpoints=[0, 25, 50, 100, 150, 200, 250, 300, 350],
        qual_checkpoints=[50, 100, 200, 350],
        target_models=[model_type],
    )
    return results


@app.function(image=image, timeout=60)
def generate_plots_cloud(metrics_dict: dict) -> dict:
    import json
    import os
    from experiments.plot_fast_ols_benchmark import plot_fast_ols_results

    tmp_dir = "/tmp/plot_fast_out"
    os.makedirs(tmp_dir, exist_ok=True)
    metrics_path = os.path.join(tmp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    plot_fast_ols_results(metrics_path, out_dir=tmp_dir)

    artifacts = {}
    for fn in [
        "fig33_fast_ols_loss_curves.png",
        "fig34_fast_ols_perplexity_comparison.png",
        "fig35_fast_ols_training_speedup_pareto.png",
    ]:
        p = os.path.join(tmp_dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                artifacts[fn] = f.read()
    return artifacts


@app.local_entrypoint()
def main():
    print("🚀 Launching EXP-13 Fast-Flash-OLS SLM Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    
    # Train Fast-Flash-OLS V3 on GPU
    fast_results = train_fast_model_worker.remote("fast_flash_ols")

    # Load EXP-12 baseline metrics for Softmax and Baseline Flash-OLS V2
    exp12_path = "results/exp12_slm_early_pretraining/metrics.json"
    combined_results = {}
    if os.path.exists(exp12_path):
        with open(exp12_path, "r") as f:
            exp12_data = json.load(f)
        if "Softmax SLM (Llama-Nano)" in exp12_data:
            combined_results["Softmax SLM (Llama-Nano)"] = exp12_data["Softmax SLM (Llama-Nano)"]
        if "Flash-OLS SLM (FlashOLS-Nano)" in exp12_data:
            combined_results["Flash-OLS SLM (Baseline V2)"] = exp12_data["Flash-OLS SLM (FlashOLS-Nano)"]

    combined_results.update(fast_results)

    dest_dir = "results/exp13_fast_ols_acceleration"
    os.makedirs(dest_dir, exist_ok=True)
    metrics_path = os.path.join(dest_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n✅ All 3 model metrics combined and saved locally to {metrics_path}!")

    print("📊 Generating publication-grade acceleration figures (Fig 33: Loss, Fig 34: Perplexity, Fig 35: Speedup Pareto)...")
    plot_artifacts = generate_plots_cloud.remote(combined_results)
    for filename, content in sorted(plot_artifacts.items()):
        dest = os.path.join(dest_dir, filename)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 EXP-13 benchmark completed! Artifacts saved in {dest_dir}/")


if __name__ == "__main__":
    main()
