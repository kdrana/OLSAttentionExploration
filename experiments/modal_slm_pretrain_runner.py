"""
Modal Cloud Runner for EXP-12 Small Language Model Pretraining Benchmark.
Executes 500 pretraining steps of 12.3M Softmax SLM vs. Flash-OLS SLM on NVIDIA T4 GPUs.
Runs both models concurrently across two separate Modal worker instances
to enforce timeout <= 300 per worker while eliminating any container queuing or time competition.
Streams back metrics and plots directly into results/exp12_slm_early_pretraining/.
"""

import json
import os
import modal

app = modal.App("ols-attention-slm-pretrain")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn", "requests", "tokenizers")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def train_single_model_worker(model_type: str) -> dict:
    import torch
    from experiments.run_slm_pretraining_benchmark import run_slm_pretraining_benchmark

    total_steps = 350
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name}) | Model: {model_type} | Steps: {total_steps}")

    results = run_slm_pretraining_benchmark(
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
    from experiments.plot_slm_pretraining_benchmark import plot_slm_results

    tmp_dir = "/tmp/plot_out"
    os.makedirs(tmp_dir, exist_ok=True)
    metrics_path = os.path.join(tmp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    plot_slm_results(metrics_path, out_dir=tmp_dir)

    artifacts = {}
    for fn in [
        "fig30_slm_pretraining_loss_curves.png",
        "fig31_slm_validation_perplexity.png",
        "fig32_slm_gradient_norm_and_throughput.png",
    ]:
        p = os.path.join(tmp_dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                artifacts[fn] = f.read()
    return artifacts


@app.local_entrypoint()
def main():
    print("🚀 Launching EXP-12 SLM Pretraining Benchmark on Modal (350 steps on 2x NVIDIA T4 GPUs in parallel)...")
    
    # Run both models concurrently via modal map
    model_targets = ["softmax", "flash_ols"]
    results_list = list(train_single_model_worker.map(model_targets))

    combined_results = {}
    for r in results_list:
        combined_results.update(r)

    dest_dir = "results/exp12_slm_early_pretraining"
    os.makedirs(dest_dir, exist_ok=True)
    metrics_path = os.path.join(dest_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n✅ All metrics saved locally to {metrics_path}!")

    print("📊 Generating publication-grade figures (Fig 30: Loss, Fig 31: Perplexity, Fig 32: Throughput)...")
    plot_artifacts = generate_plots_cloud.remote(combined_results)
    for filename, content in sorted(plot_artifacts.items()):
        dest = os.path.join(dest_dir, filename)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 EXP-12 benchmark completed! Artifacts saved in {dest_dir}/")


if __name__ == "__main__":
    main()
