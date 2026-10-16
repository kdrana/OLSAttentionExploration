"""
Modal Cloud Runner for EXP-06 Unified Mathematical Solution vs Pieces Benchmark.
Runs on cloud GPU (NVIDIA T4, 16GB) with timeout=300 and streams results and plots back.
"""

import os
import modal

app = modal.App("ols-attention-unified")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def run_cloud_unified():
    import json
    import os
    import torch
    from experiments.run_unified_benchmark import run_unified_suite
    from experiments.plot_unified_benchmark import plot_unified_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name})")

    os.makedirs("results", exist_ok=True)

    # Run the full EXP-06 Unified vs Pieces benchmark suite
    all_results = run_unified_suite(device=device)

    results_path = "results/unified_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Rendering Unified Attention plots...")
    plot_unified_results(results_path, out_dir="results")

    # Collect artifacts
    artifacts = {}
    target_files = [
        "unified_results.json",
        "fig_mqar_unified_comparison.png",
        "fig_unified_training_wallclock.png",
        "fig_unified_inference_memory.png",
        "fig_unified_per_token_latency.png",
    ]
    for filename in target_files:
        path = os.path.join("results", filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                artifacts[filename] = f.read()

    print(f"[Modal Cloud Worker] Packaged {len(artifacts)} artifact(s). Returning to client.")
    return artifacts


@app.local_entrypoint()
def main():
    print("🚀 Launching Unified vs Pieces Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_unified.remote()

    dest_dir = "results/exp06_unified_vs_pieces"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n✅ Remote execution completed! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        local_name = "metrics.json" if filename == "unified_results.json" else filename
        dest = os.path.join(dest_dir, local_name)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 Unified benchmark plots & metrics synchronized locally in {dest_dir}/!")
