"""
Modal Cloud Runner for Training Convergence Benchmark.
Runs 3 models across 3 datasets on an NVIDIA T4 GPU and streams results back.
"""

import json
import os
import modal

app = modal.App("ols-attention-convergence")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn", "pandas")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def run_cloud_convergence():
    import json
    import os
    import torch
    from experiments.run_convergence import run_convergence_suite
    from experiments.plot_convergence import plot_convergence_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active GPU: {device} ({gpu_name})")

    os.makedirs("results", exist_ok=True)

    # Run the full 3x3 convergence benchmark for 35 epochs
    all_results = run_convergence_suite(device=device, n_epochs=35)

    results_path = "results/convergence_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Generating convergence figures...")
    plot_convergence_results(results_path, out_dir="results")

    # Collect artifacts
    artifacts = {}
    target_files = [
        "convergence_results.json",
        "fig_convergence_epochs.png",
        "fig_convergence_wallclock.png",
        "fig_final_r2_comparison.png",
    ]
    for filename in target_files:
        path = os.path.join("results", filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                artifacts[filename] = f.read()

    print(f"[Modal Cloud Worker] Successfully packaged {len(artifacts)} artifacts.")
    return artifacts


@app.local_entrypoint()
def main():
    print("🚀 Launching Convergence Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_convergence.remote()

    dest_dir = "results/exp02_convergence_benchmark"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n✅ Remote execution completed! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        # Map convergence_results.json to metrics.json
        local_name = "metrics.json" if filename == "convergence_results.json" else filename
        dest = os.path.join(dest_dir, local_name)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 Convergence plots & metrics synchronized locally in {dest_dir}/!")
