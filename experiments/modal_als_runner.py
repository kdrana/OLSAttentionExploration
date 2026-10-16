"""
Modal Cloud Runner for EXP-10: Alternating Least Squares (ALS) Solvable Refinement Benchmark.
Runs on cloud GPU (NVIDIA T4, 16GB) with timeout=180 and streams results and plots back.
"""

import os
import modal

app = modal.App("ols-attention-als-refinement")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn", "xgboost", "lightgbm", "pandas")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=180)
def run_cloud_als():
    import json
    import os
    import torch
    from experiments.run_als_benchmark import run_als_suite
    from experiments.plot_als_benchmark import plot_als_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name})")

    os.makedirs("results", exist_ok=True)

    # Run the full EXP-10 Alternating Least Squares benchmark suite
    all_results = run_als_suite(device=device)

    results_path = "results/als_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Rendering EXP-10 publication plots...")
    plot_als_results(results_path, out_dir="results")

    # Collect artifacts
    artifacts = {}
    target_files = [
        "als_results.json",
        "fig_als_r2_improvement.png",
        "fig_als_convergence_trajectory.png",
        "fig_als_wallclock_comparison.png",
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
    print("🚀 Launching EXP-10 Alternating Least Squares Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_als.remote()

    dest_dir = "results/exp10_als_solvable_refinement"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n✅ Remote execution completed! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        local_name = "metrics.json" if filename == "als_results.json" else filename
        dest = os.path.join(dest_dir, local_name)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 EXP-10 ALS Refinement plots & metrics synchronized locally in {dest_dir}/!")


if __name__ == "__main__":
    main()
