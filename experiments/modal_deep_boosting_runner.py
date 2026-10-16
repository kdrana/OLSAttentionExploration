"""
Modal Cloud Runner for EXP-09: Deep Closed-Form Boosting & Solvable Networks Benchmark.
Runs on cloud GPU (NVIDIA T4, 16GB) with timeout=300 and streams results and plots back.
"""

import os
import modal

app = modal.App("ols-attention-deep-boosting")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn", "xgboost", "lightgbm", "pandas")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def run_cloud_deep_boosting():
    import json
    import os
    import torch
    from experiments.run_deep_boosting_benchmark import run_deep_boosting_suite
    from experiments.plot_deep_boosting_benchmark import plot_deep_boosting_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name})")

    os.makedirs("results", exist_ok=True)

    # Run the full EXP-09 Deep Closed-Form Boosting benchmark suite
    all_results = run_deep_boosting_suite(device=device)

    results_path = "results/deep_boosting_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Rendering EXP-09 publication plots...")
    plot_deep_boosting_results(results_path, out_dir="results")

    # Collect artifacts
    artifacts = {}
    target_files = [
        "deep_boosting_results.json",
        "fig_dsn_r2_comparison.png",
        "fig_dsn_depth_progression.png",
        "fig_dsn_training_wallclock.png",
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
    print("🚀 Launching EXP-09 Deep Closed-Form Boosting Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_deep_boosting.remote()

    dest_dir = "results/exp09_deep_boosting_solvable"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n✅ Remote execution completed! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        local_name = "metrics.json" if filename == "deep_boosting_results.json" else filename
        dest = os.path.join(dest_dir, local_name)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 EXP-09 Deep Boosting plots & metrics synchronized locally in {dest_dir}/!")


if __name__ == "__main__":
    main()
