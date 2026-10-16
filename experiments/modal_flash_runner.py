"""
Modal Cloud Runner for EXP-07 Flash-OLS Ultra-Long Context Scaling Benchmark.
Runs on cloud GPU (NVIDIA T4, 16GB) with timeout=300 and streams results and plots back.
"""

import os
import modal

app = modal.App("ols-attention-flash")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def run_cloud_flash_ols():
    import json
    import os
    import torch
    from experiments.run_flash_ols_benchmark import run_flash_ols_suite
    from experiments.plot_flash_ols_benchmark import plot_flash_ols_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name})")

    os.makedirs("results", exist_ok=True)

    # Run the full EXP-07 Flash-OLS benchmark suite
    all_results = run_flash_ols_suite(device=device)

    results_path = "results/flash_ols_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Rendering Flash-OLS scaling and fidelity plots...")
    plot_flash_ols_results(results_path, out_dir="results")

    # Collect artifacts
    artifacts = {}
    target_files = [
        "flash_ols_results.json",
        "fig_flash_ols_memory_scaling.png",
        "fig_flash_ols_latency_scaling.png",
        "fig_flash_ols_fidelity_verification.png",
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
    print("🚀 Launching Flash-OLS Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_flash_ols.remote()

    dest_dir = "results/exp07_flash_ols_scaling"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n✅ Remote execution completed! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        local_name = "metrics.json" if filename == "flash_ols_results.json" else filename
        dest = os.path.join(dest_dir, local_name)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 Flash-OLS benchmark plots & metrics synchronized locally in {dest_dir}/!")


if __name__ == "__main__":
    main()
