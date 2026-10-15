"""
Modal Cloud Runner for Kalman & RLS Attention Benchmark (Phase 3 / EXP-04).
Runs on cloud GPU (NVIDIA T4, 16GB) with timeout=300 and streams results back.
"""

import os
import modal

app = modal.App("ols-attention-kalman")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def run_cloud_kalman():
    import json
    import os
    import torch
    from experiments.run_kalman_benchmark import run_kalman_suite
    from experiments.plot_kalman_benchmark import plot_kalman_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name})")

    os.makedirs("results", exist_ok=True)

    # Run the full Phase 3 Kalman/RLS benchmark suite
    all_results = run_kalman_suite(device=device)

    results_path = "results/kalman_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Rendering Kalman & RLS plots...")
    plot_kalman_results(results_path, out_dir="results")

    # Collect artifacts
    artifacts = {}
    target_files = [
        "kalman_results.json",
        "fig_mqar_kalman_accuracy.png",
        "fig_inference_memory_o1.png",
        "fig_generation_per_token_latency.png",
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
    print("🚀 Launching Kalman & RLS Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_kalman.remote()

    dest_dir = "results/exp04_kalman_rls_attention"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n✅ Remote execution completed! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        local_name = "metrics.json" if filename == "kalman_results.json" else filename
        dest = os.path.join(dest_dir, local_name)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 Kalman plots & metrics synchronized locally in {dest_dir}/!")
