"""
Modal Cloud Runner for Sequence Modeling Benchmark (Phase 2 / EXP-03).
Runs on cloud GPU (NVIDIA T4, 16GB) with timeout=300 and streams results back.
"""

import os
import modal

app = modal.App("ols-attention-sequence")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib", "scikit-learn")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="T4", timeout=300)
def run_cloud_sequence():
    import json
    import os
    import torch
    from experiments.run_sequence_benchmark import run_sequence_suite
    from experiments.plot_sequence_benchmark import plot_sequence_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name})")

    os.makedirs("results", exist_ok=True)

    # Run the full sequence suite (MQAR accuracy + scaling across T up to 4096)
    all_results = run_sequence_suite(device=device)

    results_path = "results/sequence_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Rendering sequence benchmark plots...")
    plot_sequence_results(results_path, out_dir="results")

    # Collect artifacts
    artifacts = {}
    target_files = [
        "sequence_results.json",
        "fig_mqar_accuracy.png",
        "fig_scaling_latency.png",
        "fig_scaling_memory.png",
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
    print("🚀 Launching Sequence Benchmark on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_sequence.remote()

    dest_dir = "results/exp03_sequence_bulk_margin"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n✅ Remote execution completed! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        local_name = "metrics.json" if filename == "sequence_results.json" else filename
        dest = os.path.join(dest_dir, local_name)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print(f"\n🎉 Sequence plots & metrics synchronized locally in {dest_dir}/!")
