"""
Modal Cloud Runner for OLS vs. Softmax Attention experiments.
Runs on cloud GPU (NVIDIA T4) and streams plots/results back to local machine.
"""

import os
import modal

app = modal.App("ols-attention-exploration")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy", "matplotlib")
    .add_local_python_source("kernels", "metrics", "tasks", "run_experiments", "plot_results")
)


@app.function(image=image, gpu="T4", timeout=600)
def run_cloud_experiments():
    import os
    import json
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({gpu_name})")

    # Import project modules (automatically mounted by Modal from current working directory)
    from run_experiments import (
        experiment_basis_approximation_vs_beta,
        experiment_bessel_harmonic_spectrum,
        experiment_downstream_r2,
        experiment_entropy_and_rank,
    )
    from plot_results import plot_all

    os.makedirs("results", exist_ok=True)

    print("[Modal Cloud Worker] Running Experiment 1: Basis Error vs Beta...")
    exp1_data = experiment_basis_approximation_vs_beta(device)

    print("[Modal Cloud Worker] Running Experiment 2: Bessel Spectrum...")
    exp2_data = experiment_bessel_harmonic_spectrum(d=16, max_order=4)

    print("[Modal Cloud Worker] Running Experiment 3: In-Context Regression R^2...")
    exp3_data = experiment_downstream_r2(device)

    print("[Modal Cloud Worker] Running Experiment 4: Entropy & Effective Rank...")
    exp4_data = experiment_entropy_and_rank(device)

    all_results = {
        "device": str(device),
        "gpu_name": gpu_name,
        "exp1_approx_error": exp1_data,
        "exp2_harmonic_spectrum": exp2_data,
        "exp3_downstream_r2": exp3_data,
        "exp4_entropy_rank": exp4_data,
    }

    with open("results/experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("[Modal Cloud Worker] Rendering publication plots...")
    plot_all()

    # Package output files to stream back to local client
    output_files = {}
    for filename in os.listdir("results"):
        filepath = os.path.join("results", filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                output_files[filename] = f.read()

    print("[Modal Cloud Worker] Run completed. Returning artifacts to client.")
    return output_files


@app.local_entrypoint()
def main():
    print("🚀 Launching OLS-Attention testbed on Modal (NVIDIA T4 Cloud GPU)...")
    results = run_cloud_experiments.remote()

    os.makedirs("results", exist_ok=True)
    print(f"\n✅ Modal execution finished! Received {len(results)} artifact(s):")
    for filename, content in sorted(results.items()):
        dest = os.path.join("results", filename)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"   • Saved: {dest} ({len(content):,} bytes)")

    print("\n🎉 All experimental figures and JSON metrics updated in results/!")
