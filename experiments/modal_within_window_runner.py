"""
Modal Cloud Runner for Within-Window Memory Benchmark.
=====================================================
Runs the evaluation across window sizes W in [16, 32, 64, 128, 256, 512] on NVIDIA A10G GPU.
Enforces timeout <= 300s.
"""

import modal

app = modal.App("ols-within-window-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_benchmark_worker() -> dict:
    from experiments.test_within_window_performance import evaluate_within_window
    return evaluate_within_window(device="cuda")


@app.local_entrypoint()
def main():
    print("[Local Entrypoint] Running Within-Window Benchmark on Modal Cloud GPU...")
    res = run_benchmark_worker.remote()
    print("\nCompleted! Results:")
    import json
    print(json.dumps(res, indent=2))
