"""
Modal test runner for Fused Triton Flash-OLS Kernel.
Verifies Triton JIT compilation and numeric equivalence on NVIDIA T4/A10G GPU.
Timeout <= 120s.
"""

import modal

app = modal.App("ols-attention-triton-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy")
    .add_local_python_source("src", "experiments")
)

@app.function(image=image, gpu="T4", timeout=120)
def run_test() -> dict:
    import torch
    from experiments.test_triton_ols import test_fused_triton_correctness

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Triton test on Modal: {device} ({torch.cuda.get_device_name(0)})")
    return test_fused_triton_correctness(device=device)

@app.local_entrypoint()
def main():
    print("Launching Triton test on Modal Cloud GPU...")
    res = run_test.remote()
    print("Result:", res)
