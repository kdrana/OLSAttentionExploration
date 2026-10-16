"""
Unit test and GPU correctness verification for Fused Triton Flash-OLS Kernel.
Tests:
1. Triton forward kernel vs PyTorch reference.
2. Triton analytical backward kernel vs PyTorch autograd.
3. Speed benchmark on GPU.
"""

import math
import time
import torch
import torch.nn as nn
from src.fused_ols_triton import FusedCausalOLSAccumFunction, CausalOLSSolveFunction

def test_fused_triton_correctness(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Triton Test] Testing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    torch.manual_seed(42)
    B, H, T, r, dv = 4, 8, 256, 16, 16
    gamma = 0.99
    lambda_reg = 0.05

    phi_k = torch.randn(B, H, T, r, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, T, dv, device=device, dtype=torch.float32, requires_grad=True)

    # 1. Forward Pass
    G_triton, R_triton = FusedCausalOLSAccumFunction.apply(phi_k, v, gamma, lambda_reg)

    # Reference forward pass in PyTorch
    G_ref = torch.zeros(B, H, T, r, r, device=device, dtype=torch.float32)
    R_ref = torch.zeros(B, H, T, r, dv, device=device, dtype=torch.float32)
    curr_G = torch.zeros(B, H, r, r, device=device, dtype=torch.float32)
    curr_R = torch.zeros(B, H, r, dv, device=device, dtype=torch.float32)
    eye_r = torch.eye(r, device=device, dtype=torch.float32).view(1, 1, r, r)

    for t in range(T):
        curr_G = gamma * curr_G + torch.einsum("bhi,bhj->bhij", phi_k[:, :, t], phi_k[:, :, t])
        curr_R = gamma * curr_R + torch.einsum("bhi,bhj->bhij", phi_k[:, :, t], v[:, :, t])
        G_ref[:, :, t] = curr_G + lambda_reg * eye_r
        R_ref[:, :, t] = curr_R

    diff_G = (G_triton - G_ref).abs().max().item()
    diff_R = (R_triton - R_ref).abs().max().item()
    print(f"Forward diff G: {diff_G:.2e}, diff R: {diff_R:.2e}")
    assert diff_G < 1e-4, f"Forward diff G too high: {diff_G}"
    assert diff_R < 1e-4, f"Forward diff R too high: {diff_R}"

    # 2. Backward Pass
    grad_G = torch.randn_like(G_triton)
    grad_G = 0.5 * (grad_G + grad_G.transpose(-1, -2))
    grad_R = torch.randn_like(R_triton)

    # Autograd with Triton function
    loss_triton = (G_triton * grad_G).sum() + (R_triton * grad_R).sum()
    loss_triton.backward()

    triton_grad_phi = phi_k.grad.clone()
    triton_grad_v = v.grad.clone()

    # Reference PyTorch autograd
    phi_k_ref = phi_k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    G_ref_2 = torch.zeros(B, H, T, r, r, device=device, dtype=torch.float32)
    R_ref_2 = torch.zeros(B, H, T, r, dv, device=device, dtype=torch.float32)
    curr_G = torch.zeros(B, H, r, r, device=device, dtype=torch.float32)
    curr_R = torch.zeros(B, H, r, dv, device=device, dtype=torch.float32)
    for t in range(T):
        curr_G = gamma * curr_G + torch.einsum("bhi,bhj->bhij", phi_k_ref[:, :, t], phi_k_ref[:, :, t])
        curr_R = gamma * curr_R + torch.einsum("bhi,bhj->bhij", phi_k_ref[:, :, t], v_ref[:, :, t])
        G_ref_2[:, :, t] = curr_G + lambda_reg * eye_r
        R_ref_2[:, :, t] = curr_R

    loss_ref = (G_ref_2 * grad_G).sum() + (R_ref_2 * grad_R).sum()
    loss_ref.backward()

    diff_grad_phi = (triton_grad_phi - phi_k_ref.grad).abs().max().item()
    diff_grad_v = (triton_grad_v - v_ref.grad).abs().max().item()
    print(f"Backward diff grad_phi: {diff_grad_phi:.2e}, diff grad_v: {diff_grad_v:.2e}")
    assert diff_grad_phi < 1e-4, f"Backward diff grad_phi too high: {diff_grad_phi}"
    assert diff_grad_v < 1e-4, f"Backward diff grad_v too high: {diff_grad_v}"
    print("✨ ALL TRITON TESTS PASSED! Numerical equivalence verified.")

    # 3. Latency Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Warmup
        for _ in range(10):
            G_t, R_t = FusedCausalOLSAccumFunction.apply(phi_k, v, gamma, lambda_reg)
            torch.cuda.synchronize()

        start = time.perf_counter()
        n_iters = 100
        for _ in range(n_iters):
            G_t, R_t = FusedCausalOLSAccumFunction.apply(phi_k, v, gamma, lambda_reg)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / n_iters * 1000.0
        print(f"⚡ Triton Kernel Latency: {triton_ms:.3f} ms per call")

    return {
        "forward_diff_G": diff_G,
        "forward_diff_R": diff_R,
        "backward_diff_phi": diff_grad_phi,
        "backward_diff_v": diff_grad_v,
    }

if __name__ == "__main__":
    test_fused_triton_correctness()
