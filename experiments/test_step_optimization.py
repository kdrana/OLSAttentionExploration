"""
Optimize step() in FlashCausalOLSAttentionV2 using direct torch.bmm / batched GEMV.
"""
import time
import torch

B, H, r, dh = 2, 4, 16, 16
gamma = 0.99

P = torch.eye(r).repeat(B, H, 1, 1)
R = torch.zeros(B, H, r, dh)
phi_k = torch.randn(B, H, r)
phi_q = torch.randn(B, H, r)
v = torch.randn(B, H, dh)

# 1. Baseline einsum step
def step_einsum(P, R, phi_k, phi_q, v):
    P_k = torch.einsum("bhde,bhe->bhd", P, phi_k)
    k_P_k = torch.einsum("bhd,bhd->bh", phi_k, P_k)
    denom = (gamma + k_P_k).unsqueeze(-1).unsqueeze(-1)
    P_outer = torch.einsum("bhd,bhe->bhde", P_k, P_k)
    new_P = (1.0 / gamma) * (P - P_outer / denom)
    new_R = gamma * R + torch.einsum("bhd,bhe->bhde", phi_k, v)
    W = torch.einsum("bhde,bhef->bhdf", new_P, new_R)
    y = torch.einsum("bhd,bhde->bhe", phi_q, W)
    return y, new_P, new_R

# 2. Optimized direct batched GEMV / GEMM step
def step_optimized(P, R, phi_k, phi_q, v):
    # Flatten (B, H) -> BH
    BH = B * H
    P_flat = P.view(BH, r, r)
    R_flat = R.view(BH, r, dh)
    k_flat = phi_k.view(BH, r, 1)
    q_flat = phi_q.view(BH, 1, r)
    v_flat = v.view(BH, 1, dh)

    # Batched matrix-vector: P * k -> (BH, r, 1)
    Pk = torch.bmm(P_flat, k_flat)
    # k^T P k -> (BH, 1, 1)
    kPk = torch.bmm(k_flat.transpose(1, 2), Pk)
    denom = gamma + kPk

    # P_outer = Pk * Pk^T -> (BH, r, r)
    P_outer = torch.bmm(Pk, Pk.transpose(1, 2))
    new_P = (1.0 / gamma) * (P_flat - P_outer / denom)

    # R update: gamma * R + k * v^T -> (BH, r, dh)
    new_R = gamma * R_flat + torch.bmm(k_flat, v_flat)

    # W = new_P * new_R -> (BH, r, dh)
    W = torch.bmm(new_P, new_R)

    # y = q * W -> (BH, 1, dh)
    y = torch.bmm(q_flat, W).view(B, H, dh)
    return y, new_P.view(B, H, r, r), new_R.view(B, H, r, dh)

# Warmup
y1, P1, R1 = step_einsum(P, R, phi_k, phi_q, v)
y2, P2, R2 = step_optimized(P, R, phi_k, phi_q, v)

print(f"Numerical Difference: {(y1 - y2).abs().max().item():.4e}")

# Latency benchmark
N = 1000
t0 = time.perf_counter()
curr_P, curr_R = P, R
for _ in range(N):
    _, curr_P, curr_R = step_einsum(curr_P, curr_R, phi_k, phi_q, v)
t_einsum = (time.perf_counter() - t0) * 1000.0 / N

t0 = time.perf_counter()
curr_P, curr_R = P, R
for _ in range(N):
    _, curr_P, curr_R = step_optimized(curr_P, curr_R, phi_k, phi_q, v)
t_opt = (time.perf_counter() - t0) * 1000.0 / N

print(f"step_einsum:    {t_einsum*1000:.2f} microseconds per token ({1000.0/t_einsum:,.0f} tok/s)")
print(f"step_optimized: {t_opt*1000:.2f} microseconds per token ({1000.0/t_opt:,.0f} tok/s)")
print(f"Speedup: {t_einsum/t_opt:.2f}x")
