"""
Fused Triton Flash-OLS Causal Gram & RHS Accumulation Kernels.

Eliminates the intermediate 6D outer-product tensor allocations and cumsum launches
by computing causal Gram matrix G_t and cross-covariance R_t directly in GPU registers/SRAM.

Forward:
    G_t = gamma * G_{t-1} + phi_k(t) phi_k(t)^T + lambda * I
    R_t = gamma * R_{t-1} + phi_k(t) v(t)^T

Backward (Analytical Adjoint):
    G_future(t) = grad_G(t) + gamma * G_future(t+1)
    R_future(t) = grad_R(t) + gamma * R_future(t+1)
    grad_phi_k(t) = 2 * G_future(t) phi_k(t) + R_future(t) v(t)
    grad_v(t)     = R_future(t)^T phi_k(t)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_causal_accum_fwd_kernel(
        phi_k_ptr,
        v_ptr,
        G_ptr,
        R_ptr,
        T: int,
        gamma: float,
        lambda_reg: float,
        stride_kb: int, stride_kh: int, stride_kt: int, stride_kr: int,
        stride_vb: int, stride_vh: int, stride_vt: int, stride_vd: int,
        stride_gb: int, stride_gh: int, stride_gt: int, stride_gr1: int, stride_gr2: int,
        stride_rb: int, stride_rh: int, stride_rt: int, stride_rr: int, stride_rd: int,
        H: int,
        BLOCK_R: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H

        offs_r = tl.arange(0, BLOCK_R)
        offs_dv = tl.arange(0, BLOCK_DV)

        offs_r_row = offs_r[:, None]
        offs_r_col = offs_r[None, :]
        offs_dv_col = offs_dv[None, :]

        eye_r = (offs_r_row == offs_r_col).to(tl.float32)

        G_acc = tl.zeros([BLOCK_R, BLOCK_R], dtype=tl.float32)
        R_acc = tl.zeros([BLOCK_R, BLOCK_DV], dtype=tl.float32)

        phi_base = phi_k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        G_base = G_ptr + b * stride_gb + h * stride_gh
        R_base = R_ptr + b * stride_rb + h * stride_rh

        for t in range(T):
            phi_col = tl.load(phi_base + t * stride_kt + offs_r_row * stride_kr)
            phi_row = tl.load(phi_base + t * stride_kt + offs_r_col * stride_kr)
            v_row = tl.load(v_base + t * stride_vt + offs_dv_col * stride_vd)

            dG = phi_col * phi_row
            dR = phi_col * v_row

            G_acc = gamma * G_acc + dG
            R_acc = gamma * R_acc + dR

            G_out = G_acc + lambda_reg * eye_r

            tl.store(
                G_base + t * stride_gt + offs_r_row * stride_gr1 + offs_r_col * stride_gr2,
                G_out
            )
            tl.store(
                R_base + t * stride_rt + offs_r_row * stride_rr + offs_dv_col * stride_rd,
                R_acc
            )


    @triton.jit
    def _fused_causal_accum_bwd_kernel(
        grad_G_ptr,
        grad_R_ptr,
        phi_k_ptr,
        v_ptr,
        grad_phi_ptr,
        grad_v_ptr,
        T: int,
        gamma: float,
        stride_gb: int, stride_gh: int, stride_gt: int, stride_gr1: int, stride_gr2: int,
        stride_rb: int, stride_rh: int, stride_rt: int, stride_rr: int, stride_rd: int,
        stride_kb: int, stride_kh: int, stride_kt: int, stride_kr: int,
        stride_vb: int, stride_vh: int, stride_vt: int, stride_vd: int,
        stride_okb: int, stride_okh: int, stride_okt: int, stride_okr: int,
        stride_ovb: int, stride_ovh: int, stride_ovt: int, stride_ovd: int,
        H: int,
        BLOCK_R: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H

        offs_r = tl.arange(0, BLOCK_R)
        offs_dv = tl.arange(0, BLOCK_DV)

        offs_r_row = offs_r[:, None]
        offs_r_col = offs_r[None, :]
        offs_dv_col = offs_dv[None, :]

        G_future = tl.zeros([BLOCK_R, BLOCK_R], dtype=tl.float32)
        R_future = tl.zeros([BLOCK_R, BLOCK_DV], dtype=tl.float32)

        grad_G_base = grad_G_ptr + b * stride_gb + h * stride_gh
        grad_R_base = grad_R_ptr + b * stride_rb + h * stride_rh
        phi_base = phi_k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        grad_phi_base = grad_phi_ptr + b * stride_okb + h * stride_okh
        grad_v_base = grad_v_ptr + b * stride_ovb + h * stride_ovh

        # Anti-causal loop running backwards in time
        for step in range(T):
            t = T - 1 - step

            grad_G_t = tl.load(grad_G_base + t * stride_gt + offs_r_row * stride_gr1 + offs_r_col * stride_gr2)
            grad_R_t = tl.load(grad_R_base + t * stride_rt + offs_r_row * stride_rr + offs_dv_col * stride_rd)

            G_future = grad_G_t + gamma * G_future
            R_future = grad_R_t + gamma * R_future

            phi_col = tl.load(phi_base + t * stride_kt + offs_r_row * stride_kr)
            phi_row = tl.load(phi_base + t * stride_kt + offs_r_col * stride_kr)
            v_row = tl.load(v_base + t * stride_vt + offs_dv_col * stride_vd)

            # dL / dphi = 2 * (G_future @ phi) + R_future @ v
            term_G = 2.0 * tl.sum(G_future * phi_row, axis=1)
            term_R = tl.sum(R_future * v_row, axis=1)
            grad_phi_t = term_G + term_R

            # dL / dv = R_future^T @ phi
            grad_v_t = tl.sum(R_future * phi_col, axis=0)

            tl.store(grad_phi_base + t * stride_okt + offs_r * stride_okr, grad_phi_t)
            tl.store(grad_v_base + t * stride_ovt + offs_dv * stride_ovd, grad_v_t)


class FusedCausalOLSAccumFunction(torch.autograd.Function):
    """
    Fused Triton Autograd Function for Causal Gram (G) and Cross-Covariance (R) accumulation.
    Executes in-SRAM without allocating any intermediate 6D tensors.
    """
    @staticmethod
    def forward(ctx, phi_k: torch.Tensor, v: torch.Tensor, gamma: float = 0.99, lambda_reg: float = 0.05):
        orig_phi_dtype = phi_k.dtype
        orig_v_dtype = v.dtype

        phi_k = phi_k.contiguous().float()
        v = v.contiguous().float()

        B, H, T, r = phi_k.shape
        _, _, _, dv = v.shape

        assert r in (8, 16, 32, 64), f"Rank {r} must be power-of-2 (8 to 64) for Triton tile."
        assert dv in (8, 16, 32, 64, 128, 256), f"dv {dv} must be power-of-2 (8 to 256) for Triton tile."

        if not phi_k.is_cuda or not TRITON_AVAILABLE:
            # High-fidelity PyTorch CPU fallback
            G = torch.zeros(B, H, T, r, r, device=phi_k.device, dtype=torch.float32)
            R = torch.zeros(B, H, T, r, dv, device=v.device, dtype=torch.float32)
            curr_G = torch.zeros(B, H, r, r, device=phi_k.device, dtype=torch.float32)
            curr_R = torch.zeros(B, H, r, dv, device=v.device, dtype=torch.float32)
            eye_r = torch.eye(r, device=phi_k.device, dtype=torch.float32).view(1, 1, r, r)

            for t in range(T):
                p_t = phi_k[:, :, t]
                v_t = v[:, :, t]
                curr_G = gamma * curr_G + torch.einsum("bhi,bhj->bhij", p_t, p_t)
                curr_R = gamma * curr_R + torch.einsum("bhi,bhj->bhij", p_t, v_t)
                G[:, :, t] = curr_G + lambda_reg * eye_r
                R[:, :, t] = curr_R
        else:
            G = torch.empty(B, H, T, r, r, device=phi_k.device, dtype=torch.float32)
            R = torch.empty(B, H, T, r, dv, device=v.device, dtype=torch.float32)

            grid = (B * H,)
            n_warps = 8 if (r >= 64 or dv >= 256) else 4
            _fused_causal_accum_fwd_kernel[grid](
                phi_k, v, G, R,
                T, float(gamma), float(lambda_reg),
                phi_k.stride(0), phi_k.stride(1), phi_k.stride(2), phi_k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
                R.stride(0), R.stride(1), R.stride(2), R.stride(3), R.stride(4),
                H,
                BLOCK_R=r,
                BLOCK_DV=dv,
                num_warps=n_warps,
            )

        ctx.save_for_backward(phi_k, v)
        ctx.gamma = gamma
        ctx.orig_phi_dtype = orig_phi_dtype
        ctx.orig_v_dtype = orig_v_dtype
        ctx.r = r
        ctx.dv = dv
        return G, R

    @staticmethod
    def backward(ctx, grad_G: torch.Tensor, grad_R: torch.Tensor):
        phi_k, v = ctx.saved_tensors
        gamma = ctx.gamma
        r = ctx.r
        dv = ctx.dv
        B, H, T, _ = phi_k.shape

        grad_G = grad_G.contiguous().float()
        grad_R = grad_R.contiguous().float()

        if not phi_k.is_cuda or not TRITON_AVAILABLE:
            # High-fidelity PyTorch CPU fallback
            grad_phi = torch.zeros_like(phi_k)
            grad_v = torch.zeros_like(v)
            G_future = torch.zeros(B, H, r, r, device=phi_k.device, dtype=torch.float32)
            R_future = torch.zeros(B, H, r, dv, device=v.device, dtype=torch.float32)

            for step in range(T):
                t = T - 1 - step
                G_future = grad_G[:, :, t] + gamma * G_future
                R_future = grad_R[:, :, t] + gamma * R_future

                p_t = phi_k[:, :, t]
                v_t = v[:, :, t]

                # dL/dphi = 2 * G_future @ phi + R_future @ v
                term_G = 2.0 * torch.matmul(G_future, p_t.unsqueeze(-1)).squeeze(-1)
                term_R = torch.matmul(R_future, v_t.unsqueeze(-1)).squeeze(-1)
                grad_phi[:, :, t] = term_G + term_R

                # dL/dv = R_future^T @ phi
                grad_v[:, :, t] = torch.matmul(R_future.transpose(-1, -2), p_t.unsqueeze(-1)).squeeze(-1)
        else:
            grad_phi = torch.empty_like(phi_k)
            grad_v = torch.empty_like(v)

            grid = (B * H,)
            n_warps = 8 if (r >= 64 or dv >= 256) else 4
            _fused_causal_accum_bwd_kernel[grid](
                grad_G, grad_R, phi_k, v, grad_phi, grad_v,
                T, float(gamma),
                grad_G.stride(0), grad_G.stride(1), grad_G.stride(2), grad_G.stride(3), grad_G.stride(4),
                grad_R.stride(0), grad_R.stride(1), grad_R.stride(2), grad_R.stride(3), grad_R.stride(4),
                phi_k.stride(0), phi_k.stride(1), phi_k.stride(2), phi_k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                grad_phi.stride(0), grad_phi.stride(1), grad_phi.stride(2), grad_phi.stride(3),
                grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
                H,
                BLOCK_R=r,
                BLOCK_DV=dv,
                num_warps=n_warps,
            )

        return grad_phi.to(ctx.orig_phi_dtype), grad_v.to(ctx.orig_v_dtype), None, None


class CausalOLSSolveFunction(torch.autograd.Function):
    """
    Exact analytical implicit differentiation for causal regularized OLS normal equations.
    Forward: W* = G^{-1} R via batched Cholesky.
    Backward: dR = G^{-1} dW, dG = -0.5 * (dR W^T + W dR^T) via a single batched solve.
    """
    @staticmethod
    def forward(ctx, G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        L = torch.linalg.cholesky(G)
        W = torch.cholesky_solve(R, L)
        ctx.save_for_backward(L, W)
        return W

    @staticmethod
    def backward(ctx, grad_W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L, W = ctx.saved_tensors
        grad_R = torch.cholesky_solve(grad_W, L)
        grad_G = -0.5 * (torch.matmul(grad_R, W.transpose(-1, -2)) + torch.matmul(W, grad_R.transpose(-1, -2)))
        return grad_G, grad_R


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    B, H, T, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, -1, 2))
    freqs = freqs_cis[:T].view(1, 1, T, -1)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, T, d)
    return x_rotated.type_as(x)


class FusedLlamaFlashOLSAttention(nn.Module):
    """
    Fused Flash-OLS V4 (Triton SRAM Accumulator):
    1. In-register/SRAM causal recurrence via Triton JIT kernel.
    2. Zero intermediate 6D outer-product tensor allocations in DRAM.
    3. Analytical adjoint backward pass without cumsum DAG traversal.
    4. Exact 1:1 parameter parity with Fast-Flash-OLS V3 and Softmax SLM (9,946,368 params).
    """
    def __init__(self, d_model: int, n_heads: int, rank: int = 16, lambda_reg: float = 0.05, gamma: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.n_blocks = 2
        self.dh_sub = self.d_head // self.n_blocks
        self.lambda_reg = lambda_reg
        self.gamma = gamma

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        d_raw = self.d_head * 2
        proj = torch.randn(d_raw, self.n_blocks * self.rank)
        q_orth, _ = torch.linalg.qr(proj)
        self.subspace_proj = nn.Parameter(q_orth.view(d_raw, self.n_blocks, self.rank), requires_grad=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.cat([x, torch.tanh(x ** 2)], dim=-1)
        return torch.einsum("bhtd,dkr->bhtkr", phi, self.subspace_proj)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B, T, C_dim = x.shape
        H, dh, r = self.n_heads, self.d_head, self.rank

        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        H_eff = H * self.n_blocks
        phi_q = phi_q.permute(0, 1, 3, 2, 4).reshape(B, H_eff, T, r)
        phi_k = phi_k.permute(0, 1, 3, 2, 4).reshape(B, H_eff, T, r)
        v_sub = v.view(B, H, T, self.n_blocks, self.dh_sub).permute(0, 1, 3, 2, 4).reshape(B, H_eff, T, self.dh_sub)

        # Fused Causal Accumulation in GPU SRAM
        G, R = FusedCausalOLSAccumFunction.apply(phi_k, v_sub, self.gamma, self.lambda_reg)

        total_batch = B * H_eff * T
        W = CausalOLSSolveFunction.apply(
            G.reshape(total_batch, r, r),
            R.reshape(total_batch, r, self.dh_sub)
        ).view(B, H_eff, T, r, self.dh_sub)

        # Token-wise OLS projection
        y_sub = torch.matmul(phi_q.unsqueeze(-2), W).squeeze(-2)
        y_out = y_sub.view(B, H, self.n_blocks, T, self.dh_sub).permute(0, 1, 3, 2, 4).reshape(B, H, T, dh)
        y = y_out.to(orig_dtype).transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.out_proj(y)

