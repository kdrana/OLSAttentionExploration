"""
Mathematical basis functions and kernel feature maps for approximating Softmax Attention.

Includes:
1. Exact Softmax Attention
2. Truncated Taylor / Polynomial Feature Maps (Degree 1 = OLS-like, Degree 2 = Pairwise interactions)
3. Positive Random Features (FAVOR+ / Performer)
4. Gegenbauer (Ultraspherical) Polynomial Kernel on S^{d-1} with Bessel coefficients
"""

import math
import torch
import numpy as np
from scipy.special import iv, gamma


def exact_softmax_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard scaled dot-product softmax attention.
    Q, K: (B, T, d)
    V: (B, T, d_v)
    Returns: (Output (B, T, d_v), Attention Matrix (B, T, T))
    """
    scores = beta * torch.matmul(Q, K.transpose(-2, -1))
    A = torch.softmax(scores, dim=-1)
    out = torch.matmul(A, V)
    return out, A


def taylor_polynomial_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, beta: float, degree: int = 1, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximates exp(beta * q^T k) using truncated Taylor polynomial feature maps.
    
    Degree 1: phi(x) = [1, sqrt(beta)*x]
    Degree 2: phi(x) = [1, sqrt(beta)*x, sqrt(beta^2 / 2)*vec(x * x^T)]
    
    Decoupled linear computation: (phi(Q) * (phi(K)^T * V)) / (phi(Q) * sum(phi(K)))
    """
    B, T_Q, d = Q.shape
    _, T_K, _ = K.shape

    ones_Q = torch.ones((B, T_Q, 1), device=Q.device, dtype=Q.dtype)
    ones_K = torch.ones((B, T_K, 1), device=K.device, dtype=K.dtype)

    if degree == 1:
        phi_Q = torch.cat([ones_Q, math.sqrt(beta) * Q], dim=-1)
        phi_K = torch.cat([ones_K, math.sqrt(beta) * K], dim=-1)
    elif degree == 2:
        deg1_Q = math.sqrt(beta) * Q
        deg1_K = math.sqrt(beta) * K
        outer_Q = (Q.unsqueeze(-1) * Q.unsqueeze(-2)).reshape(B, T_Q, d * d)
        outer_K = (K.unsqueeze(-1) * K.unsqueeze(-2)).reshape(B, T_K, d * d)
        deg2_Q = math.sqrt(beta**2 / 2.0) * outer_Q
        deg2_K = math.sqrt(beta**2 / 2.0) * outer_K
        phi_Q = torch.cat([ones_Q, deg1_Q, deg2_Q], dim=-1)
        phi_K = torch.cat([ones_K, deg1_K, deg2_K], dim=-1)
    else:
        raise ValueError(f"Degree {degree} not implemented (use 1 or 2).")

    # Kernel matrix K_mat = phi_Q * phi_K^T
    K_mat = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
    
    # Avoid negative/zero denominator by clamping
    K_mat_pos = torch.clamp(K_mat, min=1e-8)
    row_sum = K_mat_pos.sum(dim=-1, keepdim=True) + eps
    A = K_mat_pos / row_sum
    out = torch.matmul(A, V)
    return out, A


def positive_random_features_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, beta: float, num_features: int = 128, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FAVOR+ (Choromanski et al., 2020) Positive Random Features (PRF) for exp(beta * q^T k).
    Guarantees non-negative kernel elements without trigonometric instability.
    """
    B, T, d = Q.shape
    device = Q.device
    dtype = Q.dtype

    # Sample random Gaussian projection vectors omega: (num_features, d)
    torch.manual_seed(42)
    omega = torch.randn((num_features, d), device=device, dtype=dtype)

    # Scaling factor for each sample: exp(-beta * ||x||^2 / 2)
    norm_sq_Q = torch.sum(Q**2, dim=-1, keepdim=True)  # (B, T, 1)
    norm_sq_K = torch.sum(K**2, dim=-1, keepdim=True)

    h_Q = torch.exp(-0.5 * beta * norm_sq_Q)
    h_K = torch.exp(-0.5 * beta * norm_sq_K)

    # Projections: (B, T, num_features)
    proj_Q = math.sqrt(beta) * torch.matmul(Q, omega.T)
    proj_K = math.sqrt(beta) * torch.matmul(K, omega.T)

    # Positive feature maps: phi(x) = h(x)/sqrt(M) * exp(proj - 0.5 * beta * ||omega||^2)
    omega_norm_sq = torch.sum(omega**2, dim=-1)  # (num_features,)
    shift = 0.5 * beta * omega_norm_sq

    phi_Q = (h_Q / math.sqrt(num_features)) * torch.exp(proj_Q - shift)
    phi_K = (h_K / math.sqrt(num_features)) * torch.exp(proj_K - shift)

    # Attention matrix
    K_mat = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
    row_sum = K_mat.sum(dim=-1, keepdim=True) + eps
    A = K_mat / row_sum
    out = torch.matmul(A, V)
    return out, A


def gegenbauer_bessel_spectrum(beta: float, d: int, max_order: int = 6) -> np.ndarray:
    """
    Computes the exact harmonic expansion coefficients of exp(beta * <q, k>) on S^{d-1}
    using Gegenbauer polynomials and modified Bessel functions of the first kind (I_nu).
    
    Formula: c_n(beta, d) = 2^(alpha) * Gamma(alpha) * (n + alpha) * I_{n + alpha}(beta) / beta^alpha
    where alpha = (d - 2) / 2.
    """
    alpha = (d - 2) / 2.0
    coeffs = []

    for n in range(max_order + 1):
        nu = n + alpha
        # Modified Bessel function of first kind
        bessel_val = iv(nu, beta)
        
        if alpha == 0:
            # Circle S^1 special case (alpha -> 0)
            c_n = (2.0 if n > 0 else 1.0) * bessel_val
        else:
            # General hypersphere S^{d-1}
            gamma_alpha = gamma(alpha)
            scaling = (2.0**alpha) * gamma_alpha * (n + alpha) / (beta**alpha + 1e-12)
            c_n = scaling * bessel_val
        coeffs.append(max(0.0, float(c_n)))

    coeffs = np.array(coeffs)
    total = np.sum(coeffs)
    if total > 0:
        return coeffs / total  # Normalized spectral power
    return coeffs
