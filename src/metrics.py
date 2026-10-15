"""
Mathematical evaluation metrics for attention analysis.
"""

import torch
import numpy as np


def relative_frobenius_error(A_approx: torch.Tensor, A_true: torch.Tensor) -> float:
    """
    Computes relative Frobenius norm distance: ||A_approx - A_true||_F / ||A_true||_F
    """
    diff = A_approx - A_true
    num = torch.norm(diff, p="fro").item()
    denom = torch.norm(A_true, p="fro").item() + 1e-12
    return float(num / denom)


def attention_entropy(A: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes mean entropy of the attention distribution across tokens:
    H(A) = - 1/T sum_{i} sum_{j} A_{ij} log(A_{ij} + eps)
    """
    A_clamped = torch.clamp(A, min=eps)
    ent = -torch.sum(A_clamped * torch.log(A_clamped), dim=-1)
    return float(torch.mean(ent).item())


def effective_rank(X: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes the stable / effective rank of a representation matrix X:
    erank(X) = exp( - sum_i p_i log(p_i) ) where p_i = s_i / sum_k s_k (normalized singular values).
    """
    # X: (T, d)
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])
    try:
        _, S, _ = torch.linalg.svd(X.float(), full_matrices=False)
        S_sum = torch.sum(S) + eps
        p = S / S_sum
        p = torch.clamp(p, min=eps)
        H_s = -torch.sum(p * torch.log(p))
        return float(torch.exp(H_s).item())
    except Exception:
        return 1.0


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes out-of-sample R^2 (coefficient of determination).
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2).item()
    ss_tot = torch.sum((y_true_flat - torch.mean(y_true_flat)) ** 2).item() + 1e-12
    return float(1.0 - (ss_res / ss_tot))
