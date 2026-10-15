"""
Main experiment runner for exploring basis function approximations of Softmax Attention.
Supports Apple Silicon (MPS), CUDA, and standard CPU.
"""

import os
import json
import math
import torch
import numpy as np
from kernels import (
    exact_softmax_attention,
    taylor_polynomial_attention,
    positive_random_features_attention,
    gegenbauer_bessel_spectrum,
)
from metrics import relative_frobenius_error, attention_entropy, effective_rank, r2_score
from tasks import generate_synthetic_data


def get_device() -> torch.device:
    """Selects Apple Silicon MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def experiment_basis_approximation_vs_beta(
    device: torch.device, T: int = 128, d: int = 16, n_trials: int = 5
) -> dict:
    """
    Exp 1: Evaluates relative Frobenius error of basis approximations against exact Softmax
    as inverse temperature beta increases from 0.05 to 5.0.
    """
    print("Running Experiment 1: Basis Approximation Error vs. Beta...")
    betas = np.linspace(0.05, 4.0, 25).tolist()

    results = {
        "betas": betas,
        "deg1_error": [],
        "deg2_error": [],
        "prf_128_error": [],
        "prf_512_error": [],
    }

    for beta in betas:
        err_deg1_list, err_deg2_list, err_prf128_list, err_prf512_list = [], [], [], []

        for trial in range(n_trials):
            # Random tokens on S^{d-1}
            torch.manual_seed(1000 * trial + int(beta * 100))
            Q = torch.nn.functional.normalize(torch.randn(1, T, d, device=device), p=2, dim=-1)
            K = torch.nn.functional.normalize(torch.randn(1, T, d, device=device), p=2, dim=-1)
            V = torch.randn(1, T, d, device=device)

            # Ground truth: Exact Softmax Attention
            _, A_true = exact_softmax_attention(Q, K, V, beta)

            # Approximations
            _, A_deg1 = taylor_polynomial_attention(Q, K, V, beta, degree=1)
            _, A_deg2 = taylor_polynomial_attention(Q, K, V, beta, degree=2)
            _, A_prf128 = positive_random_features_attention(Q, K, V, beta, num_features=128)
            _, A_prf512 = positive_random_features_attention(Q, K, V, beta, num_features=512)

            err_deg1_list.append(relative_frobenius_error(A_deg1, A_true))
            err_deg2_list.append(relative_frobenius_error(A_deg2, A_true))
            err_prf128_list.append(relative_frobenius_error(A_prf128, A_true))
            err_prf512_list.append(relative_frobenius_error(A_prf512, A_true))

        results["deg1_error"].append(float(np.mean(err_deg1_list)))
        results["deg2_error"].append(float(np.mean(err_deg2_list)))
        results["prf_128_error"].append(float(np.mean(err_prf128_list)))
        results["prf_512_error"].append(float(np.mean(err_prf512_list)))

    return results


def experiment_bessel_harmonic_spectrum(d: int = 16, max_order: int = 4) -> dict:
    """
    Exp 2: Evaluates exact analytical Gegenbauer / Bessel harmonic power spectrum
    across orders n=0 (uniform), n=1 (linear/OLS), n=2 (quadratic), etc.
    """
    print("Running Experiment 2: Gegenbauer-Bessel Harmonic Power Spectrum...")
    betas = np.linspace(0.01, 8.0, 50).tolist()
    orders = list(range(max_order + 1))
    
    spectrum_by_order = {n: [] for n in orders}

    for beta in betas:
        c_coeffs = gegenbauer_bessel_spectrum(beta, d=d, max_order=max_order)
        for n in orders:
            spectrum_by_order[n].append(float(c_coeffs[n]))

    return {"betas": betas, "spectrum": spectrum_by_order, "d": d}


def experiment_downstream_r2(device: torch.device, n_train: int = 500, n_test: int = 200, d: int = 8) -> dict:
    """
    Exp 3: Evaluates out-of-sample R^2 for In-Context / Non-parametric regression
    comparing Exact Softmax, Degree 1 (OLS-like), Degree 2 (Quadratic), and PRF.
    """
    print("Running Experiment 3: In-Context Regression R^2 across DGPs...")
    dgps = ["linear", "interaction", "friedman"]
    beta = 1.0 / math.sqrt(d)  # Canonical 1/sqrt(d) temperature

    results = {}

    for dgp in dgps:
        X_train, y_train = generate_synthetic_data(dgp=dgp, n_samples=n_train, d=d, noise_std=0.05, seed=42)
        X_test, y_test = generate_synthetic_data(dgp=dgp, n_samples=n_test, d=d, noise_std=0.05, seed=123)

        X_train, y_train = X_train.unsqueeze(0).to(device), y_train.unsqueeze(0).to(device)
        X_test, y_test = X_test.unsqueeze(0).to(device), y_test.unsqueeze(0).to(device)

        # Project features using OLS/Ridge precision metric: W = (X_train^T X_train + lambda*I)^{-1/2}
        X_tr_mat = X_train.squeeze(0)  # (N, d)
        Gram = torch.matmul(X_tr_mat.T, X_tr_mat) + 1e-3 * torch.eye(d, device=device)
        eigvals, eigvecs = torch.linalg.eigh(Gram)
        inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(eigvals, min=1e-6)))
        W_proj = torch.matmul(torch.matmul(eigvecs, inv_sqrt), eigvecs.T)  # (d, d)

        # Projected queries and keys
        Q = torch.matmul(X_test, W_proj)
        K = torch.matmul(X_train, W_proj)

        # 1. Exact Linear OLS / Attention (Degree 1 unnormalized: Q * K^T * y)
        y_pred_ols = torch.matmul(torch.matmul(Q, K.transpose(-2, -1)), y_train)

        # 2. Softmax Attention with learned metric W
        y_pred_softmax, _ = exact_softmax_attention(Q, K, y_train, beta=1.0)

        # 3. Degree 2 Polynomial Attention (Pairwise features + metric)
        # Degree-2 feature expansion: [1, x, vec(x x^T)]
        def poly2_features(X_mat):
            B_s, N_s, d_s = X_mat.shape
            ones = torch.ones((B_s, N_s, 1), device=device)
            outer = (X_mat.unsqueeze(-1) * X_mat.unsqueeze(-2)).reshape(B_s, N_s, d_s * d_s)
            return torch.cat([ones, X_mat, outer / math.sqrt(2.0)], dim=-1)

        Phi_train = poly2_features(X_train).squeeze(0)
        Phi_test = poly2_features(X_test).squeeze(0)
        Gram_poly = torch.matmul(Phi_train.T, Phi_train) + 1e-2 * torch.eye(Phi_train.shape[-1], device=device)
        w_poly = torch.linalg.solve(Gram_poly, torch.matmul(Phi_train.T, y_train.squeeze(0)))
        y_pred_deg2 = torch.matmul(Phi_test, w_poly).unsqueeze(0)

        # 4. Positive Random Features (FAVOR+)
        y_pred_prf, _ = positive_random_features_attention(Q, K, y_train, beta=1.0, num_features=256)

        results[dgp] = {
            "Degree 1 (Exact OLS)": r2_score(y_test, y_pred_ols),
            "Degree 2 (Polynomial)": r2_score(y_test, y_pred_deg2),
            "Softmax Attention": r2_score(y_test, y_pred_softmax),
            "Positive Random Features": r2_score(y_test, y_pred_prf),
        }

    return results


def experiment_entropy_and_rank(device: torch.device, T: int = 128, d: int = 16) -> dict:
    """
    Exp 4: Measures attention entropy H(A) and effective rank of output representation
    as temperature beta varies.
    """
    print("Running Experiment 4: Attention Entropy & Output Effective Rank...")
    betas = np.linspace(0.05, 5.0, 30).tolist()
    entropies = []
    eranks = []

    for beta in betas:
        torch.manual_seed(42)
        Q = torch.nn.functional.normalize(torch.randn(1, T, d, device=device), p=2, dim=-1)
        K = torch.nn.functional.normalize(torch.randn(1, T, d, device=device), p=2, dim=-1)
        V = torch.randn(1, T, d, device=device)

        out, A = exact_softmax_attention(Q, K, V, beta)
        entropies.append(attention_entropy(A))
        eranks.append(effective_rank(out[0]))

    return {"betas": betas, "entropies": entropies, "eranks": eranks}


def main():
    device = get_device()
    print(f"Using device: {device} ({'Apple Silicon Metal' if device.type == 'mps' else device.type.upper()})")

    os.makedirs("results", exist_ok=True)

    # Run experiments
    exp1_data = experiment_basis_approximation_vs_beta(device)
    exp2_data = experiment_bessel_harmonic_spectrum(d=16, max_order=4)
    exp3_data = experiment_downstream_r2(device)
    exp4_data = experiment_entropy_and_rank(device)

    # Save data to json
    all_results = {
        "device": str(device),
        "exp1_approx_error": exp1_data,
        "exp2_harmonic_spectrum": exp2_data,
        "exp3_downstream_r2": exp3_data,
        "exp4_entropy_rank": exp4_data,
    }

    with open("results/experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("All experiments completed successfully! Results saved to results/experiment_results.json")


if __name__ == "__main__":
    main()
