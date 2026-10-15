"""
Synthetic Data-Generating Processes (DGPs) to test linear vs. polynomial vs. full-attention regression.
"""

import math
import torch


def generate_synthetic_data(
    dgp: str = "linear", n_samples: int = 1000, d: int = 8, noise_std: float = 0.1, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic regression data (X, y).
    X is normalized to lie on S^{d-1} (unit hypersphere), mirroring LayerNorm.
    """
    torch.manual_seed(seed)
    
    # Generate X and normalize to S^{d-1}
    X_raw = torch.randn(n_samples, d)
    X = torch.nn.functional.normalize(X_raw, p=2, dim=-1)

    # Fix DGP ground truth parameters
    g_dgp = torch.Generator().manual_seed(999)

    if dgp == "linear":
        w = torch.randn(d, 1, generator=g_dgp)
        w = w / torch.norm(w)
        y = torch.matmul(X, w)

    elif dgp == "interaction":
        w = torch.randn(d, 1, generator=g_dgp) * 0.5
        y = torch.matmul(X, w)
        for i in range(min(4, d)):
            for j in range(i + 1, min(4, d)):
                y += (X[:, i : i + 1] * X[:, j : j + 1]) * 1.5

    elif dgp == "friedman":
        # Nonlinear benchmark (Friedman-1 style on sphere)
        # Uses sin, square, and linear terms
        x0 = X[:, 0:1]
        x1 = X[:, 1:2] if d > 1 else X[:, 0:1]
        x2 = X[:, 2:3] if d > 2 else X[:, 0:1]
        x3 = X[:, 3:4] if d > 3 else X[:, 0:1]

        y = 2.0 * torch.sin(math.pi * x0 * x1) + 3.0 * (x2**2) + 1.5 * x3
    else:
        raise ValueError(f"Unknown DGP: {dgp}")

    # Add Gaussian noise
    y += torch.randn_like(y) * noise_std
    return X, y
