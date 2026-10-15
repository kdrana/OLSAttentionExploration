"""
Data loader for California Housing, Kin8nm, and Friedman-1 benchmarks.
Applies standard train/test splits and z-score standardization.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, fetch_openml, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_benchmark_dataset(
    name: str, test_size: float = 0.2, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Loads and standardizes benchmark dataset.
    Returns: X_train, y_train, X_test, y_test, y_mean, y_std
    """
    name = name.lower()

    if name == "california":
        cal = fetch_california_housing()
        X = cal.data
        y = cal.target.reshape(-1, 1)

    elif name == "kin8nm":
        kin = fetch_openml("kin8nm", version=1, as_frame=False)
        X = np.array(kin.data, dtype=np.float32)
        y = np.array(kin.target, dtype=np.float32).reshape(-1, 1)

    elif name == "friedman1":
        X, y = make_friedman1(n_samples=10000, n_features=10, noise=0.1, random_state=seed)
        y = y.reshape(-1, 1)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Standardize X and y using training split parameters
    x_scaler = StandardScaler()
    X_train_std = x_scaler.fit_transform(X_train)
    X_test_std = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_std = y_scaler.fit_transform(y_train)
    y_test_std = y_scaler.transform(y_test)

    y_mean = float(y_scaler.mean_[0])
    y_std = float(y_scaler.scale_[0])

    return (
        torch.tensor(X_train_std, dtype=torch.float32),
        torch.tensor(y_train_std, dtype=torch.float32),
        torch.tensor(X_test_std, dtype=torch.float32),
        torch.tensor(y_test_std, dtype=torch.float32),
        y_mean,
        y_std,
    )
