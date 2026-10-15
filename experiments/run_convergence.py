"""
Training loop and convergence benchmarking for:
1. Standard FT-Transformer (Random Init)
2. Gram Metric Warm-Start FT-Transformer
3. Solvable Degree-2 Regression Block Model
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_benchmark_dataset
from src.models import StandardFTTransformer, GramInitFTTransformer, SolvableRegressionBlockModel
from src.metrics import r2_score


def train_single_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Trains a model and records trajectory: epochs, wallclock time, test R^2.
    """
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {
        "epochs": [],
        "wallclock_sec": [],
        "test_r2": [],
        "test_mse": [],
    }

    # Record Epoch 0 (initial performance before any SGD step!)
    model.eval()
    with torch.no_grad():
        y_pred_0 = model(X_test)
        initial_r2 = r2_score(y_test, y_pred_0)
        initial_mse = criterion(y_pred_0, y_test).item()

    history["epochs"].append(0)
    history["wallclock_sec"].append(0.0)
    history["test_r2"].append(initial_r2)
    history["test_mse"].append(initial_mse)

    start_time = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        elapsed = time.perf_counter() - start_time

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            test_mse = criterion(y_pred, y_test).item()
            test_r2 = r2_score(y_test, y_pred)

        history["epochs"].append(epoch)
        history["wallclock_sec"].append(elapsed)
        history["test_r2"].append(test_r2)
        history["test_mse"].append(test_mse)

    return history


def run_convergence_suite(
    device: torch.device = torch.device("cpu"), n_epochs: int = 40
) -> dict:
    """
    Runs the full 3x3 benchmark:
    Models: [Standard FT-T, Gram Warm-Start FT-T, Solvable RegBlock]
    Datasets: [California, Kin8nm, Friedman-1]
    """
    datasets = ["california", "kin8nm", "friedman1"]
    all_results = {}

    for d_name in datasets:
        print(f"\n=======================================================")
        print(f"Loading Benchmark Dataset: {d_name.upper()}...")
        print(f"=======================================================")
        X_tr, y_tr, X_te, y_te, _, _ = load_benchmark_dataset(d_name, seed=42)
        n_features = X_tr.shape[1]
        print(f"Train: {X_tr.shape}, Test: {X_te.shape}, Features: {n_features}")

        all_results[d_name] = {}

        # -------------------------------------------------------------
        # Model A: Standard FT-Transformer (Random Init)
        # -------------------------------------------------------------
        print("\n--- Training Model A: Standard FT-Transformer (Random Init) ---")
        torch.manual_seed(42)
        model_a = StandardFTTransformer(n_features=n_features, d_token=32, n_heads=4, d_ffn=64).to(device)
        hist_a = train_single_model(model_a, X_tr, y_tr, X_te, y_te, n_epochs=n_epochs, device=device)
        all_results[d_name]["Standard FT-T (Random Init)"] = hist_a
        print(f"Final Test R^2: {hist_a['test_r2'][-1]:.4f} in {hist_a['wallclock_sec'][-1]:.2f}s")

        # -------------------------------------------------------------
        # Model B: Gram Metric Warm-Start FT-Transformer
        # -------------------------------------------------------------
        print("\n--- Training Model B: Gram Metric Warm-Start FT-Transformer ---")
        torch.manual_seed(42)
        model_b = GramInitFTTransformer(n_features=n_features, d_token=32, n_heads=4, d_ffn=64).to(device)
        # Pre-align attention metric using empirical Gram precision matrix
        model_b.warm_start_gram_metric(X_tr[:1000].to(device))
        hist_b = train_single_model(model_b, X_tr, y_tr, X_te, y_te, n_epochs=n_epochs, device=device)
        all_results[d_name]["Gram Warm-Start FT-T"] = hist_b
        print(f"Final Test R^2: {hist_b['test_r2'][-1]:.4f} (Epoch 0 R^2: {hist_b['test_r2'][0]:.4f}) in {hist_b['wallclock_sec'][-1]:.2f}s")

        # -------------------------------------------------------------
        # Model C: Solvable Degree-2 Regression Block Model
        # -------------------------------------------------------------
        print("\n--- Training Model C: Solvable Degree-2 Regression Block ---")
        torch.manual_seed(42)
        model_c = SolvableRegressionBlockModel(n_features=n_features, d_token=32, n_components=64, d_ffn=64).to(device)
        # Solve attention mixing layer in a single closed-form Ridge step!
        model_c.fit_pca_and_solve_closed_form(X_tr.to(device), y_tr.to(device))
        hist_c = train_single_model(model_c, X_tr, y_tr, X_te, y_te, n_epochs=n_epochs, device=device)
        all_results[d_name]["Solvable RegBlock (Degree-2)"] = hist_c
        print(f"Final Test R^2: {hist_c['test_r2'][-1]:.4f} (Epoch 0 R^2: {hist_c['test_r2'][0]:.4f}) in {hist_c['wallclock_sec'][-1]:.2f}s")

    return all_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running convergence suite on {device}...")
    results = run_convergence_suite(device=device, n_epochs=35)

    out_dir = os.path.join(os.path.dirname(__file__), "../results/exp02_convergence_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Convergence results saved to {out_path}")


if __name__ == "__main__":
    main()
