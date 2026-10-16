"""
Benchmark Runner for EXP-10: Alternating Least Squares (ALS) Solvable Refinement.
Compares Greedy Solvable Cascade against ALS Block-Coordinate Refinement,
Gradient Boosted Decision Trees (XGBoost, LightGBM), and FT-Transformer (AdamW).

Evaluates on Kin8nm, Friedman-1, and California Housing benchmarks.
"""

import os
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from src.data_loader import load_benchmark_dataset
from src.models import DeepSolvableCascade, StandardFTTransformer, compute_poly2_features
from src.metrics import r2_score


def eval_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_std: float = 1.0) -> tuple[float, float]:
    """Computes R^2 and unstandardized RMSE."""
    r2 = r2_score(y_true, y_pred)
    mse = float(torch.mean((y_true - y_pred) ** 2).item())
    rmse = math.sqrt(max(0.0, mse)) * y_std
    return r2, rmse


def train_ft_transformer_with_history(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    y_std: float,
    device: torch.device,
    n_epochs: int = 35,
    batch_size: int = 256,
) -> dict:
    """Trains Standard FT-Transformer and records epoch trajectory."""
    n_features = X_train.shape[1]
    model = StandardFTTransformer(n_features=n_features, d_token=32, n_heads=4, d_ffn=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train.to(device), y_train.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X_te = X_test.to(device)
    y_te = y_test.to(device)
    X_tr = X_train.to(device)
    y_tr = y_train.to(device)

    epoch_history = []
    t0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(X_te)
            pred_tr = model(X_tr)
            r2_tr, rmse_tr = eval_metrics(y_tr, pred_tr, y_std)
            r2_te, rmse_te = eval_metrics(y_te, pred_te, y_std)

        epoch_history.append({
            "epoch": epoch,
            "train_r2": round(r2_tr, 5),
            "test_r2": round(r2_te, 5),
            "test_rmse": round(rmse_te, 5),
        })

    fit_time = time.perf_counter() - t0

    return {
        "train_r2": round(r2_tr, 5),
        "test_r2": round(r2_te, 5),
        "train_rmse": round(rmse_tr, 5),
        "test_rmse": round(rmse_te, 5),
        "fit_time_sec": round(fit_time, 4),
        "fit_time_ms": round(fit_time * 1000.0, 2),
        "epoch_history": epoch_history,
        "approach": "Iterative Backprop (AdamW)",
    }


def run_benchmark_for_dataset(dataset_name: str, device: torch.device, n_stages: int = 4, n_als_cycles: int = 5) -> dict:
    print(f"\n{'='*70}\n>>> Running EXP-10 Benchmark on: {dataset_name.upper()} <<<\n{'='*70}")
    X_tr, y_tr, X_te, y_te, y_mean, y_std = load_benchmark_dataset(dataset_name, test_size=0.2, seed=42)
    N_tr, d = X_tr.shape
    N_te = X_te.shape[0]
    print(f"Dataset stats: Train N={N_tr}, Test N={N_te}, Features d={d}")

    dataset_results = {
        "dataset": dataset_name,
        "n_train": N_tr,
        "n_test": N_te,
        "n_features": d,
        "y_mean": y_mean,
        "y_std": y_std,
        "models": {},
        "als_cycle_progression": [],
    }

    X_tr_dev = X_tr.to(device)
    y_tr_dev = y_tr.to(device)
    X_te_dev = X_te.to(device)
    y_te_dev = y_te.to(device)

    # 1. Greedy DSN Fit (Baseline from EXP-09)
    print("\n[1/5] Fitting Greedy Deep Solvable Cascade (DSN Stage 4)...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = DeepSolvableCascade(n_stages=n_stages, d_latent=8, shrinkage=0.5, lambda_reg=1e-3, use_rff=False).to(device)
    model.fit(X_tr_dev, y_tr_dev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    greedy_fit_time = time.perf_counter() - t0

    pred_tr_greedy = model.predict(X_tr_dev)
    pred_te_greedy = model.predict(X_te_dev)
    r2_tr_greedy, rmse_tr_greedy = eval_metrics(y_tr_dev, pred_tr_greedy, y_std)
    r2_te_greedy, rmse_te_greedy = eval_metrics(y_te_dev, pred_te_greedy, y_std)

    dataset_results["models"]["Greedy DSN (EXP-09)"] = {
        "train_r2": round(r2_tr_greedy, 5),
        "test_r2": round(r2_te_greedy, 5),
        "train_rmse": round(rmse_tr_greedy, 5),
        "test_rmse": round(rmse_te_greedy, 5),
        "fit_time_sec": round(greedy_fit_time, 4),
        "fit_time_ms": round(greedy_fit_time * 1000.0, 2),
        "approach": "Greedy Forward Boosting",
    }
    print(f"  -> Greedy DSN: Test R2 = {r2_te_greedy:.4f}, Train R2 = {r2_tr_greedy:.4f} (Fit = {greedy_fit_time*1000:.1f} ms)")

    # 2. Alternating Least Squares (ALS) Refinement
    print(f"\n[2/5] Running Blueprint 3: Alternating Least Squares ({n_als_cycles} Cycles)...")
    # Cache feature representations for all stages
    Phis_tr = []
    Phis_te = []
    prev_Z_tr = None
    prev_Z_te = None

    for stage, s in enumerate(model.stages):
        if stage == 0:
            U_tr = X_tr_dev
            U_te = X_te_dev
        else:
            U_tr = torch.cat([X_tr_dev, prev_Z_tr], dim=-1)
            U_te = torch.cat([X_te_dev, prev_Z_te], dim=-1)

        Phi_tr = compute_poly2_features(U_tr, model.clip_val)
        Phi_te = compute_poly2_features(U_te, model.clip_val)
        Phis_tr.append(Phi_tr)
        Phis_te.append(Phi_te)

        # Update Z
        Phi_tr_centered = Phi_tr - s["Phi_mean"]
        Z_pca_raw = torch.matmul(Phi_tr_centered, s["top_vecs"])
        Z_pca = torch.tanh(Z_pca_raw / s["pca_scale"])
        pred_stage = torch.matmul(Phi_tr, s["W"])
        pred_norm = torch.tanh(pred_stage / s["pred_scale"])
        prev_Z_tr = torch.cat([pred_norm, Z_pca], dim=-1)

        Phi_te_centered = Phi_te - s["Phi_mean"]
        Z_pca_raw_te = torch.matmul(Phi_te_centered, s["top_vecs"])
        Z_pca_te = torch.tanh(Z_pca_raw_te / s["pca_scale"])
        pred_stage_te = torch.matmul(Phi_te, s["W"])
        pred_norm_te = torch.tanh(pred_stage_te / s["pred_scale"])
        prev_Z_te = torch.cat([pred_norm_te, Z_pca_te], dim=-1)

    # Precompute Gram inverses
    inv_Grams = []
    for l in range(n_stages):
        Phi = Phis_tr[l]
        D = Phi.shape[1]
        G = torch.matmul(Phi.T, Phi) + (model.lambda_reg * N_tr) * torch.eye(D, device=device)
        inv_Grams.append(torch.linalg.inv(G))

    W_stages = [s["W"].clone() for s in model.stages]
    shrinkages = [s["step_shrink"] for s in model.stages]

    # Initial cycle 0 state
    dataset_results["als_cycle_progression"].append({
        "cycle": 0,
        "train_r2": round(r2_tr_greedy, 5),
        "test_r2": round(r2_te_greedy, 5),
        "test_rmse": round(rmse_te_greedy, 5),
        "cumulative_time_ms": round(greedy_fit_time * 1000.0, 2),
    })

    t0 = time.perf_counter()
    for cycle in range(1, n_als_cycles + 1):
        for l in range(n_stages):
            pred_others = torch.zeros(N_tr, 1, device=device)
            for m in range(n_stages):
                if m != l:
                    pred_others = pred_others + shrinkages[m] * torch.matmul(Phis_tr[m], W_stages[m])

            target_residual = y_tr_dev - pred_others
            rhs = torch.matmul(Phis_tr[l].T, target_residual / shrinkages[l])
            W_stages[l] = torch.matmul(inv_Grams[l], rhs)

        # Evaluate cycle performance
        total_pred_tr = sum(shrinkages[m] * torch.matmul(Phis_tr[m], W_stages[m]) for m in range(n_stages))
        total_pred_te = sum(shrinkages[m] * torch.matmul(Phis_te[m], W_stages[m]) for m in range(n_stages))
        r2_tr_c, rmse_tr_c = eval_metrics(y_tr_dev, total_pred_tr, y_std)
        r2_te_c, rmse_te_c = eval_metrics(y_te_dev, total_pred_te, y_std)
        cycle_cum_ms = (greedy_fit_time + (time.perf_counter() - t0)) * 1000.0

        dataset_results["als_cycle_progression"].append({
            "cycle": cycle,
            "train_r2": round(r2_tr_c, 5),
            "test_r2": round(r2_te_c, 5),
            "test_rmse": round(rmse_te_c, 5),
            "cumulative_time_ms": round(cycle_cum_ms, 2),
        })
        print(f"  -> ALS Cycle {cycle}: Test R2 = {r2_te_c:.4f}, Train R2 = {r2_tr_c:.4f} (Delta: {r2_te_c - r2_te_greedy:+.4f})")

    if device.type == "cuda":
        torch.cuda.synchronize()
    als_refine_time = time.perf_counter() - t0
    total_als_time = greedy_fit_time + als_refine_time

    dataset_results["models"]["ALS-Refined DSN (5 Cycles)"] = {
        "train_r2": round(r2_tr_c, 5),
        "test_r2": round(r2_te_c, 5),
        "train_rmse": round(rmse_tr_c, 5),
        "test_rmse": round(rmse_te_c, 5),
        "fit_time_sec": round(total_als_time, 4),
        "fit_time_ms": round(total_als_time * 1000.0, 2),
        "refine_only_ms": round(als_refine_time * 1000.0, 2),
        "net_gain_test_r2": round(r2_te_c - r2_te_greedy, 5),
        "approach": "Bi-Convex Coordinate Solves (Zero Backprop)",
    }

    # 3. XGBoost Baseline
    print("\n[3/5] Evaluating XGBoost Baseline (100 Trees)...")
    X_tr_np = X_tr.numpy()
    y_tr_np = y_tr.numpy().squeeze()
    X_te_np = X_te.numpy()
    y_te_np = y_te.numpy().squeeze()

    t0 = time.perf_counter()
    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_tr_np, y_tr_np)
    fit_time_xgb = time.perf_counter() - t0

    pred_te_xgb = xgb_reg.predict(X_te_np)
    pred_tr_xgb = xgb_reg.predict(X_tr_np)
    r2_tr_xgb, rmse_tr_xgb = eval_metrics(torch.tensor(y_tr_np), torch.tensor(pred_tr_xgb), y_std)
    r2_te_xgb, rmse_te_xgb = eval_metrics(torch.tensor(y_te_np), torch.tensor(pred_te_xgb), y_std)

    dataset_results["models"]["XGBoost (100 Trees)"] = {
        "train_r2": round(r2_tr_xgb, 5),
        "test_r2": round(r2_te_xgb, 5),
        "train_rmse": round(rmse_tr_xgb, 5),
        "test_rmse": round(rmse_te_xgb, 5),
        "fit_time_sec": round(fit_time_xgb, 4),
        "fit_time_ms": round(fit_time_xgb * 1000.0, 2),
        "approach": "Greedy Histogram Tree Splitting",
    }
    print(f"  -> XGBoost: Test R2 = {r2_te_xgb:.4f}, Fit time = {fit_time_xgb*1000:.2f} ms")

    # 4. LightGBM Baseline
    print("\n[4/5] Evaluating LightGBM Baseline (100 Trees)...")
    t0 = time.perf_counter()
    lgb_reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    lgb_reg.fit(X_tr_np, y_tr_np)
    fit_time_lgb = time.perf_counter() - t0

    pred_te_lgb = lgb_reg.predict(X_te_np)
    pred_tr_lgb = lgb_reg.predict(X_tr_np)
    r2_tr_lgb, rmse_tr_lgb = eval_metrics(torch.tensor(y_tr_np), torch.tensor(pred_tr_lgb), y_std)
    r2_te_lgb, rmse_te_lgb = eval_metrics(torch.tensor(y_te_np), torch.tensor(pred_te_lgb), y_std)

    dataset_results["models"]["LightGBM (100 Trees)"] = {
        "train_r2": round(r2_tr_lgb, 5),
        "test_r2": round(r2_te_lgb, 5),
        "train_rmse": round(rmse_tr_lgb, 5),
        "test_rmse": round(rmse_te_lgb, 5),
        "fit_time_sec": round(fit_time_lgb, 4),
        "fit_time_ms": round(fit_time_lgb * 1000.0, 2),
        "approach": "Leaf-wise Tree Splitting",
    }
    print(f"  -> LightGBM: Test R2 = {r2_te_lgb:.4f}, Fit time = {fit_time_lgb*1000:.2f} ms")

    # 5. FT-Transformer Baseline
    print("\n[5/5] Evaluating FT-Transformer (35 Epochs AdamW)...")
    ft_res = train_ft_transformer_with_history(X_tr, y_tr, X_te, y_te, y_std, device=device, n_epochs=35, batch_size=256)
    dataset_results["models"]["FT-Transformer (AdamW)"] = ft_res
    print(f"  -> FT-Transformer: Test R2 = {ft_res['test_r2']:.4f}, Fit time = {ft_res['fit_time_sec']:.2f} s")

    return dataset_results


def run_als_suite(device: torch.device = None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting EXP-10 Alternating Least Squares (ALS) Suite on Device: {device}")

    # CUDA context, cuSOLVER / cuBLAS, and cascade allocator warmup
    if device.type == "cuda":
        dummy_x = torch.randn(64, 8, device=device)
        dummy_y = torch.randn(64, 1, device=device)
        warm_model = DeepSolvableCascade(n_stages=4).to(device)
        warm_model.fit(dummy_x, dummy_y)
        torch.cuda.synchronize()
        print("  -> CUDA context, cuSOLVER/cuBLAS & cascade memory pools fully warmed up.")

    all_results = {}
    for ds_name in ["kin8nm", "friedman1", "california"]:
        all_results[ds_name] = run_benchmark_for_dataset(ds_name, device=device, n_stages=4, n_als_cycles=5)

    return all_results


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run_als_suite(device=dev)
    os.makedirs("results/exp10_als_solvable_refinement", exist_ok=True)
    out_file = "results/exp10_als_solvable_refinement/metrics.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {out_file}!")
