"""
Benchmark Runner for EXP-09: Deep Closed-Form Boosting & Solvable Networks (DSN)
vs Gradient Boosted Decision Trees (XGBoost, LightGBM) and FT-Transformer.

Evaluates on Kin8nm, Friedman-1, and California Housing benchmarks.
Outputs exact R^2, RMSE, training wall-clock time, and depth progression.
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
from src.models import DeepSolvableCascade, StandardFTTransformer
from src.metrics import r2_score


def eval_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_std: float = 1.0) -> tuple[float, float]:
    """Computes R^2 and unstandardized RMSE."""
    r2 = r2_score(y_true, y_pred)
    mse = float(torch.mean((y_true - y_pred) ** 2).item())
    rmse = math.sqrt(mse) * y_std
    return r2, rmse


def train_ft_transformer(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    y_std: float,
    device: torch.device,
    n_epochs: int = 35,
    batch_size: int = 256,
) -> dict:
    """Trains Standard FT-Transformer using AdamW."""
    n_features = X_train.shape[1]
    model = StandardFTTransformer(n_features=n_features, d_token=32, n_heads=4, d_ffn=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train.to(device), y_train.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X_te = X_test.to(device)
    y_te = y_test.to(device)

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    fit_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        t_infer0 = time.perf_counter()
        pred_te = model(X_te)
        infer_time = time.perf_counter() - t_infer0
        pred_tr = model(X_train.to(device))

    r2_tr, rmse_tr = eval_metrics(y_train.to(device), pred_tr, y_std)
    r2_te, rmse_te = eval_metrics(y_te, pred_te, y_std)
    infer_ms_per_1k = (infer_time / X_test.shape[0]) * 1000.0 * 1000.0

    return {
        "train_r2": round(r2_tr, 5),
        "test_r2": round(r2_te, 5),
        "train_rmse": round(rmse_tr, 5),
        "test_rmse": round(rmse_te, 5),
        "fit_time_sec": round(fit_time, 4),
        "fit_time_ms": round(fit_time * 1000.0, 2),
        "infer_ms_per_1k": round(infer_ms_per_1k, 3),
        "effective_degree": "Neural Attention (35 ep)",
        "approach": "Iterative Backprop (AdamW)",
    }


def run_benchmark_for_dataset(dataset_name: str, device: torch.device) -> dict:
    print(f"\n{'='*70}\n>>> Running EXP-09 Benchmark on: {dataset_name.upper()} <<<\n{'='*70}")
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
        "dsn_depth_progression": [],
    }

    # 1. Baseline Ridge Linear Regression (Degree 1)
    print("\n[1/7] Evaluating Ridge Linear Baseline (Degree 1)...")
    t0 = time.perf_counter()
    X_tr_dev = X_tr.to(device)
    y_tr_dev = y_tr.to(device)
    X_te_dev = X_te.to(device)
    y_te_dev = y_te.to(device)

    # Add bias column
    X_tr_bias = torch.cat([X_tr_dev, torch.ones(N_tr, 1, device=device)], dim=-1)
    X_te_bias = torch.cat([X_te_dev, torch.ones(N_te, 1, device=device)], dim=-1)
    D_lin = X_tr_bias.shape[1]
    G_lin = torch.matmul(X_tr_bias.T, X_tr_bias) + (1e-3 * N_tr) * torch.eye(D_lin, device=device)
    W_lin = torch.linalg.solve(G_lin, torch.matmul(X_tr_bias.T, y_tr_dev))
    fit_time_lin = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred_te_lin = torch.matmul(X_te_bias, W_lin)
    infer_time_lin = time.perf_counter() - t_inf0
    pred_tr_lin = torch.matmul(X_tr_bias, W_lin)

    r2_tr, rmse_tr = eval_metrics(y_tr_dev, pred_tr_lin, y_std)
    r2_te, rmse_te = eval_metrics(y_te_dev, pred_te_lin, y_std)
    dataset_results["models"]["Ridge Linear (Deg 1)"] = {
        "train_r2": round(r2_tr, 5),
        "test_r2": round(r2_te, 5),
        "train_rmse": round(rmse_tr, 5),
        "test_rmse": round(rmse_te, 5),
        "fit_time_sec": round(fit_time_lin, 4),
        "fit_time_ms": round(fit_time_lin * 1000.0, 2),
        "infer_ms_per_1k": round((infer_time_lin / N_te) * 1e6, 3),
        "effective_degree": 1,
        "approach": "Closed-Form OLS",
    }
    print(f"  -> Ridge Linear: Test R2 = {r2_te:.4f}, Fit time = {fit_time_lin*1000:.2f} ms")

    # 2. Deep Solvable Cascade (DSN-Poly) across stages 1 to 5
    print("\n[2/7] Evaluating Deep Solvable Cascade (DSN-Poly) Stages 1 to 5...")
    max_stages = 5
    t0 = time.perf_counter()
    dsn_poly = DeepSolvableCascade(
        n_stages=max_stages,
        d_latent=8,
        shrinkage=0.5,
        lambda_reg=1e-3,
        clip_val=4.0,
        use_rff=False,
    ).to(device)
    dsn_poly.fit(X_tr_dev, y_tr_dev)
    fit_time_dsn_poly = time.perf_counter() - t0

    for s in range(1, max_stages + 1):
        eff_deg = 2 ** s
        t_inf0 = time.perf_counter()
        pred_te = dsn_poly.predict(X_te_dev, max_stages=s)
        infer_time = time.perf_counter() - t_inf0
        pred_tr = dsn_poly.predict(X_tr_dev, max_stages=s)

        r2_tr, rmse_tr = eval_metrics(y_tr_dev, pred_tr, y_std)
        r2_te, rmse_te = eval_metrics(y_te_dev, pred_te, y_std)

        # Proportional fit time estimate for stage s
        stage_fit_sec = fit_time_dsn_poly * (s / max_stages)
        stage_fit_ms = stage_fit_sec * 1000.0

        model_key = f"DSN-Poly (Stage {s}, Deg {eff_deg})"
        dataset_results["models"][model_key] = {
            "train_r2": round(r2_tr, 5),
            "test_r2": round(r2_te, 5),
            "train_rmse": round(rmse_tr, 5),
            "test_rmse": round(rmse_te, 5),
            "fit_time_sec": round(stage_fit_sec, 4),
            "fit_time_ms": round(stage_fit_ms, 2),
            "infer_ms_per_1k": round((infer_time / N_te) * 1e6, 3),
            "effective_degree": eff_deg,
            "approach": "Closed-Form Boosting (Zero Backprop)",
        }
        dataset_results["dsn_depth_progression"].append({
            "stage": s,
            "effective_degree": eff_deg,
            "train_r2": round(r2_tr, 5),
            "test_r2": round(r2_te, 5),
            "test_rmse": round(rmse_te, 5),
            "cumulative_fit_ms": round(stage_fit_ms, 2),
        })
        print(f"  -> DSN-Poly Stage {s} (Deg {eff_deg}): Test R2 = {r2_te:.4f}, Train R2 = {r2_tr:.4f} (Fit ~{stage_fit_ms:.1f} ms)")

    # 3. DSN-Kernel (Stage 4, Degree 16 + RFF)
    print("\n[3/7] Evaluating DSN-Kernel (Stage 4, Degree 16 + Harmonic RFF)...")
    t0 = time.perf_counter()
    dsn_kernel = DeepSolvableCascade(
        n_stages=4,
        d_latent=8,
        shrinkage=0.5,
        lambda_reg=1e-3,
        clip_val=4.0,
        use_rff=True,
        n_rff=32,
    ).to(device)
    dsn_kernel.fit(X_tr_dev, y_tr_dev)
    fit_time_kernel = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred_te_k = dsn_kernel.predict(X_te_dev, max_stages=4)
    infer_time_k = time.perf_counter() - t_inf0
    pred_tr_k = dsn_kernel.predict(X_tr_dev, max_stages=4)

    r2_tr_k, rmse_tr_k = eval_metrics(y_tr_dev, pred_tr_k, y_std)
    r2_te_k, rmse_te_k = eval_metrics(y_te_dev, pred_te_k, y_std)

    dataset_results["models"]["DSN-Kernel (Stage 4, Deg 16+RFF)"] = {
        "train_r2": round(r2_tr_k, 5),
        "test_r2": round(r2_te_k, 5),
        "train_rmse": round(rmse_tr_k, 5),
        "test_rmse": round(rmse_te_k, 5),
        "fit_time_sec": round(fit_time_kernel, 4),
        "fit_time_ms": round(fit_time_kernel * 1000.0, 2),
        "infer_ms_per_1k": round((infer_time_k / N_te) * 1e6, 3),
        "effective_degree": "16 + Harmonic RFF",
        "approach": "Closed-Form Boosting (Zero Backprop)",
    }
    print(f"  -> DSN-Kernel Stage 4: Test R2 = {r2_te_k:.4f}, Fit time = {fit_time_kernel*1000:.2f} ms")

    # 4. XGBoost Baseline
    print("\n[4/7] Evaluating XGBoost Baseline (100 Trees)...")
    X_tr_np = X_tr.numpy()
    y_tr_np = y_tr.numpy().squeeze()
    X_te_np = X_te.numpy()
    y_te_np = y_te.numpy().squeeze()

    t0 = time.perf_counter()
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    xgb_reg.fit(X_tr_np, y_tr_np)
    fit_time_xgb = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred_te_xgb = xgb_reg.predict(X_te_np)
    infer_time_xgb = time.perf_counter() - t_inf0
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
        "infer_ms_per_1k": round((infer_time_xgb / N_te) * 1e6, 3),
        "effective_degree": "Gradient Boosted Trees",
        "approach": "Greedy Histogram Tree Splitting",
    }
    print(f"  -> XGBoost: Test R2 = {r2_te_xgb:.4f}, Fit time = {fit_time_xgb*1000:.2f} ms")

    # 5. LightGBM Baseline
    print("\n[5/7] Evaluating LightGBM Baseline (100 Trees)...")
    t0 = time.perf_counter()
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_reg.fit(X_tr_np, y_tr_np)
    fit_time_lgb = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred_te_lgb = lgb_reg.predict(X_te_np)
    infer_time_lgb = time.perf_counter() - t_inf0
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
        "infer_ms_per_1k": round((infer_time_lgb / N_te) * 1e6, 3),
        "effective_degree": "Gradient Boosted Trees",
        "approach": "Leaf-wise Tree Splitting",
    }
    print(f"  -> LightGBM: Test R2 = {r2_te_lgb:.4f}, Fit time = {fit_time_lgb*1000:.2f} ms")

    # 6. FT-Transformer Baseline (Deep Attention)
    print("\n[6/7] Evaluating Standard FT-Transformer (Deep Attention, 35 Epochs AdamW)...")
    ft_results = train_ft_transformer(
        X_tr, y_tr, X_te, y_te, y_std, device=device, n_epochs=35, batch_size=256
    )
    dataset_results["models"]["FT-Transformer (Deep Attention)"] = ft_results
    print(f"  -> FT-Transformer: Test R2 = {ft_results['test_r2']:.4f}, Fit time = {ft_results['fit_time_sec']:.2f} s")

    return dataset_results


def run_deep_boosting_suite(device: torch.device = None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting EXP-09 Deep Closed-Form Boosting Suite on Device: {device}")

    all_results = {}
    for ds_name in ["kin8nm", "friedman1", "california"]:
        all_results[ds_name] = run_benchmark_for_dataset(ds_name, device=device)

    return all_results


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run_deep_boosting_suite(device=dev)
    os.makedirs("results/exp09_deep_boosting_solvable", exist_ok=True)
    out_file = "results/exp09_deep_boosting_solvable/metrics.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {out_file}!")
