"""
EXP-11: TabPFN & TabICL Hybridization for Discontinuous Tabular Benchmarks.
Evaluates closed-form solvable models with Quantile Preconditioning (TabPFN),
Piecewise Linear Splines (PLE), and Landmark In-Context Kernel Anchors (TabICL)
against XGBoost, LightGBM, and baseline Deep Solvable Cascades.
"""

import json
import os
import time
import numpy as np
import torch
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data_loader import load_benchmark_dataset
from src.models import DeepSolvableCascade, TabularSplineAnchorSolvableModel


def eval_r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def run_benchmark():
    print("=" * 80)
    print("EXP-11: TabPFN & TabICL Hybridization Benchmark on Tabular Data")
    print("=" * 80)

    results = {}
    output_dir = "results/exp11_tabpfn_tabicl_solvable"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. CALIFORNIA HOUSING (Canonical Discontinuous Spatial Data)
    # ---------------------------------------------------------
    print("\n--- 1. California Housing Benchmark ---")
    cal = fetch_california_housing()
    X_raw, y_raw = cal.data, cal.target.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    y_mean, y_std = float(np.mean(y_tr)), float(np.std(y_tr))
    y_tr_norm = (y_tr - y_mean) / y_std
    y_te_norm = (y_te - y_mean) / y_std

    y_tr_t = torch.tensor(y_tr_norm, dtype=torch.float32)
    y_te_t = torch.tensor(y_te_norm, dtype=torch.float32)

    cal_results = {}

    # 1.1 Linear Ridge Baseline
    print("Evaluating Linear Ridge...")
    t0 = time.perf_counter()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr_norm.ravel())
    t_ridge = (time.perf_counter() - t0) * 1000
    p_tr_ridge = ridge.predict(X_tr)
    p_te_ridge = ridge.predict(X_te)
    cal_results["Linear Ridge"] = {
        "train_r2": eval_r2(y_tr_norm, p_tr_ridge),
        "test_r2": eval_r2(y_te_norm, p_te_ridge),
        "fit_time_ms": t_ridge,
    }
    print(f"  Linear Ridge: Test R2 = {cal_results['Linear Ridge']['test_r2']:.4f} ({t_ridge:.1f} ms)")

    # 1.2 XGBoost
    print("Evaluating XGBoost...")
    t0 = time.perf_counter()
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_tr, y_tr_norm.ravel())
    t_xgb = (time.perf_counter() - t0) * 1000
    p_tr_xgb = xgb.predict(X_tr)
    p_te_xgb = xgb.predict(X_te)
    cal_results["XGBoost"] = {
        "train_r2": eval_r2(y_tr_norm, p_tr_xgb),
        "test_r2": eval_r2(y_te_norm, p_te_xgb),
        "fit_time_ms": t_xgb,
    }
    print(f"  XGBoost: Test R2 = {cal_results['XGBoost']['test_r2']:.4f} ({t_xgb:.1f} ms)")

    # 1.3 LightGBM
    print("Evaluating LightGBM...")
    t0 = time.perf_counter()
    lgb = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_tr, y_tr_norm.ravel())
    t_lgb = (time.perf_counter() - t0) * 1000
    p_tr_lgb = lgb.predict(X_tr)
    p_te_lgb = lgb.predict(X_te)
    cal_results["LightGBM"] = {
        "train_r2": eval_r2(y_tr_norm, p_tr_lgb),
        "test_r2": eval_r2(y_te_norm, p_te_lgb),
        "fit_time_ms": t_lgb,
    }
    print(f"  LightGBM: Test R2 = {cal_results['LightGBM']['test_r2']:.4f} ({t_lgb:.1f} ms)")

    # 1.4 Deep Solvable Cascade (EXP-10 baseline)
    print("Evaluating Deep Solvable Cascade (Standard DSN + ALS from EXP-10)...")
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler()
    X_tr_s = torch.tensor(x_scaler.fit_transform(X_tr), dtype=torch.float32)
    X_te_s = torch.tensor(x_scaler.transform(X_te), dtype=torch.float32)
    t0 = time.perf_counter()
    dsn = DeepSolvableCascade(n_stages=4, lambda_reg=1e-3, shrinkage=0.5, d_latent=8, use_rff=False)
    dsn.fit_with_als(X_tr_s, y_tr_t, n_als_cycles=5)
    t_dsn = (time.perf_counter() - t0) * 1000
    p_tr_dsn = dsn.predict(X_tr_s).numpy()
    p_te_dsn = dsn.predict(X_te_s).numpy()
    cal_results["Standard DSN + ALS (EXP-10)"] = {
        "train_r2": eval_r2(y_tr_norm, p_tr_dsn),
        "test_r2": eval_r2(y_te_norm, p_te_dsn),
        "fit_time_ms": t_dsn,
    }
    print(f"  Standard DSN + ALS: Test R2 = {cal_results['Standard DSN + ALS (EXP-10)']['test_r2']:.4f} ({t_dsn:.1f} ms)")

    # 1.5 TabularSplineAnchorSolvableModel (Stage 1 Single Closed-Form Solve)
    print("Evaluating TabPFN + TabICL Solvable Block (Stage 1 Closed-Form Solve)...")
    model_st1 = TabularSplineAnchorSolvableModel(
        n_stages=1,
        n_knots=8,
        n_anchors=96,
        anchor_feature_indices=[6, 7],
        anchor_sigmas=[0.10, 0.25],
        interaction_feature_indices=[0, 1],
        lambda_reg=1e-3,
        n_quantiles=2000,
        random_state=42,
    )
    t0 = time.perf_counter()
    model_st1.fit(X_tr, y_tr_t)
    t_st1 = (time.perf_counter() - t0) * 1000
    p_tr_st1 = model_st1.predict(X_tr).numpy()
    p_te_st1 = model_st1.predict(X_te).numpy()
    cal_results["TabPFN+TabICL Solvable (Stage 1)"] = {
        "train_r2": eval_r2(y_tr_norm, p_tr_st1),
        "test_r2": eval_r2(y_te_norm, p_te_st1),
        "fit_time_ms": t_st1,
        "n_features": model_st1.stages[0]["D"],
    }
    print(f"  TabPFN+TabICL Stage 1: Test R2 = {cal_results['TabPFN+TabICL Solvable (Stage 1)']['test_r2']:.4f} ({t_st1:.1f} ms)")

    # 1.6 TabularSplineAnchorSolvableModel (Full 5-Stage Cascade + ALS)
    print("Evaluating TabPFN + TabICL Full 5-Stage Cascade + ALS Refinement...")
    model_full = TabularSplineAnchorSolvableModel(
        n_stages=5,
        n_knots=8,
        n_anchors=96,
        anchor_feature_indices=[6, 7],
        anchor_sigmas=[0.10, 0.25],
        interaction_feature_indices=[0, 1],
        lambda_reg=1e-3,
        shrinkage=0.5,
        use_als=True,
        n_als_cycles=5,
        n_quantiles=2000,
        random_state=42,
    )
    t0 = time.perf_counter()
    model_full.fit(X_tr, y_tr_t)
    t_full = (time.perf_counter() - t0) * 1000
    p_tr_full = model_full.predict(X_tr).numpy()
    p_te_full = model_full.predict(X_te).numpy()
    cal_results["TabPFN+TabICL Full Cascade + ALS"] = {
        "train_r2": eval_r2(y_tr_norm, p_tr_full),
        "test_r2": eval_r2(y_te_norm, p_te_full),
        "fit_time_ms": t_full,
        "stages": len(model_full.stages),
    }
    print(f"  TabPFN+TabICL Full Cascade + ALS: Test R2 = {cal_results['TabPFN+TabICL Full Cascade + ALS']['test_r2']:.4f} ({t_full:.1f} ms)")
    results["california_housing"] = cal_results

    # ---------------------------------------------------------
    # 2. KIN8NM (Continuous Robotic Forward Kinematics)
    # ---------------------------------------------------------
    print("\n--- 2. Kin8nm Benchmark ---")
    X_tr_k_s, y_tr_k, X_te_k_s, y_te_k, y_mean_k, y_std_k = load_benchmark_dataset("kin8nm")
    X_tr_k = X_tr_k_s.numpy()
    X_te_k = X_te_k_s.numpy()
    y_tr_k_np = y_tr_k.numpy()
    y_te_k_np = y_te_k.numpy()

    kin_results = {}

    # XGBoost
    t0 = time.perf_counter()
    xgb_k = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_k.fit(X_tr_k, y_tr_k_np.ravel())
    t_xgb_k = (time.perf_counter() - t0) * 1000
    kin_results["XGBoost"] = {
        "train_r2": eval_r2(y_tr_k_np, xgb_k.predict(X_tr_k)),
        "test_r2": eval_r2(y_te_k_np, xgb_k.predict(X_te_k)),
        "fit_time_ms": t_xgb_k,
    }

    # LightGBM
    t0 = time.perf_counter()
    lgb_k = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_k.fit(X_tr_k, y_tr_k_np.ravel())
    t_lgb_k = (time.perf_counter() - t0) * 1000
    kin_results["LightGBM"] = {
        "train_r2": eval_r2(y_tr_k_np, lgb_k.predict(X_tr_k)),
        "test_r2": eval_r2(y_te_k_np, lgb_k.predict(X_te_k)),
        "fit_time_ms": t_lgb_k,
    }

    # Deep Solvable Cascade + ALS (EXP-10)
    t0 = time.perf_counter()
    dsn_k = DeepSolvableCascade(n_stages=4, lambda_reg=1e-3, shrinkage=0.5, d_latent=8, use_rff=False)
    dsn_k.fit_with_als(X_tr_k_s, y_tr_k, n_als_cycles=5)
    t_dsn_k = (time.perf_counter() - t0) * 1000
    kin_results["Standard DSN + ALS (EXP-10)"] = {
        "train_r2": eval_r2(y_tr_k_np, dsn_k.predict(X_tr_k_s).numpy()),
        "test_r2": eval_r2(y_te_k_np, dsn_k.predict(X_te_k_s).numpy()),
        "fit_time_ms": t_dsn_k,
    }

    # TabPFN+TabICL Model on Kin8nm (General Prototype Anchors)
    t0 = time.perf_counter()
    model_kin = TabularSplineAnchorSolvableModel(
        n_stages=4,
        n_knots=4,
        n_anchors=48,
        anchor_feature_indices=None, # full archetype clustering
        anchor_sigmas=[0.25, 0.50],
        interaction_feature_indices=[0, 1, 2],
        lambda_reg=1e-3,
        shrinkage=0.5,
        use_als=True,
        n_als_cycles=5,
        n_quantiles=1000,
        random_state=42,
    )
    model_kin.fit(X_tr_k, y_tr_k)
    t_kin = (time.perf_counter() - t0) * 1000
    kin_results["TabPFN+TabICL Full Cascade + ALS"] = {
        "train_r2": eval_r2(y_tr_k_np, model_kin.predict(X_tr_k).numpy()),
        "test_r2": eval_r2(y_te_k_np, model_kin.predict(X_te_k).numpy()),
        "fit_time_ms": t_kin,
    }
    print(f"  Kin8nm XGBoost: Test R2 = {kin_results['XGBoost']['test_r2']:.4f} ({t_xgb_k:.1f} ms)")
    print(f"  Kin8nm LightGBM: Test R2 = {kin_results['LightGBM']['test_r2']:.4f} ({t_lgb_k:.1f} ms)")
    print(f"  Kin8nm Standard DSN + ALS: Test R2 = {kin_results['Standard DSN + ALS (EXP-10)']['test_r2']:.4f} ({t_dsn_k:.1f} ms)")
    print(f"  Kin8nm TabPFN+TabICL: Test R2 = {kin_results['TabPFN+TabICL Full Cascade + ALS']['test_r2']:.4f} ({t_kin:.1f} ms)")
    results["kin8nm"] = kin_results

    # ---------------------------------------------------------
    # 3. FRIEDMAN-1 (Non-linear trigonometric + interactions)
    # ---------------------------------------------------------
    print("\n--- 3. Friedman-1 Benchmark ---")
    X_tr_f_s, y_tr_f, X_te_f_s, y_te_f, y_mean_f, y_std_f = load_benchmark_dataset("friedman1")
    X_tr_f = X_tr_f_s.numpy()
    X_te_f = X_te_f_s.numpy()
    y_tr_f_np = y_tr_f.numpy()
    y_te_f_np = y_te_f.numpy()

    fried_results = {}

    # XGBoost
    t0 = time.perf_counter()
    xgb_f = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_f.fit(X_tr_f, y_tr_f_np.ravel())
    t_xgb_f = (time.perf_counter() - t0) * 1000
    fried_results["XGBoost"] = {
        "train_r2": eval_r2(y_tr_f_np, xgb_f.predict(X_tr_f)),
        "test_r2": eval_r2(y_te_f_np, xgb_f.predict(X_te_f)),
        "fit_time_ms": t_xgb_f,
    }

    # LightGBM
    t0 = time.perf_counter()
    lgb_f = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_f.fit(X_tr_f, y_tr_f_np.ravel())
    t_lgb_f = (time.perf_counter() - t0) * 1000
    fried_results["LightGBM"] = {
        "train_r2": eval_r2(y_tr_f_np, lgb_f.predict(X_tr_f)),
        "test_r2": eval_r2(y_te_f_np, lgb_f.predict(X_te_f)),
        "fit_time_ms": t_lgb_f,
    }

    # Deep Solvable Cascade + ALS (EXP-10)
    t0 = time.perf_counter()
    dsn_f = DeepSolvableCascade(n_stages=4, lambda_reg=1e-3, shrinkage=0.5, d_latent=8, use_rff=False)
    dsn_f.fit_with_als(X_tr_f_s, y_tr_f, n_als_cycles=5)
    t_dsn_f = (time.perf_counter() - t0) * 1000
    fried_results["Standard DSN + ALS (EXP-10)"] = {
        "train_r2": eval_r2(y_tr_f_np, dsn_f.predict(X_tr_f_s).numpy()),
        "test_r2": eval_r2(y_te_f_np, dsn_f.predict(X_te_f_s).numpy()),
        "fit_time_ms": t_dsn_f,
    }

    # TabPFN+TabICL Model on Friedman-1
    t0 = time.perf_counter()
    model_fried = TabularSplineAnchorSolvableModel(
        n_stages=4,
        n_knots=6,
        n_anchors=64,
        anchor_feature_indices=[0, 1, 2],
        anchor_sigmas=[0.20, 0.40],
        interaction_feature_indices=[0, 1],
        lambda_reg=1e-3,
        shrinkage=0.5,
        use_als=True,
        n_als_cycles=5,
        n_quantiles=1000,
        random_state=42,
    )
    model_fried.fit(X_tr_f, y_tr_f)
    t_fried = (time.perf_counter() - t0) * 1000
    fried_results["TabPFN+TabICL Full Cascade + ALS"] = {
        "train_r2": eval_r2(y_tr_f_np, model_fried.predict(X_tr_f).numpy()),
        "test_r2": eval_r2(y_te_f_np, model_fried.predict(X_te_f).numpy()),
        "fit_time_ms": t_fried,
    }
    print(f"  Friedman-1 XGBoost: Test R2 = {fried_results['XGBoost']['test_r2']:.4f} ({t_xgb_f:.1f} ms)")
    print(f"  Friedman-1 LightGBM: Test R2 = {fried_results['LightGBM']['test_r2']:.4f} ({t_lgb_f:.1f} ms)")
    print(f"  Friedman-1 Standard DSN + ALS: Test R2 = {fried_results['Standard DSN + ALS (EXP-10)']['test_r2']:.4f} ({t_dsn_f:.1f} ms)")
    print(f"  Friedman-1 TabPFN+TabICL: Test R2 = {fried_results['TabPFN+TabICL Full Cascade + ALS']['test_r2']:.4f} ({t_fried:.1f} ms)")
    results["friedman1"] = fried_results

    # Save to metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Metrics saved successfully to {metrics_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
