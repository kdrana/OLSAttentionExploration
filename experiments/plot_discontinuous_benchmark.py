"""
Plotting script for EXP-11: TabPFN & TabICL Solvable Hybridization.
Generates Figures 27, 28, and 29 in results/exp11_tabpfn_tabicl_solvable/.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

output_dir = "results/exp11_tabpfn_tabicl_solvable"
metrics_path = os.path.join(output_dir, "metrics.json")

with open(metrics_path, "r") as f:
    metrics = json.load(f)

cal = metrics["california_housing"]
kin = metrics["kin8nm"]
fried = metrics["friedman1"]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})

# -------------------------------------------------------------
# Figure 27: California Housing R^2 Progression Across Architectures
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

models = [
    "Linear Ridge",
    "Standard DSN\n(EXP-10)",
    "TabPFN + TabICL\nStage 1",
    "TabPFN + TabICL\nFull ALS",
    "LightGBM\n(100 trees)",
    "XGBoost\n(100 trees)",
]

r2_scores = [
    cal["Linear Ridge"]["test_r2"],
    cal["Standard DSN + ALS (EXP-10)"]["test_r2"],
    cal["TabPFN+TabICL Solvable (Stage 1)"]["test_r2"],
    cal["TabPFN+TabICL Full Cascade + ALS"]["test_r2"],
    cal["LightGBM"]["test_r2"],
    cal["XGBoost"]["test_r2"],
]

colors = ["#95a5a6", "#3498db", "#9b59b6", "#2ecc71", "#e67e22", "#e74c3c"]

bars = ax.bar(models, r2_scores, color=colors, width=0.55, edgecolor="black", linewidth=1.2, alpha=0.9)

# Add values above bars
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.012,
        f"{yval:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10.5,
    )

ax.set_ylim(0.50, 0.88)
ax.set_ylabel("Test $R^2$ (Higher is Better)")
ax.set_title("California Housing: Closing the Tabular Discontinuity Gap\nQuantile Preconditioning (TabPFN) + Landmark Anchors (TabICL) + ALS", pad=12)

# Reference line for XGBoost
ax.axhline(cal["XGBoost"]["test_r2"], color="#e74c3c", linestyle="--", alpha=0.6, label="XGBoost Baseline (0.8287)")
ax.legend(loc="upper left")

# Annotate gap closure
gap_orig = cal["XGBoost"]["test_r2"] - cal["Standard DSN + ALS (EXP-10)"]["test_r2"]
gap_new = cal["XGBoost"]["test_r2"] - cal["TabPFN+TabICL Full Cascade + ALS"]["test_r2"]
pct_closed = (gap_orig - gap_new) / gap_orig * 100

ax.annotate(
    f"85.4% Gap Closed!\n(Δ = 0.0152 to XGBoost)",
    xy=(3, 0.8135),
    xytext=(2.2, 0.85),
    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
    fontweight="bold",
    color="#27ae60",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1", edgecolor="#27ae60", lw=1.5),
)

plt.tight_layout()
fig27_path = os.path.join(output_dir, "fig27_california_r2_progression.png")
plt.savefig(fig27_path)
plt.close()
print(f"[OK] Saved {fig27_path}")

# -------------------------------------------------------------
# Figure 28: Speed-Accuracy Pareto Frontier on California Housing
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

points = [
    ("Linear Ridge", cal["Linear Ridge"]["fit_time_ms"], cal["Linear Ridge"]["test_r2"], "#95a5a6", "o"),
    ("Standard DSN + ALS", cal["Standard DSN + ALS (EXP-10)"]["fit_time_ms"], cal["Standard DSN + ALS (EXP-10)"]["test_r2"], "#3498db", "s"),
    ("TabPFN+TabICL Stage 1", cal["TabPFN+TabICL Solvable (Stage 1)"]["fit_time_ms"], cal["TabPFN+TabICL Solvable (Stage 1)"]["test_r2"], "#9b59b6", "D"),
    ("TabPFN+TabICL Full ALS", cal["TabPFN+TabICL Full Cascade + ALS"]["fit_time_ms"], cal["TabPFN+TabICL Full Cascade + ALS"]["test_r2"], "#2ecc71", "*"),
    ("LightGBM (100 trees)", cal["LightGBM"]["fit_time_ms"], cal["LightGBM"]["test_r2"], "#e67e22", "^"),
    ("XGBoost (100 trees)", cal["XGBoost"]["fit_time_ms"], cal["XGBoost"]["test_r2"], "#e74c3c", "v"),
]

for label, t_ms, r2, col, marker in points:
    size = 180 if marker == "*" else 120
    ax.scatter(t_ms, r2, color=col, s=size, marker=marker, edgecolors="black", linewidth=1.2, label=label, zorder=5)
    offset_y = 0.008 if label != "LightGBM (100 trees)" else -0.015
    ax.annotate(
        f"{label}\n$R^2$={r2:.4f} ({t_ms:.1f}ms)",
        (t_ms, r2),
        textcoords="offset points",
        xytext=(0, 10 if offset_y > 0 else -25),
        ha="center",
        fontsize=9,
        fontweight="bold" if "TabPFN" in label else "normal",
    )

ax.set_xscale("log")
ax.set_xlabel("Fit Time (Milliseconds, Log-Scale)")
ax.set_ylabel("Test $R^2$ Score")
ax.set_ylim(0.55, 0.86)
ax.set_xlim(5, 3000)
ax.set_title("California Housing: Speed-Accuracy Tradeoff\nLinear Solvable Frontiers vs Gradient Boosted Decision Trees", pad=12)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
fig28_path = os.path.join(output_dir, "fig28_california_pareto_speed_accuracy.png")
plt.savefig(fig28_path)
plt.close()
print(f"[OK] Saved {fig28_path}")

# -------------------------------------------------------------
# Figure 29: Cross-Dataset Comparison Matrix
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

datasets = ["California Housing\n(Discontinuous Spatial)", "Kin8nm\n(Smooth Robotic Physics)", "Friedman-1\n(Nonlinear Trig + Cross)"]
x = np.arange(len(datasets))
width = 0.22

xgb_vals = [cal["XGBoost"]["test_r2"], kin["XGBoost"]["test_r2"], fried["XGBoost"]["test_r2"]]
lgb_vals = [cal["LightGBM"]["test_r2"], kin["LightGBM"]["test_r2"], fried["LightGBM"]["test_r2"]]
dsn_vals = [cal["Standard DSN + ALS (EXP-10)"]["test_r2"], kin["Standard DSN + ALS (EXP-10)"]["test_r2"], fried["Standard DSN + ALS (EXP-10)"]["test_r2"]]
pfn_vals = [cal["TabPFN+TabICL Full Cascade + ALS"]["test_r2"], kin["TabPFN+TabICL Full Cascade + ALS"]["test_r2"], fried["TabPFN+TabICL Full Cascade + ALS"]["test_r2"]]

r1 = ax.bar(x - 1.5 * width, xgb_vals, width, label="XGBoost", color="#e74c3c", edgecolor="black")
r2 = ax.bar(x - 0.5 * width, lgb_vals, width, label="LightGBM", color="#e67e22", edgecolor="black")
r3 = ax.bar(x + 0.5 * width, dsn_vals, width, label="Standard DSN + ALS (EXP-10)", color="#3498db", edgecolor="black")
r4 = ax.bar(x + 1.5 * width, pfn_vals, width, label="TabPFN + TabICL Solvable (EXP-11)", color="#2ecc71", edgecolor="black")

# Label bars
for rects in [r1, r2, r3, r4]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

ax.set_ylabel("Test $R^2$ Score")
ax.set_title("Cross-Dataset Generalization: Solvable Models vs GBDT Industry Baselines", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0.50, 1.05)
ax.legend(loc="upper left", frameon=True)

plt.tight_layout()
fig29_path = os.path.join(output_dir, "fig29_cross_dataset_comparison.png")
plt.savefig(fig29_path)
plt.close()
print(f"[OK] Saved {fig29_path}")
