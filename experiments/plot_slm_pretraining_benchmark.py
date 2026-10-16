"""
Plotting script for EXP-12: Small Language Model Pretraining Dynamics.
Generates Figures 30, 31, and 32 in results/exp12_slm_early_pretraining/.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_slm_results(metrics_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(metrics_path, "r") as f:
        data = json.load(f)

    sm = data["Softmax SLM (Llama-Nano)"]
    ols = data["Flash-OLS SLM (FlashOLS-Nano)"]

    sm_trace = sm["trace"]
    ols_trace = ols["trace"]

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
    # Figure 30: Training & Validation Loss Trajectory
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Validation steps & loss
    sm_val_steps = [e["step"] for e in sm_trace]
    sm_val_loss = [e["val_loss"] for e in sm_trace]
    ols_val_steps = [e["step"] for e in ols_trace]
    ols_val_loss = [e["val_loss"] for e in ols_trace]

    ax1.plot(sm_val_steps, sm_val_loss, "o-", color="#e74c3c", linewidth=2.2, label="Softmax SLM (Llama-Nano 12.3M)", markersize=6)
    ax1.plot(ols_val_steps, ols_val_loss, "s-", color="#2ecc71", linewidth=2.2, label="Flash-OLS SLM (FlashOLS-Nano 12.3M)", markersize=6)
    ax1.set_title(r"Validation Loss ($\mathcal{L}_{\mathrm{val}}$) vs. Optimization Steps")
    ax1.set_xlabel("Optimization Steps")
    ax1.set_ylabel("Cross-Entropy Loss (Nats)")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Training Loss if recorded
    sm_tr_steps = [e["step"] for e in sm_trace if "train_loss" in e and e["train_loss"] is not None]
    sm_tr_loss = [e["train_loss"] for e in sm_trace if "train_loss" in e and e["train_loss"] is not None]
    ols_tr_steps = [e["step"] for e in ols_trace if "train_loss" in e and e["train_loss"] is not None]
    ols_tr_loss = [e["train_loss"] for e in ols_trace if "train_loss" in e and e["train_loss"] is not None]

    if sm_tr_steps and ols_tr_steps:
        ax2.plot(sm_tr_steps, sm_tr_loss, "o--", color="#e74c3c", alpha=0.8, linewidth=1.8, label="Softmax Train Loss")
        ax2.plot(ols_tr_steps, ols_tr_loss, "s--", color="#2ecc71", alpha=0.8, linewidth=1.8, label="Flash-OLS Train Loss")
        ax2.set_title(r"Training Loss ($\mathcal{L}_{\mathrm{train}}$) vs. Optimization Steps")
        ax2.set_xlabel("Optimization Steps")
        ax2.set_ylabel("Cross-Entropy Loss (Nats)")
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig30_path = os.path.join(out_dir, "fig30_slm_pretraining_loss_curves.png")
    plt.savefig(fig30_path)
    plt.close()
    print(f"[OK] Saved {fig30_path}")

    # -------------------------------------------------------------
    # Figure 31: Validation Perplexity Trajectory
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    sm_ppl = [e["val_ppl"] for e in sm_trace]
    ols_ppl = [e["val_ppl"] for e in ols_trace]

    ax.plot(sm_val_steps, sm_ppl, "o-", color="#e74c3c", linewidth=2.5, label="Softmax SLM (12.3M Baseline)", markersize=7)
    ax.plot(ols_val_steps, ols_ppl, "s-", color="#2ecc71", linewidth=2.5, label="Flash-OLS SLM (12.3M Active Subspace)", markersize=7)

    ax.set_yscale("log")
    ax.set_title(r"Validation Perplexity ($\mathrm{PPL}_{\mathrm{val}}$) Trajectory" + "\nFrom-Scratch Pretraining on TinyStories", pad=12)
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Perplexity (Log Scale)")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Annotate end perplexities
    ax.annotate(f"Softmax: {sm_ppl[-1]:.1f}", (sm_val_steps[-1], sm_ppl[-1]), textcoords="offset points", xytext=(-10, 10), ha="right", fontweight="bold", color="#e74c3c")
    ax.annotate(f"Flash-OLS: {ols_ppl[-1]:.1f}", (ols_val_steps[-1], ols_ppl[-1]), textcoords="offset points", xytext=(-10, -15), ha="right", fontweight="bold", color="#2ecc71")

    plt.tight_layout()
    fig31_path = os.path.join(out_dir, "fig31_slm_validation_perplexity.png")
    plt.savefig(fig31_path)
    plt.close()
    print(f"[OK] Saved {fig31_path}")

    # -------------------------------------------------------------
    # Figure 32: Optimization Dynamics & Step Throughput
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Gradient Norms
    sm_gn = [e.get("grad_norm", 0.0) for e in sm_trace if e.get("grad_norm") is not None]
    ols_gn = [e.get("grad_norm", 0.0) for e in ols_trace if e.get("grad_norm") is not None]
    gn_steps = [e["step"] for e in sm_trace if e.get("grad_norm") is not None]

    if sm_gn and ols_gn:
        ax1.plot(gn_steps, sm_gn, "o-", color="#e74c3c", label=r"Softmax $\|g\|_2$")
        ax1.plot(gn_steps, ols_gn, "s-", color="#2ecc71", label=r"Flash-OLS $\|g\|_2$")
        ax1.set_title(r"Gradient Norm ($\|\nabla \theta\|_2$) Stability")
        ax1.set_xlabel("Optimization Steps")
        ax1.set_ylabel("L2 Gradient Norm")
        ax1.legend(frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)

    # Throughput bar comparison
    models = ["Softmax SLM\n(Llama-Nano)", "Flash-OLS SLM\n(FlashOLS-Nano)"]
    tok_per_sec = [sm["tokens_per_sec"], ols["tokens_per_sec"]]
    bars = ax2.bar(models, tok_per_sec, color=["#e74c3c", "#2ecc71"], width=0.45, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:,.0f} tok/s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Pretraining Throughput (Tokens / Sec)")
    ax2.set_title("Hardware Pretraining Throughput")
    ax2.set_ylim(0, max(tok_per_sec) * 1.25)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig32_path = os.path.join(out_dir, "fig32_slm_gradient_norm_and_throughput.png")
    plt.savefig(fig32_path)
    plt.close()
    print(f"[OK] Saved {fig32_path}")


if __name__ == "__main__":
    plot_slm_results("results/exp12_slm_early_pretraining/metrics.json", "results/exp12_slm_early_pretraining")
