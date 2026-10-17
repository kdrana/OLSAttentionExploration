"""
Modal Cloud Runner for EXP-22: Native Solvable Attention Layer (NSAL) End-to-End Training & Alignment.

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300s.
Saves:
- results/exp22_native_solvable_attention/metrics.json
- results/exp22_native_solvable_attention/fig58_nsal_mqar_length_generalization.png
- results/exp22_native_solvable_attention/fig59_nsal_decode_memory_latency.png
- results/exp22_native_solvable_attention/fig60_nsal_training_dynamics_loss.png
"""

import os
import io
import json
import modal

app = modal.App("ols-attention-nsal-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "matplotlib")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_exp22_worker() -> dict:
    import torch
    from experiments.run_nsal_training import run_exp22_benchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")
    return run_exp22_benchmark(device=device)


@app.function(image=image, timeout=60)
def generate_exp22_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    seq_lens = [128, 256, 512, 1024, 2048]

    # Extract accuracy arrays
    softmax_acc = [metrics["softmax"]["accuracies"][f"acc_T{t}"] for t in seq_lens]
    linear_acc = [metrics["linear"]["accuracies"][f"acc_T{t}"] for t in seq_lens]
    nsal_acc = [metrics["nsal"]["accuracies"][f"acc_T{t}"] for t in seq_lens]

    # -------------------------------------------------------------
    # Figure 58: MQAR Length and Capacity Scaling (2 Subplots)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Context Length Scaling
    x_indices = np.arange(len(seq_lens))
    ax1.plot(x_indices, softmax_acc, marker="o", linewidth=2.5, markersize=8, color="#2b5c8f", label="Standard Softmax (O(T) KV-Cache)")
    ax1.plot(x_indices, linear_acc, marker="s", linewidth=2.5, markersize=8, color="#d95f02", linestyle="--", label="Linear Attn (Katharopoulos)")
    ax1.plot(x_indices, nsal_acc, marker="^", linewidth=3.0, markersize=10, color="#1b9e77", label="Native Solvable Attention (NSAL, O(1))")

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"T={t}" for t in seq_lens], fontsize=10, fontweight="bold")
    ax1.set_xlabel("Context Length (T) at Evaluation", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Associative Recall Accuracy (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylim(-5, 105)
    ax1.set_title("(a) Generalization vs Context Length (8 KV Pairs)", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower left")

    # Subplot 2: Capacity Scaling (Number of KV Pairs at T=512)
    pairs_list = [4, 8, 12, 16]
    softmax_pairs = [metrics["softmax"]["accuracies_pairs"][f"acc_pairs{p}"] for p in pairs_list]
    linear_pairs = [metrics["linear"]["accuracies_pairs"][f"acc_pairs{p}"] for p in pairs_list]
    nsal_pairs = [metrics["nsal"]["accuracies_pairs"][f"acc_pairs{p}"] for p in pairs_list]

    p_indices = np.arange(len(pairs_list))
    ax2.plot(p_indices, softmax_pairs, marker="o", linewidth=2.5, markersize=8, color="#2b5c8f", label="Standard Softmax")
    ax2.plot(p_indices, linear_pairs, marker="s", linewidth=2.5, markersize=8, color="#d95f02", linestyle="--", label="Standard Linear Attn")
    ax2.plot(p_indices, nsal_pairs, marker="^", linewidth=3.0, markersize=10, color="#1b9e77", label="Native Solvable Attention (NSAL)")

    ax2.set_xticks(p_indices)
    ax2.set_xticklabels([f"{p} Pairs" for p in pairs_list], fontsize=10, fontweight="bold")
    ax2.set_xlabel("Stored Memory Item Capacity (at T=512)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Associative Recall Accuracy (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylim(-5, 105)
    ax2.set_title("(b) Capacity vs Number of Concurrent Pairs", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower left")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig58_nsal_mqar_length_generalization.png"] = buf.getvalue()
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 59: Decode Memory & Step Latency Scaling
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    steps_eval = [128, 512, 1024, 2048]
    mem_softmax = [metrics["softmax"]["decode_metrics"]["memory_profile_kb"][f"mem_kb_T{t}"] for t in steps_eval]
    mem_linear = [metrics["linear"]["decode_metrics"]["memory_profile_kb"][f"mem_kb_T{t}"] for t in steps_eval]
    mem_nsal = [metrics["nsal"]["decode_metrics"]["memory_profile_kb"][f"mem_kb_T{t}"] for t in steps_eval]

    # Memory plot
    ax1.plot(steps_eval, mem_softmax, marker="o", linewidth=2.5, color="#2b5c8f", label="Softmax KV Cache (O(T))")
    ax1.plot(steps_eval, mem_linear, marker="s", linewidth=2.5, color="#d95f02", linestyle="--", label="Linear Attn State (O(1))")
    ax1.plot(steps_eval, mem_nsal, marker="^", linewidth=3.0, color="#1b9e77", label="NSAL Operator State (O(1))")
    ax1.set_xlabel("Context Tokens Generated (T)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Resident State Memory (KB)", fontsize=11, fontweight="bold")
    ax1.set_title("Decode Cache Footprint (2 Layers, H=4)", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    # Latency plot
    lat_softmax = [metrics["softmax"]["decode_metrics"]["latency_profile_ms"][f"latency_ms_T{t}"] for t in steps_eval]
    lat_linear = [metrics["linear"]["decode_metrics"]["latency_profile_ms"][f"latency_ms_T{t}"] for t in steps_eval]
    lat_nsal = [metrics["nsal"]["decode_metrics"]["latency_profile_ms"][f"latency_ms_T{t}"] for t in steps_eval]

    ax2.plot(steps_eval, lat_softmax, marker="o", linewidth=2.5, color="#2b5c8f", label="Softmax Generation")
    ax2.plot(steps_eval, lat_linear, marker="s", linewidth=2.5, color="#d95f02", linestyle="--", label="Linear Generation")
    ax2.plot(steps_eval, lat_nsal, marker="^", linewidth=3.0, color="#1b9e77", label="NSAL RLS Generation")
    ax2.set_xlabel("Context Tokens Generated (T)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Step Decode Latency (ms/token)", fontsize=11, fontweight="bold")
    ax2.set_title("Autoregressive Step Latency", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig59_nsal_decode_memory_latency.png"] = buf.getvalue()
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 60: Training Convergence Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)

    for arch, col, ls, lbl in [
        ("softmax", "#2b5c8f", "-", "Softmax Cross-Entropy"),
        ("linear", "#d95f02", "--", "Linear Attention Cross-Entropy"),
        ("nsal", "#1b9e77", "-", "NSAL End-to-End Solvable Inversion"),
    ]:
        loss_curve = metrics[arch]["loss_history"]
        steps = [i * 10 for i in range(len(loss_curve))]
        ax.plot(steps, loss_curve, linewidth=2.2, color=col, linestyle=ls, label=f"{lbl} (Final: {metrics[arch]['final_loss']:.4f})")

    ax.set_xlabel("Optimization Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss (Associative Recall)", fontsize=11, fontweight="bold")
    ax.set_title("EXP-22: End-to-End Training Loss Convergence (MQAR)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plots["fig60_nsal_training_dynamics_loss.png"] = buf.getvalue()
    plt.close(fig)

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print("  Triggering Modal Cloud Run for EXP-22: Native Solvable Attention")
    print("  Target: NVIDIA A10G Cloud GPU (timeout <= 300s)")
    print("==================================================================")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "exp22_native_solvable_attention")
    os.makedirs(out_dir, exist_ok=True)

    metrics = run_exp22_worker.remote()
    print("\nCloud benchmark finished! Generating publication figures...")
    plots = generate_exp22_plots_cloud.remote(metrics)

    # Save metrics.json
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Saved Metrics]: {metrics_path}")

    # Save plots
    for fname, pbytes in plots.items():
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            f.write(pbytes)
        print(f"[Saved Plot]: {fpath}")

    print("\nEXP-22 execution fully complete!")


if __name__ == "__main__":
    main()
