"""
Modal Cloud Runner for EXP-17: Direct Functional Operator Estimation of KV-Cache.

Executes on NVIDIA A10G Cloud GPU with strict timeout <= 300.
Serializes and downloads:
- results/exp17_direct_operator_estimation/metrics.json
- results/exp17_direct_operator_estimation/operator_adapter.pt
- Publication-quality comparison figures (fig43, fig44).
"""

import os
import io
import json
import modal

app = modal.App("ols-attention-direct-operator")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def direct_operator_worker() -> tuple[dict, bytes]:
    import torch
    from experiments.run_direct_operator_estimation import run_direct_operator_estimation_experiment

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")

    metrics, adapter_state = run_direct_operator_estimation_experiment(device=device)

    # Serialize adapter state to bytes
    buf = io.BytesIO()
    torch.save(adapter_state, buf)
    adapter_bytes = buf.getvalue()
    print(f"[Modal Cloud Worker] Serialized Direct Operator Adapter: {len(adapter_bytes) / 1024:.1f} KB")

    return metrics, adapter_bytes


@app.function(image=image, timeout=60)
def generate_operator_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    trace = metrics["trace"]
    steps = [e["step"] for e in trace]
    val_r2 = [e["val_r2"] for e in trace]
    val_cos = [e["val_cos_sim"] for e in trace]
    train_loss = [e["train_loss"] for e in trace]

    # -------------------------------------------------------------
    # Figure 43: Operator Fidelity Curves (R^2 & Cosine Similarity)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # R^2 and Cosine Similarity
    ax1.plot(steps, val_r2, "o-", color="#27ae60", linewidth=2.4, label=r"Operator $R^2$ Variance Explained")
    ax1.plot(steps, val_cos, "s--", color="#2980b9", linewidth=2.2, label=r"Directional Cosine Similarity")
    ax1.set_title(r"Flash-OLS Operator Fidelity vs Softmax ($F_{\text{OLS}} \approx F_{\text{Softmax}}$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fitting Steps (Zero Model Backprop)", fontsize=11)
    ax1.set_ylabel("Fidelity Score", fontsize=11)
    ax1.set_ylim(-0.1, 1.05)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss convergence
    ax2.plot(steps, train_loss, "d-", color="#8e44ad", linewidth=2.2, label=r"Operator Regression Loss ($\text{MSE} + \text{Cosine}$)")
    ax2.set_title(r"Operator Regression Loss Convergence", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Fitting Steps", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_yscale("log")
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig43_operator_fidelity_r2.png"] = buf.getvalue()

    # -------------------------------------------------------------
    # Figure 44: Domain-by-Domain Fidelity Breakdown
    # -------------------------------------------------------------
    domain_data = metrics["domain_fidelity"]
    domain_names = [d.upper() for d in domain_data.keys()]
    domain_r2 = [domain_data[d]["r2"] for d in domain_data.keys()]
    domain_cos = [domain_data[d]["cosine_sim"] for d in domain_data.keys()]

    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=300)
    x = np.arange(len(domain_names))
    width = 0.35

    rects1 = ax.bar(x - width/2, domain_r2, width, label=r"Operator $R^2$ Variance Explained", color="#27ae60", edgecolor="black", linewidth=1.1)
    rects2 = ax.bar(x + width/2, domain_cos, width, label=r"Directional Cosine Similarity", color="#2980b9", edgecolor="black", linewidth=1.1)

    ax.set_title(r"Flash-OLS Operator Fidelity Across Domains (Zero Semantic Bias)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fidelity Metric", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(domain_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    plots["fig44_operator_domain_comparison.png"] = buf.getvalue()

    return plots


@app.local_entrypoint()
def main():
    print("==================================================================")
    print(" Launching EXP-17: Direct Functional Operator Estimation on Modal")
    print(" Target: NVIDIA A10G Cloud GPU (Timeout <= 300s)")
    print("==================================================================")

    # 1. Execute Benchmark Worker
    metrics, adapter_bytes = direct_operator_worker.remote()

    # 2. Generate Publication Plots
    print("\nGenerating publication figures in Modal cloud worker...")
    plot_bytes_dict = generate_operator_plots_cloud.remote(metrics)

    # 3. Save Artifacts Locally
    out_dir = "results/exp17_direct_operator_estimation"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    adapter_path = os.path.join(out_dir, "operator_adapter.pt")
    with open(adapter_path, "wb") as f:
        f.write(adapter_bytes)
    print(f"Saved: {adapter_path} ({len(adapter_bytes) / 1024:.1f} KB)")

    for fname, data in plot_bytes_dict.items():
        p_path = os.path.join(out_dir, fname)
        with open(p_path, "wb") as f:
            f.write(data)
        print(f"Saved: {p_path}")

    print("\n[EXP-17 Complete] All artifacts synchronized to local repository.")


if __name__ == "__main__":
    main()
