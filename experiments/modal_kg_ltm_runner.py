"""
Modal Cloud Runner for EXP-25: Cognitive Knowledge-Graph Long-Term Memory (CKG-LTM).
==================================================================================
Runs the benchmark comparing:
1. 2-Layer Softmax Attention (Linear KV-Cache, suffers entropy diffusion over long contexts)
2. Pure Recurrent State (Linear/RLS without consolidation, suffers noise saturation)
3. Naive Surprise-Only Memory (distractor noise poisoning)
4. Cognitive KG-LTM (Surprise Working Memory + Top-Down & Repetition Gating + Closed-Form Solvable KG)

Strictly enforces timeout <= 300s on NVIDIA A10G Cloud GPU.
"""

import os
import json
import modal

app = modal.App("ols-cognitive-kg-ltm-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "matplotlib")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_exp25_worker() -> dict:
    import torch
    import torch.nn.functional as F
    from experiments.run_cognitive_kg_ltm_benchmark import (
        generate_noisy_relational_dataset,
        TwoLayerSoftmaxBaseline,
        PureRecurrentBaseline,
        NaiveSurpriseOnlyBaseline,
        CognitiveKnowledgeGraphAttention,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Modal Cloud Worker] Active Device: {device} ({torch.cuda.get_device_name(0)})")

    seq_lens = [256, 512, 1024, 2048]
    d_model = 128

    metrics = {
        "sequence_lengths": seq_lens,
        "models": {
            "softmax": {"accuracy": [], "memory_kb": [], "noise_rejection": []},
            "pure_recurrent": {"accuracy": [], "memory_kb": [], "noise_rejection": []},
            "naive_surprise": {"accuracy": [], "memory_kb": [], "noise_rejection": []},
            "cognitive_kg_ltm": {"accuracy": [], "memory_kb": [], "noise_rejection": []},
        },
        "diagnostics": {},
    }

    for T in seq_lens:
        print(f"\n--- Evaluating Sequence Length T = {T} ---")
        X, queries, targets, goals = generate_noisy_relational_dataset(
            num_samples=64, seq_len=T, d_model=d_model, num_chains=4, num_hops=2, noise_ratio=0.5, seed=42
        )
        X, queries, targets, goals = X.to(device), queries.to(device), targets.to(device), goals.to(device)

        m_softmax = TwoLayerSoftmaxBaseline(d_model).to(device)
        m_recurrent = PureRecurrentBaseline(d_model).to(device)
        m_naive = NaiveSurpriseOnlyBaseline(d_model).to(device)
        m_ckg = CognitiveKnowledgeGraphAttention(d_model, surprise_thresh=0.4, consolidation_thresh=0.5).to(device)

        # 1. 2-Layer Softmax Attention
        with torch.no_grad():
            out_sm = m_softmax(X, queries)
            sim_sm = F.cosine_similarity(out_sm, targets, dim=-1)
            acc_sm = (sim_sm > 0.60).float().mean().item() * 100.0
            mem_sm_kb = (2 * T * d_model * 4 * 2) / 1024.0  # 2 layers of KV cache
            metrics["models"]["softmax"]["accuracy"].append(round(acc_sm, 2))
            metrics["models"]["softmax"]["memory_kb"].append(round(mem_sm_kb, 1))
            metrics["models"]["softmax"]["noise_rejection"].append(0.0)

        # 2. Pure Recurrent Attention
        with torch.no_grad():
            out_rec = m_recurrent(X, queries)
            sim_rec = F.cosine_similarity(out_rec, targets, dim=-1)
            acc_rec = (sim_rec > 0.60).float().mean().item() * 100.0
            mem_rec_kb = (2 * d_model * d_model * 4) / 1024.0
            metrics["models"]["pure_recurrent"]["accuracy"].append(round(acc_rec, 2))
            metrics["models"]["pure_recurrent"]["memory_kb"].append(round(mem_rec_kb, 1))
            metrics["models"]["pure_recurrent"]["noise_rejection"].append(0.0)

        # 3. Naive Surprise-Only Baseline
        with torch.no_grad():
            out_naive, rej_naive = m_naive(X, queries, goals)
            sim_naive = F.cosine_similarity(out_naive, targets, dim=-1)
            acc_naive = (sim_naive > 0.60).float().mean().item() * 100.0
            mem_naive_kb = (2 * d_model * d_model * 4) / 1024.0
            metrics["models"]["naive_surprise"]["accuracy"].append(round(acc_naive, 2))
            metrics["models"]["naive_surprise"]["memory_kb"].append(round(mem_naive_kb, 1))
            metrics["models"]["naive_surprise"]["noise_rejection"].append(round(rej_naive * 100.0, 1))

        # 4. Cognitive KG-LTM
        with torch.no_grad():
            out_ckg, diag = m_ckg(X, queries, goal=goals)
            sim_ckg = F.cosine_similarity(out_ckg, targets, dim=-1)
            acc_ckg = (sim_ckg > 0.60).float().mean().item() * 100.0
            mem_ckg_kb = (2 * d_model * d_model * 4) / 1024.0
            metrics["models"]["cognitive_kg_ltm"]["accuracy"].append(round(acc_ckg, 2))
            metrics["models"]["cognitive_kg_ltm"]["memory_kb"].append(round(mem_ckg_kb, 1))
            metrics["models"]["cognitive_kg_ltm"]["noise_rejection"].append(
                round(diag["noise_rejection_rate"] * 100.0, 1)
            )

        print(f"Results at T={T}:")
        print(f"  2-Layer Softmax       : Acc = {acc_sm:.2f}%, Cache Memory = {mem_sm_kb:.1f} KB")
        print(f"  Pure Recurrent (RLS)  : Acc = {acc_rec:.2f}%, State Memory = {mem_rec_kb:.1f} KB")
        print(f"  Naive Surprise-Only   : Acc = {acc_naive:.2f}%, Noise Rejection = {rej_naive*100:.1f}%, State = {mem_naive_kb:.1f} KB")
        print(
            f"  Cognitive KG-LTM      : Acc = {acc_ckg:.2f}%, Noise Rejection = {diag['noise_rejection_rate']*100:.1f}%, Memory = {mem_ckg_kb:.1f} KB"
        )

    return metrics


@app.function(image=image, timeout=60)
def generate_exp25_plots_cloud(metrics: dict) -> dict[str, bytes]:
    import io
    import matplotlib.pyplot as plt
    import numpy as np

    seq_lens = metrics["sequence_lengths"]
    plots = {}

    # Figure 67: Multi-Hop Retrieval Accuracy Across Sequence Lengths
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for model_name, label, color, marker in [
        ("cognitive_kg_ltm", "Cognitive KG-LTM (Surprise + Top-Down + Repetition)", "#10B981", "o"),
        ("softmax", "2-Layer Softmax Attention (Linear KV-Cache)", "#EF4444", "s"),
        ("naive_surprise", "Naive Surprise-Only (No Repetition Gate)", "#F59E0B", "x"),
        ("pure_recurrent", "Pure Recurrent / RLS (Unguided State)", "#3B82F6", "^"),
    ]:
        ax.plot(
            seq_lens,
            metrics["models"][model_name]["accuracy"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=8,
        )
    ax.set_title("EXP-25: Multi-Hop Relational Retrieval Accuracy vs Sequence Length", fontsize=12, fontweight="bold")
    ax.set_xlabel("Context Sequence Length (T tokens)", fontsize=11)
    ax.set_ylabel("Multi-Hop Retrieval Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    plots["fig67_kg_ltm_multihop_accuracy.png"] = buf.getvalue()

    # Figure 68: Distractor Noise Rejection
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    models = [
        "Softmax\n(All Cached)",
        "Pure RLS\n(All Ingested)",
        "Naive Surprise\n(Noise Poisoned)",
        "Cognitive KG-LTM\n(Consolidated)",
    ]
    rejections = [
        0.0,
        0.0,
        metrics["models"]["naive_surprise"]["noise_rejection"][-1],
        metrics["models"]["cognitive_kg_ltm"]["noise_rejection"][-1],
    ]
    colors = ["#EF4444", "#3B82F6", "#F59E0B", "#10B981"]
    bars = ax.bar(models, rejections, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("EXP-25: Distractor Noise Rejection Rate from Long-Term Memory", fontsize=12, fontweight="bold")
    ax.set_ylabel("One-Off Noise Rejection Rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    plots["fig68_kg_ltm_noise_rejection.png"] = buf.getvalue()

    # Figure 69: Long-Term Memory State Footprint Scaling
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(
        seq_lens,
        metrics["models"]["softmax"]["memory_kb"],
        label="Softmax KV-Cache: O(T) Explosion",
        color="#EF4444",
        marker="s",
        linewidth=2.5,
    )
    ax.plot(
        seq_lens,
        metrics["models"]["cognitive_kg_ltm"]["memory_kb"],
        label="Cognitive KG-LTM: O(1) Invariant State",
        color="#10B981",
        marker="o",
        linewidth=2.5,
        linestyle="--",
    )
    ax.set_title("EXP-25: Serving State Memory Footprint vs Sequence Length", fontsize=12, fontweight="bold")
    ax.set_xlabel("Context Sequence Length (T tokens)", fontsize=11)
    ax.set_ylabel("State Memory Footprint (KB)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    plots["fig69_kg_ltm_memory_scaling.png"] = buf.getvalue()

    return plots


@app.local_entrypoint()
def main():
    print("[Local Entrypoint] Triggering Modal GPU worker for EXP-25...")
    metrics = run_exp25_worker.remote()

    print("[Local Entrypoint] Generating publication plots in Modal cloud...")
    plots = generate_exp25_plots_cloud.remote(metrics)

    out_dir = "results/exp25_cognitive_kg_ltm"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Local Entrypoint] Metrics saved to {metrics_path}")

    for filename, img_bytes in plots.items():
        img_path = os.path.join(out_dir, filename)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        print(f"[Local Entrypoint] Plot saved to {img_path}")

    print("\nEXP-25 Cloud Benchmark Completed Successfully!")
