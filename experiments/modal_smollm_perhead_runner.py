"""
Modal Cloud Runner for EXP-30: SmolLM2-1.7B Per-Head Ablation Suite.
=====================================================================
Dispatches 3 parallel cloud GPU workers (NVIDIA A10G, timeout <= 300s):
- Worker 1: UUID Multi-Token Identifier Recall (n=30)
- Worker 2: S-NIAH Passkey Recall (n=30)
- Worker 3: Displaced Context Retroactive Recall (n=30)

Generates:
- results/exp30_smollm_perhead_ablation/metrics.json
- results/exp30_smollm_perhead_ablation/fig86_smollm_perhead_vs_shared.png
"""

import os
import io
import json
import modal

app = modal.App("ols-attention-exp30-smollm-perhead")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy", "matplotlib", "requests")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def run_task_worker(task_name: str) -> dict:
    try:
        from experiments.run_smollm_perhead_ablation import run_smollm_task_eval
        return run_smollm_task_eval(task_name=task_name, device="cuda")
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.function(image=image, timeout=120)
def compile_and_plot_cloud(all_task_results: dict) -> dict:
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from experiments.run_smollm_perhead_ablation import compile_experiment_results

    compiled_metrics = compile_experiment_results(all_task_results)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plots = {}

    summary = compiled_metrics["summary"]
    methods = ["original_softmax", "snapkv", "halo_shared", "halo_perhead"]
    method_labels = [
        "Full Softmax (Ref)",
        "SnapKV (Shared)",
        "HALO (Shared)",
        "HALO-PerHead (Ours)",
    ]
    colors = ["#2b5c8f", "#386cb0", "#e66101", "#1b9e77"]

    tasks = ["uuid", "sniah", "retroactive", "composite"]
    task_titles = ["Multi-Token UUID", "S-NIAH (Passkey)", "Displaced Context", "Composite (n=90)"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    for ax_idx, (t_key, t_title) in enumerate(zip(tasks, task_titles)):
        ax = axes[ax_idx]
        means = [summary[m][t_key]["mean"] for m in methods]
        ci95s = [summary[m][t_key]["ci95"] for m in methods]

        bars = ax.bar(
            range(len(methods)),
            means,
            yerr=ci95s,
            capsize=4,
            color=colors,
            alpha=0.88,
            edgecolor="black",
            linewidth=1.2,
            width=0.55,
        )

        ax.set_title(t_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, val in zip(bars, means):
            ax.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    axes[0].set_ylabel("Retrieval Accuracy (%)", fontsize=11, fontweight="bold")
    plt.suptitle(
        "SmolLM2-1.7B (32-Head MHA, 22 Layers): Resolving Head Starvation with HALO-PerHead\n"
        "(Context T=4096, Window W=512, Landmark Cache M=128, Compression 5.33x)",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plots["fig86_smollm_perhead_vs_shared.png"] = buf.getvalue()
    plt.close()

    return {
        "metrics": compiled_metrics,
        "plots": plots,
    }


@app.local_entrypoint()
def main():
    print("=" * 75)
    print("DISPATCHING EXP-30: SMOLLM2-1.7B PER-HEAD ABLATION ON MODAL GPU")
    print("=" * 75)

    # Launch the 3 task workers concurrently
    print("\n[Modal] Spawning Worker 1 (UUID), Worker 2 (S-NIAH), Worker 3 (Retroactive)...")
    uuid_handle = run_task_worker.spawn("uuid")
    sniah_handle = run_task_worker.spawn("sniah")
    retro_handle = run_task_worker.spawn("retroactive")

    print("[Modal] Waiting for cloud workers to complete (timeout <= 300s)...")
    uuid_res = uuid_handle.get()
    sniah_res = sniah_handle.get()
    retro_res = retro_handle.get()

    for name, res in [("UUID", uuid_res), ("S-NIAH", sniah_res), ("Retroactive", retro_res)]:
        if "error" in res:
            print(f"\n[FATAL ERROR] {name} worker failed:\n{res['error']}\n{res.get('traceback', '')}")
            raise RuntimeError(f"Worker {name} failed: {res['error']}")
        print(f"[Modal] Worker {name} completed successfully.")

    task_results = {
        "uuid": uuid_res,
        "sniah": sniah_res,
        "retroactive": retro_res,
    }

    print("\n[Modal] Compiling master metrics & rendering figure in cloud container...")
    cloud_output = compile_and_plot_cloud.remote(task_results)
    compiled_metrics = cloud_output["metrics"]
    plots = cloud_output["plots"]

    # Save to results directory
    out_dir = os.path.join("results", "exp30_smollm_perhead_ablation")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(compiled_metrics, f, indent=2)
    print(f"\n[Artifact] Saved exact metrics to: {metrics_path}")

    for plot_name, plot_bytes in plots.items():
        plot_path = os.path.join(out_dir, plot_name)
        with open(plot_path, "wb") as f:
            f.write(plot_bytes)
        print(f"[Artifact] Saved figure to: {plot_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print("EXP-30 RESULTS SUMMARY (SmolLM2-1.7B, 32-Head MHA):")
    print("=" * 75)
    print(f"{'Method':<22} | {'Composite (n=90)':<18} | {'UUID (n=30)':<14} | {'S-NIAH (n=30)':<14} | {'Retro (n=30)':<14}")
    print("-" * 92)
    summary = compiled_metrics["summary"]
    for m in ["original_softmax", "snapkv", "halo_shared", "halo_perhead"]:
        c = f"{summary[m]['composite']['mean']}% (±{summary[m]['composite']['ci95']}%)"
        u = f"{summary[m]['uuid']['mean']}%"
        s = f"{summary[m]['sniah']['mean']}%"
        r = f"{summary[m]['retroactive']['mean']}%"
        print(f"{m:<22} | {c:<18} | {u:<14} | {s:<14} | {r:<14}")
    print("=" * 75)
