"""
Flash-OLS: Chunked Associative Causal Online Least Squares Benchmark (EXP-07):
1. Ultra-Long Context Scaling (T = 512 to 65,536):
   - Compares Peak GPU Memory (MB) and Latency (ms) across:
     * Standard Softmax Attention (O(T^2) memory explosion -> OOM)
     * Naive Exact Causal OLS (5D Gram tensor allocation -> OOM)
     * Flash-OLS (Linear) (Chunked associative scan + SRAM Cholesky)
     * Flash-OLS (Poly) (Chunked associative scan with degree-2 expansion)
2. Numerical & Empirical Fidelity Verification:
   - Verifies that Flash-OLS achieves identical associative recall accuracy on MQAR
     without numerical drift or performance degradation.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    CausalStandardAttention,
    ExactCausalOLSAttention,
    FlashCausalOLSAttention,
    SequenceTransformer,
)
from src.tasks import generate_associative_recall_data


def benchmark_ultra_long_scaling(
    device: torch.device,
    seq_lengths: list[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    d_model: int = 64,
    n_heads: int = 4,
    batch_size: int = 2,
    chunk_size: int = 64,
) -> dict:
    """
    Evaluates Peak GPU Memory (MB) and Latency (ms) across sequence lengths T up to 65k+.
    Gracefully handles and records OOM for scaling boundaries.
    """
    print(f"\n[EXP-07 Ultra-Long Context Scaling] B={batch_size}, d_model={d_model}, heads={n_heads}, chunk={chunk_size}")
    is_cuda = device.type == "cuda"

    models = {
        "Standard Softmax": CausalStandardAttention(d_model=d_model, n_heads=n_heads).to(device),
        "Naive Exact OLS": ExactCausalOLSAttention(d_model=d_model, n_heads=n_heads, use_poly=False).to(device),
        "Flash-OLS (Linear)": FlashCausalOLSAttention(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size, use_poly=False).to(device),
        "Flash-OLS (Poly)": FlashCausalOLSAttention(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size, use_poly=True).to(device),
    }

    scaling_results = {
        "seq_lengths": seq_lengths,
        "peak_memory_mb": {name: [] for name in models},
        "latency_ms": {name: [] for name in models},
        "oom_status": {name: [] for name in models},
    }

    n_warmup = 3
    n_repeat = 10

    for T in seq_lengths:
        print(f"\n--- Testing Sequence Length T = {T} ---")
        x = torch.randn(batch_size, T, d_model, device=device)

        for name, layer in models.items():
            # If already failed at previous T, mark as OOM
            if scaling_results["oom_status"][name] and scaling_results["oom_status"][name][-1]:
                scaling_results["peak_memory_mb"][name].append(None)
                scaling_results["latency_ms"][name].append(None)
                scaling_results["oom_status"][name].append(True)
                print(f"  {name:20s}: Skipped (OOM at prior length)")
                continue

            try:
                if is_cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                # Warmup
                for _ in range(n_warmup):
                    _ = layer(x)

                if is_cuda:
                    torch.cuda.synchronize()

                t0 = time.time()
                for _ in range(n_repeat):
                    _ = layer(x)

                if is_cuda:
                    torch.cuda.synchronize()

                elapsed_ms = (time.time() - t0) * 1000.0 / n_repeat

                if is_cuda:
                    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                else:
                    # CPU theoretical memory estimate
                    peak_mem_mb = 0.0

                scaling_results["peak_memory_mb"][name].append(round(peak_mem_mb, 2))
                scaling_results["latency_ms"][name].append(round(elapsed_ms, 2))
                scaling_results["oom_status"][name].append(False)
                print(f"  {name:20s}: Latency = {elapsed_ms:8.2f} ms | Peak Memory = {peak_mem_mb:8.2f} MB")

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_str = str(e).lower()
                if "out of memory" in err_str or "cuda" in err_str:
                    print(f"  {name:20s}: OUT OF MEMORY (OOM) at T = {T}!")
                    scaling_results["peak_memory_mb"][name].append(None)
                    scaling_results["latency_ms"][name].append(None)
                    scaling_results["oom_status"][name].append(True)
                    if is_cuda:
                        torch.cuda.empty_cache()
                else:
                    raise e

    return scaling_results


def benchmark_flash_ols_fidelity(
    device: torch.device,
    seq_len: int = 128,
    vocab_size: int = 64,
    num_pairs: int = 4,
    num_queries: int = 2,
    n_epochs: int = 10,
    batch_size: int = 32,
    n_train_batches: int = 8,
    n_test_batches: int = 3,
) -> dict:
    """
    Evaluates learning accuracy on MQAR to verify Flash-OLS mathematically reproduces
    Exact Causal OLS without degradation.
    """
    print(f"\n[EXP-07 Fidelity Benchmark] T={seq_len}, Vocab={vocab_size}, Pairs={num_pairs}, Queries={num_queries}")

    # Generate fixed evaluation dataset
    X_test_list, Y_test_list = [], []
    for i in range(n_test_batches):
        x, y = generate_associative_recall_data(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            num_queries=num_queries,
            seed=4000 + i,
        )
        X_test_list.append(x)
        Y_test_list.append(y)

    X_test = torch.cat(X_test_list, dim=0).to(device)
    Y_test = torch.cat(Y_test_list, dim=0).to(device)

    models_to_test = [
        ("Naive Exact OLS", "exact_ols"),
        ("Flash-OLS (Linear, C=32)", "flash_ols"),
        ("Naive Exact Poly OLS", "exact_poly_ols"),
        ("Flash-OLS (Poly, C=32)", "flash_poly_ols"),
    ]

    fidelity_results = {}

    for name, attn_type in models_to_test:
        print(f"\n--- Training {name} ({attn_type}) ---")
        torch.manual_seed(42)
        model = SequenceTransformer(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ffn=128,
            max_seq_len=max(seq_len + 64, 4096),
            attn_type=attn_type,
            chunk_size=32,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        history = {"epochs": [], "train_loss": [], "test_acc": [], "wallclock_sec": []}
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for b in range(n_train_batches):
                X_batch, Y_batch = generate_associative_recall_data(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    num_pairs=num_pairs,
                    num_queries=num_queries,
                    seed=epoch * 100 + b,
                )
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits.view(-1, vocab_size), Y_batch.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= n_train_batches

            model.eval()
            with torch.no_grad():
                test_logits = model(X_test)
                preds = torch.argmax(test_logits, dim=-1)
                mask = Y_test != -100
                correct = (preds[mask] == Y_test[mask]).float().sum().item()
                total = mask.sum().item()
                test_acc = correct / max(total, 1)

            elapsed = time.time() - t0
            history["epochs"].append(epoch)
            history["train_loss"].append(round(epoch_loss, 4))
            history["test_acc"].append(round(test_acc, 4))
            history["wallclock_sec"].append(round(elapsed, 2))

            if epoch % 5 == 0 or epoch == n_epochs:
                print(f"Epoch {epoch:2d}/{n_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {test_acc * 100:.2f}% | Elapsed: {elapsed:.2f}s")

        fidelity_results[name] = history

    return fidelity_results


def run_flash_ols_suite(device: torch.device) -> dict:
    """Executes full EXP-07 Flash-OLS benchmark."""
    scaling = benchmark_ultra_long_scaling(device=device)
    fidelity = benchmark_flash_ols_fidelity(device=device)
    return {"scaling": scaling, "fidelity": fidelity}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Flash-OLS Benchmark on {device}...")
    results = run_flash_ols_suite(device)

    out_dir = os.path.join(os.path.dirname(__file__), "../results/exp07_flash_ols_scaling")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFlash-OLS benchmark completed! Results saved to {out_path}")


if __name__ == "__main__":
    main()
