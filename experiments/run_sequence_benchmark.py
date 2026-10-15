"""
Sequence Modeling Benchmark (Phase 2 / EXP-03):
1. Multi-Query Associative Recall (MQAR) Accuracy:
   - Standard Softmax Attention vs Pure Linear Attention vs Two-Stream Bulk+Margin
2. Sequence Length Scaling Benchmark (T = 256, 512, 1024, 2048, 4096):
   - Wall-clock time (forward + backward ms)
   - Peak GPU Memory allocation (MB)
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SequenceTransformer
from src.tasks import generate_associative_recall_data


def benchmark_mqar(
    device: torch.device,
    seq_len: int = 512,
    vocab_size: int = 128,
    num_pairs: int = 8,
    num_queries: int = 4,
    n_epochs: int = 15,
    batch_size: int = 32,
    n_train_batches: int = 20,
    n_test_batches: int = 5,
) -> dict:
    """
    Evaluates learning accuracy on Multi-Query Associative Recall (MQAR).
    """
    print(f"\n[MQAR Benchmark] Sequence Length T={seq_len}, Vocab={vocab_size}, Pairs={num_pairs}, Queries={num_queries}")

    # Generate fixed evaluation dataset
    X_test_list, Y_test_list = [], []
    for i in range(n_test_batches):
        x, y = generate_associative_recall_data(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            num_queries=num_queries,
            seed=1000 + i,
        )
        X_test_list.append(x)
        Y_test_list.append(y)

    X_test = torch.cat(X_test_list, dim=0).to(device)
    Y_test = torch.cat(Y_test_list, dim=0).to(device)

    models_to_test = [
        ("Standard Softmax", "standard"),
        ("Pure Linear Attention", "linear"),
        ("Two-Stream Bulk+Margin", "two_stream"),
    ]

    mqar_results = {}

    for name, attn_type in models_to_test:
        print(f"\n--- Training {name} ---")
        torch.manual_seed(42)
        model = SequenceTransformer(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ffn=128,
            max_seq_len=max(seq_len + 64, 4096),
            attn_type=attn_type,
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
                logits = model(X_batch)  # (B, T, V)
                loss = criterion(logits.view(-1, vocab_size), Y_batch.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= n_train_batches

            # Evaluate test accuracy on query positions
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

            if epoch % 3 == 0 or epoch == n_epochs:
                print(f"Epoch {epoch:2d}/{n_epochs} | Train Loss: {epoch_loss:.4f} | Query Accuracy: {test_acc * 100:.2f}% | Elapsed: {elapsed:.2f}s")

        mqar_results[name] = history

    return mqar_results


def benchmark_scaling(
    device: torch.device,
    seq_lengths: list[int] = [256, 512, 1024, 2048, 4096],
    batch_size: int = 4,
    vocab_size: int = 128,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> dict:
    """
    Measures forward+backward pass wall-clock latency (ms) and peak GPU memory (MB)
    across sequence lengths T in [256, 4096].
    """
    print("\n[Scaling Benchmark] Evaluating forward+backward latency and peak memory across sequence lengths...")

    models_to_test = [
        ("Standard Softmax", "standard"),
        ("Pure Linear Attention", "linear"),
        ("Two-Stream Bulk+Margin", "two_stream"),
    ]

    scaling_results = {
        "seq_lengths": seq_lengths,
        "latency_ms": {name: [] for name, _ in models_to_test},
        "memory_mb": {name: [] for name, _ in models_to_test},
    }

    is_cuda = device.type == "cuda"

    for name, attn_type in models_to_test:
        print(f"\n--- Scaling Profile: {name} ---")
        for T in seq_lengths:
            torch.cuda.empty_cache() if is_cuda else None
            model = SequenceTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ffn=d_model * 2,
                max_seq_len=max(seq_lengths) + 128,
                attn_type=attn_type,
            ).to(device)
            model.train()

            x = torch.randint(0, vocab_size, (batch_size, T), device=device)
            target = torch.randint(0, vocab_size, (batch_size, T), device=device)
            criterion = nn.CrossEntropyLoss()

            # Warmup passes
            for _ in range(n_warmup):
                out = model(x)
                loss = criterion(out.view(-1, vocab_size), target.view(-1))
                loss.backward()
                model.zero_grad()

            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # Timed passes
            t0 = time.time()
            for _ in range(n_iters):
                out = model(x)
                loss = criterion(out.view(-1, vocab_size), target.view(-1))
                loss.backward()
                model.zero_grad()

            if is_cuda:
                torch.cuda.synchronize()
            total_time = (time.time() - t0) * 1000.0 / n_iters  # ms per pass

            peak_mem = 0.0
            if is_cuda:
                peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

            scaling_results["latency_ms"][name].append(round(total_time, 2))
            scaling_results["memory_mb"][name].append(round(peak_mem, 2))

            print(f"T = {T:4d} | Latency: {total_time:6.2f} ms | Peak Memory: {peak_mem:6.1f} MB")

    return scaling_results


def run_sequence_suite(device: torch.device) -> dict:
    """Executes complete sequence benchmark suite."""
    mqar = benchmark_mqar(device=device)
    scaling = benchmark_scaling(device=device)
    return {"mqar": mqar, "scaling": scaling}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Sequence Benchmark on {device}...")
    results = run_sequence_suite(device)

    out_dir = os.path.join(os.path.dirname(__file__), "../results/exp03_sequence_bulk_margin")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSequence benchmark completed! Results saved to {out_path}")


if __name__ == "__main__":
    main()
