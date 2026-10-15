"""
Kalman & Recursive Least Squares (RLS) Attention Benchmark (Phase 3 / EXP-04):
1. Multi-Query Associative Recall (MQAR) Accuracy:
   - Standard Softmax vs Pure Linear vs Delta/LMS vs Causal RLS/Kalman Attention
2. Autoregressive Generation Scaling (O(1) Memory vs Expanding KV-Cache):
   - Measure per-token latency and memory footprint during token-by-token generation across T in [256, 4096].
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

from src.models import SequenceTransformer, CausalRLSAttention, CausalDeltaAttention, CausalStandardAttention
from src.tasks import generate_associative_recall_data


def benchmark_mqar_kalman(
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
    Evaluates learning accuracy on Multi-Query Associative Recall across:
    1. Standard Softmax
    2. Pure Linear Attention
    3. Delta / LMS Attention
    4. Causal RLS / Kalman Attention
    """
    print(f"\n[Phase 3 MQAR Benchmark] Sequence Length T={seq_len}, Vocab={vocab_size}, Pairs={num_pairs}, Queries={num_queries}")

    # Generate fixed evaluation dataset
    X_test_list, Y_test_list = [], []
    for i in range(n_test_batches):
        x, y = generate_associative_recall_data(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            num_queries=num_queries,
            seed=2000 + i,
        )
        X_test_list.append(x)
        Y_test_list.append(y)

    X_test = torch.cat(X_test_list, dim=0).to(device)
    Y_test = torch.cat(Y_test_list, dim=0).to(device)

    models_to_test = [
        ("Standard Softmax", "standard"),
        ("Pure Linear Attention", "linear"),
        ("Delta / LMS Attention", "delta"),
        ("Causal RLS / Kalman Attention", "rls"),
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
                logits = model(X_batch)
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

            if epoch % 5 == 0 or epoch == n_epochs:
                print(f"Epoch {epoch:2d}/{n_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {test_acc * 100:.2f}% | Elapsed: {elapsed:.2f}s")

        mqar_results[name] = history

    return mqar_results


def benchmark_autoregressive_inference(
    device: torch.device,
    seq_lengths: list[int] = [256, 512, 1024, 2048, 4096],
    d_model: int = 64,
    n_heads: int = 4,
    batch_size: int = 1,
) -> dict:
    """
    Simulates token-by-token autoregressive generation comparing:
    - Standard Attention KV-cache (O(T) linear growth in memory)
    - Causal RLS / Kalman Attention (O(1) constant memory)
    """
    print("\n[Inference Benchmark] Simulating token-by-token generation across sequence lengths...")
    is_cuda = device.type == "cuda"
    d_head = d_model // n_heads

    inference_results = {
        "seq_lengths": seq_lengths,
        "memory_mb": {"KV-Cache (Standard Softmax)": [], "Causal RLS / Kalman Attention": []},
        "per_token_ms": {"KV-Cache (Standard Softmax)": [], "Causal RLS / Kalman Attention": []},
    }

    rls_layer = CausalRLSAttention(d_model=d_model, n_heads=n_heads).to(device)

    for T in seq_lengths:
        torch.cuda.empty_cache() if is_cuda else None

        # -------------------------------------------------------------
        # 1. Causal RLS Constant State Generation
        # -------------------------------------------------------------
        rls_state = rls_layer.init_state(batch_size=batch_size, device=device)
        dummy_input = torch.randn(batch_size, d_model, device=device)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(min(T, 512)):  # Run generation steps
            with torch.no_grad():
                out, rls_state = rls_layer.step(dummy_input, rls_state)

        if is_cuda:
            torch.cuda.synchronize()
        rls_ms_per_token = (time.time() - t0) * 1000.0 / min(T, 512)

        # Theoretical memory of state (S in R^{B x H x d x d}, P in R^{B x H x d x d})
        # 2 tensors of size B * H * d * d * 4 bytes
        rls_mem = (2 * batch_size * n_heads * d_head * d_head * 4) / (1024 * 1024)  # MB

        # -------------------------------------------------------------
        # 2. KV-Cache Memory Growth (Standard Softmax)
        # -------------------------------------------------------------
        # KV Cache stores K and V of shape (B, H, T, d) across 2 layers:
        # 2 * 2 * B * H * T * d * 4 bytes
        kv_cache_mem = (2 * 2 * batch_size * n_heads * T * d_head * 4) / (1024 * 1024)  # MB

        # KV-cache attention computation at step T: dot product over T tokens
        q_dummy = torch.randn(batch_size, n_heads, 1, d_head, device=device)
        k_cache = torch.randn(batch_size, n_heads, T, d_head, device=device)
        v_cache = torch.randn(batch_size, n_heads, T, d_head, device=device)

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            with torch.no_grad():
                scores = torch.matmul(q_dummy, k_cache.transpose(-2, -1))
                attn = torch.softmax(scores, dim=-1)
                _ = torch.matmul(attn, v_cache)
        if is_cuda:
            torch.cuda.synchronize()
        kv_ms_per_token = (time.time() - t0) * 1000.0 / 50

        inference_results["memory_mb"]["Causal RLS / Kalman Attention"].append(round(rls_mem, 4))
        inference_results["memory_mb"]["KV-Cache (Standard Softmax)"].append(round(kv_cache_mem, 4))
        inference_results["per_token_ms"]["Causal RLS / Kalman Attention"].append(round(rls_ms_per_token, 4))
        inference_results["per_token_ms"]["KV-Cache (Standard Softmax)"].append(round(kv_ms_per_token, 4))

        print(f"T = {T:4d} | KV-Cache Memory: {kv_cache_mem:6.3f} MB | RLS Memory: {rls_mem:6.3f} MB (O(1) Flat)")

    return inference_results


def run_kalman_suite(device: torch.device) -> dict:
    """Executes complete Phase 3 Kalman/RLS suite."""
    mqar = benchmark_mqar_kalman(device=device)
    inference = benchmark_autoregressive_inference(device=device)
    return {"mqar": mqar, "inference": inference}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Kalman / RLS Benchmark on {device}...")
    results = run_kalman_suite(device)

    out_dir = os.path.join(os.path.dirname(__file__), "../results/exp04_kalman_rls_attention")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nKalman / RLS benchmark completed! Results saved to {out_path}")


if __name__ == "__main__":
    main()
