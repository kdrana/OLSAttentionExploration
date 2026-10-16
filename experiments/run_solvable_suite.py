"""
Solvable Attention Architectures Suite Benchmark (Phase 4 / EXP-05):
Evaluates 7 Attention Architectures:
1. Standard Softmax Attention (Classical O(T^2) baseline)
2. Pure Linear Attention (Unnormalized additive accumulator)
3. Delta / LMS Attention (Innovation error with scalar gate)
4. Causal RLS / Kalman Attention (Full matrix Kalman gain)
5. Chunkwise Closed-Form Ridge Attention (Intra-chunk closed-form GPU solve)
6. Diagonal Kalman Attention (O(d) diagonal precision tracking + learnable gate)
7. Chunkwise Degree-2 Polynomial Solvable Attention (Pairwise cross-terms solved in closed form)

Metrics evaluated:
- Multi-Query Associative Recall (MQAR) Accuracy & Loss Curves
- Training Wall-Clock Seconds & Throughput
- Autoregressive Generation Latency (ms/token) across T in [256, 4096]
- Autoregressive State Memory Footprint (MB) across T in [256, 4096]
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
    SequenceTransformer,
    CausalStandardAttention,
    CausalRLSAttention,
    CausalDeltaAttention,
    ChunkwiseRidgeAttention,
    DiagonalKalmanAttention,
    ChunkwisePolySolvableAttention,
)
from src.tasks import generate_associative_recall_data


def benchmark_mqar_suite(
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
    """Evaluates learning accuracy on Multi-Query Associative Recall across 7 architectures."""
    print(f"\n[EXP-05 MQAR Suite] Sequence Length T={seq_len}, Vocab={vocab_size}, Pairs={num_pairs}, Queries={num_queries}")

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
        ("Causal RLS / Kalman", "rls"),
        ("Chunkwise Ridge Attention", "chunk_ridge"),
        ("Diagonal Kalman Attention", "diag_kalman"),
        ("Chunkwise Poly Solvable", "chunk_poly"),
    ]

    mqar_results = {}

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


def benchmark_autoregressive_inference_suite(
    device: torch.device,
    seq_lengths: list[int] = [256, 512, 1024, 2048, 4096],
    d_model: int = 64,
    n_heads: int = 4,
    batch_size: int = 1,
) -> dict:
    """
    Simulates token-by-token autoregressive generation comparing:
    - Standard Attention KV-cache (O(T) linear growth in memory)
    - Causal RLS / Kalman Attention (O(1) constant full matrix state)
    - Diagonal Kalman Attention (O(1) constant diagonal state)
    - Chunkwise Ridge Attention (O(1) bounded state + chunk buffer)
    - Chunkwise Poly Solvable Attention (O(1) bounded poly state + chunk buffer)
    """
    print("\n[EXP-05 Inference Benchmark] Simulating token-by-token generation across sequence lengths...")
    is_cuda = device.type == "cuda"
    d_head = d_model // n_heads

    model_names = [
        "KV-Cache (Standard Softmax)",
        "Causal RLS / Kalman",
        "Diagonal Kalman Attention",
        "Chunkwise Ridge Attention",
        "Chunkwise Poly Solvable",
    ]

    inference_results = {
        "seq_lengths": seq_lengths,
        "memory_mb": {name: [] for name in model_names},
        "per_token_ms": {name: [] for name in model_names},
    }

    # Instantiate layer modules
    rls_layer = CausalRLSAttention(d_model=d_model, n_heads=n_heads).to(device)
    diag_layer = DiagonalKalmanAttention(d_model=d_model, n_heads=n_heads).to(device)
    chunk_ridge_layer = ChunkwiseRidgeAttention(d_model=d_model, n_heads=n_heads, chunk_size=16).to(device)
    chunk_poly_layer = ChunkwisePolySolvableAttention(d_model=d_model, n_heads=n_heads, chunk_size=16).to(device)

    # Pre-calculate theoretical memory sizes for 2-layer sequence models:
    # 1. KV-Cache: 2 layers * 2 (K and V) * B * H * T * d_head * 4 bytes
    # 2. RLS: 2 layers * (S in R^{B x H x d x d} + P in R^{B x H x d x d}) * 4 bytes
    # 3. Diag Kalman: 2 layers * (S in R^{B x H x d x d} + p in R^{B x H x d}) * 4 bytes
    # 4. Chunk Ridge: 2 layers * (S in R^{B x H x d x d} + 2 * C * d_head) * 4 bytes
    # 5. Chunk Poly: 2 layers * (S in R^{B x H x d_poly x d} + 2 * C * d_poly) * 4 bytes (d_poly = 152)
    d_poly = chunk_poly_layer.d_poly
    chunk_size = 16

    rls_mem_mb = (2 * 2 * batch_size * n_heads * d_head * d_head * 4) / (1024 * 1024)
    diag_mem_mb = (2 * batch_size * n_heads * (d_head * d_head + d_head) * 4) / (1024 * 1024)
    chunk_ridge_mem_mb = (2 * batch_size * n_heads * (d_head * d_head + 2 * chunk_size * d_head) * 4) / (1024 * 1024)
    chunk_poly_mem_mb = (2 * batch_size * n_heads * (d_poly * d_head + 2 * chunk_size * d_poly) * 4) / (1024 * 1024)

    dummy_input = torch.randn(batch_size, d_model, device=device)
    n_warmup = 10
    n_measure = 50

    for T in seq_lengths:
        torch.cuda.empty_cache() if is_cuda else None
        kv_cache_mem_mb = (2 * 2 * batch_size * n_heads * T * d_head * 4) / (1024 * 1024)

        # 1. KV-cache benchmark at step T
        q_dummy = torch.randn(batch_size, n_heads, 1, d_head, device=device)
        k_cache = torch.randn(batch_size, n_heads, T, d_head, device=device)
        v_cache = torch.randn(batch_size, n_heads, T, d_head, device=device)

        if is_cuda:
            torch.cuda.synchronize()
        for _ in range(n_warmup):
            scores = torch.matmul(q_dummy, k_cache.transpose(-2, -1))
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v_cache)
        if is_cuda:
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_measure):
            scores = torch.matmul(q_dummy, k_cache.transpose(-2, -1))
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v_cache)
        if is_cuda:
            torch.cuda.synchronize()
        kv_ms = (time.time() - t0) * 1000.0 / n_measure

        # 2. Causal RLS step
        rls_state = rls_layer.init_state(batch_size=batch_size, device=device)
        for _ in range(n_warmup):
            _, rls_state = rls_layer.step(dummy_input, rls_state)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _, rls_state = rls_layer.step(dummy_input, rls_state)
        if is_cuda:
            torch.cuda.synchronize()
        rls_ms = (time.time() - t0) * 1000.0 / n_measure

        # 3. Diagonal Kalman step
        diag_state = diag_layer.init_state(batch_size=batch_size, device=device)
        for _ in range(n_warmup):
            _, diag_state = diag_layer.step(dummy_input, diag_state)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _, diag_state = diag_layer.step(dummy_input, diag_state)
        if is_cuda:
            torch.cuda.synchronize()
        diag_ms = (time.time() - t0) * 1000.0 / n_measure

        # 4. Chunkwise Ridge step
        chunk_ridge_state = chunk_ridge_layer.init_state(batch_size=batch_size, device=device)
        for _ in range(n_warmup):
            _, chunk_ridge_state = chunk_ridge_layer.step(dummy_input, chunk_ridge_state)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _, chunk_ridge_state = chunk_ridge_layer.step(dummy_input, chunk_ridge_state)
        if is_cuda:
            torch.cuda.synchronize()
        chunk_ridge_ms = (time.time() - t0) * 1000.0 / n_measure

        # 5. Chunkwise Poly step
        chunk_poly_state = chunk_poly_layer.init_state(batch_size=batch_size, device=device)
        for _ in range(n_warmup):
            _, chunk_poly_state = chunk_poly_layer.step(dummy_input, chunk_poly_state)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _, chunk_poly_state = chunk_poly_layer.step(dummy_input, chunk_poly_state)
        if is_cuda:
            torch.cuda.synchronize()
        chunk_poly_ms = (time.time() - t0) * 1000.0 / n_measure

        # Log metrics
        inference_results["memory_mb"]["KV-Cache (Standard Softmax)"].append(round(kv_cache_mem_mb, 4))
        inference_results["memory_mb"]["Causal RLS / Kalman"].append(round(rls_mem_mb, 4))
        inference_results["memory_mb"]["Diagonal Kalman Attention"].append(round(diag_mem_mb, 4))
        inference_results["memory_mb"]["Chunkwise Ridge Attention"].append(round(chunk_ridge_mem_mb, 4))
        inference_results["memory_mb"]["Chunkwise Poly Solvable"].append(round(chunk_poly_mem_mb, 4))

        inference_results["per_token_ms"]["KV-Cache (Standard Softmax)"].append(round(kv_ms, 4))
        inference_results["per_token_ms"]["Causal RLS / Kalman"].append(round(rls_ms, 4))
        inference_results["per_token_ms"]["Diagonal Kalman Attention"].append(round(diag_ms, 4))
        inference_results["per_token_ms"]["Chunkwise Ridge Attention"].append(round(chunk_ridge_ms, 4))
        inference_results["per_token_ms"]["Chunkwise Poly Solvable"].append(round(chunk_poly_ms, 4))

        print(f"T = {T:4d} | KV-Cache: {kv_cache_mem_mb:6.3f} MB ({kv_ms:5.2f} ms) | DiagKalman: {diag_mem_mb:6.3f} MB ({diag_ms:5.2f} ms) | Ridge: {chunk_ridge_mem_mb:6.3f} MB ({chunk_ridge_ms:5.2f} ms)")

    return inference_results


def run_solvable_suite(device: torch.device) -> dict:
    """Executes complete Phase 4 / EXP-05 Solvable Attention suite."""
    mqar = benchmark_mqar_suite(device=device)
    inference = benchmark_autoregressive_inference_suite(device=device)
    return {"mqar": mqar, "inference": inference}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Solvable Attention Suite Benchmark on {device}...")
    results = run_solvable_suite(device)

    out_dir = os.path.join(os.path.dirname(__file__), "../results/exp05_solvable_attention_suite")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSolvable Attention suite completed! Results saved to {out_path}")


if __name__ == "__main__":
    main()
