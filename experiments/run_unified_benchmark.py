"""
Unified Direct Mathematical Solution vs Pieces vs Traditional Attention (EXP-06):
Confronts three paradigms on Multi-Query Associative Recall (MQAR) & Inference Scaling:
1. Traditional Architecture:
   - Standard Softmax Attention (Classical O(T^2) quadratic attention + O(T) KV-cache)
2. Solutions in Pieces (Recurrent / Chunked Approximations):
   - Delta / LMS Attention (Innovation error with scalar step size)
   - Causal RLS / Kalman Attention (Sequential rank-1 Riccati precision updates)
   - Chunkwise Poly Solvable (Chunk-partitioned dual Ridge solves)
3. The Unified Direct Mathematical Solution:
   - Exact Causal OLS Attention (Unified cumulative Gram matrix G_t + parallel batched Cholesky solve)
   - Exact Causal Poly OLS Attention (Exact cumulative Gram matrix with degree-2 feature mapping)
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
    ChunkwisePolySolvableAttention,
    ExactCausalOLSAttention,
)
from src.tasks import generate_associative_recall_data


def benchmark_mqar_unified(
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
    """Evaluates learning accuracy on MQAR across Traditional, Pieces, and Unified OLS."""
    print(f"\n[EXP-06 MQAR Benchmark] T={seq_len}, Vocab={vocab_size}, Pairs={num_pairs}, Queries={num_queries}")

    # Generate fixed evaluation dataset
    X_test_list, Y_test_list = [], []
    for i in range(n_test_batches):
        x, y = generate_associative_recall_data(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_pairs=num_pairs,
            num_queries=num_queries,
            seed=3000 + i,
        )
        X_test_list.append(x)
        Y_test_list.append(y)

    X_test = torch.cat(X_test_list, dim=0).to(device)
    Y_test = torch.cat(Y_test_list, dim=0).to(device)

    models_to_test = [
        ("Traditional (Standard Softmax)", "standard", "Traditional"),
        ("Pieces: Delta / LMS Attention", "delta", "Pieces"),
        ("Pieces: Causal RLS / Kalman", "rls", "Pieces"),
        ("Pieces: Chunkwise Poly Solvable", "chunk_poly", "Pieces"),
        ("Unified: Exact Causal OLS", "exact_ols", "Unified"),
        ("Unified: Exact Poly OLS", "exact_poly_ols", "Unified"),
    ]

    mqar_results = {}

    for name, attn_type, category in models_to_test:
        print(f"\n--- Training {name} ({attn_type}) [{category}] ---")
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

        history = {"category": category, "epochs": [], "train_loss": [], "test_acc": [], "wallclock_sec": []}
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

            # Evaluate test accuracy
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


def benchmark_inference_unified(
    device: torch.device,
    seq_lengths: list[int] = [256, 512, 1024, 2048, 4096],
    d_model: int = 64,
    n_heads: int = 4,
    batch_size: int = 1,
) -> dict:
    """
    Simulates token-by-token autoregressive generation comparing:
    - Traditional KV-cache (O(T) memory explosion)
    - Pieces: Causal RLS (O(1) Riccati state)
    - Pieces: Chunkwise Poly (O(1) buffer + state)
    - Unified: Exact Causal OLS (O(1) Gram G_t + cross-cov R_t state)
    - Unified: Exact Poly OLS (O(1) Gram G_t + cross-cov R_t state)
    """
    print("\n[EXP-06 Inference Benchmark] Evaluating token generation across sequence lengths...")
    is_cuda = device.type == "cuda"
    d_head = d_model // n_heads

    model_names = [
        "Traditional (KV-Cache)",
        "Pieces: Causal RLS",
        "Pieces: Chunkwise Poly",
        "Unified: Exact Causal OLS",
        "Unified: Exact Poly OLS",
    ]

    inference_results = {
        "seq_lengths": seq_lengths,
        "memory_mb": {name: [] for name in model_names},
        "per_token_ms": {name: [] for name in model_names},
    }

    # Module instantiations
    rls_layer = CausalRLSAttention(d_model=d_model, n_heads=n_heads).to(device)
    poly_chunk_layer = ChunkwisePolySolvableAttention(d_model=d_model, n_heads=n_heads, chunk_size=16).to(device)
    ols_layer = ExactCausalOLSAttention(d_model=d_model, n_heads=n_heads, use_poly=False).to(device)
    poly_ols_layer = ExactCausalOLSAttention(d_model=d_model, n_heads=n_heads, use_poly=True).to(device)

    # Theoretical memory calculations for 2-layer sequence models:
    # 1. KV-Cache: 2 * 2 * B * H * T * d_head * 4 bytes
    # 2. RLS: 2 * (S (d*d) + P (d*d)) * B * H * 4 bytes = 4 * B * H * d * d * 4 bytes
    # 3. Chunk Poly: 2 * (S (dp*d) + 2*C*dp) * B * H * 4 bytes
    # 4. Exact OLS: 2 * (G (d*d) + R (d*d)) * B * H * 4 bytes = 4 * B * H * d * d * 4 bytes
    # 5. Exact Poly OLS: 2 * (G (2d*2d) + R (2d*d)) * B * H * 4 bytes
    d_poly = poly_chunk_layer.d_poly
    chunk_size = 16

    rls_mem_mb = (4 * batch_size * n_heads * d_head * d_head * 4) / (1024 * 1024)
    poly_chunk_mem_mb = (2 * batch_size * n_heads * (d_poly * d_head + 2 * chunk_size * d_poly) * 4) / (1024 * 1024)
    ols_mem_mb = (4 * batch_size * n_heads * d_head * d_head * 4) / (1024 * 1024)
    poly_ols_mem_mb = (2 * batch_size * n_heads * (4 * d_head * d_head + 2 * d_head * d_head) * 4) / (1024 * 1024)

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

        # 2. Causal RLS
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

        # 3. Pieces: Chunkwise Poly
        chunk_poly_state = poly_chunk_layer.init_state(batch_size=batch_size, device=device)
        for _ in range(n_warmup):
            _, chunk_poly_state = poly_chunk_layer.step(dummy_input, chunk_poly_state)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _, chunk_poly_state = poly_chunk_layer.step(dummy_input, chunk_poly_state)
        if is_cuda:
            torch.cuda.synchronize()
        chunk_poly_ms = (time.time() - t0) * 1000.0 / n_measure

        # 4. Unified: Exact Causal OLS
        ols_state = ols_layer.init_state(batch_size=batch_size, device=device)
        for _ in range(n_warmup):
            _, ols_state = ols_layer.step(dummy_input, ols_state)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _, ols_state = ols_layer.step(dummy_input, ols_state)
        if is_cuda:
            torch.cuda.synchronize()
        ols_ms = (time.time() - t0) * 1000.0 / n_measure

        # 5. Unified: Exact Poly OLS
        poly_ols_state = poly_ols_layer.init_state(batch_size=batch_size, device=device)
        for _ in range(n_warmup):
            _, poly_ols_state = poly_ols_layer.step(dummy_input, poly_ols_state)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_measure):
            _, poly_ols_state = poly_ols_layer.step(dummy_input, poly_ols_state)
        if is_cuda:
            torch.cuda.synchronize()
        poly_ols_ms = (time.time() - t0) * 1000.0 / n_measure

        inference_results["memory_mb"]["Traditional (KV-Cache)"].append(round(kv_cache_mem_mb, 4))
        inference_results["memory_mb"]["Pieces: Causal RLS"].append(round(rls_mem_mb, 4))
        inference_results["memory_mb"]["Pieces: Chunkwise Poly"].append(round(poly_chunk_mem_mb, 4))
        inference_results["memory_mb"]["Unified: Exact Causal OLS"].append(round(ols_mem_mb, 4))
        inference_results["memory_mb"]["Unified: Exact Poly OLS"].append(round(poly_ols_mem_mb, 4))

        inference_results["per_token_ms"]["Traditional (KV-Cache)"].append(round(kv_ms, 4))
        inference_results["per_token_ms"]["Pieces: Causal RLS"].append(round(rls_ms, 4))
        inference_results["per_token_ms"]["Pieces: Chunkwise Poly"].append(round(chunk_poly_ms, 4))
        inference_results["per_token_ms"]["Unified: Exact Causal OLS"].append(round(ols_ms, 4))
        inference_results["per_token_ms"]["Unified: Exact Poly OLS"].append(round(poly_ols_ms, 4))

        print(f"T = {T:4d} | KV-Cache: {kv_cache_mem_mb:6.3f} MB | Unified OLS: {ols_mem_mb:6.3f} MB | Unified Poly: {poly_ols_mem_mb:6.3f} MB (O(1) Flat)")

    return inference_results


def run_unified_suite(device: torch.device) -> dict:
    """Executes complete EXP-06 Unified vs Pieces benchmark."""
    mqar = benchmark_mqar_unified(device=device)
    inference = benchmark_inference_unified(device=device)
    return {"mqar": mqar, "inference": inference}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Unified vs Pieces Benchmark on {device}...")
    results = run_unified_suite(device)

    out_dir = os.path.join(os.path.dirname(__file__), "../results/exp06_unified_vs_pieces")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUnified benchmark completed! Results saved to {out_path}")


if __name__ == "__main__":
    main()
