"""
Multi-Hop Associative Reasoning in Series Benchmark (Phase 7 / EXP-08):
Confronts 1-Layer vs 2-Layer vs 4-Layer Transformer architectures on 2-Hop Associative Recall:
  Prefix Facts: A_i -> B_i  and  B_i -> C_i  (injected at disjoint random positions)
  Test Query:   A_q -> ? (Target: C_q)

Theoretical Hypothesis:
1. 1-Layer models cannot compose two associative hops in a single attention step
   (provably limited to 1-hop lookups; accuracy must fail near random chance).
2. Stacking layers in series (n_layers >= 2) creates a compositional induction chain:
   Layer 1 retrieves intermediate entity B_q; residual bridge enriches Layer 2 query;
   Layer 2 retrieves final target C_q.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SequenceTransformer
from src.tasks import generate_multihop_associative_recall_data


def benchmark_multihop_series(
    device: torch.device,
    seq_len: int = 128,
    vocab_size: int = 128,
    num_chains: int = 4,
    num_queries: int = 2,
    num_hops: int = 2,
    n_epochs: int = 12,
    batch_size: int = 32,
    n_train_batches: int = 10,
    n_test_batches: int = 4,
) -> dict:
    """Evaluates compositional 2-hop recall across varying network depths and attention mechanisms."""
    print(f"\n[EXP-08 Multi-Hop Series Benchmark] T={seq_len}, Vocab={vocab_size}, Chains={num_chains}, Queries={num_queries}, Hops={num_hops}")

    # Generate fixed evaluation dataset
    X_test_list, Y_test_list = [], []
    for i in range(n_test_batches):
        x, y = generate_multihop_associative_recall_data(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_chains=num_chains,
            num_queries=num_queries,
            num_hops=num_hops,
            seed=5000 + i,
        )
        X_test_list.append(x)
        Y_test_list.append(y)

    X_test = torch.cat(X_test_list, dim=0).to(device)
    Y_test = torch.cat(Y_test_list, dim=0).to(device)

    # Models to test across 1-Layer vs 2-Layer vs 4-Layer depth
    models_to_test = [
        ("Standard Softmax (1-Layer)", "standard", 1),
        ("Standard Softmax (2-Layers)", "standard", 2),
        ("Delta-Net (1-Layer)", "delta", 1),
        ("Delta-Net (2-Layers)", "delta", 2),
        ("Flash-OLS Linear (1-Layer)", "flash_ols", 1),
        ("Flash-OLS Linear (2-Layers)", "flash_ols", 2),
        ("Flash-OLS Poly (1-Layer)", "flash_poly_ols", 1),
        ("Flash-OLS Poly (2-Layers)", "flash_poly_ols", 2),
        ("Flash-OLS Poly (4-Layers)", "flash_poly_ols", 4),
    ]

    multihop_results = {}

    for name, attn_type, n_layers in models_to_test:
        print(f"\n--- Training {name} (type={attn_type}, layers={n_layers}) ---")
        torch.manual_seed(42)
        model = SequenceTransformer(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=n_layers,
            d_ffn=128,
            max_seq_len=max(seq_len + 64, 4096),
            attn_type=attn_type,
            chunk_size=32,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        history = {
            "n_layers": n_layers,
            "attn_type": attn_type,
            "epochs": [],
            "train_loss": [],
            "test_acc": [],
            "wallclock_sec": [],
        }
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for b in range(n_train_batches):
                X_batch, Y_batch = generate_multihop_associative_recall_data(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    num_chains=num_chains,
                    num_queries=num_queries,
                    num_hops=num_hops,
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

            # Evaluate test accuracy on 2-hop target tokens
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

            if epoch % 4 == 0 or epoch == n_epochs:
                print(f"Epoch {epoch:2d}/{n_epochs} | Loss: {epoch_loss:.4f} | 2-Hop Accuracy: {test_acc * 100:.2f}% | Elapsed: {elapsed:.2f}s")

        multihop_results[name] = history

    return multihop_results


def run_multihop_suite(device: torch.device) -> dict:
    """Executes full EXP-08 Multi-Hop Series benchmark."""
    multihop = benchmark_multihop_series(device=device)
    return {"multihop": multihop}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Multi-Hop Series Benchmark on {device}...")
    results = run_multihop_suite(device)

    out_dir = os.path.join(os.path.dirname(__file__), "../results/exp08_multihop_series")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMulti-Hop Series benchmark completed! Results saved to {out_path}")


if __name__ == "__main__":
    main()
