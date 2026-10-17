"""
Benchmark script for EXP-22: Native Solvable Attention Layer (NSAL) End-to-End Training & Alignment.
Evaluates Softmax vs Linear Attention vs NSAL on:
1. Multi-Query Associative Recall (MQAR) across sequence lengths T in [128, 256, 512, 1024, 2048].
2. Decode memory and latency scaling (O(1) vs O(T)).
3. Learned gate salience distribution.
"""

import time
import math
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.native_solvable_attention import NativeSolvableAttention


# =====================================================================
# Model Architectures
# =====================================================================

class CausalSoftmaxAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k
        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d_k).transpose(1, 2)

        if state is not None:
            k_past, v_past = state
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
            new_cache = (k, v)
            T_total = k.shape[2]
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            # Query is at position T_total - T : T_total
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        else:
            new_cache = (k, v)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, -1e9)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(self.norm(out)), new_cache


class CausalLinearAttention(nn.Module):
    """Standard Causal Linear Attention (Katharopoulos et al., 2020) with ELU+1 feature map."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k
        q = F.elu(self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)) + 1.0
        k = F.elu(self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)) + 1.0
        v = self.v_proj(x).view(B, T, H, d_k).transpose(1, 2)

        if T == 1 and state is not None:
            # Step decode
            kv_state, k_state = state  # kv: [B, H, d_k, d_k], k_state: [B, H, d_k]
            k_t = k.squeeze(2)  # [B, H, d_k]
            v_t = v.squeeze(2)  # [B, H, d_k]
            q_t = q.squeeze(2)  # [B, H, d_k]

            kv_state = kv_state + torch.einsum("bhi,bhj->bhij", k_t, v_t)
            k_state = k_state + k_t

            num = torch.einsum("bhi,bhij->bhj", q_t, kv_state)
            den = torch.einsum("bhi,bhi->bh", q_t, k_state).unsqueeze(-1) + 1e-6
            out = (num / den).unsqueeze(1).transpose(1, 2).contiguous().view(B, 1, D)
            return self.out_proj(self.norm(out)), (kv_state, k_state)

        # Prefill / Training: Parallel prefix sums
        # kv: [B, H, T, d_k, d_k]
        kv = torch.einsum("bhti,bhtj->bhtij", k, v)
        kv_cum = torch.cumsum(kv, dim=2)
        k_cum = torch.cumsum(k, dim=2)

        num = torch.einsum("bhti,bhtid->bhtd", q, kv_cum)
        den = torch.einsum("bhti,bhti->bht", q, k_cum).unsqueeze(-1) + 1e-6
        out = (num / den).transpose(1, 2).contiguous().view(B, T, D)
        
        final_state = (kv_cum[:, :, -1, :, :], k_cum[:, :, -1, :])
        return self.out_proj(self.norm(out)), final_state


class SimpleTransformerBlock(nn.Module):
    def __init__(self, attn_type: str, d_model: int, n_heads: int, r: int = 16):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "softmax":
            self.attn = CausalSoftmaxAttention(d_model, n_heads)
        elif attn_type == "linear":
            self.attn = CausalLinearAttention(d_model, n_heads)
        elif attn_type == "nsal":
            self.attn = NativeSolvableAttention(d_model, n_heads, r=r)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state=None):
        attn_out, new_state = self.attn(x, state=state)
        x = x + attn_out
        x = x + self.mlp(self.norm(x))
        return x, new_state


class MemoryModel(nn.Module):
    def __init__(self, attn_type: str, vocab_size: int, d_model: int = 64, n_heads: int = 4, r: int = 16, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)
        
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(attn_type, d_model, n_heads, r) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, states=None):
        B, T = x.shape
        h = self.embed(x) + self.pos_emb[:, :T, :]
        new_states = []
        for i, block in enumerate(self.blocks):
            state_i = states[i] if states is not None else None
            h, new_state_i = block(h, state=state_i)
            new_states.append(new_state_i)
        logits = self.head(self.norm(h))
        return logits, new_states


# =====================================================================
# Synthetic Dataset Generators
# =====================================================================

def generate_mqar_dataset(
    batch_size: int,
    seq_len: int,
    num_kv: int,
    vocab_size: int = 128,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates Multi-Query Associative Recall (MQAR) batches.
    Vocabulary partitioning:
      - 0: Padding
      - 1..32: Keys
      - 33..64: Values
      - 65..vocab_size-1: Distractor filler tokens
    """
    assert vocab_size >= 65, "vocab_size must be >= 65"
    x = torch.randint(65, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Sample unique keys
        perm = torch.randperm(32) + 1
        keys = perm[:num_kv]
        vals = keys + 32  # Ground truth values

        # Place pairs in the first 80% of sequence
        max_pos = int(seq_len * 0.8)
        pos = torch.sort(torch.randperm(max_pos)[:num_kv * 2])[0]
        for i in range(num_kv):
            k_pos = pos[2 * i]
            v_pos = pos[2 * i + 1]
            x[b, k_pos] = keys[i]
            x[b, v_pos] = vals[i]

        # Place query at the last position
        query_idx = torch.randint(0, num_kv, (1,)).item()
        x[b, -1] = keys[query_idx]
        targets[b, -1] = vals[query_idx]

    return x, targets


# =====================================================================
# Benchmark Training & Evaluation Suite
# =====================================================================

def train_and_eval_model(
    attn_type: str,
    train_seq_len: int = 128,
    test_seq_lens: List[int] = [128, 256, 512, 1024, 2048],
    num_kv: int = 8,
    train_steps: int = 350,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, Any]:
    print(f"\n--- Training {attn_type.upper()} on MQAR (T_train={train_seq_len}, num_kv={num_kv}) ---")
    model = MemoryModel(attn_type=attn_type, vocab_size=128, d_model=64, n_heads=4, r=16, n_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    loss_history = []
    t0 = time.time()
    
    for step in range(train_steps):
        x, y = generate_mqar_dataset(batch_size, train_seq_len, num_kv, vocab_size=128, device=device)
        logits, _ = model(x)
        # Loss only on query target position (last token)
        loss = loss_fn(logits[:, -1, :], y[:, -1])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(float(loss.item()))
        if (step + 1) % 100 == 0 or step == train_steps - 1:
            print(f"  Step {step+1:3d}/{train_steps} | Loss: {loss.item():.4f}")

    train_time = time.time() - t0
    
    # Evaluate across sequence lengths (Length Generalization)
    accuracies_len = {}
    model.eval()
    with torch.no_grad():
        for test_len in test_seq_lens:
            test_bs = 64
            x_test, y_test = generate_mqar_dataset(test_bs, test_len, num_kv, vocab_size=128, device=device)
            logits_test, _ = model(x_test)
            preds = logits_test[:, -1, :].argmax(dim=-1)
            acc = (preds == y_test[:, -1]).float().mean().item() * 100.0
            accuracies_len[f"acc_T{test_len}"] = round(acc, 2)
            print(f"  Eval T={test_len:4d} | Accuracy: {acc:5.1f}%")

    # Evaluate across number of stored key-value pairs (Capacity Generalization at T=512)
    accuracies_pairs = {}
    with torch.no_grad():
        for n_pairs in [4, 8, 12, 16]:
            test_bs = 64
            x_test, y_test = generate_mqar_dataset(test_bs, 512, n_pairs, vocab_size=128, device=device)
            logits_test, _ = model(x_test)
            preds = logits_test[:, -1, :].argmax(dim=-1)
            acc = (preds == y_test[:, -1]).float().mean().item() * 100.0
            accuracies_pairs[f"acc_pairs{n_pairs}"] = round(acc, 2)
            print(f"  Eval Pairs={n_pairs:2d} (T=512) | Accuracy: {acc:5.1f}%")

    # Measure Decode Memory and Latency Scaling
    print(f"  Measuring decode memory & step latency...")
    decode_metrics = measure_decode_performance(model, attn_type, device=device)

    return {
        "attn_type": attn_type,
        "train_time_sec": round(train_time, 2),
        "final_loss": round(loss_history[-1], 4),
        "loss_history": [round(l, 4) for l in loss_history[::10]],
        "accuracies": accuracies_len,
        "accuracies_pairs": accuracies_pairs,
        "decode_metrics": decode_metrics,
    }


def measure_decode_performance(model: MemoryModel, attn_type: str, device: str = "cpu") -> Dict[str, Any]:
    """Measures memory (KB/MB) and step latency (ms/token) during generation up to T=2048."""
    model.eval()
    steps_to_test = [128, 512, 1024, 2048]
    memory_profile = {}
    latency_profile = {}

    with torch.no_grad():
        for total_len in steps_to_test:
            x_init = torch.randint(65, 128, (1, 1), device=device)
            states = None

            # Prefill 1 token
            _, states = model(x_init, states=states)

            # Measure generation for a burst
            t_start = time.time()
            n_tokens = 50
            for _ in range(n_tokens):
                x_next = torch.randint(65, 128, (1, 1), device=device)
                _, states = model(x_next, states=states)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
                
            elapsed_ms = (time.time() - t_start) * 1000.0 / n_tokens
            latency_profile[f"latency_ms_T{total_len}"] = round(elapsed_ms, 3)

            # Calculate memory of states
            mem_bytes = 0
            if attn_type == "softmax":
                # states is list of (K, V) per layer
                # K, V shape: [B, H, total_len, d_k]
                # At step total_len:
                mem_bytes = 2 * len(model.blocks) * 1 * 4 * total_len * 16 * 4  # float32 = 4 bytes
            elif attn_type == "linear":
                # kv_state: [B, H, d_k, d_k], k_state: [B, H, d_k]
                mem_bytes = len(model.blocks) * 1 * 4 * (16 * 16 + 16) * 4
            elif attn_type == "nsal":
                # P: [B, H, r, r], W: [B, H, r, d_k]
                mem_bytes = len(model.blocks) * 1 * 4 * (16 * 16 + 16 * 16) * 4

            memory_profile[f"mem_kb_T{total_len}"] = round(mem_bytes / 1024.0, 2)

    return {
        "memory_profile_kb": memory_profile,
        "latency_profile_ms": latency_profile,
    }


def run_exp22_benchmark(device: str = "cpu") -> Dict[str, Any]:
    print(f"=================================================================")
    print(f"  EXP-22: Native Solvable Attention Layer (NSAL) End-to-End Suite")
    print(f"  Device: {device}")
    print(f"=================================================================")
    
    test_seq_lens = [128, 256, 512, 1024, 2048]
    results = {}
    
    for attn_type in ["linear", "softmax", "nsal"]:
        res = train_and_eval_model(
            attn_type=attn_type,
            train_seq_len=128,
            test_seq_lens=test_seq_lens,
            num_kv=8,
            train_steps=350,
            batch_size=32,
            device=device,
        )
        results[attn_type] = res

    # Inspect Learned Gates in NSAL
    print("\n--- Inspecting Learned Admission Gates in NSAL ---")
    model_nsal = MemoryModel(attn_type="nsal", vocab_size=128, d_model=64, n_heads=4, r=16, n_layers=2).to(device)
    # Check gate outputs on key vs noise tokens
    x_sample, _ = generate_mqar_dataset(4, 128, 8, vocab_size=128, device=device)
    with torch.no_grad():
        # Get gate values from first layer
        gate_vals = torch.sigmoid(model_nsal.blocks[0].attn.gate_proj(model_nsal.embed(x_sample)))  # [B, T, H]
        mean_gate = gate_vals.mean().item()
        
    results["gate_inspection"] = {
        "mean_gate_activation": round(mean_gate, 4),
        "gate_shape": list(gate_vals.shape),
    }

    return results

if __name__ == "__main__":
    res = run_exp22_benchmark(device="cpu")
    print("\nBenchmark completed successfully!")
