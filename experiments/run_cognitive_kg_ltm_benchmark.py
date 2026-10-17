"""
EXP-25 Benchmark: Cognitive Knowledge-Graph Long-Term Memory (CKG-LTM).
======================================================================
Tests the cognitive memory consolidation hypothesis:
1. "Surprise" (predictive innovation error) flags candidate salient entities in working memory.
2. Top-down goal alignment and repetition/rehearsal gate what is consolidated into LTM.
3. LTM stores a structured Knowledge Graph of entities and solvable relational operators.

Compared against:
- 2-Layer Softmax Attention (O(T) KV-cache storing all distractors, suffers entropy diffusion)
- Pure Recurrent Attention (Linear/RLS without consolidation gating, suffers noise saturation)
- Naive Surprise-Only Memory (corrupted by one-off high-entropy noise)
- Cognitive KG-LTM (Surprise + Top-Down & Repetition Gating + Closed-Form Solvable KG)
"""

import os
import json
import math
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cognitive_kg_ltm import (
    SurpriseWorkingMemory,
    ConsolidationGate,
    StructuredKnowledgeGraphLTM,
    CognitiveKnowledgeGraphAttention,
)


def generate_noisy_relational_dataset(
    num_samples: int = 64,
    seq_len: int = 1024,
    d_model: int = 128,
    num_chains: int = 4,
    num_hops: int = 2,
    noise_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates synthetic long-context relational reasoning sequences with distractor noise:
    1. Multi-hop chains: A_i -> B_i -> C_i (2 hops)
    2. True factual entities appear with repetition/rehearsal across the sequence (3 occurrences).
    3. Distractor noise: Random high-entropy vectors injected at random positions that appear only ONCE.
    4. Query: Suffix query specifies start entity A_q; target is final entity C_q.
    5. Goal: Top-down vector specifying task context.
    """
    gen = torch.Generator().manual_seed(seed)
    
    X = torch.randn(num_samples, seq_len, d_model, generator=gen) * 0.05  # background context
    queries = torch.zeros(num_samples, d_model)
    targets = torch.zeros(num_samples, d_model)
    goals = torch.zeros(num_samples, d_model)
    
    # Ground truth relation operators: R_1, R_2 [D, D]
    W_r1 = torch.linalg.qr(torch.randn(d_model, d_model, generator=gen))[0]
    W_r2 = torch.linalg.qr(torch.randn(d_model, d_model, generator=gen))[0]
    
    for b in range(num_samples):
        # Sample distinct base entities A_i [num_chains, D]
        A_entities = F.normalize(torch.randn(num_chains, d_model, generator=gen), p=2, dim=-1)
        B_entities = F.normalize(torch.matmul(A_entities, W_r1), p=2, dim=-1)
        C_entities = F.normalize(torch.matmul(B_entities, W_r2), p=2, dim=-1)
        
        # Inject reinforced factual pairs at multiple disjoint windows across the sequence
        avail_pos = torch.randperm(seq_len - 32, generator=gen).tolist()
        pos_idx = 0
        
        for rep in range(3):
            for c in range(num_chains):
                # Pair A -> B at consecutive positions
                pA = avail_pos[pos_idx]
                pB = pA + 1
                X[b, pA] = A_entities[c]
                X[b, pB] = B_entities[c]
                
                # Pair B -> C at consecutive positions
                pB2 = avail_pos[pos_idx + 2]
                pC = pB2 + 1
                X[b, pB2] = B_entities[c]
                X[b, pC] = C_entities[c]
                pos_idx += 4
                
        # Inject one-off high-surprise distractor noise (appears only once, never repeated)
        num_distractors = int(seq_len * noise_ratio)
        distractor_pos = avail_pos[pos_idx : pos_idx + num_distractors]
        for dp in distractor_pos:
            # Distractors have large magnitude to create high initial surprise!
            X[b, dp] = F.normalize(torch.randn(d_model, generator=gen), p=2, dim=-1) * 2.0
            
        # Target chain for retrieval: chain 0
        target_chain_idx = 0
        queries[b] = A_entities[target_chain_idx]  # Query prompt is start entity A_0
        targets[b] = C_entities[target_chain_idx]  # Ground truth is C_0
        goals[b] = A_entities[target_chain_idx]    # Goal aligned with entity
        
    return X, queries, targets, goals


class TwoLayerSoftmaxBaseline(nn.Module):
    """
    2-Layer Softmax Attention Baseline:
    Keys: K = X[:, :-1, :], Values: V = X[:, 1:, :]
    Layer 1: Query A -> Intermediate B
    Layer 2: Intermediate B -> Final C
    """
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def _attn_step(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, D = k.shape
        H, d_k = self.n_heads, self.d_k
        q_h = q.view(B, 1, H, d_k).transpose(1, 2)
        k_h = k.view(B, T, H, d_k).transpose(1, 2)
        v_h = v.view(B, T, H, d_k).transpose(1, 2)
        
        scores = torch.einsum("bhqd,bhtd->bhqt", q_h, k_h) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqt,bhtd->bhqd", attn, v_h).transpose(1, 2).contiguous().view(B, D)
        return out

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        k = x[:, :-1, :]
        v = x[:, 1:, :]
        # Layer 1: Query A -> Intermediate B
        hop1 = self._attn_step(query, k, v)
        # Layer 2: Intermediate B -> Final C
        hop2 = self._attn_step(hop1, k, v)
        return F.normalize(hop2, p=2, dim=-1)


class PureRecurrentBaseline(nn.Module):
    """
    Pure Recurrent / Linear RLS Attention without Consolidation Gating.
    Ingests all tokens (including 50% one-off distractors) into the Gramian.
    """
    def __init__(self, d_model: int, reg: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.reg = reg

    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # Unguided solve across all adjacent pairs in sequence: t -> t+1
        S = x[:, :-1, :].reshape(-1, D)
        O = x[:, 1:, :].reshape(-1, D)
        
        G = torch.matmul(S.t(), S) + self.reg * torch.eye(D, device=x.device, dtype=x.dtype)
        R = torch.matmul(S.t(), O)
        W_all = torch.matmul(torch.linalg.pinv(G.contiguous()), R.contiguous())
        
        # 2-hop retrieval using unguided operator: q * W * W
        curr = F.normalize(query, p=2, dim=-1)
        curr = torch.matmul(curr, W_all)
        curr = F.normalize(curr, p=2, dim=-1)
        curr = torch.matmul(curr, W_all)
        return F.normalize(curr, p=2, dim=-1)


class NaiveSurpriseOnlyBaseline(nn.Module):
    """
    Admits tokens purely on Surprise without top-down or repetition gating.
    Gets poisoned by high-entropy one-off distractor noise.
    """
    def __init__(self, d_model: int, surprise_thresh: float = 0.4, reg: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.reg = reg
        self.wm = SurpriseWorkingMemory(d_model, surprise_threshold=surprise_thresh)

    def forward(self, x: torch.Tensor, query: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, float]:
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        surprise, candidate_mask = self.wm(x)
        
        goal_norm = F.normalize(goal, p=2, dim=-1)
        hop1_subjects, hop1_objects = [], []
        hop2_subjects, hop2_objects = [], []
        
        for b in range(B):
            indices = torch.nonzero(candidate_mask[b], as_tuple=True)[0]
            if len(indices) < 2:
                continue
            for i in range(len(indices) - 1):
                p1 = indices[i].item()
                p2 = indices[i + 1].item()
                if p2 - p1 == 1:
                    v1 = x[b, p1]
                    v2 = x[b, p2]
                    sim_to_goal = torch.dot(F.normalize(v1, p=2, dim=-1), goal_norm[b]).item()
                    if sim_to_goal > 0.4:
                        hop1_subjects.append(v1)
                        hop1_objects.append(v2)
                    else:
                        hop2_subjects.append(v1)
                        hop2_objects.append(v2)
                        
        if len(hop1_subjects) >= 2:
            S1 = torch.stack(hop1_subjects, dim=0)
            O1 = torch.stack(hop1_objects, dim=0)
            G1 = torch.matmul(S1.t(), S1) + self.reg * torch.eye(D, device=device, dtype=dtype)
            R1 = torch.matmul(S1.t(), O1)
            W1 = torch.matmul(torch.linalg.pinv(G1.contiguous()), R1.contiguous())
        else:
            W1 = torch.eye(D, device=device, dtype=dtype)
            
        if len(hop2_subjects) >= 2:
            S2 = torch.stack(hop2_subjects, dim=0)
            O2 = torch.stack(hop2_objects, dim=0)
            G2 = torch.matmul(S2.t(), S2) + self.reg * torch.eye(D, device=device, dtype=dtype)
            R2 = torch.matmul(S2.t(), O2)
            W2 = torch.matmul(torch.linalg.pinv(G2.contiguous()), R2.contiguous())
        else:
            W2 = torch.eye(D, device=device, dtype=dtype)
            
        q_norm = F.normalize(query, p=2, dim=-1)
        hop1 = F.normalize(torch.matmul(q_norm, W1), p=2, dim=-1)
        hop2 = F.normalize(torch.matmul(hop1, W2), p=2, dim=-1)
        
        noise_rejection = 1.0 - candidate_mask.float().mean().item()
        return hop2, noise_rejection


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running EXP-25: Cognitive KG-LTM Benchmark on device: {device}")
    
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
            # Memory: 2 relation operators [D, D]
            mem_ckg_kb = (2 * d_model * d_model * 4) / 1024.0
            metrics["models"]["cognitive_kg_ltm"]["accuracy"].append(round(acc_ckg, 2))
            metrics["models"]["cognitive_kg_ltm"]["memory_kb"].append(round(mem_ckg_kb, 1))
            metrics["models"]["cognitive_kg_ltm"]["noise_rejection"].append(round(diag["noise_rejection_rate"] * 100.0, 1))
            
        print(f"Results at T={T}:")
        print(f"  2-Layer Softmax       : Acc = {acc_sm:.2f}%, Cache Memory = {mem_sm_kb:.1f} KB")
        print(f"  Pure Recurrent (RLS)  : Acc = {acc_rec:.2f}%, State Memory = {mem_rec_kb:.1f} KB")
        print(f"  Naive Surprise-Only   : Acc = {acc_naive:.2f}%, Noise Rejection = {rej_naive*100:.1f}%, State = {mem_naive_kb:.1f} KB")
        print(f"  Cognitive KG-LTM      : Acc = {acc_ckg:.2f}%, Noise Rejection = {diag['noise_rejection_rate']*100:.1f}%, Memory = {mem_ckg_kb:.1f} KB")
        
    out_dir = "/home/kdrana/Projects/OLSAttentionExploration/results/exp25_cognitive_kg_ltm"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] Saved metrics to {out_dir}/metrics.json")
    
    # Plotting
    # Figure 67
    plt.figure(figsize=(8, 5))
    for model_name, label, color, marker in [
        ("cognitive_kg_ltm", "Cognitive KG-LTM (Surprise + Repetition + Top-Down)", "#10B981", "o"),
        ("softmax", "2-Layer Softmax Attention (Linear KV-Cache)", "#EF4444", "s"),
        ("naive_surprise", "Naive Surprise-Only (No Repetition Gate)", "#F59E0B", "x"),
        ("pure_recurrent", "Pure Recurrent / RLS (Unguided State)", "#3B82F6", "^"),
    ]:
        plt.plot(seq_lens, metrics["models"][model_name]["accuracy"], label=label, color=color, marker=marker, linewidth=2.2, markersize=8)
    plt.title("EXP-25: Multi-Hop Relational Retrieval Accuracy vs Sequence Length", fontsize=12, fontweight="bold")
    plt.xlabel("Context Sequence Length (T tokens)", fontsize=11)
    plt.ylabel("Multi-Hop Retrieval Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig67_kg_ltm_multihop_accuracy.png"), dpi=300)
    plt.close()
    
    # Figure 68
    plt.figure(figsize=(7, 5))
    models = ["Softmax\n(All Cached)", "Pure RLS\n(All Ingested)", "Naive Surprise\n(Noise Poisoned)", "Cognitive KG-LTM\n(Consolidated)"]
    rejections = [
        0.0,
        0.0,
        metrics["models"]["naive_surprise"]["noise_rejection"][-1],
        metrics["models"]["cognitive_kg_ltm"]["noise_rejection"][-1],
    ]
    colors = ["#EF4444", "#3B82F6", "#F59E0B", "#10B981"]
    bars = plt.bar(models, rejections, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.title("EXP-25: Distractor Noise Rejection Rate from Long-Term Memory", fontsize=12, fontweight="bold")
    plt.ylabel("One-Off Noise Rejection Rate (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig68_kg_ltm_noise_rejection.png"), dpi=300)
    plt.close()
    
    # Figure 69
    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, metrics["models"]["softmax"]["memory_kb"], label="Softmax KV-Cache: O(T) Explosion", color="#EF4444", marker="s", linewidth=2.5)
    plt.plot(seq_lens, metrics["models"]["cognitive_kg_ltm"]["memory_kb"], label="Cognitive KG-LTM: O(1) Invariant State", color="#10B981", marker="o", linewidth=2.5, linestyle="--")
    plt.title("EXP-25: Serving State Memory Footprint vs Sequence Length", fontsize=12, fontweight="bold")
    plt.xlabel("Context Sequence Length (T tokens)", fontsize=11)
    plt.ylabel("State Memory Footprint (KB)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig69_kg_ltm_memory_scaling.png"), dpi=300)
    plt.close()
    
    print("[+] All figures generated successfully.")


if __name__ == "__main__":
    run_benchmark()
