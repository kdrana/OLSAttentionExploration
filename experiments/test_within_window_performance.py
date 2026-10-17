"""
Experiment: Within-Window Softmax vs. Cognitive KG-LTM Benchmark.
================================================================
Evaluates retrieval performance when items are placed within the active memory window W:
W in [16, 32, 64, 128, 256, 512]

Compares:
1. Local Softmax Attention (calibrated temperature tau=0.2)
2. Pure Recurrent RLS
3. Cognitive KG-LTM (Consolidation Gate + Relational Graph)
4. Dual-Control Memory (Local STM Window + KG-LTM)
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cognitive_kg_ltm import (
    SurpriseWorkingMemory,
    ConsolidationGate,
    StructuredKnowledgeGraphLTM,
    CognitiveKnowledgeGraphAttention,
)


def evaluate_within_window(device: str = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Within-Window Memory Benchmark on {device}")
    
    windows = [16, 32, 64, 128, 256, 512]
    d_model = 128
    
    results = {
        "windows": windows,
        "softmax_calibrated": [],
        "pure_recurrent": [],
        "cognitive_kg_ltm": [],
        "dual_control_stm_ltm": [],
    }
    
    gen = torch.Generator().manual_seed(123)
    W_r1 = torch.linalg.qr(torch.randn(d_model, d_model, generator=gen))[0].to(device)
    W_r2 = torch.linalg.qr(torch.randn(d_model, d_model, generator=gen))[0].to(device)
    
    for W in windows:
        num_samples = 64
        # Generate on CPU and move to device
        X = (torch.randn(num_samples, W, d_model, generator=gen) * 0.05).to(device)
        queries = torch.zeros(num_samples, d_model, device=device)
        targets = torch.zeros(num_samples, d_model, device=device)
        goals = torch.zeros(num_samples, d_model, device=device)
        
        for b in range(num_samples):
            # Chains A -> B -> C
            A = F.normalize(torch.randn(1, d_model, generator=gen), p=2, dim=-1).to(device)
            B = F.normalize(torch.matmul(A, W_r1), p=2, dim=-1)
            C = F.normalize(torch.matmul(B, W_r2), p=2, dim=-1)
            
            # Place facts within the window W
            pA = 0
            pB = 1
            pB2 = 2
            pC = 3
            
            X[b, pA] = A
            X[b, pB] = B
            X[b, pB2] = B
            X[b, pC] = C
            
            # If W is larger than 16, repeat the fact to simulate rehearsal
            if W >= 32:
                mid = W // 2
                X[b, mid] = A
                X[b, mid + 1] = B
                X[b, mid + 2] = B
                X[b, mid + 3] = C
                
            # Fill remaining with distractor noise
            distractor_indices = list(range(4, W // 2 if W >= 32 else W))
            for di in distractor_indices:
                X[b, di] = F.normalize(torch.randn(d_model, generator=gen), p=2, dim=-1).to(device) * 1.5
                
            queries[b] = A.squeeze(0)
            targets[b] = C.squeeze(0)
            goals[b] = A.squeeze(0)
            
        # 1. Calibrated Softmax Attention (tau = 0.2)
        # Layer 1: Query A -> B
        k = X[:, :-1, :]
        v = X[:, 1:, :]
        q1 = F.normalize(queries, p=2, dim=-1).unsqueeze(1)
        sim1 = torch.bmm(q1, k.transpose(1, 2)) / 0.2  # calibrated sharp temperature
        attn1 = F.softmax(sim1, dim=-1)
        hop1 = F.normalize(torch.bmm(attn1, v).squeeze(1), p=2, dim=-1)
        
        # Layer 2: Hop 1 (B) -> C
        q2 = hop1.unsqueeze(1)
        sim2 = torch.bmm(q2, k.transpose(1, 2)) / 0.2
        attn2 = F.softmax(sim2, dim=-1)
        hop2 = F.normalize(torch.bmm(attn2, v).squeeze(1), p=2, dim=-1)
        
        acc_sm = (F.cosine_similarity(hop2, targets, dim=-1) > 0.60).float().mean().item() * 100.0
        results["softmax_calibrated"].append(round(acc_sm, 1))
        
        # 2. Pure Recurrent (RLS)
        S = X[:, :-1, :].reshape(-1, d_model)
        O = X[:, 1:, :].reshape(-1, d_model)
        G = torch.matmul(S.t(), S) + 0.05 * torch.eye(d_model, device=device)
        R = torch.matmul(S.t(), O)
        W_all = torch.matmul(torch.linalg.pinv(G.contiguous()), R.contiguous())
        rec_out = F.normalize(torch.matmul(F.normalize(torch.matmul(queries, W_all), p=2, dim=-1), W_all), p=2, dim=-1)
        acc_rec = (F.cosine_similarity(rec_out, targets, dim=-1) > 0.60).float().mean().item() * 100.0
        results["pure_recurrent"].append(round(acc_rec, 1))
        
        # 3. Cognitive KG-LTM
        ckg = CognitiveKnowledgeGraphAttention(d_model, surprise_thresh=0.3, consolidation_thresh=0.4).to(device)
        out_ckg, diag = ckg(X, queries, goal=goals)
        acc_ckg = (F.cosine_similarity(out_ckg, targets, dim=-1) > 0.60).float().mean().item() * 100.0
        results["cognitive_kg_ltm"].append(round(acc_ckg, 1))
        
        # 4. Dual-Control Memory (STM local buffer + LTM Knowledge Graph)
        # In dual control: if query matches STM immediate keys with high confidence (>0.85), trust STM;
        # otherwise query LTM Knowledge Graph!
        conf_sm = torch.max(attn1, dim=-1)[0].squeeze(1)  # [B]
        dual_out = torch.where(conf_sm.unsqueeze(-1) > 0.85, hop2, out_ckg)
        acc_dual = (F.cosine_similarity(dual_out, targets, dim=-1) > 0.60).float().mean().item() * 100.0
        results["dual_control_stm_ltm"].append(round(acc_dual, 1))
        
        print(f"Window W={W:3d} | Softmax (Calibrated): {acc_sm:5.1f}% | Pure Recurrent: {acc_rec:5.1f}% | Cognitive KG-LTM: {acc_ckg:5.1f}% | Dual-Control: {acc_dual:5.1f}%")
        
    return results


if __name__ == "__main__":
    res = evaluate_within_window("cpu")
    print("\nBenchmark Results:", json.dumps(res, indent=2))
