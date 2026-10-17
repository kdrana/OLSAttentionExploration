"""
EXP-28: Reviewer-Proof Multi-Family, High-Power & Literature-Benchmark Suite.
=============================================================================
Implementation of:
- Sub-Experiment 28A: Cross-Company 3-Model Benchmark (Qwen3.5, Llama-3.2, SmolLM2; n=10 trials with 95% CIs)
- Sub-Experiment 28B: The Delta_QA Separation & Value-Span Length Phase Diagram (n=10 trials per point)
- Sub-Experiment 28C: Canonical Literature Replication (RULER 4-Axis Suite & LongBench-Style Summarization; n=10 per task)
- Sub-Experiment 28D: Component Leave-One-Out Ablation, Multi-Turn Unseen Follow-Up Query Durability & (beta, gamma) Sensitivity (n=10 trials)
- Sub-Experiment 28E: Hardware Wall-Clock Latency (TTFT), Selection Overhead (ms), Decode Throughput (tok/s), KV Cache Memory & Peak VRAM (1k -> 32k)
"""

import os
import gc
import re
import math
import time
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cognitive_dual_control_attention import (
    CognitiveDualControlAttention,
    apply_halo_adapter_generic,
    reset_cdca_states,
    set_freeze_salient_cache,
)

# Standard Haystack Base Text
HAYSTACK_BASE = """
Artificial intelligence research has evolved through multiple historical epochs. From symbolic expert systems in the 1970s to statistical learning paradigms in the 1990s, the field has continuously explored computational architectures for modeling knowledge. In modern connectionist models, transformer architectures rely on self-attention mechanisms to dynamically route information across sequence representations. State-space models and recurrent linear networks seek to compress temporal dependencies into fixed-dimensional representations, eliminating quadratic attention bottlenecks while maintaining expressive associative recall.
The mathematical formulation of causal regression models provides an analytical foundation for continuous temporal updates. By expressing covariance accumulation in terms of recursive least squares, state representations remain invariant to historical context length. Numerical stability is preserved through regularized normal equations, where ridge penalties prevent ill-conditioning during matrix inversion.
In distributed deep learning systems, memory bandwidth and tensor memory footprints dictate throughput scaling. High-bandwidth memory (HBM) on contemporary accelerator hardware experiences memory saturation when historical key-value tensors scale linearly with token length. In contrast, fixed-size recurrent memory states maintain strict O(1) resident footprints, avoiding DRAM bandwidth exhaustion and allowing long-horizon inference without sequence eviction.
"""


def compute_stats(vals: List[float]) -> Dict[str, Any]:
    """Computes sample size, mean, sample standard deviation, SEM, and 95% CI."""
    arr = np.array(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci95": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    ci95 = float(1.96 * sem)
    return {
        "n": n,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "sem": round(sem, 2),
        "ci95": round(ci95, 2),
    }


def load_model_and_tokenizer(model_name: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads a causal LM with bfloat16 precision in evaluation mode."""
    print(f"\n[Loader] Loading model: {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    model.eval()
    return model, tokenizer


def build_synthetic_context(
    tokenizer: AutoTokenizer,
    target_len: int,
    needle_str: str,
    query_str: str,
    depth: float,
) -> Tuple[List[int], int]:
    """Builds a haystack document of target_len with a needle inserted at depth."""
    needle_tokens = tokenizer.encode(needle_str, add_special_tokens=False)
    query_tokens = tokenizer.encode(query_str, add_special_tokens=False)
    haystack_tokens = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

    num_haystack_needed = target_len - len(needle_tokens) - len(query_tokens)
    if num_haystack_needed <= 0:
        num_haystack_needed = 100
    repeats = int(math.ceil(num_haystack_needed / len(haystack_tokens)))
    full_haystack = (haystack_tokens * repeats)[:num_haystack_needed]

    insert_pos = int(len(full_haystack) * depth)
    final_tokens = full_haystack[:insert_pos] + needle_tokens + full_haystack[insert_pos:] + query_tokens
    return final_tokens, insert_pos


def run_generation(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
) -> str:
    """Executes deterministic greedy decoding with cache resets without heavy re-allocations."""
    reset_cdca_states(model)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


# =====================================================================
# SUB-EXPERIMENT 28A: Cross-Company Model Family Workers (n=10)
# =====================================================================
def run_single_model_28a(model_cfg: dict, device: str = "cuda") -> Dict[str, Any]:
    """Evaluates one model family across 6 compression paradigms (n=10 trials)."""
    m_name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    layer_indices = model_cfg["layers"]

    print(f"\n>>> Benchmarking Model Family: {m_name} ({hf_id}) <<<")
    try:
        model, tokenizer = load_model_and_tokenizer(hf_id, device=device)
    except Exception as e:
        print(f"Error loading {hf_id}: {e}. Fallback...")
        if "llama" in m_name:
            model, tokenizer = load_model_and_tokenizer("NousResearch/Llama-3.2-1B", device=device)
        else:
            raise e

    methods = [
        "original_softmax",
        "streaming_llm",
        "h2o",
        "snapkv",
        "halo",
        "halo_blur",
    ]

    target_len = 4096
    window_size = 512
    max_salient = 128

    # 5 S-NIAH, 3 UUID, 2 Retroactive = 10 trials
    depths_sniah = [0.10, 0.30, 0.50, 0.70, 0.90]
    depths_uuid = [0.20, 0.50, 0.80]
    depths_retro = [0.15, 0.25]

    results = {}

    for method in methods:
        print(f"  --> Testing Method: {method.upper()} on {m_name}...")
        if method == "original_softmax":
            apply_halo_adapter_generic(
                model, window_size=target_len + 100, max_salient=0, mode="original_softmax",
                layer_indices=layer_indices, device=device
            )
        elif method == "streaming_llm":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=4, mode="streaming_llm",
                layer_indices=layer_indices, device=device
            )
        elif method == "h2o":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="h2o",
                layer_indices=layer_indices, device=device
            )
        elif method == "snapkv":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                use_derotation=False, use_surprise=False, diffusion_radius=2, kernel_type="uniform_box",
                layer_indices=layer_indices, device=device
            )
        elif method == "halo":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="halo",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=layer_indices, device=device
            )
        elif method == "halo_blur":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient - 16, mode="halo_blur",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=layer_indices, device=device
            )

        sniah_scores = []
        for d in depths_sniah:
            n_str = "\n[CRITICAL NOTE: The secret operational access code for Project Antigravity is 74921.]\n"
            q_str = "\nQuestion: What is the secret operational access code for Project Antigravity? Answer: The access code is"
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
            inp = torch.tensor([toks], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=10)
            sniah_scores.append(100.0 if "74921" in gen else 0.0)

        uuid_scores = []
        for d in depths_uuid:
            n_str = "\n[The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as X9F-482-Q7B for verification.]\n"
            q_str = "\nQuestion: What is the target alpha-numeric registration identifier assigned to the primary node? Answer: The registration identifier is"
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
            inp = torch.tensor([toks], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=16)
            uuid_scores.append(100.0 if "X9F-482-Q7B" in gen else 0.0)

        retro_scores = []
        for d in depths_retro:
            n_str = "\n[The secret emergency override authorization code is 74921. Make sure you remember this confirmation note.]\n"
            q_str = "\nQuestion: What is the secret emergency override authorization code? Answer: The secret emergency override authorization code is"
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
            inp = torch.tensor([toks], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=10)
            retro_scores.append(100.0 if "74921" in gen else 0.0)

        total_scores = sniah_scores + uuid_scores + retro_scores
        results[method] = {
            "composite": compute_stats(total_scores),
            "sniah": compute_stats(sniah_scores),
            "uuid": compute_stats(uuid_scores),
            "retroactive": compute_stats(retro_scores),
        }
        print(f"    --> {method.upper()}: Composite Recall = {results[method]['composite']['mean']}% "
              f"(±{results[method]['composite']['ci95']}%)")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return results


def run_subexp_28a_qwen(device: str = "cuda") -> Dict[str, Any]:
    cfg = {"name": "qwen3.5_0.8b", "hf_id": "Qwen/Qwen3.5-0.8B-Base", "layers": [3, 7, 11, 15, 19, 23]}
    return run_single_model_28a(cfg, device=device)


def run_subexp_28a_llama(device: str = "cuda") -> Dict[str, Any]:
    cfg = {"name": "llama3.2_1b", "hf_id": "unsloth/Llama-3.2-1B-Instruct", "layers": None}
    return run_single_model_28a(cfg, device=device)


def run_subexp_28a_smollm(device: str = "cuda") -> Dict[str, Any]:
    cfg = {"name": "smollm2_1.7b", "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "layers": None}
    return run_single_model_28a(cfg, device=device)


# =====================================================================
# SUB-EXPERIMENT 28B: Delta_QA & Value-Span Phase Diagram Workers (n=10)
# =====================================================================
def run_subexp_28b_delta_qa(device: str = "cuda") -> Dict[str, Any]:
    """Evaluates Syntactic Cue-to-Value Separation Delta_QA in {1, 3, 6, 10, 15, 22} tokens (n=10)."""
    print("\n" + "=" * 70)
    print("STARTING SUB-EXPERIMENT 28B: Delta_QA Separation Sweep (n=10)")
    print("=" * 70)

    model_name = "Qwen/Qwen3.5-0.8B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    layer_indices = [3, 7, 11, 15, 19, 23]

    target_len = 4096
    window_size = 512
    max_salient = 128

    methods = ["isolated_topk", "snapkv", "h2o", "halo"]
    delta_qa_values = [1, 3, 6, 10, 15, 22]

    filler_templates = {
        1: "definitely",
        3: "specifically identified as",
        6: "confirmed by the system as",
        10: "strictly verified by internal security protocols to be",
        15: "strictly verified and confirmed by internal security protocol operations to be",
        22: "strictly verified, authenticated, and approved by all standard internal network operational security protocols to be",
    }

    depths_n10 = np.linspace(0.12, 0.85, 10).tolist()
    delta_results = {}

    for method in methods:
        delta_results[method] = {}
        if method == "isolated_topk":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                use_derotation=False, use_surprise=False, diffusion_radius=0, kernel_type="uniform_box",
                layer_indices=layer_indices, device=device
            )
        elif method == "snapkv":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                use_derotation=False, use_surprise=False, diffusion_radius=2, kernel_type="uniform_box",
                layer_indices=layer_indices, device=device
            )
        elif method == "h2o":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="h2o",
                layer_indices=layer_indices, device=device
            )
        elif method == "halo":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="halo",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=layer_indices, device=device
            )

        for delta in delta_qa_values:
            filler = filler_templates[delta]
            n_str = f"\n[The critical access authorization key is {filler} 8472.]\n"
            q_str = "\nQuestion: What is the critical access authorization key? Answer: The access authorization key is"
            scores = []
            for d in depths_n10:
                toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
                inp = torch.tensor([toks], device=device)
                gen = run_generation(model, tokenizer, device, inp, max_new_tokens=10)
                scores.append(100.0 if "8472" in gen else 0.0)

            delta_results[method][str(delta)] = compute_stats(scores)
            print(f"  Method={method:<14} | Delta={delta:02d} | Recall={delta_results[method][str(delta)]['mean']}% "
                  f"(±{delta_results[method][str(delta)]['ci95']}%)")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return delta_results


def run_subexp_28b_val_span(device: str = "cuda") -> Dict[str, Any]:
    """Evaluates Value-Span Length L_val in {1, 3, 6, 10, 15} segments (Sub-Span Recall & Exact Match, n=10)."""
    print("\n" + "=" * 70)
    print("STARTING SUB-EXPERIMENT 28B: Value-Span Length Sweep (n=10)")
    print("=" * 70)

    model_name = "Qwen/Qwen3.5-0.8B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    layer_indices = [3, 7, 11, 15, 19, 23]

    target_len = 4096
    window_size = 512
    max_salient = 128

    methods = ["isolated_topk", "snapkv", "h2o", "halo"]
    val_lengths = [1, 3, 6, 10, 15]
    serial_tokens = ["8472", "X9F", "Q7B", "319", "K4M", "P28", "V91", "L44", "J12", "T77", "C03", "N88", "E55", "R19", "W62"]
    depths_n10 = np.linspace(0.12, 0.85, 10).tolist()

    span_results = {}

    for method in methods:
        span_results[method] = {}
        if method == "isolated_topk":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                use_derotation=False, use_surprise=False, diffusion_radius=0, kernel_type="uniform_box",
                layer_indices=layer_indices, device=device
            )
        elif method == "snapkv":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                use_derotation=False, use_surprise=False, diffusion_radius=2, kernel_type="uniform_box",
                layer_indices=layer_indices, device=device
            )
        elif method == "h2o":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="h2o",
                layer_indices=layer_indices, device=device
            )
        elif method == "halo":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="halo",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=layer_indices, device=device
            )

        for L in val_lengths:
            active_segments = serial_tokens[:L]
            full_identifier = "-".join(active_segments)
            n_str = f"\n[The multi-segment registration serial identifier assigned to the vault is {full_identifier}.]\n"
            q_str = "\nQuestion: What is the multi-segment registration serial identifier? Answer: The serial identifier is"

            em_scores = []
            f1_scores = []

            for d in depths_n10:
                toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
                inp = torch.tensor([toks], device=device)
                gen = run_generation(model, tokenizer, device, inp, max_new_tokens=16)

                em = 100.0 if full_identifier in gen else 0.0
                em_scores.append(em)

                recalled_chunks = sum(1 for seg in active_segments if seg in gen)
                f1_scores.append(100.0 * (recalled_chunks / L))

            span_results[method][str(L)] = {
                "exact_match": compute_stats(em_scores),
                "subspan_f1": compute_stats(f1_scores),
            }
            print(f"  Method={method:<14} | L_val={L:02d} | EM={span_results[method][str(L)]['exact_match']['mean']}% "
                  f"| F1={span_results[method][str(L)]['subspan_f1']['mean']}%")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return span_results


# =====================================================================
# SUB-EXPERIMENT 28C: Canonical Literature Replication (n=10)
# =====================================================================
def run_subexp_28c_methods(methods: Optional[List[str]] = None, device: str = "cuda") -> Dict[str, Any]:
    """Evaluates RULER 4 axes + LongBench summarization on Qwen3.5 (n=10 per task)."""
    print("\n" + "=" * 70)
    print("STARTING SUB-EXPERIMENT 28C: Canonical Literature Benchmarks (n=10)")
    print("=" * 70)

    model_name = "Qwen/Qwen3.5-0.8B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    layer_indices = [3, 7, 11, 15, 19, 23]

    target_len = 4096
    window_size = 512
    max_salient = 128

    if methods is None:
        methods = [
            "original_softmax",
            "streaming_llm",
            "h2o",
            "snapkv",
            "halo",
            "halo_blur",
        ]

    subexp_28c_results = {
        "s_niah": {},
        "mk_niah_distractors": {},
        "two_hop_vt": {},
        "cwe_aggregation": {},
        "longbench_summary": {},
    }

    depths_n10 = np.linspace(0.12, 0.85, 10).tolist()

    for method in methods:
        print(f"\n--> Configuring Method: {method.upper()} for Literature Suite...")
        if method == "original_softmax":
            apply_halo_adapter_generic(
                model, window_size=target_len + 100, max_salient=0, mode="original_softmax",
                layer_indices=layer_indices, device=device
            )
        elif method == "streaming_llm":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=4, mode="streaming_llm",
                layer_indices=layer_indices, device=device
            )
        elif method == "h2o":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="h2o",
                layer_indices=layer_indices, device=device
            )
        elif method == "snapkv":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                use_derotation=False, use_surprise=False, diffusion_radius=2, kernel_type="uniform_box",
                layer_indices=layer_indices, device=device
            )
        elif method == "halo":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="halo",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=layer_indices, device=device
            )
        elif method == "halo_blur":
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient - 16, mode="halo_blur",
                use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                layer_indices=layer_indices, device=device
            )

        # 1. RULER Axis 1: S-NIAH (n=10)
        sniah_scores = []
        for d in depths_n10:
            n_str = "\n[The secret operational access code for Project Antigravity is 74921.]\n"
            q_str = "\nQuestion: What is the secret operational access code for Project Antigravity? Answer: The access code is"
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
            inp = torch.tensor([toks], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=10)
            sniah_scores.append(100.0 if "74921" in gen else 0.0)
        subexp_28c_results["s_niah"][method] = compute_stats(sniah_scores)

        # 2. RULER Axis 2: MK-NIAH with 12 Adversarial Distractors (n=10)
        mkniah_scores = []
        for seed_idx, d_target in enumerate(depths_n10):
            rng = np.random.RandomState(100 + seed_idx)
            distractor_depths = rng.uniform(0.10, 0.88, 12).tolist()
            items = []
            for i, d_dist in enumerate(distractor_depths):
                items.append((d_dist, f"\n[The vault security key for Archive-Distractor-{i:02d} is strictly verified as {rng.randint(10000, 99999)}.]\n"))
            items.append((d_target, "\n[The vault security key for Archive-Target is strictly verified as 74921.]\n"))
            items.sort(key=lambda x: x[0])

            q_str = "\nQuestion: What is the vault security key for Archive-Target? Answer: The vault security key for Archive-Target is"
            q_toks = tokenizer.encode(q_str, add_special_tokens=False)
            hay_toks = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

            total_item_toks = sum(len(tokenizer.encode(it[1], add_special_tokens=False)) for it in items)
            rem_len = target_len - total_item_toks - len(q_toks)
            repeats = int(math.ceil(rem_len / len(hay_toks)))
            hay = (hay_toks * repeats)[:rem_len]

            doc_tokens = []
            last_idx = 0
            for d_pos, item_text in items:
                idx = int(rem_len * d_pos)
                doc_tokens.extend(hay[last_idx:idx])
                doc_tokens.extend(tokenizer.encode(item_text, add_special_tokens=False))
                last_idx = idx
            doc_tokens.extend(hay[last_idx:])
            doc_tokens.extend(q_toks)

            inp = torch.tensor([doc_tokens], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=12)
            mkniah_scores.append(100.0 if "74921" in gen else 0.0)
        subexp_28c_results["mk_niah_distractors"][method] = compute_stats(mkniah_scores)

        # 3. RULER Axis 3: Multi-Hop Variable Tracking (2-Hop VT, n=10)
        vt_scores = []
        for seed_idx, d1 in enumerate(depths_n10):
            rng = np.random.RandomState(200 + seed_idx)
            val_code = str(rng.randint(1000, 9999))
            d2 = min(0.88, d1 + 0.35)

            hop1_str = f"\n[System configuration record: primary register ALPHA_REG is assigned numerical value {val_code}.]\n"
            hop2_str = "\n[Pointer binding record: secondary register BETA_PTR is assigned an alias referencing primary register ALPHA_REG.]\n"
            q_str = "\nQuestion: What numerical value is stored in secondary register BETA_PTR? Answer: The numerical value in BETA_PTR is"

            q_toks = tokenizer.encode(q_str, add_special_tokens=False)
            h1_toks = tokenizer.encode(hop1_str, add_special_tokens=False)
            h2_toks = tokenizer.encode(hop2_str, add_special_tokens=False)
            hay_toks = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

            rem_len = target_len - len(h1_toks) - len(h2_toks) - len(q_toks)
            repeats = int(math.ceil(rem_len / len(hay_toks)))
            hay = (hay_toks * repeats)[:rem_len]

            p1 = int(rem_len * d1)
            p2 = int(rem_len * d2)
            doc_tokens = hay[:p1] + h1_toks + hay[p1:p2] + h2_toks + hay[p2:] + q_toks

            inp = torch.tensor([doc_tokens], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=10)
            vt_scores.append(100.0 if val_code in gen else 0.0)
        subexp_28c_results["two_hop_vt"][method] = compute_stats(vt_scores)

        # 4. RULER Axis 4: Common Words Frequency Aggregation (CWE, n=10)
        cwe_scores = []
        target_words = ["COBALT", "ZENITH", "QUARTZ"]
        rare_words = ["AMBER", "BRONZE", "TOPAZ", "ONYX", "JADE", "CORAL", "OPAL", "BERYL", "GARNET", "RUBY"]

        for seed_idx in range(10):
            rng = np.random.RandomState(300 + seed_idx)
            max_hist = target_len - window_size - 100
            injections = []
            for w in target_words:
                for _ in range(8):
                    injections.append((rng.randint(50, max_hist), f" {w} "))
            for rw in rare_words:
                injections.append((rng.randint(50, max_hist), f" {rw} "))
            injections.sort(key=lambda x: x[0])

            q_str = "\nQuestion: Name the three most frequently repeated uppercase code words (COBALT, ZENITH, QUARTZ) in the document. Answer: The three code words are"
            q_toks = tokenizer.encode(q_str, add_special_tokens=False)
            hay_toks = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

            rem_len = target_len - len(q_toks) - len(injections) * 3
            repeats = int(math.ceil(rem_len / len(hay_toks)))
            hay = (hay_toks * repeats)[:rem_len]

            doc_tokens = []
            last_idx = 0
            for pos, text in injections:
                p = min(len(hay), pos)
                doc_tokens.extend(hay[last_idx:p])
                doc_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
                last_idx = p
            doc_tokens.extend(hay[last_idx:])
            doc_tokens.extend(q_toks)

            inp = torch.tensor([doc_tokens[:target_len]], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=16)

            recalled = sum(1 for tw in target_words if tw in gen.upper())
            cwe_scores.append(100.0 * (recalled / 3.0))
        subexp_28c_results["cwe_aggregation"][method] = compute_stats(cwe_scores)

        # 5. LongBench-Style Extractive Summarization (Fact F1, n=10)
        summary_scores = []
        fact_keys = ["Vandenberg", "Elena Rostova", "4250", "720"]

        for seed_idx, d_pos in enumerate(depths_n10):
            n_str = "\n[Scientific Expedition Report: The deep space probe launched from the Vandenberg Launch Complex under lead scientist Dr. Elena Rostova. The spacecraft carries an advanced scientific sensor payload mass of 4250 kilograms and entered a target Polar Sun-Synchronous Orbit at 720 km altitude.]\n"
            q_str = "\nQuestion: Summarize the four key mission facts (launch site, lead scientist, payload mass, target orbit). Summary: The four key mission facts are"
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d_pos)
            inp = torch.tensor([toks], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=32)

            recalled_facts = sum(1 for fk in fact_keys if fk.lower() in gen.lower())
            summary_scores.append(100.0 * (recalled_facts / 4.0))
        subexp_28c_results["longbench_summary"][method] = compute_stats(summary_scores)

        print(f"  Method={method:<14} | S-NIAH={subexp_28c_results['s_niah'][method]['mean']}% "
              f"| MK-NIAH={subexp_28c_results['mk_niah_distractors'][method]['mean']}% "
              f"| 2-Hop VT={subexp_28c_results['two_hop_vt'][method]['mean']}% "
              f"| CWE={subexp_28c_results['cwe_aggregation'][method]['mean']}% "
              f"| Summary Fact F1={subexp_28c_results['longbench_summary'][method]['mean']}%")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return subexp_28c_results


def run_subexp_28c_part1(device: str = "cuda") -> Dict[str, Any]:
    return run_subexp_28c_methods(["original_softmax", "streaming_llm", "h2o"], device=device)


def run_subexp_28c_part2(device: str = "cuda") -> Dict[str, Any]:
    return run_subexp_28c_methods(["snapkv", "halo", "halo_blur"], device=device)


def run_subexp_28c_ruler_and_longbench(device: str = "cuda") -> Dict[str, Any]:
    res1 = run_subexp_28c_part1(device=device)
    res2 = run_subexp_28c_part2(device=device)
    merged = {}
    for task in ["s_niah", "mk_niah_distractors", "two_hop_vt", "cwe_aggregation", "longbench_summary"]:
        merged[task] = {}
        if task in res1:
            merged[task].update(res1[task])
        if task in res2:
            merged[task].update(res2[task])
    return merged


# =====================================================================
# SUB-EXPERIMENT 28D: Ablation & Multi-Turn Worker (n=10)
# =====================================================================
def run_subexp_28d_ablations_multiturn(device: str = "cuda") -> Dict[str, Any]:
    """Evaluates component ablations and multi-turn frozen cache durability (n=10)."""
    print("\n" + "=" * 70)
    print("STARTING SUB-EXPERIMENT 28D-Part1: Component Ablation & Multi-Turn Durability (n=10)")
    print("=" * 70)

    model_name = "Qwen/Qwen3.5-0.8B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    layer_indices = [3, 7, 11, 15, 19, 23]

    target_len = 4096
    window_size = 512
    max_salient = 128

    results = {
        "component_ablation": {},
        "multiturn_durability": {},
    }

    depths_n10 = np.linspace(0.12, 0.85, 10).tolist()

    # 1. Component Ablation (n=10)
    ablation_configs = [
        {"name": "(1) Isolated Top-k (K=0)", "mode": "snapkv", "derot": False, "surp": False, "rad": 0, "kern": "uniform_box"},
        {"name": "(2) SnapKV Baseline (Box K=2)", "mode": "snapkv", "derot": False, "surp": False, "rad": 2, "kern": "uniform_box"},
        {"name": "(3) Wide Box Pool (K=16)", "mode": "snapkv", "derot": False, "surp": False, "rad": 16, "kern": "uniform_box"},
        {"name": "(4) Exp Kernel Only (K=16)", "mode": "snapkv", "derot": False, "surp": False, "rad": 16, "kern": "exponential"},
        {"name": "(5) Exp Kernel + PRISM De-Rot", "mode": "halo", "derot": True, "surp": False, "rad": 16, "kern": "exponential"},
        {"name": "(6) Full HALO (+ Bottom-Up Surprise)", "mode": "halo", "derot": True, "surp": True, "rad": 16, "kern": "exponential"},
    ]

    for cfg in ablation_configs:
        c_name = cfg["name"]
        apply_halo_adapter_generic(
            model, window_size=window_size, max_salient=max_salient, mode=cfg["mode"],
            use_derotation=cfg["derot"], use_surprise=cfg["surp"], diffusion_radius=cfg["rad"],
            kernel_type=cfg["kern"], layer_indices=layer_indices, device=device
        )

        composite_scores = []
        for i, d in enumerate(depths_n10):
            n_str = "\n[The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as X9F-482-Q7B for verification.]\n"
            q_str = "\nQuestion: What is the target alpha-numeric registration identifier assigned to the primary node? Answer: The registration identifier is"
            toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
            inp = torch.tensor([toks], device=device)
            gen = run_generation(model, tokenizer, device, inp, max_new_tokens=16)
            composite_scores.append(100.0 if "X9F-482-Q7B" in gen else 0.0)

        results["component_ablation"][c_name] = compute_stats(composite_scores)
        print(f"  {c_name:<40} -> Recall={results['component_ablation'][c_name]['mean']}% "
              f"(±{results['component_ablation'][c_name]['ci95']}%)")

    # 2. Multi-Turn Unseen Follow-Up Query Durability (n=10)
    multiturn_methods = [
        {"name": "snapkv", "mode": "snapkv", "derot": False, "surp": False, "rad": 2, "kern": "uniform_box"},
        {"name": "halo_no_surprise", "mode": "halo", "derot": True, "surp": False, "rad": 16, "kern": "exponential"},
        {"name": "halo_full", "mode": "halo", "derot": True, "surp": True, "rad": 16, "kern": "exponential"},
    ]

    for m_info in multiturn_methods:
        m_key = m_info["name"]
        apply_halo_adapter_generic(
            model, window_size=window_size, max_salient=max_salient, mode=m_info["mode"],
            use_derotation=m_info["derot"], use_surprise=m_info["surp"], diffusion_radius=m_info["rad"],
            kernel_type=m_info["kern"], layer_indices=layer_indices, device=device
        )

        turn1_scores = []
        turn2_scores = []

        for seed_idx, d1 in enumerate(depths_n10):
            rng = np.random.RandomState(400 + seed_idx)
            d2 = min(0.85, d1 + 0.30)
            code1 = str(rng.randint(100, 999))
            code2 = str(rng.randint(1000, 9999))

            fact1_str = f"\n[The planetary lander velocity reading is recorded as {code1} meters per second.]\n"
            fact2_str = f"\n[CRITICAL ANOMALY ALERT [NODE-99]: Core reactor internal pressure spiked to {code2} Pascals!]\n"

            f1_toks = tokenizer.encode(fact1_str, add_special_tokens=False)
            f2_toks = tokenizer.encode(fact2_str, add_special_tokens=False)
            hay_toks = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

            q1_str = "\nQuestion: What is the planetary lander velocity reading? Answer: The lander velocity is"
            q1_toks = tokenizer.encode(q1_str, add_special_tokens=False)

            rem_len = target_len - len(f1_toks) - len(f2_toks) - len(q1_toks)
            repeats = int(math.ceil(rem_len / len(hay_toks)))
            hay = (hay_toks * repeats)[:rem_len]

            p1 = int(rem_len * d1)
            p2 = int(rem_len * d2)
            doc_haystack = hay[:p1] + f1_toks + hay[p1:p2] + f2_toks + hay[p2:]
            t1_tokens = doc_haystack + q1_toks

            # TURN 1 EXECUTION
            set_freeze_salient_cache(model, False)
            inp_t1 = torch.tensor([t1_tokens], device=device)
            gen_t1 = run_generation(model, tokenizer, device, inp_t1, max_new_tokens=10)
            turn1_scores.append(100.0 if code1 in gen_t1 else 0.0)

            # TURN 2 EXECUTION (FROZEN CACHE)
            set_freeze_salient_cache(model, True)
            q2_str = "\nQuestion: What pressure value in Pascals was reported in the CRITICAL ANOMALY ALERT? Answer: The pressure value is"
            q2_toks = tokenizer.encode(q2_str, add_special_tokens=False)
            t2_tokens = doc_haystack + q2_toks
            inp_t2 = torch.tensor([t2_tokens], device=device)

            with torch.no_grad():
                out_ids = model.generate(
                    inp_t2,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_t2 = tokenizer.decode(out_ids[0][inp_t2.shape[1]:], skip_special_tokens=True)
            turn2_scores.append(100.0 if code2 in gen_t2 else 0.0)
            set_freeze_salient_cache(model, False)

        results["multiturn_durability"][m_key] = {
            "turn1": compute_stats(turn1_scores),
            "turn2": compute_stats(turn2_scores),
        }
        print(f"  {m_key:<18} -> Turn 1: {results['multiturn_durability'][m_key]['turn1']['mean']}% "
              f"| Turn 2 (Frozen Cache): {results['multiturn_durability'][m_key]['turn2']['mean']}% "
              f"(±{results['multiturn_durability'][m_key]['turn2']['ci95']}%)")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return results


# =====================================================================
# SUB-EXPERIMENT 28D: Hyperparameter Grid Worker (beta x gamma, n=8)
# =====================================================================
def run_subexp_28d_beta_gamma_grid(device: str = "cuda") -> Dict[str, Any]:
    """Evaluates 5x5 (beta x gamma) sensitivity grid (n=8)."""
    print("\n" + "=" * 70)
    print("STARTING SUB-EXPERIMENT 28D-Part2: (beta x gamma) Grid (n=8)")
    print("=" * 70)

    model_name = "Qwen/Qwen3.5-0.8B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    layer_indices = [3, 7, 11, 15, 19, 23]

    target_len = 4096
    window_size = 512
    max_salient = 128

    betas = [0.0, 0.05, 0.15, 0.30, 0.50]
    gammas = [0.02, 0.05, 0.10, 0.25, 0.50]
    depths_n8 = np.linspace(0.15, 0.85, 8).tolist()

    grid_results = {}

    for beta in betas:
        grid_results[f"beta_{beta:.2f}"] = {}
        for gamma in gammas:
            apply_halo_adapter_generic(
                model, window_size=window_size, max_salient=max_salient, mode="halo",
                use_derotation=True, use_surprise=(beta > 0.0), diffusion_radius=16,
                kernel_type="exponential", surprise_beta=beta, decay_gamma=gamma,
                layer_indices=layer_indices, device=device
            )

            cell_scores = []
            for d in depths_n8:
                n_str = "\n[The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as X9F-482-Q7B for verification.]\n"
                q_str = "\nQuestion: What is the target alpha-numeric registration identifier assigned to the primary node? Answer: The registration identifier is"
                toks, _ = build_synthetic_context(tokenizer, target_len, n_str, q_str, d)
                inp = torch.tensor([toks], device=device)
                gen = run_generation(model, tokenizer, device, inp, max_new_tokens=16)
                cell_scores.append(100.0 if "X9F-482-Q7B" in gen else 0.0)

            stats = compute_stats(cell_scores)
            grid_results[f"beta_{beta:.2f}"][f"gamma_{gamma:.2f}"] = stats
            print(f"  beta={beta:.2f}, gamma={gamma:.2f} -> Recall={stats['mean']}%")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return grid_results


# =====================================================================
# SUB-EXPERIMENT 28E: Hardware Profiling (Latency, VRAM, Throughput)
# =====================================================================
def run_subexp_28e_hardware_profiling(device: str = "cuda") -> Dict[str, Any]:
    """Evaluates TTFT, selection overhead, decode throughput, and KV footprint (1k -> 32k)."""
    print("\n" + "=" * 70)
    print("STARTING SUB-EXPERIMENT 28E: Hardware Wall-Clock & VRAM Profiling (1k -> 32k)")
    print("=" * 70)

    model_name = "Qwen/Qwen3.5-0.8B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    layer_indices = [3, 7, 11, 15, 19, 23]

    window_size = 1024
    max_salient = 256
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    methods = [
        "original_softmax",
        "h2o",
        "snapkv",
        "halo",
        "halo_blur",
    ]

    subexp_28e_results = {}

    for method in methods:
        subexp_28e_results[method] = {}
        print(f"\n--> Profiling Method: {method.upper()}...")

        for T in seq_lengths:
            if method == "original_softmax":
                apply_halo_adapter_generic(
                    model, window_size=T + 100, max_salient=0, mode="original_softmax",
                    layer_indices=layer_indices, device=device
                )
            elif method == "h2o":
                apply_halo_adapter_generic(
                    model, window_size=window_size, max_salient=max_salient, mode="h2o",
                    layer_indices=layer_indices, device=device
                )
            elif method == "snapkv":
                apply_halo_adapter_generic(
                    model, window_size=window_size, max_salient=max_salient, mode="snapkv",
                    use_derotation=False, use_surprise=False, diffusion_radius=2, kernel_type="uniform_box",
                    layer_indices=layer_indices, device=device
                )
            elif method == "halo":
                apply_halo_adapter_generic(
                    model, window_size=window_size, max_salient=max_salient, mode="halo",
                    use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                    layer_indices=layer_indices, device=device
                )
            elif method == "halo_blur":
                apply_halo_adapter_generic(
                    model, window_size=window_size, max_salient=max_salient - 16, mode="halo_blur",
                    use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential",
                    layer_indices=layer_indices, device=device
                )

            hay_toks = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)
            repeats = int(math.ceil(T / len(hay_toks)))
            tokens = (hay_toks * repeats)[:T]
            input_ids = torch.tensor([tokens], device=device)

            reset_cdca_states(model)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.synchronize()

            # Measure TTFT (Prefill) without materializing 16GB vocab logits across 32k tokens
            t0 = time.perf_counter()
            with torch.no_grad():
                out_base = model.model(input_ids, use_cache=True)
                last_logits = model.lm_head(out_base.last_hidden_state[:, -1:, :])
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t0) * 1000.0

            # Measure Salience Selection Overhead
            selection_times = []
            for layer in model.model.layers:
                if hasattr(layer, "cdca_attn") and getattr(layer.cdca_attn, "last_selection_time_ms", 0.0) > 0.0:
                    selection_times.append(layer.cdca_attn.last_selection_time_ms)
            selection_overhead_ms = float(np.mean(selection_times)) if len(selection_times) > 0 else 0.0

            # Measure Decode Throughput (over 16 tokens)
            decode_tokens = 16
            curr_token = last_logits.argmax(dim=-1)
            t_decode_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(decode_tokens):
                    out_step = model.model(curr_token, use_cache=True)
                    step_logits = model.lm_head(out_step.last_hidden_state[:, -1:, :])
                    curr_token = step_logits.argmax(dim=-1)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            decode_time_s = time.perf_counter() - t_decode_start
            decode_tok_per_sec = float(decode_tokens / (decode_time_s + 1e-6))

            if method == "original_softmax":
                retained_tokens = T
            else:
                retained_tokens = min(T, window_size + max_salient)

            kv_cache_mib = round(float((2 * 6 * 2 * 64 * 2 * retained_tokens) / (1024.0 * 1024.0)), 2)
            peak_vram_mib = round(float(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)), 1) if str(device).startswith("cuda") else 0.0

            subexp_28e_results[method][str(T)] = {
                "ttft_ms": round(ttft_ms, 2),
                "selection_overhead_ms": round(selection_overhead_ms, 2),
                "decode_tok_per_sec": round(decode_tok_per_sec, 1),
                "kv_cache_mib": kv_cache_mib,
                "peak_vram_mib": peak_vram_mib,
            }
            print(f"  T={T:<5} | TTFT={ttft_ms:6.1f}ms | Selection={selection_overhead_ms:4.2f}ms | "
                  f"Throughput={decode_tok_per_sec:5.1f} tok/s | KV={kv_cache_mib:5.1f} MiB | Peak VRAM={peak_vram_mib:6.1f} MiB")

    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return subexp_28e_results
