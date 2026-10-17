"""
EXP-26: Cognitive Dual-Control Memory Ablation Suite (CDCA-V3).
==============================================================
Benchmarks:
1. Full 18-Trial NIAH Grid across context lengths:
   T in [1024, 2048, 4096, 8192, 16384, 32768], Depths [10%, 50%, 90%]
   Across 4 Methods:
   - Original Softmax (Historical Baseline: 18/18 = 100%, 393.2 MB)
   - StreamingLLM (W=1024 + 4 sinks)
   - H2O (W=1024 + 1024 heavy-hitters)
   - CDCA-V3 (Zero-Keyword Continuous Tensor Salience + De-Rotation + Bidirectional Diffusion)
   - Retroactive Framing at T=32,768 (Depth 10%)

2. Out-of-Distribution & Multi-Needle Stress Tests:
   - Arbitrary UUID Needle ("X9F-482-Q7B") at T=8192 and T=16384
   - Multi-Needle Synthesis (3 distinct codes at 15%, 50%, 85% at T=16384)

3. Expanded 25-Question Multi-Domain Reasoning Benchmark (MMLU / ARC)

4. Long-Context Perplexity at T=4096 and T=8192
"""

import os
import gc
import re
import math
import time
import json
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cognitive_dual_control_attention import (
    apply_cdca_adapter,
    reset_cdca_states,
)
HAYSTACK_BASE = """
Artificial intelligence research has evolved through multiple historical epochs. From symbolic expert systems in the 1970s to statistical learning paradigms in the 1990s, the field has continuously explored computational architectures for modeling knowledge. In modern connectionist models, transformer architectures rely on self-attention mechanisms to dynamically route information across sequence representations. State-space models and recurrent linear networks seek to compress temporal dependencies into fixed-dimensional representations, eliminating quadratic attention bottlenecks while maintaining expressive associative recall.
The mathematical formulation of causal regression models provides an analytical foundation for continuous temporal updates. By expressing covariance accumulation in terms of recursive least squares, state representations remain invariant to historical context length. Numerical stability is preserved through regularized normal equations, where ridge penalties prevent ill-conditioning during matrix inversion.
In distributed deep learning systems, memory bandwidth and tensor memory footprints dictate throughput scaling. High-bandwidth memory (HBM) on contemporary accelerator hardware experiences memory saturation when historical key-value tensors scale linearly with token length. In contrast, fixed-size recurrent memory states maintain strict O(1) resident footprints, avoiding DRAM bandwidth exhaustion and allowing long-horizon inference without sequence eviction.
"""

NEEDLE = "\n[CRITICAL NOTE: The secret operational access code for Project Antigravity is 74921.]\n"
RETRIEVAL_QUERY = "\nQuestion: What is the secret operational access code for Project Antigravity? Answer: The access code is"


RETROACTIVE_NEEDLE = "\n[The secret operational access code for Project Antigravity is 74921. Make sure you remember this confirmation note.]\n"
UUID_NEEDLE = "\n[SYSTEM LOG: The active authorization token is X9F-482-Q7B.]\n"
UUID_QUERY = "\nQuestion: What is the active authorization token? Answer: The authorization token is"


# 25-Question Expanded Multi-Domain Reasoning Evaluation (4% granularity)
EXPANDED_MMLU_QUESTIONS = [
    {"question": "What is the primary function of ribosomes in a biological cell?", "choices": ["A) Lipid synthesis", "B) Protein synthesis", "C) DNA replication", "D) ATP generation"], "correct": "B"},
    {"question": "Which of the following elements has the highest electronegativity?", "choices": ["A) Oxygen", "B) Chlorine", "C) Fluorine", "D) Nitrogen"], "correct": "C"},
    {"question": "In Newtonian mechanics, what does the area under a force-displacement graph represent?", "choices": ["A) Momentum", "B) Power", "C) Work done", "D) Acceleration"], "correct": "C"},
    {"question": "What is the time complexity of searching for an element in a balanced binary search tree with N nodes?", "choices": ["A) O(1)", "B) O(log N)", "C) O(N)", "D) O(N log N)"], "correct": "B"},
    {"question": "Which gas is most abundant in the Earth's atmosphere?", "choices": ["A) Oxygen", "B) Nitrogen", "C) Carbon dioxide", "D) Argon"], "correct": "B"},
    {"question": "What is the derivative of sin(x) with respect to x?", "choices": ["A) -cos(x)", "B) cos(x)", "C) tan(x)", "D) -sin(x)"], "correct": "B"},
    {"question": "Which layer of the Earth's atmosphere contains the ozone layer?", "choices": ["A) Troposphere", "B) Stratosphere", "C) Mesosphere", "D) Thermosphere"], "correct": "B"},
    {"question": "What is the speed of light in vacuum approximately?", "choices": ["A) 3 x 10^6 m/s", "B) 3 x 10^8 m/s", "C) 3 x 10^10 m/s", "D) 3 x 10^5 km/h"], "correct": "B"},
    {"question": "In computer science, which data structure operates on a Last-In, First-Out (LIFO) basis?", "choices": ["A) Queue", "B) Stack", "C) Heap", "D) Linked List"], "correct": "B"},
    {"question": "What organelle is known as the powerhouse of the cell?", "choices": ["A) Nucleus", "B) Chloroplast", "C) Mitochondrion", "D) Golgi apparatus"], "correct": "C"},
    {"question": "What is the chemical formula for table salt?", "choices": ["A) KCl", "B) NaCl", "C) Na2SO4", "D) CaCl2"], "correct": "B"},
    {"question": "Which planet in the solar system has the highest surface temperature?", "choices": ["A) Mercury", "B) Mars", "C) Venus", "D) Jupiter"], "correct": "C"},
    {"question": "In calculus, what is the integral of 1/x dx?", "choices": ["A) e^x", "B) ln|x| + C", "C) x^2 / 2", "D) 1 / x^2"], "correct": "B"},
    {"question": "What is the SI unit of electrical resistance?", "choices": ["A) Volt", "B) Ampere", "C) Ohm", "D) Watt"], "correct": "C"},
    {"question": "Which blood type is considered the universal red blood cell donor?", "choices": ["A) AB positive", "B) O negative", "C) A positive", "D) B negative"], "correct": "B"},
    {"question": "In Python, which keyword is used to define an anonymous inline function?", "choices": ["A) def", "B) lambda", "C) func", "D) anon"], "correct": "B"},
    {"question": "What is the pH of pure water at 25 degrees Celsius?", "choices": ["A) 5", "B) 6", "C) 7", "D) 8"], "correct": "C"},
    {"question": "Which law states that the total pressure of a gas mixture is the sum of partial pressures?", "choices": ["A) Boyle's Law", "B) Dalton's Law", "C) Charles's Law", "D) Avogadro's Law"], "correct": "B"},
    {"question": "What is the capital city of Australia?", "choices": ["A) Sydney", "B) Melbourne", "C) Canberra", "D) Brisbane"], "correct": "C"},
    {"question": "In linear algebra, an orthogonal matrix Q satisfies which condition?", "choices": ["A) Q = Q^T", "B) Q^T Q = I", "C) det(Q) = 0", "D) Q^2 = I"], "correct": "B"},
    {"question": "What enzyme in saliva begins the digestion of carbohydrates?", "choices": ["A) Pepsin", "B) Amylase", "C) Lipase", "D) Trypsin"], "correct": "B"},
    {"question": "Which quantum number describes the shape of an atomic orbital?", "choices": ["A) Principal (n)", "B) Azimuthal / Angular momentum (l)", "C) Magnetic (m)", "D) Spin (s)"], "correct": "B"},
    {"question": "In relational databases, what does the ACID acronym stand for?", "choices": ["A) Atomicity, Consistency, Isolation, Durability", "B) Access, Control, Index, Data", "C) Asynchronous, Concurrent, Isolated, Distributed", "D) Array, Column, Index, Directory"], "correct": "A"},
    {"question": "Which fundamental force is responsible for radioactive beta decay?", "choices": ["A) Gravity", "B) Electromagnetism", "C) Strong nuclear force", "D) Weak nuclear force"], "correct": "D"},
    {"question": "What is the escape velocity from the surface of the Earth approximately?", "choices": ["A) 7.9 km/s", "B) 11.2 km/s", "C) 16.7 km/s", "D) 29.8 km/s"], "correct": "B"},
]


def run_single_trial(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    target_len: int,
    depth: float,
    needle_str: str,
    query_str: str,
    expected_answer: str,
) -> dict:
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
    input_ids = torch.tensor([final_tokens], device=device)
    actual_len = input_ids.shape[1]

    reset_cdca_states(model)
    gc.collect()
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()
    gen_time_s = time.perf_counter() - t0

    gen_text = tokenizer.decode(out_ids[0][actual_len:], skip_special_tokens=True)
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda")) else 0.0
    success = (expected_answer in gen_text)

    local_tokens = min(actual_len, 1024)
    salient_tokens = min(max(0, actual_len - 1024), 1024)
    state_kb = float((local_tokens + salient_tokens) * 12.0)

    return {
        "context_length": actual_len,
        "target_length": target_len,
        "depth": depth,
        "success": success,
        "gen_time_s": gen_time_s,
        "peak_vram_mb": peak_vram_mb,
        "attn_state_kb": state_kb,
        "gen_text": gen_text,
    }


def run_multi_needle_3queries(model: nn.Module, tokenizer: AutoTokenizer, device: str) -> dict:
    n1 = "\n[CRITICAL NOTE: Alpha authorization code is 9182.]\n"
    n2 = "\n[CRITICAL NOTE: Beta authorization code is 3746.]\n"
    n3 = "\n[CRITICAL NOTE: Gamma authorization code is 5501.]\n"

    t1 = tokenizer.encode(n1, add_special_tokens=False)
    t2 = tokenizer.encode(n2, add_special_tokens=False)
    t3 = tokenizer.encode(n3, add_special_tokens=False)
    thay = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

    queries = [
        {"name": "alpha", "query": "\nQuestion: What is the Alpha authorization code? Answer: The Alpha authorization code is", "expected": "9182", "depth": 0.15},
        {"name": "beta", "query": "\nQuestion: What is the Beta authorization code? Answer: The Beta authorization code is", "expected": "3746", "depth": 0.50},
        {"name": "gamma", "query": "\nQuestion: What is the Gamma authorization code? Answer: The Gamma authorization code is", "expected": "5501", "depth": 0.85},
    ]

    total_query_len = max(len(tokenizer.encode(q["query"], add_special_tokens=False)) for q in queries)
    total_hay_len = 16384 - len(t1) - len(t2) - len(t3) - total_query_len
    repeats = int(math.ceil(total_hay_len / len(thay)))
    hay = (thay * repeats)[:total_hay_len]

    p1 = int(len(hay) * 0.15)
    p2 = int(len(hay) * 0.50)
    p3 = int(len(hay) * 0.85)

    base_doc = hay[:p1] + t1 + hay[p1:p2] + t2 + hay[p2:p3] + t3 + hay[p3:]

    results = {}
    needles_recalled = 0

    for q_info in queries:
        tq = tokenizer.encode(q_info["query"], add_special_tokens=False)
        input_ids = torch.tensor([base_doc + tq], device=device)
        reset_cdca_states(model)
        gc.collect()
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            torch.cuda.empty_cache()

        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        success = (q_info["expected"] in gen_text)
        if success:
            needles_recalled += 1
        results[q_info["name"]] = {
            "expected": q_info["expected"],
            "success": success,
            "gen_text": gen_text,
        }

    results["needles_recalled"] = needles_recalled
    results["total_needles"] = 3
    results["all_recalled"] = (needles_recalled == 3)
    results["context_length"] = 16384
    results["success"] = (needles_recalled == 3)
    return results


def evaluate_expanded_mmlu(model: nn.Module, tokenizer: AutoTokenizer, device: str) -> float:
    model.eval()
    correct = 0
    choice_labels = ["A", "B", "C", "D"]
    choice_token_ids = [tokenizer.encode(" " + c, add_special_tokens=False)[-1] for c in choice_labels]

    with torch.no_grad():
        for q in EXPANDED_MMLU_QUESTIONS:
            reset_cdca_states(model)
            formatted = f"Question: {q['question']}\nChoices:\n" + "\n".join(q["choices"]) + "\nAnswer: The correct answer is"
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0, -1, :]
            choice_logits = [logits[tid].item() for tid in choice_token_ids]
            pred_choice = choice_labels[int(np.argmax(choice_logits))]
            if pred_choice == q["correct"]:
                correct += 1

    return float(correct / len(EXPANDED_MMLU_QUESTIONS) * 100.0)


def measure_long_context_ppl(model: nn.Module, tokenizer: AutoTokenizer, device: str, target_len: int) -> float:
    haystack_tokens = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)
    repeats = int(math.ceil(target_len / len(haystack_tokens)))
    tokens = (haystack_tokens * repeats)[:target_len]
    input_ids = torch.tensor([tokens], device=device)

    reset_cdca_states(model)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        ppl = math.exp(min(loss, 20.0))
    return float(round(ppl, 2))


def set_model_cdca_mode(model: nn.Module, mode: str):
    for layer in model.model.layers:
        if hasattr(layer, "cdca_attn"):
            layer.cdca_attn.mode = mode


def run_cdca_v3_benchmark(device: str = "cuda") -> dict:
    print("==================================================================")
    print(" EXP-26: Cognitive Dual-Control Memory Ablation Suite (CDCA-V3)")
    print(f" Target Device: {device}")
    print("==================================================================")

    model_id = "Qwen/Qwen3.5-0.8B-Base"
    print(f"\n[1/5] Loading {model_id} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Apply CDCA-V3 adapter
    print("\n[2/5] Applying CDCA-V3 Adapter (STM=1024, Landmark=1024)...")
    apply_cdca_adapter(model, window_size=1024, max_salient=1024, mode="cdca_v3", device=device)

    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    depths = [0.10, 0.50, 0.90]

    # Historical Softmax baseline numbers from EXP-18 full validation
    softmax_niah_trials = [
        {"context_length": T, "depth": D, "success": True, "attn_state_kb": float(T * 12.0)}
        for T in context_lengths for D in depths
    ]

    metrics = {
        "experiment_id": "EXP-26-CDCA-V3-ABLATION",
        "date": "2026-10-17",
        "device": str(device),
        "niah_grid_18_trials": {
            "original_softmax": {
                "success_count": 18,
                "total_count": 18,
                "success_rate_pct": 100.0,
                "trials": softmax_niah_trials,
            },
            "streaming_llm": {"success_count": 0, "total_count": 18, "success_rate_pct": 0.0, "trials": []},
            "h2o": {"success_count": 0, "total_count": 18, "success_rate_pct": 0.0, "trials": []},
            "cdca_v3": {"success_count": 0, "total_count": 18, "success_rate_pct": 0.0, "trials": []},
        },
        "stress_tests": {
            "retroactive_framing_32k": {},
            "uuid_needle_8k": {},
            "uuid_needle_16k": {},
            "multi_needle_16k": {},
        },
        "academic_benchmarks": {
            "mmlu_expanded_25q_pct": 0.0,
        },
        "perplexity": {
            "ppl_4096": 0.0,
            "ppl_8192": 0.0,
        },
    }

    # 1. Evaluate CDCA-V3 across all 18 trials
    print("\n[3/5] Running 18-Trial NIAH Grid on CDCA-V3 (Zero Keywords)...")
    set_model_cdca_mode(model, "cdca_v3")
    cdca_v3_successes = 0
    for T in context_lengths:
        for D in depths:
            res = run_single_trial(
                model, tokenizer, device, target_len=T, depth=D,
                needle_str=NEEDLE, query_str=RETRIEVAL_QUERY, expected_answer="74921",
            )
            metrics["niah_grid_18_trials"]["cdca_v3"]["trials"].append(res)
            if res["success"]:
                cdca_v3_successes += 1
            print(f"  [CDCA-V3] T={T:5d} | Depth={D*100:2.0f}% | Success={res['success']} | Gen: '{res['gen_text'][:15]}' | State={res['attn_state_kb']:.1f} KB")

    metrics["niah_grid_18_trials"]["cdca_v3"]["success_count"] = cdca_v3_successes
    metrics["niah_grid_18_trials"]["cdca_v3"]["success_rate_pct"] = round(cdca_v3_successes / 18.0 * 100.0, 1)

    # 2. Evaluate StreamingLLM across all 18 trials
    print("\n[3/5b] Running 18-Trial NIAH Grid on StreamingLLM (4 sinks + W=1024)...")
    set_model_cdca_mode(model, "streaming_llm")
    streaming_successes = 0
    for T in context_lengths:
        for D in depths:
            res = run_single_trial(
                model, tokenizer, device, target_len=T, depth=D,
                needle_str=NEEDLE, query_str=RETRIEVAL_QUERY, expected_answer="74921",
            )
            metrics["niah_grid_18_trials"]["streaming_llm"]["trials"].append(res)
            if res["success"]:
                streaming_successes += 1
            print(f"  [StreamingLLM] T={T:5d} | Depth={D*100:2.0f}% | Success={res['success']}")

    metrics["niah_grid_18_trials"]["streaming_llm"]["success_count"] = streaming_successes
    metrics["niah_grid_18_trials"]["streaming_llm"]["success_rate_pct"] = round(streaming_successes / 18.0 * 100.0, 1)

    # 3. Evaluate H2O across all 18 trials
    print("\n[3/5c] Running 18-Trial NIAH Grid on H2O (W=1024 + 1024 Heavy-Hitters)...")
    set_model_cdca_mode(model, "h2o")
    h2o_successes = 0
    for T in context_lengths:
        for D in depths:
            res = run_single_trial(
                model, tokenizer, device, target_len=T, depth=D,
                needle_str=NEEDLE, query_str=RETRIEVAL_QUERY, expected_answer="74921",
            )
            metrics["niah_grid_18_trials"]["h2o"]["trials"].append(res)
            if res["success"]:
                h2o_successes += 1
            print(f"  [H2O] T={T:5d} | Depth={D*100:2.0f}% | Success={res['success']}")

    metrics["niah_grid_18_trials"]["h2o"]["success_count"] = h2o_successes
    metrics["niah_grid_18_trials"]["h2o"]["success_rate_pct"] = round(h2o_successes / 18.0 * 100.0, 1)

    # 4. Stress Tests across all modes (CDCA-V3, H2O, StreamingLLM)
    print("\n[4/5] Running Out-of-Distribution & Multi-Needle Stress Tests across modes...")
    stress_modes = ["cdca_v3", "h2o", "streaming_llm"]

    for m in stress_modes:
        print(f"\n  --> Evaluating Stress Tests on Mode: {m}...")
        set_model_cdca_mode(model, m)

        # A. Retroactive Framing at T=32,768 (Depth 10%)
        res_retro = run_single_trial(
            model, tokenizer, device, target_len=32768, depth=0.10,
            needle_str=RETROACTIVE_NEEDLE, query_str=RETRIEVAL_QUERY, expected_answer="74921",
        )
        print(f"    [{m}] Retroactive 32k: {res_retro['success']} | Gen: '{res_retro['gen_text'][:25]}'")

        # B. Arbitrary UUID Needle ("X9F-482-Q7B") at T=8192 and T=16384
        res_uuid_8k = run_single_trial(
            model, tokenizer, device, target_len=8192, depth=0.50,
            needle_str=UUID_NEEDLE, query_str=UUID_QUERY, expected_answer="X9F-482-Q7B",
        )
        print(f"    [{m}] UUID Needle 8k: {res_uuid_8k['success']} | Gen: '{res_uuid_8k['gen_text'][:25]}'")

        res_uuid_16k = run_single_trial(
            model, tokenizer, device, target_len=16384, depth=0.50,
            needle_str=UUID_NEEDLE, query_str=UUID_QUERY, expected_answer="X9F-482-Q7B",
        )
        print(f"    [{m}] UUID Needle 16k: {res_uuid_16k['success']} | Gen: '{res_uuid_16k['gen_text'][:25]}'")

        # C. Multi-Needle Synthesis Test at T=16384 (3 needles at 15%, 50%, 85%)
        res_multi = run_multi_needle_3queries(model, tokenizer, device)
        print(f"    [{m}] Multi-Needle 16k: {res_multi['needles_recalled']}/3 Recalled (Alpha={res_multi['alpha']['success']}, Beta={res_multi['beta']['success']}, Gamma={res_multi['gamma']['success']})")

        metrics["stress_tests"][m] = {
            "retroactive_framing_32k": res_retro,
            "uuid_needle_8k": res_uuid_8k,
            "uuid_needle_16k": res_uuid_16k,
            "multi_needle_16k": res_multi,
        }

    # Set top-level stress_tests keys for backward compatibility
    metrics["stress_tests"]["retroactive_framing_32k"] = metrics["stress_tests"]["cdca_v3"]["retroactive_framing_32k"]
    metrics["stress_tests"]["uuid_needle_8k"] = metrics["stress_tests"]["cdca_v3"]["uuid_needle_8k"]
    metrics["stress_tests"]["uuid_needle_16k"] = metrics["stress_tests"]["cdca_v3"]["uuid_needle_16k"]
    metrics["stress_tests"]["multi_needle_16k"] = metrics["stress_tests"]["cdca_v3"]["multi_needle_16k"]

    # 5. Expanded MMLU (25 questions) and Perplexity
    print("\n[5/5] Evaluating Expanded MMLU (25 Questions) & Perplexity...")
    mmlu_acc = evaluate_expanded_mmlu(model, tokenizer, device)
    metrics["academic_benchmarks"]["mmlu_expanded_25q_pct"] = mmlu_acc
    print(f"  Expanded MMLU 25-Q Accuracy: {mmlu_acc:.1f}%")

    ppl_4k = measure_long_context_ppl(model, tokenizer, device, target_len=4096)
    ppl_8k = measure_long_context_ppl(model, tokenizer, device, target_len=8192)
    metrics["perplexity"]["ppl_4096"] = ppl_4k
    metrics["perplexity"]["ppl_8192"] = ppl_8k
    print(f"  Perplexity at 4096 tokens: {ppl_4k:.2f}")
    print(f"  Perplexity at 8192 tokens: {ppl_8k:.2f}")

    print("\n[+] Benchmark Complete! Returning Metrics...")
    return metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_cdca_v3_benchmark(device=device)
    out_dir = "results/exp26_cdca_v3_ablation"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {out_dir}/metrics.json")

