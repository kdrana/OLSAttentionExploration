"""
EXP-17: Direct Functional Operator Estimation of the KV-Cache.

Directly estimates the non-linear attention retrieval operator F:
    y_t = Softmax(q_t K^T / sqrt(d)) V
by fitting an isolated Solvable Flash-OLS Hat Operator:
    y_hat_t = phi_q(q_t)^T (Phi_K^T Phi_K + lambda I)^(-1) Phi_K^T V
across diverse domains (Python Code, Science, Math, General Text).

Avoids end-to-end language modeling gradient distortion and vocabulary drift.
"""

import math
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from src.fused_ols_triton import FusedCausalOLSAccumFunction, CausalOLSSolveFunction
from src.qwen_ols_adapter import QwenFlashOLSAttention, reset_ols_states


# Diverse multi-domain prompts for operator calibration
DOMAIN_PROMPTS = {
    "code": [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\nresult = quicksort([3, 6, 8, 10, 1, 2, 1])",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, value):\n        new_node = Node(value)\n        if not self.head:\n            self.head = new_node\n            return\n        curr = self.head\n        while curr.next:\n            curr = curr.next\n        curr.next = new_node",
        "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, in_features, hidden_dim, out_features):\n        super().__init__()\n        self.fc1 = nn.Linear(in_features, hidden_dim)\n        self.act = nn.GELU()\n        self.fc2 = nn.Linear(hidden_dim, out_features)\n\n    def forward(self, x):\n        return self.fc2(self.act(self.fc1(x)))",
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
    ],
    "science": [
        "The Sun is the star at the center of the Solar System. It is a nearly perfect sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core. The Sun radiates this energy mainly as light, ultraviolet, and infrared radiation, and is the most important source of energy for life on Earth.",
        "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose. The overall chemical equation is 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2, releasing oxygen gas into the Earth's atmosphere.",
        "Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at the atomic and subatomic scale. A core principle is wave-particle duality, where photons and electrons exhibit both wave-like and particle-like properties depending on the measurement.",
        "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915. It unifies special relativity and Newton's law of universal gravitation, describing gravity not as a force, but as a consequence of the curvature of spacetime caused by the uneven distribution of mass.",
    ],
    "math": [
        "In linear algebra, an eigenvector of a linear transformation is a non-zero vector that changes at most by a scalar factor when that linear transformation is applied. The corresponding eigenvalue is the factor by which the eigenvector is scaled, satisfying the equation A v = lambda v.",
        "The Pythagorean theorem states that in any right triangle, the square of the length of the hypotenuse c is equal to the sum of the squares of the lengths of the other two sides a and b: a^2 + b^2 = c^2. This geometric relationship is fundamental in Euclidean geometry.",
        "A prime number is a natural number greater than 1 that is not a product of two smaller natural numbers. The Fundamental Theorem of Arithmetic asserts that every integer greater than 1 either is a prime itself or can be represented as the product of prime numbers uniquely up to order.",
        "Bayes' theorem calculates the probability of an event based on prior knowledge of conditions that might be related to the event: P(A|B) = P(B|A) * P(A) / P(B), where P(A|B) is the posterior probability given observed evidence B.",
    ],
    "language": [
        "The history of linguistics is inextricably linked to the development of writing systems across ancient civilizations. From early cuneiform tablets in Mesopotamia to the Sanskrit grammar composed by Panini, human language has consistently reflected complex combinatorial grammar.",
        "The Industrial Revolution marked a profound turning point in modern human history. Beginning in Great Britain during the late eighteenth century, the transition from manual artisan production to machine-powered manufacturing fundamentally reshaped global trade and urban demographics.",
        "Once upon a time, in a quiet coastal village nestled between rolling emerald hills and the deep blue sea, an elderly clockmaker discovered an ancient mechanical device hidden beneath the floorboards of his workshop.",
        "Cognitive architecture refers to both a theory about the structure of the human mind and to a computational instantiation of such a theory used in artificial intelligence research to model human reasoning, memory, and perception.",
    ]
}


class DirectOperatorFitter(nn.Module):
    """
    Directly fits projection matrices U_k and U_q to estimate F_Softmax without
    running backpropagation through the base LLM.
    """
    def __init__(self, head_dim: int = 256, num_heads: int = 8, num_kv_heads: int = 2, rank: int = 16):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads  # 4
        self.rank = rank
        d_raw = head_dim * 2  # 512

        # Initialize orthogonal projection matrices
        init_k = torch.randn(num_kv_heads, d_raw, rank)
        for h in range(num_kv_heads):
            q, _ = torch.linalg.qr(init_k[h])
            init_k[h] = q
        self.proj_k = nn.Parameter(init_k)

        init_q = torch.randn(num_heads, d_raw, rank)
        for h in range(num_heads):
            q, _ = torch.linalg.qr(init_q[h])
            init_q[h] = q
        self.proj_q = nn.Parameter(init_q)

    def feature_map_k(self, k: torch.Tensor) -> torch.Tensor:
        # k: (B, H_kv, T, head_dim)
        raw_k = torch.cat([k, torch.tanh(k ** 2)], dim=-1)  # (B, H_kv, T, 512)
        return torch.einsum("bhtd,hdr->bhtr", raw_k, self.proj_k.to(raw_k.dtype))

    def feature_map_q(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B, H_q, T, head_dim)
        raw_q = torch.cat([q, torch.tanh(q ** 2)], dim=-1)  # (B, H_q, T, 512)
        return torch.einsum("bhtd,hdr->bhtr", raw_q, self.proj_q.to(raw_q.dtype))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: float = 0.99, lambda_reg: float = 0.05) -> torch.Tensor:
        B, H_kv, T, d = v.shape
        orig_dtype = v.dtype

        phi_k = self.feature_map_k(k)  # (B, H_kv, T, r)
        phi_q = self.feature_map_q(q)  # (B, H_q, T, r)

        # Fused Causal Gram and RHS accumulator
        G, R = FusedCausalOLSAccumFunction.apply(phi_k, v, gamma, lambda_reg)

        # Batched closed-form Cholesky solve
        total_kv = B * H_kv * T
        W_star = CausalOLSSolveFunction.apply(
            G.reshape(total_kv, self.rank, self.rank),
            R.reshape(total_kv, self.rank, self.head_dim)
        ).view(B, H_kv, T, self.rank, self.head_dim).to(orig_dtype)

        # Expand across query groups
        W_expanded = W_star.repeat_interleave(self.num_kv_groups, dim=1)

        # Output projection
        y_hat = torch.matmul(phi_q.to(orig_dtype).unsqueeze(-2), W_expanded).squeeze(-2)
        return y_hat  # (B, H_q, T, head_dim)


def extract_attention_activations(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    target_layer_idx: int = 3,
    seq_len: int = 256,
) -> dict:
    """
    Extracts Q, K, V activations and true Softmax outputs Y_true from target layer
    using a non-intrusive forward hook during native model execution.
    """
    model.eval()
    orig_attn = model.model.layers[target_layer_idx].self_attn

    extracted = {"domain": [], "q": [], "k": [], "v": [], "y_true": []}
    current_domain = [None]

    orig_forward = orig_attn.forward
    head_dim = orig_attn.head_dim
    num_heads = orig_attn.q_proj.weight.shape[0] // (head_dim * 2)  # 8
    num_kv_heads = orig_attn.k_proj.weight.shape[0] // head_dim     # 2
    num_groups = num_heads // num_kv_heads                           # 4

    def intercepted_forward(*args, **kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        pos_emb = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        query_states, gate = torch.chunk(
            orig_attn.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2), 2, dim=-1
        )
        query_states = orig_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = orig_attn.k_norm(orig_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = orig_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Compute exact Softmax attention output: Y_true
        key_expanded = key_states.repeat_interleave(num_groups, dim=1)
        val_expanded = value_states.repeat_interleave(num_groups, dim=1)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(query_states, key_expanded.transpose(-1, -2)) * scale
        T = query_states.shape[2]
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=hidden_states.device), diagonal=1)
        scores = scores + causal_mask.view(1, 1, T, T)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        y_true = torch.matmul(attn_weights, val_expanded)  # (B, H_q, T, head_dim)

        extracted["domain"].append(current_domain[0])
        extracted["q"].append(query_states.detach().cpu())
        extracted["k"].append(key_states.detach().cpu())
        extracted["v"].append(value_states.detach().cpu())
        extracted["y_true"].append(y_true.detach().cpu())

        return orig_forward(*args, **kwargs)

    print(f"\n[Activation Extraction] Capturing activations for Layer {target_layer_idx} across 4 domains...")
    orig_attn.forward = intercepted_forward

    with torch.no_grad():
        for domain_name, texts in DOMAIN_PROMPTS.items():
            current_domain[0] = domain_name
            for text in texts:
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
                _ = model(inp.input_ids)

    orig_attn.forward = orig_forward
    print(f"  Extracted {len(extracted['q'])} sequences (4 domains x 4 sequences)")
    return extracted



def run_direct_operator_estimation_experiment(device: str = "cuda") -> dict:
    print("==================================================================")
    print(" EXP-17: Direct Functional Operator Estimation of KV-Cache")
    print(f" Target Device: {device}")
    print("==================================================================")

    # 1. Load Pre-Trained Model & Tokenizer
    model_id = "Qwen/Qwen3.5-0.8B-Base"
    print(f"\n[1/4] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    # 2. Extract Activations for Layer 3 across all domains
    target_layer_idx = 3
    extracted = extract_attention_activations(model, tokenizer, device, target_layer_idx=target_layer_idx, seq_len=256)

    # 3. Direct Operator Regression (Isolated Fitting of U_k, U_q)
    print(f"\n[2/4] Directly Fitting Operator F on Layer {target_layer_idx} (1,000 steps, zero model backprop)...")
    operator = DirectOperatorFitter(head_dim=256, num_heads=8, num_kv_heads=2, rank=16).to(device)
    optimizer = torch.optim.AdamW(operator.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-4)

    total_seqs = len(extracted["q"])
    n_train = int(total_seqs * 0.75)
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, total_seqs))

    trace = []
    t0 = time.perf_counter()

    for step in range(1001):
        # Sample sequences with gradient accumulation across variable lengths
        optimizer.zero_grad()
        b_indices = np.random.choice(train_indices, size=2, replace=False)
        total_loss = 0.0
        for idx in b_indices:
            q_seq = extracted["q"][idx].to(device)
            k_seq = extracted["k"][idx].to(device)
            v_seq = extracted["v"][idx].to(device)
            y_true_seq = extracted["y_true"][idx].to(device)

            y_hat = operator(q_seq, k_seq, v_seq, gamma=0.99, lambda_reg=0.05)
            mse_loss = F.mse_loss(y_hat, y_true_seq)
            cos_sim = F.cosine_similarity(y_hat.float(), y_true_seq.float(), dim=-1).mean()
            loss = (mse_loss + 0.1 * (1.0 - cos_sim)) / len(b_indices)
            loss.backward()
            total_loss += loss.item() * len(b_indices)

        torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            with torch.no_grad():
                # Evaluate on validation sequences
                val_r2_list = []
                val_cos_list = []
                for vi in val_indices:
                    vq = extracted["q"][vi].to(device)
                    vk = extracted["k"][vi].to(device)
                    vv = extracted["v"][vi].to(device)
                    vy = extracted["y_true"][vi].to(device)
                    v_hat = operator(vq, vk, vv)

                    var_true = torch.var(vy)
                    mse_val = F.mse_loss(v_hat, vy)
                    r2 = 1.0 - (mse_val / (var_true + 1e-8))
                    cos = F.cosine_similarity(v_hat.float(), vy.float(), dim=-1).mean()

                    val_r2_list.append(r2.item())
                    val_cos_list.append(cos.item())

                mean_r2 = float(np.mean(val_r2_list))
                mean_cos = float(np.mean(val_cos_list))
                trace.append({
                    "step": step,
                    "train_loss": float(total_loss),
                    "val_r2": mean_r2,
                    "val_cos_sim": mean_cos,
                })
                print(f"  Step {step:4d}/1000 | Train Loss: {total_loss:.5f} | Val R^2: {mean_r2:.4f} | Cos Sim: {mean_cos:.4f}")

    fit_time_s = time.perf_counter() - t0
    print(f"  Operator Fitting Complete in {fit_time_s:.2f}s ({fit_time_s/1000*1000:.1f}ms/step)")

    # 4. Domain-by-Domain Fidelity Breakdown
    print("\n[3/4] Evaluating Domain-Specific Operator Fidelity...")
    domain_results = {}
    with torch.no_grad():
        for d_name in ["code", "science", "math", "language"]:
            d_indices = [i for i, d in enumerate(extracted["domain"]) if d == d_name]
            d_r2s = []
            d_coss = []
            for di in d_indices:
                dq = extracted["q"][di].to(device)
                dk = extracted["k"][di].to(device)
                dv = extracted["v"][di].to(device)
                dy = extracted["y_true"][di].to(device)
                d_hat = operator(dq, dk, dv)

                var_true = torch.var(dy)
                mse_d = F.mse_loss(d_hat, dy)
                r2 = 1.0 - (mse_d / (var_true + 1e-8))
                cos = F.cosine_similarity(d_hat.float(), dy.float(), dim=-1).mean()
                d_r2s.append(r2.item())
                d_coss.append(cos.item())

            domain_results[d_name] = {
                "r2": float(np.mean(d_r2s)),
                "cosine_sim": float(np.mean(d_coss)),
            }
            print(f"  - Domain: {d_name.upper():8s} | R^2: {domain_results[d_name]['r2']:.4f} | Cosine Sim: {domain_results[d_name]['cosine_sim']:.4f}")

    # 5. Plug-Back Verification & Qualitative Generation
    print("\n[4/4] Plugging Estimated Operator back into Qwen3.5...")
    # Wrap Layer 3 with QwenFlashOLSAttention using the directly fitted U_k, U_q
    layer = model.model.layers[target_layer_idx]
    ols_attn = QwenFlashOLSAttention(layer.self_attn, rank=16, lambda_reg=0.05, gamma=0.99).to(device)
    ols_attn.proj_k.data.copy_(operator.proj_k.data)
    ols_attn.proj_q.data.copy_(operator.proj_q.data)
    layer.self_attn = ols_attn

    verification_prompts = [
        ("Science (Sun)", "The sun is a star located at the center of the Solar System. It is composed primarily of"),
        ("Python Code", "def binary_search(arr, target):\n    \"\"\"Return index of target in sorted arr.\"\"\"\n    low, high = 0, len(arr) - 1\n    while low <= high:\n"),
        ("Math Formula", "To calculate the area of a circle with radius r, the formula is"),
        ("Story", "Once upon a time, there was a little bird named Pip who lived in a"),
    ]

    completions = []
    print("  Testing Plug-Back Generations (Zero Domain Distortion Check):")
    model.eval()
    with torch.no_grad():
        for category, p_text in verification_prompts:
            reset_ols_states(model)
            inp = tokenizer(p_text, return_tensors="pt").to(device)
            p_len = inp.input_ids.shape[1]

            out_ids = model.generate(
                **inp,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True).replace('\n', ' ')
            completions.append({
                "category": category,
                "prompt": p_text,
                "completion": gen_text,
            })
            print(f"  - [{category}] \"{p_text[:35]}...\"")
            print(f"    -> \"{gen_text[len(p_text):len(p_text)+80]}...\"")

    final_metrics = {
        "experiment_id": "EXP-17-DIRECT-OPERATOR",
        "model_id": model_id,
        "target_layer_idx": target_layer_idx,
        "rank": 16,
        "fitting_time_s": fit_time_s,
        "fitting_steps": 1000,
        "final_val_r2": trace[-1]["val_r2"],
        "final_val_cos_sim": trace[-1]["val_cos_sim"],
        "trace": trace,
        "domain_fidelity": domain_results,
        "plug_back_completions": completions,
    }

    # Extract operator weights dictionary for saving
    adapter_state = {
        f"layer_{target_layer_idx}.proj_k": operator.proj_k.data.cpu(),
        f"layer_{target_layer_idx}.proj_q": operator.proj_q.data.cpu(),
        f"layer_{target_layer_idx}.rank": 16,
        f"layer_{target_layer_idx}.gamma": 0.99,
        f"layer_{target_layer_idx}.lambda_reg": 0.05,
    }

    return final_metrics, adapter_state


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics, _ = run_direct_operator_estimation_experiment(device=device)
    print("Self-test completed!")
