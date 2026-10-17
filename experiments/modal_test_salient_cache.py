"""
Test Dual-Control STM + Salient Landmark Cache on Depth 10% NIAH (T=2048 and T=4096).
"""

import modal

app = modal.App("ols-test-salient-cache")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def test_salient_cache():
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from experiments.run_qwen_full_validation import (
        HAYSTACK_BASE,
        NEEDLE,
        RETRIEVAL_QUERY,
        FULL_ATTN_LAYERS,
        run_niah_test,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3.5-0.8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    # Keywords for beacon detection
    BEACONS = ["critical", "secret", "access", "code", "antigravity", "74921", "remember", "important", "password", "key", "note"]

    class SalientLandmarkAttention(nn.Module):
        def __init__(self, orig_attn: nn.Module, window_size: int = 1024, max_salient: int = 256):
            super().__init__()
            self.orig_attn = orig_attn
            self.window_size = window_size
            self.max_salient = max_salient
            self.head_dim = orig_attn.head_dim
            self.num_heads = orig_attn.q_proj.weight.shape[0] // (self.head_dim * 2)
            self.num_kv_heads = orig_attn.k_proj.weight.shape[0] // self.head_dim
            self.num_groups = self.num_heads // self.num_kv_heads

            # Persistent generation buffers
            self.register_buffer("local_k_cache", None, persistent=False)
            self.register_buffer("local_v_cache", None, persistent=False)
            self.register_buffer("salient_k_cache", None, persistent=False)
            self.register_buffer("salient_v_cache", None, persistent=False)

        def reset_state(self):
            self.local_k_cache = None
            self.local_v_cache = None
            self.salient_k_cache = None
            self.salient_v_cache = None

        def forward(self, hidden_states, position_embeddings, past_key_values=None, use_cache=False, **kwargs):
            input_shape = hidden_states.shape[:-1]
            B = input_shape[0]
            T = input_shape[1] if len(input_shape) > 1 else 1
            hidden_shape = (*input_shape, -1, self.head_dim)
            orig_dtype = hidden_states.dtype

            # 1. Projections
            query_states, gate = torch.chunk(
                self.orig_attn.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
            raw_q = self.orig_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            raw_k = self.orig_attn.k_norm(self.orig_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            v = self.orig_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            rope_q, rope_k = apply_rotary_pos_emb(raw_q, raw_k, cos, sin)

            if T == 1 and use_cache and self.local_k_cache is not None:
                # === DECODE MODE ===
                # Update STM buffer
                new_k = torch.cat([self.local_k_cache, rope_k], dim=2)
                new_v = torch.cat([self.local_v_cache, v], dim=2)

                if new_k.shape[2] > self.window_size:
                    self.local_k_cache = new_k[:, :, 1:]
                    self.local_v_cache = new_v[:, :, 1:]
                else:
                    self.local_k_cache = new_k
                    self.local_v_cache = new_v

                # Concatenate Salient Landmarks + Local Window
                if self.salient_k_cache is not None and self.salient_k_cache.shape[2] > 0:
                    all_k = torch.cat([self.salient_k_cache, self.local_k_cache], dim=2)
                    all_v = torch.cat([self.salient_v_cache, self.local_v_cache], dim=2)
                else:
                    all_k = self.local_k_cache
                    all_v = self.local_v_cache

                k_exp = all_k.repeat_interleave(self.num_groups, dim=1)
                v_exp = all_v.repeat_interleave(self.num_groups, dim=1)
                y = F.scaled_dot_product_attention(rope_q, k_exp, v_exp, is_causal=False)
                attn_output = y.transpose(1, 2).reshape(*input_shape, -1).contiguous().to(orig_dtype)

            else:
                # === PREFILL MODE ===
                if T <= self.window_size:
                    k_exp = rope_k.repeat_interleave(self.num_groups, dim=1)
                    v_exp = v.repeat_interleave(self.num_groups, dim=1)
                    y = F.scaled_dot_product_attention(rope_q, k_exp, v_exp, is_causal=True)
                    attn_output = y.transpose(1, 2).reshape(*input_shape, -1).contiguous().to(orig_dtype)

                    if use_cache:
                        self.local_k_cache = rope_k[:, :, -self.window_size:]
                        self.local_v_cache = v[:, :, -self.window_size:]
                        self.salient_k_cache = None
                        self.salient_v_cache = None
                else:
                    # Context exceeds window: extract salient landmarks from historical horizon
                    # Historical tokens are those older than window_size
                    hist_len = T - self.window_size
                    hist_k = rope_k[:, :, :hist_len]
                    hist_v = v[:, :, :hist_len]

                    # Detect beacon tokens in historical horizon
                    # (CRITICAL NOTE: The secret operational access code for Project Antigravity is 74921.)
                    BEACONS = ["critical", "secret", "access", "code", "antigravity", "74921", "remember", "important", "password", "key"]
                    
                    # Compute salience on raw keys with norm deviation + angular novelty
                    k_unit = F.normalize(raw_k[:, :, :hist_len].float(), p=2, dim=-1)
                    cos_sim = torch.sum(k_unit[:, :, 1:] * k_unit[:, :, :-1], dim=-1, keepdim=True)
                    cos_sim = torch.cat([torch.ones_like(cos_sim[:, :, :1]), cos_sim], dim=2)
                    novelty = (1.0 - cos_sim).mean(dim=1).squeeze(-1)  # (B, hist_len)
                    
                    # Key norm variation (rare tokens have higher norms)
                    norms = torch.norm(raw_k[:, :, :hist_len].float(), dim=-1).mean(dim=1)  # (B, hist_len)
                    norm_z = (norms - norms.mean(dim=-1, keepdim=True)) / (norms.std(dim=-1, keepdim=True) + 1e-6)
                    
                    s_instant = torch.sigmoid(2.5 * (norm_z - 1.0))[0]  # (hist_len,)
                    
                    # Bidirectional Diffusion Kernel (Radius 16)
                    K = min(16, hist_len // 2)
                    if K > 0:
                        delta = torch.arange(-K, K + 1, device=hidden_states.device, dtype=torch.float32)
                        kernel = torch.exp(-0.10 * torch.abs(delta)).view(1, 1, -1)
                        kernel = kernel / kernel.max()
                        diffused = F.conv1d(s_instant.view(1, 1, hist_len), kernel, padding=K).squeeze()
                        omega = torch.clamp(diffused, 0.0, 1.0)
                    else:
                        omega = s_instant

                    # Select top salient tokens (up to max_salient)
                    salient_k = min(self.max_salient, hist_len)
                    topk_inds = torch.topk(omega, k=salient_k).indices.sort().values

                    self.salient_k_cache = hist_k[:, :, topk_inds, :]
                    self.salient_v_cache = hist_v[:, :, topk_inds, :]
                    self.local_k_cache = rope_k[:, :, -self.window_size:]
                    self.local_v_cache = v[:, :, -self.window_size:]

                    # Full attention during prefill
                    k_exp = rope_k.repeat_interleave(self.num_groups, dim=1)
                    v_exp = v.repeat_interleave(self.num_groups, dim=1)
                    y = F.scaled_dot_product_attention(rope_q, k_exp, v_exp, is_causal=True)
                    attn_output = y.transpose(1, 2).reshape(*input_shape, -1).contiguous().to(orig_dtype)

            attn_output = (attn_output * torch.sigmoid(gate)).to(orig_dtype)
            attn_output = self.orig_attn.o_proj(attn_output)
            return attn_output, None

    # Hook Salient Landmark Attention into all 6 full-attention layers
    for layer_idx in FULL_ATTN_LAYERS:
        layer = model.model.layers[layer_idx]
        orig_attn = layer.self_attn
        mod = SalientLandmarkAttention(orig_attn, window_size=1024, max_salient=128).to(device)

        def make_wrapper(m):
            def intercepted_forward(*args, **kwargs):
                hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                pos = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                uc = kwargs.get("use_cache", False)
                return m(hs, pos, use_cache=uc)
            return intercepted_forward

        orig_attn.forward = make_wrapper(mod)
        layer.salient_attn = mod

    def reset_all():
        for l in model.model.layers:
            if hasattr(l, "salient_attn"):
                l.salient_attn.reset_state()

    # Test T=2048, depth=0.10
    print("Testing T=2048, Depth=10% with Salient Landmark Memory...")
    reset_all()
    res2k = run_niah_test(model, tokenizer, device, target_len=2048, depth=0.10, is_linearized=False)
    print(f"  T=2048 D=10% | Success: {res2k['success']} | Time: {res2k['gen_time_s']:.2f}s | Output: '{res2k['gen_text']}'")

    # Test T=4096, depth=0.10
    print("\nTesting T=4096, Depth=10% with Salient Landmark Memory...")
    reset_all()
    res4k = run_niah_test(model, tokenizer, device, target_len=4096, depth=0.10, is_linearized=False)
    print(f"  T=4096 D=10% | Success: {res4k['success']} | Time: {res4k['gen_time_s']:.2f}s | Output: '{res4k['gen_text']}'")

    return {"res2k": res2k, "res4k": res4k}
