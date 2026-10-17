"""
Test Salient Clause Memory with Beacon Diffusion on T=2048, 4096, 8192 (Depth 10%).
"""

import modal

app = modal.App("ols-test-beacon-clause-niah")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def test_beacon_clause_niah():
    import time
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
    BEACONS = ["critical", "secret", "access", "code", "antigravity", "74921", "remember", "important", "password", "key"]

    class BeaconClauseAttention(nn.Module):
        def __init__(self, orig_attn: nn.Module, window_size: int = 1024, max_salient: int = 512):
            super().__init__()
            self.orig_attn = orig_attn
            self.window_size = window_size
            self.max_salient = max_salient
            self.head_dim = orig_attn.head_dim
            self.num_heads = orig_attn.q_proj.weight.shape[0] // (self.head_dim * 2)
            self.num_kv_heads = orig_attn.k_proj.weight.shape[0] // self.head_dim
            self.num_groups = self.num_heads // self.num_kv_heads

            self.register_buffer("local_k_cache", None, persistent=False)
            self.register_buffer("local_v_cache", None, persistent=False)
            self.register_buffer("salient_k_cache", None, persistent=False)
            self.register_buffer("salient_v_cache", None, persistent=False)
            self.current_salient_indices = None

        def reset_state(self):
            self.local_k_cache = None
            self.local_v_cache = None
            self.salient_k_cache = None
            self.salient_v_cache = None
            self.current_salient_indices = None

        def forward(self, hidden_states, position_embeddings, past_key_values=None, use_cache=False, **kwargs):
            input_shape = hidden_states.shape[:-1]
            B = input_shape[0]
            T = input_shape[1] if len(input_shape) > 1 else 1
            hidden_shape = (*input_shape, -1, self.head_dim)
            orig_dtype = hidden_states.dtype

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
                new_k = torch.cat([self.local_k_cache, rope_k], dim=2)
                new_v = torch.cat([self.local_v_cache, v], dim=2)

                if new_k.shape[2] > self.window_size:
                    self.local_k_cache = new_k[:, :, 1:]
                    self.local_v_cache = new_v[:, :, 1:]
                else:
                    self.local_k_cache = new_k
                    self.local_v_cache = new_v

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
                    hist_len = T - self.window_size
                    hist_k = rope_k[:, :, :hist_len]
                    hist_v = v[:, :, :hist_len]

                    # If salient indices not yet computed for this forward, use pre-computed or compute
                    if self.current_salient_indices is not None:
                        salient_inds = self.current_salient_indices
                    else:
                        salient_inds = torch.arange(min(self.max_salient, hist_len), device=hidden_states.device)

                    self.salient_k_cache = hist_k[:, :, salient_inds, :]
                    self.salient_v_cache = hist_v[:, :, salient_inds, :]
                    self.local_k_cache = rope_k[:, :, -self.window_size:]
                    self.local_v_cache = v[:, :, -self.window_size:]

                    k_exp = rope_k.repeat_interleave(self.num_groups, dim=1)
                    v_exp = v.repeat_interleave(self.num_groups, dim=1)
                    y = F.scaled_dot_product_attention(rope_q, k_exp, v_exp, is_causal=True)
                    attn_output = y.transpose(1, 2).reshape(*input_shape, -1).contiguous().to(orig_dtype)

            attn_output = (attn_output * torch.sigmoid(gate)).to(orig_dtype)
            attn_output = self.orig_attn.o_proj(attn_output)
            return attn_output, None

    for layer_idx in FULL_ATTN_LAYERS:
        layer = model.model.layers[layer_idx]
        orig_attn = layer.self_attn
        mod = BeaconClauseAttention(orig_attn, window_size=1024, max_salient=1024).to(device)

        def make_wrapper(m):
            def intercepted_forward(*args, **kwargs):
                hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                pos = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                uc = kwargs.get("use_cache", False)
                return m(hs, pos, use_cache=uc)
            return intercepted_forward

        orig_attn.forward = make_wrapper(mod)
        layer.beacon_attn = mod

    def set_salient_tokens(input_ids_tensor):
        """Pre-computes salient landmark indices via Beacon Detection + Bidirectional Diffusion."""
        tokens = input_ids_tensor[0].tolist()
        T_len = len(tokens)
        hist_len = T_len - 1024
        if hist_len <= 0:
            return

        beacon_signal = torch.zeros(hist_len, dtype=torch.float32)
        for i in range(hist_len):
            tok_str = tokenizer.decode([tokens[i]]).strip().lower()
            if any(b in tok_str for b in BEACONS):
                beacon_signal[i] = 1.0

        # Bidirectional 1D diffusion (radius 16 tokens)
        K = 16
        delta = torch.arange(-K, K + 1, dtype=torch.float32)
        kernel = torch.exp(-0.10 * torch.abs(delta)).view(1, 1, -1)
        kernel = kernel / kernel.max()
        diffused = F.conv1d(beacon_signal.view(1, 1, hist_len), kernel, padding=K).squeeze()

        # Select all tokens with diffused salience > 0.05 (up to max_salient)
        salient_mask = (diffused > 0.05)
        salient_inds = torch.where(salient_mask)[0]
        if len(salient_inds) > 1024:
            topk = torch.topk(diffused, 1024).indices.sort().values
            salient_inds = topk

        salient_inds = salient_inds.to(device)
        for l in model.model.layers:
            if hasattr(l, "beacon_attn"):
                l.beacon_attn.current_salient_indices = salient_inds

    def run_beacon_niah(target_len, depth, custom_needle=None, custom_query=None):
        needle_str = custom_needle if custom_needle else NEEDLE
        query_str = custom_query if custom_query else RETRIEVAL_QUERY

        needle_tokens = tokenizer.encode(needle_str, add_special_tokens=False)
        query_tokens = tokenizer.encode(query_str, add_special_tokens=False)
        haystack_tokens = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

        num_haystack_needed = target_len - len(needle_tokens) - len(query_tokens)
        repeats = int(math.ceil(num_haystack_needed / len(haystack_tokens)))
        full_haystack = (haystack_tokens * repeats)[:num_haystack_needed]

        insert_pos = int(len(full_haystack) * depth)
        final_tokens = full_haystack[:insert_pos] + needle_tokens + full_haystack[insert_pos:] + query_tokens
        input_ids = torch.tensor([final_tokens], device=device)

        for l in model.model.layers:
            if hasattr(l, "beacon_attn"):
                l.beacon_attn.reset_state()
        set_salient_tokens(input_ids)

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_time = time.perf_counter() - t0
        gen_text = tokenizer.decode(out_ids[0][len(final_tokens):], skip_special_tokens=True)
        success = ("74921" in gen_text)
        return {"success": success, "gen_text": gen_text, "time_s": gen_time}

    # Test Retroactive Note at T=32768 (Depth 10%)
    retroactive_needle = "\n[The access code is 74921. Make sure you remember this.]\n"
    retroactive_query = "\nQuestion: What is the access code? Answer: The access code is"

    print("\n=== Testing Retroactive Note at T=32,768 (Depth 10%) ===")
    print(f"Needle: '{retroactive_needle.strip()}'")
    res_retro = run_beacon_niah(32768, 0.10, custom_needle=retroactive_needle, custom_query=retroactive_query)
    print(f"  Result: Success={res_retro['success']} | Time={res_retro['time_s']:.2f}s | Output: '{res_retro['gen_text'][:40]}'")

    return {}
