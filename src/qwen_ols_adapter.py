"""
Qwen3.5 to Flash-OLS Adapter Module.

Wraps pre-trained Qwen3.5 Full-Attention layers (Gated Attention with GQA) with
Fused Flash-OLS Solvable Gram State, eliminating the O(T) KV-cache.

Supports:
1. Training/Distillation mode: Fused Triton in-SRAM accumulation + closed-form analytical autograd.
2. Autoregressive Decode mode: O(1) constant-memory Sherman-Morrison rank-1 update.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from src.fused_ols_triton import FusedCausalOLSAccumFunction, CausalOLSSolveFunction


class QwenFlashOLSAttention(nn.Module):
    """
    Drop-in replacement for Qwen3_5Attention.
    Replaces quadratic Softmax attention with rank-r Solvable Flash-OLS.
    """
    def __init__(self, orig_attn: nn.Module, rank: int = 16, lambda_reg: float = 0.05, gamma: float = 0.99):
        super().__init__()
        self.orig_attn = orig_attn
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.gamma = gamma

        # Extract dimensions from original attention module
        self.head_dim = orig_attn.head_dim  # 256
        self.num_heads = orig_attn.q_proj.weight.shape[0] // (self.head_dim * 2)  # 4096 / (256 * 2) = 8
        self.num_key_value_heads = orig_attn.k_proj.weight.shape[0] // self.head_dim  # 512 / 256 = 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 8 // 2 = 4

        # Freeze original projection weights
        for p in self.orig_attn.parameters():
            p.requires_grad = False

        # Learnable active polynomial subspace projections
        d_raw = self.head_dim * 2  # [x, tanh(x^2)] = 512
        
        # Subspace projectors for keys (2 heads) and queries (8 heads)
        init_proj_k = torch.randn(self.num_key_value_heads, d_raw, rank)
        for h in range(self.num_key_value_heads):
            q, _ = torch.linalg.qr(init_proj_k[h])
            init_proj_k[h] = q
        self.proj_k = nn.Parameter(init_proj_k)  # (2, 512, 16)

        init_proj_q = torch.randn(self.num_heads, d_raw, rank)
        for h in range(self.num_heads):
            q, _ = torch.linalg.qr(init_proj_q[h])
            init_proj_q[h] = q
        self.proj_q = nn.Parameter(init_proj_q)  # (8, 512, 16)

        # Persistent autoregressive OLS state for generation (O(1) memory)
        self.register_buffer("P_state", None, persistent=False)  # (B, H_kv, r, r)
        self.register_buffer("R_state", None, persistent=False)  # (B, H_kv, r, head_dim)

    def feature_map_k(self, k: torch.Tensor) -> torch.Tensor:
        # k: (B, H_kv, T, head_dim)
        raw_k = torch.cat([k, torch.tanh(k ** 2)], dim=-1)  # (B, H_kv, T, 512)
        return torch.einsum("bhtd,hdr->bhtr", raw_k, self.proj_k.to(raw_k.dtype))

    def feature_map_q(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B, H_q, T, head_dim)
        raw_q = torch.cat([q, torch.tanh(q ** 2)], dim=-1)  # (B, H_q, T, 512)
        return torch.einsum("bhtd,hdr->bhtr", raw_q, self.proj_q.to(raw_q.dtype))

    def reset_generation_state(self):
        self.P_state = None
        self.R_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values = None,
        use_cache: bool = False,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        B = input_shape[0]
        T = input_shape[1] if len(input_shape) > 1 else 1
        hidden_shape = (*input_shape, -1, self.head_dim)
        orig_dtype = hidden_states.dtype

        # 1. Project Q, Gate, K, V using pre-trained weights
        query_states, gate = torch.chunk(
            self.orig_attn.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        # 2. QK-Norm & Transpose to (B, H, T, d)
        query_states = self.orig_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.orig_attn.k_norm(self.orig_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.orig_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 3. Apply Rotary Position Embedding (RoPE)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 4. Mode Branch: Autoregressive Single-Token Decode (T == 1) vs Sequence Prefill (T > 1)
        if T == 1 and use_cache and self.P_state is not None:
            # === DECODE MODE: Exact O(1) Sherman-Morrison Rank-1 Update ===
            k_t = key_states.squeeze(2)  # (B, H_kv, head_dim)
            v_t = value_states.squeeze(2)  # (B, H_kv, head_dim)
            phi_k = self.feature_map_k(key_states).squeeze(2).float()  # (B, H_kv, r)
            phi_q = self.feature_map_q(query_states).squeeze(2).float()  # (B, H_q, r)

            P = self.P_state
            R = self.R_state

            # P_phi: (B, H_kv, r)
            P_phi = torch.matmul(P, phi_k.unsqueeze(-1)).squeeze(-1)
            denom = self.gamma + torch.sum(phi_k * P_phi, dim=-1, keepdim=True).unsqueeze(-1)
            P_next = (1.0 / self.gamma) * (P - torch.matmul(P_phi.unsqueeze(-1), P_phi.unsqueeze(-2)) / denom)
            R_next = self.gamma * R + torch.matmul(phi_k.unsqueeze(-1), v_t.float().unsqueeze(-2))

            self.P_state = P_next
            self.R_state = R_next

            # W* = P_next @ R_next: (B, H_kv, r, head_dim)
            W_star = torch.matmul(P_next, R_next).to(orig_dtype)

            # Repeat across query head groups: (B, H_q, r, head_dim)
            W_expanded = W_star.repeat_interleave(self.num_key_value_groups, dim=1)

            # y = phi_q @ W_star: (B, H_q, head_dim)
            y = torch.matmul(phi_q.to(orig_dtype).unsqueeze(-2), W_expanded).squeeze(-2)
            attn_output = y.reshape(*input_shape, -1).contiguous()

        else:
            # === PREFILL / TRAINING MODE: Fused Triton Causal Accumulator ===
            phi_k = self.feature_map_k(key_states)  # (B, H_kv, T, r)
            phi_q = self.feature_map_q(query_states)  # (B, H_q, T, r)

            # Fused Causal Gram & RHS accumulator
            G, R = FusedCausalOLSAccumFunction.apply(phi_k, value_states, self.gamma, self.lambda_reg)

            # Batched closed-form Cholesky solve
            total_kv = B * self.num_key_value_heads * T
            W_star = CausalOLSSolveFunction.apply(
                G.reshape(total_kv, self.rank, self.rank),
                R.reshape(total_kv, self.rank, self.head_dim)
            ).view(B, self.num_key_value_heads, T, self.rank, self.head_dim).to(orig_dtype)

            # Cache the end-of-prefill state for subsequent O(1) decode if requested
            if use_cache:
                G_end = G[:, :, -1].float()  # (B, H_kv, r, r)
                R_end = R[:, :, -1].float()  # (B, H_kv, r, head_dim)
                self.P_state = torch.linalg.inv(G_end)
                self.R_state = R_end

            # Expand W_star across query head groups: (B, H_q, T, r, head_dim)
            W_expanded = W_star.repeat_interleave(self.num_key_value_groups, dim=1)

            # Output token projection: (B, H_q, T, head_dim)
            y = torch.matmul(phi_q.to(orig_dtype).unsqueeze(-2), W_expanded).squeeze(-2)
            attn_output = y.transpose(1, 2).reshape(*input_shape, -1).contiguous()

        # 5. Apply Gated Sigmoid Activation and Final Linear Projection
        attn_output = attn_output * torch.sigmoid(gate).to(orig_dtype)
        attn_output = self.orig_attn.o_proj(attn_output)

        return attn_output, None
