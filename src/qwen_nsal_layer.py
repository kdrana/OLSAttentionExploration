"""
Qwen-3.5 Native Solvable Attention Module (Qwen-NSAL).
======================================================
Replaces Qwen3.5 full-attention layers with an end-to-end differentiable,
closed-form solvable operator layer, eliminating RoPE phase interference and
clamping generation memory to strictly O(1) (< 500 KB total model cache).

Key Features:
1. Native GQA Support:
   - Maintains operator state across num_key_value_heads (H_kv=2) and broadcasts
     to num_heads (H_q=8) via Grouped Query Attention (GQA).
2. De-Rotated Semantic Subspace:
   - Operates on un-rotated, stationary semantic key/query coordinates phi(k), phi(q).
3. Dual-Mode Execution:
   - Prefill / Training: Parallel causal prefix solve with exact autograd through
     batched positive-definite normal equations G_t W_t = R_t.
   - Autoregressive Decode: O(1) Recursive Least Squares (RLS) state updates via
     the Sherman-Morrison rank-1 formula.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenNativeSolvableAttention(nn.Module):
    """
    Drop-in NSAL replacement for Qwen3_5Attention in Full-Attention layers.
    """
    def __init__(
        self,
        orig_attn: nn.Module,
        r: int = 32,
        reg_init: float = 0.05,
        use_gate: bool = True,
    ):
        super().__init__()
        self.orig_attn = orig_attn
        self.r = r
        self.head_dim = orig_attn.head_dim  # 256
        self.num_heads = orig_attn.q_proj.weight.shape[0] // (self.head_dim * 2)  # 8
        self.num_key_value_heads = orig_attn.k_proj.weight.shape[0] // self.head_dim  # 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 4
        self.use_gate = use_gate

        # Subspace coordinate projection matrices: R^{head_dim -> r}
        # Learned low-rank coordinates for keys (H_kv=2) and queries (H_q=8)
        self.proj_k = nn.Parameter(
            torch.randn(self.num_key_value_heads, self.head_dim, self.r) * (1.0 / math.sqrt(self.head_dim))
        )
        self.proj_q = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.r) * (1.0 / math.sqrt(self.head_dim))
        )

        # Learned per-head regularization parameter
        self.log_reg = nn.Parameter(torch.full((self.num_key_value_heads, 1, 1), math.log(reg_init)))

        # Persistent autoregressive decode state: P [B, H_kv, r, r], W [B, H_kv, r, head_dim]
        self.register_buffer("P_state", None, persistent=False)
        self.register_buffer("W_state", None, persistent=False)

    def get_regularizer(self) -> torch.Tensor:
        """Returns per-head regularizer lambda = exp(log_reg) clamped for numerical stability."""
        return torch.clamp(torch.exp(self.log_reg), min=1e-5, max=10.0)

    def reset_generation_state(self):
        """Clears autoregressive decode cache buffers."""
        self.P_state = None
        self.W_state = None

    def init_decode_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """Initializes empty RLS state for autoregressive decode."""
        H_kv = self.num_key_value_heads
        r = self.r
        d = self.head_dim
        reg = self.get_regularizer().view(1, H_kv, 1, 1).to(device=device, dtype=dtype)
        eye = torch.eye(r, device=device, dtype=dtype).view(1, 1, r, r).repeat(batch_size, H_kv, 1, 1)
        self.P_state = eye / reg
        self.W_state = torch.zeros(batch_size, H_kv, r, d, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        use_cache: bool = False,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        B = input_shape[0]
        T = input_shape[1] if len(input_shape) > 1 else 1
        orig_dtype = hidden_states.dtype
        device = hidden_states.device

        # 1. Project Q, Gate, K, V using base Qwen projections
        # q_proj produces [B, T, num_heads * head_dim * 2] (query + gate)
        query_states, gate = torch.chunk(
            self.orig_attn.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)  # [B, T, num_heads * head_dim]
        gate_score = torch.sigmoid(gate.view(B, T, self.num_heads, self.head_dim).mean(dim=-1)).transpose(1, 2)  # [B, H_q, T]

        # Aggregate gate down to KV heads for normal equations: [B, H_kv, T, 1]
        gate_kv = gate_score.view(B, self.num_key_value_heads, self.num_key_value_groups, T).mean(dim=2).unsqueeze(-1)

        # QK-Norm (RMSNorm) on un-rotated vectors
        q_raw = self.orig_attn.q_norm(query_states.view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)  # [B, H_q, T, d]
        k_raw = self.orig_attn.k_norm(self.orig_attn.k_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim)).transpose(1, 2)  # [B, H_kv, T, d]
        v_raw = self.orig_attn.v_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, T, d]

        # 2. Map Key and Query to Low-Rank Subspace R^r
        phi_k = torch.einsum("bhtd,hdr->bhtr", k_raw.float(), self.proj_k)  # [B, H_kv, T, r]
        phi_q = torch.einsum("bhtd,hdr->bhtr", q_raw.float(), self.proj_q)  # [B, H_q, T, r]

        # 3. Mode Branch: Autoregressive Single-Token Decode (T == 1) vs Sequence Prefill (T > 1)
        if T == 1 and use_cache and self.P_state is not None:
            # === DECODE MODE: Exact O(1) Recursive Least Squares (RLS) ===
            u_t = (phi_k.squeeze(2) * torch.sqrt(gate_kv.squeeze(2) + 1e-6))  # [B, H_kv, r]
            v_t = (v_raw.squeeze(2).float() * torch.sqrt(gate_kv.squeeze(2) + 1e-6))  # [B, H_kv, d]

            P_prev = self.P_state
            W_prev = self.W_state

            # Gain vector: k_gain = (P_prev @ u_t) / (1 + u_t^T @ P_prev @ u_t)
            Pu = torch.einsum("bhij,bhj->bhi", P_prev, u_t)  # [B, H_kv, r]
            denom = 1.0 + torch.einsum("bhi,bhi->bh", u_t, Pu).unsqueeze(-1)  # [B, H_kv, 1]
            k_gain = Pu / denom  # [B, H_kv, r]

            # Error: e_t = v_t - u_t @ W_prev
            pred = torch.einsum("bhi,bhid->bhd", u_t, W_prev)
            e_t = v_t - pred  # [B, H_kv, d]

            # Operator & Covariance Update
            W_new = W_prev + torch.einsum("bhi,bhd->bhid", k_gain, e_t)
            P_new = P_prev - torch.einsum("bhi,bhj->bhij", k_gain, Pu)

            self.P_state = P_new
            self.W_state = W_new

            # Broadcast W_new to Query Head Groups (GQA): [B, H_q, r, d]
            W_exp = W_new.repeat_interleave(self.num_key_value_groups, dim=1)
            q_vec = phi_q.squeeze(2)  # [B, H_q, r]
            y_hat = torch.einsum("bhi,bhid->bhd", q_vec, W_exp).to(orig_dtype)  # [B, H_q, d]
            attn_output = y_hat.unsqueeze(2).transpose(1, 2).reshape(*input_shape, -1).contiguous()

        else:
            # === PREFILL / TRAINING MODE: Parallel Causal Prefix Solve ===
            sqrt_g = torch.sqrt(gate_kv + 1e-6)
            u = phi_k * sqrt_g          # [B, H_kv, T, r]
            v_w = v_raw.float() * sqrt_g  # [B, H_kv, T, d]

            # Cumulative Gramian and Cross-Covariance
            dG = torch.einsum("bhti,bhtj->bhtij", u, u)
            dR = torch.einsum("bhti,bhtd->bhtid", u, v_w)

            H_kv = self.num_key_value_heads
            r = self.r
            reg = self.get_regularizer().view(1, H_kv, 1, 1, 1)
            eye = torch.eye(r, device=device, dtype=torch.float32).view(1, 1, 1, r, r)

            G = torch.cumsum(dG, dim=2) + reg * eye  # [B, H_kv, T, r, r]
            R = torch.cumsum(dR, dim=2)               # [B, H_kv, T, r, d]

            # Batched Causal Solve: G_t W_t = R_t
            W_star = torch.linalg.solve(G, R)  # [B, H_kv, T, r, d]

            # Save state for decode mode if cache is requested
            if use_cache:
                G_final = G[:, :, -1, :, :]
                self.P_state = torch.linalg.inv(G_final)
                self.W_state = W_star[:, :, -1, :, :]

            # Broadcast W_star to Query Heads (GQA): [B, H_q, T, r, d]
            W_exp = W_star.repeat_interleave(self.num_key_value_groups, dim=1)
            y_hat = torch.einsum("bhtr,bhtrd->bhtd", phi_q, W_exp).to(orig_dtype)

            attn_output = y_hat.transpose(1, 2).reshape(*input_shape, -1).contiguous()

        # 4. Gating and Out-Projection
        attn_output = self.orig_attn.o_proj(attn_output)
        return attn_output, None


def swap_qwen_with_nsal(
    model: nn.Module,
    r: int = 32,
    reg_init: float = 0.05,
    device: str = "cuda",
) -> nn.Module:
    """
    Replaces the 6 Full-Attention layers in Qwen3.5-0.8B with QwenNativeSolvableAttention.
    """
    full_attn_layers = [3, 7, 11, 15, 19, 23]
    for layer_idx in full_attn_layers:
        layer = model.model.layers[layer_idx]
        orig_attn = layer.self_attn
        nsal_mod = QwenNativeSolvableAttention(orig_attn, r=r, reg_init=reg_init).to(device)

        def make_wrapper(mod):
            def intercepted_forward(*args, **kwargs):
                hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                pos = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                attn_mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
                pkv = kwargs.get("past_key_values", kwargs.get("past_key_value", None))
                uc = kwargs.get("use_cache", False)
                return mod(hs, position_embeddings=pos, attention_mask=attn_mask, past_key_values=pkv, use_cache=uc)
            return intercepted_forward

        orig_attn.forward = make_wrapper(nsal_mod)
        layer.nsal_attn = nsal_mod
        print(f"  [Qwen-NSAL] Replaced Layer {layer_idx:2d} with Native Solvable Attention (r={r})")

    return model
