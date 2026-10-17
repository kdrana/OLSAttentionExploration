"""
Native Solvable Attention Layer (NSAL)
=====================================
A drop-in replacement for standard Multi-Head Attention that eliminates
the KV-cache memory explosion and RoPE rotatory interference by casting
attention as an online, end-to-end differentiable Ordinary Least Squares (OLS)
operator estimation problem.

Key Mathematical Properties:
1. Dual-Mode Execution:
   - Parallel / Causal Prefix Solve during prefill and training (exact gradient flow).
   - O(1) Recursive Least Squares (RLS) state updates during autoregressive generation.
2. Invariant Subspace Projection:
   - Uses stationary semantic coordinates with learned low-rank projection phi(k), phi(q).
   - Learned salience admission gate g_t in [0, 1] weights critical tokens (beacons, keys).
3. Whiten-and-De-correlate:
   - The regularized inverse Gramian (G + lambda I)^{-1} orthogonalizes redundant keys,
     preventing eigenvalue collapse and outperforming standard Linear Attention on associative recall.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeSolvableAttention(nn.Module):
    """
    Native Solvable Attention Layer (NSAL).
    
    Args:
        d_model (int): Model embedding dimension.
        n_heads (int): Number of attention heads.
        r (int): Rank of the solvable subspace projection (default: 32).
        reg_init (float): Initial Tikhonov regularization lambda (default: 0.01).
        use_gate (bool): Whether to use a learned salience admission gate (default: True).
        dropout (float): Dropout probability on output projection (default: 0.0).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        r: int = 32,
        reg_init: float = 0.01,
        use_gate: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.r = min(r, self.d_k)
        self.use_gate = use_gate
        
        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Subspace coordinate projection matrices: R^{d_k -> r}
        self.proj_k = nn.Parameter(torch.randn(n_heads, self.d_k, self.r) * (1.0 / math.sqrt(self.d_k)))
        self.proj_q = nn.Parameter(torch.randn(n_heads, self.d_k, self.r) * (1.0 / math.sqrt(self.d_k)))
        
        # Learned admission gate: assigns salience weight g_t in [0, 1] per token per head
        if self.use_gate:
            self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
            # Initialize bias slightly positive so initial gate allows information through (~0.7)
            nn.init.constant_(self.gate_proj.bias, 1.0)
            
        # Learnable log regularization per head
        self.log_reg = nn.Parameter(torch.full((n_heads, 1, 1), math.log(reg_init)))
        
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def get_regularizer(self) -> torch.Tensor:
        """Returns per-head regularizer lambda = exp(log_reg) clamped for numerical safety."""
        return torch.clamp(torch.exp(self.log_reg), min=1e-5, max=10.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Native Solvable Attention.
        
        Args:
            x (torch.Tensor): Input sequence [B, T, D].
            state (Tuple[torch.Tensor, torch.Tensor], optional):
                Past autoregressive state (P_prev, W_prev) for O(1) decode mode.
                - P: Inverse Gramian [B, H, r, r]
                - W: Operator state [B, H, r, d_k]
            use_causal_mask (bool): Whether to enforce causal sequence dependencies.
            
        Returns:
            out (torch.Tensor): Attention output [B, T, D].
            new_state (Tuple[torch.Tensor, torch.Tensor], optional): Updated state for decode mode.
        """
        B, T, D = x.shape
        H = self.n_heads
        d_k = self.d_k
        r = self.r
        
        # Project inputs to Q, K, V: [B, T, H, d_k] -> transpose to [B, H, T, d_k]
        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d_k).transpose(1, 2)
        
        # Salience admission gate: [B, H, T, 1]
        if self.use_gate:
            g = torch.sigmoid(self.gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        else:
            g = torch.ones(B, H, T, 1, device=x.device, dtype=x.dtype)
            
        # Map key and query to solvable subspace R^r
        # k: [B, H, T, d_k], proj_k: [H, d_k, r] -> phi_k: [B, H, T, r]
        phi_k = torch.einsum("bhtd,hdr->bhtr", k, self.proj_k)
        phi_q = torch.einsum("bhtd,hdr->bhtr", q, self.proj_q)
        
        # Decode Mode (T == 1 and state is provided)
        if T == 1 and state is not None:
            return self._step_decode(phi_q, phi_k, v, g, state)
            
        # Prefill / Training Mode (Parallel Causal Inversion)
        out, new_state = self._parallel_causal_solve(phi_q, phi_k, v, g, use_causal_mask)
        
        # RMSNorm and output projection
        out = self.out_proj(self.dropout(self.norm(out)))
        return out, new_state

    def _parallel_causal_solve(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        use_causal_mask: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes parallel causal solvable attention via cumulative normal equations.
        Exact vector-Jacobian products flow through torch.linalg.solve.
        """
        B, H, T, r = phi_k.shape
        d_k = self.d_k
        
        # Apply gate weights: u_t = sqrt(g_t) * phi_k(t), v_w = sqrt(g_t) * v_t
        sqrt_g = torch.sqrt(g + 1e-6)
        u = phi_k * sqrt_g            # [B, H, T, r]
        v_w = v * sqrt_g              # [B, H, T, d_k]
        
        # Outer products
        # dG: u_t * u_t^T -> [B, H, T, r, r]
        dG = torch.einsum("bhti,bhtj->bhtij", u, u)
        # dR: u_t * v_w^T -> [B, H, T, r, d_k]
        dR = torch.einsum("bhti,bhtd->bhtid", u, v_w)
        
        reg = self.get_regularizer()  # [H, 1, 1]
        eye = torch.eye(r, device=phi_k.device, dtype=phi_k.dtype).view(1, 1, 1, r, r)
        
        if use_causal_mask:
            # Prefix cumulative sum enforces causal history: t only sees s <= t
            G = torch.cumsum(dG, dim=2) + reg.view(1, H, 1, 1, 1) * eye
            R = torch.cumsum(dR, dim=2)
            
            # Batched solve: G_t W_t = R_t across all (B, H, T)
            # PyTorch solves all (B * H * T) r x r linear systems in parallel
            W_star = torch.linalg.solve(G, R)  # [B, H, T, r, d_k]
            
            # Query retrieval: y_hat = phi_q * W_star -> [B, H, T, d_k]
            y_hat = torch.einsum("bhtr,bhtrd->bhtd", phi_q, W_star)
            
            # Prepare state for subsequent decoding from the final timestep
            G_final = G[:, :, -1, :, :]  # [B, H, r, r]
            P_final = torch.linalg.inv(G_final)
            W_final = W_star[:, :, -1, :, :]  # [B, H, r, d_k]
            final_state = (P_final, W_final)
        else:
            # Bidirectional / Non-causal: pool across all T
            G_total = torch.sum(dG, dim=2, keepdim=True) + reg.view(1, H, 1, 1, 1) * eye
            R_total = torch.sum(dR, dim=2, keepdim=True)
            W_star = torch.linalg.solve(G_total, R_total)
            y_hat = torch.einsum("bhtr,bh1rd->bhtd", phi_q, W_star)
            
            P_final = torch.linalg.inv(G_total.squeeze(2))
            W_final = W_star.squeeze(2)
            final_state = (P_final, W_final)
            
        # Transpose back: [B, H, T, d_k] -> [B, T, H * d_k]
        out = y_hat.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return out, final_state

    def _step_decode(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        O(1) Step-by-Step Autoregressive Decode using Recursive Least Squares (RLS).
        
        Args:
            phi_q: [B, H, 1, r]
            phi_k: [B, H, 1, r]
            v: [B, H, 1, d_k]
            g: [B, H, 1, 1]
            state: (P_prev, W_prev)
                P_prev: [B, H, r, r]
                W_prev: [B, H, r, d_k]
        """
        B, H, _, r = phi_k.shape
        d_k = self.d_k
        P_prev, W_prev = state
        
        sqrt_g = torch.sqrt(g + 1e-6)
        u = (phi_k * sqrt_g).squeeze(2)      # [B, H, r]
        v_w = (v * sqrt_g).squeeze(2)        # [B, H, d_k]
        
        # Gain vector: k = (P_prev @ u) / (1 + u^T @ P_prev @ u)
        Pu = torch.einsum("bhij,bhj->bhi", P_prev, u)  # [B, H, r]
        denom = 1.0 + torch.einsum("bhi,bhi->bh", u, Pu).unsqueeze(-1)  # [B, H, 1]
        k_gain = Pu / denom  # [B, H, r]
        
        # Prediction error: e = v_w - u @ W_prev
        pred = torch.einsum("bhi,bhid->bhd", u, W_prev)  # [B, H, d_k]
        e = v_w - pred                                   # [B, H, d_k]
        
        # Operator update: W_new = W_prev + k_gain (x) e
        W_new = W_prev + torch.einsum("bhi,bhd->bhid", k_gain, e)  # [B, H, r, d_k]
        
        # Covariance update (Sherman-Morrison): P_new = P_prev - k_gain (x) Pu
        P_new = P_prev - torch.einsum("bhi,bhj->bhij", k_gain, Pu)  # [B, H, r, r]
        
        # Query output: y_hat = phi_q @ W_new
        q_vec = phi_q.squeeze(2)  # [B, H, r]
        y_hat = torch.einsum("bhi,bhid->bhd", q_vec, W_new).unsqueeze(2)  # [B, H, 1, d_k]
        
        out = y_hat.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        out = self.out_proj(self.dropout(self.norm(out)))
        
        return out, (P_new, W_new)

    def init_decode_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes empty state for decode mode."""
        H = self.n_heads
        r = self.r
        d_k = self.d_k
        reg = self.get_regularizer().view(1, H, 1, 1)
        # P_0 = (lambda I)^{-1} = (1 / lambda) * I
        eye = torch.eye(r, device=device, dtype=dtype).view(1, 1, r, r).repeat(batch_size, H, 1, 1)
        P0 = eye / reg
        W0 = torch.zeros(batch_size, H, r, d_k, device=device, dtype=dtype)
        return (P0, W0)
