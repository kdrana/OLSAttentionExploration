"""
Full Native Solvable Attention Language Model (NSAL-LM).
=========================================================
A complete, standalone Causal Language Model constructed entirely from
Native Solvable Attention Layers (NSAL), achieving true O(1) inference memory
(< 400 KB total cache across all layers) with zero Softmax and zero RoPE.

Architecture:
- Token & Learned Positional Embeddings
- Stack of L NSAL Transformer Blocks (RMSNorm + NSAL + SwiGLU/MLP)
- Dual-Mode Execution:
  * Mode A (Training / Prefill): Parallel batched causal normal equation solve.
  * Mode B (Autoregressive Decode): Exact O(1) Recursive Least Squares (RLS).
- Final RMSNorm + Tied Linear LM Head
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.native_solvable_attention import NativeSolvableAttention


class NSALTransformerBlock(nn.Module):
    """
    Transformer block consisting of Native Solvable Attention + MLP with RMSNorm residuals.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        r: int = 32,
        d_ff: Optional[int] = None,
        reg_init: float = 0.05,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        d_ff = d_ff or (4 * d_model)
        
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = NativeSolvableAttention(
            d_model=d_model,
            n_heads=n_heads,
            r=r,
            reg_init=reg_init,
            dropout=dropout,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN residual connection for attention
        normed_x = self.norm1(x)
        attn_out, new_state = self.attn(normed_x, state=state, use_causal_mask=use_causal_mask)
        x = x + attn_out
        
        # Pre-LN residual connection for MLP
        x = x + self.mlp(self.norm2(x))
        return x, new_state


class NSALLanguageModel(nn.Module):
    """
    Full Native Solvable Attention Causal Language Model.
    
    Args:
        vocab_size (int): Size of the token vocabulary.
        d_model (int): Hidden dimension size (default: 256).
        n_layers (int): Number of transformer layers (default: 6).
        n_heads (int): Number of attention heads (default: 8).
        r (int): Rank of the solvable subspace projection (default: 32).
        max_seq_len (int): Maximum sequence length for learned positions (default: 4096).
        d_ff (int, optional): Intermediate MLP dimension.
        reg_init (float): Initial regularizer value (default: 0.05).
        dropout (float): Dropout probability (default: 0.0).
        tie_weights (bool): Whether to tie input embedding and output projection (default: True).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        r: int = 32,
        max_seq_len: int = 4096,
        d_ff: Optional[int] = None,
        reg_init: float = 0.05,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.r = r
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Layers
        self.blocks = nn.ModuleList([
            NSALTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                r=r,
                d_ff=d_ff,
                reg_init=reg_init,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Final Norm and Head
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_decode_states(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initializes empty O(1) decode states for all layers."""
        return [block.attn.init_decode_state(batch_size, device, dtype) for block in self.blocks]

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_offset: int = 0,
        use_causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs tensor of shape [B, T].
            states: Optional list of past layer states [(P_l, W_l), ...] for decode mode.
            position_offset: Current sequence position offset for positional embeddings during decode.
            use_causal_mask: Whether to enforce causal sequence dependencies.
            
        Returns:
            logits: Logits tensor of shape [B, T, vocab_size].
            new_states: List of updated layer states.
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Position slice: [1, T, d_model]
        pos = self.pos_emb[:, position_offset : position_offset + T, :]
        x = self.drop(self.tok_emb(input_ids) + pos)
        
        new_states = []
        for i, block in enumerate(self.blocks):
            state_i = states[i] if states is not None else None
            x, new_state_i = block(x, state=state_i, use_causal_mask=use_causal_mask)
            new_states.append(new_state_i)
            
        logits = self.lm_head(self.final_norm(x))
        return logits, new_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generates text autoregressively using pure O(1) Recursive Least Squares (RLS) state updates.
        Never stores past tokens in memory.
        """
        self.eval()
        B, T_prompt = input_ids.shape
        device = input_ids.device
        dtype = self.tok_emb.weight.dtype
        
        # 1. Prefill Prompt: processes all prompt tokens in parallel causal solve
        # Returns final states at timestep T_prompt
        _, states = self.forward(input_ids, states=None, position_offset=0, use_causal_mask=True)
        
        generated = input_ids
        curr_token = input_ids[:, -1:]
        pos_offset = T_prompt
        
        for _ in range(max_new_tokens):
            logits, states = self.forward(curr_token, states=states, position_offset=pos_offset, use_causal_mask=True)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            curr_token = next_token
            pos_offset += 1
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
                
        return generated
