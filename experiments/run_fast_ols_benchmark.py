"""
EXP-13: Fast Analytical Autograd & Block-Diagonal Subspace Acceleration Benchmark.
Compares:
1. Softmax SLM (Llama-Nano-12M) - FlashAttention-2 Baseline
2. Flash-OLS SLM (FlashOLS-Nano-12M Baseline V2) - Dense r=32 Cholesky
3. Fast-Flash-OLS SLM (FlashOLS-Nano-12M V3) - Block-Diagonal 2x16 + Analytical Closed-Form Autograd

All three models feature exact 1:1 parameter parity: 9,946,368 trainable parameters.
"""

import json
import os
import time
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer


# =============================================================================
# 1. ANALYTICAL CLOSED-FORM OLS AUTOGRAD SOLVER
# =============================================================================

class CausalOLSSolveFunction(torch.autograd.Function):
    """
    Exact analytical implicit differentiation for causal regularized OLS normal equations.
    Forward: W* = G^{-1} R via batched Cholesky.
    Backward: dR = G^{-1} dW, dG = -0.5 * (dR W^T + W dR^T) via a single batched solve.
    Completely eliminates PyTorch's generic autograd Cholesky backward overhead.
    """
    @staticmethod
    def forward(ctx, G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        L = torch.linalg.cholesky(G)
        W = torch.cholesky_solve(R, L)
        ctx.save_for_backward(L, W)
        return W

    @staticmethod
    def backward(ctx, grad_W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L, W = ctx.saved_tensors
        # Solve G^{-1} grad_W in one single triangular pass reusing forward Cholesky factor L:
        grad_R = torch.cholesky_solve(grad_W, L)
        # Symmetrized gradient with respect to G:
        grad_G = -0.5 * (torch.matmul(grad_R, W.transpose(-1, -2)) + torch.matmul(W, grad_R.transpose(-1, -2)))
        return grad_G, grad_R


# =============================================================================
# 2. CORE NEURAL COMPONENTS (RMSNorm, RoPE, SwiGLU)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0, device: torch.device = None) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    B, H, T, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, -1, 2))
    freqs = freqs_cis[:T].view(1, 1, T, -1)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, T, d)
    return x_rotated.type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# =============================================================================
# 3. ATTENTION MODULES: SOFTMAX, BASELINE FLASH-OLS, AND FAST FLASH-OLS V3
# =============================================================================

class LlamaCausalSoftmaxAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class LlamaFlashOLSAttention(nn.Module):
    """Dense r=32 Flash-OLS V2 Baseline."""
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64, rank: int = 32, lambda_reg: float = 0.05, gamma: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.d_head = d_model // n_heads
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.gamma = gamma

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        d_raw_feat = self.d_head * 2
        proj = torch.randn(d_raw_feat, rank)
        q_orth, _ = torch.linalg.qr(proj)
        self.subspace_proj = nn.Parameter(q_orth, requires_grad=False)

        chunk_p = torch.arange(chunk_size, dtype=torch.float32)
        decay_w = (gamma ** (chunk_size - 1 - chunk_p)).view(1, 1, 1, chunk_size, 1)
        self.register_buffer("decay_weights", decay_w)
        self.register_buffer("decay_weights_sqrt", torch.sqrt(decay_w))
        self.register_buffer("c_scale_down", (gamma ** (-chunk_p)).view(1, 1, 1, chunk_size, 1, 1))
        self.register_buffer("c_scale_up", (gamma ** chunk_p).view(1, 1, 1, chunk_size, 1, 1))
        self.register_buffer("decay_to_token", (gamma ** (chunk_p + 1)).view(1, 1, 1, chunk_size, 1, 1))
        self.register_buffer("eye_r", torch.eye(rank, dtype=torch.float32).view(1, 1, 1, 1, rank, rank))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.cat([x, torch.tanh(x ** 2)], dim=-1)
        return torch.matmul(phi, self.subspace_proj.to(phi.dtype))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B, T, C_dim = x.shape
        H, dh, C, r = self.n_heads, self.d_head, self.chunk_size, self.rank

        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        Nc = (T + C - 1) // C
        T_pad = Nc * C
        pad_len = T_pad - T
        if pad_len > 0:
            phi_k = torch.cat([phi_k, torch.zeros(B, H, pad_len, r, device=x.device, dtype=orig_dtype)], dim=2)
            v = torch.cat([v, torch.zeros(B, H, pad_len, dh, device=x.device, dtype=orig_dtype)], dim=2)
            phi_q = torch.cat([phi_q, torch.zeros(B, H, pad_len, r, device=x.device, dtype=orig_dtype)], dim=2)

        phi_k_c = phi_k.view(B, H, Nc, C, r).float()
        v_c = v.view(B, H, Nc, C, dh).float()
        phi_q_c = phi_q.view(B, H, Nc, C, r).float()

        K_weighted = phi_k_c * self.decay_weights_sqrt
        G_chunk_end = torch.matmul(K_weighted.transpose(-1, -2), K_weighted)
        R_chunk_end = torch.matmul((phi_k_c * self.decay_weights).transpose(-1, -2), v_c)

        alpha = self.gamma ** C
        G_inter = torch.zeros(B, H, Nc, r, r, device=x.device, dtype=torch.float32)
        R_inter = torch.zeros(B, H, Nc, r, dh, device=x.device, dtype=torch.float32)
        curr_G = torch.zeros(B, H, r, r, device=x.device, dtype=torch.float32)
        curr_R = torch.zeros(B, H, r, dh, device=x.device, dtype=torch.float32)

        for c in range(Nc):
            G_inter[:, :, c] = curr_G
            R_inter[:, :, c] = curr_R
            curr_G = alpha * curr_G + G_chunk_end[:, :, c]
            curr_R = alpha * curr_R + R_chunk_end[:, :, c]

        Kc_outer = torch.matmul(phi_k_c.unsqueeze(-1), phi_k_c.unsqueeze(-2))
        KVc_outer = torch.matmul(phi_k_c.unsqueeze(-1), v_c.unsqueeze(-2))

        Gc_local = self.c_scale_up * torch.cumsum(Kc_outer * self.c_scale_down, dim=3)
        Rc_local = self.c_scale_up * torch.cumsum(KVc_outer * self.c_scale_down, dim=3)

        Gc_full = self.decay_to_token * G_inter.unsqueeze(3) + Gc_local + self.lambda_reg * self.eye_r
        Rc_full = self.decay_to_token * R_inter.unsqueeze(3) + Rc_local

        total_batch = B * H * Nc * C
        Lc = torch.linalg.cholesky(Gc_full.reshape(total_batch, r, r))
        Wc = torch.cholesky_solve(Rc_full.reshape(total_batch, r, dh), Lc).view(B, H, Nc, C, r, dh)
        yc = torch.matmul(phi_q_c.unsqueeze(-2), Wc).squeeze(-2)
        y_out = yc.view(B, H, T_pad, dh)

        y = y_out[:, :, :T].to(orig_dtype).transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.out_proj(y)


class FastLlamaFlashOLSAttention(nn.Module):
    """
    Fast Flash-OLS V3:
    1. 2x16 Block-Diagonal Active Subspace: 4x FLOP reduction over 32x32 Cholesky.
    2. Analytical Closed-Form Custom Autograd (CausalOLSSolveFunction): Eliminates PyTorch's generic cholesky backward.
    3. Fused tensor layouts: Batches blocks directly into effective head dimension.
    """
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64, rank: int = 16, lambda_reg: float = 0.05, gamma: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.d_head = d_model // n_heads
        self.rank = rank
        self.n_blocks = 2
        self.dh_sub = self.d_head // self.n_blocks
        self.lambda_reg = lambda_reg
        self.gamma = gamma

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        d_raw = self.d_head * 2
        proj = torch.randn(d_raw, self.n_blocks * self.rank)
        q_orth, _ = torch.linalg.qr(proj)
        self.subspace_proj = nn.Parameter(q_orth.view(d_raw, self.n_blocks, self.rank), requires_grad=False)

        chunk_p = torch.arange(chunk_size, dtype=torch.float32)
        decay_w = (gamma ** (chunk_size - 1 - chunk_p)).view(1, 1, 1, chunk_size, 1)
        self.register_buffer("decay_weights", decay_w)
        self.register_buffer("decay_weights_sqrt", torch.sqrt(decay_w))
        self.register_buffer("c_scale_down", (gamma ** (-chunk_p)).view(1, 1, 1, chunk_size, 1, 1))
        self.register_buffer("c_scale_up", (gamma ** chunk_p).view(1, 1, 1, chunk_size, 1, 1))
        self.register_buffer("decay_to_token", (gamma ** (chunk_p + 1)).view(1, 1, 1, chunk_size, 1, 1))
        self.register_buffer("eye_r", torch.eye(rank, dtype=torch.float32).view(1, 1, 1, 1, rank, rank))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.cat([x, torch.tanh(x ** 2)], dim=-1)
        return torch.einsum("bhtd,dkr->bhtkr", phi, self.subspace_proj)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B, T, C_dim = x.shape
        H, dh, C, r = self.n_heads, self.d_head, self.chunk_size, self.rank

        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        H_eff = H * self.n_blocks
        phi_q = phi_q.permute(0, 1, 3, 2, 4).reshape(B, H_eff, T, r)
        phi_k = phi_k.permute(0, 1, 3, 2, 4).reshape(B, H_eff, T, r)
        v_sub = v.view(B, H, T, self.n_blocks, self.dh_sub).permute(0, 1, 3, 2, 4).reshape(B, H_eff, T, self.dh_sub)

        Nc = (T + C - 1) // C
        T_pad = Nc * C
        pad_len = T_pad - T
        if pad_len > 0:
            phi_k = torch.cat([phi_k, torch.zeros(B, H_eff, pad_len, r, device=x.device, dtype=orig_dtype)], dim=2)
            v_sub = torch.cat([v_sub, torch.zeros(B, H_eff, pad_len, self.dh_sub, device=x.device, dtype=orig_dtype)], dim=2)
            phi_q = torch.cat([phi_q, torch.zeros(B, H_eff, pad_len, r, device=x.device, dtype=orig_dtype)], dim=2)

        phi_k_c = phi_k.view(B, H_eff, Nc, C, r).float()
        v_c = v_sub.view(B, H_eff, Nc, C, self.dh_sub).float()
        phi_q_c = phi_q.view(B, H_eff, Nc, C, r).float()

        K_weighted = phi_k_c * self.decay_weights_sqrt
        G_chunk_end = torch.matmul(K_weighted.transpose(-1, -2), K_weighted)
        R_chunk_end = torch.matmul((phi_k_c * self.decay_weights).transpose(-1, -2), v_c)

        alpha = self.gamma ** C
        G_inter = torch.zeros(B, H_eff, Nc, r, r, device=x.device, dtype=torch.float32)
        R_inter = torch.zeros(B, H_eff, Nc, r, self.dh_sub, device=x.device, dtype=torch.float32)
        curr_G = torch.zeros(B, H_eff, r, r, device=x.device, dtype=torch.float32)
        curr_R = torch.zeros(B, H_eff, r, self.dh_sub, device=x.device, dtype=torch.float32)

        for c in range(Nc):
            G_inter[:, :, c] = curr_G
            R_inter[:, :, c] = curr_R
            curr_G = alpha * curr_G + G_chunk_end[:, :, c]
            curr_R = alpha * curr_R + R_chunk_end[:, :, c]

        Kc_outer = torch.matmul(phi_k_c.unsqueeze(-1), phi_k_c.unsqueeze(-2))
        KVc_outer = torch.matmul(phi_k_c.unsqueeze(-1), v_c.unsqueeze(-2))

        Gc_local = self.c_scale_up * torch.cumsum(Kc_outer * self.c_scale_down, dim=3)
        Rc_local = self.c_scale_up * torch.cumsum(KVc_outer * self.c_scale_down, dim=3)

        Gc_full = self.decay_to_token * G_inter.unsqueeze(3) + Gc_local + self.lambda_reg * self.eye_r
        Rc_full = self.decay_to_token * R_inter.unsqueeze(3) + Rc_local

        total_batch = B * H_eff * Nc * C
        # Fast custom analytical solve!
        Wc = CausalOLSSolveFunction.apply(Gc_full.reshape(total_batch, r, r), Rc_full.reshape(total_batch, r, self.dh_sub)).view(B, H_eff, Nc, C, r, self.dh_sub)
        yc = torch.matmul(phi_q_c.unsqueeze(-2), Wc).squeeze(-2)
        y_out = yc.view(B, H_eff, T_pad, self.dh_sub)

        # Un-merge H_eff -> (B, H, T_pad, dh)
        y_out = y_out.view(B, H, self.n_blocks, T_pad, self.dh_sub).permute(0, 1, 3, 2, 4).reshape(B, H, T_pad, dh)
        y = y_out[:, :, :T].to(orig_dtype).transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.out_proj(y)


# =============================================================================
# 4. FULL 12.3M PARAMETER LANGUAGE MODEL ARCHITECTURE
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, attn_type: str = "softmax"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if attn_type == "softmax":
            self.attn = LlamaCausalSoftmaxAttention(d_model, n_heads)
        elif attn_type == "flash_ols":
            self.attn = LlamaFlashOLSAttention(d_model, n_heads, chunk_size=64, rank=32)
        elif attn_type == "fast_flash_ols":
            self.attn = FastLlamaFlashOLSAttention(d_model, n_heads, chunk_size=64, rank=16)
        elif attn_type == "fused_triton_ols":
            from src.fused_ols_triton import FusedLlamaFlashOLSAttention
            self.attn = FusedLlamaFlashOLSAttention(d_model, n_heads, rank=16)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


class SmallLanguageModel12M(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        n_layers: int = 10,
        n_heads: int = 8,
        d_ffn: int = 680,
        max_seq_len: int = 1024,
        attn_type: str = "fast_flash_ols",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.d_head = d_model // n_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ffn, attn_type=attn_type)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.register_buffer("freqs_cis", precompute_rope_freqs_cis(self.d_head, max_seq_len))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx)
        freqs_cis = self.freqs_cis[:T].to(x.device)

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.final_norm(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 32, temperature: float = 0.7) -> torch.Tensor:
        self.eval()
        idx = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        self.train()
        return idx


# =============================================================================
# 5. DATA LOADER & STREAMING BPE PREPARATION
# =============================================================================

def prepare_tinystories_data(data_dir: str = "/tmp/tinystories_data") -> tuple[np.ndarray, np.ndarray, Tokenizer]:
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "TinyStories-valid.txt")
    tokenizer_path = os.path.join(data_dir, "bpe_8k.json")
    train_path = os.path.join(data_dir, "train_ids.npy")
    val_path = os.path.join(data_dir, "val_ids.npy")

    if not os.path.exists(raw_path):
        print(f"[Data] Downloading TinyStories-valid.txt (19.4 MB) to {raw_path}...")
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        urllib.request.urlretrieve(url, raw_path)
        print("[Data] Download complete!")

    if not os.path.exists(tokenizer_path):
        print("[Tokenizer] Training 8,192 Byte-level BPE tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[raw_path],
            vocab_size=8192,
            min_frequency=2,
            special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<bos>", "<eos>"]
        )
        tokenizer.save(tokenizer_path)
        print("[Tokenizer] Saved tokenizer to", tokenizer_path)

    tok = Tokenizer.from_file(tokenizer_path)

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[Data] Tokenizing TinyStories into memory-safe numpy arrays...")
        token_chunks = []
        batch_lines = []
        max_tokens = 2_000_000

        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                batch_lines.append(line)
                if len(batch_lines) >= 1000:
                    text_block = "".join(batch_lines)
                    enc = tok.encode(text_block)
                    token_chunks.extend(enc.ids)
                    batch_lines = []
                    if len(token_chunks) >= max_tokens:
                        break

        if batch_lines and len(token_chunks) < max_tokens:
            text_block = "".join(batch_lines)
            enc = tok.encode(text_block)
            token_chunks.extend(enc.ids)

        all_ids = np.array(token_chunks[:max_tokens], dtype=np.uint16)
        n_val = int(len(all_ids) * 0.15)
        train_ids = all_ids[:-n_val]
        val_ids = all_ids[-n_val:]
        np.save(train_path, train_ids)
        np.save(val_path, val_ids)
        print(f"[Data] Created Train ({len(train_ids):,} tokens) and Val ({len(val_ids):,} tokens).")
    else:
        train_ids = np.load(train_path)
        val_ids = np.load(val_path)

    return train_ids, val_ids, tok


class TokenDataset:
    def __init__(self, data: np.ndarray, seq_len: int, batch_size: int):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size

    def get_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_idx = len(self.data) - self.seq_len - 1
        starts = np.random.randint(0, max_idx, size=self.batch_size)
        x = np.stack([self.data[s : s + self.seq_len] for s in starts])
        y = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return (
            torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device),
        )


# =============================================================================
# 6. BENCHMARK EXECUTION HARNESS
# =============================================================================

@torch.no_grad()
def evaluate_validation(model: nn.Module, val_dataset: TokenDataset, device: torch.device, eval_steps: int = 15) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    for _ in range(eval_steps):
        x, y = val_dataset.get_batch(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += float(loss.item())
    model.train()
    mean_loss = total_loss / eval_steps
    ppl = float(np.exp(mean_loss))
    return mean_loss, ppl


def run_fast_ols_benchmark(
    device: torch.device,
    total_steps: int = 350,
    batch_size: int = 8,
    seq_len: int = 256,
    lr: float = 1e-3,
    checkpoints: list[int] = [0, 25, 50, 100, 150, 200, 250, 300, 350],
    qual_checkpoints: list[int] = [50, 100, 200, 350],
    target_models: list[str] = None,
) -> dict:
    print("=" * 80)
    print("EXP-13: Fast Analytical Autograd & Block-Diagonal Subspace Benchmark")
    print(f"Device: {device} | Total Steps: {total_steps} | Batch: {batch_size} | Seq Len: {seq_len}")
    print("=" * 80)

    train_ids, val_ids, tokenizer = prepare_tinystories_data()
    train_dataset = TokenDataset(train_ids, seq_len=seq_len, batch_size=batch_size)
    val_dataset = TokenDataset(val_ids, seq_len=seq_len, batch_size=batch_size)

    prompt_text = "Once upon a time, there was a little bird named Pip who lived in a"
    prompt_tokens = torch.tensor(tokenizer.encode(prompt_text).ids, dtype=torch.long, device=device).unsqueeze(0)

    all_models = {
        "Softmax SLM (Llama-Nano)": lambda: SmallLanguageModel12M(vocab_size=8192, d_model=256, n_layers=10, n_heads=8, d_ffn=680, attn_type="softmax").to(device),
        "Flash-OLS SLM (Baseline V2)": lambda: SmallLanguageModel12M(vocab_size=8192, d_model=256, n_layers=10, n_heads=8, d_ffn=680, attn_type="flash_ols").to(device),
        "Fast-Flash-OLS SLM (Fast V3)": lambda: SmallLanguageModel12M(vocab_size=8192, d_model=256, n_layers=10, n_heads=8, d_ffn=680, attn_type="fast_flash_ols").to(device),
        "Fused-Flash-OLS SLM (Triton V4)": lambda: SmallLanguageModel12M(vocab_size=8192, d_model=256, n_layers=10, n_heads=8, d_ffn=680, attn_type="fused_triton_ols").to(device),
    }

    def _matches(target, name):
        words = target.lower().replace("_", " ").replace("-", " ").split()
        name_clean = name.lower()
        return any(w in name_clean for w in words if len(w) > 3)

    selected_names = [k for k in all_models if target_models is None or any(_matches(t, k) for t in target_models)]
    models = {k: all_models[k]() for k in selected_names}

    results = {}

    for name, model in models.items():
        n_params = model.count_parameters()
        print(f"\n[{name}] Total Trainable Parameters: {n_params:,}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
        
        warmup_steps = 20
        def get_lr(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

        trace = []
        step_times = []

        print(f"Starting {total_steps} pretraining steps for {name}...")
        t_start = time.perf_counter()

        for step in range(total_steps + 1):
            if step in checkpoints:
                val_loss, val_ppl = evaluate_validation(model, val_dataset, device)
                
                sample_text = ""
                if step in qual_checkpoints or step == 0:
                    gen_ids = model.generate(prompt_tokens, max_new_tokens=32, temperature=0.7)
                    sample_text = tokenizer.decode(gen_ids[0].tolist())

                current_entry = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "sample_generation": sample_text,
                }
                if step > 0 and trace:
                    current_entry["train_loss"] = trace[-1].get("train_loss", None)
                    current_entry["grad_norm"] = trace[-1].get("grad_norm", None)
                trace.append(current_entry)

                print(f"  Step {step:4d}/{total_steps} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:7.2f}")
                if sample_text:
                    clean_gen = sample_text.replace('\n', ' ')[:100]
                    print(f"    Sample: \"{clean_gen}...\"")

            if step == total_steps:
                break

            t0 = time.perf_counter()
            x, y = train_dataset.get_batch(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
            optimizer.step()
            scheduler.step()

            step_time = (time.perf_counter() - t0) * 1000.0
            step_times.append(step_time)

            if trace and trace[-1]["step"] == step:
                trace[-1]["train_loss"] = float(loss.item())
                trace[-1]["grad_norm"] = grad_norm
                trace[-1]["step_time_ms"] = step_time

        total_time_s = time.perf_counter() - t_start
        mean_step_time = float(np.mean(step_times)) if step_times else 0.0
        tokens_per_sec = (total_steps * batch_size * seq_len) / max(total_time_s, 1e-5)

        print(f"[{name}] Finished in {total_time_s:.2f}s | Mean Step Time: {mean_step_time:.1f}ms | Throughput: {tokens_per_sec:,.0f} tok/s")

        results[name] = {
            "n_parameters": n_params,
            "total_time_s": total_time_s,
            "mean_step_time_ms": mean_step_time,
            "tokens_per_sec": tokens_per_sec,
            "trace": trace,
        }

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run_fast_ols_benchmark(device=device, total_steps=2, batch_size=2, seq_len=64, checkpoints=[0, 2], qual_checkpoints=[2])
    print("Self-test completed successfully!")
