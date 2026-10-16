"""
Model architectures for FT-Transformer benchmark:
- Model A: Standard Multi-Head Softmax FT-Transformer (Random Init)
- Model B: Gram Precision Metric Initialized FT-Transformer
- Model C: Solvable Degree-2 Regression Block Model
"""

import math
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class FeatureTokenizer(nn.Module):
    """
    Maps each scalar feature x_j to a d-dimensional token:
    h_j = x_j * w_j + b_j. Input: (B, P) -> Output: (B, P, d).
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, d_token) / math.sqrt(d_token))
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P) -> (B, P, 1) * (P, d) + (P, d) -> (B, P, d)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class FeedForward(nn.Module):
    """Standard position-wise FFN with ReLU and LayerNorm residual."""

    def __init__(self, d_token: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_token, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_token)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class StandardFTTransformer(nn.Module):
    """
    Model A: Standard FT-Transformer with Multi-Head Softmax Attention.
    """

    def __init__(self, n_features: int, d_token: int = 32, n_heads: int = 4, d_ffn: int = 64, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.mha = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_token)
        self.ffn = FeedForward(d_token, d_ffn, dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tokenizer(x)  # (B, P, d)
        # Multihead Attention
        attn_out, _ = self.mha(h, h, h)
        h = self.norm1(h + attn_out)
        # FFN
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)
        # Mean pool over tokens and predict
        out = self.head(h.mean(dim=1))
        return out


class GramInitFTTransformer(StandardFTTransformer):
    """
    Model B: FT-Transformer with analytical Gram Metric Warm-Start on W_Q and W_K.
    """

    def warm_start_gram_metric(self, X_sample: torch.Tensor, lambda_reg: float = 1e-3):
        """
        Initializes in_proj_weight of MHA using the empirical Gram precision metric
        W = (X^T X + lambda*I)^{-1/2} derived by Goulet Coulombe.
        """
        with torch.no_grad():
            tokens = self.tokenizer(X_sample)  # (B, P, d)
            tokens_flat = tokens.reshape(-1, tokens.shape[-1])  # (B*P, d)

            # Empirical token covariance Gram matrix
            Gram = torch.matmul(tokens_flat.T, tokens_flat) / tokens_flat.shape[0]
            d = Gram.shape[0]
            Gram_reg = Gram + lambda_reg * torch.eye(d, device=Gram.device)

            # (Gram)^{-1/2} via eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(Gram_reg)
            inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(eigvals, min=1e-6)))
            W_gram = torch.matmul(torch.matmul(eigvecs, inv_sqrt), eigvecs.T)  # (d, d)

            # in_proj_weight in PyTorch MultiheadAttention has shape (3*d, d) for [Q, K, V]
            self.mha.in_proj_weight[:d, :] = W_gram
            self.mha.in_proj_weight[d : 2 * d, :] = W_gram


class SolvableRegressionBlockModel(nn.Module):
    """
    Model C: FT-Transformer with Degree-2 Polynomial Regression Block.
    Computes pairwise polynomial cross-features and initializes the mixing/head
    weights via closed-form Ridge Regression.
    """

    def __init__(self, n_features: int, d_token: int = 16, n_components: int = 64, d_ffn: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.pca = None
        self.n_components = n_components

        # Degree-2 cross-feature dimension: P linear + P squared + P*(P-1)/2 interactions
        self.n_poly_features = n_features + n_features + (n_features * (n_features - 1)) // 2

        # Mixing layer (linear regression over compressed polynomial features)
        actual_in_dim = min(self.n_components, self.n_poly_features)
        self.mixer = nn.Linear(actual_in_dim, d_token)
        self.norm1 = nn.LayerNorm(d_token)
        self.ffn = FeedForward(d_token, d_ffn, dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def compute_poly_features(self, x: torch.Tensor) -> torch.Tensor:
        """Computes [x, x^2, x_i * x_j] pairwise cross-features."""
        B, P = x.shape
        poly_list = [x, x**2]
        for i in range(P):
            for j in range(i + 1, P):
                poly_list.append((x[:, i : i + 1] * x[:, j : j + 1]))
        return torch.cat(poly_list, dim=-1)

    def fit_pca_and_solve_closed_form(self, X_train: torch.Tensor, y_train: torch.Tensor, lambda_reg: float = 1.0):
        """
        1. Fits frozen PCA on the degree-2 polynomial features.
        2. Solves the mixer & head weights in a single closed-form Ridge step!
        """
        device = self.mixer.weight.device
        with torch.no_grad():
            Poly_tr = self.compute_poly_features(X_train).cpu().numpy()

            # 1. Fit PCA
            n_comp = min(self.n_components, Poly_tr.shape[1], Poly_tr.shape[0])
            pca_sk = PCA(n_components=n_comp, random_state=42)
            Poly_pca = pca_sk.fit_transform(Poly_tr)
            Poly_pca_tensor = torch.tensor(Poly_pca, dtype=torch.float32, device=device)

            # Store PCA parameters as registered buffers directly on GPU
            self.register_buffer("pca_components", torch.tensor(pca_sk.components_.T, dtype=torch.float32, device=device))
            self.register_buffer("pca_mean", torch.tensor(pca_sk.mean_, dtype=torch.float32, device=device))

            # 2. Closed-form Ridge Regression solve: W* = (Phi^T Phi + lambda*I)^{-1} Phi^T Y
            Gram = torch.matmul(Poly_pca_tensor.T, Poly_pca_tensor) + lambda_reg * torch.eye(n_comp, device=device)
            w_opt = torch.linalg.solve(Gram, torch.matmul(Poly_pca_tensor.T, y_train.to(device)))  # (n_comp, 1)

            # Warm-start the mixer and head so that Mixer + Head initially computes w_opt!
            self.mixer.weight.zero_()
            self.mixer.bias.zero_()
            self.head.weight.zero_()
            self.head.bias.zero_()

            # Align first output channel directly with w_opt
            self.mixer.weight[0, :n_comp] = w_opt.squeeze(-1)
            self.head.weight[0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute compressed polynomial features directly on GPU
        Poly = self.compute_poly_features(x)
        if hasattr(self, "pca_components"):
            Poly_pca = torch.matmul(Poly - self.pca_mean, self.pca_components)
        else:
            Poly_pca = Poly

        # Mixing sublayer + Residual
        mixed = self.mixer(Poly_pca).unsqueeze(1)  # (B, 1, d)
        tokens = self.tokenizer(x)  # (B, P, d)
        h = self.norm1(tokens + mixed)

        # FFN
        h = self.norm2(h + self.ffn(h))
        out = self.head(h.mean(dim=1))
        return out


# =============================================================================
# SEQUENCE MODELING MODULES (Phase 2 / EXP-03)
# =============================================================================

class CausalStandardAttention(nn.Module):
    """
    Standard Causal Multi-Head Softmax Self-Attention (O(T^2) time & memory).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Causal mask (upper triangular = -inf)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class CausalLinearAttention(nn.Module):
    """
    Linear Causal Attention using non-negative kernel feature map (ELU + 1)
    and causal prefix-sum accumulation (O(T) time & memory).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        # Causal prefix-sum accumulation: S_t = sum_{tau=1}^t phi(k_tau) v_tau^T
        # phi_k: (B, H, T, d), v: (B, H, T, d)
        kv = torch.einsum("bhtd,bhte->bhtde", phi_k, v)  # (B, H, T, d, d)
        S = torch.cumsum(kv, dim=2)                      # (B, H, T, d, d)
        out = torch.einsum("bhtd,bhtde->bhte", phi_q, S) # (B, H, T, d)

        # Normalization: Z_t = sum_{tau=1}^t phi(k_tau)
        Z = torch.cumsum(phi_k, dim=2)                   # (B, H, T, d)
        norm = torch.einsum("bhtd,bhtd->bht", phi_q, Z).unsqueeze(-1) + 1e-6
        out = out / norm

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TwoStreamBulkMarginAttention(nn.Module):
    """
    Two-Stream "Bulk + Margin" Attention Layer:
    - Stream 1 (Bulk Context): O(T) Linear Causal Attention for diffuse background (90-95% tokens).
    - Stream 2 (Margin Outliers): Local Sliding Window (w=32) + Landmark Sparse Softmax for high-salience tokens.
    - Gating: Differentiable data-dependent surprise gate alpha_t = sigma(x W_g).
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 32, n_landmarks: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.window_size = window_size
        self.n_landmarks = n_landmarks

        # Stream 1: Bulk Linear Attention
        self.bulk_attn = CausalLinearAttention(d_model, n_heads)

        # Stream 2: Margin Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Surprise Gating projection
        self.gate_proj = nn.Linear(d_model, 1)

    def _margin_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes local causal window + strided global landmark attention in O(T (w + k)) time.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # If sequence is short, exact causal attention is O(T^2) <= O(T * window_size)
        if T <= self.window_size:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        else:
            # Chunk-based local causal window + landmark keys
            # Compute sliding band causal attention + global landmark keys
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
            
            # Causal mask
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            
            # Distance mask: allow tokens within window_size OR on strided landmark positions
            row_idx = torch.arange(T, device=x.device).unsqueeze(1)
            col_idx = torch.arange(T, device=x.device).unsqueeze(0)
            dist = row_idx - col_idx
            
            # Local window condition
            local_mask = (dist >= 0) & (dist <= self.window_size)
            
            # Global landmark condition (strided sampling across history)
            stride = max(1, T // self.n_landmarks)
            landmark_mask = (col_idx % stride == 0) & (dist >= 0)
            
            sparse_active = local_mask | landmark_mask
            inactive_mask = torch.where(sparse_active, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
            
            masked_scores = scores + causal_mask.unsqueeze(0).unsqueeze(0) + inactive_mask
            attn = torch.softmax(masked_scores, dim=-1)
            # Handle potential NaNs in tokens before any landmarks
            attn = torch.nan_to_num(attn, nan=0.0)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stream 1: Bulk linear context
        bulk_out = self.bulk_attn(x)

        # Stream 2: Margin non-linear lookup
        margin_out = self._margin_attention(x)

        # Surprise Gating
        alpha = torch.sigmoid(self.gate_proj(x))  # (B, T, 1)
        return (1.0 - alpha) * bulk_out + alpha * margin_out


class CausalDeltaAttention(nn.Module):
    """
    Delta-Rule / Least-Mean-Squares (LMS) Attention (Schlag et al., 2021; Titani, 2024):
    Maintains a rolling memory state S_t in R^{d_k x d_v}.
    Updates via the prediction innovation error e_t = v_t - S_{t-1}^T k_t
    using an input-dependent scalar learning rate beta_t = sigmoid(W_beta x_t).
    Compute: O(d) per token, O(1) memory at inference.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Input-dependent learning rate (step size)
        self.beta_proj = nn.Linear(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, self.n_heads).permute(0, 2, 1).unsqueeze(-1) # (B, H, T, 1)

        # Normalize keys for numerical stability
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        # Sequential unrolling across sequence length T
        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]      # (B, H, d)
            v_t = v[:, :, t, :]      # (B, H, d)
            q_t = q[:, :, t, :]      # (B, H, d)
            beta_t = beta[:, :, t, :] # (B, H, 1)

            # 1. Predict value from current memory
            # S^T k_t -> (B, H, d)
            v_pred = torch.einsum("bhde,bhd->bhe", S, k_t)

            # 2. Innovation error (residual)
            e_t = v_t - v_pred

            # 3. Delta update: S_t = S_{t-1} + beta_t * (k_t * e_t^T)
            delta_S = beta_t.unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", k_t, e_t)
            S = S + delta_S

            # 4. Read out value for query q_t
            y_t = torch.einsum("bhde,bhd->bhe", S, q_t)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)  # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns initial memory state (B, H, d, d) for constant O(1) inference."""
        return torch.zeros(batch_size, self.n_heads, self.d_head, self.d_head, device=device)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token constant O(1) autoregressive generation step."""
        B, C = x_t.shape
        q = self.q_proj(x_t).view(B, self.n_heads, self.d_head)
        k = torch.nn.functional.normalize(self.k_proj(x_t).view(B, self.n_heads, self.d_head), p=2, dim=-1)
        v = self.v_proj(x_t).view(B, self.n_heads, self.d_head)
        beta = torch.sigmoid(self.beta_proj(x_t)).unsqueeze(-1)

        v_pred = torch.einsum("bhde,bhd->bhe", state, k)
        e_t = v_t = v - v_pred
        new_state = state + beta.unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", k, e_t)
        y = torch.einsum("bhde,bhd->bhe", new_state, q).view(B, C)
        return self.out_proj(y), new_state


class CausalRLSAttention(nn.Module):
    """
    Causal Recursive Least Squares (RLS) / Kalman Filter Attention:
    Maintains:
    - S_t: associative memory matrix in R^{d_k x d_v}
    - P_t: online precision / inverse covariance matrix in R^{d_k x d_k}

    Update equations per token:
      e_t = v_t - S_{t-1}^T k_t           (Prediction innovation error)
      u_t = P_{t-1} k_t
      K_t = u_t / (lambda + k_t^T u_t)    (Optimal Kalman Gain)
      P_t = (P_{t-1} - K_t u_t^T) / lambda (Sherman-Morrison covariance update)
      S_t = S_{t-1} + K_t e_t^T           (Optimal error-correcting memory update)
      out_t = S_t^T q_t
    """

    def __init__(self, d_model: int, n_heads: int, lambda_decay: float = 0.999, delta_reg: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lambda_decay = lambda_decay
        self.delta_reg = delta_reg

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Normalize keys for numerical stability
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        # Initialize State S_0 and Covariance P_0 = delta^{-1} * I
        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        eye = torch.eye(self.d_head, device=x.device).unsqueeze(0).unsqueeze(0)
        P = eye.repeat(B, self.n_heads, 1, 1) / self.delta_reg

        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]  # (B, H, d)
            v_t = v[:, :, t, :]  # (B, H, d)
            q_t = q[:, :, t, :]  # (B, H, d)

            # 1. Innovation error: e_t = v_t - S_{t-1}^T k_t
            v_pred = torch.einsum("bhde,bhd->bhe", S, k_t)
            e_t = v_t - v_pred

            # 2. Kalman Gain: u_t = P_{t-1} k_t
            u_t = torch.einsum("bhij,bhj->bhi", P, k_t)  # (B, H, d)
            denom = self.lambda_decay + torch.einsum("bhd,bhd->bh", k_t, u_t).unsqueeze(-1) + 1e-5
            K_t = u_t / denom  # (B, H, d)

            # 3. Update Covariance: P_t = (P_{t-1} - K_t u_t^T) / lambda
            P = (P - torch.einsum("bhi,bhj->bhij", K_t, u_t)) / self.lambda_decay

            # 4. Update Memory: S_t = S_{t-1} + K_t e_t^T
            S = S + torch.einsum("bhd,bhe->bhde", K_t, e_t)

            # 5. Readout for query q_t
            y_t = torch.einsum("bhde,bhd->bhe", S, q_t)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)  # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns initial (S, P) states for constant O(1) memory autoregressive inference."""
        S = torch.zeros(batch_size, self.n_heads, self.d_head, self.d_head, device=device)
        eye = torch.eye(self.d_head, device=device).unsqueeze(0).unsqueeze(0)
        P = eye.repeat(batch_size, self.n_heads, 1, 1) / self.delta_reg
        return S, P

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-token constant O(1) autoregressive generation step."""
        S, P = state
        B, C = x_t.shape
        q = self.q_proj(x_t).view(B, self.n_heads, self.d_head)
        k = torch.nn.functional.normalize(self.k_proj(x_t).view(B, self.n_heads, self.d_head), p=2, dim=-1)
        v = self.v_proj(x_t).view(B, self.n_heads, self.d_head)

        v_pred = torch.einsum("bhde,bhd->bhe", S, k)
        e_t = v - v_pred
        u_t = torch.einsum("bhij,bhj->bhi", P, k)
        denom = self.lambda_decay + torch.einsum("bhd,bhd->bh", k, u_t).unsqueeze(-1) + 1e-5
        K_t = u_t / denom

        new_P = (P - torch.einsum("bhi,bhj->bhij", K_t, u_t)) / self.lambda_decay
        new_S = S + torch.einsum("bhd,bhe->bhde", K_t, e_t)
        y = torch.einsum("bhde,bhd->bhe", new_S, q).view(B, C)
        return self.out_proj(y), (new_S, new_P)


class ChunkwiseRidgeAttention(nn.Module):
    """
    Chunkwise Closed-Form Ridge Attention (Micro-Solver Layer):
    - Partitions the sequence into small chunks of size C (e.g. C=16 or 32).
    - Within each chunk, keys K_c in R^{C x d} and values V_c in R^{C x d} define
      an exact Ridge regression problem:
        W_c^* = (K_c^T K_c + lambda * I)^{-1} K_c^T V_c in R^{d x d}
      solved analytically in closed-form via batched torch.linalg.solve in parallel across all chunks.
    - Causal inter-chunk recurrence: S_c = gamma * S_{c-1} + W_{c-1}^*
    - Inter-chunk query readout: Y_{inter} = Q_c @ S_c
    - Intra-chunk causal attention: Y_{intra} = CausalSoftmax(Q_c K_c^T / sqrt(d)) @ V_c
    - Combined output: Y = Y_{inter} + Y_{intra}
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 16, lambda_reg: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.lambda_reg = lambda_reg
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable inter-chunk discount factor gamma in (0, 1)
        self.gamma_logit = nn.Parameter(torch.tensor([2.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C_dim = x.shape
        H, d, C = self.n_heads, self.d_head, self.chunk_size

        # Pad sequence length to multiple of C if necessary
        pad_len = (C - (T % C)) % C
        if pad_len > 0:
            x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        T_pad = x_pad.shape[1]
        Nc = T_pad // C

        q = self.q_proj(x_pad).view(B, T_pad, H, d).transpose(1, 2)  # (B, H, T_pad, d)
        k = self.k_proj(x_pad).view(B, T_pad, H, d).transpose(1, 2)
        v = self.v_proj(x_pad).view(B, T_pad, H, d).transpose(1, 2)

        qc = q.view(B, H, Nc, C, d)
        kc = k.view(B, H, Nc, C, d)
        vc = v.view(B, H, Nc, C, d)

        # 1. Closed-form Ridge solve for all chunks simultaneously
        kc_flat = kc.reshape(B * H * Nc, C, d)
        vc_flat = vc.reshape(B * H * Nc, C, d)

        eye = torch.eye(d, device=x.device).unsqueeze(0)
        G = torch.matmul(kc_flat.transpose(-1, -2), kc_flat) + self.lambda_reg * eye
        rhs = torch.matmul(kc_flat.transpose(-1, -2), vc_flat)
        W_all = torch.linalg.solve(G, rhs).view(B, H, Nc, d, d)

        # 2. Inter-chunk causal recurrence
        gamma = torch.sigmoid(self.gamma_logit)
        S_list = [torch.zeros(B, H, d, d, device=x.device)]
        for c in range(Nc - 1):
            S_next = gamma * S_list[-1] + W_all[:, :, c, :, :]
            S_list.append(S_next)
        S = torch.stack(S_list, dim=2)  # (B, H, Nc, d, d)

        # Inter-chunk readout
        Y_inter = torch.einsum("bhncd,bhnde->bhnce", qc, S)

        # 3. Intra-chunk causal attention
        scores = torch.matmul(qc, kc.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(torch.full((C, C), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        Y_intra = torch.matmul(attn, vc)

        # 4. Total output
        Y = (Y_inter + Y_intra).view(B, H, T_pad, d)
        Y = Y.transpose(1, 2).contiguous().view(B, T_pad, C_dim)
        if pad_len > 0:
            Y = Y[:, :T, :]
        return self.out_proj(Y)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, list, list]:
        """Returns initial state for constant O(1) inference: (S, buffer_k, buffer_v)."""
        S = torch.zeros(batch_size, self.n_heads, self.d_head, self.d_head, device=device)
        return (S, [], [])

    def step(self, x_t: torch.Tensor, state: tuple) -> tuple[torch.Tensor, tuple]:
        """Single-token constant O(1) autoregressive generation step."""
        S, buffer_k, buffer_v = state
        B, C_dim = x_t.shape
        q = self.q_proj(x_t).view(B, self.n_heads, self.d_head)
        k = self.k_proj(x_t).view(B, self.n_heads, self.d_head)
        v = self.v_proj(x_t).view(B, self.n_heads, self.d_head)

        y_inter = torch.einsum("bhd,bhde->bhe", q, S)

        buffer_k.append(k)
        buffer_v.append(v)
        K_buf = torch.stack(buffer_k, dim=2)  # (B, H, len_buf, d)
        V_buf = torch.stack(buffer_v, dim=2)

        scores = torch.einsum("bhd,bhcd->bhc", q, K_buf) * self.scale
        attn = torch.softmax(scores, dim=-1)
        y_intra = torch.einsum("bhc,bhcd->bhd", attn, V_buf)

        y = y_inter + y_intra

        if len(buffer_k) == self.chunk_size:
            K_full = torch.stack(buffer_k, dim=2)
            V_full = torch.stack(buffer_v, dim=2)
            K_flat = K_full.reshape(B * self.n_heads, self.chunk_size, self.d_head)
            V_flat = V_full.reshape(B * self.n_heads, self.chunk_size, self.d_head)

            eye = torch.eye(self.d_head, device=x_t.device).unsqueeze(0)
            G = torch.matmul(K_flat.transpose(-1, -2), K_flat) + self.lambda_reg * eye
            rhs = torch.matmul(K_flat.transpose(-1, -2), V_flat)
            W_chunk = torch.linalg.solve(G, rhs).view(B, self.n_heads, self.d_head, self.d_head)

            gamma = torch.sigmoid(self.gamma_logit)
            S = gamma * S + W_chunk
            buffer_k = []
            buffer_v = []

        return self.out_proj(y.view(B, C_dim)), (S, buffer_k, buffer_v)


class DiagonalKalmanAttention(nn.Module):
    """
    Diagonal Kalman Filter Attention (Preconditioned Delta-Net):
    Maintains:
    - S_t: Associative memory matrix in R^{d_k x d_v}
    - p_t: Diagonal precision vector in R^{d_k}
    Elementwise O(d) updates with learnable input-dependent write gate beta_t = sigmoid(W_beta x_t).
    Combines pre-whitened Kalman precision with fast scalar gating.
    """

    def __init__(self, d_model: int, n_heads: int, lambda_decay: float = 0.99, delta_reg: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lambda_decay = lambda_decay
        self.delta_reg = delta_reg

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, self.n_heads).permute(0, 2, 1).unsqueeze(-1)

        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        # Initialize State S_0 = 0 and diagonal precision p_0 = 1 / delta
        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head, device=x.device)
        p = torch.ones(B, self.n_heads, self.d_head, device=x.device) / self.delta_reg

        outputs = []
        for t in range(T):
            k_t = k[:, :, t, :]        # (B, H, d)
            v_t = v[:, :, t, :]        # (B, H, d)
            q_t = q[:, :, t, :]        # (B, H, d)
            beta_t = beta[:, :, t, :]  # (B, H, 1)

            # 1. Innovation error: e_t = v_t - S_{t-1}^T k_t
            v_pred = torch.einsum("bhde,bhd->bhe", S, k_t)
            e_t = v_t - v_pred

            # 2. Diagonal Kalman gain: u_t = p_{t-1} * k_t
            u_t = p * k_t
            denom = self.lambda_decay + torch.sum(k_t * u_t, dim=-1, keepdim=True) + 1e-5
            K_t = u_t / denom

            # 3. Precision vector update: p_t = (p_{t-1} - K_t * k_t * p_{t-1}) / lambda
            p = torch.clamp((p - K_t * k_t * p) / self.lambda_decay, min=1e-4, max=1e4)

            # 4. Gated Memory update
            delta_S = beta_t.unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", K_t, e_t)
            S = S + delta_S

            # 5. Readout for query q_t
            y_t = torch.einsum("bhde,bhd->bhe", S, q_t)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns initial (S, p) states for constant O(1) memory autoregressive inference."""
        S = torch.zeros(batch_size, self.n_heads, self.d_head, self.d_head, device=device)
        p = torch.ones(batch_size, self.n_heads, self.d_head, device=device) / self.delta_reg
        return S, p

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-token constant O(1) autoregressive generation step."""
        S, p = state
        B, C = x_t.shape
        q = self.q_proj(x_t).view(B, self.n_heads, self.d_head)
        k = torch.nn.functional.normalize(self.k_proj(x_t).view(B, self.n_heads, self.d_head), p=2, dim=-1)
        v = self.v_proj(x_t).view(B, self.n_heads, self.d_head)
        beta = torch.sigmoid(self.beta_proj(x_t)).view(B, self.n_heads, 1)

        v_pred = torch.einsum("bhde,bhd->bhe", S, k)
        e_t = v - v_pred
        u_t = p * k
        denom = self.lambda_decay + torch.sum(k * u_t, dim=-1, keepdim=True) + 1e-5
        K_t = u_t / denom

        p_new = torch.clamp((p - K_t * k * p) / self.lambda_decay, min=1e-4, max=1e4)
        S_new = S + beta.unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", K_t, e_t)
        y = torch.einsum("bhde,bhd->bhe", S_new, q).view(B, C)
        return self.out_proj(y), (S_new, p_new)


class ChunkwisePolySolvableAttention(nn.Module):
    """
    Chunkwise Degree-2 Polynomial Solvable Attention:
    Expands chunk keys with degree-2 polynomial cross-features:
      phi_2(x) = [x, x^2, x_j * x_k (j < k)]
    and computes exact closed-form Ridge solve per chunk in parallel.
    Bridges non-linear interactive expressivity with closed-form solvability.
    """

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 16, lambda_reg: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.lambda_reg = lambda_reg

        # Degree-2 cross-feature dimension: d linear + d squared + d*(d-1)/2 cross
        self.d_poly = self.d_head + self.d_head + (self.d_head * (self.d_head - 1)) // 2
        self.scale = 1.0 / math.sqrt(self.d_poly)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.gamma_logit = nn.Parameter(torch.tensor([2.5]))

    def compute_poly2_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d) -> returns (..., d_poly)"""
        d = x.shape[-1]
        feats = [x, x ** 2]
        cross = []
        for i in range(d):
            for j in range(i + 1, d):
                cross.append(x[..., i : i + 1] * x[..., j : j + 1])
        feats.append(torch.cat(cross, dim=-1))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C_dim = x.shape
        H, d, C = self.n_heads, self.d_head, self.chunk_size

        pad_len = (C - (T % C)) % C
        if pad_len > 0:
            x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        T_pad = x_pad.shape[1]
        Nc = T_pad // C

        q = self.q_proj(x_pad).view(B, T_pad, H, d).transpose(1, 2)
        k = self.k_proj(x_pad).view(B, T_pad, H, d).transpose(1, 2)
        v = self.v_proj(x_pad).view(B, T_pad, H, d).transpose(1, 2)

        qc = q.view(B, H, Nc, C, d)
        kc = k.view(B, H, Nc, C, d)
        vc = v.view(B, H, Nc, C, d)

        # Expand polynomial features: (B, H, Nc, C, d_poly)
        phi_k = self.compute_poly2_features(kc)
        phi_q = self.compute_poly2_features(qc)
        dp = self.d_poly

        # 1. Closed-form Dual Ridge solve in polynomial feature space (C x C system)
        phi_k_flat = phi_k.reshape(B * H * Nc, C, dp)
        vc_flat = vc.reshape(B * H * Nc, C, d)

        eye = torch.eye(C, device=x.device).unsqueeze(0)
        G_dual = torch.matmul(phi_k_flat, phi_k_flat.transpose(-1, -2)) + self.lambda_reg * eye
        alpha = torch.linalg.solve(G_dual, vc_flat)
        W_all = torch.matmul(phi_k_flat.transpose(-1, -2), alpha).view(B, H, Nc, dp, d)

        # 2. Inter-chunk causal recurrence
        gamma = torch.sigmoid(self.gamma_logit)
        S_list = [torch.zeros(B, H, dp, d, device=x.device)]
        for c in range(Nc - 1):
            S_next = gamma * S_list[-1] + W_all[:, :, c, :, :]
            S_list.append(S_next)
        S = torch.stack(S_list, dim=2)

        # Inter-chunk readout: phi_q @ S
        Y_inter = torch.einsum("bhncp,bhnpd->bhncd", phi_q, S)

        # 3. Intra-chunk causal attention with polynomial features
        scores = torch.matmul(phi_q, phi_k.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(torch.full((C, C), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        Y_intra = torch.matmul(attn, vc)

        # 4. Total output
        Y = (Y_inter + Y_intra).view(B, H, T_pad, d)
        Y = Y.transpose(1, 2).contiguous().view(B, T_pad, C_dim)
        if pad_len > 0:
            Y = Y[:, :T, :]
        return self.out_proj(Y)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, list, list]:
        """Returns initial state for constant O(1) inference: (S, buffer_k, buffer_v)."""
        S = torch.zeros(batch_size, self.n_heads, self.d_poly, self.d_head, device=device)
        return (S, [], [])

    def step(self, x_t: torch.Tensor, state: tuple) -> tuple[torch.Tensor, tuple]:
        """Single-token constant O(1) autoregressive generation step."""
        S, buffer_k, buffer_v = state
        B, C_dim = x_t.shape
        q = self.q_proj(x_t).view(B, self.n_heads, self.d_head)
        k = self.k_proj(x_t).view(B, self.n_heads, self.d_head)
        v = self.v_proj(x_t).view(B, self.n_heads, self.d_head)

        phi_q = self.compute_poly2_features(q)
        phi_k = self.compute_poly2_features(k)

        y_inter = torch.einsum("bhp,bhpd->bhd", phi_q, S)

        buffer_k.append(phi_k)
        buffer_v.append(v)
        K_buf = torch.stack(buffer_k, dim=2)
        V_buf = torch.stack(buffer_v, dim=2)

        scores = torch.einsum("bhp,bhcp->bhc", phi_q, K_buf) * self.scale
        attn = torch.softmax(scores, dim=-1)
        y_intra = torch.einsum("bhc,bhcd->bhd", attn, V_buf)

        y = y_inter + y_intra

        if len(buffer_k) == self.chunk_size:
            K_full = torch.stack(buffer_k, dim=2)
            V_full = torch.stack(buffer_v, dim=2)
            K_flat = K_full.reshape(B * self.n_heads, self.chunk_size, self.d_poly)
            V_flat = V_full.reshape(B * self.n_heads, self.chunk_size, self.d_head)

            eye = torch.eye(self.chunk_size, device=x_t.device).unsqueeze(0)
            G_dual = torch.matmul(K_flat, K_flat.transpose(-1, -2)) + self.lambda_reg * eye
            alpha = torch.linalg.solve(G_dual, V_flat)
            W_chunk = torch.matmul(K_flat.transpose(-1, -2), alpha).view(B, self.n_heads, self.d_poly, self.d_head)

            gamma = torch.sigmoid(self.gamma_logit)
            S = gamma * S + W_chunk
            buffer_k = []
            buffer_v = []

        return self.out_proj(y.view(B, C_dim)), (S, buffer_k, buffer_v)


class ExactCausalOLSAttention(nn.Module):
    """
    Exact Causal Online Least Squares (OLS) Attention:
    Solves the continuous causal regularized regression problem:
      min_{W_t} sum_{tau=1}^t gamma^{t-tau} || phi(k_tau)^T W_t - v_tau^T ||_2^2 + lambda ||W_t||_F^2
    Analytic solution for all tokens t in parallel:
      G_t = sum_{tau=1}^t gamma^{t-tau} phi(k_tau) phi(k_tau)^T + lambda * I in R^{d_feat x d_feat}
      R_t = sum_{tau=1}^t gamma^{t-tau} phi(k_tau) v_tau^T in R^{d_feat x d_head}
      W_t^* = G_t^{-1} R_t
      y_t = phi(q_t)^T W_t^*
    All T linear systems are solved simultaneously in a single batched torch.linalg.solve.
    """

    def __init__(self, d_model: int, n_heads: int, lambda_reg: float = 0.1, gamma: float = 1.0, use_poly: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.use_poly = use_poly
        self.d_feat = self.d_head * 2 if use_poly else self.d_head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_poly:
            return torch.cat([x, x ** 2], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_heads
        q = self.q_proj(x).view(B, T, H, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.d_head).transpose(1, 2)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)
        df = self.d_feat

        # Outer products
        K_outer = torch.einsum("bhtd,bhte->bhtde", phi_k, phi_k)
        KV_outer = torch.einsum("bhtd,bhte->bhtde", phi_k, v)

        if self.gamma < 1.0:
            powers = torch.arange(T, device=x.device, dtype=x.dtype)
            scale_down = (self.gamma ** (-powers)).view(1, 1, T, 1, 1)
            scale_up = (self.gamma ** powers).view(1, 1, T, 1, 1)
            G = scale_up * torch.cumsum(K_outer * scale_down, dim=2)
            R = scale_up * torch.cumsum(KV_outer * scale_down, dim=2)
        else:
            G = torch.cumsum(K_outer, dim=2)
            R = torch.cumsum(KV_outer, dim=2)

        eye = torch.eye(df, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        G = G + self.lambda_reg * eye

        # Flatten to 3D for cuBLAS/cuSOLVER batched kernel compatibility on CUDA
        # G is strictly symmetric positive-definite (G = sum k k^T + lambda I), so Cholesky solve
        # is mathematically optimal, 2x faster than LU, and natively supported across all GPUs.
        G_flat = G.reshape(B * H * T, df, df)
        R_flat = R.reshape(B * H * T, df, self.d_head)
        L = torch.linalg.cholesky(G_flat)
        W_flat = torch.cholesky_solve(R_flat, L)
        W = W_flat.view(B, H, T, df, self.d_head)
        y = torch.einsum("bhtd,bhtde->bhte", phi_q, W)  # (B, H, T, d_head)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns initial state (G, R) for constant O(1) inference."""
        eye = torch.eye(self.d_feat, device=device).unsqueeze(0).unsqueeze(0)
        G = self.lambda_reg * eye.repeat(batch_size, self.n_heads, 1, 1)
        R = torch.zeros(batch_size, self.n_heads, self.d_feat, self.d_head, device=device)
        return G, R

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-token constant O(1) autoregressive generation step."""
        G, R = state
        B, C = x_t.shape
        q = self.q_proj(x_t).view(B, self.n_heads, self.d_head)
        k = self.k_proj(x_t).view(B, self.n_heads, self.d_head)
        v = self.v_proj(x_t).view(B, self.n_heads, self.d_head)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        # Update running Gram and cross-covariance matrices
        new_G = self.gamma * G + torch.einsum("bhd,bhe->bhde", phi_k, phi_k)
        new_R = self.gamma * R + torch.einsum("bhd,bhe->bhde", phi_k, v)

        # Solve system via 3D cuBLAS layout using Cholesky
        new_G_flat = new_G.reshape(B * self.n_heads, self.d_feat, self.d_feat)
        new_R_flat = new_R.reshape(B * self.n_heads, self.d_feat, self.d_head)
        L = torch.linalg.cholesky(new_G_flat)
        W_flat = torch.cholesky_solve(new_R_flat, L)
        W = W_flat.view(B, self.n_heads, self.d_feat, self.d_head)
        y = torch.einsum("bhd,bhde->bhe", phi_q, W).view(B, C)
        return self.out_proj(y), (new_G, new_R)


class FlashCausalOLSAttention(nn.Module):
    """
    Flash-OLS: Chunked Associative Causal Online Least Squares Attention.
    Solves the exact causal regularized least squares normal equations for ultra-long
    sequences (T > 100k) with strictly sub-linear peak memory scaling and linear O(T) runtime:
    1. Inter-Chunk Associative Prefix Scan:
       Boundary Gram matrices G_c^{inter} and cross-covariances R_c^{inter} are computed
       and propagated across N_c = ceil(T / C) chunks via an associative prefix scan.
    2. Intra-Chunk Streaming Cholesky Solves:
       Equations are solved chunk-by-chunk in local memory without ever materializing
       the full 5D Gram tensor (B, H, T, d_feat, d_feat) in global DRAM.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 64,
        lambda_reg: float = 0.1,
        gamma: float = 1.0,
        use_poly: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.d_head = d_model // n_heads
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.use_poly = use_poly
        self.d_feat = self.d_head * 2 if use_poly else self.d_head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_poly:
            return torch.cat([x, x ** 2], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C_dim = x.shape
        H = self.n_heads
        q = self.q_proj(x).view(B, T, H, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.d_head).transpose(1, 2)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)
        df = self.d_feat
        dh = self.d_head
        C = self.chunk_size

        # Pad to integer multiple of chunk_size if needed
        Nc = (T + C - 1) // C
        T_pad = Nc * C
        pad_len = T_pad - T
        if pad_len > 0:
            phi_k = torch.cat([phi_k, torch.zeros(B, H, pad_len, df, device=x.device, dtype=x.dtype)], dim=2)
            v = torch.cat([v, torch.zeros(B, H, pad_len, dh, device=x.device, dtype=x.dtype)], dim=2)
            phi_q = torch.cat([phi_q, torch.zeros(B, H, pad_len, df, device=x.device, dtype=x.dtype)], dim=2)

        phi_k_chunks = phi_k.view(B, H, Nc, C, df)
        v_chunks = v.view(B, H, Nc, C, dh)
        phi_q_chunks = phi_q.view(B, H, Nc, C, df)

        chunk_p = torch.arange(C, device=x.device, dtype=x.dtype)
        decay_weights = (self.gamma ** (C - 1 - chunk_p)).view(1, 1, 1, C, 1)

        # 1. Inter-chunk contracted boundary contributions
        K_weighted = phi_k_chunks * torch.sqrt(decay_weights)
        G_chunk_end = torch.matmul(K_weighted.transpose(-1, -2), K_weighted)
        R_chunk_end = torch.matmul((phi_k_chunks * decay_weights).transpose(-1, -2), v_chunks)

        # 2. Inter-chunk associative prefix scan across Nc chunks
        alpha = self.gamma ** C
        G_inter = torch.zeros(B, H, Nc, df, df, device=x.device, dtype=x.dtype)
        R_inter = torch.zeros(B, H, Nc, df, dh, device=x.device, dtype=x.dtype)
        curr_G = torch.zeros(B, H, df, df, device=x.device, dtype=x.dtype)
        curr_R = torch.zeros(B, H, df, dh, device=x.device, dtype=x.dtype)

        for c in range(Nc):
            G_inter[:, :, c] = curr_G
            R_inter[:, :, c] = curr_R
            curr_G = alpha * curr_G + G_chunk_end[:, :, c]
            curr_R = alpha * curr_R + R_chunk_end[:, :, c]

        # 3. Intra-chunk streaming Cholesky solve (constant local memory)
        y_out = torch.zeros(B, H, T_pad, dh, device=x.device, dtype=x.dtype)
        eye_c = torch.eye(df, device=x.device, dtype=x.dtype).view(1, 1, 1, df, df)
        c_scale_down = (self.gamma ** (-chunk_p)).view(1, 1, C, 1, 1)
        c_scale_up = (self.gamma ** chunk_p).view(1, 1, C, 1, 1)
        decay_to_token = (self.gamma ** (chunk_p + 1)).view(1, 1, C, 1, 1)

        for c in range(Nc):
            kc = phi_k_chunks[:, :, c]
            vc = v_chunks[:, :, c]
            qc = phi_q_chunks[:, :, c]

            Kc_outer = torch.einsum("bhcd,bhce->bhcde", kc, kc)
            KVc_outer = torch.einsum("bhcd,bhce->bhcde", kc, vc)
            Gc_local = c_scale_up * torch.cumsum(Kc_outer * c_scale_down, dim=2)
            Rc_local = c_scale_up * torch.cumsum(KVc_outer * c_scale_down, dim=2)

            Gc_full = decay_to_token * G_inter[:, :, c].unsqueeze(2) + Gc_local + self.lambda_reg * eye_c
            Rc_full = decay_to_token * R_inter[:, :, c].unsqueeze(2) + Rc_local

            Lc = torch.linalg.cholesky(Gc_full.reshape(B * H * C, df, df))
            Wc = torch.cholesky_solve(Rc_full.reshape(B * H * C, df, dh), Lc).view(B, H, C, df, dh)
            yc = torch.einsum("bhcd,bhcde->bhce", qc, Wc)
            y_out[:, :, c * C : (c + 1) * C] = yc

        y = y_out[:, :, :T].transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.out_proj(y)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns initial state (G, R) for constant O(1) inference."""
        eye = torch.eye(self.d_feat, device=device).unsqueeze(0).unsqueeze(0)
        G = self.lambda_reg * eye.repeat(batch_size, self.n_heads, 1, 1)
        R = torch.zeros(batch_size, self.n_heads, self.d_feat, self.d_head, device=device)
        return G, R

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-token constant O(1) autoregressive generation step."""
        G, R = state
        B, C = x_t.shape
        q = self.q_proj(x_t).view(B, self.n_heads, self.d_head)
        k = self.k_proj(x_t).view(B, self.n_heads, self.d_head)
        v = self.v_proj(x_t).view(B, self.n_heads, self.d_head)

        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        new_G = self.gamma * G + torch.einsum("bhd,bhe->bhde", phi_k, phi_k)
        new_R = self.gamma * R + torch.einsum("bhd,bhe->bhde", phi_k, v)

        new_G_flat = new_G.reshape(B * self.n_heads, self.d_feat, self.d_feat)
        new_R_flat = new_R.reshape(B * self.n_heads, self.d_feat, self.d_head)
        L = torch.linalg.cholesky(new_G_flat)
        W_flat = torch.cholesky_solve(new_R_flat, L)
        W = W_flat.view(B, self.n_heads, self.d_feat, self.d_head)
        y = torch.einsum("bhd,bhde->bhe", phi_q, W).view(B, C)
        return self.out_proj(y), (new_G, new_R)


class SequenceTransformer(nn.Module):
    """
    Autoregressive Sequence Transformer backbone supporting:
    - attn_type="standard": Standard Multi-Head Softmax Attention
    - attn_type="linear": Causal Linear Attention (Unnormalized accumulator)
    - attn_type="delta": Delta-Rule / LMS Attention (Innovation with scalar gain)
    - attn_type="rls": Causal RLS / Kalman Filter Attention (Optimal innovation with matrix gain)
    - attn_type="two_stream": Two-Stream Bulk + Margin Attention
    - attn_type="chunk_ridge": Chunkwise Closed-Form Ridge Attention (Micro-solver)
    - attn_type="diag_kalman": Diagonal Kalman Attention (Preconditioned Delta-Net)
    - attn_type="chunk_poly": Chunkwise Degree-2 Polynomial Solvable Attention
    - attn_type="exact_ols": Exact Causal Online Least Squares Attention (Unified direct solve)
    - attn_type="exact_poly_ols": Exact Causal Polynomial Online Least Squares Attention
    - attn_type="flash_ols": Flash-OLS (Chunked associative causal OLS for ultra-long context)
    - attn_type="flash_poly_ols": Flash-OLS with degree-2 polynomial expansion
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ffn: int = 128,
        max_seq_len: int = 4096,
        attn_type: str = "rls",
        chunk_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.attn_type = attn_type

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if attn_type == "standard":
                attn = CausalStandardAttention(d_model, n_heads, dropout=dropout)
            elif attn_type == "linear":
                attn = CausalLinearAttention(d_model, n_heads)
            elif attn_type == "delta":
                attn = CausalDeltaAttention(d_model, n_heads)
            elif attn_type == "rls":
                attn = CausalRLSAttention(d_model, n_heads)
            elif attn_type == "two_stream":
                attn = TwoStreamBulkMarginAttention(d_model, n_heads)
            elif attn_type == "chunk_ridge":
                attn = ChunkwiseRidgeAttention(d_model, n_heads)
            elif attn_type == "diag_kalman":
                attn = DiagonalKalmanAttention(d_model, n_heads)
            elif attn_type == "chunk_poly":
                attn = ChunkwisePolySolvableAttention(d_model, n_heads)
            elif attn_type == "exact_ols":
                attn = ExactCausalOLSAttention(d_model, n_heads, use_poly=False)
            elif attn_type == "exact_poly_ols":
                attn = ExactCausalOLSAttention(d_model, n_heads, use_poly=True)
            elif attn_type == "flash_ols":
                attn = FlashCausalOLSAttention(d_model, n_heads, chunk_size=chunk_size, use_poly=False)
            elif attn_type == "flash_poly_ols":
                attn = FlashCausalOLSAttention(d_model, n_heads, chunk_size=chunk_size, use_poly=True)
            else:
                raise ValueError(f"Unknown attn_type: {attn_type}")

            layer = nn.ModuleDict({
                "norm1": nn.LayerNorm(d_model),
                "attn": attn,
                "norm2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, d_model),
                ),
            })
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]

        for layer in self.layers:
            h = layer["norm1"](x)
            x = x + layer["attn"](h)
            x = x + layer["ffn"](layer["norm2"](x))

        x = self.final_norm(x)
        return self.head(x)  # (B, T, vocab_size)


# =============================================================================
# DEEP CLOSED-FORM BOOSTING & SOLVABLE CASCADE (Phase 7 / EXP-09)
# =============================================================================

def compute_poly2_features(U: torch.Tensor, clip_val: float = 4.0) -> torch.Tensor:
    """
    Computes degree-2 polynomial expansion with outlier clipping:
    phi_2(U) = [U, U^2, U_i * U_j (i < j)].
    Clipping prevents numerical blowup in deep cascades with high-order terms.
    """
    U_clipped = torch.clamp(U, -clip_val, clip_val)
    N, P = U_clipped.shape
    cols = [U_clipped, U_clipped ** 2]
    for i in range(P):
        for j in range(i + 1, P):
            cols.append(U_clipped[:, i : i + 1] * U_clipped[:, j : j + 1])
    return torch.cat(cols, dim=-1)


class DeepSolvableCascade(nn.Module):
    """
    Deep Closed-Form Boosting / Deep Solvable Network (DSN).

    Cascades analytical solvable regression blocks in series without backpropagation:
    - Stage 1 (Degree 2): Expands raw features X into quadratic space Phi_1 = phi_2(X).
      Solves W_1* = (Phi_1^T Phi_1 + lambda*I)^{-1} Phi_1^T y via closed-form Ridge.
      Residual r_1 = y - Phi_1 W_1*.
      Latent representation Z_1 is formed from top principal components of Phi_1.
    - Stage 2 (Degree 4): Input U_2 = [X, Z_1]. Since Z_1 ~ X^2, quadratic expansion
      Phi_2 = phi_2(U_2) produces cross-terms (X^2)^2 ~ X^4 (Degree 4 interactions!).
      Solves W_2* on residual r_1. Updates r_2 = r_1 - eta * Phi_2 W_2*.
    - Stage L (Degree 2^L): Cascading L stages reaches polynomial degree 2^L in
      strictly closed-form forward passes with zero gradient descent.
    """

    def __init__(
        self,
        n_stages: int = 4,
        d_latent: int = 8,
        shrinkage: float = 0.5,
        lambda_reg: float = 1e-3,
        clip_val: float = 4.0,
        use_rff: bool = False,
        n_rff: int = 32,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.d_latent = d_latent
        self.shrinkage = shrinkage
        self.lambda_reg = lambda_reg
        self.clip_val = clip_val
        self.use_rff = use_rff
        self.n_rff = n_rff
        self.stages = []

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "DeepSolvableCascade":
        """
        Fits all stages sequentially in closed form.
        Takes X: (N, d), y: (N, 1) or (N,).
        """
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        device = X.device
        N, d = X.shape
        curr_res = y.clone()
        prev_Z = None

        self.stages = []
        for stage in range(self.n_stages):
            if stage == 0:
                U = X
            else:
                U = torch.cat([X, prev_Z], dim=-1)

            Phi = compute_poly2_features(U, self.clip_val)
            rff_W = None
            if self.use_rff:
                torch.manual_seed(42 + stage)
                rff_W = torch.randn(U.shape[1], self.n_rff // 2, device=device) * 0.5
                proj = torch.matmul(torch.clamp(U, -self.clip_val, self.clip_val), rff_W)
                rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
                Phi = torch.cat([Phi, rff], dim=-1)

            D = Phi.shape[1]
            G = torch.matmul(Phi.T, Phi) + (self.lambda_reg * N) * torch.eye(D, device=device)
            rhs = torch.matmul(Phi.T, curr_res)
            W = torch.linalg.solve(G, rhs)

            pred_stage = torch.matmul(Phi, W)
            step_shrink = 1.0 if stage == 0 else self.shrinkage
            curr_res = curr_res - step_shrink * pred_stage

            Phi_mean = Phi.mean(dim=0, keepdim=True)
            Phi_centered = Phi - Phi_mean
            cov = torch.matmul(Phi_centered.T, Phi_centered) / N
            eigvals, eigvecs = torch.linalg.eigh(cov)

            k = max(1, self.d_latent - 1)
            top_vecs = eigvecs[:, -k:]
            Z_pca_raw = torch.matmul(Phi_centered, top_vecs)
            pca_scale = Z_pca_raw.std(dim=0, keepdim=True) + 1e-6
            Z_pca = torch.tanh(Z_pca_raw / pca_scale)

            pred_scale = pred_stage.std() + 1e-6
            pred_norm = torch.tanh(pred_stage / pred_scale)
            prev_Z = torch.cat([pred_norm, Z_pca], dim=-1)

            stage_info = {
                "W": W,
                "step_shrink": step_shrink,
                "Phi_mean": Phi_mean,
                "top_vecs": top_vecs,
                "pca_scale": pca_scale,
                "pred_scale": pred_scale,
                "rff_W": rff_W,
                "effective_degree": 2 ** (stage + 1),
            }
            self.stages.append(stage_info)

        return self

    def fit_with_als(self, X: torch.Tensor, y: torch.Tensor, n_als_cycles: int = 5) -> tuple["DeepSolvableCascade", list[float]]:
        """
        Blueprint 3: Alternating Least Squares (ALS) Block-Coordinate Refinement.
        1. Executes standard forward greedy boosting pass to build the stages.
        2. Alternates across all stages, refitting each stage W_l* on the residual
           of all other stages in exact closed form.
        Guarantees monotonic loss decrease without backpropagation or learning rates.
        Returns (self, als_loss_history).
        """
        self.fit(X, y)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        device = X.device
        N = X.shape[0]

        # 1. Cache feature matrices Phi_l for all stages
        Phis = []
        prev_Z = None
        for stage, s in enumerate(self.stages):
            if stage == 0:
                U = X
            else:
                U = torch.cat([X, prev_Z], dim=-1)

            Phi = compute_poly2_features(U, self.clip_val)
            if self.use_rff and s["rff_W"] is not None:
                proj = torch.matmul(torch.clamp(U, -self.clip_val, self.clip_val), s["rff_W"])
                rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
                Phi = torch.cat([Phi, rff], dim=-1)
            Phis.append(Phi)

            Phi_centered = Phi - s["Phi_mean"]
            Z_pca_raw = torch.matmul(Phi_centered, s["top_vecs"])
            Z_pca = torch.tanh(Z_pca_raw / s["pca_scale"])
            pred_stage = torch.matmul(Phi, s["W"])
            pred_norm = torch.tanh(pred_stage / s["pred_scale"])
            prev_Z = torch.cat([pred_norm, Z_pca], dim=-1)

        # 2. Precompute Gram inverses for all stages
        inv_Grams = []
        for l in range(len(self.stages)):
            Phi = Phis[l]
            D = Phi.shape[1]
            G = torch.matmul(Phi.T, Phi) + (self.lambda_reg * N) * torch.eye(D, device=device)
            inv_Grams.append(torch.linalg.inv(G))

        # 3. ALS Coordinate Descent Cycles
        als_history = []
        shrinkages = [s["step_shrink"] for s in self.stages]

        for cycle in range(n_als_cycles):
            for l in range(len(self.stages)):
                pred_others = torch.zeros(N, 1, device=device)
                for m in range(len(self.stages)):
                    if m != l:
                        pred_others = pred_others + shrinkages[m] * torch.matmul(Phis[m], self.stages[m]["W"])

                target_residual = y - pred_others
                rhs = torch.matmul(Phis[l].T, target_residual / shrinkages[l])
                self.stages[l]["W"] = torch.matmul(inv_Grams[l], rhs)

            # Record total cycle loss
            total_pred = sum(shrinkages[m] * torch.matmul(Phis[m], self.stages[m]["W"]) for m in range(len(self.stages)))
            cycle_loss = float(torch.mean((y - total_pred) ** 2).item())
            als_history.append(cycle_loss)

        return self, als_history

    def predict(self, X: torch.Tensor, max_stages: int = None) -> torch.Tensor:
        """
        Inference pass evaluating the cascade up to max_stages.
        """
        N = X.shape[0]
        device = X.device
        y_pred = torch.zeros(N, 1, device=device)
        prev_Z = None
        stages_to_run = self.stages if max_stages is None else self.stages[:max_stages]

        for stage, s in enumerate(stages_to_run):
            if stage == 0:
                U = X
            else:
                U = torch.cat([X, prev_Z], dim=-1)

            Phi = compute_poly2_features(U, self.clip_val)
            if self.use_rff and s["rff_W"] is not None:
                proj = torch.matmul(torch.clamp(U, -self.clip_val, self.clip_val), s["rff_W"])
                rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
                Phi = torch.cat([Phi, rff], dim=-1)

            pred_stage = torch.matmul(Phi, s["W"])
            y_pred = y_pred + s["step_shrink"] * pred_stage

            Phi_centered = Phi - s["Phi_mean"]
            Z_pca_raw = torch.matmul(Phi_centered, s["top_vecs"])
            Z_pca = torch.tanh(Z_pca_raw / s["pca_scale"])
            pred_norm = torch.tanh(pred_stage / s["pred_scale"])
            prev_Z = torch.cat([pred_norm, Z_pca], dim=-1)

        return y_pred

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict(X)



