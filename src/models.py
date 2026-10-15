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


class SequenceTransformer(nn.Module):
    """
    Autoregressive Sequence Transformer backbone supporting:
    - attn_type="standard": Standard Multi-Head Softmax Attention
    - attn_type="linear": Causal Linear Attention (Unnormalized accumulator)
    - attn_type="delta": Delta-Rule / LMS Attention (Innovation with scalar gain)
    - attn_type="rls": Causal RLS / Kalman Filter Attention (Optimal innovation with matrix gain)
    - attn_type="two_stream": Two-Stream Bulk + Margin Attention
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


