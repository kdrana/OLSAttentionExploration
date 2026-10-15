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
