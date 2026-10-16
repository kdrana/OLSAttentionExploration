"""
Synthetic Data-Generating Processes (DGPs) to test linear vs. polynomial vs. full-attention regression.
"""

import math
import torch


def generate_synthetic_data(
    dgp: str = "linear", n_samples: int = 1000, d: int = 8, noise_std: float = 0.1, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic regression data (X, y).
    X is normalized to lie on S^{d-1} (unit hypersphere), mirroring LayerNorm.
    """
    torch.manual_seed(seed)
    
    # Generate X and normalize to S^{d-1}
    X_raw = torch.randn(n_samples, d)
    X = torch.nn.functional.normalize(X_raw, p=2, dim=-1)

    # Fix DGP ground truth parameters
    g_dgp = torch.Generator().manual_seed(999)

    if dgp == "linear":
        w = torch.randn(d, 1, generator=g_dgp)
        w = w / torch.norm(w)
        y = torch.matmul(X, w)

    elif dgp == "interaction":
        w = torch.randn(d, 1, generator=g_dgp) * 0.5
        y = torch.matmul(X, w)
        for i in range(min(4, d)):
            for j in range(i + 1, min(4, d)):
                y += (X[:, i : i + 1] * X[:, j : j + 1]) * 1.5

    elif dgp == "friedman":
        # Nonlinear benchmark (Friedman-1 style on sphere)
        # Uses sin, square, and linear terms
        x0 = X[:, 0:1]
        x1 = X[:, 1:2] if d > 1 else X[:, 0:1]
        x2 = X[:, 2:3] if d > 2 else X[:, 0:1]
        x3 = X[:, 3:4] if d > 3 else X[:, 0:1]

        y = 2.0 * torch.sin(math.pi * x0 * x1) + 3.0 * (x2**2) + 1.5 * x3
    else:
        raise ValueError(f"Unknown DGP: {dgp}")

    # Add Gaussian noise
    y += torch.randn_like(y) * noise_std
    return X, y


def generate_associative_recall_data(
    batch_size: int = 32,
    seq_len: int = 512,
    vocab_size: int = 128,
    num_pairs: int = 8,
    num_queries: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic Multi-Query Associative Recall (MQAR) sequences:
    - Vocab contains: 0 (distractor token), Keys in [1, V//2 - 1], Values in [V//2, V - 1].
    - Injects `num_pairs` key-value pairs (k_i, v_i) at random locations in the prefix.
    - Suffix presents `num_queries` test keys k_q; target is to predict the associated value v_q.
    - Returns: tokens (B, T), targets (B, T) with -100 on non-evaluated tokens.
    """
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    key_pool = list(range(1, vocab_size // 2))
    val_pool = list(range(vocab_size // 2, vocab_size))

    prefix_max = seq_len - num_queries - 4

    for b in range(batch_size):
        # Sample distinct keys and values
        k_indices = torch.randperm(len(key_pool), generator=gen)[:num_pairs]
        v_indices = torch.randperm(len(val_pool), generator=gen)[:num_pairs]

        sampled_keys = [key_pool[i] for i in k_indices]
        sampled_vals = [val_pool[i] for i in v_indices]
        kv_map = dict(zip(sampled_keys, sampled_vals))

        # Sample non-overlapping prefix positions for (key, value) pairs
        avail_positions = torch.randperm(prefix_max // 2, generator=gen)[:num_pairs] * 2 + 1
        for i, pos in enumerate(avail_positions):
            tokens[b, pos] = sampled_keys[i]
            tokens[b, pos + 1] = sampled_vals[i]

        # Select queries from sampled keys
        q_idx = torch.randperm(num_pairs, generator=gen)[:num_queries]
        for q_i, q_k_idx in enumerate(q_idx):
            q_pos = seq_len - num_queries + q_i
            query_key = sampled_keys[q_k_idx]
            tokens[b, q_pos] = query_key
            targets[b, q_pos] = kv_map[query_key]

    return tokens, targets


def generate_multihop_associative_recall_data(
    batch_size: int = 32,
    seq_len: int = 128,
    vocab_size: int = 128,
    num_chains: int = 4,
    num_queries: int = 2,
    num_hops: int = 2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic Multi-Hop Associative Recall sequences:
    - Chains of length `num_hops + 1`: A_i -> B_i -> C_i ...
    - Pairs (A_i, B_i) and (B_i, C_i) are injected at disjoint random positions in the prefix.
    - Suffix presents start entity A_q; target is to predict the final entity (e.g. C_q for 2 hops).
    - Requires at least `num_hops` attention layers in series to solve compositional reasoning.
    """
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    pool_size = (vocab_size - 1) // (num_hops + 1)
    pools = []
    for h in range(num_hops + 1):
        start = 1 + h * pool_size
        end = start + pool_size
        pools.append(list(range(start, end)))

    prefix_max = seq_len - num_queries - 4

    for b in range(batch_size):
        # Sample distinct entities for num_chains
        chains = []
        for h in range(num_hops + 1):
            perm = torch.randperm(len(pools[h]), generator=gen)[:num_chains]
            chains.append([pools[h][i] for i in perm])

        pairs = []
        for i in range(num_chains):
            for h in range(num_hops):
                pairs.append((chains[h][i], chains[h + 1][i]))

        n_pairs = len(pairs)
        avail_positions = (torch.randperm(prefix_max // 2, generator=gen)[:n_pairs] * 2 + 1).tolist()
        for idx, (k, v) in enumerate(pairs):
            pos = avail_positions[idx]
            tokens[b, pos] = k
            tokens[b, pos + 1] = v

        # Sample queries from chain starts (chains[0] -> target is chains[-1])
        q_indices = torch.randperm(num_chains, generator=gen)[:num_queries]
        for q_i, c_idx in enumerate(q_indices):
            q_pos = seq_len - num_queries + q_i
            start_entity = chains[0][c_idx]
            final_entity = chains[-1][c_idx]
            tokens[b, q_pos] = start_entity
            targets[b, q_pos] = final_entity

    return tokens, targets


