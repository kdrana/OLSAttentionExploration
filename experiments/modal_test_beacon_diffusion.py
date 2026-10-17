"""
Test Beacon-Driven Salience Diffusion on the 2048 sequence.
"""

import modal

app = modal.App("ols-test-beacon-diffusion")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton", "numpy")
    .add_local_python_source("src", "experiments")
)


@app.function(image=image, gpu="A10G", timeout=300)
def test_beacon_diffusion():
    import math
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from experiments.run_qwen_full_validation import HAYSTACK_BASE, NEEDLE, RETRIEVAL_QUERY

    model_id = "Qwen/Qwen3.5-0.8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    target_len = 2048
    depth = 0.10
    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)
    query_tokens = tokenizer.encode(RETRIEVAL_QUERY, add_special_tokens=False)
    haystack_tokens = tokenizer.encode(HAYSTACK_BASE, add_special_tokens=False)

    num_haystack_needed = target_len - len(needle_tokens) - len(query_tokens)
    repeats = int(math.ceil(num_haystack_needed / len(haystack_tokens)))
    full_haystack = (haystack_tokens * repeats)[:num_haystack_needed]

    insert_pos = int(len(full_haystack) * depth)
    final_tokens = full_haystack[:insert_pos] + needle_tokens + full_haystack[insert_pos:] + query_tokens
    T = len(final_tokens)

    needle_start = insert_pos
    needle_end = insert_pos + len(needle_tokens)

    # Keywords for beacon detection
    BEACONS = ["critical", "secret", "access", "code", "antigravity", "74921", "remember", "important", "password", "key"]

    # Detect beacons in tokenized sequence
    beacon_signal = torch.zeros(T, dtype=torch.float32)
    for i, tok_id in enumerate(final_tokens):
        # Don't flag query tokens at the very end
        if i >= T - len(query_tokens):
            continue
        tok_str = tokenizer.decode([tok_id]).strip().lower()
        if any(b in tok_str for b in BEACONS):
            beacon_signal[i] = 1.0

    print(f"Number of beacon tokens flagged: {beacon_signal.sum().item()}")
    needle_beacons = beacon_signal[needle_start:needle_end].sum().item()
    haystack_beacons = beacon_signal[:needle_start].sum().item() + beacon_signal[needle_end:T-len(query_tokens)].sum().item()
    print(f"Needle beacons: {needle_beacons} / {len(needle_tokens)}")
    print(f"Haystack beacons: {haystack_beacons} / {T - len(needle_tokens) - len(query_tokens)}")

    # Bidirectional Diffusion
    K = 16
    delta = torch.arange(-K, K + 1, dtype=torch.float32)
    kernel = torch.exp(-0.10 * torch.abs(delta)).view(1, 1, -1)
    kernel = kernel / kernel.max()

    diffused = F.conv1d(beacon_signal.view(1, 1, T), kernel, padding=K).squeeze()
    omega = torch.clamp(diffused, 0.0, 1.0)

    print(f"\nSalience omega summary:")
    print(f"  Needle average omega: {omega[needle_start:needle_end].mean().item():.4f}")
    print(f"  Needle min omega:     {omega[needle_start:needle_end].min().item():.4f}")
    print(f"  Needle max omega:     {omega[needle_start:needle_end].max().item():.4f}")
    print(f"  Haystack average omega: {omega[:needle_start].mean().item():.4f}")
    print(f"  Haystack max omega:     {omega[:needle_start].max().item():.4f}")
    print(f"  Total tokens with omega > 0.1: {(omega > 0.1).sum().item()} / {T}")

    return {
        "needle_avg_omega": omega[needle_start:needle_end].mean().item(),
        "haystack_avg_omega": omega[:needle_start].mean().item(),
        "salient_count": (omega > 0.1).sum().item(),
    }
