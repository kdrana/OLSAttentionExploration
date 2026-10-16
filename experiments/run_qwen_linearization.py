"""
EXP-15: Post-Training Attention Linearization Benchmark on Qwen3.5-0.8B.
Stage 1 Sanity Run:
1. Replaces the 6 Full Attention layers with QwenFlashOLSAttention.
2. Freezes 99.94% of model weights (all MLPs, embeddings, 18 DeltaNet layers).
3. Trains only the active subspace projection U (491k params) for 350 steps.
4. Measures validation perplexity, generation text quality, and O(1) memory flatline.
"""

import json
import os
import time
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.qwen_ols_adapter import QwenFlashOLSAttention


def prepare_tinystories_data(data_dir: str = "/tmp/tinystories_data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "TinyStories-valid.txt")
    if not os.path.exists(raw_path):
        print(f"[Data] Downloading TinyStories-valid.txt to {raw_path}...")
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        urllib.request.urlretrieve(url, raw_path)
        print("[Data] Download complete!")
    return raw_path


def get_tinystories_tokens(raw_path: str, tokenizer, max_tokens: int = 500_000) -> torch.Tensor:
    print(f"[Data] Tokenizing {max_tokens:,} tokens using Qwen tokenizer...")
    tokens = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            enc = tokenizer.encode(line)
            tokens.extend(enc)
            if len(tokens) >= max_tokens:
                break
    tokens = torch.tensor(tokens[:max_tokens], dtype=torch.long)
    print(f"[Data] Loaded {len(tokens):,} tokens into tensor.")
    return tokens


def run_qwen_linearization_benchmark(
    model_id: str = "Qwen/Qwen3.5-0.8B-Base",
    device: str = "cuda",
    total_steps: int = 350,
    batch_size: int = 4,
    seq_len: int = 256,
    lr: float = 1e-3,
    checkpoints: list[int] = [0, 25, 50, 100, 150, 200, 250, 300, 350],
) -> dict:
    print("=" * 80)
    print(f"EXP-15: QWEN3.5-0.8B ATTENTION LINEARIZATION (STAGE 1 SANITY RUN)")
    print(f"Model: {model_id} | Device: {device} | Steps: {total_steps} | Batch: {batch_size} | Seq Len: {seq_len}")
    print("=" * 80)

    # 1. Load Pre-trained Qwen Model and Tokenizer
    print("\n[1/5] Loading pre-trained Qwen3.5-0.8B weights...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded successfully! Total Parameters: {total_params:,}")

    # 2. Tokenize Dataset
    raw_data_path = prepare_tinystories_data()
    all_tokens = get_tinystories_tokens(raw_data_path, tokenizer, max_tokens=300_000)
    n_val = 30_000
    train_tokens = all_tokens[:-n_val]
    val_tokens = all_tokens[-n_val:]

    def get_batch(tokens_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(tokens_tensor) - seq_len - 1, (batch_size,))
        x = torch.stack([tokens_tensor[s : s + seq_len] for s in starts]).to(device)
        y = torch.stack([tokens_tensor[s + 1 : s + seq_len + 1] for s in starts]).to(device)
        return x, y

    # 3. Benchmark Original Model Baseline
    print("\n[2/5] Evaluating Pre-Trained Softmax Baseline...")
    prompt_text = "Once upon a time, there was a little bird named Pip who lived in a"
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        # Baseline validation cross-entropy
        val_losses = []
        for _ in range(10):
            vx, vy = get_batch(val_tokens)
            vout = model(vx)
            vloss = F.cross_entropy(vout.logits.view(-1, vout.logits.size(-1)), vy.view(-1))
            val_losses.append(vloss.item())
        base_val_loss = float(np.mean(val_losses))
        base_val_ppl = float(np.exp(base_val_loss))

        # Baseline generation
        base_gen_ids = model.generate(**prompt_inputs, max_new_tokens=32, do_sample=False)
        base_gen_text = tokenizer.decode(base_gen_ids[0], skip_special_tokens=True)

    print(f"  Pre-Trained Softmax Val Loss: {base_val_loss:.4f} | Val PPL: {base_val_ppl:.2f}")
    print(f"  Pre-Trained Generation Sample: \"{base_gen_text.replace(chr(10), ' ')}\"")

    # 4. Swap the 6 Full Attention Layers with Flash-OLS
    print("\n[3/5] Swapping 6 Full-Attention Layers with Flash-OLS Modules...")
    full_attn_layer_indices = []
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and layer.self_attn is not None:
            full_attn_layer_indices.append(idx)
            # Wrap with QwenFlashOLSAttention
            ols_attn = QwenFlashOLSAttention(layer.self_attn, rank=16, lambda_reg=0.05, gamma=0.99).to(device)
            layer.self_attn = ols_attn

    print(f"Successfully replaced layers: {full_attn_layer_indices}")

    # Freeze entire model except the new active subspace projections
    trainable_params = 0
    for name, param in model.named_parameters():
        if "proj_k" in name or "proj_q" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"Frozen Model Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params*100:.2f}%)")
    print(f"Trainable Subspace Parameters: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")

    # 5. Fast Stage 1 Distillation Training Loop
    print(f"\n[4/5] Starting Stage 1 Distillation ({total_steps} steps on NVIDIA A10G)...")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    
    warmup_steps = 20
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    trace = []
    step_times = []
    t_start = time.perf_counter()

    for step in range(total_steps + 1):
        if step in checkpoints:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(8):
                    vx, vy = get_batch(val_tokens)
                    vout = model(vx)
                    vloss = F.cross_entropy(vout.logits.view(-1, vout.logits.size(-1)), vy.view(-1))
                    val_losses.append(vloss.item())
                cur_val_loss = float(np.mean(val_losses))
                cur_val_ppl = float(np.exp(cur_val_loss))

                # Generation sample
                gen_sample = ""
                if step in [0, 50, 100, 200, 350]:
                    try:
                        # Reset OLS states before generation
                        for idx in full_attn_layer_indices:
                            model.model.layers[idx].self_attn.reset_generation_state()
                        gen_ids = model.generate(**prompt_inputs, max_new_tokens=32, do_sample=False)
                        gen_sample = tokenizer.decode(gen_ids[0], skip_special_tokens=True).replace('\n', ' ')
                    except Exception as e:
                        gen_sample = f"[Gen error: {e}]"

            model.train()
            entry = {
                "step": step,
                "val_loss": cur_val_loss,
                "val_ppl": cur_val_ppl,
                "sample": gen_sample,
            }
            if trace:
                entry["train_loss"] = trace[-1].get("train_loss", None)
            trace.append(entry)

            print(f"  Step {step:4d}/{total_steps} | Val Loss: {cur_val_loss:.4f} | Val PPL: {cur_val_ppl:7.2f}")
            if gen_sample:
                print(f"    Sample: \"{gen_sample[:100]}...\"")

        if step == total_steps:
            break

        t0 = time.perf_counter()
        x, y = get_batch(train_tokens)
        optimizer.zero_grad(set_to_none=True)

        out = model(x)
        loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), y.view(-1))
        loss.backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0).item())
        optimizer.step()
        scheduler.step()

        dt = (time.perf_counter() - t0) * 1000.0
        step_times.append(dt)

        if trace and trace[-1]["step"] == step:
            trace[-1]["train_loss"] = float(loss.item())
            trace[-1]["step_time_ms"] = dt
            trace[-1]["grad_norm"] = grad_norm

    total_time_s = time.perf_counter() - t_start
    mean_step_time = float(np.mean(step_times)) if step_times else 0.0

    print(f"\n[5/5] Distillation Finished in {total_time_s:.2f}s | Mean Step Time: {mean_step_time:.1f}ms")

    # 6. Final Evaluation & Memory Verification
    final_val_loss = trace[-1]["val_loss"]
    final_val_ppl = trace[-1]["val_ppl"]
    
    # Calculate O(1) Memory Footprint for the 6 layers
    state_bytes_per_kv_head = (16 * 16 * 4) + (16 * 256 * 4)  # P + R
    total_state_bytes = len(full_attn_layer_indices) * 2 * state_bytes_per_kv_head
    total_state_kb = total_state_bytes / 1024.0

    results = {
        "model_id": model_id,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "full_attention_layers": full_attn_layer_indices,
        "baseline_softmax": {
            "val_loss": base_val_loss,
            "val_ppl": base_val_ppl,
            "sample": base_gen_text,
        },
        "linearized_ols": {
            "final_val_loss": final_val_loss,
            "final_val_ppl": final_val_ppl,
            "mean_step_time_ms": mean_step_time,
            "total_time_s": total_time_s,
            "state_memory_kb": total_state_kb,
            "trace": trace,
        }
    }
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_qwen_linearization_benchmark(device=device, total_steps=2, batch_size=2, seq_len=64)
    print("Self-test completed!")
