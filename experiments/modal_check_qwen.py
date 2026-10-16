"""
Modal inspection script to verify Qwen model loading and inspect layer structures.
Tests Qwen/Qwen3.5-0.8B-Base and Qwen/Qwen2.5-0.5B.
Enforces timeout <= 120s.
"""

import modal

app = modal.App("ols-attention-qwen-check")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "triton")
)


@app.function(image=image, gpu="A10G", timeout=120)
def check_qwen_model(model_id: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\n[Modal] Testing Model: {model_id} on {torch.cuda.get_device_name(0)}")
    
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"Config loaded successfully! Model type: {getattr(config, 'model_type', 'unknown')}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"Tokenizer loaded! Vocab size: {len(tokenizer)}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded into VRAM successfully!")
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {n_params:,}")

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            import inspect
            attn3 = model.model.layers[3].self_attn
            print("=== Qwen3_5Attention.forward Source ===")
            print(inspect.getsource(attn3.forward))
        
        # Test a 1-token forward pass
        sample_input = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**sample_input)
            logits_shape = list(out.logits.shape)
        print(f"Forward pass successful! Logits shape: {logits_shape}")

        return {
            "status": "success",
            "model_id": model_id,
            "n_params": n_params,
            "logits_shape": logits_shape,
        }
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        print(f"Failed to load {model_id}:\n{err_str}")
        return {
            "status": "error",
            "model_id": model_id,
            "error": str(e),
            "traceback": err_str,
        }


@app.local_entrypoint()
def main():
    print("Checking Qwen models on Modal GPU...")
    # First try Qwen3.5-0.8B-Base
    res = check_qwen_model.remote("Qwen/Qwen3.5-0.8B-Base")
    print("\nResult for Qwen3.5-0.8B-Base:", res["status"])
    if res["status"] != "success":
        print("Falling back to Qwen2.5-0.5B check...")
        res2 = check_qwen_model.remote("Qwen/Qwen2.5-0.5B")
        print("\nResult for Qwen2.5-0.5B:", res2)
    else:
        print("Success details:", res)
