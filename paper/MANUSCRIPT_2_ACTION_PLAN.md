# Action Plan v3: Narrative Overhaul & KV-Cache Alternatives Benchmark (`EXP-27`)

**Target Model for Execution:** Gemini Flash 3.8  
**Estimated Cloud GPU Cost:** $< \$0.15$ (Single Modal NVIDIA A10G run, `timeout=300`, ~180–220s)  
**Goal:** Transform [`paper/manuscript_2_foundation_model_adaptation.md`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/manuscript_2_foundation_model_adaptation.md) from a dense, chronological lab notebook dump into a clear, readable, model-agnostic scientific paper titled ***"Preserving Exact Long-Range Recall in Compressed Transformers"***, backed by a unified **Alternatives to the KV-Cache** experimental suite (`EXP-27`).

---

## 0. Mandatory Repository Rules (Read Before Executing)
1. **Reusable Code in `src/`:** All model/attention wrappers belong in [`src/cognitive_dual_control_attention.py`](file:///home/kdrana/Projects/OLSAttentionExploration/src/cognitive_dual_control_attention.py).
2. **Executable Benchmarks in `experiments/`:** Create [`experiments/run_kv_alternatives_suite.py`](file:///home/kdrana/Projects/OLSAttentionExploration/experiments/run_kv_alternatives_suite.py) and [`experiments/modal_kv_alternatives_runner.py`](file:///home/kdrana/Projects/OLSAttentionExploration/experiments/modal_kv_alternatives_runner.py).
3. **Structured Artifacts in `results/exp27_kv_alternatives/`:** Save `metrics.json` and the new figures (`fig73_salience_halo_trace.png`, `fig74_halo_ablation_and_pareto.png`, `fig75_six_paradigm_comparison.png`, `fig76_cross_model_qwen25_15b.png`) inside `results/exp27_kv_alternatives/`.
4. **Modal Discipline:** Enforce `timeout=300` on `@app.function`, run `modal app list` before and after to verify 0 running containers.
5. **No Hallucinated Metrics:** Every number in [`paper/manuscript_2_foundation_model_adaptation.md`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/manuscript_2_foundation_model_adaptation.md) and [`EXPERIMENTS.md`](file:///home/kdrana/Projects/OLSAttentionExploration/EXPERIMENTS.md) must come directly from `results/exp27_kv_alternatives/metrics.json` (and existing `exp18`, `exp21`, `exp23`, `exp26` JSONs).

---

## Step 1: Upgrade `src/cognitive_dual_control_attention.py` to Support All KV-Cache Alternatives & Ablation Knobs

Update [`src/cognitive_dual_control_attention.py`](file:///home/kdrana/Projects/OLSAttentionExploration/src/cognitive_dual_control_attention.py) so `CognitiveDualControlAttention` supports:
1. **All 5 Compressed Memory Paradigms (`mode`):**
   - `"halo"` (and alias `"cdca_v3"`): Full **HALO** pipeline — PRISM De-Rotated Pre-Cache Translation + RoPE Observation-Window Alignment + Bottom-Up Geometric Surprise ($z_{\text{norm}} + z_{\text{nov}}$) + Exponential Salience Halo ($\kappa(\delta) = \exp(-0.10|\delta|)$ with radius $K = \text{diffusion\_radius}$).
   - `"snapkv"`: Faithful **SnapKV** baseline (Li et al., 2024) — Post-RoPE observation-window attention only (`use_derotation=False`, `use_surprise=False`) with narrow local uniform average pooling (kernel size $5$, radius $K = 2$).
   - `"h2o"`: True **Causal-Prefix $\text{H}_2\text{O}$** (Zhang et al., 2023) — Cumulative causal attention across the historical prefix (`t <= sample_idx[q]`).
   - `"streaming_llm"`: **StreamingLLM** (Xiao et al., 2024) — Initial 4 attention sink tokens + local sliding window $W$.
   - `"hybrid_blur"`: **Matrix-State Replacement (Hybrid-BLUR)** — Replaces the historical token cache ($\tau < T - W$) with a fixed-rank ($r = 16$) un-rotated OLS / BLUR sufficient-statistic state compressed into $r = 16$ basis summary vectors via ridge-regularized Gramian projection, paired with the exact local sliding window $W$.
2. **Fine-Grained Ablation Flags on `CognitiveDualControlAttention`:**
   - `use_derotation: bool = True` (toggles PRISM pre-RoPE de-rotated alignment `derot_attn`).
   - `use_surprise: bool = True` (toggles bottom-up geometric surprise `s_bu`).
   - `diffusion_radius: int = 16` (sets $K \in \{0, 2, 4, 8, 16, 32\}$; when $K = 0$, no convolution is applied, selecting isolated top-$k$ tokens).
   - Store `self.last_salience_trace = {"s_td": ..., "s_bu": ..., "s_total": ..., "diffused": ...}` (detached CPU float lists when `record_trace=True`) so we can plot the exact 1D token-by-token salience trace around a needle for Figure 2!
3. **Cross-Architecture Support for Standard All-Attention Transformers (`Qwen/Qwen2.5-1.5B`):**
   - In `Qwen3.5-0.8B`, `q_proj` outputs `head_dim * 2` (query + sigmoid gate) and uses `q_norm`/`k_norm`.
   - In standard `Qwen2ForCausalLM` (`Qwen/Qwen2.5-1.5B`, 28 full-attention layers), `q_proj` outputs `num_heads * head_dim` directly (no gate chunking, no `q_norm`/`k_norm`, and `q_proj`/`k_proj`/`v_proj` have biases).
   - Detect whether `hasattr(orig_attn, "q_norm")` is True (`Qwen3.5` gated attention) or False (standard `Qwen2` / `Llama` attention) inside `CognitiveDualControlAttention.__init__` and `forward`, and add a helper `apply_halo_adapter_generic(model, layer_indices=None, ...)` so the exact same module wraps both `Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen2.5-1.5B` cleanly!

---

## Step 2: Create & Run `EXP-27` (`experiments/run_kv_alternatives_suite.py` & `experiments/modal_kv_alternatives_runner.py`)

Create [`experiments/run_kv_alternatives_suite.py`](file:///home/kdrana/Projects/OLSAttentionExploration/experiments/run_kv_alternatives_suite.py) and [`experiments/modal_kv_alternatives_runner.py`](file:///home/kdrana/Projects/OLSAttentionExploration/experiments/modal_kv_alternatives_runner.py) to execute four focused, fast experiments on an NVIDIA A10G GPU (`timeout=300`):

### Experiment 27A: The 6-Paradigm Alternatives to the KV-Cache Benchmark (`Qwen/Qwen3.5-0.8B-Base`)
Evaluate all 6 memory paradigms across our long-context stress suite at a selective landmark budget ($N = 256$ landmarks, $W = 1024$ local window, and $N = 1024$ landmarks, $W = 1024$ local window):
1. `original_softmax` (Full $O(T)$ KV-cache baseline)
2. `hybrid_blur` (Fixed-rank $r=16$ matrix-state replacement + local window)
3. `streaming_llm` (4 attention sinks + local window)
4. `h2o` (Causal-prefix heavy-hitter eviction)
5. `snapkv` (Post-RoPE observation window + narrow $K=2$ pooling, no PRISM/surprise)
6. `halo` (Full **HALO**: PRISM de-rotation + surprise + $K=16$ exponential salience halo)

Test each paradigm on:
- **18-Trial NIAH Grid** ($T \in \{1\text{k}, 2\text{k}, 4\text{k}, 8\text{k}, 16\text{k}, 32\text{k}\}$, depths $10\%, 50\%, 90\%$)
- **Retroactive Clause Framing at $16\text{k}$ & $32\text{k}$** (where the payload `"74921"` precedes the semantic cue `"secret emergency override authorization code"` by ~10 tokens—testing why `SnapKV`'s narrow $K=2$ window fails while `HALO`'s $K=16$ halo succeeds!)
- **Clause-Separated Unseen UUID (`"X9F-482-Q7B"`) at $16\text{k}$** (`"The target alpha-numeric registration identifier assigned to the primary node is strictly recorded in the vault as X9F-482-Q7B for verification."`)
- **3-Query Multi-Needle Synthesis at $16\text{k}$** (Alpha=`9182` at $15\%$, Beta=`3746` at $50\%$, Gamma=`5501` at $85\%$)
- **Expanded 25-Question MMLU & Long-Context Perplexity**

### Experiment 27B: Ablating the Halo Radius ($K \in \{0, 2, 4, 8, 16\}$) & Component Contributions
At $T = 16,384$ and landmark budget $N = 256$ (where token selection is genuinely selective: $256 / 15360 = 1.6\%$ of historical tokens retained!), evaluate composite exact-match recall across 6 diverse needle queries (Proactive $10\%$, Proactive $50\%$, Retroactive $10\%$, Clause-Separated UUID $50\%$, Multi-Needle Alpha $15\%$, Multi-Needle Beta $50\%$):
1. **Radius Sweep:** $K = 0$ (no halo / isolated top-$k$), $K = 2$ (narrow SnapKV radius), $K = 4$, $K = 8$, $K = 16$ (HALO default).
2. **Component Ablation (at $N = 256, T = 16,384$):**
   - Full HALO (`PRISM + RoPE + Surprise + Halo K=16`)
   - w/o Bottom-Up Surprise (`use_surprise=False`)
   - w/o PRISM Phase Rectification (`use_derotation=False`, post-RoPE only)
   - w/o Exponential Halo (`K=0`, isolated tokens only)
3. **1D Token Salience Trace Recording:** Record the exact token-by-token raw salience $s_\tau^{\text{total}}$ and diffused halo $\omega_\tau$ in a $\pm 35$-token window around the UUID needle `"X9F-482-Q7B"` at Layer 23, along with the decoded token strings, and save to `metrics["salience_trace_example"]`.

### Experiment 27C: Cache Budget Pareto Curve ($N \in \{64, 128, 256, 512, 1024\}$ at $T = 16,384$)
Evaluate composite needle retrieval accuracy (%) as a function of landmark cache capacity $N \in \{64, 128, 256, 512, 1024\}$ across `HALO`, `SnapKV`, `Causal H2O`, `StreamingLLM`, and `Hybrid-BLUR`.

### Experiment 27D: Cross-Architecture Validation on a Pure 28-Layer All-Attention Transformer (`Qwen/Qwen2.5-1.5B`)
Load `Qwen/Qwen2.5-1.5B` (a standard 1.5B-parameter, 28-layer full Softmax GQA Transformer with zero linear recurrent layers) in `bfloat16`. Wrap its attention layers with `HALO`, `SnapKV`, `Causal H2O`, and `StreamingLLM` ($W = 512, N = 256$), and evaluate NIAH retrieval across $T \in \{1024, 2048, 4096, 8192\}$ plus Retroactive Framing and UUID recall at $8\text{k}$.

### Generate 4 Publication-Ready Visual Figures in `results/exp27_kv_alternatives/`:
1. `fig73_salience_halo_trace.png`: **The Query-Answer Asymmetry & Salience Halo Trace.** A 2-panel plot showing (Left) the actual token-by-token raw query alignment $s_\tau^{\text{total}}$ vs. the diffused exponential halo $\omega_\tau$ across the tokens of the UUID clause (`"identifier ... is strictly recorded as X 9 F - 4 8 2 - Q 7 B"`), visually demonstrating how raw alignment spikes on cue words (`"identifier"`) while the exponential halo lifts the entire multi-token UUID `"X9F-482-Q7B"` above the top-$k$ admission threshold; and (Right) the PRISM De-Rotation effect ($R^2: 0.4014 \to 0.9893$ and $\mathcal{O}(1/T)$ spectral dilution curve of BLUR).
2. `fig74_halo_ablation_and_pareto.png`: **Halo Radius Ablation & Cache Budget Pareto Frontier.** (Left) Retrieval accuracy vs. Halo Diffusion Radius $K \in \{0, 2, 4, 8, 16\}$ showing why $K=0$ and $K=2$ fracture clauses while $K=16$ reaches $100\%$; (Right) Retrieval accuracy vs. Landmark Cache Budget $N \in \{64, 128, 256, 512, 1024\}$ across all methods.
3. `fig75_six_paradigm_comparison.png`: **Unified Comparison of the 6 KV-Cache Alternatives** (`Original Softmax`, `Hybrid-BLUR`, `StreamingLLM`, `Causal H2O`, `SnapKV`, `HALO`) across Single-Needle NIAH, Retroactive Framing, Clause-Separated UUID, Multi-Needle Synthesis, and MMLU Reasoning.
4. `fig76_cross_model_qwen25_15b.png`: **Cross-Architecture Generalization on `Qwen/Qwen2.5-1.5B` (28 Full-Attention Layers)** comparing `HALO` vs. `SnapKV`, `Causal H2O`, and `StreamingLLM`.

---

## Step 3: Complete Narrative Rewrite of `paper/manuscript_2_foundation_model_adaptation.md`

Completely rewrite [`paper/manuscript_2_foundation_model_adaptation.md`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/manuscript_2_foundation_model_adaptation.md) so that it is a cohesive, engaging, model-agnostic scientific paper (~250–300 lines, crisp and readable). Follow these exact structural requirements:

### 1. Title (Exact):
```markdown
# Preserving Exact Long-Range Recall in Compressed Transformers
```
*(Do NOT include "Qwen", "CDCA", or "Don't Blur the Needle" in the title.)*

### 2. Abstract (Single Cohesive Paragraph, ~160 Words, Zero `EXP-XX` Tags):
Use a single narrative paragraph explaining:
- The KV-cache bottleneck in long-context Transformer inference and the two dominant post-training compression paradigms (recurrent matrix states vs. selective token eviction).
- Why fixed-rank matrix states (**BLUR**) fail on exact recall: even when fit via closed-form operator regression and rotary phase rectification (**PRISM**) to explain $>95\%$ of attention variance and preserve $100\%$ MMLU with a local window, any rank-$r$ state acts as a low-pass spectral filter that dilutes individual facts at $\mathcal{O}(1/T)$.
- Why existing token-eviction methods (**StreamingLLM**, **$\text{H}_2\text{O}$**, **SnapKV**) fail on complex entities: causal prefix scoring misses unqueried facts, and narrow or greedy top-$k$ selection suffers from **Query-Answer Asymmetry**—retaining the semantic cue words that match the query while severing or evicting the adjacent multi-token payload (numbers, passcodes, UUIDs).
- How **HALO** resolves both limits without retraining by pairing an exact local sliding window with PRISM phase-rectified salience, bottom-up geometric surprise, and a wide **Exponential Salience Halo** ($\kappa = 16$) that preserves complete factual clauses across both hybrid (`Qwen3.5-0.8B`) and standard all-attention (`Qwen2.5-1.5B`) Transformers at $16\times$–$128\times$ KV-cache compression.

### 3. Section 1: Introduction — Alternatives to the KV-Cache and Transformer Component Replacement
Write a thorough, scholarly introduction (citing all relevant literature) organized around:
- **1.1 Why the KV-Cache Grows Linearly and Approaches to Replacing Transformer Components:** Contrast clean-slate pretraining with post-training surgical replacement. Explain why token-level distillation (*SUPRA*, *MOHAWK*, *LoLCATs*) risks domain drift on small calibration sets, motivating functional operator alignment.
- **1.2 Taxonomy of the Four Alternatives to the Full KV-Cache:**
  1. *Fixed-Size Recurrent & Test-Time Regression States ($O(1)$ Matrix Memory):* Linear Attention (Katharopoulos et al., 2020), Mamba-2 (Dao & Gu, 2024), Gated DeltaNet (Yang et al., 2025), TTT (Sun et al., 2024), Titans (Behrouz et al., 2025), and our closed-form OLS / **BLUR** recurrence.
  2. *Head- & Latent-Space KV Compression:* MQA/GQA and DeepSeek-V2 Multi-Head Latent Attention (MLA, DeepSeek-AI, 2024), highlighting the fundamental incompatibility between Rotary Position Embeddings (RoPE, Su et al., 2024) and compressed latent states.
  3. *Windowed Sinks & Online Causal Eviction:* *StreamingLLM* (Xiao et al., 2024), *$\text{H}_2\text{O}$* (Zhang et al., 2023), and *Scissorhands* (Liu et al., 2023), and why causal prefix accumulation is blind to end-of-prompt queries.
  4. *Observation-Window Selection & Contiguous Salience Diffusion:* From *SnapKV* (Li et al., 2024) and *Quest* (Tang et al., 2024) to our **HALO** architecture.
- **1.3 Summary of Core Findings:** Briefly state our three main scientific contributions (The Low-Pass Limit of Matrix States & PRISM Phase Rectification; The Query-Answer Asymmetry Principle & Entity Fragmentation; and HALO's Contiguous Salience Diffusion).

### 4. Section 2: Why Fixed-Rank Matrix States Blur Exact Recall (The BLUR Regime)
- Present the clean theory and key results without `EXP-XX` clutter:
  - **Representational Shock Theorem:** Why hard-swapping intermediate layers collapses frozen downstream MLPs (MMLU drops $100\% \to 10\%$), and why an exact local sliding window eliminates shock ($100\%$ MMLU).
  - **PRISM Phase Rectification (The De-Rotation Theorem):** Why RoPE phase spin causes high-frequency cancellation in compressed states, and how stripping RoPE via `PreCacheTranslationLayer` (**PRISM**) jumps operator regression fidelity from $R^2 = 0.4014 \to 0.9893$ (and $0.9505$ mean across layers in $21.95\text{s}$).
  - **The Spectral Dilution Horizon ($\mathcal{O}(1/T)$):** Prove why even a phase-rectified, locally windowed matrix state (**Hybrid-BLUR**, $932\times$ compression, $432\text{ KB}$) acts as a low-pass filter that scores $0\%$ on long-range passkey retrieval beyond the local window.
  - Embed `fig73_salience_halo_trace.png`.

### 5. Section 3: Query-Answer Asymmetry and the HALO Architecture
- Formulate the **Query-Answer Asymmetry Principle**: A prompt question contains the semantic *cue* (`"authorization code"`), not the unknown *answer* (`"74921"` or `"X9F-482-Q7B"`). In multi-token or retroactive clauses, the answer tokens sit $\delta \in [-16, +16]$ tokens away from the cue tokens.
- Present the 4-stage **HALO** pipeline concisely:
  1. **Exact Local Sliding Window ($W$)**
  2. **PRISM Phase-Rectified & Rotary Observation-Window Alignment ($s_\tau^{\text{td}}$)**
  3. **Bottom-Up Geometric Surprise ($s_\tau^{\text{bu}}$ via L2 norm $z_{\text{norm}}$ and angular novelty $z_{\text{nov}}$)**
  4. **The Exponential Salience Halo ($\kappa(\delta) = \exp(-0.10|\delta|), K = 16$)**

### 6. Section 4: Experimental Evaluation Across Paradigms and Model Families
- Present the unified results clearly using the new figures and focused tables (pulling exact numbers from `results/exp27_kv_alternatives/metrics.json` and `results/exp26_cdca_v3_ablation/metrics.json`):
  - **4.1 Six-Paradigm Comparison (`Original Softmax`, `Hybrid-BLUR`, `StreamingLLM`, `Causal H2O`, `SnapKV`, `HALO`):** Embed `fig75_six_paradigm_comparison.png` and a clean summary table.
  - **4.2 Ablating the Halo Radius ($K$) and Cache Budget Pareto Curve ($N$):** Embed `fig74_halo_ablation_and_pareto.png` and show the exact ablation numbers for $K \in \{0, 2, 4, 8, 16\}$ and component removals (w/o PRISM, w/o Surprise, w/o Halo).
  - **4.3 Cross-Architecture Generalization on a Pure 28-Layer All-Attention Transformer (`Qwen2.5-1.5B`) & Hardware Scaling:** Embed `fig76_cross_model_qwen25_15b.png` and `fig72_memory_and_ppl_scaling.png`.

### 7. Section 5: Discussion, Conclusion, and References
- Synthesize when practitioners should use **Hybrid-BLUR** ($932\times$ compression for global reasoning/coding where exact passkey lookup isn't needed) vs. **HALO** ($16\times$–$128\times$ compression with $100\%$ exact long-range entity recall).
- Ensure [`paper/references.bib`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/references.bib) and the References section include all cited works (Katharopoulos 2020, Dao & Gu 2024, Mercat 2024, Bick 2024, Zhang 2025, DeepSeek-AI 2024, Su 2024, Xiao 2024, Zhang 2023, Liu 2023, Li 2024, Tang 2024, Cai 2024, Sun 2024, Behrouz 2025, Yang 2025).

---

## Step 4: Update `EXPERIMENTS.md` & Verify Spin-Down
1. Add `EXP-27: Unified KV-Cache Alternatives, Halo Radius Ablation & Cross-Model Generalization` to [`EXPERIMENTS.md`](file:///home/kdrana/Projects/OLSAttentionExploration/EXPERIMENTS.md) with exact numbers and links to `results/exp27_kv_alternatives/metrics.json` and `fig73`–`fig76`.
2. Verify `modal app list` shows zero active containers.
3. Commit and push all updated files and artifacts to `main`.

---

## Execution Status: COMPLETED & VERIFIED (October 2026)
- **Step 1:** Upgraded `src/cognitive_dual_control_attention.py` with generic layer hooking, all 5 memory paradigms, ablation flags, and trace recording.
- **Step 2:** Executed `EXP-27` on Modal Cloud GPU (A10G). Generated `results/exp27_kv_alternatives/metrics.json` and figures `fig73` through `fig76`.
- **Step 3:** Rewrote `paper/manuscript_2_foundation_model_adaptation.md` under the title *"Preserving Exact Long-Range Recall in Compressed Transformers"*, updated LaTeX manuscript `paper/manuscript_2_foundation_model_adaptation.tex`, and compiled PDF with tectonic.
- **Step 4:** Logged `EXP-27` in `EXPERIMENTS.md`, verified 0 active Modal containers, and committed all changes.
