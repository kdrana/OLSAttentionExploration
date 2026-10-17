# Action Plan v4: Reviewer-Proof Multi-Family, High-Power & Literature-Benchmark Suite (`EXP-28`)

**Target Model for Execution:** Gemini Flash  
**Estimated Cloud GPU Cost:** $< \$0.35$ (Modular Modal NVIDIA A10G functions, each strictly enforcing `timeout=300`)  
**Goal:** Upgrade the empirical foundation of [`paper/manuscript_2_foundation_model_adaptation.md`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/manuscript_2_foundation_model_adaptation.md) (*"Preserving Exact Long-Range Recall in Compressed Transformers"*) to meet top-tier conference standards (NeurIPS / ICLR / ICML) by addressing five critical reviewer requirements:
1. **Cross-Company Model Diversity (3 Distinct Organizations & Architectures):** Alibaba (`Qwen/Qwen3.5-0.8B-Base`, Hybrid Gated-DeltaNet + GQA), Meta (`unsloth/Llama-3.2-1B-Instruct`, Pure GQA RoPE), and HuggingFace (`HuggingFaceTB/SmolLM2-1.7B-Instruct`, Pure MHA RoPE).
2. **High Statistical Power ($n = 25\text{–}30$ Randomized Trials per Condition):** Replace low-sample ($n=4$ or $n=6$) probes with $n = 25\text{–}30$ randomized trials per point (varying needle depths, random passkeys/UUIDs, and essay haystacks) and report **Mean $\pm$ 95% Confidence Intervals (`mean`, `std`, `ci95`)**.
3. **Novel Diagnostic Proof of Our Thesis:** The **Query-Answer Separation ($\Delta_{\text{QA}}$) & Value-Span Length ($L_{\text{val}}$) Phase Diagram**, proving mathematically and empirically why narrow pooling (`SnapKV`) collapses as soon as natural prose separates a subject cue from its value by $\Delta_{\text{QA}} \ge 4$ tokens.
4. **Canonical Literature Replication (`RULER` & `LongBench`-Style Suite) + Unifying `HALO + BLUR`:** Replicate the 4 RULER axes (Single-NIAH, Multi-Key Distractor NIAH, 2-Hop Variable Tracking, Common Word Aggregation) and multi-sentence Extractive QA / Summarization, demonstrating how unified **`HALO + BLUR`** combines exact needle retention (`HALO`) with compressed background gist (`BLUR`).
5. **Reviewer-Proof Ablations, Multi-Turn Durability & Hardware Profiling:** Leave-one-out ablation of all 4 HALO components, **Turn-1 vs. Turn-2 Unseen Follow-Up Query Recall** (where `SnapKV` fails because it evicted Turn 2's entity during Turn 1), $(\beta, \gamma)$ sensitivity heatmap, and CUDA wall-clock latency / decode throughput / peak VRAM profiling ($1\text{k} \to 32\text{k}$).

---

## 0. Mandatory Repository Rules (Read Before Executing)
1. **Reusable Code in `src/`:** All attention wrapper upgrades belong in [`src/cognitive_dual_control_attention.py`](file:///home/kdrana/Projects/OLSAttentionExploration/src/cognitive_dual_control_attention.py).
2. **Executable Benchmarks in `experiments/`:** Create [`experiments/run_comprehensive_review_suite.py`](file:///home/kdrana/Projects/OLSAttentionExploration/experiments/run_comprehensive_review_suite.py) and [`experiments/modal_exp28_runner.py`](file:///home/kdrana/Projects/OLSAttentionExploration/experiments/modal_exp28_runner.py).
3. **Structured Artifacts in `results/exp28_comprehensive_review_suite/`:** Save `metrics.json` and 6 publication-ready figures (`fig77` through `fig82`) inside `results/exp28_comprehensive_review_suite/`.
4. **Modal Discipline (`timeout=300`):** Every `@app.function` on Modal **must** set `timeout=300`. Because `EXP-28` evaluates 3 models and 5 sub-suites, **split the Modal runner into 2 or 3 separate `@app.function(..., timeout=300)` calls** (e.g., `run_part1_diagnostics_and_ruler()`, `run_part2_cross_company_models()`, `run_part3_ablations_and_profiling()`) invoked sequentially by `@app.local_entrypoint()` so no single container ever approaches 300s! Run `modal app list` before and after to verify 0 running containers.
5. **No Plot Fallback Bugs or Hallucinated Metrics:** Plots must read exact arrays from `metrics.json` without `if sum(curve) == 0:` dummy fallbacks. All numbers in [`paper/manuscript_2_foundation_model_adaptation.md`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/manuscript_2_foundation_model_adaptation.md) and [`EXPERIMENTS.md`](file:///home/kdrana/Projects/OLSAttentionExploration/EXPERIMENTS.md) must match `results/exp28_comprehensive_review_suite/metrics.json`.

---

## Step 1: Upgrade `src/cognitive_dual_control_attention.py`

Update [`src/cognitive_dual_control_attention.py`](file:///home/kdrana/Projects/OLSAttentionExploration/src/cognitive_dual_control_attention.py) to add four clean capabilities to `CognitiveDualControlAttention` and `apply_halo_adapter_generic`:

1. **Configurable Kernel Shape & Hyperparameters (`kernel_type`, `surprise_beta`, `decay_gamma`):**
   - Add `__init__` arguments:
     - `kernel_type: str = "exponential"` (`"exponential"` uses $\kappa(\delta) = \exp(-\gamma |\delta|)$; `"uniform_box"` uses uniform average pooling $\mathbf{1}/(2K+1)$ of radius $K$, allowing direct comparison of Exponential Decay vs. Uniform Box Pooling at the same radius $K=16$).
     - `surprise_beta: float = 0.15` (weight on bottom-up surprise $s_{\text{total}} = s_{\text{td}} + \beta s_{\text{bu}}$).
     - `decay_gamma: float = 0.10` (exponential decay rate in $\exp(-\gamma |\delta|)$).

2. **Unified `"halo_blur"` Mode (Combining `HALO` Exact Landmarks + `BLUR` Residual State):**
   - When `mode == "halo_blur"`:
     - Select $k_{\text{halo}} = \max(1, N - 16)$ salient tokens using the full `HALO` pipeline (`salient_inds`).
     - For the remaining evicted historical tokens ($\tau \in [0, \text{hist\_len}) \setminus \text{salient\_inds}$), partition them into $r = 16$ contiguous temporal blocks and compute their ridge-regularized basis summary keys and values (mean-pooled / L2-magnitude-preserved block summary vectors $\bar{K}_b, \bar{V}_b$ for $b = 1 \dots 16$), concatenating those $16$ `BLUR` summary states with the $N - 16$ exact `HALO` tokens!
     - This keeps the total historical cache budget strictly equal to $N$ tokens while retaining global background statistics from the evicted tokens.

3. **Multi-Turn Frozen-Cache Evaluation (`freeze_salient_cache: bool = False`):**
   - Add an attribute `self.freeze_salient_cache: bool = False` (and helper `set_freeze_salient_cache(model, freeze: bool)`).
   - When `self.freeze_salient_cache == True` and `self.salient_k_cache is not None`:
     - During a Turn-2 prefill (where the input is `[Turn 2 Question]` or when re-evaluating a new question against the already-compressed document cache), **do not recompute `salient_inds` from the full document**; instead, keep `self.salient_k_cache` and `self.salient_v_cache` frozen from Turn 1, and attend over `cat([self.salient_k_cache, local_turn2_k], dim=2)`!
     - This directly tests **Multi-Turn Durability**: in Turn 1, the user asks about Entity A (compressing the document KV cache to $N$ tokens). In Turn 2, the user asks about Entity B using only the compressed cache from Turn 1 + Turn 2's question tokens.

4. **CUDA Timing Instrumentation (`last_selection_time_ms`):**
   - Record the wall-clock time spent inside `compute_salience_indices` (using `time.perf_counter()` after `torch.cuda.synchronize()` when `record_trace=True`) as `self.last_selection_time_ms` so we can report exact selection overhead in milliseconds.

---

## Step 2: Implement `EXP-28` (`experiments/run_comprehensive_review_suite.py` & `experiments/modal_exp28_runner.py`)

Design the benchmark suite so that statistical sweeps run at context length $T = 4,096$ with local window $W = 512$ and historical cache budget $N = 128$ (retaining only $128 / 3584 = 3.5\%$ of historical tokens, i.e., **$28\times$ historical KV compression**), where a single trial takes only ~0.10s on an A10G GPU—allowing **$n = 25\text{–}30$ randomized trials per condition** to complete fast and cleanly.

Include a helper `compute_stats(binary_or_float_list)` that returns:
```python
{
    "n": len(vals),
    "mean": round(float(np.mean(vals) * 100.0), 2),
    "std": round(float(np.std(vals, ddof=1) * 100.0), 2),
    "sem": round(float(np.std(vals, ddof=1) / np.sqrt(len(vals)) * 100.0), 2),
    "ci95": round(float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) * 100.0), 2),
}
```
*(If `vals` is already on a `0..100` scale, omit the `* 100.0` multiplier.)*

---

### Sub-Experiment 28A: Cross-Company 3-Model Statistical Benchmark ($n = 30$ Trials per Method per Model)
Evaluate **three models from three different organizations**:
1. **Alibaba:** `Qwen/Qwen3.5-0.8B-Base` (24 layers: 18 Gated-DeltaNet + 6 GQA Full-Attention layers, $d_{\text{head}}=256$)
2. **Meta:** `unsloth/Llama-3.2-1B-Instruct` (16 Pure GQA RoPE layers, $d_{\text{head}}=64$; fallback to `NousResearch/Llama-3.2-1B` if needed)
3. **HuggingFace:** `HuggingFaceTB/SmolLM2-1.7B-Instruct` (24 Pure MHA RoPE layers, $d_{\text{head}}=64$)

- **Evaluation Protocol ($n = 30$ trials per method on each model):**
  - Context length $T = 4,096$, local window $W = 512$, historical landmark budget $N = 128$.
  - 5 needle depths $\{10\%, 28\%, 46\%, 64\%, 82\%\}$ (all strictly outside the final $W = 512$ local window!) $\times$ 6 randomized trials per depth (each with a freshly generated 6-character alphanumeric code `f"{rng.randint(100,999)}-{rng.choice(['AX','BQ','CZ','KM','LP'])}"` and naturally phrased clause `"The mandatory security clearance token assigned to Sector {sector_name} during the audit is recorded as {code} in the registry."`, queried at the end with `"Question: What is the mandatory security clearance token assigned to Sector {sector_name}? Answer: The token is"`).
  - Compare all 7 modes: `original_softmax`, `hybrid_blur`, `streaming_llm`, `h2o`, `snapkv`, `halo`, and `halo_blur`.
  - Log `mean`, `std`, `sem`, `ci95`, and per-depth breakdown in `metrics["exp28a_cross_company_models"]`.

---

### Sub-Experiment 28B: The Query-Answer Separation ($\Delta_{\text{QA}}$) & Value-Span Length ($L_{\text{val}}$) Phase Diagram ($n = 25$ Trials per Cell)
Run on `Qwen/Qwen3.5-0.8B-Base` ($T = 4,096, W = 512, N = 128$):
1. **Part 1 — Syntactic Cue-to-Value Separation Sweep ($\Delta_{\text{QA}} \in \{1, 3, 6, 10, 15, 22\}$ intervening words/tokens, $n = 25$ trials per distance):**
   - Construct needle sentences where the queried cue (`"Project {Name} clearance code"`) and the target answer (`"{passkey}"`) are separated by a grammatical Filler Clause of exact length $\Delta_{\text{QA}}$ words:
     - $\Delta_{\text{QA}} = 1$: `"The Project {Name} clearance code is {passkey}."`
     - $\Delta_{\text{QA}} = 3$: `"The Project {Name} clearance code currently is {passkey}."`
     - $\Delta_{\text{QA}} = 6$: `"The Project {Name} clearance code verified by the director is {passkey}."`
     - $\Delta_{\text{QA}} = 10$: `"The Project {Name} clearance code verified by the senior operations director in Geneva is {passkey}."`
     - $\Delta_{\text{QA}} = 15$: `"The Project {Name} clearance code verified by the senior operations director in Geneva during the winter summit is {passkey}."`
     - $\Delta_{\text{QA}} = 22$: `"The Project {Name} clearance code verified by the senior operations director in Geneva during the winter diplomatic summit after thorough cryptographic inspection is {passkey}."`
   - Prompt query at end of $4\text{k}$ context: `"What is the Project {Name} clearance code? Answer:"`
   - Evaluate $n = 25$ randomized trials (5 depths $\times$ 5 random passkeys) at each $\Delta_{\text{QA}}$ across:
     - `Isolated Top-k (K=0)`
     - `SnapKV (K=2)`
     - `HALO (K=8)`
     - `HALO (K=16)`
     - `Causal H2O`
2. **Part 2 — Target Value-Span Length Sweep ($L_{\text{val}} \in \{1, 3, 6, 10, 15\}$ segments/tokens, $n = 25$ trials per length):**
   - Vary the target answer from a 1-chunk code (`"8472"`) to a 3-chunk, 6-chunk, 10-chunk, and 15-chunk hyphenated serial identifier (e.g., `"8472-X9F-Q7B-319-K4M"`), measuring **Sub-Span Recall / Token F1 (%)** and **Exact Full-String Match (%)** across `Isolated Top-k (K=0)`, `SnapKV (K=2)`, `HALO (K=16)`, and `Causal H2O`.

---

### Sub-Experiment 28C: Canonical Literature Benchmarks — RULER 4-Axis Suite & Multi-Sentence QA / Summarization ($n = 25$ Trials per Task)
Replicate the four canonical task families from **RULER** (Hsieh et al., COLM 2024) plus **LongBench-style** multi-sentence extractive QA & global summarization ($T = 4,096, W = 512, N = 128$):
1. **RULER Axis 1 — Single-NIAH (S-NIAH, $n = 25$):** Standard passkey needle across 5 depths $\times$ 5 seeds.
2. **RULER Axis 2 — Multi-Key Adversarial Distractor NIAH (MK-NIAH, $n = 25$):**
   - Insert **12 adversarial distractor sentences** throughout the haystack that share the exact query keywords (e.g., `"The vault key for Archive-Distractor-{i} is normal."`) plus **1 true target needle** (`"The vault key for Archive-Target is {code}."`).
   - Tests whether distractors hijack top-down observation attention and saturate narrow caches.
3. **RULER Axis 3 — Multi-Hop Variable Tracking (2-Hop VT, $n = 25$):**
   - At depth $d_1 \in [15\%, 35\%]$: `"System initialization: primary register ALPHA_REG is set to numeric value {code}."`
   - At depth $d_2 \in [55\%, 75\%]$: `"Pointer binding: secondary register BETA_PTR is linked directly to primary register ALPHA_REG."`
   - End-of-prompt query: `"Based on the pointer binding to ALPHA_REG, what is the numeric value of BETA_PTR? The numeric value is"` (and also test the harder indirect query `"What is the numeric value of BETA_PTR? The numeric value is"`).
   - Compare `SnapKV`, `Top-Down HALO (use_surprise=False)`, and `Full HALO (use_surprise=True)`!
4. **RULER Axis 4 — Common Words / Global Frequency Aggregation (CWE, $n = 25$):**
   - Inject 3 target code words (e.g., `"COBALT"`, `"ZENITH"`, `"QUARTZ"`) repeated 8 times each across the historical prefix ($< T - W$), alongside 15 rare words repeated once. Prompt asks for the top-3 recurring code words.
   - Compare `StreamingLLM`, `Causal H2O`, `SnapKV`, `Hybrid-BLUR`, `HALO`, and **`HALO + BLUR` (`halo_blur`)**.
5. **LongBench-Style Multi-Sentence Extractive QA & Document Gist Summarization ($n = 25$ trials):**
   - Insert a 3-sentence factual report about a scientific mission (containing 4 distinct facts: launch site, lead scientist, payload mass, and orbital altitude) into the historical prefix, and evaluate **Fact F1 / ROUGE-L (%)** when asked to summarize all 4 mission facts at the end of the prompt across all methods including **`HALO + BLUR`**.

---

### Sub-Experiment 28D: Component Leave-One-Out Ablation, Multi-Turn Unseen Follow-Up Queries & $(\beta, \gamma)$ Sensitivity ($n = 25$ Trials)
1. **Incremental / Leave-One-Out Component Ablation ($n = 25$ composite trials across S-NIAH, Retroactive Framing, Clause-Separated UUID, and Multi-Hop/Distractor NIAH at $T = 4,096, W = 512, N = 128$):**
   - `(1) Isolated Top-k`: `use_derotation=False, use_surprise=False, diffusion_radius=0`
   - `(2) SnapKV Baseline`: `mode="snapkv"` (Uniform Box Pool $K=2$, post-RoPE only)
   - `(3) Wide Uniform Box Pool (K=16)`: `use_derotation=False, use_surprise=False, diffusion_radius=16, kernel_type="uniform_box"`
   - `(4) Exponential Decay Kernel Only (K=16)`: `use_derotation=False, use_surprise=False, diffusion_radius=16, kernel_type="exponential"`
   - `(5) Exponential Kernel + PRISM De-Rotation`: `use_derotation=True, use_surprise=False, diffusion_radius=16, kernel_type="exponential"`
   - `(6) Full HALO (+ Bottom-Up Surprise)`: `use_derotation=True, use_surprise=True, diffusion_radius=16, kernel_type="exponential"`
2. **Multi-Turn Unseen Follow-Up Query Durability ($n = 25$ trials at $T = 4,096, W = 512, N = 128$):**
   - Each document contains two distinct facts in the historical prefix:
     - Fact 1 (at depth $d_1$): `"The planetary lander velocity reading is {code_1} meters per second."`
     - Fact 2 (at depth $d_2$, formatted as a high-salience anomaly): `"CRITICAL ANOMALY ALERT [NODE-99]: Core reactor pressure spiked to {code_2} Pascals!"`
   - **Turn 1:** Prompt asks `"What is the planetary lander velocity reading?"` $\rightarrow$ All methods (`SnapKV`, `Top-Down HALO (use_surprise=False)`, `Full HALO (use_surprise=True)`) compress the historical KV cache to $N = 128$ tokens based on Turn 1.
   - **Turn 2 (Frozen Historical Cache):** Without re-prefilling the document, freeze `salient_k_cache` and `salient_v_cache` from Turn 1, and ask Turn 2's question: `"What pressure value in Pascals was reported in the CRITICAL ANOMALY ALERT?"`
   - Measure **Turn-1 Accuracy (%)** and **Turn-2 Unseen Query Accuracy (%)** ($n = 25$ trials). Because `Full HALO` uses query-independent **Bottom-Up Geometric Surprise ($s_\tau^{\text{bu}}$)**, it retains the anomaly alert during Turn 1 even though Turn 1 only asked about velocity, whereas pure top-down `SnapKV` evicts Fact 2!
3. **Hyperparameter Sensitivity Grid ($\beta \times \gamma$):**
   - Evaluate $\beta \in \{0.0, 0.05, 0.15, 0.30, 0.50\}$ and $\gamma \in \{0.02, 0.05, 0.10, 0.25, 0.50\}$ ($n = 15$ trials per cell) to verify a broad, stable performance plateau around $(\beta=0.15, \gamma=0.10)$.

---

### Sub-Experiment 28E: Hardware Wall-Clock Latency (TTFT), Selection Overhead (ms), Decode Throughput (tok/s) & Peak GPU VRAM ($1\text{k} \to 32\text{k}$)
Across $T \in \{1024, 2048, 4096, 8192, 16384, 32768\}$ on `Qwen/Qwen3.5-0.8B-Base` ($W = 1024, N = 256$):
- Measure and log for `original_softmax`, `h2o`, `snapkv`, `halo`, and `halo_blur`:
  1. **Prefill Latency / Time-To-First-Token (`ttft_ms`)**
  2. **Salience Selection Overhead (`selection_overhead_ms`)** (show that HALO's 1D `F.conv1d` + L2 norm takes $< 2\text{ ms}$, vs. $\text{H}_2\text{O}$'s prefix sampling overhead)
  3. **Autoregressive Decode Throughput (`decode_tok_per_sec`)** over 32 generated tokens
  4. **retained KV Cache Memory (`kv_cache_mib`)** and **Peak Allocated GPU Memory (`peak_vram_mib`)**

---

## Step 3: Generate 6 Publication-Ready Figures in `results/exp28_comprehensive_review_suite/`

Write a clean plotting function `generate_exp28_figures(metrics, out_dir)` (invoked both on Modal and locally) that creates 6 high-DPI (`dpi=300`) figures with error bars (`ci95` / `sem`):
1. `fig77_cross_company_3models.png`: **Cross-Company Generalization Across 3 Model Families ($n=30$ per bar with 95% CI error bars).** 3-panel grouped bar/depth chart comparing `Original Softmax`, `Hybrid-BLUR`, `StreamingLLM`, `Causal H2O`, `SnapKV`, `HALO`, and `HALO+BLUR` across **Alibaba (`Qwen3.5-0.8B`)**, **Meta (`Llama-3.2-1B`)**, and **HuggingFace (`SmolLM2-1.7B`)**.
2. `fig78_qa_separation_and_span_phase_transition.png`: **The Query-Answer Asymmetry Phase Transition ($n=25$ per point with 95% CI bands).** (Left) Recall (%) vs. Syntactic Cue-to-Value Separation $\Delta_{\text{QA}} \in \{1, 3, 6, 10, 15, 22\}$ tokens showing `SnapKV` ($K=2$) collapsing as soon as $\Delta_{\text{QA}} \ge 6$ while `HALO` ($K=16$) holds; (Right) Token F1 & Exact Match vs. Target Value-Span Length $L_{\text{val}} \in \{1..15\}$.
3. `fig79_ruler_and_longbench_suite.png`: **Canonical RULER 4-Axis & Extractive Summarization Suite ($n=25$ per task).** Grouped comparison across `S-NIAH`, `MK-NIAH (Distractors)`, `2-Hop Variable Tracking`, `CWE Frequency Aggregation`, and `Multi-Sentence Summarization (Fact F1)`, highlighting where `HALO` wins on exact retrieval and where `HALO + BLUR` boosts global frequency/summarization!
4. `fig80_component_ablation_and_multiturn.png`: **Leave-One-Out Component Ablation & Turn-2 Unseen Follow-Up Query Durability.** (Left) Incremental bar chart from `Isolated Top-k (K=0)` $\to$ `SnapKV (Box K=2)` $\to$ `Wide Box Pool (K=16)` $\to$ `Exp Kernel (K=16)` $\to$ `+ PRISM De-Rotation` $\to$ `Full HALO (+ Surprise)`; (Right) **Turn-1 vs. Turn-2 Unseen Follow-Up Query Recall (%)** showing `SnapKV` collapsing on Turn 2 while `Full HALO` preserves unseen anomalies via Bottom-Up Surprise.
5. `fig81_hyperparameter_sensitivity_heatmap.png`: **2D Hyperparameter Sensitivity Heatmap ($\beta \times \gamma$).** Shows a wide, flat plateau of $>90\%$ recall across bottom-up surprise weight $\beta$ and exponential decay rate $\gamma$.
6. `fig82_hardware_latency_and_vram_scaling.png`: **Hardware Efficiency Scaling ($1\text{k} \to 32\text{k}$).** 3-panel plot of (a) Retained KV Cache Memory (MiB), (b) Decode Throughput (tokens/sec), and (c) Prefill + Salience Selection Overhead (ms) across sequence lengths.

---

## Step 4: Update `EXPERIMENTS.md` and `paper/manuscript_2_foundation_model_adaptation.md`

1. **Log `EXP-28` (`EXP-28-COMPREHENSIVE-REVIEW-SUITE`) in [`EXPERIMENTS.md`](file:///home/kdrana/Projects/OLSAttentionExploration/EXPERIMENTS.md):**
   - Include exact run commands, Modal GPU time, clickable links to [`results/exp28_comprehensive_review_suite/metrics.json`](file:///home/kdrana/Projects/OLSAttentionExploration/results/exp28_comprehensive_review_suite/metrics.json) and `fig77`–`fig82`, and exact quantitative takeaways.
2. **Integrate `EXP-28` into [`paper/manuscript_2_foundation_model_adaptation.md`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/manuscript_2_foundation_model_adaptation.md):**
   - Keep the crisp title (`# Preserving Exact Long-Range Recall in Compressed Transformers`), concise single-paragraph Abstract, and Section 1 KV-Cache Taxonomy.
   - Update the empirical sections to feature:
     - **Act I:** Continuous Operator Replacement (`BLUR`) & the Unified **`HALO + BLUR`** synthesis (exact landmarks + compressed residual background).
     - **Act II:** The **Query-Answer Separation ($\Delta_{\text{QA}}$) & Value-Span Length ($L_{\text{val}}$) Phase Transition** (`fig78`, $n=25$ per cell with 95% CIs) + Leave-One-Out Component Ablation & **Turn-2 Unseen Follow-Up Query Durability** (`fig80`) + Hyperparameter Robustness (`fig81`).
     - **Act III:** Canonical **RULER 4-Axis Suite & Multi-Sentence Summarization** (`fig79`, $n=25$ per task).
     - **Act IV:** **Cross-Company Generalization Across Alibaba (`Qwen3.5-0.8B`), Meta (`Llama-3.2-1B`), and HuggingFace (`SmolLM2-1.7B`)** (`fig77`, $n=30$ per model) + **Hardware Latency, Decode Throughput & VRAM Scaling** (`fig82`).

---

## Execution Status: COMPLETED & VERIFIED (October 2026)
- **Step 1 (Codebase Upgrade):** Upgraded `src/cognitive_dual_control_attention.py` with configurable kernels (`uniform_box` vs `exponential`), `decay_gamma`, `surprise_beta`, unified `halo_blur` mode, `freeze_salient_cache` for multi-turn evaluations, and wall-clock selection timing instrumentation.
- **Step 2 (Execution):** Executed EXP-28 across 5 sub-experiments (28A-28E) on Modal NVIDIA A10G Cloud GPUs enforcing strict `timeout <= 300` per worker. Generated `results/exp28_comprehensive_review_suite/metrics.json`.
- **Step 3 (Figures):** Generated 6 publication figures (`fig77` through `fig82`) at 300 DPI with 95% CI error bars and copied to `paper/` and brain artifact directory.
- **Step 4 (Registry & Manuscript):** Logged EXP-28 in `EXPERIMENTS.md`, updated `paper/manuscript_2_foundation_model_adaptation.md`, updated LaTeX `paper/manuscript_2_foundation_model_adaptation.tex`, and compiled PDF with tectonic. Confirmed all Modal apps stopped.

