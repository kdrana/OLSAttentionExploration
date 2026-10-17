# Action Plan: Upgrading Manuscript 2 (`foundation_model_adaptation`) & Running EXP-26

**Target Manuscript:** [`paper/manuscript_2_foundation_model_adaptation.md`](manuscript_2_foundation_model_adaptation.md)  
**Supporting Bibliography:** [`paper/references.bib`](references.bib)  
**Master Registry:** [`EXPERIMENTS.md`](../EXPERIMENTS.md)  
**Core Module to Upgrade:** [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py)

> **Instructions for the Executing Agent (Flash 3.8):**  
> Follow the phases below sequentially. Adhere strictly to [`AGENTS.md`](../AGENTS.md):
> 1. Put reusable classes/functions in `src/`, runners in `experiments/`, and outputs in `results/exp26_cdca_v3_ablation/`.
> 2. Enforce `timeout=300` on all Modal GPU functions and verify spin-down via `modal app list`.
> 3. Never hallucinate or round numbers without querying the exact `results/<exp_id>/metrics.json` files.

---

## Phase 1: Fix Technical Bugs & Eliminate Keyword Leakage in `src/cognitive_dual_control_attention.py`

### 1.1 Audit of Current Issues to Fix
1. **Oracle Keyword Leakage (`ALERT_KEYWORDS`):**
   - Currently, [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py) (lines 30–34) hardcodes `ALERT_KEYWORDS = ["critical", "secret", "access", "code", "antigravity", "74921", ...]` and decodes input token IDs to strings on the CPU in `set_salient_landmark_indices()`.
   - Including `"74921"` and `"antigravity"` is target leakage. Furthermore, because `beacon_signal` is binary (`0.0` or `1.0`), when the haystack repeats to $T = 16,384$ or $32,768$, more than $1,024$ tokens match words like `"note"`, `"code"`, or `"key"`. `torch.topk(diffused, 1024)` breaks ties by picking the **first** $1,024$ tokens in the sequence, truncating or dropping needles at deeper positions (causing the `" 7."` and `" 7499."` failures in [`results/exp21_translation_layer/metrics.json`](../results/exp21_translation_layer/metrics.json)).
2. **Unused `PreCacheTranslationLayer` in `forward()`:**
   - `self.translation` (`PreCacheTranslationLayer`) is initialized on line 107 of `CognitiveDualControlAttention`, but `forward()` never calls `self.translation.translate_k()` or `translate_q()`.
3. **Prefill vs. Decode Cache Transition:**
   - In `forward()`, prefill (`T > 1`) executes full attention over all $T$ tokens (so token #1 of `.generate()` is predicted with full $O(T)$ attention), while token #2 switches to `[salient_k_cache, local_k_cache]`. If `salient_k_cache` misses any needle digit due to binary top-$k$ tie-breaking, generation halts after the first digit (`" 7."`).

### 1.2 Exact Code Upgrade: Tensor-Geometric & De-Rotated Query-Guided Salience Diffusion (`CDCA-V3`)
Update [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py) to compute token salience **100% from internal un-rotated tensors (`raw_q`, `raw_k`) inside `forward()`**, requiring zero hardcoded keywords and zero tokenizer string decoding:

1. **Bottom-Up Geometric Surprise & Structural Novelty ($s_\tau^{\text{bu}}$):**
   On un-rotated keys $\boldsymbol{k}_\tau^{\text{raw}} \in \mathbb{R}^{H_{\text{KV}} \times T_{\text{hist}} \times d}$:
   - Compute L2 key norm z-score across historical tokens:
     $$z_{\text{norm}}(\tau) = \frac{\|\boldsymbol{k}_\tau^{\text{raw}}\|_2 - \mu_{\text{norm}}}{\sigma_{\text{norm}} + \epsilon}$$
   - Compute local angular novelty (cosine distance from preceding token):
     $$\nu(\tau) = 1 - \cos\left(\boldsymbol{k}_\tau^{\text{raw}}, \boldsymbol{k}_{\tau-1}^{\text{raw}}\right), \qquad z_{\text{nov}}(\tau) = \frac{\nu(\tau) - \mu_\nu}{\sigma_\nu + \epsilon}$$
   - Bottom-up surprise score: $s_\tau^{\text{bu}} = \text{ReLU}(z_{\text{norm}}(\tau)) + \text{ReLU}(z_{\text{nov}}(\tau))$.

2. **Top-Down De-Rotated Observation Window Alignment ($s_\tau^{\text{td}}$):**
   Following the observation-window principle (cf. SnapKV, Li et al., 2024), but operating in the **de-rotated stationary semantic space** ($\tilde{\boldsymbol{q}}_t = \text{translate\_q}(\boldsymbol{q}_t^{\text{raw}})$, $\tilde{\boldsymbol{k}}_\tau = \text{translate\_k}(\boldsymbol{k}_\tau^{\text{raw}})$):
   - Take the last $W_{\text{obs}} = \min(64, T)$ query vectors at the end of the prompt $\tilde{\boldsymbol{Q}}_{\text{obs}} \in \mathbb{R}^{B \times H_Q \times W_{\text{obs}} \times d}$.
   - Compute the maximum de-rotated alignment score between the observation window queries and every historical key $\tau \in [1, T - W]$:
     $$a(\tau) = \max_{h, \, t \in \text{obs}} \left( \frac{\tilde{\boldsymbol{q}}_{h, t}^\top \tilde{\boldsymbol{k}}_{h_{\text{kv}}, \tau}}{\sqrt{d}} \right), \qquad s_\tau^{\text{td}} = \frac{a(\tau) - \mu_a}{\sigma_a + \epsilon}$$

3. **Continuous Bidirectional Salience Diffusion ($\omega_\tau$):**
   Combine bottom-up and top-down salience into a continuous signal $s_\tau = s_\tau^{\text{td}} + 0.3 \, s_\tau^{\text{bu}}$, and convolve bidirectionally with the 1D exponential diffusion kernel of radius $K = 16$:
   $$\omega_\tau = \left( s * \kappa \right)_\tau, \qquad \kappa(\delta) = \exp(-0.10 |\delta|), \quad \delta \in [-16, +16]$$
   - Because $\omega_\tau$ is **strictly continuous** (no binary `1.0` plateaus), `torch.topk(omega, k=max_salient).indices.sort().values` has **zero tie-breaking truncation**, preserving the entire clause surrounding both proactive and retroactive needles at any depth ($10\%, 50\%, 90\%$) up to $T = 32,768+$!

4. **Consistent Prefill & Decode Readout + Optional De-Rotated OLS Bulk Residual:**
   - Ensure that during prefill (`T > W`), `salient_k_cache` and `salient_v_cache` are populated automatically inside `forward()` using the continuous tensor salience indices (no external `set_salient_landmark_indices` call required, though keep `set_salient_landmark_indices` as a no-op or wrapper for backward compatibility with `EXP-21` scripts).
   - Also add two clean baseline modes to `CognitiveDualControlAttention` (via a `mode` argument: `"cdca_v3"`, `"streaming_llm"`, `"h2o"`) so all baselines can be benchmarked with the exact same harness:
     - `mode="streaming_llm"`: Retains the first $N_{\text{sink}} = 4$ tokens + local window $W = 1024$.
     - `mode="h2o"`: Retains the top $N_{\text{salient}} = 1024$ historical tokens ranked by cumulative attention from a causal sample + local window $W = 1024$.
     - `mode="cdca_v3"`: Retains the top $N_{\text{salient}} = 1024$ historical tokens ranked by De-Rotated Query + Geometric Surprise Bidirectional Diffusion + local window $W = 1024$.

---

## Phase 2: Implement & Execute `EXP-26-CDCA-V3-ABLATION` on Modal GPU

### 2.1 Files to Create
1. **Benchmark Harness:** `experiments/run_cdca_v3_ablation.py`
2. **Modal Cloud Runner:** `experiments/modal_cdca_v3_runner.py` (with `gpu="A10G"`, `timeout=300`)
3. **Output Directory:** `results/exp26_cdca_v3_ablation/`
   - `results/exp26_cdca_v3_ablation/metrics.json`
   - `results/exp26_cdca_v3_ablation/fig70_cdca_v3_ablation_niah.png`
   - `results/exp26_cdca_v3_ablation/fig71_multi_needle_and_uuid_recall.png`
   - `results/exp26_cdca_v3_ablation/fig72_memory_and_ppl_scaling.png`

### 2.2 Sub-Experiments to Run Inside `EXP-26` (Target Runtime: ~150s on A10G)
1. **Full 18-Trial NIAH Grid ($T \in \{1024, 2048, 4096, 8192, 16384, 32768\}$, Depths $\{10\%, 50\%, 90\%\}$) across 4 Methods:**
   - **Original Softmax** (from `results/exp18_qwen_full_validation/metrics.json`: $18/18 = 100.0\%$, $393.2\text{ MB}$ cache)
   - **StreamingLLM ($W=1024 + 4$ sinks):** Run live on $T \in \{1024 \dots 32768\}$. Expected: $100\%$ at $T=1024$, $0\%$ whenever needle is outside the last $1024$ tokens ($4/18 = 22.2\%$).
   - **H2O / Cumulative Attention Eviction ($W=1024 + N=1024$):** Run live. Show how position bias / pre-query accumulation degrades recall at $16\text{k}$–$32\text{k}$.
   - **CDCA-V3 (Zero-Keyword Tensor Salience + De-Rotation + Bidirectional Diffusion, $W=1024 + N=1024$):** Run live across all 18 trials + the $T=32,768$ retroactive framing trial. Verify it achieves **$18/18$ ($100.0\%$)** without any keyword list!
2. **Out-of-Distribution & Multi-Needle Stress Tests ($T = 8,192$ and $T = 16,384$):**
   - **Arbitrary Alphanumeric / UUID Needle Test:** Insert a novel unseen code identifier (`"X9F-482-Q7B"`) that never appeared in any keyword list. Verify $100\%$ exact retrieval.
   - **Multi-Needle Synthesis Test:** Insert **3 distinct needles** at Depths $15\%$, $50\%$, and $85\%$ in the same prompt (e.g., Alpha code `9182`, Beta code `3746`, Gamma code `5501`) and query all three.
3. **Expanded Multi-Domain Reasoning & Long-Context Perplexity:**
   - Evaluate MMLU/ARC science benchmark on an expanded 25-question evaluation set (so accuracy has $4\%$ granularity instead of 5-question $20\%$ steps) and measure long-context language modeling perplexity at $T = 4,096$ and $T = 8,192$.
4. **Update `EXPERIMENTS.md`:**
   - Add `EXP-26-CDCA-V3` to the summary table at the top of [`EXPERIMENTS.md`](../EXPERIMENTS.md) and append a complete detailed section at the bottom with exact numbers from `results/exp26_cdca_v3_ablation/metrics.json`.

---

## Phase 3: Expand `paper/references.bib` with Missing Related Work

Append the following BibTeX citations to [`paper/references.bib`](references.bib) so Manuscript 2 is properly grounded in the 2023–2026 literature:

```bibtex
@inproceedings{xiao2024streamingllm,
  title={Efficient Streaming Language Models with Attention Sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@inproceedings{zhang2023h2o,
  title={H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models},
  author={Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and Chen, Tianlong and Zheng, Lianmin and Cai, Ruisi and Song, Zhao and Tian, Yuandong and R{\'e}, Christopher and Barrett, Clark and Wang, Zhangyang and Chen, Beidi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023}
}

@inproceedings{li2024snapkv,
  title={SnapKV: LLM Knows What You Are Looking For Before Generation},
  author={Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and Chen, Deming},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={37},
  year={2024}
}

@inproceedings{zhang2025lolcats,
  title={LoLCATs: On Low-Rank Linearizing of Large Language Models},
  author={Zhang, Michael and Arora, Simran and Chalamala, Rahul and Wu, Alan and Spector, Benjamin and Eyuboglu, Sabri and R{\'e}, Christopher},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{mercat2024linearizing,
  title={Linearizing Large Language Models},
  author={Mercat, Jean and Vasiljevic, Igor and Keh, Sedrick and Arora, Kushal and Dave, Achal and Gaidon, Adrien and Kollar, Thomas},
  journal={arXiv preprint arXiv:2405.06640},
  year={2024}
}

@inproceedings{bick2024transformers,
  title={Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models},
  author={Bick, Aviv and Li, Kevin Y. and Xing, Eric P. and Kolter, J. Zico and Gu, Albert},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={37},
  year={2024}
}

@article{deepseek2024v2,
  title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
  author={{DeepSeek-AI}},
  journal={arXiv preprint arXiv:2405.04434},
  year={2024}
}

@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2025}
}

@article{sun2024learning,
  title={Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
  author={Sun, Yu and Li, Xinhao and Dalal, Karan and Xu, Jiarui and Vikram, Arjun and Zhang, Genghan and Dubois, Yann and Chen, Xinlei and Wang, Xiaolong and Koyejo, Sanmi and Hashimoto, Tatsunori and Guestrin, Carlos},
  journal={arXiv preprint arXiv:2407.04620},
  year={2024}
}

@inproceedings{yang2024gated,
  title={Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author={Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## Phase 4: Comprehensive Rewrite Blueprint for `paper/manuscript_2_foundation_model_adaptation.md`

Rewrite [`paper/manuscript_2_foundation_model_adaptation.md`](manuscript_2_foundation_model_adaptation.md) from its current short 186-line summary into a complete, publication-grade 8-section manuscript (~450–600 lines) that integrates **all Qwen adaptation experiments (`EXP-15` through `EXP-21`, `EXP-23`, and `EXP-26`)**:

### Section-by-Section Outline & Required Fixes

1. **Title & Abstract:**
   - Update metrics to include:
     - Direct Functional Operator Estimation ($R^2 = 0.9505$ mean across 6 layers, $R^2 = 0.9932$ at Layer 23 in $21.95\text{s}$, $40.0\%$ HumanEval pass@1).
     - The Rank-$r$ Spectral Dilution Horizon and Representational Shock Theorem.
     - CDCA-V2/V3 memory compression ($16.0\times$ to $20.9\times$ reduction at $32\text{k}$ context) and updated NIAH retrieval rate from `EXP-26`.

2. **Section 1: Introduction — The Hybrid Foundation Model Serving Dilemma**
   - **CRITICAL MATH FIX in Section 1.1:**
     Replace the buggy equation (`6 * 2 * 2 * T * 256 * 2 = 24,576 * T`) with the exact arithmetic:
     $$\text{Memory}_{\text{KV}}(T) = L_{\text{full}} \times 2_{\text{(K,V)}} \times H_{\text{KV}} \times T \times d_{\text{head}} \times 2\text{ bytes} = 6 \times 2 \times 2 \times T \times 256 \times 2 = \mathbf{12,288 \times T\text{ bytes}} \text{ ($12.0\text{ KiB/token}$)}$$
     At $T = 32,768$ tokens:
     $$\text{Memory}_{\text{KV}}(32\text{k}) = 12,288 \times 32,768 = 402,653,184\text{ bytes} = 393,216\text{ KiB} = \mathbf{384.0\text{ MiB}} \text{ ($402.7\text{ MB}$)}$$
     And at Qwen3.5's native $T = 262,144$ context window:
     $$\text{Memory}_{\text{KV}}(262\text{k}) = 12,288 \times 262,144 = 3,221,225,472\text{ bytes} = \mathbf{3.00\text{ GiB}} \text{ ($3.22\text{ GB}$ per single sequence)}$$
   - Explain why `Qwen/Qwen3.5-0.8B-Base` is an ideal testbed: 18 of its 24 layers are *already* linear Gated DeltaNet (GDN) layers, leaving the **6 Full-Attention layers (Layers 3, 7, 11, 15, 19, 23)** as the sole quadratic compute and linear memory bottleneck.

3. **Section 2: Related Work (Contextualizing Across 4 Research Streams)**
   - **2.1 Post-Training Attention Linearization & Distillation:** Contrast SUPRA (Mercat et al., 2024), MOHAWK (Bick et al., 2024), and LoLCATs (Zhang et al., 2025) against our closed-form OLS Direct Operator Estimation.
   - **2.2 Training-Free KV-Cache Eviction:** Contrast StreamingLLM (Xiao et al., 2024), H2O (Zhang et al., 2023), and SnapKV (Li et al., 2024) against our Bidirectional Salience Diffusion.
   - **2.3 RoPE Interference in Compressed Latent Spaces:** Connect DeepSeek-V2 Multi-Head Latent Attention (MLA) (DeepSeek-AI, 2024) to our De-Rotated Pre-Cache Translation Layer.
   - **2.4 Test-Time Regression & Dual Memory:** Contrast Titans (Behrouz et al., 2025) and TTT (Sun et al., 2024) against closed-form second-order Gramian inversion on frozen models.

4. **Section 3: Direct Functional Operator Estimation vs. Token Distillation (`EXP-15`, `EXP-16`, `EXP-17`, `EXP-18`, `EXP-23`)**
   - **3.1 The Representational Shock Theorem (`EXP-23`):** Show why hard-swapping unaligned NSAL layers drops MMLU to $10.0\%$ and stalls at $4.65$ loss (`results/exp23_qwen_nsal_alignment/metrics.json`).
   - **3.2 Corpus-Induced Domain Drift in Token Distillation (`EXP-15` & `EXP-16`):** Include exact metrics from `results/exp15_qwen_ols_linearization/metrics.json` and `results/exp16_qwen_benchmark/metrics.json` ($\text{PPL } 8.39 \to 9.69$ in $128.25\text{s}$, $204.0\text{ KB}$ state, $892.9\text{ tok/s}$ decode at $8\text{k}$ vs $34.4\text{ tok/s}$ Softmax), and illustrate the TinyStories vocabulary drift on scientific prompts.
   - **3.3 Direct Functional Operator Estimation (`EXP-17` & `EXP-18`):** Formulate the operator regression objective:
     $$\hat{F}_{\text{OLS}}(\boldsymbol{q}_t, \boldsymbol{K}_{\le t}, \boldsymbol{V}_{\le t}; \boldsymbol{U}_k, \boldsymbol{U}_q) = \boldsymbol{\phi}_q(\boldsymbol{q}_t)^\top \left( \sum_{\tau=1}^t \gamma^{t-\tau} \boldsymbol{\phi}_k(\boldsymbol{k}_\tau) \boldsymbol{\phi}_k(\boldsymbol{k}_\tau)^\top + \lambda \boldsymbol{I}_r \right)^{-1} \left( \sum_{\tau=1}^t \gamma^{t-\tau} \boldsymbol{\phi}_k(\boldsymbol{k}_\tau) \boldsymbol{v}_\tau^\top \right)$$
     Include the domain fidelity table from `EXP-17` (Code $R^2 = 0.9443$, Science $R^2 = 0.9619$, Math $R^2 = 0.9648$ in $8.34\text{s}$) and the 6-layer fit table from `EXP-18` (Layers 3, 7, 11, 15, 19, 23 averaging $R^2 = 0.9505$, Layer 23 reaching $R^2 = 0.9932$ in $21.95\text{s}$, and $40.0\%$ HumanEval pass rate). Embed Figures 40–45 using valid relative paths (`../results/exp16_qwen_benchmark/...`, `../results/exp17_direct_operator_estimation/...`, `../results/exp18_qwen_full_validation/...`).

5. **Section 4: Theoretical Limits of Fixed-Rank Recurrence (`EXP-18`, `EXP-19`, `EXP-21`)**
   - **4.1 The Rank-$r$ Spectral Dilution Horizon (`EXP-18` & `EXP-19`):** Present the Subspace Rank Scaling table ($r \in \{8, 16, 32, 64\}$ from `results/exp19_rank_scaling/metrics.json`: state memory $51.0\text{ KB} \to 576.0\text{ KB}$, decode FLOPs $52.2 \to 589.8\text{ KFLOPs}$) and prove why a single needle token at $T=32,768$ suffers $\frac{1}{T} \approx 3 \times 10^{-5}$ eigenvalue dilution in a fixed $r \times r$ Gram matrix.
   - **4.2 Rotary Phase Cancellation & The De-Rotation Theorem (`EXP-21`):** Prove mathematically why $\sum_\tau \boldsymbol{R}_{\theta, \tau} \boldsymbol{k} \boldsymbol{k}^\top \boldsymbol{R}_{\theta, \tau}^\top$ smears covariance across frequencies, and show that stripping RoPE before operator projection jumps Layer 3 validation $R^2$ from $0.4014 \to 0.9893$.

6. **Section 5: Cognitive Dual-Control Memory Architecture (`CDCA-V2` & `CDCA-V3`, `EXP-20`, `EXP-21`, `EXP-26`)**
   - Detail the three-component surgical architecture:
     1. **STM Local Sliding Window ($W=1024$)** for exact local syntax and $100\%$ MMLU preservation.
     2. **Tensor-Geometric & De-Rotated Query-Guided Salience Gate with Bidirectional Diffusion ($\kappa = 16$)** for proactive and retroactive clause retention without lexical heuristics.
     3. **De-Rotated Flash-OLS Bulk Operator State** for $O(1)$ background context summarization.

7. **Section 6: Comprehensive Empirical Validation (`EXP-20`, `EXP-21`, `EXP-26`)**
   - **FIX BROKEN PATHS:** Update all figure and JSON references from the non-existent `results/exp20_qwen_benchmark/` to [`../results/exp20_cognitive_dual_control/`](../results/exp20_cognitive_dual_control/), [`../results/exp21_translation_layer/`](../results/exp21_translation_layer/), and [`../results/exp26_cdca_v3_ablation/`](../results/exp26_cdca_v3_ablation/).
   - Present the complete multi-baseline comparison table (Original Softmax vs. Pure Flash-OLS vs. Hard NSAL Swap vs. StreamingLLM vs. H2O vs. CDCA-V1 vs. CDCA-V2 vs. CDCA-V3) across:
     - Memory footprint at $32\text{k}$ and $262\text{k}$
     - 18-trial NIAH accuracy ($1\text{k} \to 32\text{k}$, depths $10\%, 50\%, 90\%$)
     - Retroactive alert framing at $32\text{k}$
     - Multi-needle & unseen UUID retrieval (`EXP-26`)
     - MMLU/ARC accuracy & HumanEval coding

8. **Section 7: Practical Deployment & Discussion**
   - Show the clean, updated zero-retraining HuggingFace wrapper snippet and discuss hardware concurrency implications on 24GB GPUs.

---

## Execution Checklist for Agent

- [ ] **Step 1:** Edit [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py) to implement `CDCA-V3` (continuous tensor-geometric + de-rotated query-guided bidirectional salience diffusion, plus `"streaming_llm"` and `"h2o"` baseline modes, removing hardcoded target strings).
- [ ] **Step 2:** Create `experiments/run_cdca_v3_ablation.py` and `experiments/modal_cdca_v3_runner.py` (`timeout=300`, `gpu="A10G"`).
- [ ] **Step 3:** Run `modal run experiments/modal_cdca_v3_runner.py`, verify container spin-down via `modal app list`, and confirm `results/exp26_cdca_v3_ablation/metrics.json` and plots `fig70`–`fig72` are saved.
- [ ] **Step 4:** Log `EXP-26` in [`EXPERIMENTS.md`](../EXPERIMENTS.md) with exact numbers from `metrics.json`.
- [ ] **Step 5:** Append the 10 new BibTeX references to [`paper/references.bib`](references.bib).
- [ ] **Step 6:** Rewrite [`paper/manuscript_2_foundation_model_adaptation.md`](manuscript_2_foundation_model_adaptation.md) following the Phase 4 blueprint, verifying all file links and exact JSON metrics.
