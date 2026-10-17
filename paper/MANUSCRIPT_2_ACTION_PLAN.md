# Action Plan (v2): Resolving the 4 Code-vs-Manuscript Audit Issues in Manuscript 2 & `EXP-26`

**Target Files to Update:**
1. [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py)
2. [`experiments/run_cdca_v3_ablation.py`](../experiments/run_cdca_v3_ablation.py) & [`experiments/modal_cdca_v3_runner.py`](../experiments/modal_cdca_v3_runner.py)
3. [`results/exp26_cdca_v3_ablation/metrics.json`](../results/exp26_cdca_v3_ablation/metrics.json) (and plots `fig70`, `fig71`, `fig72`)
4. [`EXPERIMENTS.md`](../EXPERIMENTS.md)
5. [`paper/manuscript_2_foundation_model_adaptation.md`](manuscript_2_foundation_model_adaptation.md)

> **Instructions for the Executing Agent (Flash 3.8):**  
> Execute Steps 1 through 5 in order. Follow [`AGENTS.md`](../AGENTS.md) strictly:
> - Enforce `timeout=300` on Modal GPU runs and verify container spin-down via `modal app list`.
> - Every single number in [`paper/manuscript_2_foundation_model_adaptation.md`](manuscript_2_foundation_model_adaptation.md) and [`EXPERIMENTS.md`](../EXPERIMENTS.md) must match the corresponding `metrics.json` file with zero unmeasured or fabricated cells.

---

## Step 1: Fix `compute_salience_indices` in `src/cognitive_dual_control_attention.py`

### Why This Fix Is Needed
1. **Bug A (`mode="h2o"` was coded as end-of-prompt query lookahead):** In [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py) (lines 138–150), `mode == "h2o"` used `q_sample = query_states[:, :, -64:, :]` (the final 64 query tokens where the retrieval question sits). That is **SnapKV lookahead**, NOT **$\text{H}_2\text{O}$ (Heavy-Hitter Oracle, Zhang et al., 2023)**. In true $\text{H}_2\text{O}$, historical tokens in `[0, hist_len)` are evicted **causally during prefill** based on cumulative attention from historical prefix queries *before* the end-of-prompt question arrives. Because historical prefix queries attend to attention sinks and local context rather than a needle whose question hasn't been asked yet, true $\text{H}_2\text{O}$ suffers severe position bias and misses un-queried needles.
2. **Bug B (`raw_query_states`, `raw_key_states`, and `self.translation` were unused in `cdca_v3`):** Lines 151–169 ignored `raw_key_states`, `raw_query_states`, and `self.translation` (`PreCacheTranslationLayer`), computing scores only on post-RoPE `query_states` and `key_states` and omitting bottom-up geometric surprise $s_\tau^{\text{bu}}$.

### Exact Replacement Code for `compute_salience_indices` in `src/cognitive_dual_control_attention.py`
Replace `compute_salience_indices` (lines 118–170 of [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py)) with the following implementation:

```python
    def compute_salience_indices(
        self,
        raw_key_states: torch.Tensor,
        raw_query_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        hist_len: int,
    ) -> torch.Tensor:
        """
        Computes salient landmark indices across historical tokens tau in [0, hist_len).
        Requires zero hardcoded keywords and zero string decoding.
        """
        device = raw_key_states.device

        # 1. Mode: StreamingLLM (First 4 attention sinks)
        if self.mode == "streaming_llm":
            return torch.arange(min(4, hist_len), device=device)

        # 2. Mode: True H2O (Causal Prefix Heavy-Hitter Cumulative Attention Eviction)
        # In H2O, historical tokens in [0, hist_len) are scored by cumulative causal
        # attention from queries within the historical prefix (before the final window).
        if self.mode == "h2o":
            with torch.no_grad():
                num_prefix_samples = min(64, hist_len)
                sample_idx = torch.linspace(0, hist_len - 1, steps=num_prefix_samples, device=device).long()
                q_hist_sample = query_states[:, :, sample_idx, :]  # [B, H_q, S, d]
                k_hist = key_states[:, :, :hist_len, :]           # [B, H_kv, hist_len, d]
                k_hist_exp = k_hist.repeat_interleave(self.num_key_value_groups, dim=1)

                scores = torch.einsum("bhqd,bhtd->bhqt", q_hist_sample, k_hist_exp) / math.sqrt(self.head_dim)
                # Causal mask: query at sample_idx[q] can only attend to t <= sample_idx[q]
                t_idx = torch.arange(hist_len, device=device).unsqueeze(0)          # [1, hist_len]
                q_pos = sample_idx.unsqueeze(1)                                     # [S, 1]
                causal_mask = (t_idx <= q_pos).view(1, 1, num_prefix_samples, hist_len)
                scores = scores.masked_fill(~causal_mask, float("-inf"))

                attn = F.softmax(scores.float(), dim=-1)
                cum_attn = attn.sum(dim=(0, 1, 2))  # [hist_len]
                k_select = min(self.max_salient, hist_len)
                return torch.topk(cum_attn, k=k_select).indices.sort().values

        # 3. Mode: CDCA-V3 (De-Rotated Translation + Bottom-Up Geometric Surprise
        #    + Top-Down Observation Window Alignment + Bidirectional Diffusion)
        with torch.no_grad():
            W_sample = min(64, query_states.shape[2])

            # (a) De-Rotated Pre-Cache Translation Layer alignment
            q_raw_obs = self.translation.translate_q(raw_query_states[:, :, -W_sample:, :])
            k_raw_hist = self.translation.translate_k(raw_key_states[:, :, :hist_len, :])
            k_raw_exp = k_raw_hist.repeat_interleave(self.num_key_value_groups, dim=1)

            derot_scores = torch.einsum("bhqd,bhtd->bhqt", q_raw_obs.float(), k_raw_exp.float()) / math.sqrt(self.head_dim)
            derot_attn = F.softmax(derot_scores, dim=-1).sum(dim=(0, 1, 2))  # [hist_len]

            # (b) RoPE Observation-Window Query Attention Mass
            q_rope_obs = query_states[:, :, -W_sample:, :]
            k_rope_hist = key_states[:, :, :hist_len, :].repeat_interleave(self.num_key_value_groups, dim=1)
            rope_scores = torch.einsum("bhqd,bhtd->bhqt", q_rope_obs.float(), k_rope_hist.float()) / math.sqrt(self.head_dim)
            rope_attn = F.softmax(rope_scores, dim=-1).sum(dim=(0, 1, 2))    # [hist_len]

            # Top-Down combined alignment score (normalized)
            s_td = 0.5 * (derot_attn / (derot_attn.max() + 1e-6)) + 0.5 * (rope_attn / (rope_attn.max() + 1e-6))

            # (c) Bottom-Up Geometric Surprise (Key L2 Norm Z-Score + Angular Novelty)
            k_norms = torch.norm(k_raw_hist.float(), p=2, dim=-1).mean(dim=(0, 1))  # [hist_len]
            z_norm = F.relu((k_norms - k_norms.mean()) / (k_norms.std() + 1e-6))

            k_unit = F.normalize(k_raw_hist.float(), p=2, dim=-1)
            cos_adj = (k_unit[:, :, 1:, :] * k_unit[:, :, :-1, :]).sum(dim=-1).mean(dim=(0, 1))
            novelty = torch.cat([torch.zeros(1, device=device), 1.0 - cos_adj], dim=0)
            z_nov = F.relu((novelty - novelty.mean()) / (novelty.std() + 1e-6))

            s_bu = (z_norm + z_nov)
            s_bu = s_bu / (s_bu.max() + 1e-6)

            # Combined instantaneous salience signal
            s_total = s_td + 0.15 * s_bu  # [hist_len]

            # (d) Continuous Bidirectional 1D Exponential Diffusion (radius K=16)
            K = self.diffusion_radius
            delta = torch.arange(-K, K + 1, device=device, dtype=torch.float32)
            kernel = torch.exp(-0.10 * torch.abs(delta)).view(1, 1, -1)
            kernel = kernel / kernel.max()
            diffused = F.conv1d(s_total.view(1, 1, hist_len), kernel, padding=K).squeeze()  # [hist_len]

            k_select = min(self.max_salient, hist_len)
            return torch.topk(diffused, k=k_select).indices.sort().values
```

---

## Step 2: Fix the Stress Tests & Multi-Needle Evaluation in `experiments/run_cdca_v3_ablation.py`

### Why This Fix Is Needed
1. **Multi-Needle Prompt Bug:** In [`experiments/run_cdca_v3_ablation.py`](../experiments/run_cdca_v3_ablation.py) (line 339), the query was:
   `query_multi = "\nQuestion: What are the codes for Alpha, Beta, and Gamma? Answer: Alpha is 9182, Beta is 3746, Gamma is"`
   This hardcoded the answers for Alpha (`9182`) and Beta (`3746`) directly inside the prompt and only tested Gamma (`5501` at $85\%$ depth).
2. **Missing Baseline Runs on Stress Tests:** `run_cdca_v3_ablation.py` (lines 308–368) only ran `cdca_v3` on the Retroactive 32k, UUID 8k/16k, and Multi-Needle 16k stress tests, never running `streaming_llm` or `h2o` on those tests.

### Exact Modifications to `experiments/run_cdca_v3_ablation.py`
1. **Test All 3 Needles Independently on the Same 16k Multi-Needle Document:**
   In the Multi-Needle Synthesis test at $T = 16,384$ (with `n1` Alpha=`9182` at $15\%$ depth, `n2` Beta=`3746` at $50\%$ depth, and `n3` Gamma=`5501` at $85\%$ depth embedded simultaneously in the same haystack), evaluate **3 separate retrieval queries** (one asking for Alpha, one asking for Beta, one asking for Gamma, without giving away any codes in the query):
   - Query 1: `"\nQuestion: What is the Alpha authorization code? Answer: The Alpha authorization code is"` $\to$ check `"9182"` in output.
   - Query 2: `"\nQuestion: What is the Beta authorization code? Answer: The Beta authorization code is"` $\to$ check `"3746"` in output.
   - Query 3: `"\nQuestion: What is the Gamma authorization code? Answer: The Gamma authorization code is"` $\to$ check `"5501"` in output.
   - Record `alpha_success`, `beta_success`, `gamma_success`, and `needles_recalled` out of 3!
2. **Run All 4 Stress Tests Across All 3 Active Modes (`"cdca_v3"`, `"h2o"`, `"streaming_llm"`):**
   Loop over `mode_name in ["cdca_v3", "h2o", "streaming_llm"]` for:
   - `retroactive_framing_32k`
   - `uuid_needle_8k`
   - `uuid_needle_16k`
   - `multi_needle_16k` (all 3 queries: Alpha at $15\%$, Beta at $50\%$, Gamma at $85\%$)
   Store the results per mode in `metrics["stress_tests"][mode_name]`.
3. **Update Plotting in `experiments/modal_cdca_v3_runner.py` (or `run_cdca_v3_ablation.py`):**
   Regenerate `fig70_cdca_v3_ablation_niah.png`, `fig71_multi_needle_and_uuid_recall.png`, and `fig72_memory_and_ppl_scaling.png` so Figure 70 and Figure 71 visually show the true causal-prefix $\text{H}_2\text{O}$ score vs. `StreamingLLM` vs. `CDCA-V3`!

---

## Step 3: Run `EXP-26` on Modal GPU & Verify Spin-Down

1. Run:
   ```bash
   modal run experiments/modal_cdca_v3_runner.py
   ```
2. Immediately verify spin-down:
   ```bash
   modal app list
   ```
3. Inspect `results/exp26_cdca_v3_ablation/metrics.json` to get the exact new numbers for `streaming_llm`, `h2o`, and `cdca_v3` across the 18-trial NIAH grid and the 4 stress tests.

---

## Step 4: Fix Exact Numeric Tables & Formulas in `paper/manuscript_2_foundation_model_adaptation.md` and `EXPERIMENTS.md`

### 4.1 Fix Table 3 (`EXP-19` Subspace Rank Scaling) in `manuscript_2_foundation_model_adaptation.md` (Lines 187–194)
In [`results/exp19_rank_scaling/metrics.json`](../results/exp19_rank_scaling/metrics.json), `state_kb` (`51.0 KB`, `108.0 KB`, `240.0 KB`, `576.0 KB`) is **already the total 6-layer state memory** (not single-layer memory). Replace Table 3 in [`paper/manuscript_2_foundation_model_adaptation.md`](manuscript_2_foundation_model_adaptation.md) with the exact numbers from [`results/exp19_rank_scaling/metrics.json`](../results/exp19_rank_scaling/metrics.json):

| Subspace Rank ($r$) | Adapter Params | Validation $R^2$ | Cosine Similarity | Total 6-Layer State Memory | Compression vs. 32k Softmax ($384\text{ MiB}$) | Decode FLOPs / Token | Fit Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$r = 8$** | $40,960$ | $0.7080$ | $0.8365$ | **$51.0\text{ KB}$ ($8.5\text{ KB/layer}$)** | **$7,710.1\times$** | $52.2\text{ KFLOPs}$ | $5.59\text{s}$ |
| **$r = 16$** | $81,920$ | $0.4014$ | $0.7579$ | **$108.0\text{ KB}$ ($18.0\text{ KB/layer}$)** | **$3,640.9\times$** | $110.6\text{ KFLOPs}$ | $3.74\text{s}$ |
| **$r = 32$** | $163,840$ | $0.2344$ | $0.6906$ | **$240.0\text{ KB}$ ($40.0\text{ KB/layer}$)** | **$1,638.4\times$** | $245.8\text{ KFLOPs}$ | $4.67\text{s}$ |
| **$r = 64$** | $327,680$ | $0.6846$ | $0.7932$ | **$576.0\text{ KB}$ ($96.0\text{ KB/layer}$)** | **$682.7\times$** | $589.8\text{ KFLOPs}$ | $8.28\text{s}$ |

*(Also add a 1-sentence note explaining why Table 3's validation $R^2$ on Layer 3 was $0.4014$ at $r=16$: because EXP-19 projected post-RoPE keys prior to the discovery of the De-Rotation Theorem in EXP-21, which lifted $r=16$ fidelity from $0.4014 \to 0.9893$!)*

### 4.2 Update Section 5.2 in `manuscript_2_foundation_model_adaptation.md` (Lines 245–265)
Update the mathematical description of **CDCA-V3** in Section 5.2 so it matches the upgraded `compute_salience_indices()` method:
1. **De-Rotated Pre-Cache Translation Layer ($\tilde{\boldsymbol{q}}, \tilde{\boldsymbol{k}}$):**
   $$\tilde{\boldsymbol{k}}_\tau = \text{translate\_k}(\boldsymbol{k}_\tau^{\text{raw}}), \qquad \tilde{\boldsymbol{q}}_t = \text{translate\_q}(\boldsymbol{q}_t^{\text{raw}})$$
2. **Top-Down De-Rotated & Rotary Observation-Window Alignment ($s_\tau^{\text{td}}$):**
   Combines stationary de-rotated semantic alignment $\text{Softmax}(\tilde{\boldsymbol{Q}}_{\text{obs}} \tilde{\boldsymbol{K}}_{\text{hist}}^\top / \sqrt{d})$ with rotary query alignment $\text{Softmax}(\boldsymbol{Q}_{\text{obs}}^{\text{RoPE}} \boldsymbol{K}_{\text{hist}}^{\text{RoPE}\top} / \sqrt{d})$ over the final $W_{\text{obs}} = 64$ observation queries.
3. **Bottom-Up Geometric Surprise ($s_\tau^{\text{bu}}$):**
   Combines the L2 norm $z$-score $z(\|\tilde{\boldsymbol{k}}_\tau\|_2)$ and local angular novelty $z(1 - \cos(\tilde{\boldsymbol{k}}_\tau, \tilde{\boldsymbol{k}}_{\tau-1}))$.
4. **Continuous Bidirectional Exponential Diffusion ($\kappa = 16$):**
   $$\omega_\tau = \left( (s^{\text{td}} + 0.15 \, s^{\text{bu}}) * \kappa \right)_\tau, \qquad \kappa(\delta) = \exp(-0.10 |\delta|), \quad \delta \in [-16, +16]$$

### 4.3 Fix Table 4 in `manuscript_2_foundation_model_adaptation.md` (Lines 289–310)
1. **Hard NSAL Swap (EXP-23) Column:**
   Replace `0.0 MiB` and $\infty$ compression with the exact values from [`results/exp23_qwen_nsal_alignment/metrics.json`](../results/exp23_qwen_nsal_alignment/metrics.json):
   - KV-Cache at $32\text{k}$: **$0.42\text{ MiB}$ ($432.0\text{ KB}$)**
   - KV-Cache at $262\text{k}$: **$0.42\text{ MiB}$ ($432.0\text{ KB}$)**
   - Cache Compression ($32\text{k}$): **$932.0\times$**
   - Cache Compression ($262\text{k}$): **$7,456\times$**
2. **$\text{H}_2\text{O}$ Eviction & `StreamingLLM` Columns:**
   Populate every row of `H2O Eviction` and `StreamingLLM` (both the 18-trial NIAH breakdown and the 3 stress-test rows: Retroactive 32k, Unseen UUID, and 3-Needle Synthesis) with the exact measured numbers from the updated `results/exp26_cdca_v3_ablation/metrics.json`.
3. **Update `EXPERIMENTS.md`:**
   Update the `EXP-26` entry in [`EXPERIMENTS.md`](../EXPERIMENTS.md) so its tables and takeaways match the updated `results/exp26_cdca_v3_ablation/metrics.json` 100%.

---

## Execution Checklist for Agent

- [x] **1.** Update `compute_salience_indices` in [`src/cognitive_dual_control_attention.py`](../src/cognitive_dual_control_attention.py) (True causal prefix `h2o` + De-rotated & bottom-up surprise `cdca_v3`).
- [x] **2.** Update [`experiments/run_cdca_v3_ablation.py`](../experiments/run_cdca_v3_ablation.py) and [`experiments/modal_cdca_v3_runner.py`](../experiments/modal_cdca_v3_runner.py) to test all 3 needles (`9182`, `3746`, `5501`) independently and run all stress tests across `"cdca_v3"`, `"h2o"`, and `"streaming_llm"`.
- [x] **3.** Execute `modal run experiments/modal_cdca_v3_runner.py` (`timeout=300`), verify spin-down with `modal app list`, and regenerate `metrics.json` and `fig70`–`fig72`.
- [x] **4.** Update Table 3, Section 5.2, and Table 4 in [`paper/manuscript_2_foundation_model_adaptation.md`](manuscript_2_foundation_model_adaptation.md) and the `EXP-26` section in [`EXPERIMENTS.md`](../EXPERIMENTS.md) with the exact numbers from `metrics.json`.
