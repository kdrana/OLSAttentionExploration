# Scientific Writing Guidelines for High-Tier Publications
*The Craft of Scholarly Prose in the Post-LLM Era: From Data Dumps to Conceptual Mastery*

---

## 1. Executive Philosophy: Tables for Data, Prose for Meaning

Modern AI-assisted academic writing has developed a recognizable, numbing pathology: **it reads like an automated telemetry log**. Running text is suffocated by decimals, confidence intervals, sample counts, and statistical boilerplate, while every sentence strains to assert superiority using formulaic adjectives of triumph (*"decisive"*, *"flawless"*, *"surges"*, *"unmatched selectivity"*).

In contrast, the classic literature of computer science, mathematics, and natural science (e.g., Shannon 1948, Vaswani et al. 2017, He et al. 2016 ResNet, Radford et al. 2019, Breiman 2001) is distinguished by **intellectual curiosity, conceptual clarity, modest understatement, and narrative rhythm**.

> ### The Golden Rule of Scholarly Prose
> **A reader consults a table or figure to find a number. A reader reads prose to understand what that number *means*.**
>
> When running text merely recites table rows in grammatical form (`Method A achieved 56.67% ± 18.04% while Method B achieved 100.0% ± 0.0%`), the prose ceases to be an argument. It becomes an unindexed CSV.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE SCHOLARLY DIVISION OF LABOR                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • The Table's Job:   Exhaustive, auditable, high-precision quantification (mean, std, 95% CI).   │
│ • The Figure's Job:  Immediate visual geometry (phase boundaries, curves, trade-off envelopes). │
│ • The Prose's Job:   The physical phenomenon, the underlying mechanism, and the failure mode.   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Five Pathologies of LLM-Generated Academic Writing

Before learning what to write, authors must recognize and unlearn the reflexive habits of modern LLM-assisted drafting:

| Pathology | Description | How It Sounds |
| :--- | :--- | :--- |
| **1. The CSV Recitation** | Re-stating every number and error bar from the table directly into prose sentences. | *"Method X achieved 76.67% ± 8.79% (n=90), whereas Method Y achieved 5.56% ± 4.76% (n=90), an improvement of 71.11%..."* |
| **2. Adjectives of Triumph** | Using hyperbolic, promotional adjectives to manufacture importance. | *"HALO achieves a flawless 100.0% recall, decisively crushing the baseline with unmatched selectivity..."* |
| **3. The Checklist Mentality** | Structuring the narrative around sub-experiment task lists rather than scientific questions. | *"To evaluate our model, Sub-Experiment 28A was conducted across 3 models. Next, Sub-Experiment 28B was executed..."* |
| **4. Acronym Clutter** | Inventing an acronym for every sub-component and stacking them consecutively. | *"The PRISM-CDCA-NSAL framework utilizes STM and LTM states via OLS to outperform RLS on MQAR..."* |
| **5. Metronomic Rhythm** | Every sentence has the exact same length (22–28 words) and formulaic syntax (*"By doing X, Y achieves Z. Furthermore, W demonstrates..."*). | Lack of breath, absence of punchy observations, monotonous corporate cadence. |

---

## 3. Five Core Principles of Elite Scientific Prose

### Principle 1: Treat Numbers as Landmarks, Not Pavement
* **Rule:** Aim for **no more than two concrete figures per paragraph** in the main text. 
* Point the reader to the table for exhaustive quantification: `(Table 3)`.
* In the prose, describe the **behavioral shift**: *did the model truncate the string? Did it match distractor keywords? Did it suffer catastrophic forgetting, or did it maintain syntactic coherence?*

### Principle 2: Cultivate Calm Understatement
* High-tier reviewers (NeurIPS, ICLR, ICML, Nature MI) are instinctively skeptical of self-congratulation. 
* Never tell the reviewer that your result is "surprising", "revolutionary", "flawless", or "decisive". Present the evidence with calm, objective neutrality and let the reader draw the conclusion.
* A modest tone projects confidence; hyperbolic language signals defensiveness.

| Avoid (Promotional AI Slang) | Prefer (Scholarly Understatement) |
| :--- | :--- |
| *"surges validation fidelity to $R^2 = 0.9893$..."* | *"recovers over $98\%$ of attention variance ($R^2 = 0.9893$)..."* |
| *"HALO achieves a flawless $100.0\%$ recall..."* | *"HALO preserves full retrieval accuracy across all trials..."* |
| *"SnapKV collapses completely to $0.0\%$..."* | *"SnapKV fails entirely outside the sliding window..."* |
| *"decisive leave-one-out component isolation..."* | *"ablation of individual components..."* |
| *"proves our revolutionary hypothesis..."* | *"confirms the predicted geometric behavior..."* |

### Principle 3: Frame Investigations Around Physical Puzzles
* Science is not a feature demonstration; it is an investigation into why systems behave as they do.
* Frame sections around a **puzzle, contradiction, or boundary condition**:
  * *Why does a model that captures 95% of output variance fail on single-token recall?*
  * *Why does query attention peak on the question words instead of the answer?*
  * *Under what grammatical distance does local pooling break down?*

### Principle 4: Use Spatial, Mechanical Verbs
* Replace passive bureaucratic verbs (*"was benchmarked"*, *"was evaluated"*, *"resulted in"*, *"exhibited performance"*) with concrete mechanical actions:
  * *diffuses, severs, tethers, truncates, smears, cancels, projects, decays, outcompetes, clusters, saturates*.

### Principle 5: Master Sentence Cadence and Paragraph Architecture
* Vary sentence lengths deliberately:
  * Open with a **bold, clear thematic sentence** (5–12 words).
  * Follow with a **nuanced, multi-clause mechanical explanation** (25–35 words).
  * Close with a **short, definitive takeaway** (8–15 words).
* If a paragraph cannot be read comfortably aloud in a single breath, it is poorly constructed.

---

## 4. Deep Case Studies & Side-by-Side Transformations

Below are five concrete case studies drawn directly from the manuscript on compressed Transformer attention ([`paper/manuscript_2_foundation_model_adaptation.md`](file:///home/kdrana/Projects/OLSAttentionExploration/paper/manuscript_2_foundation_model_adaptation.md)).

---

### Case Study 1: The Abstract

#### ❌ The Common LLM Approach (Dense Telemetry Log)
> Serving large language models across long context horizons is critically bottlenecked by the Key-Value (KV) cache, whose linear footprint in sequence length $T$ exhausts accelerator high-bandwidth memory (HBM) and restricts serving throughput. Existing post-training compression strategies divide into two competing paradigms: compiling historical tokens into fixed-size recurrent matrix states, or selectively evicting tokens from the KV cache. In this work, we demonstrate the fundamental geometric failure modes of both approaches and introduce a principled mechanism that resolves them without fine-tuning. First, we prove the Rank-$r$ Spectral Dilution Horizon: any fixed-size recurrent state acts as a low-pass spectral filter whose energy allocation to an individual token decays as $\mathcal{O}(1/T)$, causing exact long-range recall to collapse to $0.0\%$ beyond the local window despite capturing over $95\%$ of attention output variance ($R^2 = 0.9505$) and preserving $100\%$ general reasoning accuracy. Second, we uncover the Query-Answer Asymmetry Principle: existing selective token eviction schemes fail on complex entities because causal prefix scoring is blind to terminal prompt queries, while greedy top-$k$ selection retains query-matching cue words while severing the adjacent multi-token payload. In response, we introduce HALO... Evaluated across three distinct foundation model families... HALO achieves $100.0\%$ single-needle, $90.0\%$ adversarial distractor, and $100.0\%$ multi-hop variable tracking recall while compressing KV-cache memory by $16\times$ to $25.6\times$ with sub-60ms selection overhead.

####  The Scholarly Human Approach (Clear, Essayistic, Compelling)
> Serving large language models across long horizons is fundamentally bottlenecked by the Key-Value (KV) cache, whose memory footprint scales linearly with context length. Existing post-training compression techniques face a sharp dilemma: fixed-size recurrent states compress context aggressively but wash out isolated facts, while token-eviction heuristics frequently discard essential details.
>
> In this work, we analyze the geometric limits of both paradigms and propose a simple, training-free resolution. First, we show that fixed-size recurrent states act as low-pass spectral filters: an individual token's contribution to the state Gramian decays as $\mathcal{O}(1/T)$, causing exact long-range recall to vanish even when global reasoning is well preserved. Second, we identify the *Query-Answer Asymmetry* of selective eviction: because prompt questions match semantic cues rather than unknown answers, point-wise scoring retains query keywords while evicting the adjacent factual payload.
>
> To bridge this gap, we introduce **HALO** (Hybrid Attention with Landmark Observation-diffusion). HALO pairs an exact sliding window with continuous spatial diffusion, convolving observation-window salience across adjacent tokens to preserve complete factual clauses intact. Across three independent model families (Qwen, Llama, and SmolLM) and the RULER benchmark suite, HALO preserves exact multi-token recall and distractor robustness while reducing the KV-cache footprint by up to $25\times$ with negligible latency overhead.

---

### Case Study 2: Results & Comparative Analysis

#### ❌ The Common LLM Approach (Walking Through the Table Line-by-Line)
> On HuggingFace SmolLM2-1.7B, HALO decisively outperforms SnapKV on composite recall ($60.00\% \pm 10.18\%$ vs. $48.89\% \pm 10.39\%$, an $+11.11\%$ gain). Crucially, on multi-token alphanumeric UUID needles, HALO achieves more than double the retention of SnapKV ($56.67\% \pm 18.04\%$ vs. $26.67\% \pm 16.10\%$, a $2.12\times$ improvement). Both methods decisively outperform causal eviction and streaming sinks ($5.56\%$–$6.67\%$). Across all three foundation models, HALO preserves full recall without requiring any architectural retraining or parameter updates.

####  The Scholarly Human Approach (Explaining the Behavioral Divergence)
> While both methods maintain basic conversational fluency, their behaviors diverge sharply on multi-token entities. Because the individual characters of an alphanumeric serial code carry little intrinsic semantic weight, point-wise attention scoring leaves trailing segments below the eviction threshold. SnapKV consequently truncates composite identifiers after the first hyphen.
>
> By diffusing salience across the enclosing clause, HALO preserves the entity intact, recovering more than twice as many complete identifiers on SmolLM2 without retraining (Table 3). Both observation-driven approaches decisively outperform causal prefix eviction, which remains blind to terminal prompt queries.

---

### Case Study 3: Framing an Experimental Investigation

#### ❌ The Common LLM Approach (The Sub-Experiment Telemetry Report)
> To establish the exact boundary where narrow observation pooling collapses, Sub-Experiment 28B evaluated $n=10$ randomized trials per point across two diagnostic sweeps on Qwen3.5-0.8B ($T = 4,096, W = 512, N = 128$): 
> 1. Syntactic Cue-to-Value Separation ($\Delta_{\text{QA}} \in \{1, 3, 6, 10, 15, 22\}$ tokens): At $\Delta = 1$: Isolated Top-k = $80.0\% \pm 26.13\%$, SnapKV = $100.0\%$, H2O = $0.0\%$, HALO = $100.0\%$. At $\Delta = 3$: Isolated Top-k drops sharply to $40.0\% \pm 32.01\%$, while SnapKV and HALO maintain $100.0\%$. At $\Delta = 6$: Isolated Top-k collapses to $20.0\% \pm 26.13\%$...

####  The Scholarly Human Approach (Posing the Linguistic Question)
> Under what syntactic conditions does narrow observation pooling fail? Intuitively, selection breaks down whenever sentence structure separates a queried attribute from its corresponding value. In natural prose, qualifying clauses and auxiliary verbs frequently intervene (for example, *"The clearance code, once verified by senior inspection, was 8472"*).
>
> To map this boundary, we systematically varied the grammatical distance $\Delta_{\text{QA}}$ between cue and answer. As shown in Figure 2 (Left), point-wise top-$k$ selection degrades rapidly as filler tokens are introduced, falling to $20\%$ recall at six intervening words. Convolving attention mass over a sixteen-token radius bridges this gap: HALO consistently tethers the target value to its queried cue across all tested syntactic distances.

---

### Case Study 4: Theoretical Intuition & Synthesis

#### ❌ The Common LLM Approach (Over-Formalized Yet Obscure)
> Proposition 1 proves that in frozen deep Transformers, downstream activations drift when replacing an attention layer with an approximating operator due to Lipschitz compounding. Furthermore, Theorem 3 proves that unweighted recurrent Gramians dilute needle spectral contributions as $O(1/T)$, so while arithmetic precision is maintained, Rayleigh quotients drop below detection floors, resulting in 0% recall.

####  The Scholarly Human Approach (Vivid, Geometric, Precise)
> Fixed-rank recurrence acts as a low-pass spectral filter. It accurately captures the dominant principal components of the context—preserving global topic coherence, reasoning, and fluent syntax—but washes out isolated, high-frequency facts into the background covariance. 
>
> To retrieve a needle in a haystack, one cannot rely on its blurred reflection in a matrix state. Discrete token representations must be retained.

---

### Case Study 5: Reporting Limitations and Negative Results

#### ❌ The Common LLM Approach (Defensive Hedging or Obfuscation)
> HALO + BLUR achieves top summarization F1 ($55.0\%$), but drops to $0.0\%$ on MK-NIAH due to dense adversarial distractor keys creating cross-talk in the low-rank basis vectors. However, this is expected because linear summaries retain gist rather than needles, confirming our design hypothesis.

####  The Scholarly Human Approach (Direct, Transparent Engineering Honesty)
> While residual block averaging effectively retains background context for summarization (achieving $55.0\%$ F1 on LongBench), it introduces severe cross-talk under adversarial conditions. In the 12-distractor MK-NIAH benchmark, compressing evicted tokens into sixteen summary vectors smears distractor keys across the feature space, reducing recall to $0.0\%$ (Table 2). 
>
> Consequently, pure HALO remains the strictly recommended configuration for fine-grained factual retrieval, while the hybrid variant should be reserved for document summarization.

---

## 5. Practical Author Checklist for Camera-Ready Submissions

Before submitting any manuscript or preprint, run the text through this self-audit:

1. **The Read-Aloud Test:** Read each section aloud. If you stumble, run out of breath, or find yourself reading strings of numbers like a spreadsheet, re-write the paragraph.
2. **The "Two-Number Limit":** Count the numbers in each paragraph of the discussion. If there are more than two, move the rest into the referenced Table or Figure.
3. **The Adjective Red-Pencil:** Search the document for:
   * *flawless, decisive, dramatic, surges, crushes, unmatched, revolutionary, remarkable, superior*.
   * Replace every instance with measured, objective descriptions (*consistent, substantial, higher, preserves, recovers*).
4. **The "Why" Test:** In every results paragraph, check whether you explained *why* a baseline failed (e.g., *"because the narrow window severed trailing characters"*) or merely stated *that* it failed (e.g., *"SnapKV scored 0%"*).
5. **The Acronym Pruning:** Ensure no acronym is introduced unless it appears at least five times across the paper. Prefer descriptive English nouns (*"sliding window"*, *"salience diffusion"*, *"phase de-rotation"*).

---

## 6. Recommended Reading: Masters of Scientific Exposition

To calibrate your ear for natural, authoritative scholarly prose, read these foundational works:

1. **Claude E. Shannon (1948):** *A Mathematical Theory of Communication.* (The gold standard for crystal-clear mathematical exposition without pretense).
2. **Ashish Vaswani et al. (2017):** *Attention Is All You Need.* (Economical, humble, and lucid description of a revolutionary architecture).
3. **Kaiming He et al. (2016):** *Deep Residual Learning for Image Recognition.* (Masterclass in posing an empirical puzzle—the degradation problem—and solving it elegantly).
4. **Leo Breiman (2001):** *Statistical Modeling: The Two Cultures.* (Vivid, opinionated, essayistic prose that engages the reader as an intellectual peer).
5. **Donald E. Knuth et al. (1989):** *Mathematical Writing.* Stanford Report CS-TR-89-1280. (Timeless rules on mathematical prose, punctuation, and rhythm).
