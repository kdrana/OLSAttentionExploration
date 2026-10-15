# Predictive Coding, Kalman Filtering, and Top-Down / Bottom-Up Attention

This document records the conceptual framework bridging cognitive neuroscience (predictive coding, top-down vs. bottom-up processing) with statistical filtering (Kalman/Recursive Least Squares) and modern attention architectures (Linear Attention, State Space Models, and Softmax Transformer memory).

---

## 1. Biological and Cognitive Motivation

Attention in the biological neocortex does not operate as an isotropic, static feedforward pass. Instead, it is organized around a continuous bidirectional dialogue across cortical hierarchies:

1. **Top-Down Signals (Prior Expectations):**
   - Higher-level cortical regions project priors based on past context ($h_{t-1}$).
   - These projections encode the expected state of the world / upcoming token:
     $$\hat{x}_t = \mu(h_{t-1}), \quad \text{with estimated uncertainty covariance } \Sigma_t$$
   - Top-down attention acts as a selective filter, anticipating features before they arrive.

2. **Bottom-Up Signals (Sensory Prediction Errors / Innovations):**
   - Sensory pathways carry predominantly the *prediction error* (the unexplained residual):
     $$\varepsilon_t = x_t - \hat{x}_t$$
   - When incoming tokens confirm expectations, the prediction error is near zero.
   - When incoming patterns break expectations (surprise, rare entities, syntactic shifts), the prediction error spikes.

3. **Attention as Precision Weighting (Friston's Free Energy Principle):**
   - In computational neuroscience (Feldman & Friston, 2010), attention is formally defined as the **precision** (inverse covariance) assigned to bottom-up prediction errors:
     $$\Pi_t = \Sigma_t^{-1}$$
   - Attention determines how aggressively high-level representations must change in response to new sensory evidence.

---

## 2. The "Smooth Check": Bridging Kalman Smoothness and Softmax Retrieval

Standard Transformers apply the same expensive, non-linear $O(T)$ Softmax attention indiscriminately across all tokens. Conversely, pure linear state-space models / Kalman filters struggle with sharp, winner-take-all retrieval.

To smoothly reconcile these two regimes, we introduce a **Surprise-Modulated Transition**:

### Mathematical Formulation
The surprise of an incoming token relative to top-down expectation is measured via its **Mahalanobis distance**:
$$d_M^2(x_t, \hat{x}_t) = (x_t - \hat{x}_t)^\top \Sigma_t^{-1} (x_t - \hat{x}_t)$$

We define a smooth, differentiable gating factor $\alpha_t \in [0, 1]$:
$$\alpha_t = \sigma\left( \frac{d_M^2(x_t, \hat{x}_t) - \tau}{\theta} \right)$$
where $\tau$ is an adaptive surprise threshold and $\theta$ is a temperature parameter.

### Regimes
- **Low Surprise ($d_M^2 \ll \tau \implies \alpha_t \to 0$):**
  - The token is expected (routine grammar, steady semantic flow).
  - Memory updates proceed via the **$O(1)$ linear Kalman / Recursive Least Squares (RLS)** recurrence:
    $$S_t = S_{t-1} + K_t (v_t - S_{t-1}^\top k_t)^\top$$
  - No KV-cache expansion or non-linear Softmax sweep is triggered.

- **High Surprise ($d_M^2 \gg \tau \implies \alpha_t \to 1$):**
  - The token breaks expectation (unexpected clause, rare identifier, critical entity).
  - The gate opens to recruit the **non-linear, high-gain associative memory** (e.g., Softmax attention over episodic KV cache or sparse top-$k$ routing).

---

## 3. Test-Time Recurrent Latent Reasoning (Error Reduction)

When a pattern break occurs, biological systems do not simply advance to the next time step. They pause and iterate, revising high-level representations until the surprise is resolved.

### Energy Minimization in Latent Space
At sequence step $t$, the internal consistency of thought representation $z_t$ is governed by an energy functional:
$$\mathcal{E}(z_t) = \frac{1}{2} \| x_t - g(z_t) \|_{\Sigma_x^{-1}}^2 + \frac{1}{2} \| z_t - f(h_{t-1}) \|_{\Sigma_z^{-1}}^2$$

The network iterates recurrently over internal step index $k$:
$$z_t^{(k+1)} = z_t^{(k)} - \eta \nabla_{z} \mathcal{E}\left(z_t^{(k)}\right)$$

- **Predictable tokens:** Settle in $k=1$ step (instantaneous feedforward output).
- **Surprising tokens:** Iterate across $k=1, \dots, K$ steps, shifting latent embeddings until prediction error falls below an acceptable tolerance before emitting downstream tokens.

---

## 4. Architectural Roadmap for Future Exploration

1. **Surprise-Gated Dual Memory:**
   - Stream 1: $O(d^2)$ rolling Kalman / RLS associative memory (working memory).
   - Stream 2: Sparse Softmax episodic memory, queried only when $\alpha_t > \tau$.
2. **Predictive State Covariance ($\Sigma_t$):**
   - Outputting token-level uncertainty estimates to dynamically adjust Kalman gain.
3. **Adaptive Computation Time (ACT):**
   - Allocating variable internal reasoning steps based on bottom-up innovation magnitude.
