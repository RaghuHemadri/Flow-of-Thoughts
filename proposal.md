

# Flow of Thought 
Given supervised pairs $(x,y)$ where $y$ is **final answer only** (no chain-of-thought), we can learn a **prompt-conditioned latent ODE** $\frac{dz}{dt}=v_\theta(z,t,c)$ that transports a simple prior $z(0)\sim\mathcal N(0,I)$ to an **answer-conditioned latent endpoint** $z^\star=\phi(y,c)$, such that decoding from $z(1)$ yields correct answers at competitive accuracy **without emitting CoT**.

By applying **rectified flow / reflow** to the learned latent dynamics, we can “straighten” trajectories so that **1–4 solver steps** suffice at inference time, producing a tunable **compute-quality frontier** (accuracy vs. steps/latency) rather than a fixed-cost reasoning procedure. Rectified flow is specifically motivated by the high cost of ODE/SDE-based sampling in diffusion-style models. 

With conditional flow matching from $z_0\sim\mathcal N(0,I)$ to $z^\star=\phi(y,c)$, plus **rectified reflow distillation** to enable 1–4-step inference. Uses CoT-free supervision (answer only).

---

## Method specification

### Notation & setup

* Dataset $\mathcal D=\{(x_i,y_i)\}$ where $y$ is the **final answer string** (no rationales).
* Prompt encoder $c=\mathrm{Enc}_\omega(x)$ (could be a frozen LM trunk or trainable encoder).
* Answer-latent encoder $z^\star=\phi_\eta(y,c)\in\mathbb R^{d_z}$.
* Latent dynamics: $\frac{dz}{dt}=v_\theta(z,t,c)$, $t\in[0,1]$, $z(0)=z_0\sim\mathcal N(0,I)$.
* Decoder $p_\psi(y\mid x, z(1))$ that emits **answer only**.

### Endpoint construction $z^\star=\phi_\eta(y,c)$ (crucial)

You want an endpoint that:

1. contains answer semantics, 2) is stable across paraphrases, 3) doesn’t leak CoT.

Concrete choice:

* Use a frozen pretrained LM $(F)$. Run $(F)$ on the concatenation $[x \mid \texttt{<ANS>} \mid y]$.
* Let $h_{\text{eos}}$ be the final-layer hidden state at the EOS token (or pooled answer-token hidden states).
* Set $z^\star = W h_{\text{eos}}$ with a learned projection $W\in\mathbb R^{d_z\times d_h}$.
  This makes $z^\star$ a **representation of the correct answer given the prompt**, without needing intermediate rationales.

### Training objectives

#### (i) Conditional Flow Matching loss (endpoint-only “teacher” path)

Use the simplest linear coupling path (mirrors CFM-style linear interpolation objectives). In Flow Matching, CFM yields objectives of the form in Eq. (23) for an interpolation $\psi_t$. 

For our latent endpoints:

* Sample $t\sim \mathcal U[0,1]$, $z_0\sim\mathcal N(0,I)$
* Define interpolation: $z_t=(1-t)z_0+t z^\star$
* Target velocity: $u = z^\star - z_0$
* Loss:

$$\mathcal L_{\text{CFM}}(\theta)=\mathbb E_{(x,y),z_0,t}\left[|v_\theta(z_t,t,c)-u|_2^2\right]$$

This is the digressing a vector field along an interpolation path, consistent with Flow Matching-style regression objectives. 

#### (ii) Answer decoding loss (forces correctness, discourages degenerate endpoints)

$$\mathcal L_{\text{dec}}(\psi)=\mathbb E_{(x,y),z_0}\left[-\log p_\psi\big(y\mid x, \hat z_1\big)\right]$$

where $\hat z_1$ is obtained by integrating the ODE with a training-time solver (can be few-step Euler).

#### (iii) Contrastive endpoint anchoring (bridges to C2)
Sample wrong answers $y^-$ (other dataset answers, model confusions, or arithmetic near-misses).
Define $z^-=\phi(y^-,c)$. Add:

$$\mathcal L_{\text{NCE}} = -\log \frac{\exp(\mathrm{sim}(\hat z_1,z^\star)/\tau)}{\exp(\mathrm{sim}(\hat z_1,z^\star)/\tau)+\sum_{k}\exp(\mathrm{sim}(\hat z_1,z^-_k)/\tau)}$$

**Total:**

$$\mathcal L = \mathcal L_{\text{CFM}} + \lambda_{\text{dec}} \mathcal L_{\text{dec}} + \lambda_{\text{NCE}}\mathcal L_{\text{NCE}}$$

### Rectified reflow (few-step distillation)

Rectified Flow motivates straightening because diffusion-style generation relies on numerical solvers and can be costly. 
The rectified-flow framework constructs a rectified flow with the same marginals (Theorem 3.3). 

**Adaptation to conditional reasoning:**

* Train $v_\theta^{(0)}$ with $\mathcal L$.
* For reflow iteration $k=1..K_{\text{reflow}}$:

  1. Sample $(x,y)$, compute $c$, using $v_\theta^{(k-1)}$ with a *high-step* solver to get $z_1^{(k-1)}$.
  2. **Distill** a neCFM on endpoints $(z_0, z_1^{(k-1)})$, **but keep $\mathcal L_{\text{dec}}$** so the endpoint still decodes to $y$.

This is “trajectory straightening by self-distillation,” while anchoring correctness through decoding.

### Inference procedure (compute knobs + stopping)

**Knobs:**

* $N$ = number of ODE steps (1–32)
* Solver: Euler / Heun
* Early stop: if decoded answer is stable across last 2 steps OR confidence threshold reached.

**Pseudocode (inference)**

```python
def flow_of_thought_answer(x, N, solver="euler", conf_thresh=0.9):
    c = Enc(x)
    z = Normal(0, I).sample()
    prev_ans = None
    for i in range(N):
        t = i / N
        v = v_theta(z, t, c)
        if solver == "euler":
            z = z + (1/N) * v
        elif solver == "heun":
            z_pred = z + (1/N) * v
            v_pred = v_theta(z_pred, (i+1)/N, c)
            z = z + (1/(2*N)) * (v + v_pred)

        ans, conf = decode_answer(p_psi, x, z)  # answer-only decoding
        if prev_ans is not None and ans == prev_ans and conf >= conf_thresh:
            break
        prev_ans = ans
    return ans
```

## What is technically new

**New piece = “answer-endpoint latent flow matching for reasoning” + “rectified few-step frontier.”**

* **Objective novelty:** Apply CFM-style endpoint regression (Flow Matching) to **latent reasoning states** whose target is **answer-conditioned** $z^\star=\phi(y,c)$, rather than unconditional generation or token diffusion. This uses the *form* of CFM losses (Eq. 23-style regression along an interpolation) but with a reasoning-specific endpoint. 
* **Inference novelty:** Use **rectified reflow** to make a *few-step* knob meaningful for reasoning correctness, explicitly motivated by solver cost in diffusion/ODE sampling. 
* **Novelty vs Coconut:** Coconut’s latent reasoning is trained via a **multi-stage strategy guins**, and removing that strategy collapses gains.  
  We remove the need for intermediate chains by learning trajector should work (mechanistic intuition)
* The latent ODE is a **learned iterative refinement process**: each step can be interpreted as updating an internal “scratch state” (z) connmakes the target *semantically aligned with the correct answer*, while the decoder loss makes the endpoint *operationally* correct.
* Rectification distills curved trajectories into straighter ones, which should reduce required steps (analogous to distilling many-step generation into few-step generation). 


### Dataset (contamination-resistant)

* **Synthetic multi-step arithmetic word problems** generated from templates (e.ctor sentences).
* Ground truth computed programmatically; answers canonicalized (string normalized).
* Hold out:

  * templates (OOD wording),
  * operation depth (train ≤4 ops, test 5–6 ops),
  * number ranges (train 2-digit, test 3–4 digit).

### Model (small but meaningful)

* $d_z=256$ latent.
* Encoder: small Transformer (or frozen small LM).
* Vector field $v_\theta$: 6–12 layer MLP/Transformer over $[z, \text{time-emb}, c]$.
* Decoder: answer-only generator (often <10 tokens).

### Baselines (minimum set)

1. AR model trained to output answer directly (no CoT), matched params.
2. AR + self-consistency-style sampling (but answer-only outputs; compare compute). (Self-consistency is sample-heavy in the original form. )
3. A masked iterative refinement baseline (tiny MDLM-style unmasking head) to test “iterative refinement alone” vs “latent flow.”

### Success criterion

* **Frontier win:** At $N\in\{1,2,4\}$ steps, Flow-of-Thought ≥ AR direct-answer accuracy at matched latency/FLOPs **and** shows monotonic accuracy improvement.
* **Robustness improvement:** Smaller relative drop on held-out templates/format shifts than AR.

---

# Experimental plan 

## Benchmarks / tasks (≥2 standard + ≥1 stress suite)

**Standard tasks (reasoning):**

1. GSM8K (final answer only; strip rationales)
2. MATH (final answer only; parseable subset first)

**Stress / OOD suite (tailored):**

* **Format shift:** reorder sentences, change units/number formatting, add irrelevant distractor facts.
* **Template OOD:** paraphrase problems (backtranslation / LLM paraphrase) with disjoint templates.
* **Length/generalization:** extrapolate to larger numbers / deeper compositional depth (synthetic + MATH subset).
* **Reversal-like tests:** “given result, infer input” style variants for synthetic arithmetic (controlled analog of reversal-curse probes; compare to diffusion claims like LLaDA). ([arXiv][2])

## Baselines (strong, explicit variants)

**Autoregressive (must include):**

* AR direct-answer SFT (no CoT).
* AR CoT SFT + answer extraction (upper bound but emits CoT).
* CoT prompting (few-shot) baseline. 
* Self-consistency over CoT (report compute carefully; it samples many outputs in the paper). 

**Latent/iterative refinement (must include):**

* Coconut (if reproducible in your stack) and its no-curriculum ablation as referenced by their own finding. 
* MDLM-style masked diffusion fine-tuned tokens masked/unmasked). 
* Discrete Flow Matching (DFM) or simil([proceedings.neurips.cc][1])
* LLaDA-style masked diffusion (if available code/model) as a high-capacity diffusion LM baseline. ([arXiv][2])
* Flow matching for  few steps (FlowSeq) as a “few-step FM for text” comparator. ([aclanthology.org][4])
* Latent diffusi “diffusion-in-latent” comparator. ([arXiv][3])

**Compute normalization rule (pre-registered):**

* Primary x-axis: **#model calls** (vector-field evals / denoiser evals / forward passes) and wall-clock on fixed hardware.
* Secondary: approximate FLOPs.

## Metrics (primary + ≥3 secondary)

**Primary:** Exact-answer accuracy (with strict parsing rules; ambiguous answers counted wrong unless normalized).

**Secondary (≥3):**

1. **Compute frontier:** accuracy vs (steps, NFE, wall-clock, $ cost).
2. **Robustness:** relative accuracy drop under each stress test; report worst-case across perturbations.
3. **Calibration:** ECE on answer correctness using decoder confidence (or ensemble variance).
4. **Stability / variance:** min/median/max over seeds; tail metrics (5th percentile across seeds).
5. (Optional) **Consistency:** agreement under paraphrases (same problem paraphrased K ways).

## Stress tests (systematic + worst-case)

For each test example, generate $M$ perturbations:

* distractor insertion (0/1/3 facts),
* unit conversion / number formatting,
* sentence order shuffle,
* paraphrase style changes.

Report:

* mean accuracy across perturbations,
* **worst-case accuracy** per example (min over perturbations),
* and distribution (percentiles).

## Ablations & sensitivity

**Core ablations:**

* Remove rectification/reflow (train once; vary solver steps).
* Endpoint definition variants:

  * EOS hidden vs pooled answer-token hidden,
  * frozen vs learned $\phi$,
  * with/without paraphrase-consistency regularizer.
* Remove decoder loss $\mathcal L_{\text{dec}}$ (tests whether CFM alone is sufficient).
* Remove contrastive negatives (tests collapse/shortcut).
* Path family variants:

  * linear path $z_t=(1-t)z_0+t z^\star$,
  * learned coupling / OT-inspired path (if you implement).

**Sensitivity sweeps:**

  * Steps $N\in\{1,2,4,8,16,32\}$
  * Reflow iterations $K_{\text{reflow}}\in\{0,1,2,4\}$
  * Data scaling curves (10%, 30%, 100% of training set)

## Statistical rigor

* **Seeds:** $\geq 5$ seeds for small/medium models; $\geq 3$ for large (pre-register).
* **CIs:** bootstrap CI for accuracy; paired bootstrap for baseline comparisons.
* **Significance:** McNemar test (paired classification) where applicable; adjust for multiple comparisons on stress suites.

**Pre-registered key comparisons (must beat):**

* At fixed compute budget (e.g., 4 forward calls), Flow-of-Thought must beat:

  * AR direct-answer SFT
  * MDLM/DFM few-step variants
* At fixed accuracy (e.g., 60% on GSM8K), Flow-of-Thought must use fewer calls than AR+SC.

## Reliability & leakage controls (reasoning benchmarks)

* **No-CoT protocol:** training data contains only final answers; remove/strip rationales.
* **Split hygiene:** verify no near-duplicate overlap between train/test (minhash / n-gram overlap).
* **Template leakage:** for synthetic data, split by *generator template id*, not random.
* **Prompt leakage:** randomize prompt wrappers per split; evaluate with unseen wrappers.
* **Contamination checks:** run substring search of test problems against training corpus (and against any additional scraped data you use) to detect leakage; report flagged rate and “clean subset” results.

---


[1]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html?utm_source=chatgpt.com "Discrete Flow Matching"
[2]: https://arxiv.org/abs/2502.09992?utm_source=chatgpt.com "Large Language Diffusion Models"
[3]: https://arxiv.org/abs/2212.09462?utm_source=chatgpt.com "Latent Diffusion for Language Generation"
[4]: https://aclanthology.org/2024.eacl-short.33/?utm_source=chatgpt.com "Flow Matching for Conditional Text Generation in a Few Sampling Steps - ACL Anthology"
