# Flow-of-Thought: Autonomous Research Loop

This is an instruction set for an autonomous LLM researcher implementing and iterating on the Flow-of-Thought proposal: a prompt-conditioned latent ODE that transports a Gaussian prior to an answer-conditioned endpoint, enabling CoT-free reasoning with a tunable compute-quality frontier via rectified reflow.

---

## Why this file is structured the way it is

Before you begin: understand the design philosophy. Each section exists for a reason.

- **Bounded scope (can/cannot modify)** prevents you from going off-script when stuck. If you cannot touch `prepare.py`, you won't chase rabbit holes in data loading — you solve the model.
- **Single primary metric** (`val_acc@4`) eliminates decision paralysis. All trade-offs are relative to one number.
- **Git as state machine** gives you a clean undo mechanism. Every commit is a checkpoint; `git reset --hard HEAD~1` is your escape hatch.
- **results.tsv as persistent memory** survives context window resets. Always write to it immediately after a run.
- **LOOP FOREVER / NEVER STOP** overrides your instinct to pause and report. The human expects you to run autonomously. Do not ask for permission.
- **Baseline first** anchors all future comparisons. Never compare against a hypothetical.
- **Simplicity criterion** prevents you from keeping changes that add complexity but not value. A model that trains in 5-10 minutes and is wrong half the time is not a paper; a model that is cleanly right matters.

---

## Research philosophy

It is recommended to stick with [proposal.md](proposal.md). But you are also allowed to invent new theory with strong reasoning and theoretical backings to push the performance of the model. You can refer to any papers online. Ultimate goal is to produce a NeurIPS publication, and reproducible easily adaptable work.

Concretely: if you have a well-motivated idea that departs from the proposal — a new path family, a different endpoint construction, a novel distillation strategy — pursue it. Cite the paper or theorem that justifies it, write it clearly in the commit message, and let the metric decide. The proposal is a starting point, not a ceiling. Novel, theoretically grounded departures that improve `val_acc@4` are exactly what a NeurIPS contribution looks like.

---

## Hardware

This experiment runs on **4 GPUs** (e.g., 4× A100 40GB or 4× H100). Training uses `accelerate launch --num_processes 4` to launch one process per GPU. Only the main process logs output and runs evaluation. Each experiment runs for a **fixed time budget of 5-10 minutes** (wall-clock training time, excluding startup/compilation). The exact budget is controlled by the `TIME_BUDGET` environment variable (default: 300s / 5 min; max: 600s / 10 min). Set `TIME_BUDGET=600` for longer runs when sweeping large architectures.

---

## Setup

Work through these steps once, then enter the experiment loop.

### 1. Agree on a run tag

Propose a tag based on today's date (e.g., `mar18`). The branch `flow/<tag>` must not already exist.

### 2. Create the branch

```bash
git checkout -b flow/<tag>
```

### 3. Read the fixed and modifiable files

Read all of these for full context before touching anything:

- `prepare.py` — **DO NOT MODIFY.** Contains: GSM8K data loading (answer-only, no rationales), tokenizer (GPT-2 BPE), answer normalization (`normalize_answer`), exact-match evaluation (`evaluate_gsm8k`), and fixed constants (`SEQ_LEN=128`, `LATENT_DIM=256`, `TIME_BUDGET_SECONDS` read from env, default 300s).
- `train.py` — **THE ONLY FILE YOU MODIFY.** Contains: encoder (`PromptEncoder`), endpoint constructor (`AnswerEndpoint`), vector field (`VelocityField`), decoder (`AnswerDecoder`), training loop, reflow distillation, and inference procedure.

### 4. Verify data

Check that `~/.cache/flow_of_thought/gsm8k/` contains:
- `train_answers.jsonl` — `{"problem": "...", "answer": "..."}` (answer only, rationale stripped)
- `test_answers.jsonl` — same format
- `tokenizer/` — GPT-2 tokenizer files

If missing, run: `uv run prepare.py`

The evaluation harness calls `evaluate_gsm8k(model, test_loader, n_steps=[1, 2, 4, 8])` which returns a dict `{steps: accuracy}`. It always uses greedy decoding and exact match after `normalize_answer`.

### 5. Initialize results.tsv

Create `results.tsv` with the header only:

```
commit	val_acc@1	val_acc@4	nfe_budget	memory_gb	status	description
```

Leave this file **untracked by git** (it is your lab notebook, not part of the experiment).

### 6. Confirm and go

Confirm setup looks good, then immediately begin the experiment loop.

---

## Fixed constants (from prepare.py — do not override)

| Constant | Value | Why fixed |
|---|---|---|
| `SEQ_LEN` | 128 | Covers all GSM8K problems |
| `LATENT_DIM` | 256 | Proposal spec; ablated separately |
| `TIME_BUDGET_SECONDS` | 300–600 | Wall-clock training budget (env var, default 300s / 5 min, max 600s / 10 min) |
| `EVAL_STEPS` | [1, 2, 4, 8] | Defines the frontier |
| `BATCH_SIZE` | 64 | Set for A100 40GB baseline |

---

## What you CAN modify in train.py

Everything in `train.py` is fair game:

- **PromptEncoder**: architecture (frozen GPT-2 embeddings, trainable transformer layers), hidden dim, pooling strategy.
- **AnswerEndpoint** (`φ_η`): frozen LM choice, pooling (EOS vs mean answer-token hidden states), projection `W` dimensionality, paraphrase consistency regularizer.
- **VelocityField** (`v_θ`): depth (default 8 layers), width, architecture type (MLP, MLP-Mixer, small Transformer), time embedding (sinusoidal vs learned), conditioning mechanism (FiLM vs concatenation vs cross-attention).
- **AnswerDecoder** (`p_ψ`): head type (linear projection vs small AR), answer length budget.
- **Loss weights**: `λ_dec`, `λ_NCE`, temperature `τ`.
- **Optimizer**: AdamW, Lion, Muon; learning rate, weight decay, schedule (cosine, linear warmup).
- **Training loop**: gradient accumulation, mixed precision, curriculum ordering.
- **Reflow**: number of reflow iterations `K_reflow` ∈ {0, 1, 2, 4}, high-step solver for teacher trajectories.
- **Path family**: linear interpolation (default) vs OT-minibatch coupling vs variance-exploding.
- **Curriculum** (NeurIPS-critical): staged training à la Coconut — start with high `λ_dec`, gradually increase `λ_CFM` weight. Evidence from Coconut (Hao et al., 2024) shows curriculum is essential for latent reasoning; do not skip this ablation.

## What you CANNOT modify

- `prepare.py` — read-only. Do not touch data loading, tokenizer, evaluation, or fixed constants.
- The evaluation protocol: always use `evaluate_gsm8k` from `prepare.py`.
- The GSM8K test split.
- Install new packages beyond what is in `pyproject.toml`.

---

## Primary goal

**Maximize `val_acc@4`** (exact-match accuracy on GSM8K test set at 4 ODE solver steps).

Secondary: show monotonic improvement — `val_acc@1 ≤ val_acc@2 ≤ val_acc@4 ≤ val_acc@8`. A model that does not improve with more steps has a broken ODE.

Publication bar (NeurIPS): at 4 steps, `val_acc@4` must exceed AR direct-answer SFT at matched FLOPs. At fixed accuracy, Flow-of-Thought must use fewer NFE than AR+self-consistency.

---

## Simplicity criterion

All else equal, simpler is better.

- A 0.5% `val_acc@4` improvement that adds 50 lines of architecture hacks? Probably not worth it.
- A 0.5% improvement from a clean change (e.g., FiLM conditioning)? Keep it.
- An equal result from deleting code? Definitely keep the deletion.
- Removing a loss term and getting the same accuracy = simplification win, record as `keep`.

When evaluating whether to keep a change: weigh complexity cost against improvement magnitude.

---

## The first run: establish baseline

Your very first run must be `train.py` as written (no modifications). This establishes the baseline row in `results.tsv`.

The baseline configuration should be:
- GPT-2 small (frozen) as encoder.
- 8-layer MLP velocity field, hidden dim 512, sinusoidal time embedding, FiLM conditioning.
- Linear interpolation path.
- `λ_dec=1.0`, `λ_NCE=0.1`, `τ=0.07`.
- AdamW, lr=3e-4, cosine decay, 1000-step warmup.
- No reflow (K_reflow=0).
- EOS pooling for endpoint.

---

## Output format

When training finishes, the script prints a summary block:

```
---
val_acc@1:        0.1234
val_acc@2:        0.1589
val_acc@4:        0.2103
val_acc@8:        0.2341
training_seconds: 300.2
total_seconds:    327.8
peak_vram_mb:     21504.0
nfe_budget:       4
num_steps:        1200
num_params_M:     47.3
reflow_iters:     0
```

Extract key metrics from log:

```bash
grep "^val_acc@\|^peak_vram_mb:" run.log
```

If the grep is empty, the run crashed — read the traceback:

```bash
tail -n 60 run.log
```

---

## Logging results

After every run, append one row to `results.tsv` immediately. Tab-separated, never comma-separated.

Columns:

```
commit	val_acc@1	val_acc@4	nfe_budget	memory_gb	status	description
```

1. `commit` — 7-char git hash
2. `val_acc@1` — exact-match accuracy at 1 step (e.g., `0.123456`)
3. `val_acc@4` — exact-match accuracy at 4 steps (e.g., `0.210300`)
4. `nfe_budget` — always `4` unless you change the primary knob
5. `memory_gb` — `peak_vram_mb / 1024`, rounded to 1 decimal
6. `status` — `keep`, `discard`, or `crash`
7. `description` — one short phrase; no commas

Example after several runs:

```
commit	val_acc@1	val_acc@4	nfe_budget	memory_gb	status	description
a1b2c3d	0.089200	0.156300	4	21.0	keep	baseline: 8L MLP v_θ frozen GPT-2 enc
b2c3d4e	0.091400	0.162100	4	21.1	keep	FiLM conditioning (was concat)
c3d4e5f	0.088000	0.153000	4	21.0	discard	remove λ_NCE (worse calibration)
d4e5f6g	0.000000	0.000000	4	0.0	crash	OT coupling (index error in batch)
e5f6g7h	0.094300	0.171500	4	21.4	keep	curriculum: warm λ_dec for 500 steps
```

---

## The experiment loop

LOOP FOREVER:

1. **Check git state**: `git log --oneline -5` to see where you are.

2. **Form a hypothesis**: Based on previous results and the proposal, pick one change to test. Prioritize in this order:
   - **Tier 1 (highest expected gain):** Curriculum learning schedule, reflow distillation (K_reflow), velocity field depth/architecture, endpoint pooling strategy.
   - **Tier 2 (medium gain):** Loss weight sweeps (λ_dec, λ_NCE), optimizer/LR changes, time embedding design, OT-minibatch path coupling.
   - **Tier 3 (refinement):** Contrastive negative mining strategy, decoder head design, early-stopping threshold.

3. **Edit train.py** — one focused change per experiment. Do not bundle unrelated changes.

4. **git commit** the change with a concise message.

5. **Run the experiment**:
   ```bash
   accelerate launch --num_processes 4 train.py > run.log 2>&1
   ```
   Do NOT use `tee` — let all output go to the log file. Only the main process writes the summary block.

6. **Read results**:
   ```bash
   grep "^val_acc@4:\|^val_acc@1:\|^peak_vram_mb:" run.log
   ```

7. **If grep is empty (crash)**: run `tail -n 60 run.log`. If it's a dumb bug (typo, missing import, shape mismatch), fix and re-run. If the idea is fundamentally broken (OOM with no clear fix, diverged loss), log as `crash`, `git reset --hard HEAD~1`, and move on.

8. **Log to results.tsv** immediately.

9. **Advance or revert**:
   - If `val_acc@4` improved (strictly higher): **keep** the commit, advance.
   - If `val_acc@4` is equal but the code is simpler: **keep** as a simplification win.
   - If `val_acc@4` is equal or worse and code is more complex: `git reset --hard HEAD~1`, log `discard`.

10. Repeat.

---

## Timeout and crash policy

- **Timeout**: If a run exceeds 12 minutes wall-clock, kill it (`Ctrl-C` or `kill`), log as `crash`, `git reset --hard HEAD~1`, move on.
- **Repeated crashes**: If the same idea crashes twice, abandon it. Log both, move on.
- **OOM**: First try halving the batch size or reducing velocity field width. If still OOM, abandon the idea.
- **NaN loss**: Check for missing `LayerNorm` before time embedding injection; add gradient clipping (`max_norm=1.0`). If still NaN after one fix attempt, abandon.

---

## Experiment priority queue (ordered by expected NeurIPS impact)

Work through this queue. Re-order based on what the results tell you.

### Phase 1 — Core functionality (must complete before Phase 2)

1. **Baseline** — run as-is, establish anchor.
2. **Curriculum warmup** — ramp `λ_CFM` from 0→1 over first 30% of training, keep `λ_dec=1` throughout. Coconut (2024) showed this is essential; if you skip it, the ODE cannot learn meaningful trajectories.
3. **K_reflow=1** — one round of reflow distillation using baseline teacher trajectories. This is the core compute-quality knob and the main paper claim.
4. **K_reflow=2** — does a second round help? This determines whether the frontier keeps improving.

### Phase 2 — Architecture ablations (for the ablation table)

5. **EOS vs mean pooling** in `AnswerEndpoint` — which gives better endpoint stability?
6. **Frozen vs finetuned φ_η** — does training the endpoint encoder help or overfit?
7. **FiLM vs cross-attention conditioning** in velocity field — FiLM is simpler; cross-attn may be better.
8. **MLP vs Transformer velocity field** — 8-layer MLP vs 4-layer Transformer with same param budget.
9. **Remove λ_NCE** — does contrastive anchoring help, or is the decoder loss sufficient?
10. **Remove λ_dec** — does CFM alone converge to a decodable endpoint? (Expected: no — this justifies the loss design.)

### Phase 3 — Frontier expansion (compute-quality story)

11. **Heun solver vs Euler** at same NFE — does second-order solver give free gains?
12. **Early stopping** — does the stability criterion (ans == prev_ans and conf ≥ 0.9) reduce average NFE while maintaining accuracy?
13. **Steps sweep** — evaluate at N∈{1,2,4,8,16} with and without reflow. The core figure for the paper is this frontier curve.

### Phase 4 — Robustness / OOD (stress suite for the paper)

14. **Format perturbation test** — evaluate on GSM8K with shuffled sentences and distractor facts (generated by `prepare.py`'s `perturb_gsm8k` function). Report worst-case accuracy per example.
15. **Number scaling OOD** — evaluate on 3-4 digit number variants held out from training.

---

## Baselines to implement in train.py (AR comparators)

You must implement these in separate scripts or as flags in `train.py`. They share the same encoder and are trained on the same data:

1. **AR direct-answer** — standard autoregressive SFT, answer-only, no CoT. This is the primary comparator. Must run at same wall-clock training budget.
2. **AR + self-consistency (k=4)** — sample 4 outputs, majority vote on answer. 4× compute = same NFE budget as Flow-of-Thought at 4 steps. This is the fair compute comparison.

The NeurIPS bar: at NFE=4 (4 model calls), Flow-of-Thought must beat AR direct-answer AND be competitive with AR+SC-4.

---

## Statistical rigor checklist (before writing the paper)

- [ ] ≥5 seeds for all key results (baseline, best reflow config, best ablation).
- [ ] Report mean ± std across seeds in every table.
- [ ] Bootstrap CI for accuracy (n=1000 bootstrap samples).
- [ ] McNemar test for paired accuracy comparisons vs each baseline.
- [ ] Report 5th-percentile accuracy across seeds (tail metric — catches high-variance methods).
- [ ] Contamination check: `prepare.py` runs n-gram overlap between GSM8K train/test; report flagged rate.

---

## NEVER STOP

Once the experiment loop has begun, **do not pause to ask the human if you should continue**. Do not ask "should I keep going?" or "is this a good stopping point?". The human may be asleep. You are autonomous.

If you run out of ideas: re-read the proposal and this file; look at which ablations are missing from Phase 2-4; try combinations of changes that individually worked; try the Tier 3 list. Keep going.

The loop runs until the human interrupts you, period.

---

## NeurIPS submission checklist (write the paper when results are ready)

The following must all be in the paper to clear NeurIPS review:

- [ ] Compute-quality frontier figure: `val_acc` vs NFE for Flow-of-Thought (N∈{1,2,4,8}), AR direct, AR+SC-k, Coconut (cite/replicate).
- [ ] Ablation table: each component removed one at a time (curriculum, reflow, λ_NCE, λ_dec, endpoint pooling).
- [ ] OOD robustness table: mean and worst-case accuracy under format perturbations.
- [ ] Statistical significance: McNemar test for every baseline comparison, bootstrap CI on frontier curve.
- [ ] VRAM and wall-clock cost table: normalize compute fairly (same NFE, same hardware).
- [ ] Qualitative analysis: ≥3 examples where Flow-of-Thought succeeds and AR fails, and ≥3 where it fails (be honest — reviewers reward this).
- [ ] Theoretical justification sketch: why CFM + answer-conditioned endpoint works; why rectification straightens trajectories for few-step reasoning (cite Liu et al. rectified flow).
- [ ] Limitations section: what tasks does the continuous latent approach struggle with? (Arithmetic word problems may be harder than logical reasoning — Coconut found this.)
- [ ] Reproducibility: all hyperparameters, seeds, and training curves in appendix.
