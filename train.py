"""
Flow-of-Thought: latent ODE reasoning with conditional flow matching + rectified reflow.
Multi-GPU training via Accelerate (4 GPUs).

Usage:
    accelerate launch --num_processes 4 train.py
    TIME_BUDGET=600 accelerate launch --num_processes 4 train.py  # 10-minute budget
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

from prepare import (
    LATENT_DIM, CONTEXT_DIM, MAX_SEQ_LEN, MAX_ANS_LEN,
    TIME_BUDGET, EVAL_STEPS,
    make_dataloader, evaluate_gsm8k,
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit freely (only this file)
# ---------------------------------------------------------------------------

# Frozen backbone — swap this to ablate encoder capacity.
# Must be a decoder-only causal LM available on HuggingFace.
# Ablation ladder (commit each, log val_acc@4):
#   "Qwen/Qwen2.5-0.5B"          ~1GB  — fast, low capacity
#   "Qwen/Qwen2.5-1.5B"          ~3GB  — default: strong math, Apache 2.0
#   "Qwen/Qwen2.5-3B"            ~6GB  — step up if 1.5B plateaus
#   "microsoft/Phi-3.5-mini-instruct"   ~7GB  — SOTA small math model
# Qwen2.5 chosen as default: top-ranked on math benchmarks at this scale
# (see Qwen2.5 tech report, 2024), Apache 2.0, no gated access required.
BACKBONE = "Qwen/Qwen2.5-1.5B"

# Latent / context dims (must match prepare.py constants)
D_Z = LATENT_DIM   # 256
D_C = CONTEXT_DIM  # 256

# Velocity field v_θ architecture
D_V       = 512   # hidden dim of FiLM-MLP layers
N_LAYERS  = 8     # number of FiLM-MLP layers
D_T_EMB   = 64    # sinusoidal time embedding dim

# Decoder p_ψ
D_DEC = 256       # LSTM hidden dim

# Training
BATCH_SIZE   = 64     # global batch (split across GPUs: 64/4 = 16 per GPU)
LR           = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_FRAC  = 0.05   # fraction of TIME_BUDGET for LR linear warmup
LAMBDA_DEC   = 1.0    # weight for decoder loss
LAMBDA_NCE   = 0.1    # weight for contrastive NCE loss
TAU          = 0.07   # contrastive temperature
CURRICULUM_FRAC = 0.30  # ramp λ_CFM 0→1 over first 30% of training (Coconut lesson)

# Fast-screening controls
# Use DATA_SLICE_FRAC < 1.0 to run on a smaller random subset of train/test data.
DATA_SLICE_FRAC = float(os.environ.get("DATA_SLICE_FRAC", "1.0"))
DATA_SLICE_SEED = int(os.environ.get("DATA_SLICE_SEED", "1337"))

# Reflow distillation (0 = no reflow; try 1 or 2 after baseline)
K_REFLOW     = 0
REFLOW_STEPS = 32   # high-step teacher solver for reflow pairs

# Inference
SOLVER      = "euler"   # "euler" or "heun"
CONF_THRESH = 0.9       # confidence threshold for early stopping

# ---------------------------------------------------------------------------
# Distributed setup (accelerate launch --num_processes 4)
# ---------------------------------------------------------------------------

accelerator = Accelerator()
is_master   = accelerator.is_main_process
world_size  = accelerator.num_processes
device      = accelerator.device

torch.manual_seed(42 + accelerator.process_index)
torch.set_float32_matmul_precision("high")

t_start = time.time()

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = len(tokenizer)  # includes special tokens (pad/eos beyond base vocab_size)
PAD_ID     = tokenizer.pad_token_id
BOS_ID     = tokenizer.bos_token_id or tokenizer.eos_token_id

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) scalar per sample → (B, dim) sinusoidal embedding."""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) *
        torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class PromptEncoder(nn.Module):
    """Frozen backbone + mean pooling over problem tokens → context c ∈ R^{d_c}.
    Backbone is BACKBONE (default: Qwen2.5-1.5B). Frozen — only the projection is trained.
    """

    def __init__(self, backbone: AutoModel, context_dim: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(backbone.config.hidden_size, context_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        c = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(c)  # (B, d_c)


class AnswerEndpoint(nn.Module):
    """φ_η: frozen backbone on [problem <EOS> answer] → last-token hidden state → z* ∈ R^{d_z}.
    Last token of a causal LM has attended to the full sequence; its hidden state
    encodes answer-conditioned semantics without leaking chain-of-thought.
    """

    def __init__(self, backbone: AutoModel, latent_dim: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(backbone.config.hidden_size, latent_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lens = attention_mask.sum(1) - 1  # index of last non-pad token
        with torch.no_grad():
            h = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        h_last = h[torch.arange(h.size(0), device=h.device), seq_lens]  # (B, hidden_size)
        return self.proj(h_last)  # (B, d_z)


class FiLMBlock(nn.Module):
    """Single FiLM-conditioned residual MLP block: norm → linear → FiLM(c) → SiLU."""

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.gamma  = nn.Linear(context_dim, hidden_dim)
        self.beta   = nn.Linear(context_dim, hidden_dim)
        # Init FiLM to identity (gamma=1, beta=0) for stable early training
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear(h)
        h = self.gamma(c) * h + self.beta(c)
        return x + F.silu(h)  # residual


class VelocityField(nn.Module):
    """v_θ(z_t, t, c) → v ∈ R^{d_z}.
    8-layer FiLM-conditioned MLP. Input: [z_t ‖ t_emb], conditioned on c at each layer.
    Output projection initialized to zero → starts as identity flow.
    """

    def __init__(self, latent_dim: int, context_dim: int,
                 hidden_dim: int, n_layers: int, t_emb_dim: int):
        super().__init__()
        self.t_emb_dim  = t_emb_dim
        self.input_proj = nn.Linear(latent_dim + t_emb_dim, hidden_dim)
        self.layers     = nn.ModuleList([
            FiLMBlock(hidden_dim, context_dim) for _ in range(n_layers)
        ])
        self.out_proj   = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        h = self.input_proj(torch.cat([z, t_emb], dim=-1))
        for layer in self.layers:
            h = layer(h, c)
        return self.out_proj(h)  # (B, d_z)


class AnswerDecoder(nn.Module):
    """p_ψ(y | z_1): LSTM decoder from latent endpoint to answer token sequence."""

    def __init__(self, latent_dim: int, hidden_dim: int,
                 vocab_size: int, max_len: int, pad_id: int):
        super().__init__()
        self.z_to_h  = nn.Linear(latent_dim, hidden_dim)
        self.z_to_c  = nn.Linear(latent_dim, hidden_dim)
        self.embed   = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.lstm    = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out     = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len
        self.pad_id  = pad_id

    def forward(self, z: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training forward. Returns logits (B, L-1, vocab_size)."""
        h0  = self.z_to_h(z).unsqueeze(0)   # (1, B, hidden)
        c0  = self.z_to_c(z).unsqueeze(0)
        inp = self.embed(answer_ids[:, :-1]) # (B, L-1, hidden)
        out, _ = self.lstm(inp, (h0, c0))
        return self.out(out)                 # (B, L-1, vocab_size)

    @torch.no_grad()
    def generate(self, z: torch.Tensor, bos_id: int) -> torch.Tensor:
        """Greedy decode. Returns (B, max_len) token ids."""
        B   = z.size(0)
        h   = self.z_to_h(z).unsqueeze(0)
        c   = self.z_to_c(z).unsqueeze(0)
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=z.device)
        tokens = []
        for _ in range(self.max_len):
            emb = self.embed(tok)
            out, (h, c) = self.lstm(emb, (h, c))
            tok = self.out(out.squeeze(1)).argmax(-1, keepdim=True)
            tokens.append(tok)
        return torch.cat(tokens, dim=1)  # (B, max_len)


# ---------------------------------------------------------------------------
# ODE solvers
# ---------------------------------------------------------------------------

def euler_solve(v_fn, z0: torch.Tensor, c: torch.Tensor, n_steps: int) -> torch.Tensor:
    z, dt = z0, 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((z.size(0),), i * dt, device=z.device)
        z = z + dt * v_fn(z, t, c)
    return z


def heun_solve(v_fn, z0: torch.Tensor, c: torch.Tensor, n_steps: int) -> torch.Tensor:
    z, dt = z0, 1.0 / n_steps
    for i in range(n_steps):
        t0 = torch.full((z.size(0),), i * dt,       device=z.device)
        t1 = torch.full((z.size(0),), (i+1) * dt,   device=z.device)
        v0 = v_fn(z, t0, c)
        z1 = z + dt * v0
        v1 = v_fn(z1, t1, c)
        z  = z + 0.5 * dt * (v0 + v1)
    return z


def ode_solve(v_fn, z0, c, n_steps, solver="euler"):
    return heun_solve(v_fn, z0, c, n_steps) if solver == "heun" else euler_solve(v_fn, z0, c, n_steps)


# ---------------------------------------------------------------------------
# Full Flow-of-Thought model
# ---------------------------------------------------------------------------

class FlowOfThought(nn.Module):
    def __init__(self):
        super().__init__()
        # Single frozen backbone shared by PromptEncoder and AnswerEndpoint.
        # Loaded in bfloat16 to save VRAM; frozen so it never accumulates gradients.
        backbone = AutoModel.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16)
        self.prompt_encoder  = PromptEncoder(backbone, D_C)
        self.answer_endpoint = AnswerEndpoint(backbone, D_Z)
        self.velocity        = VelocityField(D_Z, D_C, D_V, N_LAYERS, D_T_EMB)
        self.decoder         = AnswerDecoder(D_Z, D_DEC, VOCAB_SIZE, MAX_ANS_LEN, PAD_ID)

    def forward(self, batch: dict) -> tuple:
        """Returns (loss_cfm, loss_dec, loss_nce)."""
        B = batch["problem_input_ids"].size(0)

        # Encode prompt → context c ∈ R^{d_c}
        c = self.prompt_encoder(
            batch["problem_input_ids"], batch["problem_attention_mask"]
        )

        # Encode [problem <EOS> answer] → endpoint z* ∈ R^{d_z}
        z_star = self.answer_endpoint(
            batch["endpoint_input_ids"], batch["endpoint_attention_mask"]
        )

        # ── (i) CFM loss ────────────────────────────────────────────────────
        # Linear interpolation path: z_t = (1-t)·z_0 + t·z*, target u = z* - z_0
        t   = torch.rand(B, device=c.device)
        z0  = torch.randn(B, D_Z, device=c.device)
        z_t = (1 - t[:, None]) * z0 + t[:, None] * z_star
        u   = z_star - z0
        v   = self.velocity(z_t, t, c)
        loss_cfm = F.mse_loss(v, u)

        # ── (ii) Decoder loss ────────────────────────────────────────────────
        # Decode from integrated endpoint ẑ_1 so decoder and flow are coupled.
        # This matches the proposal objective L_dec(y | x, ẑ_1).
        z_hat = euler_solve(self.velocity, z0, c, n_steps=4)
        z_for_dec = z_hat + 0.1 * torch.randn_like(z_hat)
        logits    = self.decoder(z_for_dec, batch["answer_input_ids"])
        targets   = batch["answer_input_ids"][:, 1:].contiguous()
        loss_dec  = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1),
            ignore_index=PAD_ID,
        )

        # ── (iii) Contrastive NCE (in-batch negatives) ──────────────────────
        # Push predicted endpoint ẑ_1 toward z* with in-batch negatives
        z_hat_n  = F.normalize(z_hat,  dim=-1)
        z_star_n = F.normalize(z_star, dim=-1)
        sim      = z_hat_n @ z_star_n.T / TAU  # (B, B) — diagonal = positives
        loss_nce = F.cross_entropy(sim, torch.arange(B, device=c.device))

        return loss_cfm, loss_dec, loss_nce

    @torch.no_grad()
    def generate(self, batch: dict, n_steps: int = 4, solver: str = SOLVER) -> list:
        """Inference: problem → answer string. Used by evaluate_gsm8k."""
        c  = self.prompt_encoder(
            batch["problem_input_ids"], batch["problem_attention_mask"]
        )
        z0 = torch.randn(c.size(0), D_Z, device=c.device)
        z1 = ode_solve(self.velocity, z0, c, n_steps, solver)
        tok_ids = self.decoder.generate(z1, BOS_ID)  # (B, MAX_ANS_LEN)
        return [
            tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            for ids in tok_ids
        ]


# ---------------------------------------------------------------------------
# Setup: model, optimizer, dataloaders
# ---------------------------------------------------------------------------

if is_master:
    print(f"Loading backbone ({BACKBONE}) and building FlowOfThought model...")

model = FlowOfThought()
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

if is_master:
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {n_trainable / 1e6:.1f}M  |  Frozen backbone: {n_frozen / 1e6:.1f}M")

per_gpu_batch = max(1, BATCH_SIZE // world_size)


def build_loader(split: str, shuffle: bool) -> DataLoader:
    loader = make_dataloader(split, tokenizer, batch_size=per_gpu_batch, shuffle=shuffle)
    if DATA_SLICE_FRAC >= 1.0:
        return loader

    if not (0.0 < DATA_SLICE_FRAC <= 1.0):
        raise ValueError("DATA_SLICE_FRAC must be in (0, 1].")

    ds = loader.dataset
    n_keep = max(1, int(len(ds) * DATA_SLICE_FRAC))
    g = torch.Generator()
    g.manual_seed(DATA_SLICE_SEED + (0 if split == "train" else 1))
    keep_idx = torch.randperm(len(ds), generator=g)[:n_keep].tolist()
    ds_subset = Subset(ds, keep_idx)

    return DataLoader(
        ds_subset,
        batch_size=per_gpu_batch,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=loader.collate_fn,
        persistent_workers=True,
        drop_last=(split == "train"),
    )


train_loader = build_loader("train", shuffle=True)
test_loader = build_loader("test", shuffle=False)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
)

model, optimizer, train_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, test_loader
)

train_iter = iter(train_loader)

if is_master:
    print(f"Time budget:    {TIME_BUDGET}s  |  world_size: {world_size}  |  per_gpu_batch: {per_gpu_batch}")
    print(f"Data slice:     {DATA_SLICE_FRAC:.2f}  |  seed: {DATA_SLICE_SEED}")
    print(f"λ_dec: {LAMBDA_DEC}  λ_nce: {LAMBDA_NCE}  τ: {TAU}  K_reflow: {K_REFLOW}  solver: {SOLVER}")

# ---------------------------------------------------------------------------
# LR and curriculum schedules
# ---------------------------------------------------------------------------

def get_lr(elapsed: float) -> float:
    progress = min(elapsed / TIME_BUDGET, 1.0)
    if progress < WARMUP_FRAC:
        return LR * (progress / max(WARMUP_FRAC, 1e-8))
    t = (progress - WARMUP_FRAC) / max(1 - WARMUP_FRAC, 1e-8)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))  # cosine decay to 10%


def get_cfm_weight(elapsed: float) -> float:
    """Ramp λ_CFM from 0 → 1 over CURRICULUM_FRAC·TIME_BUDGET.
    Based on Coconut (Hao et al., 2024): curriculum is essential for latent reasoning.
    Starts decoder-only so z* is decodable before the flow tries to reach it.
    """
    return min(1.0, elapsed / max(CURRICULUM_FRAC * TIME_BUDGET, 1.0))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
smooth_loss = 0.0

model.train()

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    elapsed = time.time() - t_start_training
    lr      = get_lr(elapsed)
    w_cfm   = get_cfm_weight(elapsed)

    for g in optimizer.param_groups:
        g["lr"] = lr

    loss_cfm, loss_dec, loss_nce = model(batch)
    loss = w_cfm * loss_cfm + LAMBDA_DEC * loss_dec + LAMBDA_NCE * loss_nce

    optimizer.zero_grad()
    accelerator.backward(loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    torch.cuda.synchronize()
    dt = time.time() - t0

    if step > 5:
        total_training_time += dt

    loss_f = loss.item()
    if math.isnan(loss_f) or loss_f > 1e4:
        if is_master:
            print("\nFAIL — loss exploded")
        break

    smooth_loss = 0.95 * smooth_loss + 0.05 * loss_f
    debiased    = smooth_loss / (1 - 0.95 ** (step + 1))
    pct         = 100 * min(elapsed / TIME_BUDGET, 1.0)
    remaining   = max(0, TIME_BUDGET - total_training_time)

    if is_master:
        print(
            f"\rstep {step:05d} ({pct:.1f}%) | "
            f"loss: {debiased:.4f} (cfm={loss_cfm.item():.4f} dec={loss_dec.item():.4f} nce={loss_nce.item():.4f}) | "
            f"w_cfm: {w_cfm:.2f} | lr: {lr:.2e} | dt: {dt*1000:.0f}ms | rem: {remaining:.0f}s    ",
            end="", flush=True,
        )

    step += 1
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

if is_master:
    print()

# ---------------------------------------------------------------------------
# Rectified reflow distillation (K_REFLOW > 0)
# Straightens curved trajectories → fewer steps needed at inference.
# ---------------------------------------------------------------------------

for reflow_iter in range(K_REFLOW):
    if is_master:
        print(f"--- Reflow iteration {reflow_iter + 1}/{K_REFLOW} ---")

    raw_model    = accelerator.unwrap_model(model)
    raw_model.eval()

    # Collect (z0, z1, c, answer_ids) tuples from current model using high-step solver
    reflow_pairs = []
    reflow_loader = make_dataloader("train", tokenizer, batch_size=per_gpu_batch, shuffle=False)
    reflow_loader = accelerator.prepare(reflow_loader)

    with torch.no_grad():
        for batch in reflow_loader:
            c  = raw_model.prompt_encoder(
                batch["problem_input_ids"], batch["problem_attention_mask"]
            )
            z0 = torch.randn(c.size(0), D_Z, device=c.device)
            z1 = euler_solve(raw_model.velocity, z0, c, n_steps=REFLOW_STEPS)
            reflow_pairs.append((
                z0.cpu(),
                z1.cpu(),
                c.cpu(),
                batch["answer_input_ids"].cpu(),
            ))
            if len(reflow_pairs) * per_gpu_batch >= 2000:
                break

    # Retrain velocity on straightened (z0, z1) pairs while keeping decoder anchored
    raw_model.train()
    t_reflow_start = time.time()
    reflow_budget  = TIME_BUDGET // max(K_REFLOW, 1)
    ri = 0

    while time.time() - t_reflow_start < reflow_budget:
        z0_r, z1_r, c_r, ans_ids_r = reflow_pairs[ri % len(reflow_pairs)]
        z0_r = z0_r.to(device)
        z1_r = z1_r.to(device)
        c_r = c_r.to(device)
        ans_ids_r = ans_ids_r.to(device)

        t   = torch.rand(z0_r.size(0), device=device)
        z_t = (1 - t[:, None]) * z0_r + t[:, None] * z1_r
        u   = z1_r - z0_r
        v   = raw_model.velocity(z_t, t, c_r)
        rl_cfm = F.mse_loss(v, u)

        z_reflow_dec = z1_r + 0.1 * torch.randn_like(z1_r)
        logits_r = raw_model.decoder(z_reflow_dec, ans_ids_r)
        targets_r = ans_ids_r[:, 1:].contiguous()
        rl_dec = F.cross_entropy(
            logits_r.reshape(-1, VOCAB_SIZE), targets_r.reshape(-1),
            ignore_index=PAD_ID,
        )

        rl = rl_cfm + LAMBDA_DEC * rl_dec

        optimizer.zero_grad()
        accelerator.backward(rl)
        optimizer.step()
        ri += 1

    if is_master:
        print(f"  Reflow done ({ri} steps, {time.time() - t_reflow_start:.0f}s)")

# ---------------------------------------------------------------------------
# Final evaluation on GSM8K test set
# ---------------------------------------------------------------------------

accelerator.wait_for_everyone()

if is_master:
    print("Evaluating on GSM8K test set...")

val_accs = evaluate_gsm8k(
    accelerator.unwrap_model(model), test_loader, accelerator, EVAL_STEPS
)

accelerator.wait_for_everyone()

# ---------------------------------------------------------------------------
# Final summary — parsed by the experiment loop (grep "^val_acc")
# ---------------------------------------------------------------------------

if is_master:
    t_end        = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    for n in EVAL_STEPS:
        print(f"val_acc@{n}:        {val_accs.get(n, 0.0):.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_trainable / 1e6:.1f}")
    print(f"world_size:       {world_size}")
    print(f"k_reflow:         {K_REFLOW}")
    print(f"solver:           {SOLVER}")
    print(f"lambda_dec:       {LAMBDA_DEC}")
    print(f"lambda_nce:       {LAMBDA_NCE}")
