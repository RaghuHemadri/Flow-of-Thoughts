"""
Fixed data preparation for Flow-of-Thought experiments on GSM8K.
DO NOT MODIFY — this file contains the fixed evaluation harness and constants.

Usage:
    uv run prepare.py    # downloads GSM8K and verifies setup
"""

import os
import re
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants (fixed — do not override in train.py)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN  = 128   # max tokens for problem + answer (GPT-2 tokenizer)
MAX_ANS_LEN  = 16    # max answer tokens (GSM8K answers fit in ≤ 10 tokens)
LATENT_DIM   = 256   # z-space dimension (d_z)
CONTEXT_DIM  = 256   # prompt context dimension (d_c)
TIME_BUDGET  = int(os.environ.get("TIME_BUDGET", 300))  # 5-10 min wall-clock
EVAL_STEPS   = [1, 2, 4, 8]  # NFE values for compute-quality frontier curve
CACHE_DIR    = Path.home() / ".cache" / "flow_of_thought"

# ---------------------------------------------------------------------------
# Answer normalization (fixed — defines exact-match evaluation)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize a GSM8K answer to a canonical numeric string for exact match."""
    s = s.replace(",", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return s.lower()
    val = nums[-1]
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else f"{f:.4f}".rstrip("0")
    except ValueError:
        return val


def _extract_final_answer(answer_text: str) -> str:
    """Strip rationale from GSM8K answer field, returning only the number after ####."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer_text)
    return nums[-1].replace(",", "") if nums else answer_text.strip()

# ---------------------------------------------------------------------------
# GSM8K loading
# ---------------------------------------------------------------------------

def load_gsm8k(split: str) -> list:
    """Load GSM8K split with final answer only (no rationale).
    Returns list of {'question': str, 'answer': str}.
    Caches to CACHE_DIR after first download.
    """
    cache_path = CACHE_DIR / f"gsm8k_{split}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    examples = [
        {"question": ex["question"].strip(), "answer": _extract_final_answer(ex["answer"])}
        for ex in ds
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(examples, f)
    print(f"Cached {len(examples)} GSM8K {split} examples → {cache_path}")
    return examples

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GSM8KDataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_seq_len=MAX_SEQ_LEN, max_ans_len=MAX_ANS_LEN):
        self.examples    = examples
        self.tok         = tokenizer
        self.max_seq_len = max_seq_len
        self.max_ans_len = max_ans_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        q, a = ex["question"], ex["answer"]

        # Problem tokens → PromptEncoder
        q_enc = self.tok(
            q, max_length=self.max_seq_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        # [problem <EOS> answer] tokens → AnswerEndpoint (φ_η)
        qa_enc = self.tok(
            q + self.tok.eos_token + a,
            max_length=self.max_seq_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        # Answer tokens → AnswerDecoder (p_ψ) training
        a_enc = self.tok(
            a, max_length=self.max_ans_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        return {
            "problem_input_ids":       q_enc["input_ids"].squeeze(0),
            "problem_attention_mask":  q_enc["attention_mask"].squeeze(0),
            "endpoint_input_ids":      qa_enc["input_ids"].squeeze(0),
            "endpoint_attention_mask": qa_enc["attention_mask"].squeeze(0),
            "answer_input_ids":        a_enc["input_ids"].squeeze(0),
            "answer_attention_mask":   a_enc["attention_mask"].squeeze(0),
            "answer_str":              a,
        }


def _collate(batch):
    tensor_keys = [k for k in batch[0] if k != "answer_str"]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["answer_str"] = [b["answer_str"] for b in batch]
    return out


def make_dataloader(split: str, tokenizer, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = GSM8KDataset(load_gsm8k(split), tokenizer)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True, collate_fn=_collate,
        persistent_workers=True, drop_last=(split == "train"),
    )

# ---------------------------------------------------------------------------
# Fixed evaluation harness — DO NOT CHANGE
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_gsm8k(model, test_loader, accelerator, n_steps_list=EVAL_STEPS) -> dict:
    """
    Exact-match accuracy on GSM8K test set at each NFE count.

    model must implement:
        model.generate(batch: dict, n_steps: int) -> list[str]

    Returns {n_steps: accuracy} aggregated across all GPUs.
    """
    model.eval()
    results = {n: {"correct": 0, "total": 0} for n in n_steps_list}

    for batch in test_loader:
        batch = {
            k: v.to(accelerator.device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }
        for n_steps in n_steps_list:
            preds = model.generate(batch, n_steps=n_steps)
            for pred, gold in zip(preds, batch["answer_str"]):
                results[n_steps]["correct"] += int(
                    normalize_answer(pred) == normalize_answer(gold)
                )
                results[n_steps]["total"] += 1

    # Aggregate across all GPU processes
    for n in n_steps_list:
        c = torch.tensor(results[n]["correct"], device=accelerator.device, dtype=torch.long)
        t = torch.tensor(results[n]["total"],   device=accelerator.device, dtype=torch.long)
        results[n] = (
            accelerator.gather(c).sum().item() /
            max(accelerator.gather(t).sum().item(), 1)
        )

    model.train()
    return results

# ---------------------------------------------------------------------------
# Main — run to verify setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoTokenizer

    BACKBONE = os.environ.get("BACKBONE", "Qwen/Qwen2.5-1.5B")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"TIME_BUDGET:     {TIME_BUDGET}s")
    print(f"BACKBONE:        {BACKBONE}")

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ex = load_gsm8k("train")
    test_ex  = load_gsm8k("test")
    print(f"GSM8K train: {len(train_ex)} examples")
    print(f"GSM8K test:  {len(test_ex)} examples")

    loader = make_dataloader("train", tok, batch_size=4, shuffle=False)
    batch  = next(iter(loader))
    print(f"problem_input_ids shape:  {batch['problem_input_ids'].shape}")
    print(f"endpoint_input_ids shape: {batch['endpoint_input_ids'].shape}")
    print(f"answer_input_ids shape:   {batch['answer_input_ids'].shape}")
    print(f"Sample answer:            {batch['answer_str'][0]!r}")
    print(f"Normalized:               {normalize_answer(batch['answer_str'][0])!r}")
    print("Done! Ready to train.")
