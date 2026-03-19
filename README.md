# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for a fixed short budget, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. In this repository, the active setup is **Flow-of-Thought on GSM8K** with a prompt-conditioned latent ODE and answer-only supervision. The core idea is still that you're not touching most Python files like you normally would as a researcher. Instead, you are programming the `program.md` instructions that define the autonomous research loop.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads/caches GSM8K), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the Flow-of-Thought model (prompt encoder, endpoint encoder, latent velocity field, decoder), optimizer, and training loop. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-10 minute time budget** (wall clock, excluding startup/compilation), controlled by `TIME_BUDGET`. The primary metric is **`val_acc@4`** on GSM8K exact match (higher is better).

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** 4 NVIDIA GPUs, Python 3.10+, and packages installed in your base environment.

```bash
# 1. Install dependencies into your base environment
pip install -e .

# 2. Download and cache GSM8K (one-time)
python prepare.py

# 3. Run one training experiment (default 5 min budget)
accelerate launch --num_processes 4 train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training runs for a fixed wall-clock budget (`TIME_BUDGET`, default 300s, max 600s). This makes experiment comparisons fair even when architecture changes.
- **Distributed by default.** The default launch uses 4 GPUs through Accelerate (`--num_processes 4`) to match the intended experiment loop in `program.md`.
- **Minimal surface area.** Data/evaluation live in `prepare.py`, and model experimentation stays in `train.py`.

## Platform support

This code targets multi-GPU NVIDIA setups. CPU/MPS portability is out of scope for this branch.

## License

MIT
