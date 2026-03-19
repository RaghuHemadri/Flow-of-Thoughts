# Methodology

This document outlines the experimental methodology for the Flow-of-Thoughts project.

## Project Goal

The primary goal of this research is to develop and evaluate a latent Ordinary Differential Equation (ODE) based reasoning model. The model is trained on the GSM8K dataset using answer-only supervision, without relying on chain-of-thought prompting. The project aims to find a model configuration and training procedure that maximizes the reasoning accuracy on grade-school math word problems.

## Dataset

The experiments use the **GSM8K** dataset, which consists of 8.5K high-quality, linguistically diverse grade-school math word problems. The dataset is pre-processed and tokenized using a Qwen2.5-1.5B tokenizer via the `prepare.py` script. This script generates fixed training, validation, and test splits that are used for all experiments.

## Model Architecture

The model architecture is implemented in `train.py` and consists of five main components:

1.  **PromptEncoder**: A frozen Qwen2.5-1.5B language model backbone encodes the problem statement into a context vector `c`. A trainable projection layer maps the backbone's output to the final context vector `c ∈ ℝ^256`.
2.  **AnswerEndpoint (φ_η)**: Another frozen Qwen2.5-1.5B backbone encodes the concatenation of the problem, an EOS token, and the ground-truth answer. This produces an "answer endpoint" `z⋆ ∈ ℝ^256` in the latent space.
3.  **VelocityField (v_θ)**: An 8-layer MLP conditioned on the context vector `c` using FiLM layers. It takes the latent state `z_t` and a time embedding `t_emb` as input and outputs a velocity vector. This component learns the dynamics of the ODE.
4.  **AnswerDecoder**: An LSTM-based decoder that takes the final latent state `z₁` (the result of integrating the velocity field from t=0 to t=1) and decodes it into the final answer token sequence. The decoding is done greedily, with a maximum length of 16 tokens.
5.  **ODE Solvers**: Standard numerical ODE solvers are used to integrate the velocity field. The default solver is Euler, with Heun as an alternative. During evaluation, the number of integration steps `N` is varied between 1, 2, 4, and 8.

## Training Procedure

The model is trained end-to-end by jointly optimizing three loss functions:

1.  **`L_CFM` (Conditional Flow Matching)**: This is the primary loss for the VelocityField. It encourages the learned velocity field to match the velocity of a simple interpolated path between a random noise vector `z₀` and the target answer endpoint `z⋆`. A curriculum is applied where the weight of this loss (`λ_CFM`) is ramped up from 0 to 1 over the first 30% of training.
2.  **`L_dec` (Decoding Loss)**: A standard cross-entropy loss on the output of the AnswerDecoder, comparing the decoded tokens from `z₁` with the ground-truth answer tokens.
3.  **`L_NCE` (Contrastive Loss)**: An in-batch contrastive loss that pulls the integrated latent vector `z₁` closer to its corresponding answer endpoint `z⋆` and pushes it away from other endpoints in the batch.

An optional **reflow distillation** step can be enabled (`K_REFLOW > 0`) to further straighten the ODE paths.

Training is performed on 4 NVIDIA A100 or H100 GPUs using data-parallel distributed training, managed by the `accelerate` library. Each experiment is run for a fixed time budget, typically 300 seconds (5 minutes).

## Evaluation

The primary metric for this project is **`val_acc@4`**: the exact-match accuracy on the GSM8K test set when using the ODE solver with 4 integration steps. Accuracy is also reported for 1, 2, and 8 integration steps to assess the trade-off between performance and computational cost.

## Experiment Tracking

Experiments are managed by an autonomous agent that iteratively modifies `train.py`, runs training for a fixed budget, and logs the results. Key results, including validation accuracies, training time, peak VRAM usage, and hyperparameters, are logged to the console and appended to a central `results.tsv` file for tracking and comparison.
