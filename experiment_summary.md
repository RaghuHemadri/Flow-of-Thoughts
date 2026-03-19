# Experiment Summary

This document summarizes the experiments performed in the Flow-of-Thoughts repository.

## Methodology

Experiments were conducted by running the `train.py` script with different configurations. The primary metric for evaluation is `val_acc@4`, representing the exact-match accuracy on the GSM8K test set with 4 ODE integration steps.

## Experiment Results

The results are gathered from `results.tsv` and `run_py.log`.

| Commit ID | Description | `val_acc@1` | `val_acc@4` | Peak VRAM (GB) | Status |
|---|---|---|---|---|---|
| `a6e46f7` | baseline: default train.py, 300s, 25% data | 0.0 | 0.0 | 4.8 | Keep |
| `dd62f17` | K_REFLOW=1 | - | - | 4.8 | Crash |
| `9a6515b` | Baseline K_REFLOW=0, CURRICULUM_FRAC=0.30 | 0.0 | 0.0 | 4.8 | Keep |
| N/A | From run_py.log | 0.0 | 0.0 | 5.6 | Unknown |

## Analysis

- All completed experiments show a validation accuracy of 0.0 at all integration steps. This indicates that the model has not yet learned to solve the GSM8K problems.
- An experiment with `K_REFLOW=1` resulted in a crash, suggesting a potential issue with the reflow implementation.
- The experiments have so far focused on establishing a baseline and exploring the `K_REFLOW` and `CURRICULUM_FRAC` hyperparameters.
- The peak VRAM usage is between 4.8 GB and 5.6 GB.

## Conclusion

The initial experiments have established a baseline performance. The model is not yet learning, and further work is needed to debug the training process and find a set of hyperparameters that leads to convergence. The crash with `K_REFLOW=1` should also be investigated.
