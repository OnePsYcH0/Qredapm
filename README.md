# QREDAPM

Public code release for a REDAPM-style clinical risk prediction project with quantum-augmented fusion experiments.

This repository keeps only selected training and evaluation code. It does not include raw data, prediction files, model checkpoints, paper-writing materials, or private result packages.

## What Is Included

```text
src/
  Shared model, dataset, metric, training, and evaluation code.

experiments/01_baseline_models/
  Basic structured, text, weak baseline, and CPU tuning scripts.

experiments/02_faithful_redapm/
  Faithful REDAPM-style multimodal reproduction scripts.

experiments/03_strong_fusion/
  Structured/HGB and probability-level strong fusion scripts.

experiments/04_quantum_advantage_exp9_10_11/
  Quantum-enhanced fusion, operating-point analysis, and hard-case analysis suite.
```

## What Is Not Included

- raw EHR data
- `train_0.json` / `test_0.json`
- prediction CSV files
- trained checkpoints or pretrained weights
- manuscript drafts or paper-writing notes
- server packages
- credentials

## Notes

Paths in the scripts may need to be adjusted for a local environment. The repository is intended as a clean public code snapshot, not a full data or result release.
