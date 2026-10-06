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
  Historical quantum-enhanced fusion and analysis suite.

experiments/05_canonical_frozen/
  Selected canonical model, fusion and evaluation functions, with a synthetic implementation check.
```

## What Is Not Included

- raw EHR data
- `train_0.json` / `test_0.json`
- prediction CSV files
- result tables, registries, numerical audit outputs or split-index mappings
- trained checkpoints or pretrained weights
- manuscript drafts or paper-writing notes
- server packages
- credentials

## Notes

Start with [the selected canonical implementation](experiments/05_canonical_frozen/README.md) for the final model components and a runnable synthetic check. Directories 01–04 preserve historical development scripts; their folder names do not establish quantum advantage or define the final evaluation protocol.

Paths in historical scripts may need to be adjusted for a local environment. Clinical training data require approval from the data provider. This selected source release does not supply the inputs or artifacts needed to regenerate the paper results.
