# QREDAPM

Public code for **Classical Fusion and Quantum Pathways for Multimodal EHR Prediction of Depression and Anxiety**.

Starting from REDAPM, the project develops a classical multimodal fusion framework and investigates a quantum representation pathway with matched classical controls. The final implementation is in [experiments/05_canonical_frozen](experiments/05_canonical_frozen/README.md).

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

## Final implementation

The canonical directory includes the recovered REDAPM architecture, grouped compressor, classical and quantum representation branches, multimodal models, probability-level fusion, evaluation functions, dependency versions, and an in-memory synthetic implementation check. Its README explains how to run the check without clinical records or model weights.

The October 2026 publication update preserves the frozen canonical model implementations. It also restores UTF-8 strings in four historical baseline scripts; their model logic and training settings are retained. No models were retrained and no clinical results were recomputed for this publication update.
