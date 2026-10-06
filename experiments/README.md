# Experiments

This folder contains the selected public experiment code.

## 01_baseline_models

Basic reference models, including structured/text baselines and CPU-only tuning for HGB, logistic fusion, and weighted-average fusion.

## 02_faithful_redapm

Historical REDAPM-style multimodal reproduction code. The selected final cached-input implementation is in directory 05.

## 03_strong_fusion

Strong classical fusion scripts based on structured predictions, text predictions, drug-related signals, and probability-level fusion.

## 04_quantum_advantage_exp9_10_11

Historical quantum-enhanced fusion development suite:

- Experiment 9: validation-selected quantum-enhanced fusion
- Experiment 10: clinical operating-point comparison
- Experiment 11: boundary / hard-case complementarity analysis

The public repository includes code only. Required prediction CSV files and private results are not redistributed.

## 05_canonical_frozen

Selected canonical model components, quantum transformations, fusion and evaluation functions, and a CPU synthetic implementation check. See its [README](05_canonical_frozen/README.md) for dependencies and scope. No clinical data or result artifacts are included.
