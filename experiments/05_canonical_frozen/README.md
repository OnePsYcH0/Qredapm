# Selected canonical implementation

This directory contains selected implementation code from the final canonical protocol. It is a **code-only release**, not a complete experimental artifact or dataset release. The earlier numbered directories retain historical development code; this directory documents the canonical model components.

## Included code

| File | Purpose |
| --- | --- |
| `model_redapm.py`, `layers.py` | Recovered REDAPM structure, including residual layers, normalization, pooling and Transformer settings |
| `model_exp7.py` | Recovered visit/SN3 architecture and clinical-branch variants |
| `quantum_components.py` | Exact canonical dependency definitions for the grouped compressor, differentiable statevector circuit and compact head |
| `canonical_models.py` | Frozen-encoder cached-input models and matched classical/quantum substitutions |
| `text_processing.py` | Document and visit text construction; no example patient records |
| `fusion.py` | Probability/logit features, independent meta-fit logistic fusion, validation ranking rule |
| `evaluation.py` | Threshold selection, ROC-AUC/AP, weighted bootstrap summaries and calibration functions |
| `check_implementation.py` | In-memory synthetic implementation check |

The core model class bodies are preserved from the executed implementation. Project-specific imports have been made local, and the implementation check returns diagnostics in memory instead of writing an audit file. Fusion helpers expose the existing transformations and estimator settings without the original project orchestration. Historical training entry points and their local filesystem dependencies are intentionally omitted.

## Synthetic implementation check

Use Python 3.10 and install the listed dependencies in a separate environment:

```bash
python -m pip install -r experiments/05_canonical_frozen/requirements.txt
python -B experiments/05_canonical_frozen/check_implementation.py
```

This CPU check generates arbitrary tensors in memory and compares cached forwards with the recovered architectures. It also compares quantum outputs and input/weight gradients against a PennyLane reference and checks matched initialization. It does not read or create clinical files, fit a model, load pretrained weights, access a quantum service, or compute paper results. Its pass message concerns implementation consistency only.

## Use with authorized inputs

`CanonicalSN3` consumes caller-supplied visit embeddings and masks, structured features and medication vectors. Its encoder placeholder avoids loading BERT when working with already computed embeddings. The selected reference uses a staged, frozen-encoder schedule rather than joint end-to-end BERT retraining.

`fit_meta_fusion` must receive predictions from frozen base models on an independent meta-fitting pool. Align records and labels before calling it. Base training, fusion fitting and validation selection must use their respective partitions. `choose_threshold` implements the validation-F1 or target-sensitivity rule; the target-sensitivity rule takes the highest ROC threshold attaining at least the requested sensitivity. Predictions at the threshold use `p >= threshold`.

AP means Average Precision, not trapezoidal PR-AUC. Bootstrap utilities require caller-supplied cluster identifiers and aligned arrays; this source release does not supply those arrays or identifiers.

## Access boundary

Training data require approval from the data provider. This repository does not grant access to clinical data or permission to redistribute them. No training/test records, identifiers, record-index mappings, prediction files, result tables, registries, numerical audit outputs, checkpoints, embeddings or clinical caches are included. Full paper-result reproduction requires separately approved inputs and artifacts; the selected public code alone cannot regenerate the clinical benchmark.

Original architecture provenance: Feng et al., *Deep learning based prediction of depression and anxiety in patients with type 2 diabetes mellitus using regional electronic health records*, International Journal of Medical Informatics, 2025. Source availability does not change third-party software or dataset rights.
