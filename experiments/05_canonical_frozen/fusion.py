"""Probability/logit stacking helpers matching the canonical protocol.

The caller provides aligned predictions from independently trained, frozen bases.
No predictions, learned weights, clinical records or results are included here.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def fusion_features(probabilities):
    raw=np.asarray(probabilities,dtype=float)
    if raw.ndim!=2 or raw.shape[1]==0:
        raise ValueError('Expected records x base models')
    if not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):
        raise ValueError('Probabilities must be finite and between zero and one')
    clipped=np.clip(raw,1e-6,1-1e-6)
    return np.column_stack([raw,np.log(clipped/(1-clipped))])

def fit_meta_fusion(meta_probabilities,meta_labels,seed=42):
    """Fit only on the independent meta-fitting pool; align rows before calling."""
    features=fusion_features(meta_probabilities)
    labels=np.asarray(meta_labels)
    if labels.ndim!=1 or len(labels)!=len(features) or set(np.unique(labels))!={0,1}:
        raise ValueError('Aligned binary meta-fitting labels with both classes required')
    estimator=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=seed))
    return estimator.fit(features,labels)

def rank_validation_candidates(candidates):
    """Order caller-supplied validation metrics; does not compute test metrics."""
    return sorted(candidates,key=lambda r:(-r['validation_auc'],-r['validation_ap'],r['model']))
