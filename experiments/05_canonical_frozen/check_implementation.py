"""CPU synthetic implementation check. No files, training or services are used."""
from canonical_models import verify_implementations
from model_exp7 import LATENT_NAMES

def main():
    # Arbitrary schema and random tensors, not clinical feature values or records.
    columns=[f'feature_{i}' for i in range(46)]
    groups={name:columns[g::8] for g,name in enumerate(LATENT_NAMES)}
    verify_implementations(columns,groups,461)
    print('PASS: cached architecture, matched initialization, quantum forward and gradients')

if __name__=='__main__':main()
