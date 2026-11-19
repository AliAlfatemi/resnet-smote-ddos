# Residual Networks and SMOTE for DDoS Detection

[![CI](https://github.com/AliAlfatemi/resnet-smote-ddos/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAlfatemi/resnet-smote-ddos/actions/workflows/ci.yml)

The original binary residual-network model, packaged with training-only SMOTE,
a portable CSV command, and synthetic regression checks.

**Paper:** [Advancing DDoS Attack Detection: A Synergistic Approach Using Deep Residual Neural Networks and Synthetic Oversampling](https://arxiv.org/abs/2401.03116)

**Authors:** Ali Alfatemi, Mohamed Rahouti, Ruhul Amin, Sarah ALJamal, Kaiqi Xiong, Yufeng Xin

**Paper DOI:** https://doi.org/10.48550/arXiv.2401.03116

## Installation

```bash
git clone https://github.com/AliAlfatemi/resnet-smote-ddos.git
cd resnet-smote-ddos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m resnet_smote smoke
```

## Training

Use a prepared CSV with numeric flow features and a `Label` column. `BENIGN`
is class 0 by default; every other label is class 1. Pass `--benign-label 0`
for an already encoded 0/1 label column.

```bash
python -m resnet_smote train --csv /path/to/prepared-flows.csv \
  --output outputs/run-001 --epochs 100 --batch-size 64
```

Rows are partitioned into stratified 70/15/15 train/validation/test sets before
fitting the scaler. SMOTE sees training rows only. Its neighbor count adapts
to the minority training count; fewer than two examples per class is rejected.
Validation BCE selects a model; held-out test rows are evaluated after selection.
The CPU command uses the original sigmoid output, BCE loss, and Adagrad optimizer.

Generated weights, scaler, metadata, and metrics stay under the fresh output
directory. Input datasets, pretrained checkpoints, results, and paper files
are excluded from the repository.

## Implementation scope

`resnet_smote/model.py` retains `ResidualBlock` and `Model` from the original
research script. This is the original convolutional residual architecture,
not the later tabular attention-network variants found in working directories.
The surrounding CLI is a maintained release interface. Preprocessing is fitted
on training rows; independent test encoders and test-driven model selection
from later experimental scripts are not used. Software validation does not
re-establish the paper's published measurements.

The model treats feature values as channels with sequence length one, matching
the source implementation. Batch-normalized training uses batches of at least
two rows; a singleton final training batch is dropped. Evaluation uses all test
rows. This release is binary; it does not claim multiclass benchmark reproduction.

## Development and citation

```bash
python -m pytest -q
python -m build
```

CI tests Python 3.12/3.13 and the installed wheel using synthetic inputs.
See [CITATION.cff](CITATION.cff) for the full paper author order,
[CONTRIBUTING.md](CONTRIBUTING.md), and [LICENSE](LICENSE).
