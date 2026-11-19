"""Offline binary residual-network training with training-only SMOTE."""

import argparse
import copy
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .model import Model


def training_smote(X, y, seed=42):
    counts = np.bincount(y, minlength=2)
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError(
            "SMOTE needs at least two training examples of each binary class"
        )
    k = min(5, int(counts.min()) - 1)
    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)


def metrics(y, prob):
    pred = prob >= 0.5
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
    }


def train(
    csv,
    output,
    *,
    label="Label",
    benign_label="BENIGN",
    epochs=100,
    batch_size=64,
    seed=42,
):
    out = Path(output)
    if out.exists():
        raise ValueError("choose a fresh output directory")
    if epochs < 1 or batch_size < 2:
        raise ValueError("positive epochs and batch_size >= 2 are required")
    np.random.seed(seed)
    torch.manual_seed(seed)
    frame = pd.read_csv(csv)
    frame.columns = frame.columns.str.strip()
    if label not in frame:
        raise ValueError("missing label column: " + label)
    frame = frame.dropna(subset=[label])
    y = (
        (frame.pop(label).astype(str).str.strip() != str(benign_label))
        .astype(int)
        .to_numpy()
    )
    frame = frame.drop(
        columns=[
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Source Port",
            "Destination Port",
            "Timestamp",
            "attack_cat",
            "OrigLabel",
            "Unnamed: 0",
        ],
        errors="ignore",
    )
    X = frame.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1).to_numpy()
    X = X.to_numpy(np.float32)[valid]
    y = y[valid]
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both binary classes are required")
    tr, held = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )
    va, te = train_test_split(held, test_size=0.5, stratify=y[held], random_state=seed)
    scaler = StandardScaler().fit(X[tr])
    Xtr = scaler.transform(X[tr]).astype(np.float32)
    Xva = scaler.transform(X[va]).astype(np.float32)
    Xte = scaler.transform(X[te]).astype(np.float32)
    Xbal, ybal = training_smote(Xtr, y[tr], seed)
    data = TensorDataset(
        torch.tensor(Xbal), torch.tensor(ybal, dtype=torch.float32).unsqueeze(1)
    )
    batch = min(batch_size, len(data))
    loader = DataLoader(
        data, batch_size=batch, shuffle=True, drop_last=len(data) % batch == 1
    )
    model = Model(X.shape[1])
    optimizer = torch.optim.Adagrad(model.parameters())
    loss_fn = torch.nn.BCELoss()
    best = float("inf")
    state = None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(
                loss_fn(
                    model(torch.tensor(Xva)),
                    torch.tensor(y[va], dtype=torch.float32).unsqueeze(1),
                )
            )
        if val < best:
            best = val
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(Xte)).flatten().numpy()
    result = metrics(y[te], prob)
    meta = {
        "input_dim": X.shape[1],
        "features": list(frame.columns),
        "seed": seed,
        "split_sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
        "resampled_training_rows": len(ybal),
        "smote_scope": "training partition only",
        "selection": "minimum validation BCE; test rows evaluated once after model selection",
    }
    out.mkdir(parents=True)
    torch.save(model.state_dict(), out / "model.pt")
    joblib.dump(scaler, out / "scaler.joblib")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def smoke():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, weights=[0.7, 0.3], random_state=7
    )
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "synthetic.csv"
        f = pd.DataFrame(X)
        f["Label"] = y
        f.to_csv(csv, index=False)
        result = train(csv, Path(d) / "run", benign_label="0", epochs=2, batch_size=32)
        assert 0 <= result["accuracy"] <= 1
    return {"status": "passed", "scope": "synthetic residual-network and SMOTE check"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    s.add_parser("smoke")
    t = s.add_parser("train")
    t.add_argument("--csv", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--label", default="Label")
    t.add_argument("--benign-label", default="BENIGN")
    for n, v in [("epochs", 100), ("batch-size", 64), ("seed", 42)]:
        t.add_argument("--" + n, type=int, default=v)
    a = vars(p.parse_args(argv))
    c = a.pop("command")
    try:
        r = smoke() if c == "smoke" else train(**a)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    print(json.dumps(r, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
