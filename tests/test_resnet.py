import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from resnet_smote.cli import metrics, train, training_smote
from resnet_smote.model import Model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)


def test_original_binary_model_shape_and_probability_range():
    model = Model(8).eval()
    p = model(torch.randn(5, 8))
    assert p.shape == (5, 1) and bool(((p >= 0) & (p <= 1)).all())


def test_smote_balances_training_classes_without_mutating_inputs():
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.array([0] * 15 + [1] * 5)
    before = X.copy()
    rx, ry = training_smote(X, y)
    assert np.bincount(ry).tolist() == [15, 15]
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(y, np.array([0] * 15 + [1] * 5))


def test_smote_rejects_singleton_class():
    with pytest.raises(ValueError, match="two training examples"):
        training_smote(np.ones((5, 2)), np.array([0, 0, 0, 0, 1]))


def test_training_uses_train_only_scaling_and_saves_reloadable_weights(tmp_path):
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, weights=[0.7, 0.3], random_state=7
    )
    csv = tmp_path / "flows.csv"
    f = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    f["Label"] = y
    f.to_csv(csv, index=False)
    output = tmp_path / "run"
    r = train(csv, output, benign_label="0", epochs=2, batch_size=32)
    assert 0 <= r["accuracy"] <= 1
    meta = json.loads((output / "metadata.json").read_text())
    assert sum(meta["split_sizes"].values()) == 200
    tr, _ = train_test_split(np.arange(200), test_size=0.3, random_state=42, stratify=y)
    scaler = joblib.load(output / "scaler.joblib")
    np.testing.assert_allclose(scaler.mean_, X[tr].mean(axis=0), rtol=1e-6, atol=1e-6)
    model = Model(6)
    model.load_state_dict(torch.load(output / "model.pt", weights_only=True))
    with pytest.raises(ValueError, match="fresh"):
        train(csv, output, epochs=1)


def test_metrics_use_binary_threshold_and_true_labels():
    r = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.8, 0.9]))
    assert r["accuracy"] == 0.75 and r["recall"] == 1
