import json

import numpy as np
import pandas as pd

from models.tabular import predict_csv, predict_frame, predict_npy
from runtime.runner import run_task
from runtime.schemas import TaskSpec


def test_tabular_pipeline_and_audit(tmp_path):
    train = pd.DataFrame(
        {
            "number": list(range(40)),
            "kind": ["a", "b"] * 20,
            "label": [0] * 20 + [1] * 20,
        }
    )
    val = pd.DataFrame(
        {"number": [2, 8, 26, 35], "kind": ["a", "new", "b", "a"], "label": [0, 0, 1, 1]}
    )
    train_path, val_path = tmp_path / "train.csv", tmp_path / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    task = TaskSpec(
        "tabular", train_path, "label", "accuracy", "binary", validation_path=val_path,
        test_path=tmp_path / "deliberately_missing_test.csv",
    )
    result, run_dir = run_task(task, output_root=tmp_path / "outputs", price_file=tmp_path / "prices.yaml")
    assert result.status == "success"
    assert result.validation_metrics["accuracy"] == 1.0
    assert predict_frame(result.model_path, val).tolist() == val["label"].tolist()
    prediction_path = predict_csv(result.model_path, val_path, tmp_path / "predictions.csv")
    assert len(pd.read_csv(prediction_path)) == len(val)
    events = [json.loads(line)["event"] for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events == [
        "run_started", "data_analyzed", "agent_decision", "experiment_result", "agent_decision", "run_completed"
    ]


def test_formal_mode_fails_closed(tmp_path):
    import pytest

    task = TaskSpec("tabular", tmp_path / "unused.csv", "label", "accuracy", "binary", validation_fraction=0.2)
    price_file = tmp_path / "prices.yaml"
    price_file.write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(Exception, match="verified official"):
        run_task(task, output_root=tmp_path / "out", price_file=price_file, mode="formal")


def test_generic_paired_npy_pipeline(tmp_path):
    features = np.array([[i, "a" if i % 2 else "b"] for i in range(30)], dtype=object)
    labels = np.array([0] * 15 + [1] * 15)
    for split, indices in (("train", slice(None, 24)), ("val", slice(24, None))):
        np.save(tmp_path / f"X_{split}.npy", features[indices])
        np.save(tmp_path / f"y_{split}.npy", labels[indices])
    # Validation needs both classes for a useful score.
    np.save(tmp_path / "X_val.npy", features[[2, 5, 20, 25]])
    np.save(tmp_path / "y_val.npy", labels[[2, 5, 20, 25]])
    task = TaskSpec(
        "tabular", tmp_path / "X_train.npy", "label", "accuracy", "binary",
        validation_path=tmp_path / "X_val.npy",
        train_labels_path=tmp_path / "y_train.npy",
        validation_labels_path=tmp_path / "y_val.npy",
        feature_names=("numeric", "category"),
    )
    result, _ = run_task(task, output_root=tmp_path / "out", price_file=tmp_path / "prices.yaml")
    assert result.status == "success"
    output = predict_npy(result.model_path, tmp_path / "X_val.npy", tmp_path / "predictions.csv")
    assert len(pd.read_csv(output)) == 4
