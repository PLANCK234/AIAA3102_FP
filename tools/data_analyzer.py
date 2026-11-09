"""CSV and paired NPY tabular inspection. Target is always explicit."""

from pathlib import Path

import pandas as pd
import numpy as np


def read_labeled_csv(path: Path, target_column: str) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ValueError("Expected a CSV feature/label file")
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(f"Explicit target column {target_column!r} is missing in {path}")
    if frame.empty or frame[target_column].isna().any():
        raise ValueError("Empty data or missing labels cannot be trained")
    return frame


def analyze_csv(path: Path, target_column: str) -> dict:
    return _summary(read_labeled_csv(path, target_column), target_column)


def read_labeled_data(
    path: Path, target_column: str, labels_path: Path | None = None, feature_names: tuple[str, ...] | None = None
) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return read_labeled_csv(path, target_column)
    if path.suffix.lower() != ".npy":
        raise NotImplementedError("Tabular MVP accepts CSV or paired NPY arrays")
    if labels_path is None:
        raise ValueError("NPY features require a separate label path")
    # The provided arrays contain Python objects; only load trusted local task data.
    features = np.load(path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=False)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Feature and label arrays have incompatible shapes")
    if feature_names is not None and len(feature_names) != features.shape[1]:
        raise ValueError("feature_names length does not match NPY columns")
    names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=names)
    for name in names:
        converted = pd.to_numeric(frame[name], errors="coerce")
        if converted.notna().sum() == frame[name].notna().sum():
            frame[name] = converted
    frame[target_column] = labels
    if frame.empty or frame[target_column].isna().any():
        raise ValueError("Empty data or missing labels cannot be trained")
    return frame


def analyze_data(
    path: Path, target_column: str, labels_path: Path | None = None, feature_names: tuple[str, ...] | None = None
) -> dict:
    return _summary(read_labeled_data(path, target_column, labels_path, feature_names), target_column)


def _summary(frame: pd.DataFrame, target_column: str) -> dict:
    labels = frame[target_column].value_counts(dropna=False)
    if len(labels) < 2:
        raise ValueError("Classification needs at least two classes")
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "feature_columns": [x for x in frame.columns if x != target_column],
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "label_counts": {str(key): int(value) for key, value in labels.items()},
        "class_count": len(labels),
    }
