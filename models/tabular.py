"""Persisted sklearn tabular baseline and prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tools.preprocessing import make_preprocessor


def fit_baseline(features: pd.DataFrame, labels: pd.Series, seed: int, parameters: dict) -> Pipeline:
    allowed = {"C", "max_iter"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"Unsupported logistic regression parameters: {sorted(unknown)}")
    classifier = LogisticRegression(random_state=seed, max_iter=parameters.get("max_iter", 500), C=parameters.get("C", 1.0))
    pipeline = Pipeline([("preprocess", make_preprocessor(features)), ("classifier", classifier)])
    pipeline.fit(features, labels)
    return pipeline


def save_model(pipeline: Pipeline, feature_columns: list[str], target_column: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "feature_columns": feature_columns, "target_column": target_column}, path
    )


def predict_frame(model_path: Path, frame: pd.DataFrame):
    # joblib uses pickle: load only trusted model artifacts from this project.
    bundle = joblib.load(model_path)
    columns = bundle["feature_columns"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction data is missing features: {sorted(missing)}")
    return bundle["pipeline"].predict(frame[columns])


def predict_csv(model_path: Path, data_path: Path, output_path: Path) -> Path:
    frame = pd.read_csv(data_path)
    predictions = predict_frame(model_path, frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
    return output_path


def predict_npy(model_path: Path, data_path: Path, output_path: Path) -> Path:
    bundle = joblib.load(model_path)
    columns = bundle["feature_columns"]
    features = np.load(data_path, allow_pickle=True)
    if features.ndim != 2 or features.shape[1] != len(columns):
        raise ValueError("Prediction array columns do not match saved model features")
    frame = pd.DataFrame(features, columns=columns)
    for name in columns:
        converted = pd.to_numeric(frame[name], errors="coerce")
        if converted.notna().sum() == frame[name].notna().sum():
            frame[name] = converted
    predictions = bundle["pipeline"].predict(frame[columns])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
    return output_path
