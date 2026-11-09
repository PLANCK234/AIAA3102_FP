"""Validation metrics only; test evaluation is outside the development runner."""

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(truth, predictions) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
    }
