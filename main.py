"""General Algorithm SuperAgent MVP CLI."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models.tabular import predict_csv, predict_npy
from runtime.runner import load_settings, load_task, run_task
from runtime.schemas import TaskSpec

PROJECT_ROOT = Path(__file__).resolve().parent


def smoke_task(output_root: Path) -> TaskSpec:
    data_dir = output_root / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    features, labels = make_classification(
        n_samples=180, n_features=4, n_informative=3, n_redundant=0, random_state=42
    )
    frame = pd.DataFrame(features, columns=[f"numeric_{i}" for i in range(4)])
    frame["category"] = np.where(frame["numeric_0"] > 0, "high", "low")
    frame.loc[::17, "category"] = None
    frame["label"] = labels
    train, val = train_test_split(frame, test_size=0.25, random_state=42, stratify=labels)
    train_path, val_path = data_dir / "train.csv", data_dir / "val.csv"
    train.to_csv(train_path, index=False, quoting=csv.QUOTE_MINIMAL)
    val.to_csv(val_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return TaskSpec("tabular", train_path, "label", "accuracy", "binary", validation_path=val_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Development MVP; mock planner results are not course evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--mode", choices=("development", "formal"))
    run.add_argument("--settings", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    run.add_argument("--output-dir", type=Path)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--settings", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    smoke.add_argument("--output-dir", type=Path)
    predict = sub.add_parser("predict", help="Run a saved model on CSV/NPY features without fitting")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        if args.data.suffix.lower() not in (".csv", ".npy"):
            raise ValueError("Prediction data must be CSV or NPY")
        function = predict_csv if args.data.suffix.lower() == ".csv" else predict_npy
        print(function(args.model, args.data, args.output))
        return 0
    settings = load_settings(args.settings)
    output_root = args.output_dir or settings["output_dir"]
    task = load_task(args.task) if args.command == "run" else smoke_task(output_root)
    result, run_dir = run_task(
        task,
        output_root=output_root,
        price_file=settings["price_file"],
        mode=(args.mode or settings["mode"]) if args.command == "run" else "development",
        max_experiments=settings["max_experiments"],
        budget_limit_rmb=settings["budget_rmb"],
    )
    print(f"run_dir={run_dir}")
    if result.status != "success":
        print(f"experiment_failed={result.error}")
        return 1
    print(f"validation_{task.metric}={result.validation_metrics[task.metric]:.6f}")
    print("planner=mock; NOT a formal autonomous evaluation")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
