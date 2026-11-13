"""General Algorithm SuperAgent MVP CLI."""

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

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
    run.add_argument("--planner", choices=("mock", "llm", "llm-dev"), default="mock", help="llm and llm-dev make billable calls")
    run.add_argument("--max-experiments", type=int)
    run.add_argument("--max-llm-calls", type=int, help="Required explicit call limit for llm-dev")
    run.add_argument("--llm-max-output-tokens", type=int, default=2048)
    probe = sub.add_parser("llm-test", help="One real unpriced development call; requires explicit enablement")
    probe.add_argument("--model", required=True)
    probe.add_argument("--max-output-tokens", type=int, default=2048)
    probe.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--settings", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    smoke.add_argument("--output-dir", type=Path)
    predict = sub.add_parser("predict", help="Run a saved model on CSV/NPY features without fitting")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "llm-test":
        from agent.llm_client import RealLLMClient
        from runtime.logger import ExperimentLogger
        url = os.environ.get("SUPERAGENT_BASE_URL", "https://aigc-api.hkust-gz.edu.cn/api/v2")
        run_dir = args.output_dir / ("llm-test-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        logger = ExperimentLogger(run_dir / "events.jsonl")
        client = RealLLMClient(args.model, url, None, logger, None, development_mode=True, max_calls=1, timeout_seconds=90)
        print(f"run_dir={run_dir}", flush=True)
        result = client.complete("Reply with exactly the word OK.", max_output_tokens=args.max_output_tokens)
        print(f"model={result.model} input_tokens={result.input_tokens} output_tokens={result.output_tokens} reasoning_present={result.reasoning_present} finish_reason={result.finish_reason}")
        print("cost_rmb=unknown; development-only")
        return 0
    if args.command == "predict":
        if args.data.suffix.lower() not in (".csv", ".npy"):
            raise ValueError("Prediction data must be CSV or NPY")
        function = predict_csv if args.data.suffix.lower() == ".csv" else predict_npy
        print(function(args.model, args.data, args.output))
        return 0
    settings = load_settings(args.settings)
    output_root = args.output_dir or settings["output_dir"]
    task = load_task(args.task) if args.command == "run" else smoke_task(output_root)
    planner_factory = None
    if args.command == "run" and args.planner in ("llm", "llm-dev"):
        from agent.llm_client import RealLLMClient
        from agent.planner import LLMPlanner

        def planner_factory(budget, logger):
            model = os.environ.get("SUPERAGENT_MODEL")
            base_url = os.environ.get("SUPERAGENT_BASE_URL")
            if not model or not base_url:
                raise ValueError("Set SUPERAGENT_MODEL and SUPERAGENT_BASE_URL")
            if args.planner == "llm-dev":
                if args.max_llm_calls is None or args.max_llm_calls <= 0:
                    raise ValueError("llm-dev requires --max-llm-calls greater than zero")
                client = RealLLMClient(model, base_url, None, logger, None, development_mode=True,
                                       max_calls=args.max_llm_calls, timeout_seconds=90)
            else:
                upper = os.environ.get("SUPERAGENT_INPUT_TOKEN_UPPER_BOUND")
                if not upper:
                    raise ValueError("Set SUPERAGENT_INPUT_TOKEN_UPPER_BOUND")
                client = RealLLMClient(model, base_url, budget, logger, int(upper))
            return LLMPlanner(client, max_output_tokens=args.llm_max_output_tokens, event_callback=logger.log)
    result, run_dir = run_task(
        task,
        output_root=output_root,
        price_file=settings["price_file"],
        mode=(args.mode or settings["mode"]) if args.command == "run" else "development",
        max_experiments=(args.max_experiments if args.max_experiments is not None else settings["max_experiments"]) if args.command == "run" else settings["max_experiments"],
        budget_limit_rmb=settings["budget_rmb"],
        planner_factory=planner_factory,
    )
    print(f"run_dir={run_dir}")
    if result.status != "success":
        print(f"experiment_failed={result.error}")
        return 1
    print(f"validation_{task.metric}={result.validation_metrics[task.metric]:.6f}")
    print(f"planner={args.planner if args.command == 'run' else 'mock'}; NOT a formal evaluation")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
