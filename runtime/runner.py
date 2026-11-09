"""Unified development runner. Formal execution is blocked until real autonomy exists."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import yaml

from agent.controller import AgentController
from agent.llm_client import MockLLMClient
from agent.planner import MockPlanner
from runtime.budget import BudgetManager
from runtime.logger import ExperimentLogger
from runtime.schemas import ExperimentResult, TaskSpec, to_jsonable
from tools.data_analyzer import analyze_data
from tools.training import train_validate


def load_task(path: Path) -> TaskSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Task file must contain a mapping")
    root = path.resolve().parent
    for key in (
        "train_path", "validation_path", "test_path", "train_labels_path", "validation_labels_path", "test_labels_path"
    ):
        if data.get(key) is not None:
            raw = Path(data[key])
            data[key] = raw if raw.is_absolute() else (root / raw).resolve()
    if data.get("feature_names") is not None:
        data["feature_names"] = tuple(data["feature_names"])
    return TaskSpec(**data)


def load_settings(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Settings file must contain a mapping")
    root = path.resolve().parent
    for key in ("output_dir", "price_file"):
        raw = Path(data[key])
        data[key] = raw if raw.is_absolute() else (root / raw).resolve()
    data["budget_rmb"] = Decimal(str(data["budget_rmb"]))
    data["max_experiments"] = int(data["max_experiments"])
    return data


def run_task(
    task: TaskSpec,
    *,
    output_root: Path,
    price_file: Path,
    mode: str = "development",
    max_experiments: int = 1,
    budget_limit_rmb: Decimal = Decimal("20"),
) -> tuple[ExperimentResult, Path]:
    if mode not in ("development", "formal"):
        raise ValueError("mode must be development or formal")
    budget = BudgetManager(price_file, limit_rmb=budget_limit_rmb, formal=(mode == "formal"))
    if mode == "formal":
        raise RuntimeError(
            "Formal mode is blocked: real LLM adapter, autonomous planning, feedback search and usage audit are TODO"
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = ExperimentLogger(run_dir / "events.jsonl")
    logger.log("run_started", mode=mode, planner="mock", task=task, budget_rmb=budget.limit_rmb)
    controller = AgentController(MockPlanner(MockLLMClient()), max_experiments=max_experiments)
    try:
        if task.modality != "tabular":
            raise NotImplementedError("Text and image classification are TODO")
        metadata = analyze_data(task.train_path, task.target_column, task.train_labels_path, task.feature_names)
        logger.log("data_analyzed", metadata=metadata)
        decision = controller.plan(task, metadata)
        logger.log("agent_decision", decision=decision, planner="mock")
        if decision.experiment is None:
            raise RuntimeError("Planner did not provide an experiment")
        started = perf_counter()
        try:
            result = train_validate(task, decision.experiment, run_dir / "model.joblib")
        except Exception as exc:
            result = ExperimentResult("failed", {}, None, perf_counter() - started, str(exc))
        logger.log("experiment_result", result=result, selected_metric=task.metric)
        stop = controller.feedback(result)
        logger.log("agent_decision", decision=stop, planner="mock")
        if result.status == "success":
            logger.log("run_completed", model_path=result.model_path, spent_rmb=budget.spent_rmb, llm_calls=0)
        else:
            logger.log("run_failed", error_type="ExperimentFailure", error=result.error)
    except Exception as exc:
        logger.log("run_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    (run_dir / "result.json").write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, run_dir
