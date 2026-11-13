"""Unified development runner. Formal execution is blocked until real autonomy exists."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from typing import Callable

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
    planner_factory: Callable | None = None,
) -> tuple[ExperimentResult, Path]:
    if mode not in ("development", "formal"):
        raise ValueError("mode must be development or formal")
    budget = BudgetManager(price_file, limit_rmb=budget_limit_rmb, formal=(mode == "formal"))
    if mode == "formal":
        raise RuntimeError(
            "Formal mode is blocked until pricing, provider accounting and complete course task coverage are verified"
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = ExperimentLogger(run_dir / "events.jsonl")
    planner_name = "injected" if planner_factory else "mock"
    logger.log("run_started", mode=mode, planner=planner_name, task=task, budget_rmb=budget.limit_rmb)
    try:
        planner = planner_factory(budget, logger) if planner_factory else MockPlanner(MockLLMClient())
        controller = AgentController(planner, max_experiments=max_experiments)
        if task.modality != "tabular":
            raise NotImplementedError("Text and image classification are TODO")
        metadata = analyze_data(task.train_path, task.target_column, task.train_labels_path, task.feature_names)
        logger.log("data_analyzed", metadata=metadata)
        decision = controller.plan(task, metadata)
        logger.log("agent_decision", decision=decision, planner=planner_name)
        last_result = None
        while decision.action == "experiment":
            if decision.experiment is None:
                raise RuntimeError("Planner did not provide an experiment")
            index = controller.experiments_run + 1
            model_file = run_dir / ("model.joblib" if index == 1 else f"model_{index:03d}.joblib")
            started = perf_counter()
            try:
                last_result = train_validate(task, decision.experiment, model_file)
            except Exception as exc:
                last_result = ExperimentResult("failed", {}, None, perf_counter() - started, str(exc))
            logger.log("experiment_result", result=last_result, selected_metric=task.metric, experiment=decision.experiment)
            decision = controller.feedback(last_result)
            logger.log("agent_decision", decision=decision, planner=planner_name)
        result = controller.best_result
        if result is None and last_result is not None and last_result.status == "failed":
            result = last_result
        if result is None:
            raise RuntimeError("No successful experiment with a valid validation metric and model")
        if result.status == "success":
            client = getattr(planner, "client", None)
            if getattr(client, "development_mode", False):
                logger.log("run_completed", model_path=result.model_path, spent_rmb=None,
                           cost_status="unpriced_development", llm_calls=client.calls_started)
            else:
                logger.log("run_completed", model_path=result.model_path, spent_rmb=budget.spent_rmb, llm_calls=len(budget.records))
        else:
            logger.log("run_failed", error_type="ExperimentFailure", error=result.error)
    except Exception as exc:
        logger.log("run_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    (run_dir / "result.json").write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, run_dir
