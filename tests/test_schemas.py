from decimal import Decimal
from pathlib import Path

import pytest

from runtime.schemas import AgentDecision, ExperimentConfig, TaskSpec, UsageRecord, to_jsonable


def test_contract_examples():
    task = TaskSpec("tabular", Path("train.csv"), "label", "accuracy", "binary", validation_path=Path("val.csv"))
    config = ExperimentConfig("logistic_regression", {"C": 1.0}, random_seed=7)
    decision = AgentDecision("experiment", "Try baseline", config)
    usage = UsageRecord("Qwen-example", 100, 50, Decimal("2"), Decimal("4"), Decimal("0.0004"))
    assert to_jsonable(task)["target_column"] == "label"
    assert to_jsonable(decision)["experiment"]["random_seed"] == 7
    assert to_jsonable(usage)["cost_rmb"] == "0.0004"


def test_missing_target_or_experiment_rejected():
    with pytest.raises(ValueError):
        TaskSpec("tabular", Path("train.csv"), "", "accuracy", "binary", validation_fraction=0.2)
    with pytest.raises(ValueError):
        AgentDecision("experiment", "missing config")
    with pytest.raises(ValueError):
        TaskSpec("tabular", Path("X.npy"), "label", "accuracy", "binary", validation_fraction=0.2)
    with pytest.raises(ValueError, match="validation"):
        TaskSpec("tabular", Path("train.csv"), "label", "accuracy", "binary")


def test_usage_cost_must_match():
    with pytest.raises(ValueError):
        UsageRecord("Qwen-example", 1, 1, Decimal("1"), Decimal("1"), Decimal("0"))
