"""Shared contracts. Other packages depend on this module, never on each other."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

Modality = Literal["tabular", "text", "image"]
Metric = Literal["accuracy", "f1_macro"]
Action = Literal["experiment", "stop"]


@dataclass(frozen=True)
class TaskSpec:
    modality: Modality
    train_path: Path
    target_column: str
    metric: Metric
    task_type: Literal["binary", "multiclass"]
    validation_path: Path | None = None
    test_path: Path | None = None  # Never read during development runs.
    train_labels_path: Path | None = None  # Required for paired NPY arrays.
    validation_labels_path: Path | None = None
    test_labels_path: Path | None = None
    feature_names: tuple[str, ...] | None = None
    validation_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.modality not in ("tabular", "text", "image"):
            raise ValueError("Unsupported modality")
        if self.metric not in ("accuracy", "f1_macro"):
            raise ValueError("Unsupported metric")
        if not self.target_column:
            raise ValueError("target_column must be explicit")
        if self.validation_path is None and self.validation_fraction is None:
            raise ValueError("Explicit validation_path or validation_fraction is required")
        if self.validation_fraction is not None and not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.task_type not in ("binary", "multiclass"):
            raise ValueError("Unsupported task_type")
        if self.train_path.suffix.lower() == ".npy" and self.train_labels_path is None:
            raise ValueError("Paired NPY input requires train_labels_path")
        if self.validation_path is not None and self.validation_path.suffix.lower() == ".npy" and self.validation_labels_path is None:
            raise ValueError("Paired NPY validation requires validation_labels_path")


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str
    parameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    max_seconds: int = 300

    def __post_init__(self) -> None:
        if not self.algorithm or self.max_seconds <= 0:
            raise ValueError("Algorithm and positive max_seconds are required")


@dataclass(frozen=True)
class ExperimentResult:
    status: Literal["success", "failed"]
    validation_metrics: dict[str, float]
    model_path: Path | None
    runtime_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class AgentDecision:
    action: Action
    reason: str
    experiment: ExperimentConfig | None = None
    stop_condition: str | None = None

    def __post_init__(self) -> None:
        if self.action == "experiment" and self.experiment is None:
            raise ValueError("Experiment action requires experiment config")
        if self.action == "stop" and not self.stop_condition:
            raise ValueError("Stop action requires stop_condition")


@dataclass(frozen=True)
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    input_price_per_million_rmb: Decimal
    output_price_per_million_rmb: Decimal
    cost_rmb: Decimal

    def __post_init__(self) -> None:
        if not self.model or min(self.input_tokens, self.output_tokens) < 0:
            raise ValueError("Model and nonnegative token counts are required")
        if min(self.input_price_per_million_rmb, self.output_price_per_million_rmb) < 0:
            raise ValueError("Prices must be nonnegative")
        expected = (
            self.input_price_per_million_rmb * self.input_tokens
            + self.output_price_per_million_rmb * self.output_tokens
        ) / Decimal(1_000_000)
        if self.cost_rmb != expected:
            raise ValueError("Usage cost does not match standard token prices")


def to_jsonable(value: Any) -> Any:
    """Convert contract objects to JSON safe values for structured logs."""
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (Path, Decimal)):
        return str(value)
    return value
