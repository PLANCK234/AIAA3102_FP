"""State machine for repeated experiment actions and validation-based selection."""

import math
from enum import Enum

from agent.llm_client import LLMCallLimitError
from runtime.schemas import AgentDecision, ExperimentResult, TaskSpec


class State(str, Enum):
    INITIAL = "initial"
    ANALYZED = "analyzed"
    PLANNED = "planned"
    TRAINED = "trained"
    COMPLETE = "complete"
    FAILED = "failed"


class AgentController:
    def __init__(self, planner, max_experiments: int = 1):
        if max_experiments < 1:
            raise ValueError("max_experiments must be positive")
        self.planner, self.max_experiments = planner, max_experiments
        self.experiments_run = 0
        self.state = State.INITIAL
        self.history = []
        self.best_result = None
        self._task = self._metadata = self._current = None

    def plan(self, task: TaskSpec, metadata: dict) -> AgentDecision:
        if self.state != State.INITIAL:
            raise RuntimeError("Invalid controller transition")
        self._task, self._metadata = task, metadata
        self.state = State.ANALYZED
        return self._accept(self.planner.plan(task, metadata, self.history, self.max_experiments))

    def _accept(self, decision: AgentDecision) -> AgentDecision:
        if decision.action == "experiment":
            if decision.experiment is None or self.experiments_run >= self.max_experiments:
                raise RuntimeError("Invalid experiment decision or experiment limit reached")
            self._current = decision.experiment
            self.state = State.PLANNED
        elif decision.action == "stop":
            self.state = State.COMPLETE if self.best_result is not None else State.FAILED
        else:
            raise RuntimeError("Invalid action")
        return decision

    def feedback(self, result: ExperimentResult) -> AgentDecision:
        if self.state != State.PLANNED or self._current is None or self._task is None:
            raise RuntimeError("Invalid controller transition")
        self.state = State.TRAINED
        self.experiments_run += 1
        self.history.append((self._current, result))
        self._current = None
        score = result.validation_metrics.get(self._task.metric)
        if result.status == "success" and result.model_path is not None and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
            best_score = self.best_result.validation_metrics[self._task.metric] if self.best_result else float("-inf")
            if score > best_score:
                self.best_result = result
        if self.experiments_run >= self.max_experiments:
            return self._accept(AgentDecision("stop", "Configured experiment limit reached", stop_condition="max_experiments_reached"))
        try:
            return self._accept(self.planner.plan(self._task, self._metadata, self.history, self.max_experiments))
        except LLMCallLimitError:
            if self.best_result is None:
                raise
            return self._accept(AgentDecision("stop", "Development LLM call limit reached; select best validated result",
                                              stop_condition="llm_call_limit_reached"))
