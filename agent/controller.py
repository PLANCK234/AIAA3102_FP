"""Small explicit state machine for the development MVP."""

from enum import Enum

from agent.planner import MockPlanner
from agent.reflection import reflect
from runtime.schemas import AgentDecision, ExperimentResult, TaskSpec


class State(str, Enum):
    INITIAL = "initial"
    ANALYZED = "analyzed"
    PLANNED = "planned"
    TRAINED = "trained"
    COMPLETE = "complete"
    FAILED = "failed"


class AgentController:
    def __init__(self, planner: MockPlanner, max_experiments: int = 1):
        if max_experiments < 1:
            raise ValueError("max_experiments must be positive")
        self.planner = planner
        self.max_experiments = max_experiments
        self.experiments_run = 0
        self.state = State.INITIAL

    def plan(self, task: TaskSpec, metadata: dict) -> AgentDecision:
        if self.state != State.INITIAL:
            raise RuntimeError("Invalid controller transition")
        self.state = State.ANALYZED
        decision = self.planner.plan(task, metadata)
        self.state = State.PLANNED
        return decision

    def feedback(self, result: ExperimentResult) -> AgentDecision:
        if self.state != State.PLANNED:
            raise RuntimeError("Invalid controller transition")
        self.state = State.TRAINED
        self.experiments_run += 1
        decision = reflect(result, self.experiments_run, self.max_experiments)
        self.state = State.COMPLETE if result.status == "success" else State.FAILED
        return decision
