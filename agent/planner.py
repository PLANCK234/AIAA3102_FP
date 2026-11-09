"""Development-only fixed initial planner."""

from agent.llm_client import LLMClient
from runtime.schemas import AgentDecision, ExperimentConfig, TaskSpec


class MockPlanner:
    def __init__(self, client: LLMClient | None = None):
        self.client = client  # Reserved interface; no LLM call is made.

    def plan(self, task: TaskSpec, metadata: dict) -> AgentDecision:
        if task.modality != "tabular":
            raise NotImplementedError("Text and image training are TODO")
        return AgentDecision(
            action="experiment",
            reason="Development-only fixed tabular baseline; no autonomous LLM decision",
            experiment=ExperimentConfig(algorithm="logistic_regression", random_seed=42),
        )
