"""Development-only termination hook."""

from runtime.schemas import AgentDecision, ExperimentResult


def reflect(result: ExperimentResult, experiments_run: int, max_experiments: int) -> AgentDecision:
    # TODO: replace with feedback-driven autonomous experiment selection.
    return AgentDecision(
        action="stop",
        reason="Mock controller has no feedback-driven search",
        stop_condition=(
            "failed_experiment" if result.status == "failed" else
            "max_experiments_reached" if experiments_run >= max_experiments else "mock_baseline_complete"
        ),
    )
