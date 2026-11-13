"""Agent decision flow using an in-process scripted client, with no model calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.controller import AgentController, State
from agent.planner import DecisionError, LLMPlanner, parse_decision
from runtime.schemas import ExperimentResult, TaskSpec


def experiment(c):
    return json.dumps({"action": "experiment", "reason": "Try regularization", "experiment": {
        "algorithm": "logistic_regression", "parameters": {"C": c}, "random_seed": 42, "max_seconds": 30
    }})


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def complete(self, prompt, *, max_output_tokens):
        self.prompts.append(prompt)
        return SimpleNamespace(text=next(self.replies))


def task():
    return TaskSpec("tabular", Path("private/train.csv"), "label", "accuracy", "binary",
                    validation_path=Path("private/val.csv"), test_path=Path("private/test.csv"))


def test_feedback_drives_second_experiment_and_selects_best():
    client = ScriptedClient([experiment(1), experiment(0.1)])
    controller = AgentController(LLMPlanner(client), max_experiments=2)
    assert controller.plan(task(), {"rows": 100}).experiment.parameters["C"] == 1
    first = ExperimentResult("success", {"accuracy": 0.8}, Path("first.joblib"), 1)
    decision = controller.feedback(first)
    assert decision.experiment.parameters["C"] == 0.1
    assert '"accuracy": 0.8' in client.prompts[1]
    assert "train.csv" not in client.prompts[1] and "test.csv" not in client.prompts[1]
    second = ExperimentResult("success", {"accuracy": 0.7}, Path("second.joblib"), 1)
    assert controller.feedback(second).stop_condition == "max_experiments_reached"
    assert controller.best_result == first and controller.state == State.COMPLETE
    assert len(client.prompts) == 2


def test_invalid_action_retry_and_failed_experiment_feedback():
    client = ScriptedClient(['{"action":"delete"}', experiment(1), experiment(0.5)])
    events = []
    controller = AgentController(LLMPlanner(client, max_retries=1, event_callback=lambda event, **fields: events.append((event, fields))), max_experiments=2)
    controller.plan(task(), {"rows": 20})
    assert "Previous response was rejected" in client.prompts[1]
    assert events[0][0] == "planner_rejected" and events[0][1]["attempt"] == 1
    next_decision = controller.feedback(ExperimentResult("failed", {}, None, 1, "bad input private/train.csv"))
    assert next_decision.action == "experiment"
    assert "bad input" in client.prompts[2] and "train.csv" not in client.prompts[2]


@pytest.mark.parametrize("raw", [
    '{"action":"experiment","reason":"x","experiment":{"algorithm":"unknown"}}',
    '{"action":"experiment","reason":"x","experiment":{"algorithm":"logistic_regression","parameters":{"C":NaN}}}',
    '{"action":"experiment","reason":"x","experiment":{"algorithm":"logistic_regression","parameters":{"C":-1}}}',
    '{"action":"stop","reason":"x"}',
    '{"action":"stop","action":"experiment","reason":"x","experiment":{"algorithm":"logistic_regression"}}',
    '{"action":"experiment","reason":"x","experiment":{"algorithm":"logistic_regression","parameters":{"max_iter":true}}}',
])
def test_invalid_tool_actions_rejected(raw):
    with pytest.raises(DecisionError):
        parse_decision(raw)


def test_stop_before_any_success_and_missing_metric_are_not_selected():
    stop = json.dumps({"action": "stop", "reason": "Cannot proceed", "stop_condition": "no_solution"})
    controller = AgentController(LLMPlanner(ScriptedClient([experiment(1), stop])), max_experiments=3)
    controller.plan(task(), {"rows": 20})
    decision = controller.feedback(ExperimentResult("success", {"f1_macro": 0.8}, Path("model.joblib"), 1))
    assert decision.action == "stop"
    assert controller.best_result is None and controller.state == State.FAILED


def test_local_call_limit_stops_with_best_but_unknown_usage_does_not():
    from agent.llm_client import LLMCallError, LLMCallLimitError

    class LimitedClient:
        def __init__(self, exception):
            self.calls = 0
            self.exception = exception
        def complete(self, prompt, *, max_output_tokens):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(text=experiment(1))
            raise self.exception

    controller = AgentController(LLMPlanner(LimitedClient(LLMCallLimitError("limit"))), max_experiments=3)
    controller.plan(task(), {"rows": 20})
    good = ExperimentResult("success", {"accuracy": 0.8}, Path("model.joblib"), 1)
    assert controller.feedback(good).stop_condition == "llm_call_limit_reached"
    assert controller.best_result == good
    unresolved = AgentController(LLMPlanner(LimitedClient(LLMCallError("usage unknown"))), max_experiments=3)
    unresolved.plan(task(), {"rows": 20})
    with pytest.raises(LLMCallError, match="unknown"):
        unresolved.feedback(good)


def test_runner_recovers_from_invalid_action_and_failed_tool(tmp_path, monkeypatch):
    """Scripted LLM and a deliberately failed tool exercise the real runner offline."""
    import pandas as pd
    import runtime.runner as runner

    frame = pd.DataFrame({"x": list(range(40)), "label": [0] * 20 + [1] * 20})
    train, val = tmp_path / "train.csv", tmp_path / "val.csv"
    frame.to_csv(train, index=False)
    frame.iloc[[1, 5, 24, 30]].to_csv(val, index=False)
    spec = TaskSpec("tabular", train, "label", "accuracy", "binary", validation_path=val,
                    test_path=tmp_path / "never_read.csv")
    stop = json.dumps({"action": "stop", "reason": "Useful validation result", "stop_condition": "sufficient"})
    client = ScriptedClient([experiment(-1), experiment(1), experiment(0.5), stop])
    real_train = runner.train_validate
    tried = []

    def flaky_tool(task_spec, config, model_path):
        tried.append(config.parameters["C"])
        if len(tried) == 1:
            raise ValueError("synthetic tool failure")
        return real_train(task_spec, config, model_path)

    monkeypatch.setattr(runner, "train_validate", flaky_tool)
    result, run_dir = runner.run_task(
        spec, output_root=tmp_path / "runs", price_file=tmp_path / "no-prices.yaml", max_experiments=3,
        planner_factory=lambda budget, logger: LLMPlanner(client, max_retries=1, event_callback=logger.log),
    )
    assert tried == [1, 0.5] and result.status == "success" and result.model_path.is_file()
    assert "C must be a finite number" in client.prompts[1]
    assert "synthetic tool failure" in client.prompts[2]
    assert '"validation_metrics"' in client.prompts[3]
    logs = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["event"] for entry in logs].count("planner_rejected") == 1
    experiments = [entry for entry in logs if entry["event"] == "experiment_result"]
    assert [entry["result"]["status"] for entry in experiments] == ["failed", "success"]
    assert logs[-1]["event"] == "run_completed"
