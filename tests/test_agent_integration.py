import json
import io

import pandas as pd
import pytest

from agent.llm_client import LLMResponse, RealLLMClient
from agent.planner import LLMPlanner
from runtime.runner import run_task
from runtime.schemas import TaskSpec


def test_multi_experiment_feedback_and_best_selection(tmp_path):
    train = pd.DataFrame({"feature": list(range(40)), "target": [0] * 20 + [1] * 20})
    val = pd.DataFrame({"feature": [3, 12, 25, 38], "target": [0, 0, 1, 1]})
    train_path, val_path = tmp_path / "train.csv", tmp_path / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    task = TaskSpec("tabular", train_path, "target", "accuracy", "binary",
                    validation_path=val_path, test_path=tmp_path / "secret_test.csv")

    class ScriptedClient:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, *, max_output_tokens):
            self.prompts.append(prompt)
            n = len(self.prompts)
            if n < 3:
                obj = {"action": "experiment", "reason": f"try {n}",
                       "experiment": {"algorithm": "logistic_regression", "parameters": {"C": n}}}
            else:
                obj = {"action": "stop", "reason": "finished", "stop_condition": "enough"}
            return LLMResponse(json.dumps(obj), "scripted", None, None)

    client = ScriptedClient()
    result, run_dir = run_task(task, output_root=tmp_path / "out", price_file=tmp_path / "missing.yaml",
                               max_experiments=3, planner_factory=lambda budget, logger: LLMPlanner(client))
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    experiments = [event["result"] for event in events if event["event"] == "experiment_result"]
    assert len(experiments) == 2
    assert result.validation_metrics["accuracy"] == max(e["validation_metrics"]["accuracy"] for e in experiments)
    assert experiments[0]["model_path"] != experiments[1]["model_path"]
    assert "validation_metrics" in client.prompts[1]
    assert "secret_test.csv" not in "".join(client.prompts)


def test_task_rejects_test_path_reused_for_development(tmp_path):
    with pytest.raises(ValueError, match="Test files"):
        TaskSpec("tabular", tmp_path / "same.csv", "target", "accuracy", "binary",
                 validation_fraction=0.2, test_path=tmp_path / "same.csv")


def test_real_adapter_with_simulated_provider_and_runner(tmp_path, monkeypatch):
    prices = tmp_path / "prices.yaml"
    # Fictional values; used only to exercise end-to-end accounting.
    prices.write_text("models:\n  Qwen-fixture:\n    official_release_date: '2026-01-01'\n"
                      "    price_as_of: '2026-10-21'\n    official_source_url: 'https://example.invalid'\n"
                      "    input_rmb_per_million_tokens: 1\n    output_rmb_per_million_tokens: 2\n", encoding="utf-8")
    monkeypatch.setenv("SUPERAGENT_API_KEY", "test-secret")
    frame = pd.DataFrame({"x": range(30), "y": [0] * 15 + [1] * 15})
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    task = TaskSpec("tabular", path, "y", "accuracy", "binary", validation_fraction=0.25)
    calls = []

    def transport(request, timeout):
        calls.append(1)
        decision = {"action": "experiment", "reason": "try", "experiment": {
            "algorithm": "logistic_regression", "parameters": {"C": len(calls)}}}
        payload = {"model": "Qwen-fixture", "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                   "choices": [{"message": {"content": json.dumps(decision)}}]}
        return io.BytesIO(json.dumps(payload).encode())

    def factory(budget, logger):
        client = RealLLMClient("Qwen-fixture", "https://example.invalid/v1", budget, logger, 500,
                               transport=transport)
        return LLMPlanner(client, event_callback=logger.log)

    result, run_dir = run_task(task, output_root=tmp_path / "out", price_file=prices,
                               max_experiments=2, planner_factory=factory)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result.status == "success" and len(calls) == 2
    assert len([e for e in events if e["event"] == "llm_call_usage"]) == 2
    assert events[-1]["llm_calls"] == 2 and events[-1]["spent_rmb"] == "0.0004"
    assert "test-secret" not in (run_dir / "events.jsonl").read_text(encoding="utf-8")


def test_runner_rejects_success_without_validation_score(tmp_path, monkeypatch):
    from runtime.schemas import ExperimentResult
    import runtime.runner as runner

    path = tmp_path / "train.csv"
    pd.DataFrame({"x": range(20), "y": [0] * 10 + [1] * 10}).to_csv(path, index=False)
    task = TaskSpec("tabular", path, "y", "accuracy", "binary", validation_fraction=0.2)
    monkeypatch.setattr(runner, "train_validate", lambda *_: ExperimentResult("success", {}, tmp_path / "model", 0.1))
    with pytest.raises(RuntimeError, match="valid validation metric"):
        run_task(task, output_root=tmp_path / "out", price_file=tmp_path / "missing.yaml")


def test_unpriced_runner_selects_best_at_call_limit_after_invalid_reply(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERAGENT_API_KEY", "test-secret")
    frame = pd.DataFrame({"x": range(30), "y": [0] * 15 + [1] * 15})
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    task = TaskSpec("tabular", path, "y", "accuracy", "binary", validation_fraction=0.25)
    responses = iter([
        json.dumps({"action": "experiment", "reason": "baseline", "experiment": {"algorithm": "logistic_regression"}}),
        "truncated non-json",
    ])
    def transport(request, timeout):
        payload = {"model": "Qwen3.8-27B", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                   "choices": [{"message": {"content": next(responses)}}]}
        return io.BytesIO(json.dumps(payload).encode())
    def factory(budget, logger):
        client = RealLLMClient("Qwen3.8-27B", "https://example.invalid/v2", None, logger, None,
                               development_mode=True, max_calls=2, transport=transport)
        return LLMPlanner(client, event_callback=logger.log)
    result, run_dir = run_task(task, output_root=tmp_path / "out", price_file=tmp_path / "missing.yaml",
                               max_experiments=3, planner_factory=factory)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result.status == "success"
    assert any(e["event"] == "agent_decision" and e["decision"]["stop_condition"] == "llm_call_limit_reached" for e in events)
    assert events[-1]["llm_calls"] == 2 and events[-1]["spent_rmb"] is None
