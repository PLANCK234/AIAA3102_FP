import io
import json
from decimal import Decimal

import pytest

from agent.llm_client import LLMCallError, RealLLMClient
from runtime.budget import BudgetError, BudgetManager
from runtime.logger import ExperimentLogger


@pytest.fixture
def setup_client(tmp_path, monkeypatch):
    # Purely fictional prices used for arithmetic tests, not official pricing.
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "models:\n  Qwen-fixture:\n    official_release_date: '2026-01-01'\n"
        "    price_as_of: '2026-10-21'\n    official_source_url: 'https://example.invalid/test'\n"
        "    input_rmb_per_million_tokens: 1\n    output_rmb_per_million_tokens: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPERAGENT_API_KEY", "test-secret")
    budget = BudgetManager(prices, limit_rmb=Decimal("0.01"))
    logger = ExperimentLogger(tmp_path / "events.jsonl")
    def create(transport, bound=1000):
        return RealLLMClient("Qwen-fixture", "https://example.invalid/v1", budget, logger, bound, transport=transport)
    return create, budget, logger.path


def response(payload):
    return io.BytesIO(json.dumps(payload).encode())


def test_actual_usage_is_recorded_and_sensitive_content_omitted(setup_client):
    create, budget, path = setup_client
    calls = []
    def transport(request, timeout):
        calls.append(request)
        return response({"model": "Qwen-fixture", "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                         "choices": [{"message": {"content": "answer"}}]})
    result = create(transport).complete("private prompt", max_output_tokens=100)
    assert result.text == "answer"
    assert budget.spent_rmb == Decimal("0.0002")
    assert len(calls) == 1
    events = path.read_text(encoding="utf-8")
    assert "llm_call_started" in events and "llm_call_usage" in events and "llm_call_completed" in events
    assert "test-secret" not in events and "private prompt" not in events


def test_hkust_base_url_constructs_documented_endpoint(setup_client):
    create, budget, path = setup_client
    requests = []
    def transport(request, timeout):
        requests.append(request)
        return response({"model": "Qwen-fixture", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                         "choices": [{"message": {"content": "ok"}}]})
    client = RealLLMClient("Qwen-fixture", "https://aigc-api.hkust-gz.edu.cn/api/v2", budget,
                           ExperimentLogger(path), 1000, transport=transport)
    client.complete("test", max_output_tokens=100)
    assert requests[0].full_url == "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
    assert requests[0].get_header("Content-type") == "application/json"
    assert json.loads(requests[0].data)["model"] == "Qwen-fixture"


def test_missing_usage_blocks_further_calls(setup_client):
    create, budget, path = setup_client
    calls = []
    def transport(request, timeout):
        calls.append(1)
        return response({"choices": [{"message": {"content": "answer"}}]})
    client = create(transport)
    with pytest.raises(LLMCallError, match="usage missing"):
        client.complete("prompt", max_output_tokens=100)
    with pytest.raises(LLMCallError, match="blocked"):
        client.complete("prompt", max_output_tokens=100)
    assert len(calls) == 1 and not budget.records
    assert '"usage_status": "unknown"' in path.read_text(encoding="utf-8")


def test_preflight_rejects_before_request_and_transport_failure_is_unknown(setup_client):
    create, _, path = setup_client
    def transport(request, timeout):
        raise TimeoutError("must not expose response secrets")
    with pytest.raises(BudgetError):
        create(transport, bound=20_000).complete("prompt", max_output_tokens=100)
    assert not path.exists()
    client = create(transport)
    with pytest.raises(LLMCallError, match="usage is unknown"):
        client.complete("prompt", max_output_tokens=100)
    assert '"usage_status": "unknown"' in path.read_text(encoding="utf-8")
    with pytest.raises(LLMCallError, match="blocked"):
        client.complete("prompt", max_output_tokens=100)


def test_unpriced_development_reasoning_alias_and_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERAGENT_API_KEY", "test-secret")
    path = tmp_path / "events.jsonl"
    def transport(request, timeout):
        return response({"model": "Qwen3.8-27B-w8a8", "usage": {"prompt_tokens": 12, "completion_tokens": 34},
                         "choices": [{"finish_reason": "stop", "message": {"content": "OK", "reasoning_content": "private thoughts"}}]})
    client = RealLLMClient("Qwen3.8-27B", "https://example.invalid/v2", None, ExperimentLogger(path), None,
                           development_mode=True, max_calls=1, transport=transport)
    result = client.complete("prompt", max_output_tokens=100)
    assert result.reasoning_present and result.finish_reason == "stop"
    assert result.input_tokens == 12 and result.output_tokens == 34
    assert client.usage_records[0]["cost_rmb"] is None
    with pytest.raises(LLMCallError, match="limit"):
        client.complete("prompt", max_output_tokens=100)
    contents = path.read_text(encoding="utf-8")
    assert '"cost_rmb": null' in contents and "private thoughts" not in contents and "test-secret" not in contents


def test_unpriced_missing_usage_blocks_and_reasoning_only_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERAGENT_API_KEY", "test-secret")
    path = tmp_path / "events.jsonl"
    client = RealLLMClient("DeepSeek-V4-Pro", "https://example.invalid/v2", None, ExperimentLogger(path), None,
                           development_mode=True, max_calls=2, transport=lambda *_args, **_kw: response({"choices": []}))
    with pytest.raises(LLMCallError, match="usage missing"):
        client.complete("prompt", max_output_tokens=100)
    with pytest.raises(LLMCallError, match="blocked"):
        client.complete("prompt", max_output_tokens=100)
    assert client.calls_started == 1 and '"usage_status": "unknown"' in path.read_text(encoding="utf-8")
    other = RealLLMClient("DeepSeek-V4-Pro", "https://example.invalid/v2", None, ExperimentLogger(path), None,
                          development_mode=True, max_calls=1,
                          transport=lambda *_args, **_kw: response({"model": "DSP", "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                          "choices": [{"message": {"content": None, "reasoning_content": "hidden"}, "finish_reason": "length"}]}))
    with pytest.raises(LLMCallError, match="no content"):
        other.complete("prompt", max_output_tokens=100)
    assert other.usage_records[0]["output_tokens"] == 3


def test_direct_transport_disables_inherited_proxies(tmp_path, monkeypatch):
    import requests
    monkeypatch.setenv("SUPERAGENT_API_KEY", "test-secret")
    observed = []
    class Reply:
        status_code = 200
        content = json.dumps({"model": "DeepSeek-V4.1-Flash-1017", "usage": {"prompt_tokens": 5, "completion_tokens": 7},
                              "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
        def raise_for_status(self): pass
    class Session:
        trust_env = True
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            observed.append((self.trust_env, url, kwargs["allow_redirects"]))
            return Reply()
    monkeypatch.setattr(requests, "Session", Session)
    client = RealLLMClient("DeepSeek-V4.1-Flash", "https://example.invalid/v2", None,
                           ExperimentLogger(tmp_path / "events.jsonl"), None, development_mode=True, max_calls=1)
    assert client.complete("prompt", max_output_tokens=100).text == "ok"
    assert observed == [(False, "https://example.invalid/v2/chat/completions", False)]
