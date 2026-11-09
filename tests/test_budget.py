from decimal import Decimal

import pytest

from runtime.budget import BudgetError, BudgetManager


def test_formal_requires_verified_price_table(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(BudgetError, match="verified official"):
        BudgetManager(path, formal=True)


def test_budget_reservation_and_actual_usage(tmp_path):
    # Deliberately fictional fixture values, never used as an official price claim.
    path = tmp_path / "prices.yaml"
    path.write_text(
        "models:\n  Qwen-fixture:\n    official_release_date: '2026-01-01'\n"
        "    price_as_of: '2026-10-21'\n    official_source_url: 'https://example.invalid/fixture'\n"
        "    input_rmb_per_million_tokens: 1\n    output_rmb_per_million_tokens: 2\n",
        encoding="utf-8",
    )
    budget = BudgetManager(path, limit_rmb=Decimal("0.01"))
    budget.ensure_capacity("Qwen-fixture", 1000, 1000)
    record = budget.record("Qwen-fixture", 1000, 500)
    assert record.cost_rmb == Decimal("0.002")
    assert budget.remaining_rmb == Decimal("0.008")
    with pytest.raises(BudgetError, match="Actual token usage"):
        budget.record("Qwen-fixture", None, 100)
    with pytest.raises(BudgetError, match="exceed"):
        budget.ensure_capacity("Qwen-fixture", 9000, 0)
    with pytest.raises(BudgetError, match="Unconfigured"):
        budget.record("DeepSeek-unknown", 1, 1)
