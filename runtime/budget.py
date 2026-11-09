"""Official standard-price accounting, without cache discounts."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml

from runtime.schemas import UsageRecord

CUTOFF = date(2026, 10, 21)


class BudgetError(RuntimeError):
    pass


class BudgetManager:
    def __init__(self, price_file: Path, limit_rmb: Decimal = Decimal("20"), formal: bool = False):
        self.limit_rmb = Decimal(str(limit_rmb))
        if self.limit_rmb <= 0 or self.limit_rmb > Decimal("20"):
            raise BudgetError("Per-task budget must be within (0, 20] RMB")
        self.formal = formal
        self.records: list[UsageRecord] = []
        if not price_file.is_file():
            if formal:
                raise BudgetError(f"Missing formal price file: {price_file}")
            self.prices = {}
            return
        data = yaml.safe_load(price_file.read_text(encoding="utf-8")) or {}
        self.prices = data.get("models") or {}
        if formal and not self.prices:
            raise BudgetError("Formal mode needs verified official model prices")
        if formal:
            for name in self.prices:
                self._validated_prices(name)

    def _validated_prices(self, model: str) -> tuple[Decimal, Decimal]:
        entry = self.prices.get(model)
        if not isinstance(entry, dict):
            raise BudgetError(f"Unconfigured model: {model}")
        try:
            released = date.fromisoformat(str(entry["official_release_date"]))
            as_of = date.fromisoformat(str(entry["price_as_of"]))
            input_price = Decimal(str(entry["input_rmb_per_million_tokens"]))
            output_price = Decimal(str(entry["output_rmb_per_million_tokens"]))
            source = str(entry["official_source_url"])
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise BudgetError(f"Incomplete price evidence for {model}") from exc
        if not model.lower().startswith(("qwen", "deepseek")):
            raise BudgetError("Only Qwen or DeepSeek models are permitted")
        if released > CUTOFF or as_of != CUTOFF or not source.startswith("https://"):
            raise BudgetError(f"Model release or official price date invalid: {model}")
        if input_price < 0 or output_price < 0:
            raise BudgetError("Negative price")
        return input_price, output_price

    @property
    def spent_rmb(self) -> Decimal:
        return sum((x.cost_rmb for x in self.records), Decimal("0"))

    @property
    def remaining_rmb(self) -> Decimal:
        return self.limit_rmb - self.spent_rmb

    def ensure_capacity(self, model: str, input_tokens: int, max_output_tokens: int) -> None:
        """Reserve a worst-case call amount before sending a billable request."""
        if input_tokens < 0 or max_output_tokens < 0:
            raise BudgetError("Token counts cannot be negative")
        ip, op = self._validated_prices(model)
        reserved = (ip * input_tokens + op * max_output_tokens) / Decimal(1_000_000)
        if reserved > self.remaining_rmb:
            raise BudgetError("LLM call would exceed the task budget")

    def record(self, model: str, input_tokens: int | None, output_tokens: int | None) -> UsageRecord:
        if input_tokens is None or output_tokens is None:
            raise BudgetError("Actual token usage is required; zero cost cannot be assumed")
        if input_tokens < 0 or output_tokens < 0:
            raise BudgetError("Token counts cannot be negative")
        ip, op = self._validated_prices(model)
        cost = (ip * input_tokens + op * output_tokens) / Decimal(1_000_000)
        if cost > self.remaining_rmb:
            raise BudgetError("LLM usage exceeded the task budget")
        record = UsageRecord(model, input_tokens, output_tokens, ip, op, cost)
        self.records.append(record)
        return record
