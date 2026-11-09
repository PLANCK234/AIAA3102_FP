"""Append-only JSONL audit trail."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime.schemas import to_jsonable


class ExperimentLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event}
        row.update(to_jsonable(fields))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
