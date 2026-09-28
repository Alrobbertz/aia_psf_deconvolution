from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schema import json_safe


class ResultWriter:
    def __init__(self, path: Path, run_id: str, config_hash: str) -> None:
        self.path = path
        self.run_id = run_id
        self.config_hash = config_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Refusing to reuse result log: {self.path}")
        self.path.open("x", encoding="utf-8").close()

    def append(self, row: dict[str, Any]) -> None:
        record = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            **row,
        }
        encoded = json.dumps(
            json_safe(record), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
