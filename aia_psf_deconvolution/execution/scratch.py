from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def clear_scratch(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


@contextmanager
def ephemeral_fits(root: Path, stem: str):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stem}_{uuid4().hex[:8]}.fits"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
