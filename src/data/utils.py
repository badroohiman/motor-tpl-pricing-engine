from __future__ import annotations

from pathlib import Path


def ensure_dir(p: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def rate(count: int, denom: int) -> float:
    """Safe rate helper: returns 0.0 when denom <= 0."""
    if denom <= 0:
        return 0.0
    return float(count) / float(denom)

