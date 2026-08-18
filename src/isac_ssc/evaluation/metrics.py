"""Small reusable helpers for deterministic environment evaluation."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Iterable

from isac_ssc.core.entities import RequestState


def safe_ratio(numerator: Real, denominator: Real) -> float | None:
    """Return ``numerator / denominator`` or ``None`` for a zero denominator."""
    left, right = float(numerator), float(denominator)
    if not isfinite(left) or not isfinite(right) or right < 0.0:
        raise ValueError("ratio inputs must be finite and the denominator non-negative")
    return None if right == 0.0 else left/right


def request_state_counts(requests: Iterable[object]) -> tuple[tuple[str, int], ...]:
    """Count requests in the canonical lifecycle order."""
    counts = {state: 0 for state in RequestState}
    for request in requests:
        counts[request.state] += 1
    return tuple((state.value, counts[state]) for state in RequestState)