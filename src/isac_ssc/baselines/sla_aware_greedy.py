"""Deterministic current-state SLA-margin baseline."""

from __future__ import annotations

from typing import Iterable

from isac_ssc.baselines.selectors import (
    ImmediateServiceCandidate, canonical_action_key, fallback_action,
)
from isac_ssc.envs.action_space import EnvironmentAction


def select_action(
    candidates: Iterable[ImmediateServiceCandidate], *, defer_feasible: bool,
) -> EnvironmentAction:
    values = tuple(candidates)
    if not values:
        return fallback_action(defer_feasible)
    return min(values, key=lambda item: (
        -item.normalized_sla_margin,
        item.incremental_current_cost,
        canonical_action_key(item.action),
    )).action