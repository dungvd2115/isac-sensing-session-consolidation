"""Deterministic no-consolidation baseline."""

from __future__ import annotations

from typing import Iterable

from isac_ssc.baselines.selectors import (
    ImmediateServiceCandidate, canonical_action_key, fallback_action,
)
from isac_ssc.envs.action_space import ActionType, EnvironmentAction


def select_action(
    candidates: Iterable[ImmediateServiceCandidate], *, defer_feasible: bool,
) -> EnvironmentAction:
    creates = tuple(
        item for item in candidates
        if item.action.action_type is ActionType.CREATE
    )
    if not creates:
        return fallback_action(defer_feasible)
    return min(creates, key=lambda item: (
        item.absolute_profile_cost, canonical_action_key(item.action),
    )).action