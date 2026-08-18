"""Deterministic static compatibility-merge baseline."""

from __future__ import annotations

from typing import Iterable

from isac_ssc.baselines.selectors import (
    ImmediateServiceCandidate, canonical_action_key, fallback_action,
)
from isac_ssc.envs.action_space import ActionType, EnvironmentAction


def _minimum_profile_cost(
    candidates: tuple[ImmediateServiceCandidate, ...],
) -> EnvironmentAction:
    return min(candidates, key=lambda item: (
        item.absolute_profile_cost, canonical_action_key(item.action),
    )).action


def select_action(
    candidates: Iterable[ImmediateServiceCandidate], *, defer_feasible: bool,
) -> EnvironmentAction:
    values = tuple(candidates)
    merges = tuple(
        item for item in values
        if item.action.action_type is ActionType.MERGE
    )
    if merges:
        return _minimum_profile_cost(merges)
    creates = tuple(
        item for item in values
        if item.action.action_type is ActionType.CREATE
    )
    return _minimum_profile_cost(creates) if creates else fallback_action(defer_feasible)