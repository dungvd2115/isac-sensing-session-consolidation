"""Canonical completed-value, slot-reward, and finite-horizon return primitives."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Iterable

from isac_ssc.core.entities import RequestState, SensingRequest


class UtilityValidationError(ValueError):
    """Raised when utility reconstruction receives an invalid finite-horizon quantity."""


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise UtilityValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise UtilityValidationError(f"{name} must be >= {minimum}")
    return number


def completed_request_value(completion_events: Iterable[SensingRequest]) -> float:
    """Sum value only for requests that complete successfully in the current slot."""
    requests = tuple(completion_events)
    if any(not isinstance(request, SensingRequest) for request in requests):
        raise UtilityValidationError("completion_events must contain SensingRequest values")
    return float(sum(
        request.completion_value for request in requests if request.state is RequestState.COMPLETED
    ))


def slot_reward(completed_value: float, sensing_resource_cost: float,
                sensing_resource_cost_weight: float) -> float:
    """Compute completed value minus the weighted normalized sensing-resource cost."""
    value = _finite(completed_value, "completed_value", minimum=0.0)
    cost = _finite(sensing_resource_cost, "sensing_resource_cost", minimum=0.0)
    weight = _finite(sensing_resource_cost_weight, "sensing_resource_cost_weight", minimum=0.0)
    return float(value - weight * cost)


def finite_horizon_return(slot_rewards: Iterable[float]) -> float:
    """Reconstruct the undiscounted scientific return over the fixed horizon."""
    return float(sum(_finite(reward, "slot_reward") for reward in slot_rewards))