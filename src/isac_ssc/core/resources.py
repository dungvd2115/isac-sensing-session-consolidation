"""Deterministic update calendars and shared-resource accounting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from typing import Iterable

from isac_ssc.core.entities import CommunicationUser, EntityId, ResourceProfile, SensingSession


class ResourceValidationError(ValueError):
    """Raised when resource accounting violates hard capacity."""


def _finite(value: object, name: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ResourceValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and (number <= minimum if strict else number < minimum):
        operator = ">" if strict else ">="
        raise ResourceValidationError(f"{name} must be {operator} {minimum}")
    return number


def _slot(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceValidationError(f"{name} must be a non-negative integer slot")
    return value


def _identifier_key(value: EntityId) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _within_capacity(used: float, total: float) -> bool:
    return used <= total+1e-12*max(1.0, abs(total))


@dataclass(frozen=True, slots=True)
class SensingResourceUsage:
    sensing_bandwidth_hz: float
    sensing_power_w: float
    updating_session_ids: tuple[EntityId, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensing_bandwidth_hz", _finite(
            self.sensing_bandwidth_hz, "sensing_bandwidth_hz", minimum=0.0,
        ))
        object.__setattr__(self, "sensing_power_w", _finite(
            self.sensing_power_w, "sensing_power_w", minimum=0.0,
        ))
        identifiers = tuple(self.updating_session_ids)
        if len(set(identifiers)) != len(identifiers):
            raise ResourceValidationError("updating_session_ids must be unique")
        object.__setattr__(self, "updating_session_ids", identifiers)


@dataclass(frozen=True, slots=True)
class SlotResourceUsage:
    slot: int
    sensing_bandwidth_hz: float
    sensing_power_w: float
    updating_session_ids: tuple[EntityId, ...]


@dataclass(frozen=True, slots=True)
class ResidualCommunicationResources:
    communication_bandwidth_hz: float
    communication_power_w: float


@dataclass(frozen=True, slots=True)
class CommunicationAllocation:
    user_id: EntityId
    bandwidth_hz: float
    power_w: float


def scheduled_update_slots(session: SensingSession) -> tuple[int, ...]:
    """Return all known committed update slots for one active session."""
    if session.next_update_slot > session.final_active_slot:
        return ()
    return tuple(range(
        session.next_update_slot, session.final_active_slot+1, session.profile.update_period_slots,
    ))


def candidate_update_slots(
    current_slot: int, final_active_slot: int, profile: ResourceProfile,
) -> tuple[int, ...]:
    """Return the reset update calendar after a successful CREATE or MERGE."""
    current = _slot(current_slot, "current_slot")
    final = _slot(final_active_slot, "final_active_slot")
    if final < current:
        raise ResourceValidationError("final_active_slot must not precede current_slot")
    return tuple(range(current, final+1, profile.update_period_slots))


def next_scheduled_update_slot(
    current_slot: int, final_active_slot: int, profile: ResourceProfile,
) -> int | None:
    """Return the first periodic update after the current serviced slot, if one remains."""
    slots = candidate_update_slots(current_slot, final_active_slot, profile)
    return slots[1] if len(slots) > 1 else None


def session_updates_at(session: SensingSession, slot: int) -> bool:
    """Check whether a session consumes sensing resources in a physical slot."""
    current = _slot(slot, "slot")
    if current < session.next_update_slot or current > session.final_active_slot:
        return False
    return (current-session.next_update_slot) % session.profile.update_period_slots == 0


def sensing_resource_usage(sessions: Iterable[SensingSession], slot: int) -> SensingResourceUsage:
    """Aggregate current-slot sensing bandwidth and power for scheduled sessions."""
    current = _slot(slot, "slot")
    updating = tuple(session for session in sessions if session_updates_at(session, current))
    identifiers = tuple(sorted((session.session_id for session in updating), key=_identifier_key))
    return SensingResourceUsage(
        sum(session.profile.sensing_bandwidth_hz for session in updating),
        sum(session.profile.sensing_power_w for session in updating), identifiers,
    )


def residual_communication_resources(
    total_bandwidth_hz: float, total_power_w: float, sensing_usage: SensingResourceUsage,
) -> ResidualCommunicationResources:
    """Subtract sensing use and fail when hard capacity is exceeded."""
    total_bandwidth = _finite(total_bandwidth_hz, "total_bandwidth_hz", minimum=0.0, strict=True)
    total_power = _finite(total_power_w, "total_power_w", minimum=0.0, strict=True)
    if not _within_capacity(sensing_usage.sensing_bandwidth_hz, total_bandwidth):
        raise ResourceValidationError("committed sensing bandwidth exceeds total bandwidth")
    if not _within_capacity(sensing_usage.sensing_power_w, total_power):
        raise ResourceValidationError("committed sensing power exceeds total power")
    return ResidualCommunicationResources(
        max(0.0, total_bandwidth-sensing_usage.sensing_bandwidth_hz),
        max(0.0, total_power-sensing_usage.sensing_power_w),
    )


def equal_share_communication_resources(
    users: Iterable[CommunicationUser], residual: ResidualCommunicationResources,
) -> tuple[CommunicationAllocation, ...]:
    """Allocate residual bandwidth and power equally among active-demand users."""
    user_list = tuple(users)
    if len({user.user_id for user in user_list}) != len(user_list):
        raise ResourceValidationError("communication user identifiers must be unique")
    active_count = sum(user.demand_bit_per_s > 0.0 for user in user_list)
    bandwidth_share = residual.communication_bandwidth_hz/active_count if active_count else 0.0
    power_share = residual.communication_power_w/active_count if active_count else 0.0
    allocations = (
        CommunicationAllocation(
            user.user_id, bandwidth_share if user.demand_bit_per_s > 0.0 else 0.0,
            power_share if user.demand_bit_per_s > 0.0 else 0.0,
        )
        for user in user_list
    )
    return tuple(sorted(allocations, key=lambda item: _identifier_key(item.user_id)))


def _usage_by_slot(
    sessions: Iterable[SensingSession], start_slot: int, end_slot: int | None,
) -> tuple[SlotResourceUsage, ...]:
    session_list = tuple(sessions)
    if len({session.session_id for session in session_list}) != len(session_list):
        raise ResourceValidationError("session identifiers must be unique")
    start = _slot(start_slot, "start_slot")
    if end_slot is None:
        end = max((session.final_active_slot for session in session_list), default=start-1)
    else:
        end = _slot(end_slot, "end_slot")
        if end < start:
            raise ResourceValidationError("end_slot must not precede start_slot")
    slots = sorted({
        slot
        for session in session_list
        for slot in scheduled_update_slots(session)
        if start <= slot <= end
    })
    return tuple(
        SlotResourceUsage(
            slot, usage.sensing_bandwidth_hz, usage.sensing_power_w, usage.updating_session_ids,
        )
        for slot in slots
        for usage in (sensing_resource_usage(session_list, slot),)
    )


def committed_resource_usage(
    sessions: Iterable[SensingSession], total_bandwidth_hz: float, total_power_w: float,
    *, start_slot: int = 0, end_slot: int | None = None,
) -> tuple[SlotResourceUsage, ...]:
    """Return all known reservations and reject an already-overcommitted state."""
    total_bandwidth = _finite(total_bandwidth_hz, "total_bandwidth_hz", minimum=0.0, strict=True)
    total_power = _finite(total_power_w, "total_power_w", minimum=0.0, strict=True)
    usage = _usage_by_slot(sessions, start_slot, end_slot)
    for item in usage:
        if not _within_capacity(item.sensing_bandwidth_hz, total_bandwidth):
            raise ResourceValidationError(f"sensing bandwidth exceeds capacity in slot {item.slot}")
        if not _within_capacity(item.sensing_power_w, total_power):
            raise ResourceValidationError(f"sensing power exceeds capacity in slot {item.slot}")
    return usage


def reservation_feasible(
    sessions: Iterable[SensingSession], total_bandwidth_hz: float, total_power_w: float,
    *, start_slot: int = 0, end_slot: int | None = None,
) -> bool:
    """Check every current and future known update without reserving unknown arrivals."""
    total_bandwidth = _finite(total_bandwidth_hz, "total_bandwidth_hz", minimum=0.0, strict=True)
    total_power = _finite(total_power_w, "total_power_w", minimum=0.0, strict=True)
    return all(
        _within_capacity(item.sensing_bandwidth_hz, total_bandwidth)
        and _within_capacity(item.sensing_power_w, total_power)
        for item in _usage_by_slot(sessions, start_slot, end_slot)
    )


def profile_dominates(candidate: ResourceProfile, reference: ResourceProfile) -> bool:
    """Apply the sufficient physical partial order between joint profiles."""
    candidate_psd = candidate.sensing_power_w/candidate.sensing_bandwidth_hz
    reference_psd = reference.sensing_power_w/reference.sensing_bandwidth_hz
    candidate_range_proxy = candidate.sensing_power_w*candidate.sensing_bandwidth_hz
    reference_range_proxy = reference.sensing_power_w*reference.sensing_bandwidth_hz
    return (
        candidate_psd >= reference_psd
        and candidate_range_proxy >= reference_range_proxy
        and candidate.update_period_slots <= reference.update_period_slots
    )


def normalized_sensing_resource_cost(
    sensing_usage: SensingResourceUsage, total_bandwidth_hz: float, total_power_w: float,
    bandwidth_weight: float, power_weight: float,
) -> float:
    """Compute normalized sensing-resource cost for one slot."""
    total_bandwidth = _finite(total_bandwidth_hz, "total_bandwidth_hz", minimum=0.0, strict=True)
    total_power = _finite(total_power_w, "total_power_w", minimum=0.0, strict=True)
    bandwidth_weight = _finite(bandwidth_weight, "bandwidth_weight", minimum=0.0)
    power_weight = _finite(power_weight, "power_weight", minimum=0.0)
    if not isclose(bandwidth_weight+power_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ResourceValidationError("bandwidth_weight and power_weight must sum to one")
    if not _within_capacity(sensing_usage.sensing_bandwidth_hz, total_bandwidth):
        raise ResourceValidationError("sensing bandwidth exceeds total bandwidth")
    if not _within_capacity(sensing_usage.sensing_power_w, total_power):
        raise ResourceValidationError("sensing power exceeds total power")
    return float(
        bandwidth_weight*sensing_usage.sensing_bandwidth_hz/total_bandwidth
        + power_weight*sensing_usage.sensing_power_w/total_power
    )