"""Request-level sensing-SLA and communication-QoS accounting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterable

from isac_ssc.core.entities import EntityId, RequestState, SensingRequest, TaskDurationMap


class SlaValidationError(ValueError):
    """Raised when SLA or communication accounting is outside its domain."""


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise SlaValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise SlaValidationError(f"{name} must be >= {minimum}")
    return number


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SlaValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class RequestSlaUpdate:
    updated_request: SensingRequest
    valid_output_event: bool
    first_violation_event: bool
    completed_event: bool
    failed_event: bool


@dataclass(frozen=True, slots=True)
class TenantSlaSummary:
    tenant_id: EntityId
    arrived_count: int
    accepted_count: int
    first_violated_count: int
    completed_count: int
    failed_count: int
    rejected_count: int
    expired_count: int

    @property
    def violation_rate(self) -> float | None:
        return None if self.accepted_count == 0 else self.first_violated_count/self.accepted_count


@dataclass(frozen=True, slots=True)
class CommunicationQosSlot:
    active_demand: bool
    effective_target_bit_per_s: float
    normalized_shortfall: float
    residual: float


@dataclass(frozen=True, slots=True)
class CommunicationQosSummary:
    active_demand_slots: int
    shortfall_sum: float
    residual_sum: float

    @property
    def mean_shortfall(self) -> float | None:
        return None if self.active_demand_slots == 0 else self.shortfall_sum/self.active_demand_slots


def initialize_admitted_request(request: SensingRequest) -> SensingRequest:
    """Initialize a newly ACTIVE request before its mandatory current-slot update."""
    if request.state is not RequestState.ACTIVE:
        raise SlaValidationError("admission initialization requires an ACTIVE SensingRequest")
    if any((
        request.valid_output_age_slots is not None, request.valid_output_count != 0,
        request.sla_violated, request.first_violation_slot is not None,
    )):
        raise SlaValidationError("admitted request has already been initialized or accounted")
    return request.with_accounting(
        valid_output_age_slots=request.valid_output_interval_slots+1, valid_output_count=0,
        sla_violated=False, first_violation_slot=None,
    )


def update_request_sla(
    request: SensingRequest, current_slot: int, service_durations: TaskDurationMap,
    *, valid_output: bool,
) -> RequestSlaUpdate:
    """Apply freshness, first violation, and final-slot outcome in canonical order."""
    if request.state is not RequestState.ACTIVE:
        raise SlaValidationError("SLA update requires an ACTIVE SensingRequest")
    if type(valid_output) is not bool:
        raise SlaValidationError("valid_output must be boolean")
    if isinstance(current_slot, bool) or not isinstance(current_slot, int) or current_slot < 0:
        raise SlaValidationError("current_slot must be a non-negative integer")
    final_slot = request.final_service_slot(service_durations)
    assert final_slot is not None
    if current_slot < request.admission_slot or current_slot > final_slot:
        raise SlaValidationError("current_slot lies outside the accepted service interval")

    pre_age = request.valid_output_age_slots
    if pre_age is None:
        pre_age = request.valid_output_interval_slots+1
    next_age = 0 if valid_output else pre_age+1
    next_count = request.valid_output_count+int(valid_output)
    first_violation = not request.sla_violated and (
        next_age > request.valid_output_interval_slots
        or current_slot == final_slot and next_count == 0
    )
    violated = request.sla_violated or first_violation
    updated = request.with_accounting(
        valid_output_age_slots=next_age, valid_output_count=next_count, sla_violated=violated,
        first_violation_slot=current_slot if first_violation else request.first_violation_slot,
    )

    completed = failed = False
    if current_slot == final_slot:
        completed = next_count >= 1 and not violated
        failed = not completed
        updated = updated.transition(RequestState.COMPLETED if completed else RequestState.FAILED)
    return RequestSlaUpdate(updated, valid_output, first_violation, completed, failed)


def tenant_slot_sla_residual(
    accepted_count: int, first_violation_count: int, sla_violation_budget: float,
) -> float:
    """Return the additive per-tenant slot residual from request-level events."""
    accepted = _count(accepted_count, "accepted_count")
    violated = _count(first_violation_count, "first_violation_count")
    budget = _finite(sla_violation_budget, "sla_violation_budget", minimum=0.0)
    if budget > 1.0:
        raise SlaValidationError("sla_violation_budget must lie in [0, 1]")
    return float(violated-budget*accepted)


def summarize_tenant_requests(
    requests: Iterable[SensingRequest], tenant_id: EntityId,
) -> TenantSlaSummary:
    """Reconstruct request-level tenant diagnostics from lifecycle states."""
    tenant_requests = tuple(request for request in requests if request.tenant_id == tenant_id)
    accepted_states = {RequestState.ACTIVE, RequestState.COMPLETED, RequestState.FAILED}
    return TenantSlaSummary(
        tenant_id, len(tenant_requests),
        sum(request.state in accepted_states for request in tenant_requests),
        sum(request.sla_violated for request in tenant_requests),
        sum(request.state is RequestState.COMPLETED for request in tenant_requests),
        sum(request.state is RequestState.FAILED for request in tenant_requests),
        sum(request.state is RequestState.REJECTED for request in tenant_requests),
        sum(request.state is RequestState.EXPIRED for request in tenant_requests),
    )


def tenant_episode_sla_residual(summary: TenantSlaSummary, sla_violation_budget: float) -> float:
    """Reconstruct the exact finite-horizon tenant residual from episode counts."""
    return tenant_slot_sla_residual(
        summary.accepted_count, summary.first_violated_count, sla_violation_budget,
    )


def _communication_values(
    demand_bit_per_s: float, minimum_rate_bit_per_s: float, served_rate_bit_per_s: float | None = None,
) -> tuple[float, float, float | None]:
    demand = _finite(demand_bit_per_s, "demand_bit_per_s", minimum=0.0)
    minimum_rate = _finite(minimum_rate_bit_per_s, "minimum_rate_bit_per_s", minimum=0.0)
    served = None if served_rate_bit_per_s is None else _finite(
        served_rate_bit_per_s, "served_rate_bit_per_s", minimum=0.0,
    )
    return demand, min(demand, minimum_rate), served


def effective_communication_target_bit_per_s(
    demand_bit_per_s: float, minimum_rate_bit_per_s: float,
) -> float:
    demand, target, _ = _communication_values(demand_bit_per_s, minimum_rate_bit_per_s)
    return float(target if demand > 0.0 else 0.0)


def normalized_communication_shortfall(
    demand_bit_per_s: float, minimum_rate_bit_per_s: float, served_rate_bit_per_s: float,
) -> float:
    demand, target, served = _communication_values(
        demand_bit_per_s, minimum_rate_bit_per_s, served_rate_bit_per_s,
    )
    if demand == 0.0:
        return 0.0
    return float(max(0.0, target-served)/target)


def communication_qos_slot(
    demand_bit_per_s: float, minimum_rate_bit_per_s: float, served_rate_bit_per_s: float,
    normalized_shortfall_budget: float,
) -> CommunicationQosSlot:
    """Compute one user's additive communication-QoS residual for one slot."""
    demand, target, served = _communication_values(
        demand_bit_per_s, minimum_rate_bit_per_s, served_rate_bit_per_s,
    )
    budget = _finite(normalized_shortfall_budget, "normalized_shortfall_budget", minimum=0.0)
    if demand == 0.0:
        return CommunicationQosSlot(False, 0.0, 0.0, 0.0)
    shortfall = float(max(0.0, target-served)/target)
    return CommunicationQosSlot(True, target, shortfall, shortfall-budget)


def summarize_communication_qos(
    slots: Iterable[CommunicationQosSlot],
) -> CommunicationQosSummary:
    """Aggregate one user's communication-QoS events over a finite horizon."""
    active = tuple(slot for slot in slots if slot.active_demand)
    return CommunicationQosSummary(
        len(active), sum(slot.normalized_shortfall for slot in active),
        sum(slot.residual for slot in active),
    )