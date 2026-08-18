"""Deterministic primitive traces, single-session plans, and raw oracle accounting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from isac_ssc.core.compatibility import (
    SensingPrimitiveState, evaluate_create_profile, evaluate_merge_profile,
    request_shared_output_valid,
)
from isac_ssc.core.entities import (
    CommunicationUser, DiskAOI, EntityId, Matrix4, RequestState, ResourceProfile,
    SensingRequest, SensingSession, Task,
)
from isac_ssc.core.quality import (
    CommunicationParameters, SensingParameters, evaluate_communication_quality,
    evaluate_shared_sensing_quality, predict_tracking_covariance, tracking_pcrb_m,
)
from isac_ssc.core.resources import (
    SensingResourceUsage, equal_share_communication_resources,
    normalized_sensing_resource_cost, residual_communication_resources,
    session_updates_at,
)
from isac_ssc.core.sla import (
    communication_qos_slot, initialize_admitted_request, tenant_slot_sla_residual,
    update_request_sla,
)
from isac_ssc.core.utility import completed_request_value, finite_horizon_return, slot_reward
from isac_ssc.envs.dynamics import CommunicationSlotPrimitive, PrimitiveTrace, TargetSlotPrimitive
from isac_ssc.utils.config import CanonicalConfig


class ReferenceValidationError(ValueError):
    """Raised when a deterministic oracle instance is malformed."""


def _identifier_key(value: EntityId) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _matrix4_diagonal(diagonal: Iterable[float]) -> Matrix4:
    values = tuple(float(value) for value in diagonal)
    return tuple(tuple(values[row] if row == column else 0.0 for column in range(4)) for row in range(4))


@dataclass(frozen=True, slots=True)
class OracleInstance:
    """A small clairvoyant optimization instance over the shared primitive-trace model."""

    primitive_trace: PrimitiveTrace
    requests: tuple[SensingRequest, ...]
    available_profiles: tuple[ResourceProfile, ...]
    communication_users: tuple[CommunicationUser, ...] = ()

    @property
    def trace_id(self) -> str:
        return self.primitive_trace.trace_id

    @property
    def horizon_slots(self) -> int:
        return self.primitive_trace.horizon_slots

    def request(self, request_id: EntityId) -> SensingRequest:
        return next(request for request in self.requests if request.request_id == request_id)

    def target(self, target_id: EntityId, slot: int) -> TargetSlotPrimitive:
        return self.primitive_trace.target_at(target_id, slot)

    def communication_at(
        self, slot: int, config: CanonicalConfig,
    ) -> tuple[tuple[CommunicationSlotPrimitive, CommunicationUser], ...]:
        contracts = {user.user_id: user for user in self.communication_users}
        values = []
        for primitive in self.primitive_trace.communication_at(slot):
            contract = contracts.get(primitive.user_id)
            minimum_rate = (
                config.communication["minimum_rate_bit_per_s"]
                if contract is None else contract.minimum_rate_bit_per_s
            )
            budget = (
                config.communication["normalized_shortfall_budget"]
                if contract is None else contract.normalized_shortfall_budget
            )
            user = CommunicationUser(
                primitive.user_id, primitive.position_m, primitive.velocity_m_per_s,
                primitive.demand_bit_per_s, minimum_rate, budget,
            )
            values.append((primitive, user))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class AdmissionEvent:
    slot: int
    request_id: EntityId
    profile_id: str


@dataclass(frozen=True, slots=True)
class SharedOutputRecord:
    plan_id: str
    slot: int
    profile_id: str
    member_request_ids: tuple[EntityId, ...]
    detection_probability: float
    peb_m: float | None
    pcrb_m: float | None
    tracking_measurement_updated: bool | None
    request_validity: tuple[tuple[EntityId, bool], ...]


@dataclass(frozen=True, slots=True)
class TrackingSlotRecord:
    slot: int
    predicted: bool
    measurement_updated: bool
    prior_covariance: Matrix4
    posterior_covariance: Matrix4
    pcrb_m: float


@dataclass(frozen=True, slots=True)
class RequestPlanOutcome:
    plan_id: str
    request_id: EntityId
    tenant_id: EntityId
    admission_slot: int
    final_service_slot: int
    state: RequestState
    valid_output_count: int
    sla_violated: bool
    first_violation_slot: int | None
    valid_output_slots: tuple[int, ...]

    @property
    def completed(self) -> bool:
        return self.state is RequestState.COMPLETED


@dataclass(frozen=True, slots=True)
class PlanSlotAccounting:
    slot: int
    sensing_bandwidth_hz: float
    sensing_power_w: float
    normalized_sensing_cost: float
    completed_value: float
    reward: float


@dataclass(frozen=True, slots=True)
class SingleSessionPlan:
    trace_id: str
    plan_id: str
    target_id: EntityId
    aoi: DiskAOI
    creator_request_id: EntityId
    creation_slot: int
    admissions: tuple[AdmissionEvent, ...]
    member_request_ids: tuple[EntityId, ...]
    final_session_slot: int
    update_slots: tuple[int, ...]
    shared_outputs: tuple[SharedOutputRecord, ...]
    tracking_slots: tuple[TrackingSlotRecord, ...]
    request_outcomes: tuple[RequestPlanOutcome, ...]
    slot_accounting: tuple[PlanSlotAccounting, ...]
    objective: float

    def request_ids(self) -> frozenset[EntityId]:
        return frozenset(self.member_request_ids)

    def admission_slots(self) -> frozenset[int]:
        return frozenset(event.slot for event in self.admissions)


@dataclass(frozen=True, slots=True)
class CommunicationSlotOutcome:
    slot: int
    user_id: EntityId
    served_rate_bit_per_s: float
    normalized_shortfall: float
    residual: float


@dataclass(frozen=True, slots=True)
class JointSlotAccounting:
    slot: int
    sensing_bandwidth_hz: float
    sensing_power_w: float
    normalized_sensing_cost: float
    completed_value: float
    reward: float


@dataclass(frozen=True, slots=True)
class JointSelectionAccounting:
    trace_id: str
    selected_plan_ids: tuple[str, ...]
    request_assignments: tuple[tuple[EntityId, str], ...]
    request_outcomes: tuple[RequestPlanOutcome, ...]
    shared_outputs: tuple[SharedOutputRecord, ...]
    slot_accounting: tuple[JointSlotAccounting, ...]
    tenant_residuals: tuple[tuple[EntityId, float], ...]
    communication_outcomes: tuple[CommunicationSlotOutcome, ...]
    communication_residuals: tuple[tuple[EntityId, float], ...]
    objective: float


def validate_reference_instance(instance: OracleInstance, config: CanonicalConfig) -> None:
    """Check the small-instance boundary once before plan enumeration or optimization."""
    if not isinstance(instance, OracleInstance):
        raise ReferenceValidationError("oracle input must be an OracleInstance")
    limits = config.oracle["instance_selection_limits"]
    if instance.horizon_slots > limits["horizon_slots"]:
        raise ReferenceValidationError("oracle instance exceeds the configured horizon limit")
    if len(instance.requests) > limits["max_requests"]:
        raise ReferenceValidationError("oracle instance exceeds the configured request limit")
    if not instance.available_profiles:
        raise ReferenceValidationError("oracle instance requires at least one resource profile")
    if len({request.request_id for request in instance.requests}) != len(instance.requests):
        raise ReferenceValidationError("oracle request identifiers must be unique")
    profile_ids = {profile.profile_id for profile in instance.available_profiles}
    if len(profile_ids) != len(instance.available_profiles):
        raise ReferenceValidationError("oracle profile identifiers must be unique")
    tenant_ids = {tenant.tenant_id for tenant in config.tenants}
    if any(request.tenant_id not in tenant_ids for request in instance.requests):
        raise ReferenceValidationError("oracle request references an unknown tenant")
    if any(
        request.state is not RequestState.WAITING
        or request.eligible_slot != request.arrival_slot
        or request.arrival_slot >= instance.horizon_slots
        or request.latest_start_slot >= instance.horizon_slots
        for request in instance.requests
    ):
        raise ReferenceValidationError("oracle requests must be pristine and lie inside the horizon")
    required_targets = {request.target_id for request in instance.requests}
    target_keys = {(item.slot, item.target_id) for item in instance.primitive_trace.target_states}
    expected_targets = {
        (slot, target_id)
        for slot in range(instance.horizon_slots)
        for target_id in required_targets
    }
    if not expected_targets.issubset(target_keys):
        raise ReferenceValidationError("oracle primitive trace is missing a required target state")
    user_ids = {item.user_id for item in instance.primitive_trace.communication_states}
    communication_keys = {
        (item.slot, item.user_id)
        for item in instance.primitive_trace.communication_states
    }
    expected_users = {(slot, user_id) for slot in range(instance.horizon_slots) for user_id in user_ids}
    if communication_keys != expected_users:
        raise ReferenceValidationError("oracle communication primitives must cover every user and slot")
    contract_ids = {user.user_id for user in instance.communication_users}
    if not contract_ids.issubset(user_ids):
        raise ReferenceValidationError("oracle communication contract references an unknown user")


def _plan_id(trace_id: str, admissions: tuple[AdmissionEvent, ...]) -> str:
    key = tuple((event.slot, _identifier_key(event.request_id), event.profile_id) for event in admissions)
    return f"{trace_id}:{key!r}"


def _primitive_state(
    instance: OracleInstance, target_id: EntityId, slot: int,
    tracking_prior: Matrix4 | None, config: CanonicalConfig,
) -> SensingPrimitiveState:
    target = instance.target(target_id, slot)
    return SensingPrimitiveState(
        target.position_m, tuple(config.geometry["bs_position_m"]), target.rcs_m2,
        target.shadowing_db, target.fading_power_gain, tracking_prior,
    )


def _tracking_initial_covariance(config: CanonicalConfig) -> Matrix4:
    return _matrix4_diagonal(config.sensing["tracking"]["initial_covariance_diag"])


def simulate_single_session_plan(
    instance: OracleInstance, config: CanonicalConfig,
    admissions: Iterable[AdmissionEvent],
) -> SingleSessionPlan | None:
    """Simulate one exact-target plan; return None when a declared plan event is infeasible."""
    events = tuple(admissions)
    if not events:
        raise ReferenceValidationError("a single-session plan requires a creation event")
    if tuple(sorted(events, key=lambda item: item.slot)) != events:
        raise ReferenceValidationError("admission events must be ordered by slot")
    if len({event.slot for event in events}) != len(events):
        raise ReferenceValidationError("a plan may admit at most one request per slot")
    if len({event.request_id for event in events}) != len(events):
        raise ReferenceValidationError("a request may be admitted at most once per plan")

    request_by_id = {request.request_id: request for request in instance.requests}
    profile_by_id = {profile.profile_id: profile for profile in instance.available_profiles}
    if any(event.request_id not in request_by_id for event in events):
        raise ReferenceValidationError("plan references an unknown request")
    if any(event.profile_id not in profile_by_id for event in events):
        raise ReferenceValidationError("plan references an unavailable profile")
    if any(event.slot >= instance.horizon_slots for event in events):
        raise ReferenceValidationError("plan event lies outside the trace horizon")

    plan_id = _plan_id(instance.trace_id, events)
    plan_request_ids = frozenset(event.request_id for event in events)
    state = {request_id: request_by_id[request_id] for request_id in plan_request_ids}
    event_by_slot = {event.slot: event for event in events}
    sensing_parameters = SensingParameters.from_config(config)
    initial_covariance = _tracking_initial_covariance(config)
    durations = config.service_duration_slots
    total_bandwidth = config.system["total_bandwidth_hz"]
    total_power = config.system["total_power_w"]
    minimum_coverage = config.compatibility["minimum_spatial_coverage_ratio"]
    bandwidth_weight = config.reward["sensing_cost_bandwidth_weight"]
    power_weight = config.reward["sensing_cost_power_weight"]
    resource_weight = config.reward["sensing_resource_cost_weight"]
    session: SensingSession | None = None
    shared_outputs: list[SharedOutputRecord] = []
    tracking_slots: list[TrackingSlotRecord] = []
    slot_accounting: list[PlanSlotAccounting] = []
    valid_output_slots = {request_id: [] for request_id in plan_request_ids}
    prediction_std = config.mobility["targets"]["acceleration_std_m_per_s2"]
    slot_duration = config.system["slot_duration_s"]

    for slot in range(instance.horizon_slots):
        if session is not None:
            terminal_ids = tuple(
                request_id
                for request_id in session.member_request_ids
                if state[request_id].is_terminal
            )
            if terminal_ids:
                session = session.detach_terminal_members(terminal_ids)

        event = event_by_slot.get(slot)
        if event is not None and session is None and event != events[0]:
            return None

        tracking_prior: Matrix4 | None = None
        predicted = False
        if session is not None and Task.TRACKING in session.exposed_outputs:
            assert session.tracking_covariance is not None
            if slot > session.creation_slot:
                tracking_prior = predict_tracking_covariance(
                    session.tracking_covariance, slot_duration, prediction_std,
                )
                session = replace(session, tracking_covariance=tracking_prior)
                predicted = True
            else:
                tracking_prior = session.tracking_covariance

        shared = None
        if event is not None:
            request = state[event.request_id]
            profile = profile_by_id[event.profile_id]
            if session is None:
                if event != events[0]:
                    return None
                primitive = _primitive_state(instance, request.target_id, slot, None, config)
                assessment = evaluate_create_profile(
                    request, plan_id, profile, primitive, sensing_parameters, (), durations, slot,
                    instance.horizon_slots, total_bandwidth, total_power, initial_covariance,
                )
                if (
                    not assessment.feasible
                    or assessment.candidate_session is None
                    or assessment.shared_quality is None
                ):
                    return None
                state[event.request_id] = initialize_admitted_request(
                    request.transition(RequestState.ACTIVE, slot=slot),
                )
                session = assessment.candidate_session
                shared = assessment.shared_quality
                tracking_prior = initial_covariance if Task.TRACKING in session.exposed_outputs else None
            else:
                if request.target_id != session.target_id:
                    return None
                primitive = _primitive_state(instance, session.target_id, slot, tracking_prior, config)
                members = tuple(state[request_id] for request_id in session.member_request_ids)
                assessment = evaluate_merge_profile(
                    request, session, members, config.tenants, profile, primitive,
                    sensing_parameters, (session,), durations, slot, instance.horizon_slots,
                    minimum_coverage, total_bandwidth, total_power,
                )
                if (
                    not assessment.feasible
                    or assessment.candidate_session is None
                    or assessment.shared_quality is None
                ):
                    return None
                state[event.request_id] = initialize_admitted_request(
                    request.transition(RequestState.ACTIVE, slot=slot),
                )
                session = assessment.candidate_session
                shared = assessment.shared_quality
        elif session is not None and session_updates_at(session, slot):
            primitive = _primitive_state(instance, session.target_id, slot, tracking_prior, config)
            shared = evaluate_shared_sensing_quality(
                session, primitive.target_position_m, primitive.bs_position_m,
                primitive.target_rcs_m2, primitive.sensing_shadowing_db,
                primitive.sensing_fading_power_gain, sensing_parameters, tracking_prior,
            )

        bandwidth = 0.0
        power = 0.0
        member_ids: tuple[EntityId, ...] = ()
        validity: tuple[tuple[EntityId, bool], ...] = ()
        if session is not None:
            member_ids = session.member_request_ids

        if shared is not None and session is not None:
            bandwidth = session.profile.sensing_bandwidth_hz
            power = session.profile.sensing_power_w
            validity = tuple(
                (request_id, request_shared_output_valid(state[request_id], session, shared))
                for request_id in member_ids
            )
            for request_id, valid in validity:
                if valid:
                    valid_output_slots[request_id].append(slot)

            localization = shared.localization
            tracking = shared.tracking
            shared_outputs.append(SharedOutputRecord(
                plan_id, slot, session.profile.profile_id, member_ids,
                shared.detection_probability, localization.peb_m,
                None if tracking is None else tracking.pcrb_m,
                None if tracking is None else tracking.measurement_updated, validity,
            ))
            next_update = slot+session.profile.update_period_slots
            posterior = None if tracking is None else tracking.posterior_covariance
            session = session.with_update_state(
                next_update_slot=next_update, tracking_covariance=posterior,
            )

        validity_by_id = dict(validity)
        completed_events = []
        if session is not None:
            for request_id in member_ids:
                request = state[request_id]
                update = update_request_sla(
                    request, slot, durations, valid_output=validity_by_id.get(request_id, False),
                )
                state[request_id] = update.updated_request
                if update.completed_event:
                    completed_events.append(update.updated_request)

        if session is not None and Task.TRACKING in session.exposed_outputs:
            assert tracking_prior is not None
            assert session.tracking_covariance is not None
            measurement_updated = bool(
                shared is not None
                and shared.tracking is not None
                and shared.tracking.measurement_updated
            )
            tracking_slots.append(TrackingSlotRecord(
                slot, predicted, measurement_updated, tracking_prior,
                session.tracking_covariance, tracking_pcrb_m(session.tracking_covariance),
            ))

        usage = SensingResourceUsage(bandwidth, power, (plan_id,) if shared is not None else ())
        cost = normalized_sensing_resource_cost(
            usage, total_bandwidth, total_power, bandwidth_weight, power_weight,
        )
        completed_value = completed_request_value(completed_events)
        reward = slot_reward(completed_value, cost, resource_weight)
        slot_accounting.append(PlanSlotAccounting(
            slot, bandwidth, power, cost, completed_value, reward,
        ))

    outcomes = []
    for event in events:
        request = state[event.request_id]
        final_slot = request.final_service_slot(durations)
        if request.state not in {RequestState.COMPLETED, RequestState.FAILED} or final_slot is None:
            return None
        outcomes.append(RequestPlanOutcome(
            plan_id, request.request_id, request.tenant_id, request.admission_slot,
            final_slot, request.state, request.valid_output_count, request.sla_violated,
            request.first_violation_slot, tuple(valid_output_slots[request.request_id]),
        ))

    outcome_values = tuple(sorted(outcomes, key=lambda item: _identifier_key(item.request_id)))
    target_id = request_by_id[events[0].request_id].target_id
    aoi = request_by_id[events[0].request_id].aoi
    objective = finite_horizon_return(item.reward for item in slot_accounting)

    return SingleSessionPlan(
        instance.trace_id, plan_id, target_id, aoi, events[0].request_id, events[0].slot,
        events, tuple(event.request_id for event in events),
        max(outcome.final_service_slot for outcome in outcome_values),
        tuple(output.slot for output in shared_outputs), tuple(shared_outputs),
        tuple(tracking_slots), outcome_values, tuple(slot_accounting), objective,
    )


def enumerate_single_session_plans(
    instance: OracleInstance, config: CanonicalConfig,
) -> tuple[SingleSessionPlan, ...]:
    """Enumerate the complete feasible plan set without request, session, or plan truncation."""
    validate_reference_instance(instance, config)
    plans: dict[str, SingleSessionPlan] = {}
    requests = tuple(sorted(instance.requests, key=lambda item: _identifier_key(item.request_id)))
    profiles = tuple(sorted(instance.available_profiles, key=lambda item: item.profile_id))

    def extend(plan: SingleSessionPlan) -> None:
        plans[plan.plan_id] = plan
        used = plan.request_ids()
        last_slot = plan.admissions[-1].slot

        for request in requests:
            if request.request_id in used:
                continue
            first_slot = max(last_slot+1, request.arrival_slot)
            last_feasible = min(
                request.latest_start_slot, plan.final_session_slot, instance.horizon_slots-1,
            )
            for slot in range(first_slot, last_feasible+1):
                for profile in profiles:
                    events = plan.admissions + (
                        AdmissionEvent(slot, request.request_id, profile.profile_id),
                    )
                    candidate = simulate_single_session_plan(instance, config, events)
                    if candidate is not None:
                        extend(candidate)

    for request in requests:
        last_slot = min(request.latest_start_slot, instance.horizon_slots-1)
        for slot in range(request.arrival_slot, last_slot+1):
            for profile in profiles:
                plan = simulate_single_session_plan(
                    instance, config,
                    (AdmissionEvent(slot, request.request_id, profile.profile_id),),
                )
                if plan is not None:
                    extend(plan)

    return tuple(sorted(plans.values(), key=lambda item: item.plan_id))


def evaluate_joint_selection(
    selected_plans: Iterable[SingleSessionPlan],
    instance: OracleInstance,
    config: CanonicalConfig,
    *,
    tolerance: float = 1.0e-10,
) -> JointSelectionAccounting | None:
    """Reconstruct all raw accounting for one joint plan selection and reject any violation."""
    plans = tuple(sorted(selected_plans, key=lambda item: item.plan_id))
    if len({plan.plan_id for plan in plans}) != len(plans):
        raise ReferenceValidationError("selected plan identifiers must be unique")

    assignments: dict[EntityId, str] = {}
    admission_counts = [0]*instance.horizon_slots
    outcomes = []

    for plan in plans:
        for request_id in plan.member_request_ids:
            if request_id in assignments:
                return None
            assignments[request_id] = plan.plan_id
        for event in plan.admissions:
            admission_counts[event.slot] += 1
            if admission_counts[event.slot] > 1:
                return None
        outcomes.extend(plan.request_outcomes)

    total_bandwidth = config.system["total_bandwidth_hz"]
    total_power = config.system["total_power_w"]
    bandwidth_weight = config.reward["sensing_cost_bandwidth_weight"]
    power_weight = config.reward["sensing_cost_power_weight"]
    resource_weight = config.reward["sensing_resource_cost_weight"]
    slots = []

    for slot in range(instance.horizon_slots):
        bandwidth = sum(plan.slot_accounting[slot].sensing_bandwidth_hz for plan in plans)
        power = sum(plan.slot_accounting[slot].sensing_power_w for plan in plans)
        if bandwidth > total_bandwidth+tolerance or power > total_power+tolerance:
            return None

        usage = SensingResourceUsage(
            bandwidth, power,
            tuple(
                plan.plan_id
                for plan in plans
                if plan.slot_accounting[slot].sensing_bandwidth_hz > 0.0
            ),
        )
        cost = normalized_sensing_resource_cost(
            usage, total_bandwidth, total_power, bandwidth_weight, power_weight,
        )
        value = sum(plan.slot_accounting[slot].completed_value for plan in plans)
        slots.append(JointSlotAccounting(
            slot, bandwidth, power, cost, value,
            slot_reward(value, cost, resource_weight),
        ))

    tenant_residuals = []
    for tenant in config.tenants:
        tenant_outcomes = tuple(item for item in outcomes if item.tenant_id == tenant.tenant_id)
        residual = tenant_slot_sla_residual(
            len(tenant_outcomes), sum(item.sla_violated for item in tenant_outcomes),
            tenant.sla_violation_budget,
        )
        if residual > tolerance:
            return None
        tenant_residuals.append((tenant.tenant_id, residual))

    communication_parameters = CommunicationParameters.from_config(config)
    communication_outcomes = []
    communication_residuals: dict[EntityId, float] = {}

    for slot_record in slots:
        communication = instance.communication_at(slot_record.slot, config)
        users = tuple(user for _, user in communication)
        residual_resources = residual_communication_resources(
            total_bandwidth, total_power,
            SensingResourceUsage(slot_record.sensing_bandwidth_hz, slot_record.sensing_power_w),
        )
        allocations = {
            item.user_id: item
            for item in equal_share_communication_resources(users, residual_resources)
        }

        for primitive, user in communication:
            allocation = allocations[user.user_id]
            quality = evaluate_communication_quality(
                user.position_m, config.geometry["bs_position_m"], allocation.bandwidth_hz,
                allocation.power_w, user.demand_bit_per_s, primitive.shadowing_db,
                primitive.fading_power_gain, communication_parameters,
            )
            qos = communication_qos_slot(
                user.demand_bit_per_s, user.minimum_rate_bit_per_s,
                quality.served_rate_bit_per_s, user.normalized_shortfall_budget,
            )
            communication_outcomes.append(CommunicationSlotOutcome(
                slot_record.slot, user.user_id, quality.served_rate_bit_per_s,
                qos.normalized_shortfall, qos.residual,
            ))
            communication_residuals[user.user_id] = (
                communication_residuals.get(user.user_id, 0.0)+qos.residual
            )

    if any(residual > tolerance for residual in communication_residuals.values()):
        return None

    return JointSelectionAccounting(
        instance.trace_id,
        tuple(plan.plan_id for plan in plans),
        tuple(sorted(assignments.items(), key=lambda item: _identifier_key(item[0]))),
        tuple(sorted(outcomes, key=lambda item: _identifier_key(item.request_id))),
        tuple(sorted(
            (output for plan in plans for output in plan.shared_outputs),
            key=lambda item: (item.slot, item.plan_id),
        )),
        tuple(slots),
        tuple(tenant_residuals),
        tuple(sorted(
            communication_outcomes,
            key=lambda item: (item.slot, _identifier_key(item.user_id)),
        )),
        tuple(sorted(communication_residuals.items(), key=lambda item: _identifier_key(item[0]))),
        finite_horizon_return(item.reward for item in slots),
    )