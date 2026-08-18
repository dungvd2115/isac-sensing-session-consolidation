"""Exact finite-horizon slotted CMDP environment for ISAC sensing-session consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from isac_ssc.core.compatibility import request_shared_output_valid
from isac_ssc.core.entities import (
    CommunicationUser, EntityId, RequestState, SensingRequest, SensingSession, Task,
)
from isac_ssc.core.quality import (
    CommunicationParameters, CommunicationQuality, SensingParameters, SharedSensingQuality,
    evaluate_communication_quality, evaluate_shared_sensing_quality,
    predict_tracking_covariance,
)
from isac_ssc.core.resources import (
    CommunicationAllocation, SensingResourceUsage, equal_share_communication_resources,
    normalized_sensing_resource_cost, residual_communication_resources,
    sensing_resource_usage, session_updates_at,
)
from isac_ssc.core.sla import (
    communication_qos_slot, initialize_admitted_request, tenant_slot_sla_residual,
    update_request_sla,
)
from isac_ssc.core.utility import completed_request_value, slot_reward
from isac_ssc.envs.action_masks import (
    ActionMaskSnapshot, CurrentFeasibilitySnapshot, MaskedActionEntry,
    build_action_masks, build_current_feasibility,
)
from isac_ssc.envs.action_space import ActionType, EnvironmentAction, identifier_key
from isac_ssc.envs.dynamics import (
    CommunicationSlotPrimitive, PrimitiveTrace, RequestPrimitiveDescriptor, TargetSlotPrimitive,
)
from isac_ssc.envs.observation import (
    CommunicationAccountingState, ObservationSnapshot, TenantAccountingState,
    build_observation,
)
from isac_ssc.utils.config import CanonicalConfig


class EnvironmentValidationError(ValueError):
    """Raised when the public environment API is used incorrectly."""


def _typed_index(values, attribute: str):
    return {identifier_key(getattr(item, attribute)): item for item in values}


def _sorted_ids(values: Iterable[EntityId]) -> tuple[EntityId, ...]:
    return tuple(sorted(values, key=identifier_key))


@dataclass(frozen=True, slots=True)
class SessionUpdateRecord:
    session_id: EntityId
    shared_quality: SharedSensingQuality
    valid_request_ids: tuple[EntityId, ...]


@dataclass(frozen=True, slots=True)
class CommunicationServiceRecord:
    user_id: EntityId
    demand_bit_per_s: float
    allocated_bandwidth_hz: float
    allocated_power_w: float
    achievable_rate_bit_per_s: float
    served_rate_bit_per_s: float
    normalized_shortfall: float
    residual: float


@dataclass(frozen=True, slots=True)
class StepResult:
    processed_slot: int
    action: EnvironmentAction | None
    focal_request_id: EntityId | None
    arrived_request_ids: tuple[EntityId, ...]
    expired_request_ids: tuple[EntityId, ...]
    accepted_request_ids: tuple[EntityId, ...]
    rejected_request_ids: tuple[EntityId, ...]
    tracking_prediction_session_ids: tuple[EntityId, ...]
    session_updates: tuple[SessionUpdateRecord, ...]
    communication_service: tuple[CommunicationServiceRecord, ...]
    valid_output_request_ids: tuple[EntityId, ...]
    first_violation_request_ids: tuple[EntityId, ...]
    completed_request_ids: tuple[EntityId, ...]
    failed_request_ids: tuple[EntityId, ...]
    sensing_resource_usage: SensingResourceUsage
    completed_value: float
    sensing_resource_cost: float
    reward: float
    tenant_sla_residuals: tuple[tuple[EntityId, float], ...]
    communication_qos_residuals: tuple[tuple[EntityId, float], ...]
    cumulative_completed_value: float
    cumulative_sensing_resource_cost: float
    cumulative_reward: float
    next_observation: ObservationSnapshot | None
    terminated: bool


@dataclass(frozen=True, slots=True)
class EnvironmentStateSnapshot:
    trace_id: str
    current_slot: int
    terminated: bool
    requests: tuple[SensingRequest, ...]
    active_sessions: tuple[SensingSession, ...]
    current_target_primitives: tuple[TargetSlotPrimitive, ...]
    current_communication_primitives: tuple[CommunicationSlotPrimitive, ...]
    pending_clustered_children: tuple[RequestPrimitiveDescriptor, ...]
    focal_request_id: EntityId | None
    action_masks: ActionMaskSnapshot | None
    observation: ObservationSnapshot | None
    tenant_accounting: tuple[TenantAccountingState, ...]
    communication_accounting: tuple[CommunicationAccountingState, ...]
    cumulative_completed_value: float
    cumulative_sensing_resource_cost: float
    cumulative_reward: float
    action_counts: tuple[tuple[ActionType, int], ...]
    no_request_count: int
    next_session_counter: int


class ISACSSCEnv:
    """Deterministic one-action-per-physical-slot environment over an immutable primitive trace."""

    def __init__(self, config: CanonicalConfig) -> None:
        if not isinstance(config, CanonicalConfig):
            raise EnvironmentValidationError("config must be CanonicalConfig")
        self.config = config
        self._sensing_parameters = SensingParameters.from_config(config)
        self._communication_parameters = CommunicationParameters.from_config(config)
        self._trace: PrimitiveTrace | None = None
        self._arrivals_by_slot: dict[int, tuple[SensingRequest, ...]] = {}
        self._requests: tuple[SensingRequest, ...] = ()
        self._sessions: tuple[SensingSession, ...] = ()
        self._current_slot = 0
        self._terminated = False
        self._prepared = False
        self._current_targets: tuple[TargetSlotPrimitive, ...] = ()
        self._current_communication: tuple[CommunicationSlotPrimitive, ...] = ()
        self._pending_children: tuple[RequestPrimitiveDescriptor, ...] = ()
        self._focal: SensingRequest | None = None
        self._feasibility: CurrentFeasibilitySnapshot | None = None
        self._action_masks: ActionMaskSnapshot | None = None
        self._observation: ObservationSnapshot | None = None
        self._current_arrived_ids: tuple[EntityId, ...] = ()
        self._current_expired_ids: tuple[EntityId, ...] = ()
        self._current_prediction_ids: tuple[EntityId, ...] = ()
        self._tenant_accounting: tuple[TenantAccountingState, ...] = ()
        self._communication_accounting: tuple[CommunicationAccountingState, ...] = ()
        self._cumulative_completed_value = 0.0
        self._cumulative_sensing_cost = 0.0
        self._cumulative_reward = 0.0
        self._action_counts = {kind: 0 for kind in ActionType}
        self._no_request_count = 0
        self._next_session_counter = 0

    @property
    def current_slot(self) -> int:
        self._require_reset()
        return self._current_slot

    @property
    def terminated(self) -> bool:
        self._require_reset()
        return self._terminated

    @property
    def trace_id(self) -> str:
        self._require_reset()
        assert self._trace is not None
        return self._trace.trace_id

    def reset(self, trace: PrimitiveTrace) -> ObservationSnapshot | None:
        if not isinstance(trace, PrimitiveTrace):
            raise EnvironmentValidationError("reset requires a PrimitiveTrace")
        if trace.horizon_slots != self.config.system["horizon_slots"]:
            raise EnvironmentValidationError("trace horizon does not match the environment")
        if trace.tenant_ids != tuple(tenant.tenant_id for tenant in self.config.tenants):
            raise EnvironmentValidationError("trace tenants do not match the environment")
        requests = trace.materialized_requests(self.config)
        self._trace = trace
        self._arrivals_by_slot = {
            slot: tuple(item for item in requests if item.arrival_slot == slot)
            for slot in range(trace.horizon_slots)
        }
        self._requests = ()
        self._sessions = ()
        self._current_slot = 0
        self._terminated = False
        self._prepared = False
        self._current_targets = ()
        self._current_communication = ()
        self._pending_children = ()
        self._focal = None
        self._feasibility = None
        self._action_masks = None
        self._observation = None
        self._current_arrived_ids = ()
        self._current_expired_ids = ()
        self._current_prediction_ids = ()
        self._tenant_accounting = tuple(
            TenantAccountingState(tenant.tenant_id, 0, 0, 0, 0.0)
            for tenant in self.config.tenants
        )
        user_ids = _sorted_ids({item.user_id for item in trace.communication_states})
        self._communication_accounting = tuple(
            CommunicationAccountingState(user_id, 0, 0.0, 0.0)
            for user_id in user_ids
        )
        self._cumulative_completed_value = 0.0
        self._cumulative_sensing_cost = 0.0
        self._cumulative_reward = 0.0
        self._action_counts = {kind: 0 for kind in ActionType}
        self._no_request_count = 0
        self._next_session_counter = 0
        self._prepare_current_slot()
        return self._observation

    def current_action_masks(self) -> ActionMaskSnapshot | None:
        self._require_ready()
        return self._action_masks

    def current_observation(self) -> ObservationSnapshot | None:
        self._require_ready()
        return self._observation

    def state_snapshot(self) -> EnvironmentStateSnapshot:
        self._require_reset()
        assert self._trace is not None
        return EnvironmentStateSnapshot(
            self._trace.trace_id, self._current_slot, self._terminated, self._requests,
            self._sessions, self._current_targets, self._current_communication,
            self._pending_children, None if self._focal is None else self._focal.request_id,
            self._action_masks, self._observation, self._tenant_accounting,
            self._communication_accounting, self._cumulative_completed_value,
            self._cumulative_sensing_cost, self._cumulative_reward,
            tuple((kind, self._action_counts[kind]) for kind in ActionType),
            self._no_request_count, self._next_session_counter,
        )

    def step(self, action: EnvironmentAction | None) -> StepResult:
        self._require_ready()
        if self._terminated:
            raise EnvironmentValidationError("cannot step a terminated environment")
        if self._focal is None:
            if action is not None:
                raise EnvironmentValidationError("a no-focal slot requires step(None)")
            entry = None
        else:
            if action is None:
                raise EnvironmentValidationError(
                    "a focal-request slot requires an EnvironmentAction",
                )
            if not isinstance(action, EnvironmentAction):
                raise EnvironmentValidationError("action must be EnvironmentAction")
            assert self._action_masks is not None
            try:
                entry = self._action_masks.entry_for(action)
            except KeyError as error:
                raise EnvironmentValidationError(
                    "action is not in the current structural catalogue",
                ) from error
            if not entry.feasible:
                raise EnvironmentValidationError("masked action is infeasible")
        return self._execute_step(action, entry)

    def _execute_step(
        self, action: EnvironmentAction | None, entry: MaskedActionEntry | None,
    ) -> StepResult:
        assert self._trace is not None
        current = self._current_slot
        focal_id = None if self._focal is None else self._focal.request_id
        requests = list(self._requests)
        sessions = list(self._sessions)
        request_index = _typed_index(requests, "request_id")
        session_index = _typed_index(sessions, "session_id")
        accepted_ids: list[EntityId] = []
        rejected_ids: list[EntityId] = []
        retained_quality: dict[tuple[int, str], SharedSensingQuality] = {}

        if action is None:
            self._no_request_count += 1
        else:
            assert self._focal is not None and entry is not None
            focal_key = identifier_key(self._focal.request_id)
            focal = request_index[focal_key]

            if action.action_type is ActionType.MERGE:
                assessment = entry.merge_assessment
                request_index[focal_key] = initialize_admitted_request(
                    focal.transition(RequestState.ACTIVE, slot=current),
                )
                destination_key = identifier_key(action.session_id)
                session_index[destination_key] = assessment.candidate_session
                retained_quality[destination_key] = assessment.shared_quality
                accepted_ids.append(focal.request_id)

            elif action.action_type is ActionType.CREATE:
                assessment = entry.create_assessment
                request_index[focal_key] = initialize_admitted_request(
                    focal.transition(RequestState.ACTIVE, slot=current),
                )
                candidate = assessment.candidate_session
                candidate_key = identifier_key(candidate.session_id)
                session_index[candidate_key] = candidate
                retained_quality[candidate_key] = assessment.shared_quality
                self._next_session_counter += 1
                accepted_ids.append(focal.request_id)

            elif action.action_type is ActionType.DEFER:
                request_index[focal_key] = focal.defer(
                    current, self.config.requests["defer_cooldown_slots"],
                )

            elif action.action_type is ActionType.REJECT:
                request_index[focal_key] = focal.transition(RequestState.REJECTED)
                rejected_ids.append(focal.request_id)

            self._action_counts[action.action_type] += 1

        requests = sorted(request_index.values(), key=lambda item: identifier_key(item.request_id))
        sessions = sorted(session_index.values(), key=lambda item: identifier_key(item.session_id))
        usage = sensing_resource_usage(sessions, current)
        residual_communication_resources(
            self.config.system["total_bandwidth_hz"],
            self.config.system["total_power_w"],
            usage,
        )

        target_index = _typed_index(self._current_targets, "target_id")
        request_index = _typed_index(requests, "request_id")
        session_updates: list[SessionUpdateRecord] = []
        valid_ids: set[tuple[int, str]] = set()
        updated_sessions: list[SensingSession] = []

        for session in sessions:
            if not session_updates_at(session, current):
                updated_sessions.append(session)
                continue

            key = identifier_key(session.session_id)
            target = target_index[identifier_key(session.target_id)]
            shared = retained_quality.get(key)
            if shared is None:
                shared = evaluate_shared_sensing_quality(
                    session, target.position_m,
                    tuple(self.config.geometry["bs_position_m"]),
                    target.rcs_m2, target.shadowing_db, target.fading_power_gain,
                    self._sensing_parameters, session.tracking_covariance,
                )

            member_valid = []
            for request_id in session.member_request_ids:
                request = request_index[identifier_key(request_id)]
                if request_shared_output_valid(request, session, shared):
                    valid_ids.add(identifier_key(request_id))
                    member_valid.append(request_id)

            session_updates.append(SessionUpdateRecord(
                session.session_id, shared, _sorted_ids(member_valid),
            ))
            next_slot = current + session.profile.update_period_slots
            next_slot = (
                next_slot
                if next_slot <= session.final_active_slot
                else session.final_active_slot + 1
            )
            covariance = (
                None if shared.tracking is None
                else shared.tracking.posterior_covariance
            )
            updated_sessions.append(session.with_update_state(
                next_update_slot=next_slot, tracking_covariance=covariance,
            ))

        sessions = sorted(updated_sessions, key=lambda item: identifier_key(item.session_id))
        communication_service = self._serve_communication(usage)
        communication_residuals = tuple(
            (item.user_id, item.residual) for item in communication_service
        )

        first_violation_ids: list[EntityId] = []
        completed_ids: list[EntityId] = []
        failed_ids: list[EntityId] = []
        completed_events: list[SensingRequest] = []
        request_index = _typed_index(requests, "request_id")

        for key, request in tuple(request_index.items()):
            if request.state is not RequestState.ACTIVE:
                continue
            update = update_request_sla(
                request, current, self.config.service_duration_slots,
                valid_output=key in valid_ids,
            )
            request_index[key] = update.updated_request
            if update.first_violation_event:
                first_violation_ids.append(request.request_id)
            if update.completed_event:
                completed_ids.append(request.request_id)
                completed_events.append(update.updated_request)
            if update.failed_event:
                failed_ids.append(request.request_id)

        requests = sorted(request_index.values(), key=lambda item: identifier_key(item.request_id))
        tenant_residuals = self._update_tenant_accounting(
            requests, tuple(accepted_ids), tuple(first_violation_ids),
            tuple(completed_ids),
        )
        self._update_communication_accounting(communication_service)

        completed_value = completed_request_value(completed_events)
        sensing_cost = normalized_sensing_resource_cost(
            usage, self.config.system["total_bandwidth_hz"],
            self.config.system["total_power_w"],
            self.config.reward["sensing_cost_bandwidth_weight"],
            self.config.reward["sensing_cost_power_weight"],
        )
        reward = slot_reward(
            completed_value, sensing_cost,
            self.config.reward["sensing_resource_cost_weight"],
        )

        self._cumulative_completed_value += completed_value
        self._cumulative_sensing_cost += sensing_cost
        self._cumulative_reward += reward
        self._requests = tuple(requests)
        self._sessions = tuple(sessions)
        self._focal = None
        self._feasibility = None
        self._action_masks = None
        self._observation = None

        processed_arrivals = self._current_arrived_ids
        processed_expired = self._current_expired_ids
        processed_predictions = self._current_prediction_ids

        if current == self._trace.horizon_slots - 1:
            self._current_slot = self._trace.horizon_slots
            self._terminated = True
            self._prepared = True
            self._current_targets = ()
            self._current_communication = ()
            self._pending_children = ()
            next_observation = None
        else:
            self._current_slot = current + 1
            self._prepared = False
            self._prepare_current_slot()
            next_observation = self._observation

        return StepResult(
            current, action, focal_id, processed_arrivals, processed_expired,
            _sorted_ids(accepted_ids), _sorted_ids(rejected_ids), processed_predictions,
            tuple(session_updates), communication_service,
            _sorted_ids(
                request_index[key].request_id
                for key in valid_ids
                if key in request_index
            ),
            _sorted_ids(first_violation_ids), _sorted_ids(completed_ids),
            _sorted_ids(failed_ids), usage, completed_value, sensing_cost, reward,
            tenant_residuals, communication_residuals,
            self._cumulative_completed_value, self._cumulative_sensing_cost,
            self._cumulative_reward, next_observation, self._terminated,
        )

    def _prepare_current_slot(self) -> None:
        assert self._trace is not None
        current = self._current_slot
        request_index = _typed_index(self._requests, "request_id")
        terminal_keys = {
            key for key, request in request_index.items()
            if request.state in {RequestState.COMPLETED, RequestState.FAILED}
        }

        sessions = []
        for session in self._sessions:
            detach = tuple(
                request_id for request_id in session.member_request_ids
                if identifier_key(request_id) in terminal_keys
            )
            remaining = session.detach_terminal_members(detach) if detach else session
            if remaining is not None:
                sessions.append(remaining)
        self._sessions = tuple(sorted(
            sessions, key=lambda item: identifier_key(item.session_id),
        ))

        self._current_targets = tuple(
            item for item in self._trace.target_states if item.slot == current
        )
        self._current_communication = self._trace.communication_at(current)
        self._pending_children = self._trace.pending_children_at(current)

        predicted = []
        sessions = []
        for session in self._sessions:
            if Task.TRACKING in session.exposed_outputs and session.creation_slot < current:
                covariance = predict_tracking_covariance(
                    session.tracking_covariance,
                    self.config.system["slot_duration_s"],
                    self.config.mobility["targets"]["acceleration_std_m_per_s2"],
                )
                session = session.with_update_state(
                    next_update_slot=session.next_update_slot,
                    tracking_covariance=covariance,
                )
                predicted.append(session.session_id)
            sessions.append(session)

        self._sessions = tuple(sorted(
            sessions, key=lambda item: identifier_key(item.session_id),
        ))
        self._current_prediction_ids = _sorted_ids(predicted)

        arrivals = self._arrivals_by_slot.get(current, ())
        for request in arrivals:
            request_index[identifier_key(request.request_id)] = request

        expired = []
        for key, request in tuple(request_index.items()):
            if request.state is RequestState.WAITING and current > request.latest_start_slot:
                request_index[key] = request.transition(RequestState.EXPIRED)
                expired.append(request.request_id)

        self._requests = tuple(sorted(
            request_index.values(), key=lambda item: identifier_key(item.request_id),
        ))
        self._current_arrived_ids = _sorted_ids(
            request.request_id for request in arrivals
        )
        self._current_expired_ids = _sorted_ids(expired)

        eligible = tuple(
            request for request in self._requests
            if request.state is RequestState.WAITING
            and current >= request.eligible_slot
        )
        self._focal = min(
            eligible,
            key=lambda request: (
                request.eligible_slot, request.arrival_slot,
                identifier_key(request.request_id),
            ),
            default=None,
        )

        if self._focal is None:
            self._feasibility = None
            self._action_masks = None
            self._observation = None
        else:
            prospective = self._prospective_session_id()
            self._feasibility = build_current_feasibility(
                current, self._requests, self._sessions,
                self._current_targets, prospective, self.config,
            )
            self._action_masks = build_action_masks(
                self._focal, self._requests, self._sessions,
                self._feasibility, self.config,
            )
            self._observation = build_observation(
                current, self._focal, self._requests, self._sessions,
                self._current_targets, self._current_communication,
                self._feasibility, self._action_masks,
                self._tenant_accounting, self._communication_accounting,
                self._cumulative_completed_value,
                self._cumulative_sensing_cost, self.config,
            )
        self._prepared = True

    def _serve_communication(
        self, sensing_usage: SensingResourceUsage,
    ) -> tuple[CommunicationServiceRecord, ...]:
        users = tuple(CommunicationUser(
            primitive.user_id, primitive.position_m, primitive.velocity_m_per_s,
            primitive.demand_bit_per_s,
            self.config.communication["minimum_rate_bit_per_s"],
            self.config.communication["normalized_shortfall_budget"],
        ) for primitive in self._current_communication)

        residual = residual_communication_resources(
            self.config.system["total_bandwidth_hz"],
            self.config.system["total_power_w"], sensing_usage,
        )
        allocations = _typed_index(
            equal_share_communication_resources(users, residual), "user_id",
        )
        primitives = _typed_index(self._current_communication, "user_id")
        records = []

        for user in sorted(users, key=lambda item: identifier_key(item.user_id)):
            primitive = primitives[identifier_key(user.user_id)]
            allocation: CommunicationAllocation = allocations[identifier_key(user.user_id)]
            quality: CommunicationQuality = evaluate_communication_quality(
                primitive.position_m, tuple(self.config.geometry["bs_position_m"]),
                allocation.bandwidth_hz, allocation.power_w,
                user.demand_bit_per_s, primitive.shadowing_db,
                primitive.fading_power_gain, self._communication_parameters,
            )
            qos = communication_qos_slot(
                user.demand_bit_per_s, user.minimum_rate_bit_per_s,
                quality.served_rate_bit_per_s,
                user.normalized_shortfall_budget,
            )
            records.append(CommunicationServiceRecord(
                user.user_id, user.demand_bit_per_s,
                allocation.bandwidth_hz, allocation.power_w,
                quality.achievable_rate_bit_per_s,
                quality.served_rate_bit_per_s,
                qos.normalized_shortfall, qos.residual,
            ))

        return tuple(records)

    def _update_tenant_accounting(
        self, requests: list[SensingRequest], accepted_ids: tuple[EntityId, ...],
        violated_ids: tuple[EntityId, ...], completed_ids: tuple[EntityId, ...],
    ) -> tuple[tuple[EntityId, float], ...]:
        request_index = _typed_index(requests, "request_id")
        accepted_keys = {identifier_key(item) for item in accepted_ids}
        violated_keys = {identifier_key(item) for item in violated_ids}
        completed_keys = {identifier_key(item) for item in completed_ids}
        prior = _typed_index(self._tenant_accounting, "tenant_id")
        updated = []
        slot_residuals = []

        for tenant in self.config.tenants:
            tenant_key = identifier_key(tenant.tenant_id)
            accepted = sum(
                key in accepted_keys
                and identifier_key(request.tenant_id) == tenant_key
                for key, request in request_index.items()
            )
            violated = sum(
                key in violated_keys
                and identifier_key(request.tenant_id) == tenant_key
                for key, request in request_index.items()
            )
            completed = sum(
                key in completed_keys
                and identifier_key(request.tenant_id) == tenant_key
                for key, request in request_index.items()
            )
            residual = tenant_slot_sla_residual(
                accepted, violated, tenant.sla_violation_budget,
            )
            before = prior[tenant_key]
            updated.append(TenantAccountingState(
                tenant.tenant_id, before.accepted_count + accepted,
                before.first_violated_count + violated,
                before.completed_count + completed,
                before.residual + residual,
            ))
            slot_residuals.append((tenant.tenant_id, residual))

        self._tenant_accounting = tuple(updated)
        return tuple(slot_residuals)

    def _update_communication_accounting(
        self, records: tuple[CommunicationServiceRecord, ...],
    ) -> None:
        prior = _typed_index(self._communication_accounting, "user_id")
        updated = []

        for record in records:
            key = identifier_key(record.user_id)
            before = prior[key]
            active = int(record.demand_bit_per_s > 0.0)
            updated.append(CommunicationAccountingState(
                record.user_id,
                before.active_demand_slots + active,
                before.shortfall_sum + (
                    record.normalized_shortfall if active else 0.0
                ),
                before.residual_sum + (
                    record.residual if active else 0.0
                ),
            ))

        self._communication_accounting = tuple(sorted(
            updated, key=lambda item: identifier_key(item.user_id),
        ))

    def _prospective_session_id(self) -> str:
        assert self._trace is not None
        return f"{self._trace.trace_id}:session:{self._next_session_counter}"

    def _require_reset(self) -> None:
        if self._trace is None:
            raise EnvironmentValidationError("environment must be reset before use")

    def _require_ready(self) -> None:
        self._require_reset()
        if not self._prepared:
            raise EnvironmentValidationError("environment current slot is not prepared")