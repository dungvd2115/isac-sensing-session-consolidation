"""Decision-to-decision rollout collection and regime-separated validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, isfinite
from statistics import fmean, pstdev
from typing import Callable, Iterable

import torch

from isac_ssc.algorithms.buffers import (
    ConstraintLayout, EpisodeTotals, FactorCreditTransition,
    PreparedRollout, RolloutBuffer, RolloutTransition, StoredAction,
)
from isac_ssc.algorithms.losses import normalize_advantages
from isac_ssc.baselines.ppo_common_trace import CommonTracePPOAgent
from isac_ssc.baselines.ppo_joint_credit import JointCreditPPOAgent
from isac_ssc.envs.action_masks import ActionMaskSnapshot
from isac_ssc.envs.action_space import ActionType, EnvironmentAction, identifier_key
from isac_ssc.envs.dynamics import PrimitiveTrace
from isac_ssc.envs.isac_ssc_env import CommunicationServiceRecord, ISACSSCEnv, StepResult
from isac_ssc.envs.observation import ObservationSnapshot
from isac_ssc.utils.config import CanonicalConfig, ConstrainedPPOConfig
from isac_ssc.utils.seeding import SeedContract


class RolloutCollectionError(ValueError):
    """Raised when public rollout collection cannot preserve the CMDP contract."""


_ACTION_TYPES = (
    ActionType.MERGE, ActionType.CREATE,
    ActionType.DEFER, ActionType.REJECT,
)


def _key(value: object) -> str:
    kind, text = identifier_key(value)
    return f"{'int' if kind == 0 else 'str'}:{text}"


def _action_label(action: EnvironmentAction | None) -> str:
    if action is None:
        return "none"
    session = "" if action.session_id is None else f":{_key(action.session_id)}"
    profile = "" if action.profile_id is None else f":{action.profile_id}"
    return f"{action.action_type.value}{session}{profile}"


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


@dataclass(frozen=True, slots=True)
class TenantMetrics:
    tenant_id: str
    sla_violation_budget: float
    arrived: int
    accepted: int
    completed: int
    rejected: int
    expired: int
    failed: int
    first_violated: int
    residual_total: float

    @property
    def acceptance_ratio(self) -> float | None:
        return _ratio(self.accepted, self.arrived)

    @property
    def completion_ratio(self) -> float | None:
        return _ratio(self.completed, self.arrived)

    @property
    def violation_rate(self) -> float | None:
        return _ratio(self.first_violated, self.accepted)

    @property
    def positive_residual(self) -> float:
        return max(0.0, self.residual_total)


@dataclass(frozen=True, slots=True)
class CommunicationUserMetrics:
    user_id: str
    normalized_shortfall_budget: float
    active_demand_slots: int
    demand_bit_per_s_slot_sum: float
    allocated_bandwidth_hz_slot_sum: float
    allocated_power_w_slot_sum: float
    achievable_rate_bit_per_s_slot_sum: float
    served_rate_bit_per_s_slot_sum: float
    normalized_shortfall_sum: float
    residual_total: float

    @property
    def mean_active_demand_bit_per_s(self) -> float | None:
        return None if self.active_demand_slots == 0 else self.demand_bit_per_s_slot_sum / self.active_demand_slots

    @property
    def mean_active_allocated_bandwidth_hz(self) -> float | None:
        return None if self.active_demand_slots == 0 else self.allocated_bandwidth_hz_slot_sum / self.active_demand_slots

    @property
    def mean_active_allocated_power_w(self) -> float | None:
        return None if self.active_demand_slots == 0 else self.allocated_power_w_slot_sum / self.active_demand_slots

    @property
    def mean_active_achievable_rate_bit_per_s(self) -> float | None:
        return None if self.active_demand_slots == 0 else self.achievable_rate_bit_per_s_slot_sum / self.active_demand_slots

    @property
    def mean_active_served_rate_bit_per_s(self) -> float | None:
        return None if self.active_demand_slots == 0 else self.served_rate_bit_per_s_slot_sum / self.active_demand_slots

    @property
    def mean_normalized_shortfall(self) -> float | None:
        return None if self.active_demand_slots == 0 else self.normalized_shortfall_sum / self.active_demand_slots

    @property
    def positive_residual(self) -> float:
        return max(0.0, self.residual_total)


@dataclass(frozen=True, slots=True)
class EpisodeCollectionMetrics:
    episode_index: int
    trace_id: str
    root_seed: int
    arrival_regime: str
    physical_slots: int
    focal_decisions: int
    reward_total: float
    completed_value_total: float
    arrived_request_value_total: float
    sensing_resource_cost_total: float
    sensing_bandwidth_hz_slot_sum: float
    sensing_bandwidth_hz_max: float
    sensing_power_w_slot_sum: float
    sensing_power_w_max: float
    slots_with_session_update: int
    session_update_count: int
    tracking_prediction_count: int
    post_slot_active_session_count_sum: int
    post_slot_active_session_count_max: int
    arrived: int
    accepted: int
    completed: int
    rejected: int
    expired: int
    failed: int
    valid_outputs: int
    first_violations: int
    created_sessions: int
    tenant_residual_totals: tuple[float, ...]
    communication_residual_totals: tuple[float, ...]
    tenants: tuple[TenantMetrics, ...]
    communication_users: tuple[CommunicationUserMetrics, ...]
    action_counts: tuple[tuple[str, int], ...]
    valid_actions: int
    invalid_actions: int

    @property
    def valid_action_rate(self) -> float:
        total = self.valid_actions + self.invalid_actions
        return 1.0 if total == 0 else self.valid_actions / total

    @property
    def positive_constraint_excess(self) -> float:
        return sum(max(0.0, value) for value in (*self.tenant_residual_totals, *self.communication_residual_totals))

    @property
    def reward_per_slot(self) -> float:
        return self.reward_total / self.physical_slots

    @property
    def completed_value_per_slot(self) -> float:
        return self.completed_value_total / self.physical_slots

    @property
    def normalized_completed_value(self) -> float | None:
        return _ratio(self.completed_value_total, self.arrived_request_value_total)

    @property
    def sensing_resource_cost_per_slot(self) -> float:
        return self.sensing_resource_cost_total / self.physical_slots

    @property
    def sensing_bandwidth_hz_mean(self) -> float:
        return self.sensing_bandwidth_hz_slot_sum / self.physical_slots

    @property
    def sensing_power_w_mean(self) -> float:
        return self.sensing_power_w_slot_sum / self.physical_slots

    @property
    def post_slot_active_session_count_mean(self) -> float:
        return self.post_slot_active_session_count_sum / self.physical_slots

    @property
    def acceptance_ratio(self) -> float | None:
        return _ratio(self.accepted, self.arrived)

    @property
    def completion_ratio(self) -> float | None:
        return _ratio(self.completed, self.arrived)

    @property
    def rejection_ratio(self) -> float | None:
        return _ratio(self.rejected, self.arrived)

    @property
    def requests_served_per_created_session(self) -> float | None:
        return _ratio(self.accepted, self.created_sessions)

    @property
    def network_mean_user_shortfall(self) -> float | None:
        values = tuple(item.mean_normalized_shortfall for item in self.communication_users if item.mean_normalized_shortfall is not None)
        return None if not values else fmean(values)

    @property
    def fraction_users_within_budget(self) -> float | None:
        active = tuple(item for item in self.communication_users if item.mean_normalized_shortfall is not None)
        return None if not active else sum(item.mean_normalized_shortfall <= item.normalized_shortfall_budget for item in active) / len(active)


@dataclass(frozen=True, slots=True)
class RolloutCollectionMetrics:
    episodes: int
    physical_slots: int
    focal_decisions: int
    reward_total: float
    completed_value_total: float
    arrived_request_value_total: float
    sensing_resource_cost_total: float
    sensing_bandwidth_hz_slot_sum: float
    sensing_bandwidth_hz_max: float
    sensing_power_w_slot_sum: float
    sensing_power_w_max: float
    slots_with_session_update: int
    session_update_count: int
    tracking_prediction_count: int
    post_slot_active_session_count_sum: int
    post_slot_active_session_count_max: int
    arrived: int
    accepted: int
    completed: int
    rejected: int
    expired: int
    failed: int
    valid_outputs: int
    first_violations: int
    created_sessions: int
    tenant_residual_totals: tuple[float, ...]
    communication_residual_totals: tuple[float, ...]
    tenants: tuple[TenantMetrics, ...]
    communication_users: tuple[CommunicationUserMetrics, ...]
    action_counts: tuple[tuple[str, int], ...]
    valid_actions: int
    invalid_actions: int

    @property
    def valid_action_rate(self) -> float:
        total = self.valid_actions + self.invalid_actions
        return 1.0 if total == 0 else self.valid_actions / total

    @property
    def positive_constraint_excess(self) -> float:
        return sum(max(0.0, value) for value in (*self.tenant_residual_totals, *self.communication_residual_totals))

    @property
    def reward_per_slot(self) -> float:
        return self.reward_total / self.physical_slots

    @property
    def reward_per_episode(self) -> float:
        return self.reward_total / self.episodes

    @property
    def completed_value_per_slot(self) -> float:
        return self.completed_value_total / self.physical_slots

    @property
    def completed_value_per_episode(self) -> float:
        return self.completed_value_total / self.episodes

    @property
    def normalized_completed_value(self) -> float | None:
        return _ratio(self.completed_value_total, self.arrived_request_value_total)

    @property
    def sensing_resource_cost_per_slot(self) -> float:
        return self.sensing_resource_cost_total / self.physical_slots

    @property
    def sensing_resource_cost_per_episode(self) -> float:
        return self.sensing_resource_cost_total / self.episodes

    @property
    def sensing_bandwidth_hz_mean(self) -> float:
        return self.sensing_bandwidth_hz_slot_sum / self.physical_slots

    @property
    def sensing_power_w_mean(self) -> float:
        return self.sensing_power_w_slot_sum / self.physical_slots

    @property
    def post_slot_active_session_count_mean(self) -> float:
        return self.post_slot_active_session_count_sum / self.physical_slots

    @property
    def acceptance_ratio(self) -> float | None:
        return _ratio(self.accepted, self.arrived)

    @property
    def completion_ratio(self) -> float | None:
        return _ratio(self.completed, self.arrived)

    @property
    def rejection_ratio(self) -> float | None:
        return _ratio(self.rejected, self.arrived)

    @property
    def requests_served_per_created_session(self) -> float | None:
        return _ratio(self.accepted, self.created_sessions)

    @property
    def network_mean_user_shortfall(self) -> float | None:
        values = tuple(item.mean_normalized_shortfall for item in self.communication_users if item.mean_normalized_shortfall is not None)
        return None if not values else fmean(values)

    @property
    def fraction_users_within_budget(self) -> float | None:
        active = tuple(item for item in self.communication_users if item.mean_normalized_shortfall is not None)
        return None if not active else sum(item.mean_normalized_shortfall <= item.normalized_shortfall_budget for item in active) / len(active)


@dataclass(frozen=True, slots=True)
class CollectedRollout:
    rollout: PreparedRollout
    metrics: RolloutCollectionMetrics
    episodes: tuple[EpisodeCollectionMetrics, ...]
    focal_observations: tuple[ObservationSnapshot, ...]
    next_episode_index: int


@dataclass(frozen=True, slots=True)
class ValidationEpisode:
    policy: str
    physical_slot: int
    replicate: int
    metrics: EpisodeCollectionMetrics
    feasible_merge_action_count: int | None = None
    decision_states_with_merge_opportunity: int | None = None
    action_sequence: tuple[str, ...] = ()
    focal_sequence: tuple[str | None, ...] = ()

    @property
    def trace_id(self) -> str:
        return self.metrics.trace_id

    @property
    def root_seed(self) -> int:
        return self.metrics.root_seed

    @property
    def arrival_regime(self) -> str:
        return self.metrics.arrival_regime

    @property
    def reward_total(self) -> float:
        return self.metrics.reward_total


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    policy: str
    physical_slot: int
    arrival_regime: str
    episode_count: int
    mean_return: float
    std_return: float | None
    mean_completed_value: float
    mean_normalized_completed_value: float | None
    mean_sensing_resource_cost: float
    mean_positive_constraint_excess: float
    mean_reward_per_slot: float
    mean_completed_value_per_slot: float
    mean_sensing_resource_cost_per_slot: float
    mean_network_user_shortfall: float | None
    mean_fraction_users_within_budget: float | None
    valid_action_rate: float
    all_finite: bool


@dataclass(frozen=True, slots=True)
class ValidationReport:
    policy: str
    physical_slot: int
    episodes: tuple[ValidationEpisode, ...]
    regimes: tuple[ValidationSummary, ...]
    overall: ValidationSummary

    @property
    def valid_action_rate(self) -> float:
        return self.overall.valid_action_rate

    @property
    def all_finite(self) -> bool:
        return self.overall.all_finite and all(item.all_finite for item in self.regimes)


class _Accumulator:
    def __init__(self, layout: ConstraintLayout, config: CanonicalConfig, trace: PrimitiveTrace) -> None:
        self.layout = layout
        self.slots = self.decisions = self.valid_actions = self.invalid_actions = 0
        self.reward = self.completed_value = self.arrived_request_value = self.sensing_cost = 0.0
        self.sensing_bandwidth = self.sensing_bandwidth_max = 0.0
        self.sensing_power = self.sensing_power_max = 0.0
        self.slots_with_session_update = self.session_updates = self.tracking_predictions = 0
        self.post_slot_active_session_count_sum = self.post_slot_active_session_count_max = 0
        self.arrived = self.accepted = self.completed = self.rejected = 0
        self.expired = self.failed = self.valid_outputs = self.first_violations = 0
        self.tenant = torch.zeros(layout.tenant_count, dtype=torch.float32)
        self.communication = torch.zeros(layout.communication_count, dtype=torch.float32)
        self.actions = {kind: 0 for kind in ActionType}
        requests = trace.materialized_requests(config)
        self.request_tenant = {_key(item.request_id): _key(item.tenant_id) for item in requests}
        self.request_values = {_key(item.request_id): item.completion_value for item in requests}
        self.tenant_budgets = {_key(item.tenant_id): item.sla_violation_budget for item in config.tenants}
        self.tenant_counts = {
            _key(item): {name: 0 for name in ("arrived", "accepted", "completed", "rejected", "expired", "failed", "first_violated")}
            for item in layout.tenant_ids
        }
        shortfall_budget = float(config.communication["normalized_shortfall_budget"])
        self.user_budgets = {_key(item): shortfall_budget for item in layout.communication_user_ids}
        self.users = {_key(item): [0.0] * 8 for item in layout.communication_user_ids}

    def _count_tenant_events(self, request_ids: tuple[object, ...], name: str) -> None:
        for request_id in request_ids:
            tenant = self.request_tenant.get(_key(request_id))
            if tenant is None or tenant not in self.tenant_counts:
                raise RolloutCollectionError("request event cannot be mapped to the canonical tenant layout")
            self.tenant_counts[tenant][name] += 1

    def add(self, result: StepResult, post_slot_active_session_count: int) -> None:
        self.slots += 1
        self.reward += result.reward
        self.completed_value += result.completed_value
        self.sensing_cost += result.sensing_resource_cost
        usage = result.sensing_resource_usage
        self.sensing_bandwidth += usage.sensing_bandwidth_hz
        self.sensing_bandwidth_max = max(self.sensing_bandwidth_max, usage.sensing_bandwidth_hz)
        self.sensing_power += usage.sensing_power_w
        self.sensing_power_max = max(self.sensing_power_max, usage.sensing_power_w)
        self.slots_with_session_update += int(bool(result.session_updates))
        self.session_updates += len(result.session_updates)
        self.tracking_predictions += len(result.tracking_prediction_session_ids)
        self.post_slot_active_session_count_sum += post_slot_active_session_count
        self.post_slot_active_session_count_max = max(self.post_slot_active_session_count_max, post_slot_active_session_count)
        for attribute, name in (
            (result.arrived_request_ids, "arrived"), (result.accepted_request_ids, "accepted"),
            (result.completed_request_ids, "completed"), (result.rejected_request_ids, "rejected"),
            (result.expired_request_ids, "expired"), (result.failed_request_ids, "failed"),
            (result.first_violation_request_ids, "first_violated"),
        ):
            self._count_tenant_events(attribute, name)
        self.arrived += len(result.arrived_request_ids)
        self.arrived_request_value += sum(self.request_values[_key(item)] for item in result.arrived_request_ids)
        self.accepted += len(result.accepted_request_ids)
        self.completed += len(result.completed_request_ids)
        self.rejected += len(result.rejected_request_ids)
        self.expired += len(result.expired_request_ids)
        self.failed += len(result.failed_request_ids)
        self.valid_outputs += len(result.valid_output_request_ids)
        self.first_violations += len(result.first_violation_request_ids)
        self.tenant += self.layout.pack_tenant_residuals(result.tenant_sla_residuals)
        self.communication += self.layout.pack_communication_residuals(result.communication_qos_residuals)
        for service in result.communication_service:
            self._add_user(service)

    def _add_user(self, service: CommunicationServiceRecord) -> None:
        values = self.users[_key(service.user_id)]
        active = int(service.demand_bit_per_s > 0.0)
        values[0] += active
        for index, value in enumerate((
            service.demand_bit_per_s, service.allocated_bandwidth_hz, service.allocated_power_w,
            service.achievable_rate_bit_per_s, service.served_rate_bit_per_s,
            service.normalized_shortfall if active else 0.0, service.residual if active else 0.0,
        ), start=1):
            values[index] += value

    def episode(self, trace: PrimitiveTrace, episode_index: int) -> EpisodeCollectionMetrics:
        tenant_residuals = tuple(map(float, self.tenant))
        communication_residuals = tuple(map(float, self.communication))
        tenants = tuple(
            TenantMetrics(
                _key(tenant_id), self.tenant_budgets[_key(tenant_id)],
                self.tenant_counts[_key(tenant_id)]["arrived"], self.tenant_counts[_key(tenant_id)]["accepted"],
                self.tenant_counts[_key(tenant_id)]["completed"], self.tenant_counts[_key(tenant_id)]["rejected"],
                self.tenant_counts[_key(tenant_id)]["expired"], self.tenant_counts[_key(tenant_id)]["failed"],
                self.tenant_counts[_key(tenant_id)]["first_violated"], tenant_residuals[index],
            )
            for index, tenant_id in enumerate(self.layout.tenant_ids)
        )
        users = tuple(
            CommunicationUserMetrics(_key(user_id), self.user_budgets[_key(user_id)], int(self.users[_key(user_id)][0]), *self.users[_key(user_id)][1:])
            for user_id in self.layout.communication_user_ids
        )
        return EpisodeCollectionMetrics(
            episode_index, trace.trace_id, trace.root_seed, trace.arrival_regime, self.slots, self.decisions,
            self.reward, self.completed_value, self.arrived_request_value, self.sensing_cost, self.sensing_bandwidth,
            self.sensing_bandwidth_max, self.sensing_power, self.sensing_power_max,
            self.slots_with_session_update, self.session_updates, self.tracking_predictions,
            self.post_slot_active_session_count_sum, self.post_slot_active_session_count_max,
            self.arrived, self.accepted, self.completed, self.rejected, self.expired, self.failed,
            self.valid_outputs, self.first_violations, self.actions[ActionType.CREATE], tenant_residuals,
            communication_residuals, tenants, users,
            tuple((kind.value, self.actions[kind]) for kind in ActionType),
            self.valid_actions, self.invalid_actions,
        )


def _layout_for_trace(config: CanonicalConfig, trace: PrimitiveTrace) -> ConstraintLayout:
    tenants = tuple(sorted((item.tenant_id for item in config.tenants), key=identifier_key))
    users = tuple(sorted({item.user_id for item in trace.communication_states}, key=identifier_key))
    return ConstraintLayout(tenants, users)


def collect_episode(
    env: ISACSSCEnv, agent: JointCreditPPOAgent, trace: PrimitiveTrace, layout: ConstraintLayout,
    *, deterministic: bool = False, episode_index: int = 0,
    decision_observer: Callable[[ActionMaskSnapshot | None, EnvironmentAction | None], None] | None = None,
) -> tuple[tuple[RolloutTransition, ...], EpisodeTotals, EpisodeCollectionMetrics, tuple[ObservationSnapshot, ...]]:
    if _layout_for_trace(env.config, trace) != layout:
        raise RolloutCollectionError("trace constraint layout does not match the collector")
    observation = env.reset(trace)
    accumulator = _Accumulator(layout, env.config, trace)
    transitions: list[RolloutTransition] = []
    focal_observations: list[ObservationSnapshot] = []
    with torch.no_grad():
        while not env.terminated:
            if observation is None:
                if decision_observer is not None:
                    decision_observer(None, None)
                result = env.step(None)
                accumulator.add(result, len(env.state_snapshot().active_sessions))
                observation = env.current_observation()
                continue
            masks = env.current_action_masks()
            if masks is None:
                raise RolloutCollectionError("focal observation has no public action masks")
            selection, values = agent.select(observation, deterministic=deterministic)
            action = selection.actions[0]
            if action not in masks.feasible_actions:
                accumulator.invalid_actions += 1
                raise RolloutCollectionError("policy selected an infeasible action")
            accumulator.valid_actions += 1
            accumulator.decisions += 1
            accumulator.actions[action.action_type] += 1
            focal_observations.append(observation)
            if decision_observer is not None:
                decision_observer(masks, action)
            interval_reward = 0.0
            interval_tenant = torch.zeros(layout.tenant_count, dtype=torch.float32)
            interval_communication = torch.zeros(layout.communication_count, dtype=torch.float32)
            span = 0
            result = env.step(action)
            while True:
                accumulator.add(result, len(env.state_snapshot().active_sessions))
                interval_reward += result.reward
                interval_tenant += layout.pack_tenant_residuals(result.tenant_sla_residuals)
                interval_communication += layout.pack_communication_residuals(result.communication_qos_residuals)
                span += 1
                if result.terminated or result.next_observation is not None:
                    break
                if decision_observer is not None:
                    decision_observer(None, None)
                result = env.step(None)
            transitions.append(RolloutTransition(
                observation, StoredAction.from_indices(selection.indices), interval_reward,
                tuple(map(float, interval_tenant)), tuple(map(float, interval_communication)),
                result.next_observation, result.terminated, span, float(selection.log_probability[0]),
                float(values.reward_value[0]), tuple(map(float, values.sensing_sla_values[0])),
                tuple(map(float, values.communication_qos_values[0])),
            ))
            observation = result.next_observation
    if accumulator.slots != trace.horizon_slots:
        raise RolloutCollectionError("episode did not consume the complete primitive trace")
    metrics = accumulator.episode(trace, episode_index)
    values = (
        metrics.reward_total, metrics.completed_value_total, metrics.arrived_request_value_total, metrics.sensing_resource_cost_total,
        *metrics.tenant_residual_totals, *metrics.communication_residual_totals,
    )
    if not all(isfinite(value) for value in values):
        raise RolloutCollectionError("episode accounting is non-finite")
    totals = EpisodeTotals(
        metrics.physical_slots, metrics.reward_total,
        metrics.tenant_residual_totals, metrics.communication_residual_totals,
    )
    return tuple(transitions), totals, metrics, tuple(focal_observations)


def _factor_credit_transition(
    selection, prefixes, action: StoredAction,
    layout: ConstraintLayout,
) -> FactorCreditTransition:
    type_index = action.action_type_index
    type_constraints = tuple(map(
        float, prefixes.type_constraint_values[0, type_index],
    ))
    if _ACTION_TYPES[type_index] is ActionType.MERGE:
        session_index = action.merge_session_index
        session_reward = float(
            prefixes.merge_session_reward_values[0, session_index],
        )
        session_constraints = tuple(map(
            float,
            prefixes.merge_session_constraint_values[0, session_index],
        ))
    else:
        session_reward = 0.0
        session_constraints = (0.0,) * layout.constraint_count
    components = selection.factor_log_probabilities
    return FactorCreditTransition(
        float(components.action_type[0]),
        float(components.merge_session[0]),
        float(components.profile[0]),
        float(prefixes.type_reward_values[0, type_index]),
        type_constraints,
        session_reward,
        session_constraints,
        bool(components.merge_session_applicable[0]),
        bool(components.profile_applicable[0]),
    )


def collect_common_trace_episode(
    env: ISACSSCEnv, agent: CommonTracePPOAgent,
    trace: PrimitiveTrace, layout: ConstraintLayout, *,
    deterministic: bool = False, episode_index: int = 0,
    decision_observer: Callable[[ActionMaskSnapshot | None, EnvironmentAction | None], None] | None = None,
    slot_reward_observer: Callable[[float], None] | None = None,
) -> tuple[
    tuple[RolloutTransition, ...], EpisodeTotals,
    EpisodeCollectionMetrics, tuple[ObservationSnapshot, ...],
]:
    if _layout_for_trace(env.config, trace) != layout:
        raise RolloutCollectionError(
            "trace constraint layout does not match the collector"
        )
    observation = env.reset(trace)
    accumulator = _Accumulator(layout, env.config, trace)
    transitions: list[RolloutTransition] = []
    focal_observations: list[ObservationSnapshot] = []
    with torch.no_grad():
        while not env.terminated:
            if observation is None:
                if decision_observer is not None:
                    decision_observer(None, None)
                result = env.step(None)
                if slot_reward_observer is not None:
                    slot_reward_observer(float(result.reward))
                accumulator.add(
                    result, len(env.state_snapshot().active_sessions),
                )
                observation = env.current_observation()
                continue
            masks = env.current_action_masks()
            if masks is None:
                raise RolloutCollectionError(
                    "focal observation has no public action masks"
                )
            selection, values, prefixes = agent.select(
                observation, deterministic=deterministic,
            )
            action = selection.actions[0]
            if action not in masks.feasible_actions:
                accumulator.invalid_actions += 1
                raise RolloutCollectionError(
                    "policy selected an infeasible action"
                )
            accumulator.valid_actions += 1
            accumulator.decisions += 1
            accumulator.actions[action.action_type] += 1
            focal_observations.append(observation)
            if decision_observer is not None:
                decision_observer(masks, action)
            interval_reward = 0.0
            interval_tenant = torch.zeros(
                layout.tenant_count, dtype=torch.float32,
            )
            interval_communication = torch.zeros(
                layout.communication_count, dtype=torch.float32,
            )
            span = 0
            result = env.step(action)
            while True:
                if slot_reward_observer is not None:
                    slot_reward_observer(float(result.reward))
                accumulator.add(
                    result, len(env.state_snapshot().active_sessions),
                )
                interval_reward += result.reward
                interval_tenant += layout.pack_tenant_residuals(
                    result.tenant_sla_residuals,
                )
                interval_communication += (
                    layout.pack_communication_residuals(
                        result.communication_qos_residuals,
                    )
                )
                span += 1
                if result.terminated or result.next_observation is not None:
                    break
                if decision_observer is not None:
                    decision_observer(None, None)
                result = env.step(None)
            stored_action = StoredAction.from_indices(selection.indices)
            transitions.append(RolloutTransition(
                observation,
                stored_action,
                interval_reward,
                tuple(map(float, interval_tenant)),
                tuple(map(float, interval_communication)),
                result.next_observation,
                result.terminated,
                span,
                float(selection.log_probability[0]),
                float(values.reward_value[0]),
                tuple(map(float, values.sensing_sla_values[0])),
                tuple(map(float, values.communication_qos_values[0])),
                _factor_credit_transition(
                    selection, prefixes, stored_action, layout,
                ),
            ))
            observation = result.next_observation
    if accumulator.slots != trace.horizon_slots:
        raise RolloutCollectionError(
            "episode did not consume the complete primitive trace"
        )
    metrics = accumulator.episode(trace, episode_index)
    values = (
        metrics.reward_total,
        metrics.completed_value_total,
        metrics.arrived_request_value_total,
        metrics.sensing_resource_cost_total,
        *metrics.tenant_residual_totals,
        *metrics.communication_residual_totals,
    )
    if not all(isfinite(value) for value in values):
        raise RolloutCollectionError(
            "episode accounting is non-finite"
        )
    totals = EpisodeTotals(
        metrics.physical_slots,
        metrics.reward_total,
        metrics.tenant_residual_totals,
        metrics.communication_residual_totals,
    )
    return (
        tuple(transitions), totals, metrics,
        tuple(focal_observations),
    )


def _aggregate(episodes: tuple[EpisodeCollectionMetrics, ...]) -> RolloutCollectionMetrics:
    tenant_residuals = [0.0] * len(episodes[0].tenant_residual_totals)
    communication_residuals = [0.0] * len(episodes[0].communication_residual_totals)
    actions = {kind.value: 0 for kind in ActionType}
    tenant_values = {
        item.tenant_id: [item.sla_violation_budget, 0, 0, 0, 0, 0, 0, 0, 0.0]
        for item in episodes[0].tenants
    }
    user_values = {
        item.user_id: [item.normalized_shortfall_budget, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for item in episodes[0].communication_users
    }
    for episode in episodes:
        tenant_residuals = [left + right for left, right in zip(tenant_residuals, episode.tenant_residual_totals, strict=True)]
        communication_residuals = [left + right for left, right in zip(communication_residuals, episode.communication_residual_totals, strict=True)]
        for name, count in episode.action_counts:
            actions[name] += count
        for item in episode.tenants:
            values = tenant_values[item.tenant_id]
            for index, value in enumerate((item.arrived, item.accepted, item.completed, item.rejected, item.expired, item.failed, item.first_violated), start=1):
                values[index] += value
            values[8] += item.residual_total
        for item in episode.communication_users:
            values = user_values[item.user_id]
            for index, value in enumerate((
                item.active_demand_slots, item.demand_bit_per_s_slot_sum, item.allocated_bandwidth_hz_slot_sum,
                item.allocated_power_w_slot_sum, item.achievable_rate_bit_per_s_slot_sum,
                item.served_rate_bit_per_s_slot_sum, item.normalized_shortfall_sum, item.residual_total,
            ), start=1):
                values[index] += value
    tenants = tuple(TenantMetrics(name, values[0], *values[1:]) for name, values in tenant_values.items())
    users = tuple(CommunicationUserMetrics(name, values[0], int(values[1]), *values[2:]) for name, values in user_values.items())
    return RolloutCollectionMetrics(
        len(episodes), sum(item.physical_slots for item in episodes), sum(item.focal_decisions for item in episodes),
        sum(item.reward_total for item in episodes), sum(item.completed_value_total for item in episodes),
        sum(item.arrived_request_value_total for item in episodes), sum(item.sensing_resource_cost_total for item in episodes),
        sum(item.sensing_bandwidth_hz_slot_sum for item in episodes),
        max(item.sensing_bandwidth_hz_max for item in episodes), sum(item.sensing_power_w_slot_sum for item in episodes),
        max(item.sensing_power_w_max for item in episodes), sum(item.slots_with_session_update for item in episodes),
        sum(item.session_update_count for item in episodes), sum(item.tracking_prediction_count for item in episodes),
        sum(item.post_slot_active_session_count_sum for item in episodes), max(item.post_slot_active_session_count_max for item in episodes),
        sum(item.arrived for item in episodes), sum(item.accepted for item in episodes), sum(item.completed for item in episodes),
        sum(item.rejected for item in episodes), sum(item.expired for item in episodes), sum(item.failed for item in episodes),
        sum(item.valid_outputs for item in episodes), sum(item.first_violations for item in episodes),
        sum(item.created_sessions for item in episodes), tuple(tenant_residuals), tuple(communication_residuals),
        tenants, users, tuple(actions.items()), sum(item.valid_actions for item in episodes),
        sum(item.invalid_actions for item in episodes),
    )


def collect_training_rollout(
    env: ISACSSCEnv, agent: JointCreditPPOAgent, layout: ConstraintLayout,
    trace_factory: Callable[[int, str], PrimitiveTrace], start_episode_index: int,
    target_physical_slots: int, algorithm: ConstrainedPPOConfig,
    regimes: tuple[str, ...] = ("independent", "clustered"),
) -> CollectedRollout:
    if target_physical_slots < 1 or not regimes:
        raise RolloutCollectionError("rollout target and regimes must be non-empty")
    buffer = RolloutBuffer(layout, algorithm.ppo.discount, algorithm.ppo.gae_lambda)
    episodes: list[EpisodeCollectionMetrics] = []
    observations: list[ObservationSnapshot] = []
    episode_index, collected_slots = start_episode_index, 0
    while collected_slots < target_physical_slots:
        regime = regimes[episode_index % len(regimes)]
        trace = trace_factory(episode_index, regime)
        transitions, totals, metrics, focal = collect_episode(env, agent, trace, layout, episode_index=episode_index)
        for transition in transitions:
            buffer.append(transition)
        buffer.record_episode_totals(totals)
        episodes.append(metrics)
        observations.extend(focal)
        collected_slots += metrics.physical_slots
        episode_index += 1
    episode_values = tuple(episodes)
    return CollectedRollout(buffer.finalize(), _aggregate(episode_values), episode_values, tuple(observations), episode_index)


def _common_trace_group_sizes(episode_count: int) -> tuple[int, ...]:
    if episode_count < 1:
        raise RolloutCollectionError("common-trace episode count must be positive")
    if episode_count == 1:
        return (1,)
    if episode_count % 2 == 0:
        return (2,) * (episode_count // 2)
    return (3,) + (2,) * ((episode_count - 3) // 2)


def _discounted_slot_suffix(rewards: list[float], discount: float) -> tuple[float, ...]:
    suffix = [0.0] * (len(rewards) + 1)
    for index in range(len(rewards) - 1, -1, -1):
        suffix[index] = float(rewards[index]) + discount * suffix[index + 1]
    return tuple(suffix)


def _decision_start_slots(transitions: tuple[RolloutTransition, ...], horizon_slots: int) -> tuple[int, ...]:
    if not transitions:
        return ()
    first = horizon_slots - sum(item.physical_slot_span for item in transitions)
    if first < 0:
        raise RolloutCollectionError("decision spans exceed primitive-trace horizon")
    starts: list[int] = []
    slot = first
    for transition in transitions:
        starts.append(slot)
        slot += transition.physical_slot_span
    if slot != horizon_slots:
        raise RolloutCollectionError("decision spans do not terminate at primitive-trace horizon")
    return tuple(starts)


def collect_common_trace_training_rollout(
    env: ISACSSCEnv, agent: CommonTracePPOAgent,
    layout: ConstraintLayout,
    trace_factory: Callable[[int, str], PrimitiveTrace],
    start_episode_index: int, target_physical_slots: int,
    algorithm: ConstrainedPPOConfig,
    regimes: tuple[str, ...] = ("independent", "clustered"),
) -> CollectedRollout:
    if target_physical_slots < 1 or not regimes:
        raise RolloutCollectionError("rollout target and regimes must be non-empty")
    first_trace = trace_factory(start_episode_index, regimes[0])
    horizon = first_trace.horizon_slots
    episode_budget = max(1, ceil(target_physical_slots / horizon))
    groups = _common_trace_group_sizes(episode_budget)
    buffer = RolloutBuffer(
        layout, algorithm.ppo.discount, algorithm.ppo.gae_lambda,
        factor_normalization_epsilon=algorithm.normalization.epsilon,
    )
    episodes: list[EpisodeCollectionMetrics] = []
    observations: list[ObservationSnapshot] = []
    common_advantages: list[float | None] = []
    episode_index = start_episode_index
    rollout_number = start_episode_index // episode_budget
    for group_index, replicate_count in enumerate(groups):
        regime = regimes[(rollout_number + group_index) % len(regimes)]
        trace = trace_factory(episode_index, regime)
        if trace.horizon_slots != horizon:
            raise RolloutCollectionError("common-trace groups require a fixed primitive-trace horizon")
        group_transitions: list[tuple[RolloutTransition, ...]] = []
        group_slot_rewards: list[list[float]] = []
        for _ in range(replicate_count):
            slot_rewards: list[float] = []
            transitions, totals, metrics, focal = collect_common_trace_episode(
                env, agent, trace, layout, episode_index=episode_index,
                slot_reward_observer=slot_rewards.append,
            )
            if len(slot_rewards) != trace.horizon_slots:
                raise RolloutCollectionError("common-trace slot rewards do not cover the full primitive trace")
            group_transitions.append(transitions)
            group_slot_rewards.append(slot_rewards)
            for transition in transitions:
                buffer.append(transition)
            buffer.record_episode_totals(totals)
            episodes.append(metrics)
            observations.extend(focal)
            episode_index += 1
        if replicate_count == 1:
            common_advantages.extend([None] * len(group_transitions[0]))
            continue
        suffixes = [
            _discounted_slot_suffix(values, algorithm.ppo.discount)
            for values in group_slot_rewards
        ]
        for replicate, transitions in enumerate(group_transitions):
            starts = _decision_start_slots(transitions, trace.horizon_slots)
            peers = tuple(index for index in range(replicate_count) if index != replicate)
            for start_slot in starts:
                peer_baseline = fmean(suffixes[index][start_slot] for index in peers)
                common_advantages.append(suffixes[replicate][start_slot] - peer_baseline)
    prepared = buffer.finalize()
    if len(common_advantages) != prepared.transition_count:
        raise RolloutCollectionError("common-trace advantages do not align with rollout transitions")
    reward_advantages = torch.tensor([
        float(prepared.reward_advantages[index]) if value is None else value
        for index, value in enumerate(common_advantages)
    ], dtype=torch.float32)
    if not bool(torch.isfinite(reward_advantages).all()):
        raise RolloutCollectionError("common-trace reward advantages must be finite")
    factor = prepared.factor_credit
    if factor is None:
        raise RolloutCollectionError("common-trace rollout is missing factor-credit state")
    normalized = normalize_advantages(
        reward_advantages, prepared.constraint_advantages,
        factor.normalization_epsilon,
    )
    factor_reward = factor.reward_advantages.clone()
    factor_reward[:, 0] = reward_advantages
    factor_normalized_reward = factor.normalized_reward_advantages.clone()
    factor_normalized_reward[:, 0] = normalized.reward
    factor = replace(
        factor, reward_advantages=factor_reward,
        normalized_reward_advantages=factor_normalized_reward,
    )
    prepared = replace(
        prepared, reward_advantages=reward_advantages,
        factor_credit=factor,
    )
    episode_values = tuple(episodes)
    return CollectedRollout(
        prepared, _aggregate(episode_values), episode_values,
        tuple(observations), episode_index,
    )


def _validation_episode(metrics: EpisodeCollectionMetrics, policy: str, physical_slot: int, replicate: int) -> ValidationEpisode:
    return ValidationEpisode(policy, physical_slot, replicate, metrics)


def _optional_mean(values: Iterable[float | None]) -> float | None:
    defined = tuple(value for value in values if value is not None)
    return None if not defined else fmean(defined)


def _regime_summary(policy: str, physical_slot: int, regime: str, episodes: tuple[ValidationEpisode, ...]) -> ValidationSummary:
    returns = tuple(item.metrics.reward_total for item in episodes)
    decisions = sum(item.metrics.focal_decisions for item in episodes)
    valid = 1.0 if decisions == 0 else sum(item.metrics.valid_action_rate * item.metrics.focal_decisions for item in episodes) / decisions
    normalized_value = _optional_mean(item.metrics.normalized_completed_value for item in episodes)
    network_shortfall = _optional_mean(item.metrics.network_mean_user_shortfall for item in episodes)
    users_within_budget = _optional_mean(item.metrics.fraction_users_within_budget for item in episodes)
    finite_values = (
        *returns, *(item.metrics.completed_value_total for item in episodes),
        *(item.metrics.sensing_resource_cost_total for item in episodes),
        *(item.metrics.positive_constraint_excess for item in episodes), valid,
        *(value for value in (normalized_value, network_shortfall, users_within_budget) if value is not None),
    )
    return ValidationSummary(
        policy, physical_slot, regime, len(episodes), fmean(returns),
        pstdev(returns) if len(returns) > 1 else 0.0,
        fmean(item.metrics.completed_value_total for item in episodes), normalized_value,
        fmean(item.metrics.sensing_resource_cost_total for item in episodes),
        fmean(item.metrics.positive_constraint_excess for item in episodes),
        fmean(item.metrics.reward_per_slot for item in episodes),
        fmean(item.metrics.completed_value_per_slot for item in episodes),
        fmean(item.metrics.sensing_resource_cost_per_slot for item in episodes),
        network_shortfall, users_within_budget, valid,
        all(isfinite(value) for value in finite_values),
    )


def _validation_report(policy: str, physical_slot: int, episodes: tuple[ValidationEpisode, ...]) -> ValidationReport:
    if not episodes:
        raise RolloutCollectionError("validation requires at least one episode")
    regime_order = tuple(dict.fromkeys(item.arrival_regime for item in episodes))
    regimes = tuple(
        _regime_summary(policy, physical_slot, regime, tuple(item for item in episodes if item.arrival_regime == regime))
        for regime in regime_order
    )
    macro_values = (
        fmean(item.mean_return for item in regimes), fmean(item.mean_completed_value for item in regimes),
        _optional_mean(item.mean_normalized_completed_value for item in regimes),
        fmean(item.mean_sensing_resource_cost for item in regimes), fmean(item.mean_positive_constraint_excess for item in regimes),
        fmean(item.mean_reward_per_slot for item in regimes), fmean(item.mean_completed_value_per_slot for item in regimes),
        fmean(item.mean_sensing_resource_cost_per_slot for item in regimes),
        _optional_mean(item.mean_network_user_shortfall for item in regimes),
        _optional_mean(item.mean_fraction_users_within_budget for item in regimes),
        fmean(item.valid_action_rate for item in regimes),
    )
    finite_macro = tuple(value for value in macro_values if value is not None)
    overall = ValidationSummary(
        policy, physical_slot, "overall", len(episodes), macro_values[0], None, *macro_values[1:],
        all(item.all_finite for item in regimes) and all(isfinite(value) for value in finite_macro),
    )
    return ValidationReport(policy, physical_slot, episodes, regimes, overall)


def evaluate_policy(
    config: CanonicalConfig, agent: JointCreditPPOAgent | CommonTracePPOAgent,
    traces: Iterable[PrimitiveTrace], *, physical_slot: int = 0,
    capture_decisions: bool = False,
) -> ValidationReport:
    trace_values = tuple(traces)
    if isinstance(agent, JointCreditPPOAgent):
        collector = collect_episode
    elif isinstance(agent, CommonTracePPOAgent):
        collector = collect_common_trace_episode
    else:
        raise RolloutCollectionError("unsupported learned-policy agent")
    training = agent.model.training
    agent.model.eval()
    try:
        with torch.inference_mode():
            episodes = []
            for index, trace in enumerate(trace_values):
                actions: list[str] = []
                focals: list[str | None] = []
                feasible_merges = 0
                merge_opportunities = 0

                def observe(
                    masks: ActionMaskSnapshot | None,
                    action: EnvironmentAction | None,
                ) -> None:
                    nonlocal feasible_merges, merge_opportunities
                    if masks is None:
                        focals.append(None)
                    else:
                        focals.append(_key(masks.focal_request_id))
                        merge_count = sum(
                            candidate.action_type is ActionType.MERGE
                            for candidate in masks.feasible_actions
                        )
                        feasible_merges += merge_count
                        merge_opportunities += int(merge_count > 0)
                    actions.append(_action_label(action))

                metrics = collector(
                    ISACSSCEnv(config), agent, trace,
                    _layout_for_trace(config, trace),
                    deterministic=True, episode_index=index,
                    decision_observer=observe if capture_decisions else None,
                )[2]
                episodes.append(ValidationEpisode(
                    "policy", physical_slot, 0, metrics,
                    feasible_merges if capture_decisions else None,
                    merge_opportunities if capture_decisions else None,
                    tuple(actions), tuple(focals),
                ))
            episodes = tuple(episodes)
    finally:
        agent.model.train(training)
    return _validation_report("policy", physical_slot, episodes)


def _random_episode(
    config: CanonicalConfig, trace: PrimitiveTrace, generator: torch.Generator,
    physical_slot: int, replicate: int, episode_index: int,
) -> ValidationEpisode:
    env, layout = ISACSSCEnv(config), _layout_for_trace(config, trace)
    observation = env.reset(trace)
    accumulator = _Accumulator(layout, config, trace)
    while not env.terminated:
        if observation is None:
            result = env.step(None)
        else:
            masks = env.current_action_masks()
            if masks is None or not masks.feasible_actions:
                raise RolloutCollectionError("random-valid policy has no feasible action")
            action = masks.feasible_actions[int(torch.randint(len(masks.feasible_actions), (1,), generator=generator))]
            accumulator.actions[action.action_type] += 1
            accumulator.decisions += 1
            accumulator.valid_actions += 1
            result = env.step(action)
        accumulator.add(result, len(env.state_snapshot().active_sessions))
        observation = result.next_observation
    return _validation_episode(accumulator.episode(trace, episode_index), "random_valid", physical_slot, replicate)


def evaluate_random_valid(
    config: CanonicalConfig, traces: Iterable[PrimitiveTrace], *, root_seed: int,
    replicates_per_trace: int, physical_slot: int = 0,
) -> ValidationReport:
    contract = SeedContract.from_config(config)
    episodes = []
    for trace_index, trace in enumerate(tuple(traces)):
        for replicate in range(replicates_per_trace):
            seed = contract.derive_uint64(root_seed, "random_valid", trace.root_seed, trace.arrival_regime, replicate)
            episodes.append(_random_episode(
                config, trace, torch.Generator().manual_seed(seed), physical_slot, replicate,
                trace_index * replicates_per_trace + replicate,
            ))
    return _validation_report("random_valid", physical_slot, tuple(episodes))