"""Heuristic, random-valid and selected-policy evaluation on shared primitive traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from statistics import mean, median, pstdev
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from isac_ssc.baselines import (
    greedy_incremental_cost, no_consolidation, sla_aware_greedy,
    static_compatibility_merge,
)
from isac_ssc.baselines.ppo_common_trace import build_common_trace_agent
from isac_ssc.baselines.ppo_joint_credit import build_joint_credit_agent
from isac_ssc.baselines.selectors import (
    ImmediateServiceCandidate, build_create_candidate, build_merge_candidate,
)
from isac_ssc.core.entities import EntityId, RequestState, SensingRequest
from isac_ssc.envs.action_masks import ActionMaskSnapshot
from isac_ssc.envs.action_space import ActionType, EnvironmentAction, identifier_key
from isac_ssc.envs.dynamics import PrimitiveTrace, generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import EnvironmentStateSnapshot, ISACSSCEnv
from isac_ssc.evaluation.metrics import request_state_counts, safe_ratio
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.training.checkpoint import checkpoint_sha256, load_policy_checkpoint, read_checkpoint_metadata
from isac_ssc.training.rollout import ValidationEpisode, evaluate_policy, evaluate_random_valid
from isac_ssc.utils.config import (
    COMMON_TRACE_METHOD, JOINT_CREDIT_METHOD, SUPPORTED_LEARNED_METHODS,
    CanonicalConfig, ConstrainedPPOConfig,
)
from isac_ssc.utils.seeding import SeedContract


BASELINE_REGISTRY: Mapping[str, Callable[..., EnvironmentAction]] = MappingProxyType({
    "no_consolidation": no_consolidation.select_action,
    "static_compatibility_merge": static_compatibility_merge.select_action,
    "greedy_incremental_cost": greedy_incremental_cost.select_action,
    "sla_aware_greedy": sla_aware_greedy.select_action,
})
DETERMINISTIC_ONLINE_HEURISTICS = tuple(BASELINE_REGISTRY)
RANDOM_VALID_NAME = "random_valid"
EVALUATION_METHODS = (*DETERMINISTIC_ONLINE_HEURISTICS, RANDOM_VALID_NAME)
DETERMINISTIC_METHOD_CATEGORY = "deterministic_online_heuristic"
STOCHASTIC_SANITY_METHOD_CATEGORY = "stochastic_sanity_baseline"
LEARNED_POLICY_NAME = JOINT_CREDIT_METHOD
LEARNED_POLICY_NAMES = SUPPORTED_LEARNED_METHODS
LEARNED_POLICY_CATEGORY = "learned_policy"
REPORT_METHOD_ORDER = (*EVALUATION_METHODS, *LEARNED_POLICY_NAMES)
_BOOTSTRAP_DOMAIN = "standalone_evaluation_bootstrap_v1"
_CI_METRICS = (
    "reward_total", "completed_value_total", "normalized_completed_value",
    "completion_ratio", "acceptance_ratio", "sensing_resource_cost_total",
    "positive_constraint_excess", "requests_per_created_session",
)


@dataclass(frozen=True, slots=True)
class BaselineEpisodeReport:
    baseline_name: str
    method_category: str
    replicate: int | None
    trace_id: str
    root_seed: int
    arrival_regime: str
    physical_step_count: int
    focal_decision_count: int
    no_request_slot_count: int
    merge_count: int
    create_count: int
    defer_count: int
    reject_count: int
    arrived_request_count: int
    arrived_request_value_total: float
    accepted_request_count: int
    completed_request_count: int
    failed_request_count: int
    active_request_count: int
    expired_request_count: int
    rejected_request_count: int
    terminal_waiting_request_count: int
    cumulative_completed_value: float
    normalized_completed_value: float | None
    cumulative_sensing_resource_cost: float
    cumulative_reward: float
    acceptance_ratio: float
    completion_ratio: float
    mean_active_session_count: float
    peak_active_session_count: int
    created_session_count: int
    requests_per_created_session: float | None
    feasible_merge_action_count: int | None
    decision_states_with_merge_opportunity: int | None
    tenant_sla_residual_total: float
    tenant_sla_positive_excess: float
    communication_qos_residual_total: float
    communication_qos_positive_excess: float
    positive_constraint_excess: float
    valid_action_count: int
    invalid_action_count: int
    valid_action_rate: float
    all_finite: bool
    terminal_request_state_counts: tuple[tuple[str, int], ...]
    per_tenant_accounting: tuple[tuple[str, int, int, int, float], ...]
    per_user_communication_accounting: tuple[tuple[str, int, float, float], ...]
    action_sequence: tuple[str, ...]
    focal_sequence: tuple[str | None, ...]

    @property
    def method_name(self) -> str:
        return self.baseline_name


@dataclass(frozen=True, slots=True)
class MetricConfidenceInterval:
    metric: str
    sample_count: int
    mean: float | None
    median: float | None
    standard_deviation: float | None
    ci95_lower: float | None
    ci95_upper: float | None


@dataclass(frozen=True, slots=True)
class BaselineAggregate:
    baseline_name: str
    method_category: str
    arrival_regime: str
    episode_count: int
    unique_trace_count: int
    seed_count: int
    replicates_per_trace: int
    mean_reward: float
    mean_completed_value: float
    mean_arrived_request_value: float
    mean_normalized_completed_value: float | None
    mean_sensing_resource_cost: float
    mean_completion_ratio: float
    mean_acceptance_ratio: float
    mean_requests_per_created_session: float | None
    mean_tenant_sla_residual: float
    mean_communication_qos_residual: float
    mean_positive_constraint_excess: float
    valid_action_rate: float
    all_finite: bool
    pooled_active_demand_shortfall: float | None
    confidence_intervals: tuple[MetricConfidenceInterval, ...]

    @property
    def method_name(self) -> str:
        return self.baseline_name


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationMetadata:
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_physical_slot: int | None
    algorithm_config_path: str
    method: str
    credit_assignment_schema: str
    training_seed: int
    feature_schema_digest: str
    architecture_signature: str
    environment_semantic_digest: str
    validation_protocol_digest: str
    constraint_labels: tuple[str, ...]
    normalization_clip: float
    normalization_epsilon: float


@dataclass(frozen=True, slots=True)
class BaselineEvaluationReport:
    seeds: tuple[int, ...]
    arrival_regimes: tuple[str, ...]
    methods: tuple[str, ...]
    deterministic_online_heuristics: tuple[str, ...]
    stochastic_sanity_baselines: tuple[str, ...]
    learned_policies: tuple[str, ...]
    checkpoint: CheckpointEvaluationMetadata | None
    random_valid_root_seed: int | None
    random_valid_replicates: int | None
    bootstrap_root_seed: int
    bootstrap_samples: int
    unique_primitive_trace_count: int
    deterministic_heuristic_episode_count: int
    random_valid_episode_count: int
    learned_policy_episode_count: int
    episodes: tuple[BaselineEpisodeReport, ...]
    aggregates: tuple[BaselineAggregate, ...]
    macro_aggregates: tuple[BaselineAggregate, ...]

    @staticmethod
    def _episode_dict(item: BaselineEpisodeReport) -> dict[str, Any]:
        value = asdict(item)
        value.update({
            "method_name": item.method_name,
            "completed_value_total": item.cumulative_completed_value,
            "sensing_resource_cost_total": item.cumulative_sensing_resource_cost,
            "reward_total": item.cumulative_reward,
        })
        return value

    @staticmethod
    def _aggregate_dict(item: BaselineAggregate) -> dict[str, Any]:
        value = asdict(item)
        value["method_name"] = item.method_name
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_groups": {
                "deterministic_online_heuristics": list(self.deterministic_online_heuristics),
                "stochastic_sanity_baselines": [{
                    "name": name,
                    "interpretation": "empirical lower/sanity reference; not a theoretical lower bound",
                } for name in self.stochastic_sanity_baselines],
                "learned_policies": list(self.learned_policies),
            },
            "evaluation_contract": {
                "seeds": list(self.seeds), "arrival_regimes": list(self.arrival_regimes),
                "methods": list(self.methods), "random_valid_root_seed": self.random_valid_root_seed,
                "random_valid_replicates": self.random_valid_replicates,
                "checkpoint": None if self.checkpoint is None else asdict(self.checkpoint),
                "bootstrap_root_seed": self.bootstrap_root_seed,
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_unit": "root_seed_after_within_trace_replicate_averaging",
                "macro_weighting": "equal_arrival_regime_weight_per_root_seed",
            },
            "counts": {
                "unique_primitive_traces": self.unique_primitive_trace_count,
                "deterministic_heuristic_episodes": self.deterministic_heuristic_episode_count,
                "random_valid_episodes": self.random_valid_episode_count,
                "learned_policy_episodes": self.learned_policy_episode_count,
                "total_episodes": len(self.episodes), "aggregate_groups": len(self.aggregates),
                "macro_groups": len(self.macro_aggregates),
            },
            "episodes": [self._episode_dict(item) for item in self.episodes],
            "aggregates": [self._aggregate_dict(item) for item in self.aggregates],
            "macro_aggregates": [self._aggregate_dict(item) for item in self.macro_aggregates],
        }


def _id_label(value: EntityId | None) -> str | None:
    if value is None:
        return None
    kind, raw = identifier_key(value)
    return f"{'int' if kind == 0 else 'str'}:{raw}"


def _action_label(action: EnvironmentAction | None) -> str:
    if action is None:
        return "none"
    session = "" if action.session_id is None else f":{_id_label(action.session_id)}"
    profile = "" if action.profile_id is None else f":{action.profile_id}"
    return f"{action.action_type.value}{session}{profile}"


def _focal_request(state: EnvironmentStateSnapshot) -> SensingRequest:
    key = identifier_key(state.focal_request_id)
    return next(request for request in state.requests if identifier_key(request.request_id) == key)


def _service_candidates(
    config: CanonicalConfig, state: EnvironmentStateSnapshot, masks: ActionMaskSnapshot,
) -> tuple[ImmediateServiceCandidate, ...]:
    focal = _focal_request(state)
    requests = {identifier_key(item.request_id): item for item in state.requests}
    sessions = {identifier_key(item.session_id): item for item in state.active_sessions}
    candidates = []
    for entry in masks.entries:
        if not entry.feasible or entry.action.action_type not in {ActionType.MERGE, ActionType.CREATE}:
            continue
        profile = config.resource_profiles[entry.action.profile_id]
        if entry.action.action_type is ActionType.CREATE:
            candidates.append(build_create_candidate(
                focal, profile, entry.create_assessment, state.active_sessions, state.current_slot,
                config.system["total_bandwidth_hz"], config.system["total_power_w"],
                config.reward["sensing_cost_bandwidth_weight"],
                config.reward["sensing_cost_power_weight"],
            ))
            continue
        session = sessions[identifier_key(entry.action.session_id)]
        members = tuple(requests[identifier_key(request_id)] for request_id in session.member_request_ids)
        candidates.append(build_merge_candidate(
            focal, members, session, profile, entry.merge_assessment, state.active_sessions,
            state.current_slot, config.system["total_bandwidth_hz"],
            config.system["total_power_w"], config.reward["sensing_cost_bandwidth_weight"],
            config.reward["sensing_cost_power_weight"],
        ))
    return tuple(candidates)


def select_baseline_action(
    config: CanonicalConfig, baseline_name: str, state: EnvironmentStateSnapshot,
    masks: ActionMaskSnapshot,
) -> EnvironmentAction:
    """Run one deterministic selector and return its feasible environment action."""
    try:
        selector = BASELINE_REGISTRY[baseline_name]
    except KeyError as error:
        raise ValueError(f"unknown baseline: {baseline_name!r}") from error
    action = selector(_service_candidates(config, state, masks), defer_feasible=masks.defer_feasible)
    if action not in masks.feasible_actions:
        raise RuntimeError(f"baseline {baseline_name!r} selected an infeasible action")
    return action


def _finite(values: Iterable[float | int | None]) -> bool:
    return all(value is None or isfinite(float(value)) for value in values)


def run_baseline_episode(
    config: CanonicalConfig, trace: PrimitiveTrace, baseline_name: str,
) -> BaselineEpisodeReport:
    """Run one deterministic baseline for one complete frozen trace."""
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(f"unknown baseline: {baseline_name!r}")
    env = ISACSSCEnv(config)
    env.reset(trace)
    actions: list[str] = []
    focals: list[str | None] = []
    active_session_counts: list[int] = []
    feasible_merge_actions = merge_opportunity_states = 0

    while not env.terminated:
        state = env.state_snapshot()
        masks = env.current_action_masks()
        active_session_counts.append(len(state.active_sessions))
        action = None
        if masks is not None:
            merge_count = sum(
                entry.feasible and entry.action.action_type is ActionType.MERGE for entry in masks.entries
            )
            feasible_merge_actions += merge_count
            merge_opportunity_states += int(merge_count > 0)
            action = select_baseline_action(config, baseline_name, state, masks)
        focals.append(_id_label(state.focal_request_id))
        actions.append(_action_label(action))
        env.step(action)

    terminal = env.state_snapshot()
    state_counts = request_state_counts(terminal.requests)
    states = dict(state_counts)
    action_counts = dict(terminal.action_counts)
    arrived_count = len(terminal.requests)
    arrived_value = sum(item.completion_value for item in terminal.requests)
    accepted = sum(item.accepted_count for item in terminal.tenant_accounting)
    tenant_accounting = tuple((
        _id_label(item.tenant_id), item.accepted_count, item.first_violated_count,
        item.completed_count, item.residual,
    ) for item in terminal.tenant_accounting)
    communication_accounting = tuple((
        _id_label(item.user_id), item.active_demand_slots, item.shortfall_sum, item.residual_sum,
    ) for item in terminal.communication_accounting)
    tenant_residuals = tuple(item[4] for item in tenant_accounting)
    communication_residuals = tuple(item[3] for item in communication_accounting)
    focal_decisions = sum(item is not None for item in focals)
    completed = states[RequestState.COMPLETED.value]
    normalized_completed_value = safe_ratio(terminal.cumulative_completed_value, arrived_value)
    acceptance_ratio = 0.0 if arrived_count == 0 else accepted/arrived_count
    completion_ratio = 0.0 if arrived_count == 0 else completed/arrived_count
    requests_per_created_session = safe_ratio(accepted, action_counts[ActionType.CREATE])
    primary_values = (
        arrived_value, terminal.cumulative_completed_value, normalized_completed_value,
        terminal.cumulative_sensing_resource_cost, terminal.cumulative_reward,
        acceptance_ratio, completion_ratio, requests_per_created_session,
        *tenant_residuals, *communication_residuals,
    )
    return BaselineEpisodeReport(
        baseline_name=baseline_name, method_category=DETERMINISTIC_METHOD_CATEGORY,
        replicate=None, trace_id=trace.trace_id, root_seed=trace.root_seed,
        arrival_regime=trace.arrival_regime, physical_step_count=terminal.current_slot,
        focal_decision_count=focal_decisions, no_request_slot_count=terminal.no_request_count,
        merge_count=action_counts[ActionType.MERGE], create_count=action_counts[ActionType.CREATE],
        defer_count=action_counts[ActionType.DEFER], reject_count=action_counts[ActionType.REJECT],
        arrived_request_count=arrived_count, arrived_request_value_total=arrived_value,
        accepted_request_count=accepted, completed_request_count=completed,
        failed_request_count=states[RequestState.FAILED.value],
        active_request_count=states[RequestState.ACTIVE.value],
        expired_request_count=states[RequestState.EXPIRED.value],
        rejected_request_count=states[RequestState.REJECTED.value],
        terminal_waiting_request_count=states[RequestState.WAITING.value],
        cumulative_completed_value=terminal.cumulative_completed_value,
        normalized_completed_value=normalized_completed_value,
        cumulative_sensing_resource_cost=terminal.cumulative_sensing_resource_cost,
        cumulative_reward=terminal.cumulative_reward, acceptance_ratio=acceptance_ratio,
        completion_ratio=completion_ratio,
        mean_active_session_count=mean(active_session_counts),
        peak_active_session_count=max(active_session_counts, default=0),
        created_session_count=action_counts[ActionType.CREATE],
        requests_per_created_session=requests_per_created_session,
        feasible_merge_action_count=feasible_merge_actions,
        decision_states_with_merge_opportunity=merge_opportunity_states,
        tenant_sla_residual_total=sum(tenant_residuals),
        tenant_sla_positive_excess=sum(max(0.0, value) for value in tenant_residuals),
        communication_qos_residual_total=sum(communication_residuals),
        communication_qos_positive_excess=sum(max(0.0, value) for value in communication_residuals),
        positive_constraint_excess=sum(max(0.0, value) for value in (*tenant_residuals, *communication_residuals)),
        valid_action_count=focal_decisions, invalid_action_count=0, valid_action_rate=1.0,
        all_finite=_finite(primary_values), terminal_request_state_counts=state_counts,
        per_tenant_accounting=tenant_accounting,
        per_user_communication_accounting=communication_accounting,
        action_sequence=tuple(actions), focal_sequence=tuple(focals),
    )


def _validation_episode_report(
    item: ValidationEpisode, method_name: str, method_category: str, replicate: int | None,
) -> BaselineEpisodeReport:
    metrics = item.metrics
    action_counts = dict(metrics.action_counts)
    active = metrics.accepted-metrics.completed-metrics.failed
    waiting = metrics.arrived-metrics.accepted-metrics.rejected-metrics.expired
    state_counts = (
        (RequestState.WAITING.value, waiting), (RequestState.ACTIVE.value, active),
        (RequestState.COMPLETED.value, metrics.completed), (RequestState.FAILED.value, metrics.failed),
        (RequestState.EXPIRED.value, metrics.expired), (RequestState.REJECTED.value, metrics.rejected),
    )
    tenant_accounting = tuple((
        value.tenant_id, value.accepted, value.first_violated, value.completed, value.residual_total,
    ) for value in metrics.tenants)
    communication_accounting = tuple((
        value.user_id, value.active_demand_slots, value.normalized_shortfall_sum, value.residual_total,
    ) for value in metrics.communication_users)
    tenant_residuals = metrics.tenant_residual_totals
    communication_residuals = metrics.communication_residual_totals
    acceptance_ratio = 0.0 if metrics.acceptance_ratio is None else metrics.acceptance_ratio
    completion_ratio = 0.0 if metrics.completion_ratio is None else metrics.completion_ratio
    primary_values = (
        metrics.arrived_request_value_total, metrics.completed_value_total,
        metrics.normalized_completed_value, metrics.sensing_resource_cost_total, metrics.reward_total,
        acceptance_ratio, completion_ratio, metrics.requests_served_per_created_session,
        metrics.valid_action_rate, *tenant_residuals, *communication_residuals,
    )
    return BaselineEpisodeReport(
        baseline_name=method_name, method_category=method_category, replicate=replicate,
        trace_id=metrics.trace_id, root_seed=metrics.root_seed, arrival_regime=metrics.arrival_regime,
        physical_step_count=metrics.physical_slots, focal_decision_count=metrics.focal_decisions,
        no_request_slot_count=metrics.physical_slots-metrics.focal_decisions,
        merge_count=action_counts[ActionType.MERGE.value], create_count=action_counts[ActionType.CREATE.value],
        defer_count=action_counts[ActionType.DEFER.value], reject_count=action_counts[ActionType.REJECT.value],
        arrived_request_count=metrics.arrived, arrived_request_value_total=metrics.arrived_request_value_total,
        accepted_request_count=metrics.accepted, completed_request_count=metrics.completed,
        failed_request_count=metrics.failed, active_request_count=active, expired_request_count=metrics.expired,
        rejected_request_count=metrics.rejected, terminal_waiting_request_count=waiting,
        cumulative_completed_value=metrics.completed_value_total,
        normalized_completed_value=metrics.normalized_completed_value,
        cumulative_sensing_resource_cost=metrics.sensing_resource_cost_total,
        cumulative_reward=metrics.reward_total, acceptance_ratio=acceptance_ratio,
        completion_ratio=completion_ratio, mean_active_session_count=metrics.post_slot_active_session_count_mean,
        peak_active_session_count=metrics.post_slot_active_session_count_max,
        created_session_count=metrics.created_sessions,
        requests_per_created_session=metrics.requests_served_per_created_session,
        feasible_merge_action_count=item.feasible_merge_action_count,
        decision_states_with_merge_opportunity=item.decision_states_with_merge_opportunity,
        tenant_sla_residual_total=sum(tenant_residuals),
        tenant_sla_positive_excess=sum(max(0.0, value) for value in tenant_residuals),
        communication_qos_residual_total=sum(communication_residuals),
        communication_qos_positive_excess=sum(max(0.0, value) for value in communication_residuals),
        positive_constraint_excess=metrics.positive_constraint_excess,
        valid_action_count=metrics.valid_actions, invalid_action_count=metrics.invalid_actions,
        valid_action_rate=metrics.valid_action_rate, all_finite=_finite(primary_values),
        terminal_request_state_counts=state_counts, per_tenant_accounting=tenant_accounting,
        per_user_communication_accounting=communication_accounting,
        action_sequence=item.action_sequence, focal_sequence=item.focal_sequence,
    )


def _optional_mean(values: Iterable[float | None]) -> float | None:
    defined = tuple(value for value in values if value is not None)
    return None if not defined else mean(defined)


def _metric_value(item: BaselineEpisodeReport, metric: str) -> float | None:
    values = {
        "reward_total": item.cumulative_reward,
        "completed_value_total": item.cumulative_completed_value,
        "arrived_request_value_total": item.arrived_request_value_total,
        "normalized_completed_value": item.normalized_completed_value,
        "sensing_resource_cost_total": item.cumulative_sensing_resource_cost,
        "completion_ratio": item.completion_ratio,
        "acceptance_ratio": item.acceptance_ratio,
        "requests_per_created_session": item.requests_per_created_session,
        "positive_constraint_excess": item.positive_constraint_excess,
        "tenant_sla_residual_mean": mean(row[4] for row in item.per_tenant_accounting),
        "communication_qos_residual_mean": mean(row[3] for row in item.per_user_communication_accounting),
        "active_demand_shortfall": safe_ratio(
            sum(row[2] for row in item.per_user_communication_accounting),
            sum(row[1] for row in item.per_user_communication_accounting),
        ),
        "valid_action_rate": item.valid_action_rate,
    }
    return values[metric]


def _seed_values(reports: Sequence[BaselineEpisodeReport], metric: str) -> tuple[float, ...]:
    values = []
    for seed in sorted({item.root_seed for item in reports}):
        replicate_values = tuple(
            value for value in (_metric_value(item, metric) for item in reports if item.root_seed == seed)
            if value is not None
        )
        if replicate_values:
            values.append(mean(replicate_values))
    return tuple(values)


def _macro_seed_values(
    reports: Sequence[BaselineEpisodeReport], arrival_regimes: Sequence[str], metric: str,
) -> tuple[float, ...]:
    values = []
    for seed in sorted({item.root_seed for item in reports}):
        regime_values = []
        for regime in arrival_regimes:
            replicate_values = tuple(
                value for value in (
                    _metric_value(item, metric) for item in reports
                    if item.root_seed == seed and item.arrival_regime == regime
                ) if value is not None
            )
            if replicate_values:
                regime_values.append(mean(replicate_values))
        if len(regime_values) == len(arrival_regimes):
            values.append(mean(regime_values))
    return tuple(values)


def _confidence_interval(
    config: CanonicalConfig, method_name: str, aggregate_label: str, metric: str,
    values: Sequence[float], bootstrap_root_seed: int, bootstrap_samples: int,
) -> MetricConfidenceInterval:
    if not values:
        return MetricConfidenceInterval(metric, 0, None, None, None, None, None)
    sample = np.asarray(values, dtype=np.float64)
    if len(sample) == 1:
        lower = upper = float(sample[0])
    else:
        generator = SeedContract.from_config(config).rng(
            bootstrap_root_seed, _BOOTSTRAP_DOMAIN, method_name, aggregate_label, metric,
        )
        indices = generator.integers(0, len(sample), size=(bootstrap_samples, len(sample)))
        draws = sample[indices].mean(axis=1)
        lower, upper = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
    return MetricConfidenceInterval(
        metric, len(sample), float(sample.mean()), float(median(values)),
        float(pstdev(values)) if len(values) > 1 else 0.0, lower, upper,
    )


def _aggregate_group(
    config: CanonicalConfig, reports: Sequence[BaselineEpisodeReport], aggregate_label: str,
    bootstrap_root_seed: int, bootstrap_samples: int, macro_regimes: Sequence[str] | None = None,
) -> BaselineAggregate:
    method_name = reports[0].method_name
    metric_values = {
        metric: (
            _seed_values(reports, metric) if macro_regimes is None
            else _macro_seed_values(reports, macro_regimes, metric)
        ) for metric in (*_CI_METRICS, "arrived_request_value_total", "tenant_sla_residual_mean",
                          "communication_qos_residual_mean")
    }
    confidence_intervals = tuple(
        _confidence_interval(
            config, method_name, aggregate_label, metric, metric_values[metric],
            bootstrap_root_seed, bootstrap_samples,
        ) for metric in _CI_METRICS
    )
    active_slots = sum(row[1] for item in reports for row in item.per_user_communication_accounting)
    shortfall_sum = sum(row[2] for item in reports for row in item.per_user_communication_accounting)
    if macro_regimes is None:
        active_demand_shortfall = safe_ratio(shortfall_sum, active_slots)
        valid_action_rate = mean(_seed_values(reports, "valid_action_rate"))
    else:
        active_demand_shortfall = _optional_mean(
            _macro_seed_values(reports, macro_regimes, "active_demand_shortfall")
        )
        valid_action_rate = mean(_macro_seed_values(reports, macro_regimes, "valid_action_rate"))
    unique_trace_count = len({(item.root_seed, item.arrival_regime) for item in reports})
    return BaselineAggregate(
        baseline_name=method_name, method_category=reports[0].method_category,
        arrival_regime=aggregate_label, episode_count=len(reports),
        unique_trace_count=unique_trace_count,
        seed_count=len({item.root_seed for item in reports}),
        replicates_per_trace=len(reports)//unique_trace_count,
        mean_reward=mean(metric_values["reward_total"]),
        mean_completed_value=mean(metric_values["completed_value_total"]),
        mean_arrived_request_value=mean(metric_values["arrived_request_value_total"]),
        mean_normalized_completed_value=_optional_mean(metric_values["normalized_completed_value"]),
        mean_sensing_resource_cost=mean(metric_values["sensing_resource_cost_total"]),
        mean_completion_ratio=mean(metric_values["completion_ratio"]),
        mean_acceptance_ratio=mean(metric_values["acceptance_ratio"]),
        mean_requests_per_created_session=_optional_mean(metric_values["requests_per_created_session"]),
        mean_tenant_sla_residual=mean(metric_values["tenant_sla_residual_mean"]),
        mean_communication_qos_residual=mean(metric_values["communication_qos_residual_mean"]),
        mean_positive_constraint_excess=mean(metric_values["positive_constraint_excess"]),
        valid_action_rate=valid_action_rate, all_finite=all(item.all_finite for item in reports),
        pooled_active_demand_shortfall=active_demand_shortfall, confidence_intervals=confidence_intervals,
    )


def aggregate_episode_reports(
    config: CanonicalConfig, reports: Iterable[BaselineEpisodeReport], *,
    bootstrap_root_seed: int, bootstrap_samples: int,
) -> tuple[BaselineAggregate, ...]:
    """Aggregate each method separately by arrival regime."""
    values = tuple(reports)
    aggregates = []
    for method_name in REPORT_METHOD_ORDER:
        for regime in config.trace_generation["registered_arrival_regimes"]:
            group = tuple(
                item for item in values if item.method_name == method_name and item.arrival_regime == regime
            )
            if group:
                aggregates.append(_aggregate_group(
                    config, group, regime, bootstrap_root_seed, bootstrap_samples,
                ))
    return tuple(aggregates)


def aggregate_macro_reports(
    config: CanonicalConfig, reports: Iterable[BaselineEpisodeReport],
    arrival_regimes: Sequence[str], *, bootstrap_root_seed: int, bootstrap_samples: int,
) -> tuple[BaselineAggregate, ...]:
    """Aggregate each method with equal arrival-regime weight per root seed."""
    values = tuple(reports)
    aggregates = []
    for method_name in REPORT_METHOD_ORDER:
        group = tuple(item for item in values if item.method_name == method_name)
        if group:
            aggregates.append(_aggregate_group(
                config, group, "macro", bootstrap_root_seed, bootstrap_samples,
                macro_regimes=arrival_regimes,
            ))
    return tuple(aggregates)


def _validated_inputs(
    config: CanonicalConfig, seeds: Sequence[int], arrival_regimes: Sequence[str],
    methods: Sequence[str], random_valid_root_seed: int | None,
    random_valid_replicates: int | None, bootstrap_root_seed: int,
    bootstrap_samples: int, checkpoint_requested: bool,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    for values, name in ((seeds, "seeds"), (arrival_regimes, "arrival regimes")):
        if not values:
            raise ValueError(f"{name} must not be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    if not methods and not checkpoint_requested:
        raise ValueError("at least one baseline or checkpoint must be selected")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    registered_regimes = tuple(config.trace_generation["registered_arrival_regimes"])
    unknown_regimes = set(arrival_regimes)-set(registered_regimes)
    unknown_methods = set(methods)-set(EVALUATION_METHODS)
    if unknown_regimes:
        raise ValueError(f"unknown arrival regimes: {sorted(unknown_regimes)}")
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
    random_requested = RANDOM_VALID_NAME in methods
    if random_requested and (random_valid_root_seed is None or random_valid_replicates is None):
        raise ValueError("random_valid requires --random-valid-root-seed and --random-valid-replicates")
    if not random_requested and (random_valid_root_seed is not None or random_valid_replicates is not None):
        raise ValueError("random-valid options require random_valid in --baselines")
    if random_valid_replicates is not None and random_valid_replicates < 1:
        raise ValueError("random-valid replicates must be positive")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    contract = SeedContract.from_config(config)
    for seed in seeds:
        contract.canonical_material(seed, "standalone_evaluation_trace")
    contract.canonical_material(bootstrap_root_seed, _BOOTSTRAP_DOMAIN)
    if random_valid_root_seed is not None:
        contract.canonical_material(random_valid_root_seed, RANDOM_VALID_NAME)
    canonical_regimes = tuple(regime for regime in registered_regimes if regime in set(arrival_regimes))
    canonical_methods = tuple(method for method in EVALUATION_METHODS if method in set(methods))
    return tuple(sorted(seeds)), canonical_regimes, canonical_methods


def _checkpoint_physical_slot(state: Mapping[str, Any]) -> int | None:
    progress = state.get("progress")
    if not isinstance(progress, Mapping):
        return None
    value = progress.get("completed_physical_slots")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_focal_observation(config: CanonicalConfig, traces: Sequence[PrimitiveTrace]):
    for trace in traces:
        env = ISACSSCEnv(config)
        observation = env.reset(trace)
        while observation is None and not env.terminated:
            observation = env.step(None).next_observation
        if observation is not None:
            return observation
    raise ValueError("selected evaluation traces contain no focal observation")


def _evaluate_checkpoint(
    config: CanonicalConfig, algorithm: ConstrainedPPOConfig, checkpoint_path: str | Path,
    traces: Sequence[PrimitiveTrace],
) -> tuple[tuple[BaselineEpisodeReport, ...], CheckpointEvaluationMetadata]:
    observation = _first_focal_observation(config, traces)
    layout = FeatureLayout.from_view(observation.set_view)
    checkpoint_metadata = read_checkpoint_metadata(checkpoint_path)
    builders = {
        JOINT_CREDIT_METHOD: build_joint_credit_agent,
        COMMON_TRACE_METHOD: build_common_trace_agent,
    }
    try:
        builder = builders[checkpoint_metadata.method]
    except KeyError as error:
        raise ValueError(
            f"unsupported learned-policy checkpoint method: {checkpoint_metadata.method!r}"
        ) from error
    agent = builder(
        layout, algorithm, config, model_seed=0, action_seed=1, minibatch_seed=2,
    )
    metadata, state = load_policy_checkpoint(
        checkpoint_path, agent, expected_method=checkpoint_metadata.method,
    )
    physical_slot = _checkpoint_physical_slot(state)
    validation = evaluate_policy(
        config, agent, traces,
        physical_slot=0 if physical_slot is None else physical_slot,
        capture_decisions=True,
    )
    episodes = tuple(
        _validation_episode_report(item, metadata.method, LEARNED_POLICY_CATEGORY, None)
        for item in validation.episodes
    )
    checkpoint = CheckpointEvaluationMetadata(
        checkpoint_path=Path(checkpoint_path).resolve().as_posix(),
        checkpoint_sha256=checkpoint_sha256(checkpoint_path), checkpoint_physical_slot=physical_slot,
        algorithm_config_path=algorithm.source_path.resolve().as_posix(), method=metadata.method,
        credit_assignment_schema=metadata.credit_assignment_schema,
        training_seed=metadata.training_seed, feature_schema_digest=metadata.feature_schema_digest,
        architecture_signature=metadata.architecture_signature,
        environment_semantic_digest=metadata.environment_semantic_digest,
        validation_protocol_digest=metadata.validation_protocol_digest,
        constraint_labels=metadata.constraint_labels, normalization_clip=agent.normalizer.clip,
        normalization_epsilon=agent.normalizer.epsilon,
    )
    return episodes, checkpoint


def run_baseline_evaluation(
    config: CanonicalConfig, seeds: Sequence[int], arrival_regimes: Sequence[str],
    baselines: Sequence[str] = tuple(BASELINE_REGISTRY), *,
    random_valid_root_seed: int | None = None, random_valid_replicates: int | None = None,
    checkpoint_path: str | Path | None = None, algorithm: ConstrainedPPOConfig | None = None,
    bootstrap_root_seed: int = 54001, bootstrap_samples: int = 2000,
) -> BaselineEvaluationReport:
    """Evaluate requested methods on the same generated primitive traces."""
    if (checkpoint_path is None) != (algorithm is None):
        raise ValueError("checkpoint evaluation requires both checkpoint_path and algorithm")
    seeds, arrival_regimes, methods = _validated_inputs(
        config, seeds, arrival_regimes, baselines, random_valid_root_seed, random_valid_replicates,
        bootstrap_root_seed, bootstrap_samples, checkpoint_requested=checkpoint_path is not None,
    )
    traces = tuple(generate_primitive_trace(config, seed, regime) for seed in seeds for regime in arrival_regimes)
    deterministic_methods = tuple(method for method in methods if method in BASELINE_REGISTRY)
    episodes = [
        run_baseline_episode(config, trace, method)
        for trace in traces for method in deterministic_methods
    ]
    if RANDOM_VALID_NAME in methods:
        random_report = evaluate_random_valid(
            config, traces, root_seed=random_valid_root_seed, replicates_per_trace=random_valid_replicates,
        )
        episodes.extend(
            _validation_episode_report(item, RANDOM_VALID_NAME, STOCHASTIC_SANITY_METHOD_CATEGORY, item.replicate)
            for item in random_report.episodes
        )
    checkpoint = None
    if checkpoint_path is not None:
        learned_episodes, checkpoint = _evaluate_checkpoint(config, algorithm, checkpoint_path, traces)
        episodes.extend(learned_episodes)
    method_order = {name: index for index, name in enumerate(REPORT_METHOD_ORDER)}
    regime_order = {name: index for index, name in enumerate(arrival_regimes)}
    episodes.sort(key=lambda item: (
        item.root_seed, regime_order[item.arrival_regime], method_order[item.method_name],
        -1 if item.replicate is None else item.replicate,
    ))
    reports = tuple(episodes)
    random_count = len(traces)*random_valid_replicates if RANDOM_VALID_NAME in methods else 0
    learned_count = len(traces) if checkpoint is not None else 0
    learned_methods = () if checkpoint is None else (checkpoint.method,)
    return BaselineEvaluationReport(
        seeds=seeds, arrival_regimes=arrival_regimes, methods=(*methods, *learned_methods),
        deterministic_online_heuristics=deterministic_methods,
        stochastic_sanity_baselines=(RANDOM_VALID_NAME,) if RANDOM_VALID_NAME in methods else (),
        learned_policies=learned_methods, checkpoint=checkpoint,
        random_valid_root_seed=random_valid_root_seed if RANDOM_VALID_NAME in methods else None,
        random_valid_replicates=random_valid_replicates if RANDOM_VALID_NAME in methods else None,
        bootstrap_root_seed=bootstrap_root_seed, bootstrap_samples=bootstrap_samples,
        unique_primitive_trace_count=len(traces),
        deterministic_heuristic_episode_count=len(traces)*len(deterministic_methods),
        random_valid_episode_count=random_count, learned_policy_episode_count=learned_count, episodes=reports,
        aggregates=aggregate_episode_reports(
            config, reports, bootstrap_root_seed=bootstrap_root_seed, bootstrap_samples=bootstrap_samples,
        ),
        macro_aggregates=aggregate_macro_reports(
            config, reports, arrival_regimes, bootstrap_root_seed=bootstrap_root_seed,
            bootstrap_samples=bootstrap_samples,
        ),
    )