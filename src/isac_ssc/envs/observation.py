"""Deterministic structured current-state observations for edge-free set policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, log10, log1p
from numbers import Real
from typing import Iterable

from isac_ssc.core.entities import (
    CommunicationUser, EntityId, RequestState, SensingRequest, SensingSession, Task,
)
from isac_ssc.core.quality import (
    CommunicationParameters, SensingParameters, evaluate_communication_quality,
    evaluate_shared_sensing_quality, point_in_disk,
)
from isac_ssc.core.resources import (
    committed_resource_usage, equal_share_communication_resources,
    residual_communication_resources, scheduled_update_slots, sensing_resource_usage,
    session_updates_at,
)
from isac_ssc.core.sla import communication_qos_slot
from isac_ssc.envs.action_masks import ActionMaskSnapshot, CurrentFeasibilitySnapshot
from isac_ssc.envs.action_space import identifier_key
from isac_ssc.envs.dynamics import CommunicationSlotPrimitive, TargetSlotPrimitive
from isac_ssc.utils.config import CanonicalConfig


class ObservationValidationError(ValueError):
    """Raised when a numerical observation feature is not finite."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ObservationValidationError(f"{name} must be a finite number")
    return float(value)


def _typed_index(values, attribute: str):
    return {identifier_key(getattr(item, attribute)): item for item in values}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    unit: str
    normalization: str
    canonical_source: str


@dataclass(frozen=True, slots=True)
class RequestRelationalKey:
    request_id: EntityId
    tenant_id: EntityId
    target_id: EntityId


@dataclass(frozen=True, slots=True)
class SessionRelationalKey:
    session_id: EntityId
    target_id: EntityId


@dataclass(frozen=True, slots=True)
class TenantAccountingState:
    tenant_id: EntityId
    accepted_count: int
    first_violated_count: int
    completed_count: int
    residual: float


@dataclass(frozen=True, slots=True)
class CommunicationAccountingState:
    user_id: EntityId
    active_demand_slots: int
    shortfall_sum: float
    residual_sum: float


@dataclass(frozen=True, slots=True)
class FeatureTable:
    specs: tuple[FeatureSpec, ...]
    keys: tuple[RequestRelationalKey | SessionRelationalKey, ...]
    rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ObservationView:
    request_table: FeatureTable
    session_table: FeatureTable
    global_specs: tuple[FeatureSpec, ...]
    global_features: tuple[float, ...]
    action_masks: ActionMaskSnapshot


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    set_view: ObservationView


def _spec(name: str, unit: str, normalization: str, source: str) -> FeatureSpec:
    return FeatureSpec(name, unit, normalization, source)


def _request_specs(profile_ids: tuple[str, ...]) -> tuple[FeatureSpec, ...]:
    values = [
        _spec("task_detection", "indicator", "none", "SensingRequest.task"),
        _spec("task_localization", "indicator", "none", "SensingRequest.task"),
        _spec("task_tracking", "indicator", "none", "SensingRequest.task"),
        _spec("quality_threshold_normalized", "ratio", "divide by configured task maximum", "SensingRequest.quality_threshold"),
        _spec("completion_value_normalized", "ratio", "divide by configured task maximum", "SensingRequest.completion_value"),
        _spec("merge_permission", "indicator", "none", "SensingRequest.merge_permission"),
        _spec("tenant_sla_budget", "ratio", "none", "Tenant.sla_violation_budget"),
        _spec("deadline_slack_normalized", "slot/horizon", "divide by horizon", "latest_start_slot-current_slot"),
        _spec("eligibility_slack_normalized", "slot/horizon", "divide by horizon", "eligible_slot-current_slot"),
        _spec("waiting_age_normalized", "slot/horizon", "divide by horizon", "current_slot-arrival_slot"),
        _spec("valid_output_interval_normalized", "ratio", "divide by configured maximum interval", "valid_output_interval_slots"),
        _spec("focal_indicator", "indicator", "none", "environment focal request"),
        _spec("aoi_radius_normalized", "ratio", "divide by configured maximum radius", "SensingRequest.aoi.radius_m"),
        _spec("target_relative_x_aoi", "ratio", "divide by request AOI radius", "target_x-request_aoi_center_x"),
        _spec("target_relative_y_aoi", "ratio", "divide by request AOI radius", "target_y-request_aoi_center_y"),
        _spec("target_inside_request_aoi", "indicator", "none", "core.quality.point_in_disk"),
        _spec("target_x_normalized", "ratio", "center and divide by half region width", "TargetSlotPrimitive.position_m"),
        _spec("target_y_normalized", "ratio", "center and divide by half region height", "TargetSlotPrimitive.position_m"),
        _spec("target_vx_normalized", "ratio", "divide by configured target maximum speed", "TargetSlotPrimitive.velocity_m_per_s"),
        _spec("target_vy_normalized", "ratio", "divide by configured target maximum speed", "TargetSlotPrimitive.velocity_m_per_s"),
        _spec("sensing_shadowing_standardized", "standard deviation", "divide by configured sensing shadowing std", "TargetSlotPrimitive.shadowing_db"),
        _spec("sensing_fading_real", "dimensionless", "none", "TargetSlotPrimitive.fading_real"),
        _spec("sensing_fading_imag", "dimensionless", "none", "TargetSlotPrimitive.fading_imag"),
        _spec("sensing_rcs_standardized", "standard deviation", "center dBsm and divide by configured std", "TargetSlotPrimitive.rcs_dbsm"),
        _spec("tenant_accepted_count", "count", "none", "TenantAccountingState.accepted_count"),
        _spec("tenant_first_violated_count", "count", "none", "TenantAccountingState.first_violated_count"),
        _spec("tenant_completed_count", "count", "none", "TenantAccountingState.completed_count"),
        _spec("tenant_additive_residual", "count", "none", "TenantAccountingState.residual"),
    ]
    for profile_id in profile_ids:
        prefix = f"create_{profile_id}"
        values.extend((
            _spec(f"{prefix}_bandwidth_fraction", "ratio", "divide by total bandwidth", "ResourceProfile.sensing_bandwidth_hz"),
            _spec(f"{prefix}_power_fraction", "ratio", "divide by total power", "ResourceProfile.sensing_power_w"),
            _spec(f"{prefix}_period_normalized", "slot/horizon", "divide by horizon", "ResourceProfile.update_period_slots"),
            _spec(f"{prefix}_feasible", "indicator", "none", "CreateProfileAssessment.feasible"),
            _spec(f"{prefix}_quality_margin_defined", "indicator", "none", "CreateProfileAssessment.quality_margin"),
            _spec(f"{prefix}_quality_margin_normalized", "ratio", "divide by request threshold", "CreateProfileAssessment.quality_margin"),
            _spec(f"{prefix}_period_margin_normalized", "ratio", "divide by request valid-output interval", "request interval-(profile period-1)"),
        ))
    return tuple(values)


def _session_specs(profile_ids: tuple[str, ...]) -> tuple[FeatureSpec, ...]:
    values = [
        _spec("output_detection", "indicator", "none", "SensingSession.exposed_outputs"),
        _spec("output_localization", "indicator", "none", "SensingSession.exposed_outputs"),
        _spec("output_tracking", "indicator", "none", "SensingSession.exposed_outputs"),
    ]
    values.extend(_spec(f"profile_{profile_id}", "indicator", "none", "SensingSession.profile")
                  for profile_id in profile_ids)
    values.extend((
        _spec("aoi_radius_normalized", "ratio", "divide by configured maximum radius", "SensingSession.aoi.radius_m"),
        _spec("target_relative_x_aoi", "ratio", "divide by session AOI radius", "target_x-session_aoi_center_x"),
        _spec("target_relative_y_aoi", "ratio", "divide by session AOI radius", "target_y-session_aoi_center_y"),
        _spec("target_inside_session_aoi", "indicator", "none", "core.quality.point_in_disk"),
        _spec("target_x_normalized", "ratio", "center and divide by half region width", "TargetSlotPrimitive.position_m"),
        _spec("target_y_normalized", "ratio", "center and divide by half region height", "TargetSlotPrimitive.position_m"),
        _spec("target_vx_normalized", "ratio", "divide by configured target maximum speed", "TargetSlotPrimitive.velocity_m_per_s"),
        _spec("target_vy_normalized", "ratio", "divide by configured target maximum speed", "TargetSlotPrimitive.velocity_m_per_s"),
        _spec("sensing_shadowing_standardized", "standard deviation", "divide by configured sensing shadowing std", "TargetSlotPrimitive.shadowing_db"),
        _spec("sensing_fading_real", "dimensionless", "none", "TargetSlotPrimitive.fading_real"),
        _spec("sensing_fading_imag", "dimensionless", "none", "TargetSlotPrimitive.fading_imag"),
        _spec("sensing_rcs_standardized", "standard deviation", "center dBsm and divide by configured std", "TargetSlotPrimitive.rcs_dbsm"),
        _spec("next_update_slack_normalized", "slot/horizon", "divide by horizon", "next_update_slot-current_slot"),
        _spec("remaining_lifetime_normalized", "slot/horizon", "divide by horizon", "final_active_slot-current_slot+1"),
        _spec("updates_current_slot", "indicator", "none", "core.resources.session_updates_at"),
        _spec("future_scheduled_update_count_normalized", "count/horizon", "divide by horizon", "canonical session update calendar"),
        _spec("member_count", "count", "none", "SensingSession.member_request_ids"),
        _spec("detection_member_fraction", "ratio", "none", "member task counts"),
        _spec("localization_member_fraction", "ratio", "none", "member task counts"),
        _spec("tracking_member_fraction", "ratio", "none", "member task counts"),
        _spec("distinct_tenant_count", "count", "none", "member tenant keys"),
        _spec("all_members_shareable", "indicator", "none", "member merge permissions"),
        _spec("current_profile_bandwidth_fraction", "ratio", "divide by total bandwidth", "SensingSession.profile"),
        _spec("current_profile_power_fraction", "ratio", "divide by total power", "SensingSession.profile"),
        _spec("current_profile_period_normalized", "slot/horizon", "divide by horizon", "SensingSession.profile"),
        _spec("shared_log1p_sensing_sinr", "log ratio", "log1p", "core.quality.evaluate_shared_sensing_quality"),
        _spec("shared_detection_probability", "probability", "none", "SharedSensingQuality.detection_probability"),
        _spec("shared_detection_gate", "indicator", "none", "SharedSensingQuality.detection_gate_passed"),
        _spec("shared_localization_defined", "indicator", "none", "LocalizationQuality.information_valid"),
        _spec("shared_peb_normalized", "ratio", "divide by simulation-region diagonal", "LocalizationQuality.peb_m"),
        _spec("shared_tracking_capable", "indicator", "none", "SharedSensingQuality.tracking"),
        _spec("shared_pcrb_normalized", "ratio", "divide by simulation-region diagonal", "TrackingQuality.pcrb_m"),
    ))
    return tuple(values)


def _global_specs() -> tuple[FeatureSpec, ...]:
    names = (
        ("slot_normalized", "slot/horizon", "divide by horizon", "current_slot"),
        ("waiting_request_count", "count", "none", "RequestState.WAITING"),
        ("eligible_waiting_count", "count", "none", "current eligibility"),
        ("active_request_count", "count", "none", "RequestState.ACTIVE"),
        ("active_session_count", "count", "none", "current sessions"),
        ("current_sensing_bandwidth_fraction", "ratio", "divide by total bandwidth", "core.resources.sensing_resource_usage"),
        ("current_sensing_power_fraction", "ratio", "divide by total power", "core.resources.sensing_resource_usage"),
        ("future_mean_bandwidth_fraction", "ratio", "mean over remaining slots / total bandwidth", "core.resources.committed_resource_usage"),
        ("future_mean_power_fraction", "ratio", "mean over remaining slots / total power", "core.resources.committed_resource_usage"),
        ("future_peak_bandwidth_fraction", "ratio", "peak / total bandwidth", "core.resources.committed_resource_usage"),
        ("future_peak_power_fraction", "ratio", "peak / total power", "core.resources.committed_resource_usage"),
        ("active_demand_user_count", "count", "none", "CommunicationSlotPrimitive.demand_bit_per_s"),
        ("aggregate_offered_load_normalized", "ratio", "divide by users times minimum rate", "current communication demand"),
        ("pre_action_mean_shortfall", "ratio", "mean active-demand normalized shortfall", "core.sla.communication_qos_slot"),
        ("pre_action_max_shortfall", "ratio", "maximum active-demand normalized shortfall", "core.sla.communication_qos_slot"),
        ("cumulative_completed_value", "value", "none", "environment accounting"),
        ("cumulative_sensing_cost", "cost", "none", "environment accounting"),
        ("tenant_accepted_total", "count", "none", "tenant accounting"),
        ("tenant_first_violated_total", "count", "none", "tenant accounting"),
        ("tenant_completed_total", "count", "none", "tenant accounting"),
        ("tenant_residual_sum", "count", "none", "tenant accounting"),
        ("defined_tenant_violation_fraction", "ratio", "defined tenants / all tenants", "tenant accounting"),
        ("mean_defined_tenant_violation_rate", "ratio", "mean over accepted tenants", "tenant accounting"),
        ("max_defined_tenant_violation_rate", "ratio", "maximum over accepted tenants", "tenant accounting"),
        ("communication_active_demand_slots_total", "count", "none", "communication accounting"),
        ("communication_shortfall_sum", "ratio", "none", "communication accounting"),
        ("communication_residual_sum", "ratio", "none", "communication accounting"),
    )
    return tuple(_spec(*item) for item in names)


def _position_features(state: TargetSlotPrimitive, config: CanonicalConfig) -> tuple[float, ...]:
    region = config.geometry["simulation_region"]
    x_mid = (region["x_min_m"] + region["x_max_m"]) / 2.0
    y_mid = (region["y_min_m"] + region["y_max_m"]) / 2.0
    x_half = (region["x_max_m"] - region["x_min_m"]) / 2.0
    y_half = (region["y_max_m"] - region["y_min_m"]) / 2.0
    speed_scale = config.mobility["targets"]["initial_speed_max_m_per_s"] or 1.0
    shadow_scale = config.sensing["shadowing_std_db"] or 1.0
    rcs = config.sensing["rcs"]
    rcs_mean = 10.0 * log10(rcs["median_m2"])
    rcs_scale = rcs["dbsm_std_db"] or 1.0
    return (
        (state.position_m[0] - x_mid) / x_half, (state.position_m[1] - y_mid) / y_half,
        state.velocity_m_per_s[0] / speed_scale, state.velocity_m_per_s[1] / speed_scale,
        state.shadowing_db / shadow_scale, state.fading_real, state.fading_imag,
        (state.rcs_dbsm - rcs_mean) / rcs_scale,
    )


def _task_threshold_scale(task: Task, config: CanonicalConfig) -> float:
    name = {
        Task.DETECTION: "detection_probability",
        Task.LOCALIZATION: "localization_peb_m",
        Task.TRACKING: "tracking_pcrb_m",
    }[task]
    return float(config.requests["quality_thresholds"][name]["maximum"])


def _task_value_scale(task: Task, config: CanonicalConfig) -> float:
    return float(config.requests["completion_values"][task.value]["maximum"])


def _request_table(
    current_slot: int, focal: SensingRequest, requests: tuple[SensingRequest, ...],
    target_index: dict[tuple[int, str], TargetSlotPrimitive], feasibility: CurrentFeasibilitySnapshot,
    tenant_index: dict[tuple[int, str], TenantAccountingState], config: CanonicalConfig,
) -> FeatureTable:
    waiting = tuple(sorted((item for item in requests if item.state is RequestState.WAITING),
                           key=lambda item: identifier_key(item.request_id)))
    profile_ids = feasibility.profile_ids
    specs = _request_specs(profile_ids)
    horizon = config.system["horizon_slots"]
    max_interval = max(value for task in Task for value in config.requests["update_interval_slots"][task.value]["values"])
    max_radius = config.geometry["aoi"]["radius_max_m"]
    rows, keys = [], []
    for request in waiting:
        target = target_index[identifier_key(request.target_id)]
        tenant = config.tenant(request.tenant_id)
        accounting = tenant_index[identifier_key(request.tenant_id)]
        relative = (
            (target.position_m[0] - request.aoi.center_m[0]) / request.aoi.radius_m,
            (target.position_m[1] - request.aoi.center_m[1]) / request.aoi.radius_m,
        )
        row = [
            float(request.task is Task.DETECTION), float(request.task is Task.LOCALIZATION),
            float(request.task is Task.TRACKING), request.quality_threshold / _task_threshold_scale(request.task, config),
            request.completion_value / _task_value_scale(request.task, config), float(request.merge_permission),
            tenant.sla_violation_budget, (request.latest_start_slot-current_slot) / horizon,
            (request.eligible_slot-current_slot) / horizon, (current_slot-request.arrival_slot) / horizon,
            request.valid_output_interval_slots / max_interval,
            float(identifier_key(request.request_id) == identifier_key(focal.request_id)),
            request.aoi.radius_m / max_radius, relative[0], relative[1],
            float(point_in_disk(target.position_m, request.aoi)), *_position_features(target, config),
            float(accounting.accepted_count), float(accounting.first_violated_count),
            float(accounting.completed_count), accounting.residual,
        ]
        request_assessment = feasibility.request_for(request.request_id)
        for profile_id in profile_ids:
            profile = config.resource_profiles[profile_id]
            assessment = request_assessment.create_for(profile_id)
            margin = assessment.quality_margin.margin
            row.extend((
                profile.sensing_bandwidth_hz / config.system["total_bandwidth_hz"],
                profile.sensing_power_w / config.system["total_power_w"],
                profile.update_period_slots / horizon, float(assessment.feasible),
                float(margin is not None), 0.0 if margin is None else margin / request.quality_threshold,
                (request.valid_output_interval_slots-(profile.update_period_slots-1))
                / max(1, request.valid_output_interval_slots),
            ))
        keys.append(RequestRelationalKey(request.request_id, request.tenant_id, request.target_id))
        rows.append(tuple(_finite(value, "request feature") for value in row))
    return FeatureTable(specs, tuple(keys), tuple(rows))


def _session_table(
    current_slot: int, requests: tuple[SensingRequest, ...], sessions: tuple[SensingSession, ...],
    target_index: dict[tuple[int, str], TargetSlotPrimitive], config: CanonicalConfig,
) -> FeatureTable:
    profiles = tuple(sorted(config.resource_profiles))
    specs = _session_specs(profiles)
    request_index = _typed_index(requests, "request_id")
    parameters = SensingParameters.from_config(config)
    horizon = config.system["horizon_slots"]
    max_radius = config.geometry["aoi"]["radius_max_m"]
    region = config.geometry["simulation_region"]
    diagonal = hypot(region["x_max_m"]-region["x_min_m"], region["y_max_m"]-region["y_min_m"])
    rows, keys = [], []
    for session in sessions:
        target = target_index[identifier_key(session.target_id)]
        members = tuple(request_index[identifier_key(item)] for item in session.member_request_ids)
        member_count = len(members)
        relative = (
            (target.position_m[0]-session.aoi.center_m[0]) / session.aoi.radius_m,
            (target.position_m[1]-session.aoi.center_m[1]) / session.aoi.radius_m,
        )
        shared = evaluate_shared_sensing_quality(
            session, target.position_m, tuple(config.geometry["bs_position_m"]), target.rcs_m2,
            target.shadowing_db, target.fading_power_gain, parameters, session.tracking_covariance,
        )
        scheduled = tuple(slot for slot in scheduled_update_slots(session) if slot >= current_slot)
        next_update = scheduled[0] if scheduled else session.final_active_slot+1
        future_updates = sum(slot > current_slot for slot in scheduled)
        row = [
            float(Task.DETECTION in session.exposed_outputs),
            float(Task.LOCALIZATION in session.exposed_outputs),
            float(Task.TRACKING in session.exposed_outputs),
        ]
        row.extend(float(session.profile.profile_id == profile_id) for profile_id in profiles)
        row.extend((
            session.aoi.radius_m / max_radius, relative[0], relative[1],
            float(point_in_disk(target.position_m, session.aoi)), *_position_features(target, config),
            (next_update-current_slot) / horizon,
            (session.final_active_slot-current_slot+1) / horizon,
            float(session_updates_at(session, current_slot)), future_updates / horizon,
            float(member_count), sum(item.task is Task.DETECTION for item in members) / member_count,
            sum(item.task is Task.LOCALIZATION for item in members) / member_count,
            sum(item.task is Task.TRACKING for item in members) / member_count,
            float(len({identifier_key(item.tenant_id) for item in members})),
            float(all(item.merge_permission for item in members)),
            session.profile.sensing_bandwidth_hz / config.system["total_bandwidth_hz"],
            session.profile.sensing_power_w / config.system["total_power_w"],
            session.profile.update_period_slots / horizon, log1p(shared.sensing_sinr),
            shared.detection_probability, float(shared.detection_gate_passed),
            float(shared.localization.information_valid),
            0.0 if not shared.localization.information_valid else shared.localization.peb_m / diagonal,
            float(shared.tracking is not None),
            0.0 if shared.tracking is None else shared.tracking.pcrb_m / diagonal,
        ))
        keys.append(SessionRelationalKey(session.session_id, session.target_id))
        rows.append(tuple(_finite(value, "session feature") for value in row))
    return FeatureTable(specs, tuple(keys), tuple(rows))


def _communication_current(
    current_slot: int, primitives: tuple[CommunicationSlotPrimitive, ...],
    sessions: tuple[SensingSession, ...], config: CanonicalConfig,
) -> tuple[int, float, float, float]:
    users = tuple(CommunicationUser(
        item.user_id, item.position_m, item.velocity_m_per_s, item.demand_bit_per_s,
        config.communication["minimum_rate_bit_per_s"], config.communication["normalized_shortfall_budget"],
    ) for item in primitives)
    usage = sensing_resource_usage(sessions, current_slot)
    residual = residual_communication_resources(
        config.system["total_bandwidth_hz"], config.system["total_power_w"], usage,
    )
    allocations = {identifier_key(item.user_id): item for item in equal_share_communication_resources(users, residual)}
    parameters = CommunicationParameters.from_config(config)
    shortfalls = []
    for primitive, user in zip(primitives, users, strict=True):
        allocation = allocations[identifier_key(user.user_id)]
        quality = evaluate_communication_quality(
            primitive.position_m, tuple(config.geometry["bs_position_m"]), allocation.bandwidth_hz,
            allocation.power_w, user.demand_bit_per_s, primitive.shadowing_db,
            primitive.fading_power_gain, parameters,
        )
        qos = communication_qos_slot(
            user.demand_bit_per_s, user.minimum_rate_bit_per_s, quality.served_rate_bit_per_s,
            user.normalized_shortfall_budget,
        )
        if qos.active_demand:
            shortfalls.append(qos.normalized_shortfall)
    active = len(shortfalls)
    offered = sum(item.demand_bit_per_s for item in users) / max(
        1.0, len(users) * config.communication["minimum_rate_bit_per_s"],
    )
    return active, offered, 0.0 if not shortfalls else sum(shortfalls)/active, max(shortfalls, default=0.0)


def _global_features(
    current_slot: int, requests: tuple[SensingRequest, ...], sessions: tuple[SensingSession, ...],
    communication: tuple[CommunicationSlotPrimitive, ...], tenant_accounting: tuple[TenantAccountingState, ...],
    communication_accounting: tuple[CommunicationAccountingState, ...], cumulative_completed_value: float,
    cumulative_sensing_cost: float, config: CanonicalConfig,
) -> tuple[float, ...]:
    horizon = config.system["horizon_slots"]
    total_bandwidth, total_power = config.system["total_bandwidth_hz"], config.system["total_power_w"]
    current_usage = sensing_resource_usage(sessions, current_slot)
    future_start = current_slot+1
    reservations = () if future_start >= horizon else committed_resource_usage(
        sessions, total_bandwidth, total_power, start_slot=future_start, end_slot=horizon-1,
    )
    remaining = max(1, horizon-future_start)
    bandwidth_sum = sum(item.sensing_bandwidth_hz for item in reservations)
    power_sum = sum(item.sensing_power_w for item in reservations)
    active_demand, offered, mean_shortfall, max_shortfall = _communication_current(
        current_slot, communication, sessions, config,
    )
    rates = tuple(item.first_violated_count/item.accepted_count for item in tenant_accounting
                  if item.accepted_count > 0)
    values = (
        current_slot / horizon,
        float(sum(item.state is RequestState.WAITING for item in requests)),
        float(sum(item.state is RequestState.WAITING and item.eligible_slot <= current_slot <= item.latest_start_slot
                  for item in requests)),
        float(sum(item.state is RequestState.ACTIVE for item in requests)), float(len(sessions)),
        current_usage.sensing_bandwidth_hz / total_bandwidth,
        current_usage.sensing_power_w / total_power,
        bandwidth_sum / remaining / total_bandwidth, power_sum / remaining / total_power,
        max((item.sensing_bandwidth_hz for item in reservations), default=0.0) / total_bandwidth,
        max((item.sensing_power_w for item in reservations), default=0.0) / total_power,
        float(active_demand), offered, mean_shortfall, max_shortfall,
        cumulative_completed_value, cumulative_sensing_cost,
        float(sum(item.accepted_count for item in tenant_accounting)),
        float(sum(item.first_violated_count for item in tenant_accounting)),
        float(sum(item.completed_count for item in tenant_accounting)),
        sum(item.residual for item in tenant_accounting),
        len(rates) / max(1, len(tenant_accounting)),
        0.0 if not rates else sum(rates)/len(rates), max(rates, default=0.0),
        float(sum(item.active_demand_slots for item in communication_accounting)),
        sum(item.shortfall_sum for item in communication_accounting),
        sum(item.residual_sum for item in communication_accounting),
    )
    return tuple(_finite(value, "global feature") for value in values)


def build_observation(
    current_slot: int, focal_request: SensingRequest, requests: Iterable[SensingRequest],
    sessions: Iterable[SensingSession], target_primitives: Iterable[TargetSlotPrimitive],
    communication_primitives: Iterable[CommunicationSlotPrimitive],
    feasibility: CurrentFeasibilitySnapshot, action_masks: ActionMaskSnapshot,
    tenant_accounting: Iterable[TenantAccountingState],
    communication_accounting: Iterable[CommunicationAccountingState],
    cumulative_completed_value: float, cumulative_sensing_cost: float, config: CanonicalConfig,
) -> ObservationSnapshot:
    """Build the single edge-free learner view from current scientific state."""
    request_values = tuple(sorted(
        requests, key=lambda item: identifier_key(item.request_id),
    ))
    session_values = tuple(sorted(
        sessions, key=lambda item: identifier_key(item.session_id),
    ))
    target_values = tuple(sorted(
        target_primitives, key=lambda item: identifier_key(item.target_id),
    ))
    communication_values = tuple(sorted(
        communication_primitives, key=lambda item: identifier_key(item.user_id),
    ))
    target_index = _typed_index(target_values, "target_id")
    tenant_values = tuple(sorted(
        tenant_accounting, key=lambda item: identifier_key(item.tenant_id),
    ))
    communication_accounting_values = tuple(sorted(
        communication_accounting, key=lambda item: identifier_key(item.user_id),
    ))
    tenant_index = _typed_index(tenant_values, "tenant_id")

    request_table = _request_table(
        current_slot, focal_request, request_values, target_index,
        feasibility, tenant_index, config,
    )
    session_table = _session_table(
        current_slot, request_values, session_values, target_index, config,
    )
    global_specs = _global_specs()
    global_features = _global_features(
        current_slot, request_values, session_values, communication_values,
        tenant_values, communication_accounting_values,
        float(cumulative_completed_value), float(cumulative_sensing_cost), config,
    )
    view = ObservationView(request_table, session_table, global_specs, global_features, action_masks)
    return ObservationSnapshot(view)