"""Shared candidate calculations for deterministic online baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from isac_ssc.core.compatibility import (
    CreateProfileAssessment, MergeProfileAssessment, QualityMargin,
)
from isac_ssc.core.entities import ResourceProfile, SensingRequest, SensingSession
from isac_ssc.core.resources import (
    SensingResourceUsage, normalized_sensing_resource_cost, sensing_resource_usage,
)
from isac_ssc.envs.action_space import (
    ActionType, EnvironmentAction, canonical_action_key,
)


@dataclass(frozen=True, slots=True)
class ImmediateServiceCandidate:
    action: EnvironmentAction
    absolute_profile_cost: float
    incremental_current_cost: float
    normalized_sla_margin: float


def fallback_action(defer_feasible: bool) -> EnvironmentAction:
    return EnvironmentAction(
        ActionType.DEFER if defer_feasible else ActionType.REJECT,
    )


def absolute_profile_cost(
    profile: ResourceProfile, total_bandwidth_hz: float, total_power_w: float,
    bandwidth_weight: float, power_weight: float,
) -> float:
    usage = SensingResourceUsage(
        profile.sensing_bandwidth_hz, profile.sensing_power_w,
    )
    return normalized_sensing_resource_cost(
        usage, total_bandwidth_hz, total_power_w,
        bandwidth_weight, power_weight,
    )


def _normalized_request_margin(
    request: SensingRequest, quality_margin: QualityMargin,
    profile: ResourceProfile,
) -> float:
    quality = quality_margin.margin/request.quality_threshold
    periodicity = (
        request.valid_output_interval_slots - (profile.update_period_slots-1)
    )/max(1, request.valid_output_interval_slots)
    return float(min(quality, periodicity))


def _normalized_cost(
    usage: SensingResourceUsage, total_bandwidth_hz: float, total_power_w: float,
    bandwidth_weight: float, power_weight: float,
) -> float:
    return normalized_sensing_resource_cost(
        usage, total_bandwidth_hz, total_power_w,
        bandwidth_weight, power_weight,
    )


def build_create_candidate(
    request: SensingRequest, profile: ResourceProfile,
    assessment: CreateProfileAssessment,
    active_sessions: Iterable[SensingSession], current_slot: int,
    total_bandwidth_hz: float, total_power_w: float,
    bandwidth_weight: float, power_weight: float,
) -> ImmediateServiceCandidate:
    """Build one feasible CREATE candidate from the retained mask assessment."""
    sessions = tuple(active_sessions)
    before = sensing_resource_usage(sessions, current_slot)
    after = sensing_resource_usage(
        (*sessions, assessment.candidate_session), current_slot,
    )
    incremental = _normalized_cost(
        after, total_bandwidth_hz, total_power_w,
        bandwidth_weight, power_weight,
    ) - _normalized_cost(
        before, total_bandwidth_hz, total_power_w,
        bandwidth_weight, power_weight,
    )
    return ImmediateServiceCandidate(
        EnvironmentAction(ActionType.CREATE, profile_id=profile.profile_id),
        absolute_profile_cost(
            profile, total_bandwidth_hz, total_power_w,
            bandwidth_weight, power_weight,
        ),
        incremental,
        _normalized_request_margin(
            request, assessment.quality_margin, profile,
        ),
    )


def build_merge_candidate(
    focal: SensingRequest, members: Iterable[SensingRequest],
    session: SensingSession, profile: ResourceProfile,
    assessment: MergeProfileAssessment,
    active_sessions: Iterable[SensingSession], current_slot: int,
    total_bandwidth_hz: float, total_power_w: float,
    bandwidth_weight: float, power_weight: float,
) -> ImmediateServiceCandidate:
    """Build one feasible MERGE candidate from the retained mask assessment."""
    before_sessions = tuple(active_sessions)
    sessions = list(before_sessions)
    destination = next(
        index for index, active in enumerate(sessions)
        if active.session_id == session.session_id
    )
    sessions[destination] = assessment.candidate_session
    before = sensing_resource_usage(before_sessions, current_slot)
    after = sensing_resource_usage(sessions, current_slot)
    incremental = _normalized_cost(
        after, total_bandwidth_hz, total_power_w,
        bandwidth_weight, power_weight,
    ) - _normalized_cost(
        before, total_bandwidth_hz, total_power_w,
        bandwidth_weight, power_weight,
    )
    member_by_id = {request.request_id: request for request in members}
    margins = [
        _normalized_request_margin(
            focal, assessment.focal_margin, profile,
        ),
    ]
    margins.extend(
        _normalized_request_margin(
            member_by_id[request_id], margin, profile,
        )
        for request_id, margin in assessment.member_margins
    )
    return ImmediateServiceCandidate(
        EnvironmentAction(
            ActionType.MERGE, session.session_id, profile.profile_id,
        ),
        absolute_profile_cost(
            profile, total_bandwidth_hz, total_power_w,
            bandwidth_weight, power_weight,
        ),
        incremental, min(margins),
    )


__all__ = [
    "ImmediateServiceCandidate", "absolute_profile_cost",
    "build_create_candidate", "build_merge_candidate",
    "canonical_action_key", "fallback_action",
]