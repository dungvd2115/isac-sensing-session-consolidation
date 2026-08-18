"""Deterministic request-session compatibility and profile-feasibility predicates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterable, Mapping

from isac_ssc.core.entities import (
    EntityId, Matrix4, RequestState, ResourceProfile, SensingRequest, SensingSession,
    Task, TaskDurationMap, Tenant, Vector2,
)
from isac_ssc.core.quality import (
    SharedSensingQuality, SensingParameters, aoi_coverage_ratio,
    evaluate_shared_sensing_quality, point_in_disk,
)
from isac_ssc.core.resources import reservation_feasible


class CompatibilityValidationError(ValueError):
    """Raised when compatibility evaluation receives inconsistent state."""


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise CompatibilityValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise CompatibilityValidationError(f"{name} must be >= {minimum}")
    return number


def _slot(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompatibilityValidationError(f"{name} must be a non-negative integer slot")
    return value


def _members_for_session(session: SensingSession, members: Iterable[SensingRequest]) -> tuple[SensingRequest, ...]:
    member_values = tuple(members)
    if any(not isinstance(request, SensingRequest) for request in member_values):
        raise CompatibilityValidationError("members must contain SensingRequest values")
    by_id = {request.request_id: request for request in member_values}
    if len(by_id) != len(member_values) or set(by_id) != set(session.member_request_ids):
        raise CompatibilityValidationError("member requests must match session.member_request_ids exactly")
    ordered = tuple(by_id[request_id] for request_id in session.member_request_ids)
    if any(request.state is not RequestState.ACTIVE for request in ordered):
        raise CompatibilityValidationError("current session members must be ACTIVE")
    return ordered


def _tenant_index(tenants: Iterable[Tenant]) -> tuple[tuple[Tenant, ...], Mapping[EntityId, int]]:
    values = tuple(tenants)
    if any(not isinstance(tenant, Tenant) for tenant in values):
        raise CompatibilityValidationError("tenants must contain Tenant values")
    index = {tenant.tenant_id: position for position, tenant in enumerate(values)}
    if len(index) != len(values) or any(len(tenant.authorization_row) != len(values) for tenant in values):
        raise CompatibilityValidationError("tenant authorization rows must match the canonical tenant order")
    return values, index


def _replace_session(sessions: Iterable[SensingSession], candidate: SensingSession) -> tuple[SensingSession, ...]:
    values = tuple(sessions)
    matches = sum(session.session_id == candidate.session_id for session in values)
    if matches != 1:
        raise CompatibilityValidationError("active sessions must contain the destination session exactly once")
    return tuple(candidate if session.session_id == candidate.session_id else session for session in values)


@dataclass(frozen=True, slots=True)
class SensingPrimitiveState:
    target_position_m: Vector2
    bs_position_m: Vector2
    target_rcs_m2: float
    sensing_shadowing_db: float
    sensing_fading_power_gain: float
    tracking_prior_covariance: Matrix4 | None = None

    def __post_init__(self) -> None:
        for name in ("target_rcs_m2", "sensing_fading_power_gain"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0))
        object.__setattr__(self, "sensing_shadowing_db", _finite(
            self.sensing_shadowing_db, "sensing_shadowing_db",
        ))


@dataclass(frozen=True, slots=True)
class QualityMargin:
    task: Task
    margin: float | None

    @property
    def valid(self) -> bool:
        return self.margin is not None and self.margin >= 0.0


@dataclass(frozen=True, slots=True)
class MergeProfileAssessment:
    profile_id: str
    request_start: bool
    active_session: bool
    lifetime: bool
    authorization: bool
    exact_target: bool
    spatial_coverage: bool
    focal_region: bool
    session_region: bool
    task_capability: bool
    update_period: bool
    current_quality: bool
    reservation: bool
    existing_members: bool
    focal_margin: QualityMargin
    member_margins: tuple[tuple[EntityId, QualityMargin], ...]
    candidate_session: SensingSession | None
    shared_quality: SharedSensingQuality | None

    @property
    def feasible(self) -> bool:
        return all((
            self.request_start, self.active_session, self.lifetime, self.authorization,
            self.exact_target, self.spatial_coverage, self.focal_region, self.session_region,
            self.task_capability, self.update_period, self.current_quality, self.reservation,
            self.existing_members,
        ))


@dataclass(frozen=True, slots=True)
class CreateProfileAssessment:
    profile_id: str
    request_start: bool
    lifetime: bool
    target_region: bool
    update_period: bool
    current_quality: bool
    reservation: bool
    quality_margin: QualityMargin
    candidate_session: SensingSession | None
    shared_quality: SharedSensingQuality | None

    @property
    def feasible(self) -> bool:
        return all((
            self.request_start, self.lifetime, self.target_region, self.update_period,
            self.current_quality, self.reservation,
        ))


def request_start_feasible(request: SensingRequest, current_slot: int) -> bool:
    current = _slot(current_slot, "current_slot")
    return bool(
        isinstance(request, SensingRequest) and request.state is RequestState.WAITING
        and request.eligible_slot <= current <= request.latest_start_slot
    )


def session_active(session: SensingSession, current_slot: int) -> bool:
    current = _slot(current_slot, "current_slot")
    return bool(isinstance(session, SensingSession) and session.creation_slot <= current <= session.final_active_slot)


def service_lifetime_feasible(
    request: SensingRequest, current_slot: int, service_durations: TaskDurationMap,
    horizon_slots: int, *, existing_final_slot: int | None = None,
) -> bool:
    current = _slot(current_slot, "current_slot")
    horizon = _slot(horizon_slots, "horizon_slots")
    if horizon == 0:
        return False
    request_final = current + request.service_duration_slots(service_durations) - 1
    candidate_final = max(request_final, existing_final_slot) if existing_final_slot is not None else request_final
    return candidate_final < horizon


def exact_target_match(request: SensingRequest, session: SensingSession) -> bool:
    return request.target_id == session.target_id


def spatial_coverage_feasible(request: SensingRequest, session: SensingSession,
                              minimum_ratio: float) -> bool:
    threshold = _finite(minimum_ratio, "minimum_ratio", minimum=0.0)
    if threshold > 1.0:
        raise CompatibilityValidationError("minimum_ratio must be <= 1")
    return aoi_coverage_ratio(request.aoi, session.aoi) >= threshold


def target_in_request_region(request: SensingRequest, target_position_m: Vector2) -> bool:
    return point_in_disk(target_position_m, request.aoi)


def target_in_session_region(session: SensingSession, target_position_m: Vector2) -> bool:
    return point_in_disk(target_position_m, session.aoi)


def task_output_capable(request: SensingRequest, session: SensingSession) -> bool:
    return request.task in session.exposed_outputs


def all_merge_permissions_enabled(focal: SensingRequest, members: Iterable[SensingRequest]) -> bool:
    return focal.merge_permission and all(request.merge_permission for request in members)


def cross_tenant_pairs_authorized(
    focal: SensingRequest, members: Iterable[SensingRequest], tenants: Iterable[Tenant],
) -> bool:
    tenant_values, index = _tenant_index(tenants)
    represented = {focal.tenant_id, *(request.tenant_id for request in members)}
    if not represented.issubset(index):
        raise CompatibilityValidationError("request references an unknown tenant")
    represented_indices = sorted(index[tenant_id] for tenant_id in represented)
    for left_offset, left in enumerate(represented_indices):
        for right in represented_indices[left_offset + 1:]:
            if not tenant_values[left].authorization_row[right] or not tenant_values[right].authorization_row[left]:
                return False
    return True


def merge_authorized(focal: SensingRequest, members: Iterable[SensingRequest],
                     tenants: Iterable[Tenant]) -> bool:
    member_values = tuple(members)
    return all_merge_permissions_enabled(focal, member_values) and cross_tenant_pairs_authorized(
        focal, member_values, tenants,
    )


def request_update_period_feasible(request: SensingRequest, profile: ResourceProfile) -> bool:
    return profile.update_period_slots - 1 <= request.valid_output_interval_slots


def all_members_update_period_feasible(
    focal: SensingRequest, members: Iterable[SensingRequest], profile: ResourceProfile,
) -> bool:
    return all(request_update_period_feasible(request, profile) for request in (focal, *tuple(members)))


def request_quality_margin(
    request: SensingRequest, session: SensingSession, shared_quality: SharedSensingQuality,
) -> QualityMargin:
    if not task_output_capable(request, session) or not shared_quality.target_in_session_aoi:
        return QualityMargin(request.task, None)
    if request.task is Task.DETECTION:
        return QualityMargin(request.task, shared_quality.detection_probability - request.quality_threshold)
    if request.task is Task.LOCALIZATION:
        localization = shared_quality.localization
        margin = None if not localization.information_valid else request.quality_threshold - localization.peb_m
        return QualityMargin(request.task, margin)
    tracking = shared_quality.tracking
    margin = None if tracking is None or not tracking.measurement_updated else request.quality_threshold - tracking.pcrb_m
    return QualityMargin(request.task, margin)


def request_shared_output_valid(
    request: SensingRequest, session: SensingSession, shared_quality: SharedSensingQuality,
) -> bool:
    return request_quality_margin(request, session, shared_quality).valid


def build_merge_candidate_session(
    focal: SensingRequest, session: SensingSession, profile: ResourceProfile, current_slot: int,
    service_durations: TaskDurationMap,
) -> SensingSession:
    if not request_start_feasible(focal, current_slot):
        raise CompatibilityValidationError("merge candidate requires a currently eligible waiting request")
    admitted = focal.transition(RequestState.ACTIVE, slot=current_slot)
    return session.with_member(admitted, profile, current_slot, service_durations)


def build_create_candidate_session(
    request: SensingRequest, prospective_session_id: EntityId, profile: ResourceProfile,
    current_slot: int, service_durations: TaskDurationMap,
    tracking_initial_covariance: Matrix4 | None = None,
) -> SensingSession:
    if not request_start_feasible(request, current_slot):
        raise CompatibilityValidationError("create candidate requires a currently eligible waiting request")
    admitted = request.transition(RequestState.ACTIVE, slot=current_slot)
    return SensingSession.create(
        prospective_session_id, admitted, profile, current_slot, service_durations,
        tracking_initial_covariance if request.task is Task.TRACKING else None,
    )


def existing_members_preserved(
    original: SensingSession, candidate: SensingSession, members: Iterable[SensingRequest],
    shared_quality: SharedSensingQuality, *, reservation_is_feasible: bool,
) -> bool:
    member_values = _members_for_session(original, members)
    structural = (
        set(original.member_request_ids).issubset(candidate.member_request_ids)
        and candidate.final_active_slot >= original.final_active_slot
        and candidate.aoi == original.aoi and candidate.target_id == original.target_id
        and candidate.exposed_outputs == original.exposed_outputs
    )
    periods = all(request_update_period_feasible(request, candidate.profile) for request in member_values)
    quality = all(request_shared_output_valid(request, candidate, shared_quality) for request in member_values)
    return bool(structural and periods and quality and reservation_is_feasible)


def merge_reservation_feasible(
    active_sessions: Iterable[SensingSession], destination: SensingSession, candidate: SensingSession,
    total_bandwidth_hz: float, total_power_w: float, current_slot: int,
) -> bool:
    if destination.session_id != candidate.session_id:
        raise CompatibilityValidationError("merge candidate must retain the destination session identifier")
    sessions = _replace_session(active_sessions, candidate)
    return reservation_feasible(
        sessions, total_bandwidth_hz, total_power_w, start_slot=current_slot,
        end_slot=max(session.final_active_slot for session in sessions),
    )


def create_reservation_feasible(
    active_sessions: Iterable[SensingSession], candidate: SensingSession,
    total_bandwidth_hz: float, total_power_w: float, current_slot: int,
) -> bool:
    sessions = tuple(active_sessions)
    if any(session.session_id == candidate.session_id for session in sessions):
        raise CompatibilityValidationError("prospective session identifier is already active")
    combined = sessions + (candidate,)
    return reservation_feasible(
        combined, total_bandwidth_hz, total_power_w, start_slot=current_slot,
        end_slot=max(session.final_active_slot for session in combined),
    )


def evaluate_merge_profile(
    focal: SensingRequest, session: SensingSession, members: Iterable[SensingRequest],
    tenants: Iterable[Tenant], profile: ResourceProfile, primitive: SensingPrimitiveState,
    sensing_parameters: SensingParameters, active_sessions: Iterable[SensingSession],
    service_durations: TaskDurationMap, current_slot: int, horizon_slots: int,
    minimum_coverage_ratio: float, total_bandwidth_hz: float, total_power_w: float,
) -> MergeProfileAssessment:
    member_values = _members_for_session(session, members)
    start_ok = request_start_feasible(focal, current_slot)
    active_ok = session_active(session, current_slot)
    lifetime_ok = service_lifetime_feasible(
        focal, current_slot, service_durations, horizon_slots,
        existing_final_slot=session.final_active_slot,
    )
    authorization_ok = merge_authorized(focal, member_values, tenants)
    target_ok = exact_target_match(focal, session)
    coverage_ok = spatial_coverage_feasible(focal, session, minimum_coverage_ratio)
    focal_region_ok = target_in_request_region(focal, primitive.target_position_m)
    session_region_ok = target_in_session_region(session, primitive.target_position_m)
    task_ok = task_output_capable(focal, session)
    period_ok = all_members_update_period_feasible(focal, member_values, profile)
    empty_margin = QualityMargin(focal.task, None)
    empty_members = tuple((request.request_id, QualityMargin(request.task, None)) for request in member_values)
    if not (start_ok and active_ok and target_ok):
        return MergeProfileAssessment(
            profile.profile_id, start_ok, active_ok, lifetime_ok, authorization_ok, target_ok,
            coverage_ok, focal_region_ok, session_region_ok, task_ok, period_ok, False,
            False, False, empty_margin, empty_members, None, None,
        )
    candidate = build_merge_candidate_session(focal, session, profile, current_slot, service_durations)
    shared = evaluate_shared_sensing_quality(
        candidate, primitive.target_position_m, primitive.bs_position_m, primitive.target_rcs_m2,
        primitive.sensing_shadowing_db, primitive.sensing_fading_power_gain, sensing_parameters,
        primitive.tracking_prior_covariance,
    )
    reservation = merge_reservation_feasible(
        active_sessions, session, candidate, total_bandwidth_hz, total_power_w, current_slot,
    )
    focal_margin = request_quality_margin(focal, candidate, shared)
    member_margins = tuple((request.request_id, request_quality_margin(request, candidate, shared))
                           for request in member_values)
    current_quality = focal_margin.valid and all(margin.valid for _, margin in member_margins)
    preserved = existing_members_preserved(
        session, candidate, member_values, shared, reservation_is_feasible=reservation,
    )
    return MergeProfileAssessment(
        profile.profile_id, start_ok, active_ok, lifetime_ok, authorization_ok, target_ok,
        coverage_ok, focal_region_ok, session_region_ok, task_ok, period_ok, current_quality,
        reservation, preserved, focal_margin, member_margins, candidate, shared,
    )


def evaluate_create_profile(
    request: SensingRequest, prospective_session_id: EntityId, profile: ResourceProfile,
    primitive: SensingPrimitiveState, sensing_parameters: SensingParameters,
    active_sessions: Iterable[SensingSession], service_durations: TaskDurationMap,
    current_slot: int, horizon_slots: int, total_bandwidth_hz: float, total_power_w: float,
    tracking_initial_covariance: Matrix4 | None = None,
) -> CreateProfileAssessment:
    start_ok = request_start_feasible(request, current_slot)
    lifetime_ok = service_lifetime_feasible(request, current_slot, service_durations, horizon_slots)
    region_ok = target_in_request_region(request, primitive.target_position_m)
    period_ok = request_update_period_feasible(request, profile)
    if not start_ok:
        return CreateProfileAssessment(
            profile.profile_id, False, lifetime_ok, region_ok, period_ok, False, False,
            QualityMargin(request.task, None), None, None,
        )
    candidate = build_create_candidate_session(
        request, prospective_session_id, profile, current_slot, service_durations,
        tracking_initial_covariance,
    )
    create_prior = candidate.tracking_covariance if Task.TRACKING in candidate.exposed_outputs else None
    shared = evaluate_shared_sensing_quality(
        candidate, primitive.target_position_m, primitive.bs_position_m, primitive.target_rcs_m2,
        primitive.sensing_shadowing_db, primitive.sensing_fading_power_gain, sensing_parameters,
        create_prior,
    )
    margin = request_quality_margin(request, candidate, shared)
    reservation = create_reservation_feasible(
        active_sessions, candidate, total_bandwidth_hz, total_power_w, current_slot,
    )
    return CreateProfileAssessment(
        profile.profile_id, start_ok, lifetime_ok, region_ok, period_ok, margin.valid,
        reservation, margin, candidate, shared,
    )
