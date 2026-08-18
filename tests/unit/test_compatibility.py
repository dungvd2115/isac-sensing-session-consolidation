from __future__ import annotations

from dataclasses import replace
from inspect import signature

import pytest

import isac_ssc.core.compatibility as compatibility_module
from isac_ssc.core.compatibility import (
    CompatibilityValidationError, SensingPrimitiveState, all_members_update_period_feasible,
    all_merge_permissions_enabled, build_merge_candidate_session, cross_tenant_pairs_authorized,
    evaluate_create_profile, evaluate_merge_profile,
    exact_target_match, existing_members_preserved, merge_authorized,
    request_quality_margin, request_shared_output_valid,
    request_start_feasible, request_update_period_feasible, service_lifetime_feasible,
    session_active, spatial_coverage_feasible, target_in_request_region,
    target_in_session_region, task_output_capable,
)
from isac_ssc.core.entities import (
    DiskAOI, RequestState, ResourceProfile, SensingRequest, SensingSession, Task,
)
from isac_ssc.core.quality import SensingParameters, evaluate_shared_sensing_quality
from isac_ssc.utils.config import load_config

CONFIG = load_config()
DURATIONS = CONFIG.service_duration_slots
PARAMETERS = SensingParameters.from_config(CONFIG)
TOTAL_BANDWIDTH = CONFIG.system["total_bandwidth_hz"]
TOTAL_POWER = CONFIG.system["total_power_w"]
HORIZON = CONFIG.system["horizon_slots"]
COVERAGE = CONFIG.compatibility["minimum_spatial_coverage_ratio"]
TRACKING_PRIOR = tuple(
    tuple(float(CONFIG.sensing["tracking"]["initial_covariance_diag"][row]) if row == column else 0.0
          for column in range(4)) for row in range(4)
)
AOI = DiskAOI((80.0, 0.0), 30.0)
PRIMITIVE = SensingPrimitiveState((80.0, 0.0), (0.0, 0.0), 1.0, 0.0, 1.0, TRACKING_PRIOR)


def _request(
    request_id: int, task: Task = Task.DETECTION, tenant_id: str = "tenant_1",
    target_id: int = 7, permission: bool = True, state: RequestState = RequestState.WAITING,
    arrival_slot: int = 1, latest_start_slot: int = 8, interval: int = 2,
    threshold: float | None = None, completion_value: float = 1.0,
) -> SensingRequest:
    default_threshold = 0.1 if task is Task.DETECTION else 100.0
    request = SensingRequest(
        request_id, tenant_id, arrival_slot, latest_start_slot, AOI, target_id, task,
        default_threshold if threshold is None else threshold, interval, completion_value, permission,
    )
    return request if state is RequestState.WAITING else request.transition(state, slot=arrival_slot)


def _session(profile_name: str = "balanced", creator: SensingRequest | None = None,
             session_id: int = 1) -> tuple[SensingSession, SensingRequest]:
    creator = creator or _request(1, Task.TRACKING, state=RequestState.ACTIVE, arrival_slot=0)
    session = SensingSession.create(
        session_id, creator, CONFIG.resource_profiles[profile_name], creator.admission_slot,
        DURATIONS, TRACKING_PRIOR if creator.task is Task.TRACKING else None,
    )
    return session, creator


def _merge_assessment(focal: SensingRequest | None = None, session: SensingSession | None = None,
                      members: tuple[SensingRequest, ...] | None = None,
                      profile: ResourceProfile | None = None, active_sessions=None):
    if session is None:
        session, creator = _session()
        members = (creator,)
    focal = focal or _request(2)
    members = members or ()
    return evaluate_merge_profile(
        focal, session, members, CONFIG.tenants, profile or CONFIG.resource_profiles["balanced"],
        PRIMITIVE, PARAMETERS, tuple(active_sessions) if active_sessions is not None else (session,),
        DURATIONS, 1, HORIZON, COVERAGE, TOTAL_BANDWIDTH, TOTAL_POWER,
    )


def test_lifecycle_time_target_region_and_task_predicates() -> None:
    session, creator = _session()
    focal = _request(2)
    assert request_start_feasible(focal, 1)
    assert not request_start_feasible(replace(focal, eligible_slot=2), 1)
    assert session_active(session, 1) and not session_active(session, session.final_active_slot + 1)
    assert service_lifetime_feasible(focal, 1, DURATIONS, HORIZON)
    assert not service_lifetime_feasible(focal, HORIZON - 1, DURATIONS, HORIZON)
    assert exact_target_match(focal, session)
    assert not exact_target_match(replace(focal, target_id=99), session)
    assert target_in_request_region(focal, PRIMITIVE.target_position_m)
    assert target_in_session_region(session, PRIMITIVE.target_position_m)
    assert task_output_capable(focal, session)
    detection_session, _ = _session(creator=_request(
        3, Task.DETECTION, state=RequestState.ACTIVE, arrival_slot=0,
    ), session_id=3)
    assert not task_output_capable(_request(4, Task.LOCALIZATION), detection_session)
    assert creator.state is RequestState.ACTIVE


def test_spatial_coverage_uses_exact_request_over_session_area_ratio() -> None:
    session, _ = _session()
    request = replace(_request(2), aoi=DiskAOI((80.0, 0.0), 15.0))
    assert spatial_coverage_feasible(request, session, 1.0)
    shifted = replace(request, aoi=DiskAOI((105.0, 0.0), 15.0))
    assert not spatial_coverage_feasible(shifted, session, COVERAGE)


def test_d03_b_requires_every_permission_including_same_tenant() -> None:
    focal = _request(2)
    member = _request(1, Task.TRACKING, state=RequestState.ACTIVE, arrival_slot=0)
    assert all_merge_permissions_enabled(focal, (member,))
    blocked_member = replace(member, merge_permission=False)
    assert not all_merge_permissions_enabled(focal, (blocked_member,))
    assert not merge_authorized(focal, (blocked_member,), CONFIG.tenants)
    assert not merge_authorized(replace(focal, merge_permission=False), (member,), CONFIG.tenants)


def test_cross_tenant_authorization_checks_every_represented_pair() -> None:
    focal = _request(2, tenant_id="tenant_4")
    member = _request(1, Task.TRACKING, tenant_id="tenant_1", state=RequestState.ACTIVE, arrival_slot=0)
    assert not cross_tenant_pairs_authorized(focal, (member,), CONFIG.tenants)
    assert not merge_authorized(focal, (member,), CONFIG.tenants)
    allowed = replace(focal, tenant_id="tenant_2")
    assert cross_tenant_pairs_authorized(allowed, (member,), CONFIG.tenants)


def test_update_period_checks_focal_and_every_existing_member() -> None:
    fast, slow = CONFIG.resource_profiles["rapid"], CONFIG.resource_profiles["economical"]
    focal = _request(2, interval=1)
    member = _request(1, Task.TRACKING, state=RequestState.ACTIVE, arrival_slot=0, interval=2)
    assert request_update_period_feasible(focal, fast)
    assert not request_update_period_feasible(focal, slow)
    assert all_members_update_period_feasible(focal, (member,), fast)
    assert not all_members_update_period_feasible(focal, (member,), slow)


def test_task_specific_quality_margins_use_one_shared_output() -> None:
    session, _ = _session()
    shared = evaluate_shared_sensing_quality(
        session, PRIMITIVE.target_position_m, PRIMITIVE.bs_position_m, 1.0, 0.0, 1.0,
        PARAMETERS, TRACKING_PRIOR,
    )
    detection = _request(2, Task.DETECTION, threshold=shared.detection_probability)
    localization = _request(3, Task.LOCALIZATION, threshold=shared.localization.peb_m)
    tracking = _request(4, Task.TRACKING, threshold=shared.tracking.pcrb_m)
    assert request_quality_margin(detection, session, shared).margin == pytest.approx(0.0)
    assert request_quality_margin(localization, session, shared).margin == pytest.approx(0.0)
    assert request_quality_margin(tracking, session, shared).margin == pytest.approx(0.0)
    assert all(request_shared_output_valid(request, session, shared)
               for request in (detection, localization, tracking))


def test_candidate_builders_preserve_fixed_session_contract_and_private_create_ignores_chi() -> None:
    session, creator = _session()
    focal = _request(2)
    candidate = build_merge_candidate_session(
        focal, session, CONFIG.resource_profiles["rapid"], 1, DURATIONS,
    )
    assert candidate.aoi == session.aoi and candidate.target_id == session.target_id
    assert candidate.base_task == session.base_task and candidate.exposed_outputs == session.exposed_outputs
    assert candidate.next_update_slot == 1 and candidate.member_request_ids == (creator.request_id, focal.request_id)

    private = replace(focal, merge_permission=False)
    create = evaluate_create_profile(
        private, 99, CONFIG.resource_profiles["balanced"], PRIMITIVE, PARAMETERS, (session,),
        DURATIONS, 1, HORIZON, TOTAL_BANDWIDTH, TOTAL_POWER,
    )
    assert create.feasible
    assert create.candidate_session.member_request_ids == (private.request_id,)


def test_tracking_create_uses_configured_creation_prior_not_unrelated_primitive_prior() -> None:
    request = _request(20, Task.TRACKING)
    unrelated = tuple(tuple(999.0 if row == column else 0.0 for column in range(4)) for row in range(4))
    primitive = replace(PRIMITIVE, tracking_prior_covariance=unrelated)
    assessment = evaluate_create_profile(
        request, 120, CONFIG.resource_profiles["balanced"], primitive, PARAMETERS, (),
        DURATIONS, 1, HORIZON, TOTAL_BANDWIDTH, TOTAL_POWER, TRACKING_PRIOR,
    )
    direct = evaluate_shared_sensing_quality(
        assessment.candidate_session, primitive.target_position_m, primitive.bs_position_m,
        primitive.target_rcs_m2, primitive.sensing_shadowing_db,
        primitive.sensing_fading_power_gain, PARAMETERS, TRACKING_PRIOR,
    )
    assert assessment.feasible and assessment.shared_quality == direct


def test_merge_evaluates_shared_physics_once_and_preserves_members(monkeypatch) -> None:
    session, creator = _session()
    calls = 0
    original = compatibility_module.evaluate_shared_sensing_quality

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(compatibility_module, "evaluate_shared_sensing_quality", counted)
    assessment = _merge_assessment(session=session, members=(creator,))
    assert calls == 1 and assessment.feasible and assessment.existing_members
    assert existing_members_preserved(
        session, assessment.candidate_session, (creator,), assessment.shared_quality,
        reservation_is_feasible=True,
    )


def test_member_mapping_mismatch_is_corrupted_state_not_normal_infeasibility() -> None:
    session, _ = _session()
    with pytest.raises(CompatibilityValidationError, match="match session"):
        _merge_assessment(session=session, members=())


def test_composite_different_target_returns_infeasible_instead_of_crashing() -> None:
    session, creator = _session()
    assessment = _merge_assessment(
        replace(_request(2), target_id=999), session, (creator,),
    )
    assert not assessment.exact_target and not assessment.feasible
    assert assessment.candidate_session is None and assessment.shared_quality is None


def test_resource_reservation_can_make_otherwise_valid_merge_infeasible() -> None:
    session, creator = _session("precision")
    other_creator = _request(9, Task.TRACKING, state=RequestState.ACTIVE, arrival_slot=0)
    third_creator = _request(10, Task.TRACKING, state=RequestState.ACTIVE, arrival_slot=0)
    other, _ = _session("precision", other_creator, session_id=9)
    third, _ = _session("precision", third_creator, session_id=10)
    other, third = replace(other, next_update_slot=1), replace(third, next_update_slot=1)
    assessment = _merge_assessment(
        session=session, members=(creator,), profile=CONFIG.resource_profiles["precision"],
        active_sessions=(session, other, third),
    )
    assert not assessment.reservation and not assessment.feasible


def test_feasibility_is_independent_of_request_value_and_reward_weights() -> None:
    session, creator = _session()
    low = _request(2, completion_value=0.1)
    high = replace(low, completion_value=1000.0)
    low_result = _merge_assessment(low, session, (creator,))
    high_result = _merge_assessment(high, session, (creator,))
    assert low_result.feasible == high_result.feasible
    for function in (evaluate_merge_profile, evaluate_create_profile):
        parameters = signature(function).parameters
        assert "completion_value" not in parameters and "reward_weight" not in parameters


def test_compatibility_module_has_no_graph_representation_payload() -> None:
    source = compatibility_module.__dict__
    assert "CompatibilityEdgeAssessment" not in source
    assert "ProfileEdgeFeatures" not in source
    assert "ReservationDelta" not in source
    assert "evaluate_compatibility_edge" not in source