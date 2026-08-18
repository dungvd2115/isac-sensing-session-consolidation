from __future__ import annotations

from dataclasses import replace

import pytest

from isac_ssc.core.entities import (
    CommunicationUser, DiskAOI, RequestState, ResourceProfile, SensingRequest, SensingSession, Task,
)
from isac_ssc.core.resources import (
    ResourceValidationError, SensingResourceUsage, candidate_update_slots,
    committed_resource_usage, equal_share_communication_resources, next_scheduled_update_slot,
    normalized_sensing_resource_cost, profile_dominates, reservation_feasible,
    residual_communication_resources, scheduled_update_slots, sensing_resource_usage,
    session_updates_at,
)
from isac_ssc.core.utility import completed_request_value, finite_horizon_return, slot_reward
from isac_ssc.utils.config import load_config

CONFIG = load_config()
DURATIONS = CONFIG.service_duration_slots
TOTAL_BANDWIDTH = CONFIG.system["total_bandwidth_hz"]
TOTAL_POWER = CONFIG.system["total_power_w"]


def _request(request_id: int, task: Task = Task.TRACKING, admission_slot: int = 0,
             latest_start_slot: int = 10, completion_value: float = 2.0) -> SensingRequest:
    threshold = 0.9 if task is Task.DETECTION else 4.0
    return SensingRequest(
        request_id=request_id, tenant_id="tenant_1", arrival_slot=0,
        latest_start_slot=latest_start_slot, aoi=DiskAOI((80.0, 0.0), 25.0), target_id=1,
        task=task, quality_threshold=threshold, valid_output_interval_slots=2,
        completion_value=completion_value, merge_permission=True,
    ).transition(RequestState.ACTIVE, slot=admission_slot)


def _session(session_id: int, profile_name: str = "balanced", task: Task = Task.TRACKING,
             admission_slot: int = 0, request_id: int | None = None) -> SensingSession:
    request = _request(request_id or session_id, task, admission_slot)
    tracking = tuple(tuple(1.0 if row == column else 0.0 for column in range(4)) for row in range(4))
    return SensingSession.create(
        session_id, request, CONFIG.resource_profiles[profile_name], admission_slot, DURATIONS,
        tracking if task is Task.TRACKING else None,
    )


def test_update_calendars_follow_session_state_and_admission_reset() -> None:
    session = _session(1, "balanced", Task.TRACKING)
    assert scheduled_update_slots(session) == (0, 2, 4, 6)
    assert session_updates_at(session, 0) and session_updates_at(session, 4)
    assert not session_updates_at(session, 1) and not session_updates_at(session, 7)

    rapid = CONFIG.resource_profiles["rapid"]
    assert candidate_update_slots(3, 7, rapid) == (3, 4, 5, 6, 7)
    assert next_scheduled_update_slot(3, 7, rapid) == 4
    assert next_scheduled_update_slot(7, 7, rapid) is None


def test_merge_resets_calendar_to_current_slot_and_extends_lifetime() -> None:
    session = _session(1, "economical", Task.TRACKING)
    member = _request(2, Task.TRACKING, admission_slot=2)
    merged = session.with_member(member, CONFIG.resource_profiles["balanced"], 2, DURATIONS)
    assert merged.next_update_slot == 2
    assert merged.final_active_slot == 9
    assert scheduled_update_slots(merged) == (2, 4, 6, 8)


def test_current_sensing_occupancy_and_residual_resources_conserve_totals() -> None:
    first = _session(1, "balanced")
    second = _session(2, "economical")
    usage = sensing_resource_usage((first, second), 0)
    assert usage.updating_session_ids == (1, 2)
    assert usage.sensing_bandwidth_hz == 6.0e6
    assert usage.sensing_power_w == 7.0

    residual = residual_communication_resources(TOTAL_BANDWIDTH, TOTAL_POWER, usage)
    assert usage.sensing_bandwidth_hz + residual.communication_bandwidth_hz == TOTAL_BANDWIDTH
    assert usage.sensing_power_w + residual.communication_power_w == TOTAL_POWER


def test_committed_overflow_fails_and_candidate_reservation_returns_false() -> None:
    sessions = tuple(_session(index, "precision") for index in range(3))
    assert not reservation_feasible(sessions, TOTAL_BANDWIDTH, TOTAL_POWER)
    with pytest.raises(ResourceValidationError):
        committed_resource_usage(sessions, TOTAL_BANDWIDTH, TOTAL_POWER)

    feasible = (_session(1, "rapid"), _session(2, "balanced"))
    usage = committed_resource_usage(feasible, TOTAL_BANDWIDTH, TOTAL_POWER)
    assert tuple(item.slot for item in usage) == tuple(range(8))
    assert reservation_feasible(feasible, TOTAL_BANDWIDTH, TOTAL_POWER)


def test_future_reservation_uses_only_known_session_calendars() -> None:
    first = _session(1, "economical")
    delayed = replace(_session(2, "precision"), next_update_slot=1)
    usage = committed_resource_usage((first, delayed), TOTAL_BANDWIDTH, TOTAL_POWER)
    by_slot = {item.slot: item for item in usage}
    assert by_slot[0].updating_session_ids == (1,)
    assert by_slot[1].updating_session_ids == (2,)
    assert by_slot[3].updating_session_ids == (1, 2)
    assert 8 not in by_slot


def test_equal_share_scheduler_allocates_only_to_active_demand_users() -> None:
    users = (
        CommunicationUser(2, (0.0, 0.0), (0.0, 0.0), 4.0e6, 2.0e6, 0.05),
        CommunicationUser(1, (0.0, 0.0), (0.0, 0.0), 0.0, 2.0e6, 0.05),
        CommunicationUser(3, (0.0, 0.0), (0.0, 0.0), 1.0e6, 2.0e6, 0.05),
    )
    residual = residual_communication_resources(
        TOTAL_BANDWIDTH, TOTAL_POWER, SensingResourceUsage(4.0e6, 8.0),
    )
    allocations = equal_share_communication_resources(users, residual)
    by_user = {allocation.user_id: allocation for allocation in allocations}
    assert by_user[1].bandwidth_hz == by_user[1].power_w == 0.0
    assert by_user[2].bandwidth_hz == by_user[3].bandwidth_hz == 8.0e6
    assert by_user[2].power_w == by_user[3].power_w == 16.0
    assert sum(item.bandwidth_hz for item in allocations) == residual.communication_bandwidth_hz
    assert sum(item.power_w for item in allocations) == residual.communication_power_w


def test_profile_partial_order_matches_only_the_locked_three_conditions() -> None:
    profiles = CONFIG.resource_profiles
    expected = {
        ("balanced", "economical"), ("precision", "economical"),
        ("rapid", "economical"), ("rapid", "balanced"),
    }
    actual = {
        (candidate_name, reference_name)
        for candidate_name, candidate in profiles.items()
        for reference_name, reference in profiles.items()
        if candidate_name != reference_name and profile_dominates(candidate, reference)
    }
    assert actual == expected
    assert not profile_dominates(profiles["precision"], profiles["balanced"])
    assert not profile_dominates(profiles["rapid"], profiles["precision"])


def test_normalized_sensing_cost_uses_bandwidth_and_power_once() -> None:
    usage = SensingResourceUsage(4.0e6, 8.0)
    cost = normalized_sensing_resource_cost(usage, TOTAL_BANDWIDTH, TOTAL_POWER, 0.5, 0.5)
    assert cost == pytest.approx(0.5 * 4.0e6 / TOTAL_BANDWIDTH + 0.5 * 8.0 / TOTAL_POWER)
    with pytest.raises(ResourceValidationError):
        normalized_sensing_resource_cost(usage, TOTAL_BANDWIDTH, TOTAL_POWER, 0.6, 0.5)


def test_utility_contains_only_completed_value_minus_weighted_resource_cost() -> None:
    completed = _request(1, Task.DETECTION, completion_value=1.5).transition(RequestState.COMPLETED)
    failed = _request(2, Task.DETECTION, completion_value=9.0).transition(RequestState.FAILED)
    waiting = SensingRequest(
        3, "tenant_1", 0, 4, DiskAOI((80.0, 0.0), 25.0), 1, Task.DETECTION,
        0.9, 2, 8.0, True,
    )
    value = completed_request_value((completed, failed, waiting))
    assert value == 1.5
    reward = slot_reward(value, sensing_resource_cost=0.4, sensing_resource_cost_weight=0.2)
    assert reward == pytest.approx(1.42)
    assert finite_horizon_return((reward, -0.1, 0.5)) == pytest.approx(1.82)