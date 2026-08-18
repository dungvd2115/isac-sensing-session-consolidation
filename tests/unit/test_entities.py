from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from isac_ssc.core.entities import (
    CommunicationUser, DiskAOI, EntityValidationError, RequestState, ResourceProfile,
    SensingRequest, SensingSession, Task, Tenant, task_outputs, task_service_duration_slots,
)

DURATIONS = {Task.DETECTION: 3, Task.LOCALIZATION: 4, Task.TRACKING: 8}
IDENTITY4 = (
    (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0),
)


def _profile(name: str = "balanced") -> ResourceProfile:
    return ResourceProfile(name, 4.0e6, 5.0, 2)


def _request(
    task: Task = Task.LOCALIZATION, request_id: int = 1, target_id: int = 7,
    merge_permission: bool = True, arrival_slot: int = 2, latest_start_slot: int = 6,
) -> SensingRequest:
    threshold = 0.9 if task is Task.DETECTION else 3.0
    return SensingRequest(
        request_id, "tenant_1", arrival_slot, latest_start_slot,
        DiskAOI((10.0, 5.0), 20.0), target_id, task, threshold, 2, 1.5, merge_permission,
    )


def test_task_outputs_and_service_durations_are_canonical() -> None:
    assert task_outputs(Task.DETECTION) == {Task.DETECTION}
    assert task_outputs(Task.LOCALIZATION) == {Task.DETECTION, Task.LOCALIZATION}
    assert task_outputs(Task.TRACKING) == set(Task)
    assert task_service_duration_slots(Task.TRACKING, DURATIONS) == 8

    with pytest.raises(EntityValidationError):
        task_service_duration_slots(Task.TRACKING, {Task.DETECTION: 3})


def test_entities_are_frozen_and_validate_finite_domains() -> None:
    tenant = Tenant("tenant_1", frozenset(Task), 0.05, (True, True))
    with pytest.raises(FrozenInstanceError):
        tenant.sla_violation_budget = 0.1
    with pytest.raises(EntityValidationError):
        DiskAOI((0.0, float("nan")), 1.0)
    with pytest.raises(EntityValidationError):
        CommunicationUser(1, (0.0, 0.0), (0.0, 0.0), -1.0, 1.0, 0.05)
    with pytest.raises(EntityValidationError):
        ResourceProfile("bad", 0.0, 1.0, 1)


def test_request_duration_service_interval_and_lifecycle() -> None:
    request = _request()
    assert request.state is RequestState.WAITING
    assert request.final_service_slot(DURATIONS) is None

    active = request.transition(RequestState.ACTIVE, slot=4)
    assert active.service_duration_slots(DURATIONS) == 4
    assert active.final_service_slot(DURATIONS) == 7
    assert active.service_slots(DURATIONS) == (4, 5, 6, 7)
    assert active.transition(RequestState.COMPLETED).is_terminal

    with pytest.raises(EntityValidationError):
        request.transition(RequestState.COMPLETED)
    with pytest.raises(EntityValidationError):
        request.transition(RequestState.ACTIVE, slot=7)
    with pytest.raises(EntityValidationError):
        request.transition("unknown")


def test_pre_admission_states_cannot_carry_admission_or_sla_accounting() -> None:
    kwargs = dict(
        request_id=2, tenant_id="tenant_1", arrival_slot=0, latest_start_slot=3,
        aoi=DiskAOI((0.0, 0.0), 10.0), target_id=1, task=Task.DETECTION,
        quality_threshold=0.9, valid_output_interval_slots=1, completion_value=1.0,
        merge_permission=True,
    )
    with pytest.raises(EntityValidationError):
        SensingRequest(**kwargs, state=RequestState.REJECTED, admission_slot=1)
    with pytest.raises(EntityValidationError):
        SensingRequest(**kwargs, state=RequestState.EXPIRED, valid_output_count=1)


def test_accounting_updates_are_active_only_monotone_and_absorbing() -> None:
    active = _request().transition(RequestState.ACTIVE, slot=3)
    violated = active.with_accounting(
        valid_output_age_slots=2, valid_output_count=1,
        sla_violated=True, first_violation_slot=5,
    )
    assert violated.sla_violated and violated.first_violation_slot == 5

    with pytest.raises(EntityValidationError):
        _request().with_accounting(
            valid_output_age_slots=0, valid_output_count=0,
            sla_violated=False, first_violation_slot=None,
        )
    with pytest.raises(EntityValidationError):
        violated.with_accounting(
            valid_output_age_slots=0, valid_output_count=0,
            sla_violated=True, first_violation_slot=5,
        )
    with pytest.raises(EntityValidationError):
        violated.with_accounting(
            valid_output_age_slots=0, valid_output_count=2,
            sla_violated=False, first_violation_slot=None,
        )


def test_accounting_update_validates_changed_state_fields() -> None:
    active = _request().transition(RequestState.ACTIVE, slot=3)
    invalid_updates = (
        dict(valid_output_age_slots=-1, valid_output_count=0, sla_violated=False, first_violation_slot=None),
        dict(valid_output_age_slots=0, valid_output_count=0.5, sla_violated=False, first_violation_slot=None),
        dict(valid_output_age_slots=0, valid_output_count=True, sla_violated=False, first_violation_slot=None),
        dict(valid_output_age_slots=0, valid_output_count=0, sla_violated=1, first_violation_slot=None),
        dict(valid_output_age_slots=0, valid_output_count=0, sla_violated=True, first_violation_slot=2),
    )
    for update in invalid_updates:
        with pytest.raises(EntityValidationError):
            active.with_accounting(**update)


def test_private_create_does_not_require_merge_permission() -> None:
    request = _request(merge_permission=False).transition(RequestState.ACTIVE, slot=2)
    session = SensingSession.create(1, request, _profile(), 2, DURATIONS)
    assert session.member_request_ids == (request.request_id,)


def test_session_creation_locks_target_aoi_output_set_and_current_update() -> None:
    request = _request(Task.TRACKING).transition(RequestState.ACTIVE, slot=2)
    session = SensingSession.create(1, request, _profile(), 2, DURATIONS, IDENTITY4)
    assert session.target_id == request.target_id
    assert session.aoi == request.aoi
    assert session.exposed_outputs == task_outputs(Task.TRACKING)
    assert session.next_update_slot == 2
    assert session.final_active_slot == 9

    with pytest.raises(EntityValidationError):
        SensingSession(
            1, 2, request.aoi, request.target_id, Task.TRACKING, {Task.TRACKING},
            (request.request_id,), _profile(), 2, 9, IDENTITY4,
        )


def test_merge_resets_update_to_current_slot_and_extends_end_only() -> None:
    creator = _request(request_id=1).transition(RequestState.ACTIVE, slot=2)
    session = SensingSession.create(1, creator, _profile(), 2, DURATIONS)
    member = _request(
        Task.DETECTION, request_id=2, arrival_slot=3, latest_start_slot=7,
    ).transition(RequestState.ACTIVE, slot=4)
    updated = session.with_member(member, _profile("rapid"), 4, DURATIONS)

    assert updated.aoi == session.aoi and updated.target_id == session.target_id
    assert updated.base_task == session.base_task
    assert updated.exposed_outputs == session.exposed_outputs
    assert updated.member_request_ids == (1, 2)
    assert updated.next_update_slot == 4
    assert updated.final_active_slot == 6

    post_update = updated.with_update_state(next_update_slot=5)
    assert post_update.profile == updated.profile
    with pytest.raises(TypeError):
        updated.with_update_state(profile=_profile(), next_update_slot=5)


def test_session_member_update_validates_changed_profile_and_slot() -> None:
    creator = _request(Task.DETECTION).transition(RequestState.ACTIVE, slot=2)
    session = SensingSession.create(1, creator, _profile(), 2, DURATIONS)
    member = _request(
        Task.DETECTION, request_id=2, arrival_slot=1, latest_start_slot=4,
    ).transition(RequestState.ACTIVE, slot=1)
    for invalid_slot in (False, 1.0):
        with pytest.raises(EntityValidationError, match="current_slot"):
            session.with_member(member, _profile(), invalid_slot, DURATIONS)
    with pytest.raises(EntityValidationError, match="active session interval"):
        session.with_member(member, _profile(), 1, DURATIONS)

    current_member = _request(
        Task.DETECTION, request_id=3, arrival_slot=3, latest_start_slot=4,
    ).transition(RequestState.ACTIVE, slot=3)
    with pytest.raises(EntityValidationError, match="ResourceProfile"):
        session.with_member(current_member, object(), 3, DURATIONS)


def test_session_rejects_duplicate_different_target_and_inactive_merge() -> None:
    creator = _request(Task.DETECTION).transition(RequestState.ACTIVE, slot=2)
    session = SensingSession.create(1, creator, _profile(), 2, DURATIONS)

    with pytest.raises(EntityValidationError):
        session.with_member(creator, _profile(), 2, DURATIONS)

    other_target = _request(
        Task.DETECTION, request_id=3, target_id=9,
    ).transition(RequestState.ACTIVE, slot=2)
    with pytest.raises(EntityValidationError):
        session.with_member(other_target, _profile(), 2, DURATIONS)

    late_member = _request(
        Task.DETECTION, request_id=4, arrival_slot=6, latest_start_slot=6,
    ).transition(RequestState.ACTIVE, slot=6)
    with pytest.raises(EntityValidationError):
        session.with_member(late_member, _profile(), 6, DURATIONS)


def test_tracking_session_requires_finite_4x4_covariance() -> None:
    tracking = _request(Task.TRACKING).transition(RequestState.ACTIVE, slot=2)
    with pytest.raises(EntityValidationError):
        SensingSession.create(2, tracking, _profile(), 2, DURATIONS)

        
def test_tracking_update_validates_changed_covariance() -> None:
    tracking = _request(Task.TRACKING).transition(RequestState.ACTIVE, slot=2)
    session = SensingSession.create(2, tracking, _profile(), 2, DURATIONS, IDENTITY4)
    with pytest.raises(EntityValidationError, match="4x4"):
        session.with_update_state(next_update_slot=3, tracking_covariance=((1.0, 0.0), (0.0, 1.0)))
    invalid = [list(row) for row in IDENTITY4]
    invalid[0][0] = float("nan")
    with pytest.raises(EntityValidationError, match="finite"):
        session.with_update_state(next_update_slot=3, tracking_covariance=tuple(tuple(row) for row in invalid))


def test_communication_shortfall_budget_is_nonnegative_without_probability_cap() -> None:
    user = CommunicationUser(1, (0.0, 0.0), (0.0, 0.0), 1.0, 1.0, 1.2)
    assert user.normalized_shortfall_budget == 1.2

    with pytest.raises(EntityValidationError):
        CommunicationUser(2, (0.0, 0.0), (0.0, 0.0), 1.0, 1.0, -0.1)

def test_defer_updates_eligibility_and_blocks_early_admission() -> None:
    request = _request(arrival_slot=2, latest_start_slot=6)
    assert request.eligible_slot == 2

    deferred = request.defer(current_slot=2, cooldown_slots=2)
    assert deferred.state is RequestState.WAITING
    assert deferred.eligible_slot == 4

    with pytest.raises(EntityValidationError):
        deferred.transition(RequestState.ACTIVE, slot=3)
    assert deferred.transition(RequestState.ACTIVE, slot=4).admission_slot == 4

    with pytest.raises(EntityValidationError):
        deferred.defer(current_slot=4, cooldown_slots=3)


def test_terminal_detachment_preserves_members_and_terminates_empty_session() -> None:
    creator = _request(Task.LOCALIZATION, request_id=1).transition(
        RequestState.ACTIVE, slot=2,
    )
    session = SensingSession.create(1, creator, _profile(), 2, DURATIONS)
    member = _request(
        Task.DETECTION, request_id=2, arrival_slot=3, latest_start_slot=7,
    ).transition(RequestState.ACTIVE, slot=4)
    session = session.with_member(member, _profile("rapid"), 4, DURATIONS)

    remaining = session.detach_terminal_members([1])
    assert remaining is not None
    assert remaining.member_request_ids == (2,)
    assert remaining.base_task is Task.LOCALIZATION
    assert remaining.exposed_outputs == task_outputs(Task.LOCALIZATION)

    assert remaining.detach_terminal_members([2]) is None

    with pytest.raises(EntityValidationError):
        session.detach_terminal_members([99])