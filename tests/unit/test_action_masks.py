from __future__ import annotations

from dataclasses import replace
from itertools import permutations
from types import MappingProxyType

import pytest

from isac_ssc.core.entities import DiskAOI, RequestState, SensingRequest, SensingSession, Task
from isac_ssc.envs.action_masks import build_action_masks, build_current_feasibility
from isac_ssc.envs.action_space import ActionType, EnvironmentAction
from isac_ssc.envs.dynamics import TargetSlotPrimitive
from isac_ssc.utils.config import load_config

CONFIG = load_config()
DURATIONS = CONFIG.service_duration_slots
AOI = DiskAOI((80.0, 0.0), 30.0)
PRIOR = tuple(tuple(float(CONFIG.sensing["tracking"]["initial_covariance_diag"][row])
                    if row == column else 0.0 for column in range(4)) for row in range(4))


def _request(
    request_id, *, task=Task.DETECTION, tenant="tenant_1", target=7, permission=True,
    state=RequestState.WAITING, interval=2, threshold=None, latest=8, value=1.0, aoi=AOI,
):
    threshold = (0.1 if task is Task.DETECTION else 100.0) if threshold is None else threshold
    request = SensingRequest(
        request_id, tenant, 1 if state is RequestState.WAITING else 0, latest, aoi, target,
        task, threshold, interval, value, permission,
    )
    return request if state is RequestState.WAITING else request.transition(state, slot=0)


def _session(session_id=1, *, creator=None, profile="balanced"):
    creator = creator or _request("creator", task=Task.TRACKING, state=RequestState.ACTIVE)
    session = SensingSession.create(
        session_id, creator, CONFIG.resource_profiles[profile], 0, DURATIONS,
        PRIOR if creator.task is Task.TRACKING else None,
    )
    return session, creator


def _state(*, focal=None, session=None, creator=None, requests=(), sessions=(), primitives=None,
           config=CONFIG, prospective=99):
    if session is None:
        session, creator = _session()
    focal = focal or _request("focal")
    all_requests = (creator, focal, *requests)
    all_sessions = (session, *sessions)
    primitives = primitives or (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
    )
    feasibility = build_current_feasibility(1, all_requests, all_sessions, primitives, prospective, config)
    masks = build_action_masks(focal, all_requests, all_sessions, feasibility, config)
    return all_requests, all_sessions, feasibility, masks


def test_complete_action_catalogue_and_factorized_masks_agree() -> None:
    _, _, feasibility, masks = _state()
    expected = (
        len(feasibility.session_ids) * len(feasibility.profile_ids)
        + len(feasibility.profile_ids) + 2
    )
    assert len(masks.entries) == expected
    assert tuple(action for action in masks.feasible_actions) == tuple(
        entry.action for entry in masks.entries if entry.feasible
    )
    assert dict(masks.action_type_mask)[ActionType.REJECT]
    assert masks.reject_feasible


def test_same_target_merge_is_retained_and_assessment_object_is_reused() -> None:
    _, _, feasibility, masks = _state()
    action = EnvironmentAction(ActionType.MERGE, 1, "balanced")
    entry = masks.entry_for(action)
    canonical = feasibility.request_for("focal").merge_for(1, "balanced")
    assert entry.feasible and entry.merge_assessment is canonical


def test_different_target_and_same_class_different_target_are_masked() -> None:
    focal = _request("focal", target=8)
    primitives = (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
        TargetSlotPrimitive(1, 8, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
    )
    _, _, _, masks = _state(focal=focal, primitives=primitives)
    assert not any(entry.feasible for entry in masks.entries if entry.action.action_type is ActionType.MERGE)


def test_focal_or_existing_member_without_sharing_permission_masks_every_merge() -> None:
    _, _, _, focal_masks = _state(focal=_request("focal", permission=False))
    assert not any(item.feasible for item in focal_masks.entries if item.action.action_type is ActionType.MERGE)
    creator = _request("creator", task=Task.TRACKING, state=RequestState.ACTIVE, permission=False)
    session, creator = _session(creator=creator)
    _, _, _, member_masks = _state(session=session, creator=creator)
    assert not any(item.feasible for item in member_masks.entries if item.action.action_type is ActionType.MERGE)


def test_cross_tenant_pair_authorization_masks_merge() -> None:
    creator = _request("creator", task=Task.TRACKING, tenant="tenant_1", state=RequestState.ACTIVE)
    session, creator = _session(creator=creator)
    _, _, _, masks = _state(
        session=session, creator=creator, focal=_request("focal", tenant="tenant_4"),
    )
    assert not any(item.feasible for item in masks.entries if item.action.action_type is ActionType.MERGE)


def test_private_request_can_create_even_when_every_merge_is_masked() -> None:
    _, _, _, masks = _state(focal=_request("focal", permission=False))
    assert any(item.feasible for item in masks.entries if item.action.action_type is ActionType.CREATE)


def test_task_capability_aoi_and_current_quality_are_hard_mask_predicates() -> None:
    creator = _request("creator", task=Task.DETECTION, state=RequestState.ACTIVE)
    session, creator = _session(creator=creator)
    localization = _request("focal", task=Task.LOCALIZATION)
    _, _, _, task_masks = _state(session=session, creator=creator, focal=localization)
    assert not any(item.feasible for item in task_masks.entries if item.action.action_type is ActionType.MERGE)

    shifted = _request("focal", aoi=DiskAOI((120.0, 0.0), 15.0))
    _, _, _, aoi_masks = _state(focal=shifted)
    assert not any(item.feasible for item in aoi_masks.entries if item.action.action_type is ActionType.MERGE)

    weak = (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 0.0, 0.0, -100.0),
    )
    _, _, _, quality_masks = _state(primitives=weak)
    assert not any(
        item.feasible for item in quality_masks.entries
        if item.action.action_type in {ActionType.MERGE, ActionType.CREATE}
    )


def test_update_period_mask_is_profile_specific_and_expensive_feasible_actions_remain() -> None:
    _, _, _, masks = _state(focal=_request("focal", interval=1))
    create = dict(masks.create_profile_mask)
    assert not create["economical"] and create["rapid"]
    assert create["precision"]


def test_future_reservation_and_existing_member_preservation_are_enforced() -> None:
    first, creator = _session(profile="precision")
    second_creator = _request("second", task=Task.TRACKING, state=RequestState.ACTIVE)
    second, second_creator = _session(2, creator=second_creator, profile="precision")
    third_creator = _request("third", task=Task.TRACKING, state=RequestState.ACTIVE)
    third, third_creator = _session(3, creator=third_creator, profile="precision")
    second, third = replace(second, next_update_slot=1), replace(third, next_update_slot=1)

    requests = (creator, second_creator, third_creator, _request("focal"))
    sessions = (first, second, third)
    primitive = (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
    )
    feasibility = build_current_feasibility(1, requests, sessions, primitive, 99, CONFIG)
    masks = build_action_masks(requests[-1], requests, sessions, feasibility, CONFIG)
    assert not masks.entry_for(EnvironmentAction(ActionType.MERGE, 1, "precision")).feasible


def test_defer_uses_exact_cooldown_boundary_and_reject_is_always_valid() -> None:
    _, _, _, at_boundary = _state(focal=_request("focal", latest=2))
    assert at_boundary.defer_feasible and at_boundary.reject_feasible
    _, _, _, beyond = _state(focal=_request("focal", latest=1))
    assert not beyond.defer_feasible and beyond.reject_feasible


def test_request_value_and_reward_weights_do_not_change_masks() -> None:
    requests, sessions, _, original = _state()
    focal = next(item for item in requests if item.request_id == "focal")
    changed_value = replace(focal, completion_value=9.0)
    changed_requests = tuple(
        changed_value if item.request_id == "focal" else item for item in requests
    )
    primitive = (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
    )
    changed_feasibility = build_current_feasibility(
        1, changed_requests, sessions, primitive, 99, CONFIG,
    )
    value_masks = build_action_masks(
        changed_value, changed_requests, sessions, changed_feasibility, CONFIG,
    )

    reward = dict(CONFIG.reward)
    reward["sensing_resource_cost_weight"] = 999.0
    reward_config = replace(CONFIG, reward=MappingProxyType(reward))

    reward_feasibility = build_current_feasibility(
        1, requests, sessions, primitive, 99, reward_config,
    )
    reward_masks = build_action_masks(
        focal, requests, sessions, reward_feasibility, reward_config,
    )

    expected = tuple((item.action, item.feasible) for item in original.entries)
    assert tuple((item.action, item.feasible) for item in value_masks.entries) == expected
    assert tuple((item.action, item.feasible) for item in reward_masks.entries) == expected


def test_input_permutation_and_typed_session_ids_are_deterministic() -> None:
    first, first_creator = _session(1)
    second_creator = _request(
        "creator_string", task=Task.TRACKING, state=RequestState.ACTIVE,
    )
    second, second_creator = _session("1", creator=second_creator)
    focal = _request("focal")
    requests = (first_creator, second_creator, focal)
    sessions = (first, second)
    primitive = (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
    )

    outputs = set()
    for request_order in permutations(requests):
        for session_order in permutations(sessions):
            feasibility = build_current_feasibility(
                1, request_order, session_order, primitive, 99, CONFIG,
            )
            masks = build_action_masks(
                focal, request_order, session_order, feasibility, CONFIG,
            )
            outputs.add(tuple((item.action, item.feasible) for item in masks.entries))

    assert len(outputs) == 1
    merge_ids = {
        item.action.session_id for item in masks.entries
        if item.action.action_type is ActionType.MERGE
    }
    assert merge_ids == {1, "1"}


def test_no_active_session_state_keeps_all_create_actions_and_no_merge_actions() -> None:
    focal = _request("focal")
    primitive = (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
    )
    feasibility = build_current_feasibility(
        1, (focal,), (), primitive, 99, CONFIG,
    )
    masks = build_action_masks(focal, (focal,), (), feasibility, CONFIG)
    assert not any(
        item.action.action_type is ActionType.MERGE for item in masks.entries
    )
    assert len(tuple(
        item for item in masks.entries
        if item.action.action_type is ActionType.CREATE
    )) == 4
    assert masks.reject_feasible


def test_mask_construction_never_calls_tracking_prediction(monkeypatch) -> None:
    import isac_ssc.core.quality as quality_module

    def forbidden(*args, **kwargs):
        raise AssertionError("mask construction must use the already-defined current prior")

    monkeypatch.setattr(quality_module, "predict_tracking_covariance", forbidden)
    _state()
