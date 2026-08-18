from __future__ import annotations

from dataclasses import replace
from inspect import signature
from itertools import permutations
from math import isfinite
from pathlib import Path

import pytest

from isac_ssc.core.entities import DiskAOI, RequestState, SensingRequest, SensingSession, Task
from isac_ssc.envs.action_masks import build_action_masks, build_current_feasibility
from isac_ssc.envs.dynamics import CommunicationSlotPrimitive, TargetSlotPrimitive
from isac_ssc.envs.observation import (
    CommunicationAccountingState, FeatureSpec,
    TenantAccountingState, build_observation,
)
from isac_ssc.utils.config import load_config

CONFIG = load_config()
DURATIONS = CONFIG.service_duration_slots
AOI = DiskAOI((80.0, 0.0), 30.0)
PRIOR = tuple(tuple(float(CONFIG.sensing["tracking"]["initial_covariance_diag"][row])
                    if row == column else 0.0 for column in range(4)) for row in range(4))


def _request(
    request_id, *, task=Task.DETECTION, tenant="tenant_1", target=7,
    state=RequestState.WAITING, threshold=None, value=1.0, permission=True, aoi=AOI,
):
    threshold = (0.1 if task is Task.DETECTION else 100.0) if threshold is None else threshold
    request = SensingRequest(
        request_id, tenant, 1 if state is RequestState.WAITING else 0, 8, aoi,
        target, task, threshold, 2, value, permission,
    )
    return request if state is RequestState.WAITING else request.transition(state, slot=0)


def _session(session_id=1, *, task=Task.TRACKING, target=7):
    creator = _request(
        f"creator_{type(session_id).__name__}_{session_id}",
        task=task, target=target, state=RequestState.ACTIVE,
    )
    session = SensingSession.create(
        session_id, creator, CONFIG.resource_profiles["balanced"], 0, DURATIONS,
        PRIOR if task is Task.TRACKING else None,
    )
    return session, creator


def _inputs(*, target_states=None, requests=None, sessions=None, focal=None):
    session, creator = _session()
    focal = _request("focal") if focal is None else focal
    extra = _request("other", tenant="tenant_2", target=8)
    requests = (creator, focal, extra) if requests is None else requests
    sessions = (session,) if sessions is None else sessions
    target_states = (
        TargetSlotPrimitive(
            1, 7, (80.0, 0.0), (1.0, -1.0), 1.0, 0.5, -0.25, 0.0,
        ),
        TargetSlotPrimitive(
            1, 8, (100.0, 0.0), (0.0, 0.0), -1.0, 0.25, 0.5, 1.0,
        ),
    ) if target_states is None else target_states
    communication = tuple(CommunicationSlotPrimitive(
        1, f"user_{index}", (40.0+index, 5.0), (0.0, 0.0),
        index % 2 == 0, 5.0e6 if index % 2 == 0 else 0.0,
        0.0, 1.0, 0.0,
    ) for index in range(1, 7))
    feasibility = build_current_feasibility(
        1, requests, sessions, target_states, 99, CONFIG,
    )
    masks = build_action_masks(focal, requests, sessions, feasibility, CONFIG)
    tenant = tuple(TenantAccountingState(
        item.tenant_id, index, index // 2, index // 3,
        index // 2-item.sla_violation_budget*index,
    ) for index, item in enumerate(CONFIG.tenants, start=1))
    communication_accounting = tuple(CommunicationAccountingState(
        item.user_id, index, 0.1*index, 0.05*index,
    ) for index, item in enumerate(communication, start=1))
    return {
        "current_slot": 1,
        "focal_request": focal,
        "requests": requests,
        "sessions": sessions,
        "target_primitives": target_states,
        "communication_primitives": communication,
        "feasibility": feasibility,
        "action_masks": masks,
        "tenant_accounting": tenant,
        "communication_accounting": communication_accounting,
        "cumulative_completed_value": 3.5,
        "cumulative_sensing_cost": 0.75,
        "config": CONFIG,
    }


def _observation(**overrides):
    values = _inputs()
    values.update(overrides)
    return build_observation(**values)


def _column(table, name: str, row: int = 0) -> float:
    index = tuple(item.name for item in table.specs).index(name)
    return table.rows[row][index]


def _column_global(view, name: str) -> float:
    index = tuple(item.name for item in view.global_specs).index(name)
    return view.global_features[index]


def test_feature_specs_are_explicit_immutable_and_match_every_row() -> None:
    observation = _observation()
    for table in (observation.set_view.request_table, observation.set_view.session_table):
        assert all(isinstance(item, FeatureSpec) for item in table.specs)
        assert all(len(row) == len(table.specs) for row in table.rows)
    assert len(observation.set_view.global_specs) == len(observation.set_view.global_features)
    with pytest.raises(Exception):
        observation.set_view.request_table.specs[0].name = "changed"


def test_all_observation_features_are_finite_and_ids_are_not_numeric_columns() -> None:
    observation = _observation()
    tables = (
        observation.set_view.request_table.rows, observation.set_view.session_table.rows,
        (observation.set_view.global_features,),
    )
    assert all(isfinite(value) for rows in tables for row in rows for value in row)
    names = tuple(item.name for item in (
        *observation.set_view.request_table.specs,
        *observation.set_view.session_table.specs,
        *observation.set_view.global_specs,
    ))
    assert not any(name in {"request_id", "session_id", "tenant_id", "target_id"} for name in names)


def test_relational_keys_preserve_typed_identity_and_target_equality() -> None:
    session_int, creator_int = _session(1)
    session_str, creator_str = _session("1")
    focal_int = _request(1)
    focal_str = _request("1", tenant="tenant_2")
    requests = (creator_int, creator_str, focal_int, focal_str)
    sessions = (session_int, session_str)
    targets = (
        TargetSlotPrimitive(
            1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0,
        ),
    )
    base = _inputs(
        requests=requests, sessions=sessions, focal=focal_int,
        target_states=targets,
    )
    observation = build_observation(**base)
    request_ids = tuple(
        item.request_id for item in observation.set_view.request_table.keys
    )
    session_ids = tuple(
        item.session_id for item in observation.set_view.session_table.keys
    )
    assert request_ids == (1, "1")
    assert session_ids == (1, "1")
    assert all(
        item.target_id == 7
        for item in (
            *observation.set_view.request_table.keys,
            *observation.set_view.session_table.keys,
        )
    )


def test_request_and_session_nodes_are_canonically_ordered_under_input_permutation() -> None:
    values = _inputs()
    outputs = set()
    for request_order in permutations(values["requests"]):
        for session_order in permutations(values["sessions"]):
            feasibility = build_current_feasibility(
                1, request_order, session_order,
                values["target_primitives"], 99, CONFIG,
            )
            masks = build_action_masks(
                values["focal_request"], request_order, session_order,
                feasibility, CONFIG,
            )
            observation = build_observation(
                1, values["focal_request"], request_order, session_order,
                values["target_primitives"],
                reversed(values["communication_primitives"]),
                feasibility, masks,
                reversed(values["tenant_accounting"]),
                reversed(values["communication_accounting"]),
                3.5, 0.75, CONFIG,
            )
            outputs.add(observation)
    assert len(outputs) == 1


def test_request_create_features_are_exactly_backed_by_canonical_assessments() -> None:
    observation = _observation()
    table = observation.set_view.request_table
    focal_row = next(
        index for index, key in enumerate(table.keys) if key.request_id == "focal"
    )
    values = _inputs()
    assessments = values["feasibility"].request_for("focal")
    for profile_id in values["feasibility"].profile_ids:
        assessment = assessments.create_for(profile_id)
        assert _column(
            table, f"create_{profile_id}_feasible", focal_row,
        ) == float(assessment.feasible)
        expected = (
            0.0 if assessment.quality_margin.margin is None
            else assessment.quality_margin.margin
            / values["focal_request"].quality_threshold
        )
        assert _column(
            table, f"create_{profile_id}_quality_margin_normalized", focal_row,
        ) == pytest.approx(expected)


def test_observation_snapshot_contains_only_the_single_set_view() -> None:
    observation = _observation()
    assert tuple(observation.__dataclass_fields__) == ("set_view",)
    assert tuple(observation.set_view.__dataclass_fields__) == (
        "request_table", "session_table", "global_specs", "global_features", "action_masks",
    )
    assert not hasattr(observation, "graph_view")
    assert not hasattr(observation.set_view, "edges")


def test_request_value_is_observable_but_does_not_change_masks() -> None:
    values = _inputs()
    original = build_observation(**values)
    focal = replace(values["focal_request"], completion_value=1.1)
    requests = tuple(
        focal if item.request_id == "focal" else item
        for item in values["requests"]
    )
    feasibility = build_current_feasibility(
        1, requests, values["sessions"],
        values["target_primitives"], 99, CONFIG,
    )
    masks = build_action_masks(
        focal, requests, values["sessions"], feasibility, CONFIG,
    )
    changed = build_observation(
        1, focal, requests, values["sessions"],
        values["target_primitives"], values["communication_primitives"],
        feasibility, masks, values["tenant_accounting"],
        values["communication_accounting"], 3.5, 0.75, CONFIG,
    )
    assert tuple(
        (item.action, item.feasible)
        for item in original.set_view.action_masks.entries
    ) == tuple(
        (item.action, item.feasible)
        for item in changed.set_view.action_masks.entries
    )
    assert _column(
        original.set_view.request_table, "completion_value_normalized",
    ) != _column(
        changed.set_view.request_table, "completion_value_normalized",
    )


def test_current_primitive_state_changes_current_features_without_future_trace_input() -> None:
    values = _inputs()
    original = build_observation(**values)
    targets = tuple(
        replace(
            item,
            position_m=(item.position_m[0]+1.0, item.position_m[1]),
        ) if item.target_id == 7 else item
        for item in values["target_primitives"]
    )
    feasibility = build_current_feasibility(
        1, values["requests"], values["sessions"], targets, 99, CONFIG,
    )
    masks = build_action_masks(
        values["focal_request"], values["requests"],
        values["sessions"], feasibility, CONFIG,
    )
    changed = build_observation(
        1, values["focal_request"], values["requests"],
        values["sessions"], targets, values["communication_primitives"],
        feasibility, masks, values["tenant_accounting"],
        values["communication_accounting"], 3.5, 0.75, CONFIG,
    )
    assert original.set_view.request_table != changed.set_view.request_table
    parameters = signature(build_observation).parameters
    assert "trace" not in parameters
    assert "pending_children" not in parameters
    assert "future_primitives" not in parameters


def test_running_normalization_is_not_part_of_observation_features() -> None:
    observation = _observation()
    names = " ".join(
        item.name+" "+item.normalization for item in (
            *observation.set_view.request_table.specs,
            *observation.set_view.session_table.specs,
            *observation.set_view.global_specs,
        )
    ).lower()
    assert "running" not in names
    source = (Path(__file__).parents[2] / "src/isac_ssc/envs/observation.py").read_text(encoding="utf-8")
    assert "PrimitiveTrace" not in source and "pending_children" not in source
    assert "graph_view" not in source and "EdgeTable" not in source and "_edge_table" not in source


def test_pre_action_communication_summary_uses_current_demand_and_canonical_scheduler() -> None:
    observation = _observation()
    global_view = observation.set_view
    assert _column_global(global_view, "active_demand_user_count") == 3.0
    assert 0.0 <= _column_global(
        global_view, "pre_action_mean_shortfall",
    ) <= 1.0
    assert 0.0 <= _column_global(
        global_view, "pre_action_max_shortfall",
    ) <= 1.0


def test_missing_localization_and_tracking_values_use_flags_and_finite_zero() -> None:
    session, creator = _session(task=Task.DETECTION)
    focal = _request("focal")
    other = _request("other", tenant="tenant_2", target=8)
    weak = (
        TargetSlotPrimitive(
            1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 0.0, 0.0, -100.0,
        ),
        TargetSlotPrimitive(
            1, 8, (100.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0,
        ),
    )
    values = _inputs(
        requests=(creator, focal, other), sessions=(session,),
        focal=focal, target_states=weak,
    )
    observation = build_observation(**values)
    table = observation.set_view.session_table
    assert _column(table, "shared_localization_defined") == 0.0
    assert _column(table, "shared_peb_normalized") == 0.0
    assert _column(table, "shared_tracking_capable") == 0.0
    assert _column(table, "shared_pcrb_normalized") == 0.0


def test_repeated_build_is_exactly_deterministic() -> None:
    values = _inputs()
    assert build_observation(**values) == build_observation(**values)


def test_feature_schema_contains_all_locked_descriptive_groups() -> None:
    observation = _observation()
    request_names = {item.name for item in observation.set_view.request_table.specs}
    session_names = {item.name for item in observation.set_view.session_table.specs}
    global_names = {item.name for item in observation.set_view.global_specs}
    assert {
        "task_detection", "deadline_slack_normalized",
        "tenant_additive_residual", "create_balanced_feasible",
    }.issubset(request_names)
    assert {
        "member_count", "shared_detection_probability",
        "shared_pcrb_normalized", "future_scheduled_update_count_normalized",
    }.issubset(session_names)
    assert {
        "current_sensing_bandwidth_fraction", "pre_action_mean_shortfall",
        "tenant_residual_sum", "communication_residual_sum",
    }.issubset(global_names)


def test_empty_session_state_builds_empty_session_table_and_no_merge_actions() -> None:
    focal = _request("focal")
    other = _request("other", tenant="tenant_2", target=8)
    requests = (focal, other)
    targets = (
        TargetSlotPrimitive(1, 7, (80.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
        TargetSlotPrimitive(1, 8, (100.0, 0.0), (0.0, 0.0), 0.0, 1.0, 0.0, 0.0),
    )
    values = _inputs(requests=requests, sessions=(), focal=focal, target_states=targets)
    observation = build_observation(**values)
    assert observation.set_view.session_table.rows == ()
    assert not any(
        item.action.action_type.value == "merge"
        for item in observation.set_view.action_masks.entries
    )
