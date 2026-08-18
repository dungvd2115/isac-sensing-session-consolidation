from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from math import log10, sqrt
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from isac_ssc.core.entities import (
    CommunicationUser, DiskAOI,
    SensingRequest, Task,
)
from isac_ssc.core.quality import (
    CommunicationParameters,
    evaluate_communication_quality,
)
from isac_ssc.core.resources import (
    SensingResourceUsage,
    equal_share_communication_resources,
    residual_communication_resources,
)
from isac_ssc.core.sla import communication_qos_slot
from isac_ssc.envs.dynamics import (
    CommunicationSlotPrimitive,
    PrimitiveTrace,
    TargetSlotPrimitive,
)
from isac_ssc.oracles.exhaustive import (
    authorization_matrix,
    build_restricted_coloring_instance,
    minimum_authorized_session_partition,
    solve_exhaustive_reference,
)
from isac_ssc.oracles.milp import (
    MilpSolveError,
    build_offline_reference_milp,
    solve_offline_reference_milp,
)
from isac_ssc.oracles.reference import (
    OracleInstance,
    ReferenceValidationError,
    enumerate_single_session_plans,
    evaluate_joint_selection,
)
from isac_ssc.utils.config import load_config

CONFIG = load_config()
AOI = DiskAOI((80.0, 0.0), 30.0)
TARGET_ID = 7


def _request(
    request_id: int | str,
    arrival_slot: int,
    latest_start_slot: int,
    *,
    task: Task = Task.DETECTION,
    interval: int = 2,
    threshold: float | None = None,
    completion_value: float = 1.0,
) -> SensingRequest:
    default_threshold = (
        0.1
        if task is Task.DETECTION
        else 1.0e6
    )
    return SensingRequest(
        request_id, "tenant_1",
        arrival_slot, latest_start_slot,
        AOI, TARGET_ID, task,
        default_threshold if threshold is None else threshold,
        interval, completion_value, True,
    )


def _target_primitives(
    horizon_slots: int,
    target_ids: tuple[int | str, ...] = (TARGET_ID,),
    *,
    fading_gain_by_slot: dict[int, float] | None = None,
) -> tuple[TargetSlotPrimitive, ...]:
    gains = (
        {}
        if fading_gain_by_slot is None
        else fading_gain_by_slot
    )
    return tuple(
        TargetSlotPrimitive(
            slot, target_id,
            (80.0, 0.0), (0.0, 0.0),
            0.0,
            sqrt(gains.get(slot, 1.0)),
            0.0,
            10.0*log10(1.0),
        )
        for slot in range(horizon_slots)
        for target_id in target_ids
    )


def _communication_primitives(
    horizon_slots: int,
    users: tuple[CommunicationUser, ...],
) -> tuple[CommunicationSlotPrimitive, ...]:
    return tuple(
        CommunicationSlotPrimitive(
            slot, user.user_id,
            user.position_m,
            user.velocity_m_per_s,
            user.demand_bit_per_s > 0.0,
            user.demand_bit_per_s,
            0.0, 1.0, 0.0,
        )
        for slot in range(horizon_slots)
        for user in users
    )


def _primitive_trace(
    trace_id: str,
    horizon_slots: int,
    targets: tuple[TargetSlotPrimitive, ...],
    communications: tuple[CommunicationSlotPrimitive, ...] = (),
) -> PrimitiveTrace:
    return PrimitiveTrace(
        trace_id, 0, "independent", horizon_slots,
        tuple(
            tenant.tenant_id
            for tenant in CONFIG.tenants
        ),
        tuple(
            tenant.authorization_row
            for tenant in CONFIG.tenants
        ),
        targets, (), communications, (), (), (),
    )


def _trace(
    trace_id: str,
    horizon_slots: int,
    requests: tuple[SensingRequest, ...],
    profile_name: str,
    *,
    with_communication: bool = False,
) -> OracleInstance:
    users = (
        CommunicationUser(
            "user_1",
            (50.0, 0.0), (0.0, 0.0),
            1.0e6, 2.0e6, 0.05,
        ),
    ) if with_communication else ()

    primitive_trace = _primitive_trace(
        trace_id,
        horizon_slots,
        _target_primitives(horizon_slots),
        _communication_primitives(
            horizon_slots, users,
        ),
    )
    return OracleInstance(
        primitive_trace,
        requests,
        (CONFIG.resource_profiles[profile_name],),
        users,
    )


def _edges_complete(
    vertex_count: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        combinations(range(vertex_count), 2),
    )


def _edges_path(
    vertex_count: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (vertex, vertex+1)
        for vertex in range(vertex_count-1)
    )


def _edges_cycle(
    vertex_count: int,
) -> tuple[tuple[int, int], ...]:
    return _edges_path(vertex_count) + (
        (0, vertex_count-1),
    )


@pytest.mark.parametrize(
    ("vertex_count", "edges", "expected"),
    (
        (4, (), 1),
        (4, _edges_complete(4), 4),
        (5, _edges_path(5), 2),
        (4, _edges_cycle(4), 2),
        (5, _edges_cycle(5), 3),
    ),
)
def test_restricted_graph_coloring_construction_matches_known_chromatic_numbers(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
    expected: int,
) -> None:
    instance = build_restricted_coloring_instance(
        vertex_count, edges,
    )
    matrix = authorization_matrix(instance)
    normalized_edges = {
        tuple(sorted(edge))
        for edge in edges
    }
    assert all(
        matrix[left][right] == (
            left == right
            or tuple(sorted(
                (left, right),
            )) not in normalized_edges
        )
        for left in range(vertex_count)
        for right in range(vertex_count)
    )

    result = minimum_authorized_session_partition(instance)
    assert result.minimum_session_count == expected
    assert all(
        len(partition) == expected
        for partition in result.partitions
    )


def test_single_session_plan_enumeration_is_complete_and_event_constrained() -> None:
    trace = _trace(
        "two-request-reference",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
        with_communication=True,
    )
    plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    event_keys = {
        tuple(
            (
                event.slot,
                event.request_id,
                event.profile_id,
            )
            for event in plan.admissions
        )
        for plan in plans
    }
    assert event_keys == {
        ((0, 1, "balanced"),),
        (
            (0, 1, "balanced"),
            (1, 2, "balanced"),
        ),
        ((1, 2, "balanced"),),
    }

    shared = next(
        plan
        for plan in plans
        if len(plan.member_request_ids) == 2
    )
    assert shared.creator_request_id == 1
    assert shared.update_slots == (0, 1, 3)
    assert tuple(
        event.slot
        for event in shared.admissions
    ) == (0, 1)
    assert len(shared.shared_outputs) == len(shared.update_slots)
    assert shared.shared_outputs[
        1
    ].member_request_ids == (1, 2)
    assert len(
        shared.shared_outputs[1].request_validity,
    ) == 2
    assert all(
        outcome.completed
        and not outcome.sla_violated
        for outcome in shared.request_outcomes
    )
    assert shared.slot_accounting[
        2
    ].sensing_bandwidth_hz == 0.0
    assert shared.slot_accounting[
        3
    ].sensing_bandwidth_hz == (
        CONFIG.resource_profiles[
            "balanced"
        ].sensing_bandwidth_hz
    )


def test_future_bad_shared_output_creates_one_request_level_violation_without_plan_rejection() -> None:
    request = _request(
        1, 0, 0, interval=1,
    )
    target_primitives = _target_primitives(
        3,
        fading_gain_by_slot={2: 0.0},
    )
    trace = OracleInstance(
        _primitive_trace(
            "future-quality-reference",
            3,
            target_primitives,
        ),
        (request,),
        (
            CONFIG.resource_profiles["balanced"],
        ),
    )
    plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    assert len(plans) == 1
    outcome = plans[0].request_outcomes[0]
    assert outcome.state.value == "failed"
    assert outcome.valid_output_count == 1
    assert outcome.sla_violated
    assert outcome.first_violation_slot == 2
    assert plans[0].shared_outputs[
        -1
    ].request_validity == ((1, False),)


def test_profiles_change_only_at_creation_or_member_admission_events() -> None:
    base = _trace(
        "profile-event-reference",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
    )
    trace = replace(
        base,
        available_profiles=(
            CONFIG.resource_profiles["economical"],
            CONFIG.resource_profiles["balanced"],
        ),
    )
    plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    assert len(plans) == 8

    for plan in plans:
        events = {
            event.slot: event.profile_id
            for event in plan.admissions
        }
        active_profile = None
        outputs = {
            output.slot: output
            for output in plan.shared_outputs
        }
        for slot in range(trace.horizon_slots):
            if slot in events:
                active_profile = events[slot]
            if slot in outputs:
                assert outputs[
                    slot
                ].profile_id == active_profile


def test_tracking_plan_uses_one_prediction_per_post_creation_slot_and_only_scheduled_updates() -> None:
    trace = _trace(
        "tracking-reference",
        8,
        (
            _request(
                1, 0, 0,
                task=Task.TRACKING,
                interval=2,
            ),
        ),
        "economical",
    )
    plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    assert len(plans) == 1
    plan = plans[0]
    assert plan.update_slots == (0, 3, 6)
    assert tuple(
        record.slot
        for record in plan.tracking_slots
        if record.predicted
    ) == tuple(range(1, 8))
    assert tuple(
        record.slot
        for record in plan.tracking_slots
        if record.measurement_updated
    ) == (0, 3, 6)

    for record in plan.tracking_slots:
        if not record.measurement_updated:
            assert (
                record.posterior_covariance
                == record.prior_covariance
            )
        assert record.pcrb_m > 0.0


def test_joint_selection_enforces_one_admission_per_slot() -> None:
    trace = _trace(
        "joint-hard-constraints",
        3,
        (
            _request(1, 0, 0),
            _request(2, 0, 0),
        ),
        "balanced",
    )
    plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    singleton_one = next(
        plan
        for plan in plans
        if plan.member_request_ids == (1,)
    )
    singleton_two = next(
        plan
        for plan in plans
        if plan.member_request_ids == (2,)
    )
    assert evaluate_joint_selection(
        (singleton_one, singleton_two),
        trace, CONFIG,
    ) is None
    assert all(
        len(plan.member_request_ids) == 1
        for plan in plans
    )


def test_joint_selection_rejects_duplicate_request_assignment() -> None:
    trace = _trace(
        "joint-request-exclusivity",
        4,
        (
            _request(1, 0, 1),
            _request(2, 1, 1),
        ),
        "balanced",
    )
    plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    singleton = next(
        plan
        for plan in plans
        if (
            plan.member_request_ids == (1,)
            and plan.creation_slot == 0
        )
    )
    shared = next(
        plan
        for plan in plans
        if plan.member_request_ids == (1, 2)
    )
    assert evaluate_joint_selection(
        (singleton, shared),
        trace, CONFIG,
    ) is None


def test_exhaustive_selection_prefers_shared_plan_and_reconstructs_all_accounting() -> None:
    trace = _trace(
        "joint-reference",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
        with_communication=True,
    )
    plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    result = solve_exhaustive_reference(
        plans, trace, CONFIG,
    )
    assert result.plan_count == 3
    assert result.feasible_selection_count == 5
    assert len(result.optimal_selections) == 1

    optimum = result.optimal_selections[0]
    assert len(optimum.selected_plan_ids) == 1
    selected = next(
        plan
        for plan in plans
        if plan.plan_id == optimum.selected_plan_ids[0]
    )
    assert selected.member_request_ids == (1, 2)
    assert optimum.request_assignments == (
        (1, selected.plan_id),
        (2, selected.plan_id),
    )
    assert tuple(
        output.slot
        for output in optimum.shared_outputs
    ) == (0, 1, 3)
    assert tuple(
        (
            slot.sensing_bandwidth_hz,
            slot.sensing_power_w,
        )
        for slot in optimum.slot_accounting
    ) == (
        (4.0e6, 5.0),
        (4.0e6, 5.0),
        (0.0, 0.0),
        (4.0e6, 5.0),
    )
    assert optimum.tenant_residuals[
        0
    ] == (
        "tenant_1",
        pytest.approx(-0.10),
    )
    assert all(
        residual <= 0.0
        for _, residual in optimum.communication_residuals
    )
    assert optimum.objective == pytest.approx(
        selected.objective,
    )


def test_communication_accounting_matches_direct_canonical_core_evaluation() -> None:
    trace = _trace(
        "communication-reference",
        4,
        (_request(1, 0, 0),),
        "balanced",
        with_communication=True,
    )
    plan = enumerate_single_session_plans(
        trace, CONFIG,
    )[0]
    accounting = evaluate_joint_selection(
        (plan,), trace, CONFIG,
    )
    assert accounting is not None

    slot = accounting.slot_accounting[0]
    primitive, user = trace.communication_at(
        0, CONFIG,
    )[0]
    residual = residual_communication_resources(
        CONFIG.system["total_bandwidth_hz"],
        CONFIG.system["total_power_w"],
        SensingResourceUsage(
            slot.sensing_bandwidth_hz,
            slot.sensing_power_w,
        ),
    )
    allocation = equal_share_communication_resources(
        (user,), residual,
    )[0]
    quality = evaluate_communication_quality(
        user.position_m,
        CONFIG.geometry["bs_position_m"],
        allocation.bandwidth_hz,
        allocation.power_w,
        user.demand_bit_per_s,
        primitive.shadowing_db,
        primitive.fading_power_gain,
        CommunicationParameters.from_config(CONFIG),
    )
    qos = communication_qos_slot(
        user.demand_bit_per_s,
        user.minimum_rate_bit_per_s,
        quality.served_rate_bit_per_s,
        user.normalized_shortfall_budget,
    )
    outcome = accounting.communication_outcomes[0]
    assert outcome.served_rate_bit_per_s == pytest.approx(
        quality.served_rate_bit_per_s,
    )
    assert outcome.normalized_shortfall == pytest.approx(
        qos.normalized_shortfall,
    )
    assert outcome.residual == pytest.approx(
        qos.residual,
    )


def test_oracle_selection_limits_reject_instead_of_truncating_selected_trace() -> None:
    trace = _trace(
        "over-limit",
        12,
        tuple(
            _request(request_id, 0, 0)
            for request_id in range(8)
        ),
        "balanced",
    )
    over_limit = replace(
        trace,
        primitive_trace=_primitive_trace(
            "over-limit",
            13,
            _target_primitives(13),
        ),
    )
    with pytest.raises(
        ReferenceValidationError,
        match="horizon limit",
    ):
        enumerate_single_session_plans(
            over_limit, CONFIG,
        )

    request_over_limit = replace(
        trace,
        requests=trace.requests + (
            _request(8, 0, 0),
        ),
    )
    with pytest.raises(
        ReferenceValidationError,
        match="request limit",
    ):
        enumerate_single_session_plans(
            request_over_limit, CONFIG,
        )


def _row(model, label: str) -> np.ndarray:
    return model.constraint_matrix.toarray()[
        model.row_index(label)
    ]


def test_milp_model_consumes_complete_plan_set_and_has_exact_structural_counts() -> None:
    trace = _trace(
        "milp-structure",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
        with_communication=True,
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    complete_plans = enumerate_single_session_plans(
        trace, CONFIG,
    )
    assert tuple(
        plan.plan_id for plan in model.plans
    ) == tuple(
        plan.plan_id for plan in complete_plans
    )
    assert model.plan_count == 3
    assert model.variable_count == (
        model.plan_count
        + model.aggregate_state_count
    )
    assert model.constraint_count == (
        len(trace.requests)
        + 6*trace.horizon_slots
        + len(CONFIG.tenants)
        + 1
    )
    assert np.all(model.integrality == 1)
    assert np.all(model.variable_lower_bounds == 0.0)
    assert np.all(model.variable_upper_bounds == 1.0)
    assert np.allclose(
        model.objective_coefficients[
            :model.plan_count
        ],
        [
            -plan.objective
            for plan in model.plans
        ],
    )
    assert np.all(
        model.objective_coefficients[
            model.plan_count:
        ] == 0.0
    )


def test_milp_request_admission_resource_and_tenant_rows_match_plan_coefficients() -> None:
    trace = _trace(
        "milp-rows",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )

    for request in trace.requests:
        row = _row(
            model,
            "request_exclusivity:"
            f"{type(request.request_id).__name__}:"
            f"{request.request_id}",
        )
        assert tuple(
            row[:model.plan_count]
        ) == tuple(
            float(
                request.request_id
                in plan.member_request_ids
            )
            for plan in model.plans
        )

    for slot in range(trace.horizon_slots):
        admission = _row(
            model,
            f"admission_capacity:{slot}",
        )
        bandwidth = _row(
            model,
            f"sensing_bandwidth:{slot}",
        )
        power = _row(
            model,
            f"sensing_power:{slot}",
        )
        assert tuple(
            admission[:model.plan_count]
        ) == tuple(
            float(sum(
                event.slot == slot
                for event in plan.admissions
            ))
            for plan in model.plans
        )
        assert tuple(
            bandwidth[:model.plan_count]
        ) == tuple(
            plan.slot_accounting[
                slot
            ].sensing_bandwidth_hz
            for plan in model.plans
        )
        assert tuple(
            power[:model.plan_count]
        ) == tuple(
            plan.slot_accounting[
                slot
            ].sensing_power_w
            for plan in model.plans
        )

    tenant = next(
        item
        for item in CONFIG.tenants
        if item.tenant_id == "tenant_1"
    )
    tenant_row = _row(
        model,
        "tenant_sla:str:tenant_1",
    )
    assert tuple(
        tenant_row[:model.plan_count]
    ) == tuple(
        sum(
            float(outcome.sla_violated)
            - tenant.sla_violation_budget
            for outcome in plan.request_outcomes
            if outcome.tenant_id == tenant.tenant_id
        )
        for plan in model.plans
    )


def test_milp_row_labels_distinguish_valid_mixed_type_request_ids() -> None:
    trace = _trace(
        "milp-mixed-request-ids",
        4,
        (
            _request(1, 0, 0),
            _request("1", 1, 1),
        ),
        "balanced",
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    assert "request_exclusivity:int:1" in model.row_labels
    assert "request_exclusivity:str:1" in model.row_labels
    assert len(model.row_labels) == len(set(model.row_labels))

    solution = solve_offline_reference_milp(
        trace, CONFIG,
    )
    assert solution.success
    assert solution.accounting.request_assignments


def test_milp_row_labels_distinguish_valid_mixed_type_communication_user_ids() -> None:
    horizon = 4
    users = (
        CommunicationUser(
            1, (50.0, 0.0), (0.0, 0.0),
            1.0e6, 2.0e6, 0.05,
        ),
        CommunicationUser(
            "1", (60.0, 0.0), (0.0, 0.0),
            1.0e6, 2.0e6, 0.05,
        ),
    )
    communications = _communication_primitives(
        horizon, users,
    )
    trace = OracleInstance(
        _primitive_trace(
            "milp-mixed-user-ids",
            horizon, (), communications,
        ),
        (),
        (
            CONFIG.resource_profiles["balanced"],
        ),
        users,
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    assert "communication_qos:int:1" in model.row_labels
    assert "communication_qos:str:1" in model.row_labels
    assert len(model.row_labels) == len(set(model.row_labels))

    solution = solve_offline_reference_milp(
        trace, CONFIG,
    )
    assert solution.success
    assert tuple(
        user_id
        for user_id, _ in solution.accounting.communication_residuals
    ) == (1, "1")


def test_milp_violation_coefficient_uses_one_request_level_event() -> None:
    target_primitives = _target_primitives(
        3,
        fading_gain_by_slot={2: 0.0},
    )
    trace = OracleInstance(
        _primitive_trace(
            "milp-violation-row",
            3, target_primitives,
        ),
        (
            _request(
                1, 0, 0, interval=1,
            ),
        ),
        (
            CONFIG.resource_profiles["balanced"],
        ),
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    tenant = next(
        item
        for item in CONFIG.tenants
        if item.tenant_id == "tenant_1"
    )
    assert model.plan_count == 1
    assert model.plans[
        0
    ].request_outcomes[
        0
    ].sla_violated
    assert _row(
        model,
        "tenant_sla:str:tenant_1",
    )[0] == pytest.approx(
        1.0-tenant.sla_violation_budget,
    )


def test_milp_aggregate_states_are_complete_subset_sums_and_links_are_exact() -> None:
    trace = _trace(
        "milp-states",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    total_bandwidth = Fraction(str(
        CONFIG.system["total_bandwidth_hz"],
    ))
    total_power = Fraction(str(
        CONFIG.system["total_power_w"],
    ))

    for slot, states in enumerate(model.aggregate_states):
        expected = set()
        for count in range(model.plan_count+1):
            for selection in combinations(
                model.plans, count,
            ):
                bandwidth = sum(
                    Fraction(str(
                        plan.slot_accounting[
                            slot
                        ].sensing_bandwidth_hz,
                    ))
                    for plan in selection
                )
                power = sum(
                    Fraction(str(
                        plan.slot_accounting[
                            slot
                        ].sensing_power_w,
                    ))
                    for plan in selection
                )
                if (
                    bandwidth <= total_bandwidth
                    and power <= total_power
                ):
                    expected.add((
                        float(bandwidth),
                        float(power),
                    ))

        assert tuple(
            (
                state.bandwidth_hz,
                state.power_w,
            )
            for state in states
        ) == tuple(sorted(expected))

        indices = model.state_variable_indices[slot]
        one_state = _row(
            model,
            f"aggregate_state_one:{slot}",
        )
        assert tuple(
            one_state[index]
            for index in indices
        ) == (1.0,)*len(indices)

        bandwidth_link = _row(
            model,
            f"aggregate_bandwidth_link:{slot}",
        )
        power_link = _row(
            model,
            f"aggregate_power_link:{slot}",
        )
        assert tuple(
            bandwidth_link[:model.plan_count]
        ) == tuple(
            -plan.slot_accounting[
                slot
            ].sensing_bandwidth_hz
            for plan in model.plans
        )
        assert tuple(
            power_link[:model.plan_count]
        ) == tuple(
            -plan.slot_accounting[
                slot
            ].sensing_power_w
            for plan in model.plans
        )
        assert tuple(
            bandwidth_link[index]
            for index in indices
        ) == tuple(
            state.bandwidth_hz
            for state in states
        )
        assert tuple(
            power_link[index]
            for index in indices
        ) == tuple(
            state.power_w
            for state in states
        )


def test_milp_communication_rows_use_canonical_state_residuals() -> None:
    trace = _trace(
        "milp-communication",
        4,
        (_request(1, 0, 0),),
        "balanced",
        with_communication=True,
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    row = _row(
        model,
        "communication_qos:str:user_1",
    )
    assert np.all(
        row[:model.plan_count] == 0.0,
    )

    for states, indices in zip(
        model.aggregate_states,
        model.state_variable_indices,
        strict=True,
    ):
        assert tuple(
            row[index]
            for index in indices
        ) == tuple(
            state.communication_residual("user_1")
            for state in states
        )


def test_milp_matches_exhaustive_unique_optimum_and_all_raw_accounting() -> None:
    trace = _trace(
        "milp-unique",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
        with_communication=True,
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    exhaustive = solve_exhaustive_reference(
        model.plans, trace, CONFIG,
    )
    solution = solve_offline_reference_milp(
        trace, CONFIG,
    )
    assert len(exhaustive.optimal_selections) == 1
    assert solution.success
    assert solution.status == 0
    assert solution.mip_gap <= (
        CONFIG.oracle["solver_relative_gap"]
    )
    assert solution.objective == pytest.approx(
        exhaustive.objective,
        abs=1.0e-7,
    )
    assert solution.selected_plan_ids == (
        exhaustive.optimal_selections[
            0
        ].selected_plan_ids
    )
    assert solution.accounting == (
        exhaustive.optimal_selections[0]
    )


def test_milp_solution_is_an_exhaustive_optimum_without_secondary_tie_objective() -> None:
    trace = _trace(
        "milp-multiple-optima",
        4,
        (_request(1, 0, 1),),
        "balanced",
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    exhaustive = solve_exhaustive_reference(
        model.plans, trace, CONFIG,
    )
    solution = solve_offline_reference_milp(
        trace, CONFIG,
    )
    optimal_ids = {
        item.selected_plan_ids
        for item in exhaustive.optimal_selections
    }
    assert len(optimal_ids) == 2
    assert solution.selected_plan_ids in optimal_ids
    assert solution.objective == pytest.approx(
        exhaustive.objective,
        abs=1.0e-7,
    )
    assert np.all(
        model.objective_coefficients[
            model.plan_count:
        ] == 0.0
    )


def test_milp_matches_exhaustive_when_multiple_sessions_are_selected() -> None:
    horizon = 4
    second_target = 8
    requests = (
        _request(1, 0, 0),
        SensingRequest(
            2, "tenant_1", 1, 1,
            AOI, second_target,
            Task.DETECTION,
            0.1, 2, 1.0, True,
        ),
    )
    targets = _target_primitives(
        horizon,
        (TARGET_ID, second_target),
    )
    trace = OracleInstance(
        _primitive_trace(
            "milp-multiple-sessions",
            horizon,
            targets,
        ),
        requests,
        (
            CONFIG.resource_profiles["balanced"],
        ),
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    exhaustive = solve_exhaustive_reference(
        model.plans, trace, CONFIG,
    )
    solution = solve_offline_reference_milp(
        trace, CONFIG,
    )
    assert len(solution.selected_plan_ids) == 2
    assert solution.objective == pytest.approx(
        exhaustive.objective,
        abs=1.0e-7,
    )
    assert solution.selected_plan_ids == (
        exhaustive.optimal_selections[
            0
        ].selected_plan_ids
    )
    assert solution.accounting == (
        exhaustive.optimal_selections[0]
    )


def test_milp_binding_communication_constraint_matches_exhaustive_selection() -> None:
    horizon = 4
    second_target = 8
    requests = (
        _request(1, 0, 0),
        SensingRequest(
            2, "tenant_1", 1, 1,
            AOI, second_target,
            Task.DETECTION,
            0.1, 2, 1.0, True,
        ),
    )
    targets = _target_primitives(
        horizon,
        (TARGET_ID, second_target),
    )
    users = (
        CommunicationUser(
            "limited_user",
            (180.0, 0.0), (0.0, 0.0),
            2.0e8, 2.0e8, 0.35,
        ),
    )
    communications = _communication_primitives(
        horizon, users,
    )
    trace = OracleInstance(
        _primitive_trace(
            "milp-binding-communication",
            horizon,
            targets,
            communications,
        ),
        requests,
        (
            CONFIG.resource_profiles["balanced"],
        ),
        users,
    )
    model = build_offline_reference_milp(
        trace, CONFIG,
    )
    exhaustive = solve_exhaustive_reference(
        model.plans, trace, CONFIG,
    )
    solution = solve_offline_reference_milp(
        trace, CONFIG,
    )
    assert len(model.plans) == 2
    assert len(solution.selected_plan_ids) == 1
    assert solution.selected_plan_ids in {
        item.selected_plan_ids
        for item in exhaustive.optimal_selections
    }
    assert solution.objective == pytest.approx(
        exhaustive.objective,
        abs=1.0e-7,
    )
    assert solution.accounting.communication_residuals[
        0
    ][1] <= 0.0
    both = evaluate_joint_selection(
        model.plans, trace, CONFIG,
    )
    assert both is None


def test_milp_infeasibility_is_reported_without_heuristic_fallback() -> None:
    horizon = 2
    users = (
        CommunicationUser(
            "blocked_user",
            (1.0e9, 0.0), (0.0, 0.0),
            1.0e9, 1.0e9, 0.0,
        ),
    )
    communications = _communication_primitives(
        horizon, users,
    )
    trace = OracleInstance(
        _primitive_trace(
            "milp-infeasible",
            horizon, (), communications,
        ),
        (),
        (
            CONFIG.resource_profiles["balanced"],
        ),
        users,
    )
    with pytest.raises(
        MilpSolveError,
        match="did not certify",
    ):
        solve_offline_reference_milp(
            trace, CONFIG,
        )


def test_milp_repeated_solves_are_deterministic() -> None:
    trace = _trace(
        "milp-deterministic",
        4,
        (
            _request(1, 0, 0),
            _request(2, 1, 1),
        ),
        "balanced",
        with_communication=True,
    )
    first = solve_offline_reference_milp(
        trace, CONFIG,
    )
    second = solve_offline_reference_milp(
        trace, CONFIG,
    )
    assert first.objective == second.objective
    assert first.selected_plan_ids == (
        second.selected_plan_ids
    )
    assert first.selected_state_indices == (
        second.selected_state_indices
    )
    assert first.accounting == second.accounting


def test_oracle_script_reports_required_reference_metadata() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "scripts/solve_oracle.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    required = {
        "interpretation",
        "interpretation_restriction",
        "information_relaxation",
        "exhaustive_internal_verifier",
        "exhaustive_verification",
        "exhaustive_feasible_selection_count",
        "exhaustive_optimal_selection_count",
        "exhaustive_objective",
        "milp_selection_in_exhaustive_optimal_set",
        "objective_consistency",
        "raw_accounting_consistency",
        "trace_id",
        "horizon_slots",
        "request_count",
        "profile_count",
        "plan_count",
        "variable_count",
        "constraint_count",
        "aggregate_state_count",
        "solver",
        "scipy_version",
        "configured_relative_gap",
        "status",
        "message",
        "success",
        "mip_gap",
        "mip_dual_bound",
        "mip_node_count",
        "model_build_time_s",
        "solver_wall_clock_time_s",
        "wall_clock_time_s",
        "objective",
        "selected_plan_ids",
        "request_assignments",
        "per_slot_sensing_resources",
        "tenant_residuals",
        "communication_residuals",
    }
    assert required <= report.keys()
    assert report["interpretation"] == (
        "finite-horizon offline clairvoyant "
        "constrained plan-selection reference"
    )
    assert report["interpretation_restriction"] == (
        "not a guaranteed upper bound "
        "or deployable online competitor"
    )
    assert report["exhaustive_internal_verifier"] == (
        "not a second reported oracle"
    )
    assert report["exhaustive_verification"]
    assert report[
        "milp_selection_in_exhaustive_optimal_set"
    ]
    assert report["objective_consistency"]
    assert report["raw_accounting_consistency"]
    assert report["exhaustive_objective"] == pytest.approx(
        report["objective"],
        abs=1.0e-7,
    )
    assert report["information_relaxation"] == {
        "future_primitive_information_known": True,
        "deterministic_focal_request_queue_enforced": False,
        "explicit_repeated_defer_actions_consumed": False,
        "explicit_repeated_reject_actions_consumed": False,
        "future_admission_time_represents": "offline deferral",
        "omitted_request_represents": "rejection or non-service",
        "profile_changes_permitted_only_at": (
            "session creation or member admission"
        ),
    }
    assert report["success"]
    assert report["status"] == 0
    assert report["mip_gap"] <= (
        report["configured_relative_gap"]
    )
    assert "online_optimality_gap" not in report
    assert "certified_upper_bound_gap" not in report