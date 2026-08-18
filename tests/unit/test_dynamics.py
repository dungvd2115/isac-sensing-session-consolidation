from __future__ import annotations

from dataclasses import fields
from math import cos, log10, pi, sin, sqrt
from pathlib import Path

import numpy as np
import pytest
import yaml

from isac_ssc.core.entities import RequestState, Task
from isac_ssc.envs.dynamics import (
    CommunicationSlotPrimitive, PrimitiveTrace, RequestPrimitiveDescriptor,
    TargetSlotPrimitive, demand_from_standard_normal,
    fading_transition, generate_primitive_trace,
    mobility_transition, rcs_transition, reflect_axis,
    shadowing_transition, traffic_transition,
)
from isac_ssc.utils.config import DEFAULT_CONFIG_PATH, load_config
from isac_ssc.utils.seeding import SeedContract

CONFIG = load_config()
CONTRACT = SeedContract.from_config(CONFIG)


@pytest.fixture(scope="module")
def independent_trace() -> PrimitiveTrace:
    return generate_primitive_trace(
        CONFIG, 41001, "independent",
    )


@pytest.fixture(scope="module")
def clustered_trace() -> PrimitiveTrace:
    return generate_primitive_trace(
        CONFIG, 41001, "clustered",
    )


def _config(tmp_path: Path, mutate) -> object:
    data = yaml.safe_load(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"),
    )
    mutate(data)
    path = tmp_path/"config.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(path)


def test_initial_target_state_matches_uniform_area_annulus_and_velocity_draw() -> None:
    state = generate_primitive_trace(
        CONFIG, 41001, "independent",
    ).target_at("target_1", 0)
    position_rng = CONTRACT.rng(
        41001, "independent", "initial",
        "target", "target_1", "position",
    )
    specification = CONFIG.geometry["target_initial_position"]
    radius = sqrt(float(position_rng.uniform(
        specification["minimum_radius_m"]**2,
        specification["maximum_radius_m"]**2,
    )))
    angle = float(position_rng.uniform(-pi, pi))
    assert state.position_m == pytest.approx((
        radius*cos(angle),
        radius*sin(angle),
    ))
    velocity_rng = CONTRACT.rng(
        41001, "independent", "initial",
        "target", "target_1", "velocity",
    )
    mobility = CONFIG.mobility["targets"]
    speed = float(velocity_rng.uniform(
        mobility["initial_speed_min_m_per_s"],
        mobility["initial_speed_max_m_per_s"],
    ))
    heading = float(velocity_rng.uniform(-pi, pi))
    assert state.velocity_m_per_s == pytest.approx((
        speed*cos(heading),
        speed*sin(heading),
    ))


@pytest.mark.parametrize(("position", "velocity", "expected"), (
    (201.0, 3.0, (199.0, -3.0)),
    (-203.0, -2.0, (-197.0, 2.0)),
    (-200.0, -2.0, (-200.0, 2.0)),
    (200.0, 2.0, (200.0, -2.0)),
    (-200.0, 2.0, (-200.0, 2.0)),
    (200.0, -2.0, (200.0, -2.0)),
    (1001.0, 4.0, (199.0, -4.0)),
    (-1003.0, -4.0, (-197.0, 4.0)),
))
def test_axiswise_repeated_reflection_contract(
    position, velocity, expected,
) -> None:
    assert reflect_axis(
        position, velocity, -200.0, 200.0,
    ) == pytest.approx(expected)


def test_mobility_transition_uses_exact_cv_matrix_and_corner_reflection() -> None:
    region = {
        "x_min_m": -2.0,
        "x_max_m": 2.0,
        "y_min_m": -2.0,
        "y_max_m": 2.0,
    }
    position, velocity = mobility_transition(
        (1.9, 1.8), (2.0, 3.0),
        (1.0, -2.0), 0.2, region,
    )
    raw_position = (
        1.9 + 0.2*2.0 + 0.5*0.04,
        1.8 + 0.2*3.0 - 0.5*0.08,
    )
    raw_velocity = (
        2.0 + 0.2,
        3.0 - 0.4,
    )
    assert position == pytest.approx((
        4.0-raw_position[0],
        4.0-raw_position[1],
    ))
    assert velocity == pytest.approx((
        -raw_velocity[0],
        -raw_velocity[1],
    ))


def test_shadowing_fading_and_rcs_transitions_match_formulation() -> None:
    rho = 0.9
    assert shadowing_transition(
        2.0, -1.0, rho,
    ) == pytest.approx(
        2.0*rho-sqrt(1-rho**2),
    )
    fading = fading_transition(
        1.0, -2.0, 0.5, 0.25, rho,
    )
    assert fading == pytest.approx((
        rho + sqrt(1-rho**2)*0.5,
        -2*rho + sqrt(1-rho**2)*0.25,
    ))
    expected_rcs = (
        rho*3.0
        + (1-rho)*10*log10(2.0)
        + sqrt(1-rho**2)*-0.5
    )
    assert rcs_transition(
        3.0, -0.5, rho, 2.0,
    ) == pytest.approx(expected_rcs)


def test_traffic_transition_and_demand_follow_frozen_slot_order() -> None:
    assert not traffic_transition(
        True, 0.01, 0.08, 0.20,
    )
    assert traffic_transition(
        True, 0.5, 0.08, 0.20,
    )
    assert traffic_transition(
        False, 0.1, 0.08, 0.20,
    )
    assert not traffic_transition(
        False, 0.5, 0.08, 0.20,
    )
    assert demand_from_standard_normal(
        0.0, 5.0e6, 0.45,
    ) == pytest.approx(5.0e6)


def test_trace_has_complete_target_user_and_innovation_coverage(
    independent_trace: PrimitiveTrace,
) -> None:
    horizon = CONFIG.system["horizon_slots"]
    assert len(independent_trace.target_states) == (
        horizon*CONFIG.population["physical_targets"]
    )
    assert len(independent_trace.communication_states) == (
        horizon*CONFIG.population["communication_users"]
    )
    assert len(independent_trace.target_innovations) == (
        (horizon-1)*CONFIG.population["physical_targets"]
    )
    assert len(independent_trace.communication_innovations) == (
        (horizon-1)*CONFIG.population["communication_users"]
    )


def test_target_and_communication_states_are_indexed_by_physical_entity(
    independent_trace: PrimitiveTrace,
) -> None:
    for slot in range(CONFIG.system["horizon_slots"]):
        targets = [
            item.target_id
            for item in independent_trace.target_states
            if item.slot == slot
        ]
        users = [
            item.user_id
            for item in independent_trace.communication_states
            if item.slot == slot
        ]
        assert len(targets) == len(set(targets)) == (
            CONFIG.population["physical_targets"]
        )
        assert len(users) == len(set(users)) == (
            CONFIG.population["communication_users"]
        )


def test_off_demand_is_zero_and_every_on_transition_has_a_fresh_draw(
    independent_trace: PrimitiveTrace,
) -> None:
    states = {
        (item.slot, item.user_id): item
        for item in independent_trace.communication_states
    }
    innovations = {
        (item.slot, item.user_id): item
        for item in independent_trace.communication_innovations
    }
    on_to_on_draws = []
    for slot in range(1, independent_trace.horizon_slots):
        for user_id in {
            item.user_id
            for item in independent_trace.communication_at(slot)
        }:
            current = states[(slot, user_id)]
            previous = states[(slot-1, user_id)]
            innovation = innovations[(slot, user_id)]
            if current.traffic_on:
                assert current.demand_bit_per_s > 0.0
                assert innovation.demand_standard_normal is not None
                if previous.traffic_on:
                    on_to_on_draws.append(
                        innovation.demand_standard_normal,
                    )
            else:
                assert current.demand_bit_per_s == 0.0
                assert innovation.demand_standard_normal is None
    assert len(on_to_on_draws) > 2
    assert len(set(on_to_on_draws)) == len(on_to_on_draws)


def test_independent_poisson_counts_match_keyed_draws(
    independent_trace: PrimitiveTrace,
) -> None:
    rate = CONFIG.arrivals["independent"]["per_tenant_rate_per_slot"]
    for slot in range(independent_trace.horizon_slots):
        for tenant in CONFIG.tenants:
            expected = int(CONTRACT.rng(
                41001, "arrival", "independent",
                tenant.tenant_id, "slot", slot, "count",
            ).poisson(rate))
            actual = sum(
                item.sampled_slot == slot
                and item.tenant_id == tenant.tenant_id
                for item in independent_trace.request_descriptors
            )
            assert actual == expected


def test_independent_descriptors_have_no_parent_and_materialize_pristine_requests(
    independent_trace: PrimitiveTrace,
) -> None:
    assert not independent_trace.parent_events
    assert all(
        item.source_regime == "independent"
        for item in independent_trace.request_descriptors
    )
    assert all(
        item.sampled_slot == item.arrival_slot
        for item in independent_trace.request_descriptors
    )
    assert all(
        item.parent_id is None and item.child_index is None
        for item in independent_trace.request_descriptors
    )
    requests = independent_trace.materialized_requests(CONFIG)
    assert all(
        item.state is RequestState.WAITING
        and item.eligible_slot == item.arrival_slot
        for item in requests
    )
    assert len(requests) == sum(
        not item.horizon_omitted
        for item in independent_trace.request_descriptors
    )


def test_clustered_parent_and_child_counts_match_keyed_draws(
    clustered_trace: PrimitiveTrace,
) -> None:
    specification = CONFIG.arrivals["clustered"]
    for slot in range(clustered_trace.horizon_slots):
        expected_parents = int(CONTRACT.rng(
            41001, "arrival", "clustered",
            "slot", slot, "parent_count",
        ).poisson(specification["parent_rate_per_slot"]))
        parents = tuple(
            item for item in clustered_trace.parent_events
            if item.sampled_slot == slot
        )
        assert len(parents) == expected_parents
    for parent in clustered_trace.parent_events:
        expected_children = 1 + int(CONTRACT.rng(
            41001, "parent", parent.parent_id, "child_count",
        ).poisson(specification["child_poisson_mean"]))
        children = tuple(
            item for item in clustered_trace.request_descriptors
            if item.parent_id == parent.parent_id
        )
        assert parent.child_count == expected_children == len(children)
        assert {
            item.child_index for item in children
        } == set(range(parent.child_count))


def test_pending_children_are_fully_recoverable_from_descriptors(
    clustered_trace: PrimitiveTrace,
) -> None:
    expected = tuple(
        item for item in clustered_trace.request_descriptors
        if (
            item.source_regime == "clustered"
            and not item.horizon_omitted
            and item.sampled_slot <= 20 < item.arrival_slot
        )
    )
    assert clustered_trace.pending_children_at(20) == expected
    assert all(
        item.parent_id is not None
        and item.child_index is not None
        for item in expected
    )


def test_materialized_child_aoi_uses_arrival_slot_target_position(
    clustered_trace: PrimitiveTrace,
) -> None:
    descriptors = {
        item.request_id: item
        for item in clustered_trace.request_descriptors
        if not item.horizon_omitted
    }
    requests = {
        item.request_id: item
        for item in clustered_trace.materialized_requests(CONFIG)
    }
    for request_id, request in requests.items():
        descriptor = descriptors[request_id]
        target = clustered_trace.target_at(
            descriptor.target_id,
            descriptor.arrival_slot,
        )
        expected = (
            target.position_m[0] + descriptor.aoi_displacement_m[0],
            target.position_m[1] + descriptor.aoi_displacement_m[1],
        )
        assert request.aoi.center_m == pytest.approx(expected)


def test_request_attributes_follow_declared_domains(
    independent_trace: PrimitiveTrace,
) -> None:
    aoi = CONFIG.geometry["aoi"]
    for descriptor in independent_trace.request_descriptors:
        tenant = CONFIG.tenant(descriptor.tenant_id)
        assert descriptor.task in tenant.permitted_tasks
        assert (
            aoi["radius_min_m"]
            <= descriptor.aoi_radius_m
            <= aoi["radius_max_m"]
        )
        assert np.linalg.norm(
            descriptor.aoi_displacement_m,
        ) <= (
            aoi["center_offset_max_fraction_of_radius"]
            * descriptor.aoi_radius_m
            + 1e-12
        )
        interval = CONFIG.requests[
            "update_interval_slots"
        ][descriptor.task.value]
        assert descriptor.valid_output_interval_slots in interval["values"]
        completion = CONFIG.requests[
            "completion_values"
        ][descriptor.task.value]
        assert (
            completion["minimum"]
            <= descriptor.completion_value
            <= completion["maximum"]
        )


def test_tenant_task_restriction_is_renormalized(
    tmp_path: Path,
) -> None:
    def mutate(data: dict) -> None:
        data["system"]["horizon_slots"] = 12
        data["tenant_profiles"]["tenant_1"]["permitted_tasks"] = [
            "detection",
        ]
        data["arrivals"]["independent"]["per_tenant_rate_per_slot"] = 2.0

    config = _config(tmp_path, mutate)
    trace = generate_primitive_trace(
        config, 99, "independent",
    )
    tenant_one = [
        item for item in trace.request_descriptors
        if item.tenant_id == "tenant_1"
    ]
    assert tenant_one
    assert {
        item.task for item in tenant_one
    } == {Task.DETECTION}


def test_horizon_omission_records_every_sampled_candidate_without_replacement(
    tmp_path: Path,
) -> None:
    def mutate(data: dict) -> None:
        data["system"]["horizon_slots"] = 5
        data["arrivals"]["independent"]["per_tenant_rate_per_slot"] = 2.0
        data["arrivals"]["clustered"]["parent_rate_per_slot"] = 2.0
        data["arrivals"]["clustered"]["child_poisson_mean"] = 2.0

    config = _config(tmp_path, mutate)
    contract = SeedContract.from_config(config)
    independent = generate_primitive_trace(
        config, 123, "independent",
    )
    expected_independent = sum(
        int(contract.rng(
            123, "arrival", "independent",
            tenant.tenant_id, "slot", slot, "count",
        ).poisson(2.0))
        for slot in range(5)
        for tenant in config.tenants
    )
    assert len(independent.request_descriptors) == expected_independent
    assert independent.horizon_omitted_descriptors()

    clustered = generate_primitive_trace(
        config, 123, "clustered",
    )
    assert len(clustered.request_descriptors) == sum(
        item.child_count for item in clustered.parent_events
    )
    assert clustered.horizon_omitted_descriptors()
    assert (
        len(clustered.materialized_requests(config))
        + len(clustered.horizon_omitted_descriptors())
        == len(clustered.request_descriptors)
    )


def test_trace_records_only_primitive_fields() -> None:
    names = {
        field.name
        for cls in (
            TargetSlotPrimitive,
            CommunicationSlotPrimitive,
            RequestPrimitiveDescriptor,
        )
        for field in fields(cls)
    }
    prohibited = {
        "communication_sinr", "communication_rate",
        "sensing_sinr", "detection_probability",
        "peb", "pcrb", "compatibility",
        "sla_satisfaction", "reward", "constraint_cost",
    }
    assert names.isdisjoint(prohibited)


def test_different_seed_or_regime_changes_primitive_trace(
    independent_trace: PrimitiveTrace,
    clustered_trace: PrimitiveTrace,
) -> None:
    other_seed = generate_primitive_trace(
        CONFIG, 41002, "independent",
    )
    assert independent_trace != other_seed
    assert independent_trace != clustered_trace
