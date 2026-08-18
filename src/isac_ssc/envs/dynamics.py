"""Primitive uncertainty generation for the event-driven ISAC environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot, isfinite, log, log10, pi, sin, sqrt
from numbers import Real
from typing import Iterable, Mapping

import numpy as np

from isac_ssc.core.entities import DiskAOI, EntityId, SensingRequest, Task, Tenant, Vector2
from isac_ssc.utils.config import CanonicalConfig
from isac_ssc.utils.seeding import SeedContract

_AOI_REJECTION_LIMIT = 100_000


class DynamicsValidationError(ValueError):
    """Raised when a primitive dynamics input is invalid."""


def _finite(value: object, name: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise DynamicsValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and (number <= minimum if strict else number < minimum):
        operator = ">" if strict else ">="
        raise DynamicsValidationError(f"{name} must be {operator} {minimum}")
    return number


def _identifier_key(value: EntityId) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _vector2(value: Iterable[float], name: str) -> Vector2:
    try:
        vector = tuple(value)
    except TypeError as error:
        raise DynamicsValidationError(f"{name} must contain two finite values") from error
    if len(vector) != 2:
        raise DynamicsValidationError(f"{name} must contain two values")
    return _finite(vector[0], f"{name}[0]"), _finite(vector[1], f"{name}[1]")


def _probability(value: object, name: str) -> float:
    number = _finite(value, name, minimum=0.0)
    if number > 1.0:
        raise DynamicsValidationError(f"{name} must lie in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class TargetSlotPrimitive:
    slot: int
    target_id: EntityId
    position_m: Vector2
    velocity_m_per_s: Vector2
    shadowing_db: float
    fading_real: float
    fading_imag: float
    rcs_dbsm: float

    @property
    def fading_power_gain(self) -> float:
        return self.fading_real*self.fading_real + self.fading_imag*self.fading_imag

    @property
    def rcs_m2(self) -> float:
        return 10.0**(self.rcs_dbsm/10.0)


@dataclass(frozen=True, slots=True)
class CommunicationSlotPrimitive:
    slot: int
    user_id: EntityId
    position_m: Vector2
    velocity_m_per_s: Vector2
    traffic_on: bool
    demand_bit_per_s: float
    shadowing_db: float
    fading_real: float
    fading_imag: float

    @property
    def fading_power_gain(self) -> float:
        return self.fading_real*self.fading_real + self.fading_imag*self.fading_imag


@dataclass(frozen=True, slots=True)
class TargetTransitionInnovation:
    slot: int
    target_id: EntityId
    acceleration_m_per_s2: Vector2
    shadowing_innovation_db: float
    fading_innovation_real: float
    fading_innovation_imag: float
    rcs_innovation_dbsm: float


@dataclass(frozen=True, slots=True)
class CommunicationTransitionInnovation:
    slot: int
    user_id: EntityId
    acceleration_m_per_s2: Vector2
    shadowing_innovation_db: float
    fading_innovation_real: float
    fading_innovation_imag: float
    traffic_transition_uniform: float
    demand_standard_normal: float | None


@dataclass(frozen=True, slots=True)
class ParentEventPrimitive:
    parent_id: EntityId
    sampled_slot: int
    target_id: EntityId
    task: Task
    child_count: int


@dataclass(frozen=True, slots=True)
class RequestPrimitiveDescriptor:
    request_id: EntityId
    source_regime: str
    sampled_slot: int
    arrival_slot: int
    tenant_id: EntityId
    target_id: EntityId
    task: Task
    aoi_radius_m: float
    aoi_displacement_m: Vector2
    latest_start_slack_slots: int
    valid_output_interval_slots: int
    quality_threshold: float
    completion_value: float
    merge_permission: bool
    parent_id: EntityId | None = None
    child_index: int | None = None
    horizon_omitted: bool = False


@dataclass(frozen=True, slots=True)
class PrimitiveTrace:
    trace_id: str
    root_seed: int
    arrival_regime: str
    horizon_slots: int
    tenant_ids: tuple[EntityId, ...]
    tenant_authorization_matrix: tuple[tuple[bool, ...], ...]
    target_states: tuple[TargetSlotPrimitive, ...]
    target_innovations: tuple[TargetTransitionInnovation, ...]
    communication_states: tuple[CommunicationSlotPrimitive, ...]
    communication_innovations: tuple[CommunicationTransitionInnovation, ...]
    parent_events: tuple[ParentEventPrimitive, ...]
    request_descriptors: tuple[RequestPrimitiveDescriptor, ...]

    def target_at(self, target_id: EntityId, slot: int) -> TargetSlotPrimitive:
        key = _identifier_key(target_id)
        matches = tuple(
            item for item in self.target_states
            if item.slot == slot and _identifier_key(item.target_id) == key
        )
        if len(matches) != 1:
            raise KeyError(f"target state not found: {target_id!r} at slot {slot}")
        return matches[0]

    def communication_at(self, slot: int) -> tuple[CommunicationSlotPrimitive, ...]:
        return tuple(item for item in self.communication_states if item.slot == slot)

    def request_descriptors_at(self, slot: int) -> tuple[RequestPrimitiveDescriptor, ...]:
        return tuple(item for item in self.request_descriptors if item.sampled_slot == slot)

    def pending_children_at(self, slot: int) -> tuple[RequestPrimitiveDescriptor, ...]:
        return tuple(item for item in self.request_descriptors if (
            item.source_regime == "clustered" and not item.horizon_omitted
            and item.sampled_slot <= slot < item.arrival_slot
        ))

    def horizon_omitted_descriptors(self) -> tuple[RequestPrimitiveDescriptor, ...]:
        return tuple(item for item in self.request_descriptors if item.horizon_omitted)

    def materialized_requests(self, config: CanonicalConfig) -> tuple[SensingRequest, ...]:
        requests = []
        for descriptor in self.request_descriptors:
            if descriptor.horizon_omitted:
                continue
            target = self.target_at(descriptor.target_id, descriptor.arrival_slot)
            center = (
                target.position_m[0] + descriptor.aoi_displacement_m[0],
                target.position_m[1] + descriptor.aoi_displacement_m[1],
            )
            duration = config.service_duration_slots[descriptor.task]
            latest_start = min(
                descriptor.arrival_slot + descriptor.latest_start_slack_slots,
                self.horizon_slots-duration,
            )
            requests.append(SensingRequest(
                descriptor.request_id, descriptor.tenant_id, descriptor.arrival_slot, latest_start,
                DiskAOI(center, descriptor.aoi_radius_m), descriptor.target_id, descriptor.task,
                descriptor.quality_threshold, descriptor.valid_output_interval_slots,
                descriptor.completion_value, descriptor.merge_permission,
            ))
        return tuple(sorted(
            requests, key=lambda item: (item.arrival_slot, _identifier_key(item.request_id)),
        ))

    def requests_arriving_at(self, slot: int, config: CanonicalConfig) -> tuple[SensingRequest, ...]:
        return tuple(item for item in self.materialized_requests(config) if item.arrival_slot == slot)


def reflect_axis(position: float, velocity: float, lower: float, upper: float) -> tuple[float, float]:
    """Apply the frozen repeated-mirror convention to one Cartesian axis."""
    value = _finite(position, "position")
    speed = _finite(velocity, "velocity")
    low = _finite(lower, "lower")
    high = _finite(upper, "upper")
    if high <= low:
        raise DynamicsValidationError("upper boundary must exceed lower boundary")
    while value < low or value > high:
        if value < low:
            value = 2.0*low-value
            speed = -speed
        elif value > high:
            value = 2.0*high-value
            speed = -speed
    if value == low and speed < 0.0 or value == high and speed > 0.0:
        speed = -speed
    return value, speed


def mobility_transition(
    position_m: Vector2, velocity_m_per_s: Vector2, acceleration_m_per_s2: Vector2,
    slot_duration_s: float, region: Mapping[str, float],
) -> tuple[Vector2, Vector2]:
    position = _vector2(position_m, "position_m")
    velocity = _vector2(velocity_m_per_s, "velocity_m_per_s")
    acceleration = _vector2(acceleration_m_per_s2, "acceleration_m_per_s2")
    duration = _finite(slot_duration_s, "slot_duration_s", minimum=0.0, strict=True)
    get = region.__getitem__
    next_position = (
        position[0] + duration*velocity[0] + 0.5*duration*duration*acceleration[0],
        position[1] + duration*velocity[1] + 0.5*duration*duration*acceleration[1],
    )
    next_velocity = (
        velocity[0] + duration*acceleration[0],
        velocity[1] + duration*acceleration[1],
    )
    x, vx = reflect_axis(next_position[0], next_velocity[0], get("x_min_m"), get("x_max_m"))
    y, vy = reflect_axis(next_position[1], next_velocity[1], get("y_min_m"), get("y_max_m"))
    return (x, y), (vx, vy)


def shadowing_transition(previous_db: float, innovation_db: float, correlation: float) -> float:
    rho = _probability(correlation, "correlation")
    return rho*_finite(previous_db, "previous_db") + sqrt(1.0-rho*rho)*_finite(
        innovation_db, "innovation_db",
    )


def fading_transition(
    previous_real: float, previous_imag: float, innovation_real: float,
    innovation_imag: float, correlation: float,
) -> tuple[float, float]:
    rho = _probability(correlation, "correlation")
    scale = sqrt(1.0-rho*rho)
    return (
        rho*_finite(previous_real, "previous_real")
        + scale*_finite(innovation_real, "innovation_real"),
        rho*_finite(previous_imag, "previous_imag")
        + scale*_finite(innovation_imag, "innovation_imag"),
    )


def rcs_transition(
    previous_dbsm: float, innovation_dbsm: float, correlation: float, median_m2: float,
) -> float:
    rho = _probability(correlation, "correlation")
    median = _finite(median_m2, "median_m2", minimum=0.0, strict=True)
    mean = 10.0*log10(median)
    return (
        rho*_finite(previous_dbsm, "previous_dbsm") + (1.0-rho)*mean
        + sqrt(1.0-rho*rho)*_finite(innovation_dbsm, "innovation_dbsm")
    )


def traffic_transition(
    previous_on: bool, transition_uniform: float, on_to_off: float, off_to_on: float,
) -> bool:
    if type(previous_on) is not bool:
        raise DynamicsValidationError("previous_on must be boolean")
    draw = _probability(transition_uniform, "transition_uniform")
    return (
        draw >= _probability(on_to_off, "on_to_off")
        if previous_on
        else draw < _probability(off_to_on, "off_to_on")
    )


def demand_from_standard_normal(
    standard_normal: float, median_bit_per_s: float, natural_log_std: float,
) -> float:
    return exp(
        log(_finite(median_bit_per_s, "median_bit_per_s", minimum=0.0, strict=True))
        + _finite(natural_log_std, "natural_log_std", minimum=0.0)
        * _finite(standard_normal, "standard_normal")
    )


def _sample_annulus(
    contract: SeedContract, seed: int, tokens: tuple[object, ...],
    specification: Mapping[str, object], bs_position: Vector2,
) -> Vector2:
    rng = contract.rng(seed, *tokens)
    minimum = float(specification["minimum_radius_m"])
    maximum = float(specification["maximum_radius_m"])
    radius = sqrt(float(rng.uniform(minimum*minimum, maximum*maximum)))
    angle = float(rng.uniform(-pi, pi))
    return bs_position[0] + radius*cos(angle), bs_position[1] + radius*sin(angle)


def _sample_velocity(
    contract: SeedContract, seed: int, tokens: tuple[object, ...],
    specification: Mapping[str, object],
) -> Vector2:
    rng = contract.rng(seed, *tokens)
    speed = float(rng.uniform(
        specification["initial_speed_min_m_per_s"],
        specification["initial_speed_max_m_per_s"],
    ))
    heading = float(rng.uniform(-pi, pi))
    return speed*cos(heading), speed*sin(heading)


def _complex_normal(rng: np.random.Generator) -> tuple[float, float]:
    scale = 1.0/sqrt(2.0)
    return float(rng.normal(0.0, scale)), float(rng.normal(0.0, scale))


def _categorical(
    rng: np.random.Generator, values: tuple[object, ...],
    probabilities: tuple[float, ...] | None = None,
) -> object:
    index = int(rng.choice(len(values), p=None if probabilities is None else probabilities))
    return values[index]


def _task_for_tenant(
    contract: SeedContract, seed: int, tokens: tuple[object, ...],
    tenant: Tenant, config: CanonicalConfig,
) -> Task:
    retained = tuple(task for task in Task if task in tenant.permitted_tasks)
    masses = tuple(config.task_probabilities[task] for task in retained)
    total = sum(masses)
    probabilities = tuple(value/total for value in masses)
    return _categorical(  # type: ignore[return-value]
        contract.rng(seed, *tokens), retained, probabilities,
    )


def _request_attribute(
    contract: SeedContract, seed: int, request_id: EntityId, name: str,
    minimum: float, maximum: float,
) -> float:
    return float(contract.rng(seed, "request", request_id, name).uniform(minimum, maximum))


def _sample_aoi_displacement(
    contract: SeedContract, seed: int, request_id: EntityId,
    std_m: float, maximum_radius_m: float,
) -> Vector2:
    rng = contract.rng(seed, "request", request_id, "aoi_displacement")
    for _ in range(_AOI_REJECTION_LIMIT):
        value = rng.normal(0.0, std_m, size=2)
        if hypot(float(value[0]), float(value[1])) <= maximum_radius_m:
            return float(value[0]), float(value[1])
    raise DynamicsValidationError("AOI displacement rejection sampler exceeded its attempt limit")


def _descriptor(
    contract: SeedContract, seed: int, request_id: EntityId, source_regime: str,
    sampled_slot: int, arrival_slot: int, tenant: Tenant, target_id: EntityId,
    task: Task, config: CanonicalConfig, *, parent_id: EntityId | None = None,
    child_index: int | None = None,
) -> RequestPrimitiveDescriptor:
    aoi = config.geometry["aoi"]
    radius = _request_attribute(
        contract, seed, request_id, "aoi_radius",
        aoi["radius_min_m"], aoi["radius_max_m"],
    )
    std = (
        config.arrivals["clustered"]["child_aoi_center_offset_std_m"]
        if source_regime == "clustered"
        else aoi["center_offset_std_m"]
    )
    displacement = _sample_aoi_displacement(
        contract, seed, request_id, std,
        aoi["center_offset_max_fraction_of_radius"]*radius,
    )
    slack_spec = config.requests["latest_start_slack_slots"]
    slack = int(contract.rng(seed, "request", request_id, "latest_start_slack").integers(
        slack_spec["minimum"], slack_spec["maximum"]+1,
    ))
    interval_spec = config.requests["update_interval_slots"][task.value]
    interval = int(_categorical(
        contract.rng(seed, "request", request_id, "update_interval"),
        tuple(interval_spec["values"]), tuple(interval_spec["probabilities"]),
    ))
    threshold_name = {
        Task.DETECTION: "detection_probability",
        Task.LOCALIZATION: "localization_peb_m",
        Task.TRACKING: "tracking_pcrb_m",
    }[task]
    threshold_spec = config.requests["quality_thresholds"][threshold_name]
    threshold = _request_attribute(
        contract, seed, request_id, "quality_threshold",
        threshold_spec["minimum"], threshold_spec["maximum"],
    )
    value_spec = config.requests["completion_values"][task.value]
    completion_value = _request_attribute(
        contract, seed, request_id, "completion_value",
        value_spec["minimum"], value_spec["maximum"],
    )
    merge_permission = bool(
        contract.rng(seed, "request", request_id, "merge_permission").random()
        < config.requests["sharing_permission"]["probability_true"]
    )
    duration = config.service_duration_slots[task]
    omitted = arrival_slot+duration > config.system["horizon_slots"]
    return RequestPrimitiveDescriptor(
        request_id, source_regime, sampled_slot, arrival_slot, tenant.tenant_id,
        target_id, task, radius, displacement, slack, interval, threshold,
        completion_value, merge_permission, parent_id, child_index, omitted,
    )


def _generate_independent_descriptors(
    contract: SeedContract, seed: int, slot: int, target_ids: tuple[EntityId, ...],
    config: CanonicalConfig,
) -> tuple[RequestPrimitiveDescriptor, ...]:
    descriptors = []
    rate = config.arrivals["independent"]["per_tenant_rate_per_slot"]
    for tenant in config.tenants:
        count = int(contract.rng(
            seed, "arrival", "independent", tenant.tenant_id, "slot", slot, "count",
        ).poisson(rate))
        for index in range(count):
            request_id = f"independent:{tenant.tenant_id}:{slot}:{index}"
            target_id = _categorical(
                contract.rng(seed, "request", request_id, "target"), target_ids,
            )
            task = _task_for_tenant(
                contract, seed, ("request", request_id, "task"), tenant, config,
            )
            descriptors.append(_descriptor(
                contract, seed, request_id, "independent", slot, slot,
                tenant, target_id, task, config,
            ))
    return tuple(descriptors)


def _generate_clustered_slot(
    contract: SeedContract, seed: int, slot: int,
    target_ids: tuple[EntityId, ...], config: CanonicalConfig, parent_start: int,
) -> tuple[tuple[ParentEventPrimitive, ...], tuple[RequestPrimitiveDescriptor, ...]]:
    specification = config.arrivals["clustered"]
    parent_count = int(contract.rng(
        seed, "arrival", "clustered", "slot", slot, "parent_count",
    ).poisson(specification["parent_rate_per_slot"]))
    parents = []
    descriptors = []
    global_tasks = tuple(Task(value) for value in config.requests["task"]["values"])
    global_probabilities = tuple(config.requests["task"]["probabilities"])
    tenant_values = tuple(config.tenants)
    for local_index in range(parent_count):
        parent_id = f"parent:{slot}:{parent_start+local_index}"
        target_id = _categorical(
            contract.rng(seed, "parent", parent_id, "target"), target_ids,
        )
        parent_task = _categorical(
            contract.rng(seed, "parent", parent_id, "task"),
            global_tasks, global_probabilities,
        )
        child_count = 1 + int(contract.rng(
            seed, "parent", parent_id, "child_count",
        ).poisson(specification["child_poisson_mean"]))
        parents.append(ParentEventPrimitive(
            parent_id, slot, target_id, parent_task, child_count,
        ))
        for child_index in range(child_count):
            request_id = f"clustered:{parent_id}:{child_index}"
            child_rng = contract.rng(
                seed, "parent", parent_id, "child", child_index, "arrival_offset",
            )
            arrival_slot = slot + int(child_rng.integers(
                specification["temporal_offset_min_slots"],
                specification["temporal_offset_max_slots"]+1,
            ))
            tenant = _categorical(
                contract.rng(
                    seed, "parent", parent_id, "child", child_index, "tenant",
                ),
                tenant_values,
            )
            inherit_target = bool(contract.rng(
                seed, "parent", parent_id, "child", child_index, "inherit_target",
            ).random() < specification["inherit_parent_target_probability"])
            child_target = (
                target_id
                if inherit_target
                else _categorical(
                    contract.rng(
                        seed, "parent", parent_id, "child", child_index, "target",
                    ),
                    target_ids,
                )
            )
            inherit_task = bool(contract.rng(
                seed, "parent", parent_id, "child", child_index, "inherit_task",
            ).random() < specification["inherit_parent_task_probability"])
            child_task = (
                parent_task
                if inherit_task and parent_task in tenant.permitted_tasks
                else _task_for_tenant(
                    contract, seed,
                    ("parent", parent_id, "child", child_index, "task"),
                    tenant, config,
                )
            )
            descriptors.append(_descriptor(
                contract, seed, request_id, "clustered", slot, arrival_slot,
                tenant, child_target, child_task, config,
                parent_id=parent_id, child_index=child_index,
            ))
    return tuple(parents), tuple(descriptors)


def generate_primitive_trace(
    config: CanonicalConfig, root_seed: int, arrival_regime: str,
) -> PrimitiveTrace:
    """Generate one complete trace of primitive uncertainties."""
    contract = SeedContract.from_config(config)
    if arrival_regime not in tuple(config.trace_generation["registered_arrival_regimes"]):
        raise DynamicsValidationError("arrival_regime is not registered")
    contract.canonical_material(root_seed, "trace")
    horizon = config.system["horizon_slots"]
    target_ids = tuple(
        f"target_{index+1}" for index in range(config.population["physical_targets"])
    )
    user_ids = tuple(
        f"user_{index+1}" for index in range(config.population["communication_users"])
    )
    bs_position = tuple(config.geometry["bs_position_m"])
    region = config.geometry["simulation_region"]
    duration = config.system["slot_duration_s"]
    target_states: list[TargetSlotPrimitive] = []
    target_innovations: list[TargetTransitionInnovation] = []
    communication_states: list[CommunicationSlotPrimitive] = []
    communication_innovations: list[CommunicationTransitionInnovation] = []
    target_previous: dict[EntityId, TargetSlotPrimitive] = {}
    communication_previous: dict[EntityId, CommunicationSlotPrimitive] = {}
    sensing_shadow_std = config.sensing["shadowing_std_db"]
    sensing_rcs = config.sensing["rcs"]
    rcs_mean = 10.0*log10(sensing_rcs["median_m2"])

    for target_id in target_ids:
        position = _sample_annulus(
            contract, root_seed,
            (arrival_regime, "initial", "target", target_id, "position"),
            config.geometry["target_initial_position"], bs_position,
        )
        velocity = _sample_velocity(
            contract, root_seed,
            (arrival_regime, "initial", "target", target_id, "velocity"),
            config.mobility["targets"],
        )
        shadowing = float(contract.rng(
            root_seed, arrival_regime, "initial", "target", target_id, "shadowing",
        ).normal(0.0, sensing_shadow_std))
        fading = _complex_normal(contract.rng(
            root_seed, arrival_regime, "initial", "target", target_id, "fading",
        ))
        rcs_dbsm = float(contract.rng(
            root_seed, arrival_regime, "initial", "target", target_id, "rcs",
        ).normal(rcs_mean, sensing_rcs["dbsm_std_db"]))
        state = TargetSlotPrimitive(
            0, target_id, position, velocity, shadowing, *fading, rcs_dbsm,
        )
        target_previous[target_id] = state
        target_states.append(state)

    traffic = config.communication["traffic"]
    channel = config.communication["channel"]
    for user_id in user_ids:
        position = _sample_annulus(
            contract, root_seed,
            (arrival_regime, "initial", "communication", user_id, "position"),
            config.geometry["communication_user_initial_position"], bs_position,
        )
        velocity = _sample_velocity(
            contract, root_seed,
            (arrival_regime, "initial", "communication", user_id, "velocity"),
            config.mobility["communication_users"],
        )
        shadowing = float(contract.rng(
            root_seed, arrival_regime, "initial", "communication", user_id, "shadowing",
        ).normal(0.0, channel["shadowing_std_db"]))
        fading = _complex_normal(contract.rng(
            root_seed, arrival_regime, "initial", "communication", user_id, "fading",
        ))
        traffic_on = bool(contract.rng(
            root_seed, arrival_regime, "initial", "communication", user_id, "traffic",
        ).random() < traffic["initial_on_probability"])
        demand = 0.0
        if traffic_on:
            draw = float(contract.rng(
                root_seed, arrival_regime, "initial", "communication", user_id, "demand",
            ).normal())
            demand = demand_from_standard_normal(
                draw, traffic["demand_median_bit_per_s"],
                traffic["demand_natural_log_std"],
            )
        state = CommunicationSlotPrimitive(
            0, user_id, position, velocity, traffic_on,
            demand, shadowing, *fading,
        )
        communication_previous[user_id] = state
        communication_states.append(state)

    parent_events: list[ParentEventPrimitive] = []
    request_descriptors: list[RequestPrimitiveDescriptor] = []
    parent_counter = 0
    for slot in range(horizon):
        if slot > 0:
            for target_id in target_ids:
                previous = target_previous[target_id]
                acceleration = tuple(float(value) for value in contract.rng(
                    root_seed, arrival_regime, "transition", "target",
                    target_id, "slot", slot, "acceleration",
                ).normal(
                    0.0, config.mobility["targets"]["acceleration_std_m_per_s2"], size=2,
                ))
                shadowing_innovation = float(contract.rng(
                    root_seed, arrival_regime, "transition", "target",
                    target_id, "slot", slot, "shadowing",
                ).normal(0.0, sensing_shadow_std))
                fading_innovation = _complex_normal(contract.rng(
                    root_seed, arrival_regime, "transition", "target",
                    target_id, "slot", slot, "fading",
                ))
                rcs_innovation = float(contract.rng(
                    root_seed, arrival_regime, "transition", "target",
                    target_id, "slot", slot, "rcs",
                ).normal(0.0, sensing_rcs["dbsm_std_db"]))
                innovation = TargetTransitionInnovation(
                    slot, target_id, acceleration, shadowing_innovation,
                    *fading_innovation, rcs_innovation,
                )
                position, velocity = mobility_transition(
                    previous.position_m, previous.velocity_m_per_s,
                    acceleration, duration, region,
                )
                fading = fading_transition(
                    previous.fading_real, previous.fading_imag, *fading_innovation,
                    config.sensing["fading_correlation"],
                )
                state = TargetSlotPrimitive(
                    slot, target_id, position, velocity,
                    shadowing_transition(
                        previous.shadowing_db, shadowing_innovation,
                        config.sensing["shadowing_correlation"],
                    ),
                    *fading,
                    rcs_transition(
                        previous.rcs_dbsm, rcs_innovation,
                        sensing_rcs["correlation"], sensing_rcs["median_m2"],
                    ),
                )
                target_innovations.append(innovation)
                target_states.append(state)
                target_previous[target_id] = state

            for user_id in user_ids:
                previous = communication_previous[user_id]
                acceleration = tuple(float(value) for value in contract.rng(
                    root_seed, arrival_regime, "transition", "communication",
                    user_id, "slot", slot, "acceleration",
                ).normal(
                    0.0,
                    config.mobility["communication_users"]["acceleration_std_m_per_s2"],
                    size=2,
                ))
                shadowing_innovation = float(contract.rng(
                    root_seed, arrival_regime, "transition", "communication",
                    user_id, "slot", slot, "shadowing",
                ).normal(0.0, channel["shadowing_std_db"]))
                fading_innovation = _complex_normal(contract.rng(
                    root_seed, arrival_regime, "transition", "communication",
                    user_id, "slot", slot, "fading",
                ))
                traffic_uniform = float(contract.rng(
                    root_seed, arrival_regime, "transition", "communication",
                    user_id, "slot", slot, "traffic",
                ).random())
                traffic_on = traffic_transition(
                    previous.traffic_on, traffic_uniform,
                    traffic["on_to_off_probability"],
                    traffic["off_to_on_probability"],
                )
                demand_draw = None
                demand = 0.0
                if traffic_on:
                    demand_draw = float(contract.rng(
                        root_seed, arrival_regime, "transition", "communication",
                        user_id, "slot", slot, "demand",
                    ).normal())
                    demand = demand_from_standard_normal(
                        demand_draw, traffic["demand_median_bit_per_s"],
                        traffic["demand_natural_log_std"],
                    )
                innovation = CommunicationTransitionInnovation(
                    slot, user_id, acceleration, shadowing_innovation,
                    *fading_innovation, traffic_uniform, demand_draw,
                )
                position, velocity = mobility_transition(
                    previous.position_m, previous.velocity_m_per_s,
                    acceleration, duration, region,
                )
                fading = fading_transition(
                    previous.fading_real, previous.fading_imag, *fading_innovation,
                    channel["fading_correlation"],
                )
                state = CommunicationSlotPrimitive(
                    slot, user_id, position, velocity, traffic_on, demand,
                    shadowing_transition(
                        previous.shadowing_db, shadowing_innovation,
                        channel["shadowing_correlation"],
                    ),
                    *fading,
                )
                communication_innovations.append(innovation)
                communication_states.append(state)
                communication_previous[user_id] = state

        if arrival_regime == "independent":
            request_descriptors.extend(_generate_independent_descriptors(
                contract, root_seed, slot, target_ids, config,
            ))
        else:
            parents, descriptors = _generate_clustered_slot(
                contract, root_seed, slot, target_ids, config, parent_counter,
            )
            parent_counter += len(parents)
            parent_events.extend(parents)
            request_descriptors.extend(descriptors)

    return PrimitiveTrace(
        f"{arrival_regime}:{root_seed}", root_seed, arrival_regime, horizon,
        tuple(tenant.tenant_id for tenant in config.tenants),
        tuple(tenant.authorization_row for tenant in config.tenants),
        tuple(sorted(
            target_states,
            key=lambda item: (item.slot, _identifier_key(item.target_id)),
        )),
        tuple(sorted(
            target_innovations,
            key=lambda item: (item.slot, _identifier_key(item.target_id)),
        )),
        tuple(sorted(
            communication_states,
            key=lambda item: (item.slot, _identifier_key(item.user_id)),
        )),
        tuple(sorted(
            communication_innovations,
            key=lambda item: (item.slot, _identifier_key(item.user_id)),
        )),
        tuple(sorted(
            parent_events,
            key=lambda item: (item.sampled_slot, _identifier_key(item.parent_id)),
        )),
        tuple(sorted(
            request_descriptors,
            key=lambda item: (
                item.sampled_slot, item.arrival_slot,
                _identifier_key(item.request_id),
            ),
        )),
    )
