"""Finite-horizon offline clairvoyant constrained plan-selection MILP."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose, isfinite
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, lil_matrix

from isac_ssc.core.entities import EntityId
from isac_ssc.core.quality import CommunicationParameters, evaluate_communication_quality
from isac_ssc.core.resources import (
    SensingResourceUsage, equal_share_communication_resources,
    residual_communication_resources,
)
from isac_ssc.core.sla import communication_qos_slot
from isac_ssc.oracles.reference import (
    CommunicationSlotOutcome, JointSelectionAccounting, OracleInstance,
    SingleSessionPlan, enumerate_single_session_plans, evaluate_joint_selection,
)
from isac_ssc.utils.config import CanonicalConfig

NUMERICAL_TOLERANCE = 1.0e-7


class MilpValidationError(ValueError):
    """Raised when MILP coefficients cannot be represented safely."""


class MilpSolveError(RuntimeError):
    """Raised when HiGHS does not return a certified acceptable MILP solution."""


def _identifier_key(value: EntityId) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _identifier_label(value: EntityId) -> str:
    identifier_type, identifier_value = _identifier_key(value)
    return f"{identifier_type}:{identifier_value}"


def _fraction(value: float) -> Fraction:
    if not isfinite(float(value)):
        raise MilpValidationError("resource coefficients must be finite")
    return Fraction(str(float(value)))


def _close(left: float, right: float, tolerance: float = NUMERICAL_TOLERANCE) -> bool:
    return isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


@dataclass(frozen=True, slots=True)
class AggregateSensingState:
    slot: int
    bandwidth_hz: float
    power_w: float
    communication_outcomes: tuple[CommunicationSlotOutcome, ...]

    def communication_residual(self, user_id: EntityId) -> float:
        return next((item.residual for item in self.communication_outcomes if item.user_id == user_id), 0.0)


@dataclass(slots=True)
class OfflineMilpModel:
    instance: OracleInstance
    config: CanonicalConfig
    plans: tuple[SingleSessionPlan, ...]
    aggregate_states: tuple[tuple[AggregateSensingState, ...], ...]
    objective_coefficients: np.ndarray
    integrality: np.ndarray
    variable_lower_bounds: np.ndarray
    variable_upper_bounds: np.ndarray
    constraint_matrix: csc_array
    constraint_lower_bounds: np.ndarray
    constraint_upper_bounds: np.ndarray
    row_labels: tuple[str, ...]
    plan_variable_indices: tuple[int, ...]
    state_variable_indices: tuple[tuple[int, ...], ...]

    @property
    def variable_count(self) -> int:
        return int(self.objective_coefficients.size)

    @property
    def constraint_count(self) -> int:
        return int(self.constraint_lower_bounds.size)

    @property
    def plan_count(self) -> int:
        return len(self.plans)

    @property
    def aggregate_state_count(self) -> int:
        return sum(len(states) for states in self.aggregate_states)

    def row_index(self, label: str) -> int:
        try:
            return self.row_labels.index(label)
        except ValueError as error:
            raise KeyError(label) from error


@dataclass(frozen=True, slots=True)
class OfflineMilpSolution:
    solver: str
    scipy_version: str
    configured_relative_gap: float
    status: int
    message: str
    success: bool
    mip_gap: float
    mip_dual_bound: float
    mip_node_count: int
    model_build_time_s: float
    solver_wall_clock_time_s: float
    wall_clock_time_s: float
    objective: float
    selected_plan_ids: tuple[str, ...]
    selected_state_indices: tuple[int, ...]
    accounting: JointSelectionAccounting
    plan_count: int
    variable_count: int
    constraint_count: int
    aggregate_state_count: int


def _communication_outcomes(
    instance: OracleInstance, config: CanonicalConfig,
    slot: int, bandwidth_hz: float, power_w: float,
) -> tuple[CommunicationSlotOutcome, ...]:
    communication = instance.communication_at(slot, config)
    if not communication:
        return ()

    total_bandwidth = config.system["total_bandwidth_hz"]
    total_power = config.system["total_power_w"]
    residual = residual_communication_resources(
        total_bandwidth, total_power, SensingResourceUsage(bandwidth_hz, power_w),
    )
    users = tuple(user for _, user in communication)
    allocations = {item.user_id: item for item in equal_share_communication_resources(users, residual)}
    parameters = CommunicationParameters.from_config(config)
    outcomes = []

    for primitive, user in communication:
        allocation = allocations[user.user_id]
        quality = evaluate_communication_quality(
            user.position_m, config.geometry["bs_position_m"], allocation.bandwidth_hz,
            allocation.power_w, user.demand_bit_per_s, primitive.shadowing_db,
            primitive.fading_power_gain, parameters,
        )
        qos = communication_qos_slot(
            user.demand_bit_per_s, user.minimum_rate_bit_per_s,
            quality.served_rate_bit_per_s, user.normalized_shortfall_budget,
        )
        outcomes.append(CommunicationSlotOutcome(
            slot, user.user_id, quality.served_rate_bit_per_s,
            qos.normalized_shortfall, qos.residual,
        ))

    return tuple(sorted(outcomes, key=lambda item: _identifier_key(item.user_id)))


def _attainable_states(
    plans: tuple[SingleSessionPlan, ...],
    instance: OracleInstance,
    config: CanonicalConfig,
    slot: int,
) -> tuple[AggregateSensingState, ...]:
    total_bandwidth = _fraction(config.system["total_bandwidth_hz"])
    total_power = _fraction(config.system["total_power_w"])
    exact_states = {(Fraction(0), Fraction(0))}

    for plan in plans:
        accounting = plan.slot_accounting[slot]
        contribution = (
            _fraction(accounting.sensing_bandwidth_hz),
            _fraction(accounting.sensing_power_w),
        )
        for bandwidth, power in tuple(exact_states):
            candidate = bandwidth+contribution[0], power+contribution[1]
            if candidate[0] <= total_bandwidth and candidate[1] <= total_power:
                exact_states.add(candidate)

    values = []
    for bandwidth, power in sorted(exact_states):
        bandwidth_hz = float(bandwidth)
        power_w = float(power)
        values.append(AggregateSensingState(
            slot, bandwidth_hz, power_w,
            _communication_outcomes(instance, config, slot, bandwidth_hz, power_w),
        ))
    return tuple(values)


def build_offline_reference_milp(instance: OracleInstance, config: CanonicalConfig) -> OfflineMilpModel:
    """Enumerate every feasible plan and construct the exact binary plan-selection MILP."""
    plans = enumerate_single_session_plans(instance, config)
    states = tuple(
        _attainable_states(plans, instance, config, slot)
        for slot in range(instance.horizon_slots)
    )

    plan_indices = tuple(range(len(plans)))
    state_indices = []
    cursor = len(plans)
    for slot_states in states:
        indices = tuple(range(cursor, cursor+len(slot_states)))
        state_indices.append(indices)
        cursor += len(slot_states)

    state_index_values = tuple(state_indices)
    objective = np.zeros(cursor, dtype=float)
    for index, plan in enumerate(plans):
        objective[index] = -plan.objective

    rows: list[dict[int, float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    labels: list[str] = []

    def add_row(label: str, coefficients: dict[int, float], lower: float, upper: float) -> None:
        if label in labels:
            raise MilpValidationError(f"duplicate MILP row label: {label}")
        rows.append({index: float(value) for index, value in coefficients.items() if value != 0.0})
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))
        labels.append(label)

    for request in instance.requests:
        coefficients = {
            index: 1.0
            for index, plan in enumerate(plans)
            if request.request_id in plan.request_ids()
        }
        add_row(
            f"request_exclusivity:{_identifier_label(request.request_id)}",
            coefficients, -np.inf, 1.0,
        )

    for slot in range(instance.horizon_slots):
        coefficients = {
            index: float(sum(event.slot == slot for event in plan.admissions))
            for index, plan in enumerate(plans)
        }
        add_row(f"admission_capacity:{slot}", coefficients, -np.inf, 1.0)

    for slot in range(instance.horizon_slots):
        coefficients = {
            index: plan.slot_accounting[slot].sensing_bandwidth_hz
            for index, plan in enumerate(plans)
        }
        add_row(
            f"sensing_bandwidth:{slot}", coefficients,
            -np.inf, config.system["total_bandwidth_hz"],
        )

    for slot in range(instance.horizon_slots):
        coefficients = {
            index: plan.slot_accounting[slot].sensing_power_w
            for index, plan in enumerate(plans)
        }
        add_row(
            f"sensing_power:{slot}", coefficients,
            -np.inf, config.system["total_power_w"],
        )

    for slot, (slot_states, indices) in enumerate(zip(states, state_index_values, strict=True)):
        add_row(f"aggregate_state_one:{slot}", {index: 1.0 for index in indices}, 1.0, 1.0)

        bandwidth_link = {
            index: -plan.slot_accounting[slot].sensing_bandwidth_hz
            for index, plan in enumerate(plans)
        }
        bandwidth_link.update({
            index: state.bandwidth_hz
            for index, state in zip(indices, slot_states, strict=True)
        })
        add_row(f"aggregate_bandwidth_link:{slot}", bandwidth_link, 0.0, 0.0)

        power_link = {
            index: -plan.slot_accounting[slot].sensing_power_w
            for index, plan in enumerate(plans)
        }
        power_link.update({
            index: state.power_w
            for index, state in zip(indices, slot_states, strict=True)
        })
        add_row(f"aggregate_power_link:{slot}", power_link, 0.0, 0.0)

    for tenant in config.tenants:
        coefficients = {}
        for index, plan in enumerate(plans):
            outcomes = tuple(item for item in plan.request_outcomes if item.tenant_id == tenant.tenant_id)
            coefficients[index] = sum(
                float(item.sla_violated)-tenant.sla_violation_budget
                for item in outcomes
            )
        add_row(f"tenant_sla:{_identifier_label(tenant.tenant_id)}", coefficients, -np.inf, 0.0)

    user_ids = tuple(sorted(
        {item.user_id for item in instance.primitive_trace.communication_states},
        key=_identifier_key,
    ))
    for user_id in user_ids:
        coefficients = {}
        for slot, (slot_states, indices) in enumerate(zip(states, state_index_values, strict=True)):
            for index, state in zip(indices, slot_states, strict=True):
                coefficients[index] = state.communication_residual(user_id)
        add_row(f"communication_qos:{_identifier_label(user_id)}", coefficients, -np.inf, 0.0)

    matrix = lil_matrix((len(rows), cursor), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for variable_index, coefficient in coefficients.items():
            matrix[row_index, variable_index] = coefficient

    return OfflineMilpModel(
        instance, config, plans, states, objective,
        np.ones(cursor, dtype=np.uint8), np.zeros(cursor), np.ones(cursor),
        csc_array(matrix), np.asarray(lower_bounds), np.asarray(upper_bounds),
        tuple(labels), plan_indices, state_index_values,
    )


def _validate_solver_result(result: object, configured_gap: float, variable_count: int) -> np.ndarray:
    status = getattr(result, "status", None)
    success = getattr(result, "success", None)
    x = getattr(result, "x", None)
    fun = getattr(result, "fun", None)
    mip_gap = getattr(result, "mip_gap", None)

    if success is not True or status != 0:
        raise MilpSolveError(
            "MILP solver did not certify an acceptable optimum: "
            f"{getattr(result, 'message', '')}",
        )
    if x is None or fun is None or mip_gap is None:
        raise MilpSolveError("MILP solver omitted required primal or certification fields")

    values = np.asarray(x, dtype=float)
    if values.shape != (variable_count,) or not np.all(np.isfinite(values)) or not isfinite(float(fun)):
        raise MilpSolveError("MILP solver returned invalid finite-dimensional primal data")
    if not isfinite(float(mip_gap)) or float(mip_gap) > configured_gap+NUMERICAL_TOLERANCE:
        raise MilpSolveError("MILP solver did not meet the configured relative-gap tolerance")
    if np.any(np.abs(values-np.rint(values)) > NUMERICAL_TOLERANCE):
        raise MilpSolveError("MILP solver returned a non-integral binary solution")

    rounded = np.rint(values)
    if np.any((rounded < 0.0) | (rounded > 1.0)):
        raise MilpSolveError("MILP solver returned a binary variable outside [0, 1]")
    return rounded.astype(np.uint8)


def _certify_accounting(
    model: OfflineMilpModel,
    selected_plans: tuple[SingleSessionPlan, ...],
    selected_state_indices: tuple[int, ...],
    objective: float,
    configured_gap: float,
) -> JointSelectionAccounting:
    feasibility_tolerance = max(NUMERICAL_TOLERANCE, configured_gap)
    accounting = evaluate_joint_selection(
        selected_plans, model.instance, model.config, tolerance=feasibility_tolerance,
    )
    if accounting is None:
        raise MilpSolveError("solver selection failed joint-accounting reconstruction")

    objective_tolerance = feasibility_tolerance*max(
        1.0, abs(objective), abs(accounting.objective),
    )
    if abs(accounting.objective-objective) > objective_tolerance:
        raise MilpSolveError("solver objective does not match joint accounting")

    for slot, state_index in enumerate(selected_state_indices):
        state = model.aggregate_states[slot][state_index]
        slot_accounting = accounting.slot_accounting[slot]

        if not _close(state.bandwidth_hz, slot_accounting.sensing_bandwidth_hz):
            raise MilpSolveError("selected aggregate bandwidth state does not match selected plans")
        if not _close(state.power_w, slot_accounting.sensing_power_w):
            raise MilpSolveError("selected aggregate power state does not match selected plans")

        actual = {
            item.user_id: item
            for item in accounting.communication_outcomes
            if item.slot == slot
        }
        expected = {item.user_id: item for item in state.communication_outcomes}
        if actual.keys() != expected.keys():
            raise MilpSolveError("selected communication state has a different user set")

        for user_id in actual:
            if not _close(
                actual[user_id].served_rate_bit_per_s,
                expected[user_id].served_rate_bit_per_s,
            ):
                raise MilpSolveError("selected communication-state rate does not match joint accounting")
            if not _close(
                actual[user_id].normalized_shortfall,
                expected[user_id].normalized_shortfall,
            ):
                raise MilpSolveError(
                    "selected communication-state shortfall does not match joint accounting",
                )
            if not _close(actual[user_id].residual, expected[user_id].residual):
                raise MilpSolveError(
                    "selected communication-state residual does not match joint accounting",
                )

    if any(residual > feasibility_tolerance for _, residual in accounting.tenant_residuals):
        raise MilpSolveError("selected plans violate a per-tenant sensing-SLA constraint")
    if any(residual > feasibility_tolerance for _, residual in accounting.communication_residuals):
        raise MilpSolveError("selected plans violate a communication-QoS constraint")
    return accounting


def solve_offline_reference_milp(instance: OracleInstance, config: CanonicalConfig) -> OfflineMilpSolution:
    """Build, solve, and independently certify the complete small-trace offline reference."""
    total_start = perf_counter()
    model = build_offline_reference_milp(instance, config)
    model_build_time = perf_counter()-total_start
    configured_gap = float(config.oracle["solver_relative_gap"])
    if not isfinite(configured_gap) or configured_gap < 0.0:
        raise MilpValidationError("oracle solver_relative_gap must be finite and non-negative")

    constraints = LinearConstraint(
        model.constraint_matrix, model.constraint_lower_bounds, model.constraint_upper_bounds,
    )
    start = perf_counter()
    result = milp(
        model.objective_coefficients,
        integrality=model.integrality,
        bounds=Bounds(model.variable_lower_bounds, model.variable_upper_bounds),
        constraints=constraints,
        options={"mip_rel_gap": configured_gap, "presolve": True},
    )
    solver_wall_clock = perf_counter()-start
    binary = _validate_solver_result(result, configured_gap, model.variable_count)

    selected_plans = tuple(
        plan for index, plan in enumerate(model.plans) if binary[index] == 1
    )
    selected_state_indices = []
    for indices in model.state_variable_indices:
        selected = tuple(
            offset
            for offset, variable_index in enumerate(indices)
            if binary[variable_index] == 1
        )
        if len(selected) != 1:
            raise MilpSolveError(
                "MILP solution did not select exactly one aggregate state per slot",
            )
        selected_state_indices.append(selected[0])

    objective = -float(result.fun)
    accounting = _certify_accounting(
        model, selected_plans, tuple(selected_state_indices), objective, configured_gap,
    )

    dual_bound = getattr(result, "mip_dual_bound", None)
    node_count = getattr(result, "mip_node_count", None)
    if dual_bound is None or node_count is None or not isfinite(float(dual_bound)):
        raise MilpSolveError("MILP solver omitted required certification metadata")

    return OfflineMilpSolution(
        "SciPy milp (HiGHS)", scipy.__version__, configured_gap, int(result.status),
        str(result.message), bool(result.success), float(result.mip_gap),
        float(dual_bound), int(node_count), float(model_build_time),
        float(solver_wall_clock), float(perf_counter()-total_start), objective,
        tuple(plan.plan_id for plan in selected_plans), tuple(selected_state_indices),
        accounting, model.plan_count, model.variable_count, model.constraint_count,
        model.aggregate_state_count,
    )