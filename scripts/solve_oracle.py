"""Run a deterministic offline-reference smoke instance."""

from __future__ import annotations

import json
from math import isclose

from isac_ssc.core.entities import (
    CommunicationUser, DiskAOI,
    SensingRequest, Task,
)
from isac_ssc.envs.dynamics import (
    CommunicationSlotPrimitive, PrimitiveTrace,
    TargetSlotPrimitive,
)
from isac_ssc.oracles.exhaustive import solve_exhaustive_reference
from isac_ssc.oracles.milp import solve_offline_reference_milp
from isac_ssc.oracles.reference import (
    OracleInstance, enumerate_single_session_plans,
)
from isac_ssc.utils.config import load_config


def _smoke_instance() -> OracleInstance:
    config = load_config()
    horizon = 4
    target_id = "target_smoke"
    aoi = DiskAOI((80.0, 0.0), 30.0)
    requests = (
        SensingRequest(
            "request_1", "tenant_1", 0, 0,
            aoi, target_id, Task.DETECTION,
            0.1, 2, 1.0, True,
        ),
        SensingRequest(
            "request_2", "tenant_1", 1, 1,
            aoi, target_id, Task.DETECTION,
            0.1, 2, 1.0, True,
        ),
    )
    targets = tuple(
        TargetSlotPrimitive(
            slot, target_id,
            (80.0, 0.0), (0.0, 0.0),
            0.0, 1.0, 0.0, 0.0,
        )
        for slot in range(horizon)
    )
    user = CommunicationUser(
        "user_smoke",
        (50.0, 0.0), (0.0, 0.0),
        1.0e6, 2.0e6, 0.05,
    )
    communications = tuple(
        CommunicationSlotPrimitive(
            slot, user.user_id,
            user.position_m,
            user.velocity_m_per_s,
            True, user.demand_bit_per_s,
            0.0, 1.0, 0.0,
        )
        for slot in range(horizon)
    )
    trace = PrimitiveTrace(
        "oracle_smoke", 0, "independent", horizon,
        tuple(
            tenant.tenant_id
            for tenant in config.tenants
        ),
        tuple(
            tenant.authorization_row
            for tenant in config.tenants
        ),
        targets, (), communications, (), (), (),
    )
    return OracleInstance(
        trace, requests,
        (config.resource_profiles["balanced"],),
        (user,),
    )


def main() -> None:
    config = load_config()
    instance = _smoke_instance()
    plans = enumerate_single_session_plans(
        instance, config,
    )
    exhaustive = solve_exhaustive_reference(
        plans, instance, config,
    )
    solution = solve_offline_reference_milp(
        instance, config,
    )

    optimal_by_plan_ids = {
        item.selected_plan_ids: item
        for item in exhaustive.optimal_selections
    }
    matched_accounting = optimal_by_plan_ids.get(
        solution.selected_plan_ids,
    )
    objective_consistency = isclose(
        solution.objective,
        exhaustive.objective,
        rel_tol=solution.configured_relative_gap,
        abs_tol=1.0e-7,
    )
    selection_consistency = (
        matched_accounting is not None
    )
    raw_accounting_consistency = (
        matched_accounting == solution.accounting
    )

    if (
        not objective_consistency
        or not selection_consistency
        or not raw_accounting_consistency
    ):
        raise RuntimeError(
            "exhaustive and MILP smoke-reference results are inconsistent",
        )

    accounting = solution.accounting
    report = {
        "interpretation": (
            "finite-horizon offline clairvoyant "
            "constrained plan-selection reference"
        ),
        "interpretation_restriction": (
            "not a guaranteed upper bound "
            "or deployable online competitor"
        ),
        "information_relaxation": {
            "future_primitive_information_known": True,
            "deterministic_focal_request_queue_enforced": False,
            "explicit_repeated_defer_actions_consumed": False,
            "explicit_repeated_reject_actions_consumed": False,
            "future_admission_time_represents": "offline deferral",
            "omitted_request_represents": "rejection or non-service",
            "profile_changes_permitted_only_at": (
                "session creation or member admission"
            ),
        },
        "exhaustive_internal_verifier": (
            "not a second reported oracle"
        ),
        "exhaustive_verification": True,
        "exhaustive_feasible_selection_count": (
            exhaustive.feasible_selection_count
        ),
        "exhaustive_optimal_selection_count": len(
            exhaustive.optimal_selections
        ),
        "exhaustive_objective": exhaustive.objective,
        "milp_selection_in_exhaustive_optimal_set": (
            selection_consistency
        ),
        "objective_consistency": objective_consistency,
        "raw_accounting_consistency": raw_accounting_consistency,
        "trace_id": instance.trace_id,
        "horizon_slots": instance.horizon_slots,
        "request_count": len(instance.requests),
        "profile_count": len(instance.available_profiles),
        "plan_count": solution.plan_count,
        "variable_count": solution.variable_count,
        "constraint_count": solution.constraint_count,
        "aggregate_state_count": solution.aggregate_state_count,
        "solver": solution.solver,
        "scipy_version": solution.scipy_version,
        "configured_relative_gap": solution.configured_relative_gap,
        "status": solution.status,
        "message": solution.message,
        "success": solution.success,
        "mip_gap": solution.mip_gap,
        "mip_dual_bound": solution.mip_dual_bound,
        "mip_node_count": solution.mip_node_count,
        "model_build_time_s": solution.model_build_time_s,
        "solver_wall_clock_time_s": (
            solution.solver_wall_clock_time_s
        ),
        "wall_clock_time_s": solution.wall_clock_time_s,
        "objective": solution.objective,
        "selected_plan_ids": solution.selected_plan_ids,
        "request_assignments": accounting.request_assignments,
        "per_slot_sensing_resources": tuple({
            "slot": item.slot,
            "bandwidth_hz": item.sensing_bandwidth_hz,
            "power_w": item.sensing_power_w,
        } for item in accounting.slot_accounting),
        "tenant_residuals": accounting.tenant_residuals,
        "communication_residuals": (
            accounting.communication_residuals
        ),
    }
    print(json.dumps(
        report, indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()