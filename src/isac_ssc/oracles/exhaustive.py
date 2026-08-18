"""Exact static construction and exhaustive verification for small oracle instances."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from isac_ssc.core.compatibility import merge_authorized
from isac_ssc.core.entities import DiskAOI, EntityId, ResourceProfile, SensingRequest, Task, Tenant
from isac_ssc.oracles.reference import (
    JointSelectionAccounting, OracleInstance, ReferenceValidationError,
    SingleSessionPlan, evaluate_joint_selection, validate_reference_instance,
)
from isac_ssc.utils.config import CanonicalConfig


def _identifier_key(value: EntityId) -> tuple[str, str]:
    return type(value).__name__, str(value)


@dataclass(frozen=True, slots=True)
class StaticColoringInstance:
    vertex_count: int
    edges: frozenset[tuple[int, int]]
    tenants: tuple[Tenant, ...]
    requests: tuple[SensingRequest, ...]
    profile: ResourceProfile


@dataclass(frozen=True, slots=True)
class StaticPartitionResult:
    minimum_session_count: int
    partitions: tuple[tuple[tuple[EntityId, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class ExhaustiveReferenceResult:
    plan_count: int
    feasible_selection_count: int
    objective: float
    optimal_selections: tuple[JointSelectionAccounting, ...]


def build_restricted_coloring_instance(
    vertex_count: int, edges: Iterable[tuple[int, int]],
) -> StaticColoringInstance:
    """Construct the exact graph-coloring special case from the authoritative reduction."""
    if isinstance(vertex_count, bool) or not isinstance(vertex_count, int) or vertex_count <= 0:
        raise ReferenceValidationError("vertex_count must be a positive integer")

    normalized = set()
    for edge in edges:
        if len(edge) != 2:
            raise ReferenceValidationError("each graph edge must contain two vertices")
        left, right = edge
        if any(isinstance(value, bool) or not isinstance(value, int) for value in edge):
            raise ReferenceValidationError("graph vertices must be integers")
        if left == right or not (0 <= left < vertex_count and 0 <= right < vertex_count):
            raise ReferenceValidationError("graph edges must be loop-free and inside the vertex set")
        normalized.add((min(left, right), max(left, right)))

    edge_set = frozenset(normalized)
    authorization = tuple(
        tuple(
            left == right or (min(left, right), max(left, right)) not in edge_set
            for right in range(vertex_count)
        )
        for left in range(vertex_count)
    )
    tenants = tuple(
        Tenant(f"tenant_{vertex}", frozenset({Task.DETECTION}), 0.0, authorization[vertex])
        for vertex in range(vertex_count)
    )
    aoi = DiskAOI((0.0, 0.0), 1.0)
    requests = tuple(
        SensingRequest(
            f"request_{vertex}", tenants[vertex].tenant_id, 0, 0, aoi, "target",
            Task.DETECTION, 0.5, 1, 1.0, True,
        )
        for vertex in range(vertex_count)
    )
    return StaticColoringInstance(
        vertex_count, edge_set, tenants, requests, ResourceProfile("unit", 1.0, 1.0, 1),
    )


def authorization_matrix(instance: StaticColoringInstance) -> tuple[tuple[bool, ...], ...]:
    return tuple(tenant.authorization_row for tenant in instance.tenants)


def static_group_feasible(
    instance: StaticColoringInstance, request_ids: Iterable[EntityId],
) -> bool:
    identifiers = tuple(request_ids)
    if not identifiers:
        return False

    by_id = {request.request_id: request for request in instance.requests}
    if any(identifier not in by_id for identifier in identifiers):
        raise ReferenceValidationError("static group references an unknown request")

    requests = tuple(by_id[identifier] for identifier in identifiers)
    return len(requests) == 1 or merge_authorized(requests[0], requests[1:], instance.tenants)


def minimum_authorized_session_partition(
    instance: StaticColoringInstance,
) -> StaticPartitionResult:
    """Enumerate exact authorized partitions using only the constructed sharing contract."""
    request_ids = tuple(request.request_id for request in instance.requests)
    best_count = len(request_ids)+1
    best: set[tuple[tuple[EntityId, ...], ...]] = set()

    def canonical(groups: list[list[EntityId]]) -> tuple[tuple[EntityId, ...], ...]:
        normalized = [tuple(sorted(group, key=_identifier_key)) for group in groups]
        return tuple(sorted(
            normalized,
            key=lambda group: tuple(_identifier_key(value) for value in group),
        ))

    def assign(index: int, groups: list[list[EntityId]]) -> None:
        nonlocal best_count, best
        if len(groups) > best_count:
            return
        if index == len(request_ids):
            partition = canonical(groups)
            count = len(partition)
            if count < best_count:
                best_count = count
                best = {partition}
            elif count == best_count:
                best.add(partition)
            return

        request_id = request_ids[index]
        for group in groups:
            candidate = (*group, request_id)
            if static_group_feasible(instance, candidate):
                group.append(request_id)
                assign(index+1, groups)
                group.pop()

        groups.append([request_id])
        assign(index+1, groups)
        groups.pop()

    assign(0, [])
    return StaticPartitionResult(
        best_count,
        tuple(sorted(
            best,
            key=lambda partition: tuple(
                tuple(_identifier_key(value) for value in group)
                for group in partition
            ),
        )),
    )


def solve_exhaustive_reference(
    plans: Iterable[SingleSessionPlan],
    instance: OracleInstance,
    config: CanonicalConfig,
    *,
    objective_tolerance: float = 1.0e-9,
) -> ExhaustiveReferenceResult:
    """Enumerate every feasible joint plan selection and retain every optimal accounting record."""
    validate_reference_instance(instance, config)
    plan_values = tuple(sorted(plans, key=lambda item: item.plan_id))
    if len({plan.plan_id for plan in plan_values}) != len(plan_values):
        raise ReferenceValidationError("plan identifiers must be unique")

    feasible_count = 0
    best_objective = float("-inf")
    best: list[JointSelectionAccounting] = []

    for count in range(len(plan_values)+1):
        for selection in combinations(plan_values, count):
            accounting = evaluate_joint_selection(selection, instance, config)
            if accounting is None:
                continue

            feasible_count += 1
            if accounting.objective > best_objective+objective_tolerance:
                best_objective = accounting.objective
                best = [accounting]
            elif abs(accounting.objective-best_objective) <= objective_tolerance:
                best.append(accounting)

    if not best:
        raise ReferenceValidationError("the tiny reference instance has no feasible joint selection")

    return ExhaustiveReferenceResult(
        len(plan_values), feasible_count, best_objective,
        tuple(sorted(best, key=lambda item: item.selected_plan_ids)),
    )