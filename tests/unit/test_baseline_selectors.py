from __future__ import annotations

from dataclasses import fields, replace
from itertools import permutations

import pytest

from isac_ssc.baselines.greedy_incremental_cost import (
    select_action as select_incremental_cost,
)
from isac_ssc.baselines.no_consolidation import (
    select_action as select_no_consolidation,
)
from isac_ssc.baselines.selectors import (
    ImmediateServiceCandidate, absolute_profile_cost,
    build_create_candidate, build_merge_candidate,
    canonical_action_key,
)
from isac_ssc.baselines.sla_aware_greedy import (
    select_action as select_sla_aware,
)
from isac_ssc.baselines.static_compatibility_merge import (
    select_action as select_static_merge,
)
from isac_ssc.core.compatibility import (
    SensingPrimitiveState, evaluate_create_profile, evaluate_merge_profile,
)
from isac_ssc.core.entities import (
    DiskAOI, RequestState, SensingRequest, SensingSession, Task,
)
from isac_ssc.core.quality import SensingParameters
from isac_ssc.core.resources import (
    SensingResourceUsage, normalized_sensing_resource_cost,
)
from isac_ssc.envs.action_space import ActionType, EnvironmentAction
from isac_ssc.utils.config import load_config

CONFIG = load_config()
DURATIONS = CONFIG.service_duration_slots
PARAMETERS = SensingParameters.from_config(CONFIG)
TOTAL_BANDWIDTH = CONFIG.system["total_bandwidth_hz"]
TOTAL_POWER = CONFIG.system["total_power_w"]
BANDWIDTH_WEIGHT = CONFIG.reward["sensing_cost_bandwidth_weight"]
POWER_WEIGHT = CONFIG.reward["sensing_cost_power_weight"]
HORIZON = CONFIG.system["horizon_slots"]
COVERAGE = CONFIG.compatibility["minimum_spatial_coverage_ratio"]
AOI = DiskAOI((80.0, 0.0), 30.0)
PRIMITIVE = SensingPrimitiveState(
    (80.0, 0.0), (0.0, 0.0), 1.0, 0.0, 1.0, None,
)


def _request(
    request_id: int | str, *, threshold: float = 0.1,
    interval: int = 3, state: RequestState = RequestState.WAITING,
    arrival_slot: int = 1,
) -> SensingRequest:
    request = SensingRequest(
        request_id, "tenant_1", arrival_slot, 8, AOI, 7,
        Task.DETECTION, threshold, interval, 1.0, True,
    )
    return (
        request
        if state is RequestState.WAITING
        else request.transition(state, slot=arrival_slot)
    )


def _session(
    profile_name: str = "balanced", *, session_id: int | str = 1,
    threshold: float = 0.1, interval: int = 3,
) -> tuple[SensingSession, SensingRequest]:
    creator = _request(
        f"creator_{session_id}", threshold=threshold,
        interval=interval, state=RequestState.ACTIVE,
        arrival_slot=0,
    )
    return SensingSession.create(
        session_id, creator, CONFIG.resource_profiles[profile_name],
        0, DURATIONS,
    ), creator


def _candidate(
    kind: ActionType, *, session_id=None, profile_id="balanced",
    absolute=0.2, incremental=0.1, margin=0.5,
) -> ImmediateServiceCandidate:
    return ImmediateServiceCandidate(
        EnvironmentAction(kind, session_id, profile_id),
        absolute, incremental, margin,
    )


def test_candidate_contains_only_current_selector_coefficients() -> None:
    assert tuple(field.name for field in fields(ImmediateServiceCandidate)) == (
        "action", "absolute_profile_cost",
        "incremental_current_cost", "normalized_sla_margin",
    )
    candidate = _candidate(ActionType.CREATE)
    assert not hasattr(candidate, "request_value")
    assert not hasattr(candidate, "future_primitives")


def test_canonical_action_key_uses_public_action_order_and_typed_ids() -> None:
    actions = (
        EnvironmentAction(ActionType.REJECT),
        EnvironmentAction(ActionType.DEFER),
        EnvironmentAction(ActionType.CREATE, profile_id="balanced"),
        EnvironmentAction(ActionType.MERGE, "1", "balanced"),
        EnvironmentAction(ActionType.MERGE, 1, "balanced"),
    )
    ordered = tuple(sorted(actions, key=canonical_action_key))
    assert tuple(action.action_type for action in ordered) == (
        ActionType.MERGE, ActionType.MERGE, ActionType.CREATE,
        ActionType.DEFER, ActionType.REJECT,
    )
    assert ordered[0].session_id == 1
    assert ordered[1].session_id == "1"


def test_absolute_profile_cost_uses_canonical_resource_function() -> None:
    profile = CONFIG.resource_profiles["balanced"]
    expected = normalized_sensing_resource_cost(
        SensingResourceUsage(
            profile.sensing_bandwidth_hz, profile.sensing_power_w,
        ),
        TOTAL_BANDWIDTH, TOTAL_POWER,
        BANDWIDTH_WEIGHT, POWER_WEIGHT,
    )
    assert absolute_profile_cost(
        profile, TOTAL_BANDWIDTH, TOTAL_POWER,
        BANDWIDTH_WEIGHT, POWER_WEIGHT,
    ) == pytest.approx(expected)


def test_create_candidate_uses_retained_assessment_cost_and_margin() -> None:
    request = _request(1, threshold=0.2, interval=3)
    profile = CONFIG.resource_profiles["balanced"]
    assessment = evaluate_create_profile(
        request, "new_session", profile, PRIMITIVE, PARAMETERS,
        (), DURATIONS, 1, HORIZON, TOTAL_BANDWIDTH, TOTAL_POWER,
    )
    candidate = build_create_candidate(
        request, profile, assessment, (), 1,
        TOTAL_BANDWIDTH, TOTAL_POWER,
        BANDWIDTH_WEIGHT, POWER_WEIGHT,
    )
    quality = assessment.quality_margin.margin/request.quality_threshold
    periodicity = (
        request.valid_output_interval_slots - (profile.update_period_slots-1)
    )/request.valid_output_interval_slots
    assert candidate.action == EnvironmentAction(
        ActionType.CREATE, profile_id="balanced",
    )
    assert candidate.incremental_current_cost == pytest.approx(
        candidate.absolute_profile_cost,
    )
    assert candidate.normalized_sla_margin == pytest.approx(
        min(quality, periodicity),
    )


def test_merge_candidate_uses_signed_cost_and_worst_affected_margin() -> None:
    session, creator = _session(
        "balanced", threshold=0.2, interval=3,
    )
    session = replace(session, next_update_slot=1)
    focal = _request(2, threshold=0.3, interval=3)
    profile = CONFIG.resource_profiles["economical"]
    assessment = evaluate_merge_profile(
        focal, session, (creator,), CONFIG.tenants,
        profile, PRIMITIVE, PARAMETERS, (session,),
        DURATIONS, 1, HORIZON, COVERAGE,
        TOTAL_BANDWIDTH, TOTAL_POWER,
    )
    candidate = build_merge_candidate(
        focal, (creator,), session, profile, assessment,
        (session,), 1, TOTAL_BANDWIDTH, TOTAL_POWER,
        BANDWIDTH_WEIGHT, POWER_WEIGHT,
    )
    focal_quality = (
        assessment.focal_margin.margin/focal.quality_threshold
    )
    member_quality = (
        assessment.member_margins[0][1].margin/creator.quality_threshold
    )
    focal_period = (
        focal.valid_output_interval_slots -
        (profile.update_period_slots-1)
    )/3
    member_period = (
        creator.valid_output_interval_slots -
        (profile.update_period_slots-1)
    )/3
    assert candidate.incremental_current_cost < 0.0
    assert candidate.normalized_sla_margin == pytest.approx(min(
        focal_quality, member_quality,
        focal_period, member_period,
    ))


def test_no_consolidation_never_merges_and_uses_minimum_absolute_cost() -> None:
    merge = _candidate(
        ActionType.MERGE, session_id=1,
        profile_id="economical", absolute=0.01,
    )
    expensive = _candidate(
        ActionType.CREATE, profile_id="precision", absolute=0.4,
    )
    cheap = _candidate(
        ActionType.CREATE, profile_id="balanced", absolute=0.2,
    )
    assert select_no_consolidation(
        (merge, expensive, cheap), defer_feasible=True,
    ) == cheap.action


def test_no_consolidation_uses_defer_then_reject_when_create_is_absent() -> None:
    merge = _candidate(
        ActionType.MERGE, session_id=1, absolute=0.01,
    )
    assert select_no_consolidation(
        (merge,), defer_feasible=True,
    ) == EnvironmentAction(ActionType.DEFER)
    assert select_no_consolidation(
        (merge,), defer_feasible=False,
    ) == EnvironmentAction(ActionType.REJECT)


def test_static_merge_prioritizes_merge_then_minimum_absolute_cost() -> None:
    create = _candidate(
        ActionType.CREATE, profile_id="economical", absolute=0.01,
    )
    expensive = _candidate(
        ActionType.MERGE, session_id=2,
        profile_id="precision", absolute=0.4,
    )
    cheap = _candidate(
        ActionType.MERGE, session_id=3,
        profile_id="balanced", absolute=0.2,
    )
    assert select_static_merge(
        (create, expensive, cheap), defer_feasible=True,
    ) == cheap.action


def test_incremental_cost_preserves_signed_current_delta() -> None:
    create = _candidate(ActionType.CREATE, incremental=0.0)
    merge = _candidate(
        ActionType.MERGE, session_id=1, incremental=-0.05,
    )
    assert select_incremental_cost(
        (create, merge), defer_feasible=True,
    ) == merge.action


def test_sla_aware_uses_margin_then_incremental_cost_then_action_id() -> None:
    lower = _candidate(
        ActionType.MERGE, session_id=1,
        incremental=-1.0, margin=0.4,
    )
    expensive = _candidate(
        ActionType.CREATE, profile_id="precision",
        incremental=0.3, margin=0.8,
    )
    cheap = _candidate(
        ActionType.CREATE, profile_id="balanced",
        incremental=0.1, margin=0.8,
    )
    assert select_sla_aware(
        (lower, expensive, cheap), defer_feasible=True,
    ) == cheap.action
    tied_merge = _candidate(
        ActionType.MERGE, session_id=1,
        incremental=0.1, margin=0.8,
    )
    assert select_sla_aware(
        (cheap, tied_merge), defer_feasible=False,
    ) == tied_merge.action


@pytest.mark.parametrize("selector", (
    select_no_consolidation, select_static_merge,
    select_incremental_cost, select_sla_aware,
))
def test_all_selectors_use_defer_then_reject_for_empty_service_set(selector) -> None:
    assert selector(
        (), defer_feasible=True,
    ) == EnvironmentAction(ActionType.DEFER)
    assert selector(
        (), defer_feasible=False,
    ) == EnvironmentAction(ActionType.REJECT)


@pytest.mark.parametrize("selector", (
    select_no_consolidation, select_static_merge,
    select_incremental_cost, select_sla_aware,
))
def test_selector_output_is_deterministic_under_candidate_permutation(selector) -> None:
    candidates = (
        _candidate(
            ActionType.CREATE, profile_id="balanced",
            absolute=0.2, incremental=0.2, margin=0.7,
        ),
        _candidate(
            ActionType.CREATE, profile_id="precision",
            absolute=0.4, incremental=0.4, margin=0.9,
        ),
        _candidate(
            ActionType.MERGE, session_id=2,
            profile_id="balanced", absolute=0.2,
            incremental=0.05, margin=0.8,
        ),
    )
    outputs = {
        selector(ordering, defer_feasible=True)
        for ordering in permutations(candidates)
    }
    assert len(outputs) == 1