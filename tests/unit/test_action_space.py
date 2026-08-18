from __future__ import annotations

import pytest

from isac_ssc.core.entities import (
    DiskAOI, RequestState, SensingRequest, SensingSession, Task,
)
from isac_ssc.envs.action_space import (
    ActionType, ActionValidationError, EnvironmentAction,
    build_action_catalogue, canonical_action_key,
)
from isac_ssc.utils.config import load_config

CONFIG = load_config()
DURATIONS = CONFIG.service_duration_slots
AOI = DiskAOI((80.0, 0.0), 30.0)
PRIOR = tuple(tuple(
    25.0 if row == column < 2 else 4.0 if row == column else 0.0
    for column in range(4)
) for row in range(4))


def _session(session_id):
    creator = SensingRequest(
        f"creator_{session_id}", "tenant_1", 0, 8, AOI, 7,
        Task.TRACKING, 100.0, 2, 3.0, True,
    ).transition(RequestState.ACTIVE, slot=0)
    return SensingSession.create(
        session_id, creator, CONFIG.resource_profiles["balanced"],
        0, DURATIONS, PRIOR,
    )


def test_public_action_type_is_exactly_four_agent_actions() -> None:
    assert tuple(ActionType) == (
        ActionType.MERGE, ActionType.CREATE,
        ActionType.DEFER, ActionType.REJECT,
    )


@pytest.mark.parametrize("arguments", (
    (ActionType.MERGE, None, "balanced"),
    (ActionType.MERGE, 1, None),
    (ActionType.CREATE, 1, "balanced"),
    (ActionType.CREATE, None, None),
    (ActionType.DEFER, 1, None),
    (ActionType.REJECT, None, "balanced"),
))
def test_public_action_boundary_rejects_invalid_field_combinations(arguments) -> None:
    with pytest.raises(ActionValidationError):
        EnvironmentAction(*arguments)


def test_action_catalogue_is_complete_and_canonically_ordered_for_typed_ids() -> None:
    sessions = (_session("1"), _session(1))
    profiles = tuple(reversed(tuple(CONFIG.resource_profiles.values())))
    catalogue = build_action_catalogue(sessions, profiles)
    assert len(catalogue) == 2*4+4+2
    assert catalogue == tuple(sorted(catalogue, key=canonical_action_key))
    merge_ids = tuple(
        item.session_id
        for item in catalogue
        if item.action_type is ActionType.MERGE
    )
    assert merge_ids[:4] == (1, 1, 1, 1)
    assert merge_ids[4:] == ("1", "1", "1", "1")
    assert tuple(
        item.profile_id
        for item in catalogue
        if item.action_type is ActionType.CREATE
    ) == ("balanced", "economical", "precision", "rapid")


def test_catalogue_is_independent_of_input_order() -> None:
    sessions = (_session(2), _session(1))
    profiles = tuple(CONFIG.resource_profiles.values())
    assert build_action_catalogue(sessions, profiles) == build_action_catalogue(
        reversed(sessions), reversed(profiles),
    )