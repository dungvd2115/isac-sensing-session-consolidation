"""Public actions for sensing-session consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from isac_ssc.core.entities import EntityId, ResourceProfile, SensingSession


class ActionValidationError(ValueError):
    """Raised when a public action is malformed."""


class ActionType(StrEnum):
    MERGE = "merge"
    CREATE = "create"
    DEFER = "defer"
    REJECT = "reject"


def identifier_key(value: EntityId) -> tuple[int, str]:
    """Order integer identifiers before string identifiers."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ActionValidationError("identifier must be a string or non-negative integer")
    if isinstance(value, int):
        if value < 0:
            raise ActionValidationError("integer identifier must be non-negative")
        return 0, str(value)
    if not value.strip():
        raise ActionValidationError("string identifier must not be empty")
    return 1, value


@dataclass(frozen=True, slots=True)
class EnvironmentAction:
    action_type: ActionType
    session_id: EntityId | None = None
    profile_id: str | None = None

    def __post_init__(self) -> None:
        try:
            action_type = ActionType(self.action_type)
        except (TypeError, ValueError) as error:
            raise ActionValidationError(f"unsupported action type: {self.action_type!r}") from error
        object.__setattr__(self, "action_type", action_type)
        if action_type is ActionType.MERGE:
            if self.session_id is None or not isinstance(self.profile_id, str) or not self.profile_id.strip():
                raise ActionValidationError("MERGE requires session_id and profile_id")
            identifier_key(self.session_id)
        elif action_type is ActionType.CREATE:
            if self.session_id is not None or not isinstance(self.profile_id, str) or not self.profile_id.strip():
                raise ActionValidationError("CREATE requires only profile_id")
        elif self.session_id is not None or self.profile_id is not None:
            raise ActionValidationError(f"{action_type.value.upper()} does not accept identifiers")


_ACTION_ORDER = {
    ActionType.MERGE: 0, ActionType.CREATE: 1, ActionType.DEFER: 2, ActionType.REJECT: 3,
}


def canonical_action_key(action: EnvironmentAction) -> tuple[int, tuple[int, str], str]:
    session = (-1, "") if action.session_id is None else identifier_key(action.session_id)
    return _ACTION_ORDER[action.action_type], session, action.profile_id or ""


def build_action_catalogue(
    sessions: Iterable[SensingSession], profiles: Iterable[ResourceProfile],
) -> tuple[EnvironmentAction, ...]:
    """Enumerate structural actions; feasibility is evaluated separately."""
    session_values = tuple(sorted(sessions, key=lambda item: identifier_key(item.session_id)))
    profile_values = tuple(sorted(profiles, key=lambda item: item.profile_id))
    actions = [
        EnvironmentAction(ActionType.MERGE, session.session_id, profile.profile_id)
        for session in session_values for profile in profile_values
    ]
    actions.extend(
        EnvironmentAction(ActionType.CREATE, profile_id=profile.profile_id)
        for profile in profile_values
    )
    actions.extend((EnvironmentAction(ActionType.DEFER), EnvironmentAction(ActionType.REJECT)))
    return tuple(actions)