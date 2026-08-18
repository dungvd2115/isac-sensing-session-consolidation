"""Hard action masks derived from current compatibility and resource feasibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from isac_ssc.core.compatibility import (
    CreateProfileAssessment, MergeProfileAssessment, SensingPrimitiveState,
    evaluate_create_profile, evaluate_merge_profile,
)
from isac_ssc.core.entities import (
    EntityId, Matrix4, RequestState, ResourceProfile, SensingRequest, SensingSession, Task,
)
from isac_ssc.core.quality import SensingParameters
from isac_ssc.envs.action_space import (
    ActionType, EnvironmentAction, build_action_catalogue, identifier_key,
)
from isac_ssc.envs.dynamics import TargetSlotPrimitive
from isac_ssc.utils.config import CanonicalConfig


def _profiles(config: CanonicalConfig) -> tuple[ResourceProfile, ...]:
    return tuple(sorted(config.resource_profiles.values(), key=lambda item: item.profile_id))


def _tracking_initial_covariance(config: CanonicalConfig) -> Matrix4:
    diagonal = tuple(float(value) for value in config.sensing["tracking"]["initial_covariance_diag"])
    return tuple(
        tuple(diagonal[row] if row == column else 0.0 for column in range(4))
        for row in range(4)
    )


def _by_typed_id(values, attribute: str):
    return {identifier_key(getattr(item, attribute)): item for item in values}


def _primitive(
    state: TargetSlotPrimitive, config: CanonicalConfig, tracking_prior: Matrix4 | None,
) -> SensingPrimitiveState:
    return SensingPrimitiveState(
        state.position_m, tuple(config.geometry["bs_position_m"]), state.rcs_m2,
        state.shadowing_db, state.fading_power_gain, tracking_prior,
    )


@dataclass(frozen=True, slots=True)
class SessionMergeAssessments:
    session_id: EntityId
    profile_assessments: tuple[MergeProfileAssessment, ...]


@dataclass(frozen=True, slots=True)
class RequestFeasibilityAssessment:
    request_id: EntityId
    create_assessments: tuple[CreateProfileAssessment, ...]
    session_merges: tuple[SessionMergeAssessments, ...]

    def create_for(self, profile_id: str) -> CreateProfileAssessment:
        return next(item for item in self.create_assessments if item.profile_id == profile_id)

    def merge_for(self, session_id: EntityId, profile_id: str) -> MergeProfileAssessment:
        key = identifier_key(session_id)
        session = next(
            item for item in self.session_merges
            if identifier_key(item.session_id) == key
        )
        return next(item for item in session.profile_assessments if item.profile_id == profile_id)


@dataclass(frozen=True, slots=True)
class CurrentFeasibilitySnapshot:
    current_slot: int
    prospective_session_id: EntityId
    profile_ids: tuple[str, ...]
    session_ids: tuple[EntityId, ...]
    request_assessments: tuple[RequestFeasibilityAssessment, ...]

    def request_for(self, request_id: EntityId) -> RequestFeasibilityAssessment:
        key = identifier_key(request_id)
        return next(
            item for item in self.request_assessments
            if identifier_key(item.request_id) == key
        )


def build_current_feasibility(
    current_slot: int, requests: Iterable[SensingRequest], sessions: Iterable[SensingSession],
    target_primitives: Iterable[TargetSlotPrimitive], prospective_session_id: EntityId,
    config: CanonicalConfig,
) -> CurrentFeasibilitySnapshot:
    """Evaluate every waiting-request CREATE and request-session MERGE candidate once."""
    request_values = tuple(sorted(requests, key=lambda item: identifier_key(item.request_id)))
    session_values = tuple(sorted(sessions, key=lambda item: identifier_key(item.session_id)))
    primitive_values = tuple(sorted(
        target_primitives, key=lambda item: identifier_key(item.target_id),
    ))
    request_index = _by_typed_id(request_values, "request_id")
    primitive_index = _by_typed_id(primitive_values, "target_id")
    profiles = _profiles(config)
    parameters = SensingParameters.from_config(config)
    tracking_initial = _tracking_initial_covariance(config)
    assessments = []

    for request in request_values:
        if request.state is not RequestState.WAITING:
            continue
        target = primitive_index[identifier_key(request.target_id)]
        create_primitive = _primitive(
            target, config, tracking_initial if request.task is Task.TRACKING else None,
        )
        creates = tuple(evaluate_create_profile(
            request, prospective_session_id, profile, create_primitive, parameters,
            session_values, config.service_duration_slots, current_slot,
            config.system["horizon_slots"], config.system["total_bandwidth_hz"],
            config.system["total_power_w"],
            tracking_initial if request.task is Task.TRACKING else None,
        ) for profile in profiles)

        merges = []
        for session in session_values:
            members = tuple(request_index[identifier_key(item)] for item in session.member_request_ids)
            primitive = _primitive(target, config, session.tracking_covariance)
            profile_assessments = tuple(evaluate_merge_profile(
                request, session, members, config.tenants, profile, primitive, parameters,
                session_values, config.service_duration_slots, current_slot,
                config.system["horizon_slots"],
                config.compatibility["minimum_spatial_coverage_ratio"],
                config.system["total_bandwidth_hz"], config.system["total_power_w"],
            ) for profile in profiles)
            merges.append(SessionMergeAssessments(session.session_id, profile_assessments))

        assessments.append(RequestFeasibilityAssessment(request.request_id, creates, tuple(merges)))

    return CurrentFeasibilitySnapshot(
        current_slot, prospective_session_id, tuple(item.profile_id for item in profiles),
        tuple(item.session_id for item in session_values), tuple(assessments),
    )


@dataclass(frozen=True, slots=True)
class MaskedActionEntry:
    action: EnvironmentAction
    feasible: bool
    merge_assessment: MergeProfileAssessment | None = None
    create_assessment: CreateProfileAssessment | None = None


@dataclass(frozen=True, slots=True)
class ActionMaskSnapshot:
    current_slot: int
    focal_request_id: EntityId
    prospective_session_id: EntityId
    entries: tuple[MaskedActionEntry, ...]
    feasibility: CurrentFeasibilitySnapshot

    def entry_for(self, action: EnvironmentAction) -> MaskedActionEntry:
        try:
            return next(item for item in self.entries if item.action == action)
        except StopIteration as error:
            raise KeyError(f"action is not in the current catalogue: {action!r}") from error

    @property
    def feasible_actions(self) -> tuple[EnvironmentAction, ...]:
        return tuple(item.action for item in self.entries if item.feasible)

    @property
    def action_type_mask(self) -> tuple[tuple[ActionType, bool], ...]:
        return tuple((kind, any(
            item.feasible and item.action.action_type is kind for item in self.entries
        )) for kind in ActionType)

    @property
    def merge_session_mask(self) -> tuple[tuple[EntityId, bool], ...]:
        return tuple((session_id, any(
            item.feasible
            and item.action.action_type is ActionType.MERGE
            and identifier_key(item.action.session_id) == identifier_key(session_id)
            for item in self.entries
        )) for session_id in self.feasibility.session_ids)

    @property
    def merge_profile_mask(
        self,
    ) -> tuple[tuple[EntityId, tuple[tuple[str, bool], ...]], ...]:
        return tuple((
            session_id,
            tuple((
                profile_id,
                self.entry_for(EnvironmentAction(
                    ActionType.MERGE, session_id, profile_id,
                )).feasible,
            ) for profile_id in self.feasibility.profile_ids),
        ) for session_id in self.feasibility.session_ids)

    @property
    def create_profile_mask(self) -> tuple[tuple[str, bool], ...]:
        return tuple((
            profile_id,
            self.entry_for(EnvironmentAction(
                ActionType.CREATE, profile_id=profile_id,
            )).feasible,
        ) for profile_id in self.feasibility.profile_ids)

    @property
    def defer_feasible(self) -> bool:
        return self.entry_for(EnvironmentAction(ActionType.DEFER)).feasible

    @property
    def reject_feasible(self) -> bool:
        return self.entry_for(EnvironmentAction(ActionType.REJECT)).feasible


def build_action_masks(
    focal_request: SensingRequest, requests: Iterable[SensingRequest],
    sessions: Iterable[SensingSession], feasibility: CurrentFeasibilitySnapshot,
    config: CanonicalConfig,
) -> ActionMaskSnapshot:
    """Build complete masks without ranking or reward information."""
    session_values = tuple(sorted(sessions, key=lambda item: identifier_key(item.session_id)))
    profiles = _profiles(config)
    focal = feasibility.request_for(focal_request.request_id)
    entries = []

    for action in build_action_catalogue(session_values, profiles):
        if action.action_type is ActionType.MERGE:
            assessment = focal.merge_for(action.session_id, action.profile_id)
            entries.append(MaskedActionEntry(
                action, assessment.feasible, merge_assessment=assessment,
            ))
        elif action.action_type is ActionType.CREATE:
            assessment = focal.create_for(action.profile_id)
            entries.append(MaskedActionEntry(
                action, assessment.feasible, create_assessment=assessment,
            ))
        elif action.action_type is ActionType.DEFER:
            feasible = (
                feasibility.current_slot + config.requests["defer_cooldown_slots"]
                <= focal_request.latest_start_slot
            )
            entries.append(MaskedActionEntry(action, feasible))
        else:
            entries.append(MaskedActionEntry(action, True))

    return ActionMaskSnapshot(
        feasibility.current_slot, focal_request.request_id,
        feasibility.prospective_session_id, tuple(entries), feasibility,
    )