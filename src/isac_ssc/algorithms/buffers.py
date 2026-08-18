"""Immutable decision-to-decision rollout storage for constrained PPO."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import torch

from isac_ssc.core.entities import EntityId
from isac_ssc.envs.action_space import ActionType, identifier_key
from isac_ssc.envs.observation import ObservationSnapshot
from isac_ssc.models.policy import FactorizedActionIndices
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.models.value import ValueOutput


class BufferValidationError(ValueError):
    """Raised when rollout storage violates the locked public-data contract."""


_ACTION_TYPES = (ActionType.MERGE, ActionType.CREATE, ActionType.DEFER, ActionType.REJECT)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise BufferValidationError(f"{name} must be finite")
    return float(value)


def _float_tuple(values: Iterable[object], name: str, expected: int | None = None) -> tuple[float, ...]:
    result = tuple(_finite(value, name) for value in values)
    if expected is not None and len(result) != expected:
        raise BufferValidationError(f"{name} must contain {expected} values")
    return result


@dataclass(frozen=True, slots=True)
class ConstraintLayout:
    tenant_ids: tuple[EntityId, ...]
    communication_user_ids: tuple[EntityId, ...]

    def __post_init__(self) -> None:
        for name in ("tenant_ids", "communication_user_ids"):
            values = tuple(getattr(self, name))
            keys = tuple(identifier_key(value) for value in values)
            if len(set(keys)) != len(keys):
                raise BufferValidationError(f"{name} must contain unique typed identifiers")
            if keys != tuple(sorted(keys)):
                raise BufferValidationError(f"{name} must use canonical typed-identifier order")
            object.__setattr__(self, name, values)

    @property
    def tenant_count(self) -> int:
        return len(self.tenant_ids)

    @property
    def communication_count(self) -> int:
        return len(self.communication_user_ids)

    @property
    def constraint_count(self) -> int:
        return self.tenant_count + self.communication_count

    def _pack(self, pairs: Iterable[tuple[EntityId, float]], expected: tuple[EntityId, ...], name: str) -> torch.Tensor:
        values = tuple(pairs)
        keys = tuple(identifier_key(item) for item, _ in values)
        if len(set(keys)) != len(keys):
            raise BufferValidationError(f"{name} contains duplicate typed identifiers")
        expected_keys = tuple(identifier_key(item) for item in expected)
        if set(keys) != set(expected_keys):
            raise BufferValidationError(f"{name} identifiers do not match ConstraintLayout")
        mapping = {identifier_key(item): _finite(value, f"{name} residual") for item, value in values}
        return torch.tensor([mapping[key] for key in expected_keys], dtype=torch.float32)

    def pack_tenant_residuals(self, pairs: Iterable[tuple[EntityId, float]]) -> torch.Tensor:
        return self._pack(pairs, self.tenant_ids, "tenant residuals")

    def pack_communication_residuals(self, pairs: Iterable[tuple[EntityId, float]]) -> torch.Tensor:
        return self._pack(pairs, self.communication_user_ids, "communication residuals")

    def pack_residuals(self, tenant_pairs: Iterable[tuple[EntityId, float]],
                       communication_pairs: Iterable[tuple[EntityId, float]]) -> torch.Tensor:
        return torch.cat((self.pack_tenant_residuals(tenant_pairs), self.pack_communication_residuals(communication_pairs)))


@dataclass(frozen=True, slots=True)
class StoredAction:
    action_type_index: int
    merge_session_index: int = -1
    profile_index: int = -1

    def __post_init__(self) -> None:
        for name in ("action_type_index", "merge_session_index", "profile_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise BufferValidationError(f"{name} must be an integer")
        if self.action_type_index not in range(4):
            raise BufferValidationError("action_type_index must lie in [0, 3]")
        action_type = _ACTION_TYPES[self.action_type_index]
        if action_type is ActionType.MERGE and (self.merge_session_index < 0 or self.profile_index < 0):
            raise BufferValidationError("MERGE requires session and profile indices")
        if action_type is ActionType.CREATE and (self.merge_session_index != -1 or self.profile_index < 0):
            raise BufferValidationError("CREATE forbids a session index and requires a profile index")
        if action_type in (ActionType.DEFER, ActionType.REJECT) and (self.merge_session_index != -1 or self.profile_index != -1):
            raise BufferValidationError("DEFER and REJECT forbid conditional indices")

    @classmethod
    def from_indices(cls, indices: FactorizedActionIndices, index: int = 0) -> StoredAction:
        if not isinstance(indices, FactorizedActionIndices) or isinstance(index, bool) or not isinstance(index, int):
            raise BufferValidationError("indices and integer batch index are required")
        if index < 0 or index >= indices.action_type.shape[0]:
            raise BufferValidationError("factorized-action batch index is out of range")
        return cls(int(indices.action_type[index]), int(indices.merge_session[index]), int(indices.profile[index]))

    def to_indices(self, device: str | torch.device = "cpu") -> FactorizedActionIndices:
        target = torch.device(device)
        return FactorizedActionIndices(
            torch.tensor([self.action_type_index], dtype=torch.int64, device=target),
            torch.tensor([self.merge_session_index], dtype=torch.int64, device=target),
            torch.tensor([self.profile_index], dtype=torch.int64, device=target),
        )

    def validate_for_observation(self, observation: ObservationSnapshot) -> None:
        if not isinstance(observation, ObservationSnapshot):
            raise BufferValidationError("observation must be an ObservationSnapshot")
        masks = observation.set_view.action_masks
        if tuple(action_type for action_type, _ in masks.action_type_mask) != _ACTION_TYPES:
            raise BufferValidationError("public action-type order does not match StoredAction")
        action_mask = tuple(value for _, value in masks.action_type_mask)
        if not action_mask[self.action_type_index]:
            raise BufferValidationError("stored action type is infeasible")
        action_type = _ACTION_TYPES[self.action_type_index]
        if action_type is ActionType.MERGE:
            sessions = masks.merge_session_mask
            if self.merge_session_index >= len(sessions) or not sessions[self.merge_session_index][1]:
                raise BufferValidationError("stored merge-session index is infeasible")
            profiles = masks.merge_profile_mask[self.merge_session_index][1]
            if self.profile_index >= len(profiles) or not profiles[self.profile_index][1]:
                raise BufferValidationError("stored merge-profile index is infeasible")
        elif action_type is ActionType.CREATE:
            profiles = masks.create_profile_mask
            if self.profile_index >= len(profiles) or not profiles[self.profile_index][1]:
                raise BufferValidationError("stored create-profile index is infeasible")


@dataclass(frozen=True, slots=True)
class FactorCreditTransition:
    old_action_type_log_probability: float
    old_merge_session_log_probability: float
    old_profile_log_probability: float
    old_action_type_reward_value: float
    old_action_type_constraint_values: tuple[float, ...]
    old_merge_session_reward_value: float
    old_merge_session_constraint_values: tuple[float, ...]
    merge_session_applicable: bool
    profile_applicable: bool

    def __post_init__(self) -> None:
        for name in (
            "old_action_type_log_probability", "old_merge_session_log_probability",
            "old_profile_log_probability", "old_action_type_reward_value",
            "old_merge_session_reward_value",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self, "old_action_type_constraint_values",
            _float_tuple(
                self.old_action_type_constraint_values,
                "old action-type constraint value",
            ),
        )
        object.__setattr__(
            self, "old_merge_session_constraint_values",
            _float_tuple(
                self.old_merge_session_constraint_values,
                "old merge-session constraint value",
            ),
        )
        if type(self.merge_session_applicable) is not bool or type(self.profile_applicable) is not bool:
            raise BufferValidationError("factor applicability values must be bool")

    def validate_for_observation(
        self, action: StoredAction, observation: ObservationSnapshot,
    ) -> None:
        action_type = _ACTION_TYPES[action.action_type_index]
        masks = observation.set_view.action_masks
        expected_session = False
        expected_profile = False
        if action_type is ActionType.MERGE:
            expected_session = sum(value for _, value in masks.merge_session_mask) > 1
            expected_profile = sum(
                value for _, value
                in masks.merge_profile_mask[action.merge_session_index][1]
            ) > 1
        elif action_type is ActionType.CREATE:
            expected_profile = sum(value for _, value in masks.create_profile_mask) > 1
        if (
            self.merge_session_applicable != expected_session
            or self.profile_applicable != expected_profile
        ):
            raise BufferValidationError(
                "factor applicability does not match the selected feasible branch"
            )

    def validate(self, action: StoredAction, layout: ConstraintLayout) -> None:
        if len(self.old_action_type_constraint_values) != layout.constraint_count:
            raise BufferValidationError("action-type prefix values do not match ConstraintLayout")
        if len(self.old_merge_session_constraint_values) != layout.constraint_count:
            raise BufferValidationError("merge-session prefix values do not match ConstraintLayout")
        action_type = _ACTION_TYPES[action.action_type_index]
        if self.merge_session_applicable and action_type is not ActionType.MERGE:
            raise BufferValidationError("session actor factor requires selected MERGE")
        if self.profile_applicable and action_type not in {ActionType.MERGE, ActionType.CREATE}:
            raise BufferValidationError("profile actor factor requires selected MERGE or CREATE")
        if not self.merge_session_applicable and self.old_merge_session_log_probability != 0.0:
            raise BufferValidationError("inapplicable session log-probability must be zero")
        if not self.profile_applicable and self.old_profile_log_probability != 0.0:
            raise BufferValidationError("inapplicable profile log-probability must be zero")
        if action_type is not ActionType.MERGE:
            values = (
                self.old_merge_session_reward_value,
                *self.old_merge_session_constraint_values,
            )
            if any(value != 0.0 for value in values):
                raise BufferValidationError(
                    "unselected MERGE-session prefix values must be zero"
                )


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    observation: ObservationSnapshot
    action: StoredAction
    reward: float
    tenant_residuals: tuple[float, ...]
    communication_residuals: tuple[float, ...]
    next_observation: ObservationSnapshot | None
    terminated: bool
    physical_slot_span: int
    old_log_probability: float
    old_reward_value: float
    old_tenant_values: tuple[float, ...]
    old_communication_values: tuple[float, ...]
    factor_credit: FactorCreditTransition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationSnapshot) or not isinstance(self.action, StoredAction):
            raise BufferValidationError("transition requires public observation and StoredAction")
        if self.next_observation is not None and not isinstance(self.next_observation, ObservationSnapshot):
            raise BufferValidationError("next_observation must be ObservationSnapshot or None")
        if type(self.terminated) is not bool or self.terminated != (self.next_observation is None):
            raise BufferValidationError("terminated must be true exactly when next_observation is absent")
        if isinstance(self.physical_slot_span, bool) or not isinstance(self.physical_slot_span, int) or self.physical_slot_span < 1:
            raise BufferValidationError("physical_slot_span must be a positive integer")
        object.__setattr__(self, "reward", _finite(self.reward, "reward"))
        object.__setattr__(
            self, "old_log_probability",
            _finite(self.old_log_probability, "old_log_probability"),
        )
        object.__setattr__(
            self, "old_reward_value",
            _finite(self.old_reward_value, "old_reward_value"),
        )
        object.__setattr__(
            self, "tenant_residuals",
            _float_tuple(self.tenant_residuals, "tenant residual"),
        )
        object.__setattr__(
            self, "communication_residuals",
            _float_tuple(self.communication_residuals, "communication residual"),
        )
        object.__setattr__(
            self, "old_tenant_values",
            _float_tuple(self.old_tenant_values, "old tenant value"),
        )
        object.__setattr__(
            self, "old_communication_values",
            _float_tuple(self.old_communication_values, "old communication value"),
        )
        if self.factor_credit is not None and not isinstance(
            self.factor_credit, FactorCreditTransition,
        ):
            raise BufferValidationError(
                "factor_credit must be FactorCreditTransition or None"
            )
        self.action.validate_for_observation(self.observation)
        if self.factor_credit is not None:
            self.factor_credit.validate_for_observation(self.action, self.observation)

    def validate_layout(self, layout: ConstraintLayout) -> None:
        if (
            len(self.tenant_residuals) != layout.tenant_count
            or len(self.old_tenant_values) != layout.tenant_count
        ):
            raise BufferValidationError(
                "tenant transition dimensions do not match ConstraintLayout"
            )
        if (
            len(self.communication_residuals) != layout.communication_count
            or len(self.old_communication_values) != layout.communication_count
        ):
            raise BufferValidationError(
                "communication transition dimensions do not match ConstraintLayout"
            )
        if self.factor_credit is not None:
            self.factor_credit.validate(self.action, layout)


@dataclass(frozen=True, slots=True)
class EpisodeTotals:
    physical_slot_count: int
    reward_total: float
    tenant_residual_totals: tuple[float, ...]
    communication_residual_totals: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.physical_slot_count, bool) or not isinstance(self.physical_slot_count, int) or self.physical_slot_count < 1:
            raise BufferValidationError("physical_slot_count must be a positive integer")
        object.__setattr__(self, "reward_total", _finite(self.reward_total, "episode reward total"))
        object.__setattr__(self, "tenant_residual_totals", _float_tuple(self.tenant_residual_totals, "episode tenant residual"))
        object.__setattr__(self, "communication_residual_totals",
                           _float_tuple(self.communication_residual_totals, "episode communication residual"))

    def validate_layout(self, layout: ConstraintLayout) -> None:
        if len(self.tenant_residual_totals) != layout.tenant_count or len(self.communication_residual_totals) != layout.communication_count:
            raise BufferValidationError("episode totals do not match ConstraintLayout")


@dataclass(frozen=True, slots=True)
class FactorCreditMinibatch:
    old_log_probabilities: torch.Tensor
    applicability: torch.Tensor
    old_action_type_reward_values: torch.Tensor
    old_action_type_constraint_values: torch.Tensor
    old_merge_session_reward_values: torch.Tensor
    old_merge_session_constraint_values: torch.Tensor
    normalized_reward_advantages: torch.Tensor
    normalized_constraint_advantages: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreparedFactorCredit:
    old_log_probabilities: torch.Tensor
    applicability: torch.Tensor
    old_action_type_reward_values: torch.Tensor
    old_action_type_constraint_values: torch.Tensor
    old_merge_session_reward_values: torch.Tensor
    old_merge_session_constraint_values: torch.Tensor
    reward_advantages: torch.Tensor
    constraint_advantages: torch.Tensor
    normalized_reward_advantages: torch.Tensor
    normalized_constraint_advantages: torch.Tensor
    normalization_epsilon: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "normalization_epsilon",
            _finite(self.normalization_epsilon, "normalization_epsilon"),
        )
        if self.normalization_epsilon <= 0.0:
            raise BufferValidationError("normalization_epsilon must be positive")
        n = (
            self.old_log_probabilities.shape[0]
            if isinstance(self.old_log_probabilities, torch.Tensor)
            and self.old_log_probabilities.ndim == 2
            else -1
        )
        q = (
            self.old_action_type_constraint_values.shape[1]
            if isinstance(self.old_action_type_constraint_values, torch.Tensor)
            and self.old_action_type_constraint_values.ndim == 2
            else -1
        )
        specifications = (
            (self.old_log_probabilities, torch.float32, (n, 3)),
            (self.applicability, torch.bool, (n, 3)),
            (self.old_action_type_reward_values, torch.float32, (n,)),
            (self.old_action_type_constraint_values, torch.float32, (n, q)),
            (self.old_merge_session_reward_values, torch.float32, (n,)),
            (self.old_merge_session_constraint_values, torch.float32, (n, q)),
            (self.reward_advantages, torch.float32, (n, 3)),
            (self.constraint_advantages, torch.float32, (n, 3, q)),
            (self.normalized_reward_advantages, torch.float32, (n, 3)),
            (self.normalized_constraint_advantages, torch.float32, (n, 3, q)),
        )
        if n < 0 or q < 0:
            raise BufferValidationError("PreparedFactorCredit dimensions are invalid")
        for tensor, dtype, shape in specifications:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype is not dtype
                or tuple(tensor.shape) != shape
            ):
                raise BufferValidationError(
                    "PreparedFactorCredit tensor contract is invalid"
                )
            if tensor.device.type != "cpu" or tensor.requires_grad:
                raise BufferValidationError(
                    "PreparedFactorCredit tensors must be detached CPU tensors"
                )
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise BufferValidationError(
                    "PreparedFactorCredit tensors must be finite"
                )

        if n and not bool(self.applicability[:, 0].all()):
            raise BufferValidationError("action-type factor must apply to every transition")

        session_inactive = ~self.applicability[:, 1]
        profile_inactive = ~self.applicability[:, 2]
        if bool(
            self.old_log_probabilities[:, 1]
            .masked_select(session_inactive)
            .ne(0.0)
            .any()
        ):
            raise BufferValidationError(
                "inapplicable session log-probabilities must be zero"
            )
        if bool(
            self.old_log_probabilities[:, 2]
            .masked_select(profile_inactive)
            .ne(0.0)
            .any()
        ):
            raise BufferValidationError(
                "inapplicable profile log-probabilities must be zero"
            )

        if bool(
            self.reward_advantages
            .masked_select(~self.applicability)
            .ne(0.0)
            .any()
        ):
            raise BufferValidationError(
                "inapplicable reward factor advantages must be zero"
            )
        if bool(
            self.constraint_advantages
            .masked_select(~self.applicability.unsqueeze(2))
            .ne(0.0)
            .any()
        ):
            raise BufferValidationError(
                "inapplicable constraint factor advantages must be zero"
            )
        if bool(
            self.normalized_reward_advantages
            .masked_select(~self.applicability)
            .ne(0.0)
            .any()
        ):
            raise BufferValidationError(
                "inapplicable normalized reward advantages must be zero"
            )
        if bool(
            self.normalized_constraint_advantages
            .masked_select(~self.applicability.unsqueeze(2))
            .ne(0.0)
            .any()
        ):
            raise BufferValidationError(
                "inapplicable normalized constraint advantages must be zero"
            )

    @classmethod
    def empty(
        cls, constraint_count: int, normalization_epsilon: float,
    ) -> PreparedFactorCredit:
        if (
            isinstance(constraint_count, bool)
            or not isinstance(constraint_count, int)
            or constraint_count < 0
        ):
            raise BufferValidationError(
                "constraint_count must be a non-negative integer"
            )
        floats = torch.empty((0, 3), dtype=torch.float32)
        constraints = torch.empty(
            (0, 3, constraint_count), dtype=torch.float32,
        )
        return cls(
            floats,
            torch.empty((0, 3), dtype=torch.bool),
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, constraint_count), dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, constraint_count), dtype=torch.float32),
            floats.clone(), constraints, floats.clone(), constraints.clone(),
            normalization_epsilon,
        )

    def validate(
        self, observations: tuple[ObservationSnapshot, ...],
        stored_actions: tuple[StoredAction, ...], constraint_count: int,
    ) -> None:
        if self.old_action_type_constraint_values.shape[1] != constraint_count:
            raise BufferValidationError(
                "factor-credit constraint width does not match rollout layout"
            )
        if (
            len(observations) != self.old_log_probabilities.shape[0]
            or len(stored_actions) != self.old_log_probabilities.shape[0]
        ):
            raise BufferValidationError(
                "factor-credit rows do not match observations and stored actions"
            )
        expected_rows = []
        merge_selected = []
        for observation, action in zip(observations, stored_actions, strict=True):
            action_type = _ACTION_TYPES[action.action_type_index]
            masks = observation.set_view.action_masks
            session_applicable = False
            profile_applicable = False
            if action_type is ActionType.MERGE:
                session_applicable = sum(
                    value for _, value in masks.merge_session_mask
                ) > 1
                profile_applicable = sum(
                    value for _, value
                    in masks.merge_profile_mask[action.merge_session_index][1]
                ) > 1
            elif action_type is ActionType.CREATE:
                profile_applicable = sum(
                    value for _, value in masks.create_profile_mask
                ) > 1
            expected_rows.append((True, session_applicable, profile_applicable))
            merge_selected.append(action_type is ActionType.MERGE)
        expected = torch.tensor(expected_rows, dtype=torch.bool).reshape(len(stored_actions), 3)
        if not torch.equal(self.applicability, expected):
            raise BufferValidationError(
                "factor applicability does not match selected feasible branches"
            )
        merge_mask = torch.tensor(merge_selected, dtype=torch.bool)
        if bool(
            self.old_merge_session_reward_values.masked_select(~merge_mask).ne(0.0).any()
        ) or bool(
            self.old_merge_session_constraint_values
            .masked_select((~merge_mask).unsqueeze(1)).ne(0.0).any()
        ):
            raise BufferValidationError(
                "unselected MERGE-session prefix values must be zero"
            )

    def minibatch(self, indices: torch.Tensor) -> FactorCreditMinibatch:
        return FactorCreditMinibatch(
            self.old_log_probabilities[indices],
            self.applicability[indices],
            self.old_action_type_reward_values[indices],
            self.old_action_type_constraint_values[indices],
            self.old_merge_session_reward_values[indices],
            self.old_merge_session_constraint_values[indices],
            self.normalized_reward_advantages[indices],
            self.normalized_constraint_advantages[indices],
        )


@dataclass(frozen=True, slots=True)
class RolloutMinibatch:
    observations: tuple[ObservationSnapshot, ...]
    actions: FactorizedActionIndices
    old_log_probabilities: torch.Tensor
    old_reward_values: torch.Tensor
    old_constraint_values: torch.Tensor
    reward_returns: torch.Tensor
    constraint_returns: torch.Tensor
    reward_advantages: torch.Tensor
    constraint_advantages: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreparedRollout:
    layout: ConstraintLayout
    observations: tuple[ObservationSnapshot, ...]
    next_observations: tuple[ObservationSnapshot | None, ...]
    stored_actions: tuple[StoredAction, ...]
    old_log_probabilities: torch.Tensor
    rewards: torch.Tensor
    constraint_residuals: torch.Tensor
    terminated: torch.Tensor
    physical_slot_spans: torch.Tensor
    old_reward_values: torch.Tensor
    old_constraint_values: torch.Tensor
    reward_advantages: torch.Tensor
    constraint_advantages: torch.Tensor
    reward_returns: torch.Tensor
    constraint_returns: torch.Tensor
    episode_reward_totals: torch.Tensor
    episode_constraint_totals: torch.Tensor
    episode_physical_slots: torch.Tensor
    factor_credit: PreparedFactorCredit | None = None

    def __post_init__(self) -> None:
        n, q = len(self.observations), self.layout.constraint_count
        if len(self.next_observations) != n or len(self.stored_actions) != n:
            raise BufferValidationError("PreparedRollout transitions, next observations, and actions must align")
        feature_layout = None
        for observation, next_observation, action in zip(self.observations, self.next_observations, self.stored_actions, strict=True):
            if not isinstance(observation, ObservationSnapshot) or (next_observation is not None and not isinstance(next_observation, ObservationSnapshot)):
                raise BufferValidationError("PreparedRollout observations must use the public ObservationSnapshot contract")
            if not isinstance(action, StoredAction):
                raise BufferValidationError("PreparedRollout actions must be StoredAction values")
            action.validate_for_observation(observation)
            current_layout = FeatureLayout.from_view(observation.set_view)
            feature_layout = current_layout if feature_layout is None else feature_layout
            if current_layout != feature_layout or (next_observation is not None and FeatureLayout.from_view(next_observation.set_view) != feature_layout):
                raise BufferValidationError("PreparedRollout observations must share one FeatureLayout")
        specifications = (
            (self.old_log_probabilities, torch.float32, (n,)), (self.rewards, torch.float32, (n,)),
            (self.constraint_residuals, torch.float32, (n, q)), (self.terminated, torch.bool, (n,)),
            (self.physical_slot_spans, torch.int64, (n,)), (self.old_reward_values, torch.float32, (n,)),
            (self.old_constraint_values, torch.float32, (n, q)), (self.reward_advantages, torch.float32, (n,)),
            (self.constraint_advantages, torch.float32, (n, q)), (self.reward_returns, torch.float32, (n,)),
            (self.constraint_returns, torch.float32, (n, q)),
        )
        for tensor, dtype, shape in specifications:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype is not dtype or tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.requires_grad:
                raise BufferValidationError("PreparedRollout tensor contract is invalid")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise BufferValidationError("PreparedRollout tensors must be finite")
        if bool((self.physical_slot_spans < 1).any()):
            raise BufferValidationError("physical_slot_spans must be positive")
        if any(bool(self.terminated[index]) != (next_observation is None) for index, next_observation in enumerate(self.next_observations)):
            raise BufferValidationError("terminated must be true exactly when next_observation is absent")
        if not isinstance(self.episode_reward_totals, torch.Tensor):
            raise BufferValidationError("episode_reward_totals must be a torch tensor")
        e = self.episode_reward_totals.shape[0]
        for tensor, dtype, shape in ((self.episode_reward_totals, torch.float32, (e,)),
                                     (self.episode_constraint_totals, torch.float32, (e, q)),
                                     (self.episode_physical_slots, torch.int64, (e,))):
            if not isinstance(tensor, torch.Tensor) or tensor.dtype is not dtype or tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.requires_grad:
                raise BufferValidationError("episode-total tensor contract is invalid")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise BufferValidationError("episode totals must be finite")
        if bool((self.episode_physical_slots < 1).any()):
            raise BufferValidationError("episode_physical_slots must be positive")
        if self.factor_credit is not None:
            if not isinstance(self.factor_credit, PreparedFactorCredit):
                raise BufferValidationError(
                    "factor_credit must be PreparedFactorCredit or None"
                )
            self.factor_credit.validate(self.observations, self.stored_actions, q)
            applicability = self.factor_credit.applicability.to(torch.float32)
            old_joint = (
                self.factor_credit.old_log_probabilities * applicability
            ).sum(dim=1)
            if not torch.allclose(
                old_joint, self.old_log_probabilities,
                rtol=1.0e-6, atol=1.0e-6,
            ):
                raise BufferValidationError(
                    "factor log-probabilities do not reconstruct old joint log-probabilities"
                )
            if not torch.equal(
                self.factor_credit.reward_advantages[:, 0], self.reward_advantages,
            ) or not torch.equal(
                self.factor_credit.constraint_advantages[:, 0], self.constraint_advantages,
            ):
                raise BufferValidationError(
                    "action-type factor advantages must equal rollout advantages"
                )

            from isac_ssc.algorithms.losses import normalize_advantages

            if n:
                normalized = normalize_advantages(
                    self.reward_advantages, self.constraint_advantages,
                    self.factor_credit.normalization_epsilon,
                )
                if not torch.equal(
                    self.factor_credit.normalized_reward_advantages[:, 0], normalized.reward,
                ) or not torch.equal(
                    self.factor_credit.normalized_constraint_advantages[:, 0], normalized.constraints,
                ):
                    raise BufferValidationError(
                        "normalized action-type credit must equal normalized rollout advantages"
                    )

    @property
    def transition_count(self) -> int:
        return len(self.observations)

    def minibatch(self, indices: torch.Tensor, reward_advantages: torch.Tensor, constraint_advantages: torch.Tensor) -> RolloutMinibatch:
        values = indices.tolist()
        actions = FactorizedActionIndices(
            torch.tensor([self.stored_actions[index].action_type_index for index in values], dtype=torch.int64),
            torch.tensor([self.stored_actions[index].merge_session_index for index in values], dtype=torch.int64),
            torch.tensor([self.stored_actions[index].profile_index for index in values], dtype=torch.int64),
        )
        return RolloutMinibatch(
            tuple(self.observations[index] for index in values), actions, self.old_log_probabilities[indices],
            self.old_reward_values[indices], self.old_constraint_values[indices], self.reward_returns[indices],
            self.constraint_returns[indices], reward_advantages[indices], constraint_advantages[indices],
        )


class RolloutBuffer:
    """Accumulate immutable focal transitions and complete physical-episode totals."""

    def __init__(
        self, layout: ConstraintLayout, discount: float, gae_lambda: float,
        factor_normalization_epsilon: float | None = None,
    ) -> None:
        if not isinstance(layout, ConstraintLayout):
            raise BufferValidationError("layout must be ConstraintLayout")
        self.layout = layout
        self.discount = _finite(discount, "discount")
        self.gae_lambda = _finite(gae_lambda, "gae_lambda")
        self.factor_normalization_epsilon = (
            None if factor_normalization_epsilon is None
            else _finite(
                factor_normalization_epsilon,
                "factor_normalization_epsilon",
            )
        )
        if not 0.0 <= self.discount <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise BufferValidationError("discount and gae_lambda must lie in [0, 1]")
        if (
            self.factor_normalization_epsilon is not None
            and self.factor_normalization_epsilon <= 0.0
        ):
            raise BufferValidationError(
                "factor_normalization_epsilon must be positive"
            )
        self._transitions: list[RolloutTransition] = []
        self._episode_totals: list[EpisodeTotals] = []
        self._finalized = False

    def append(self, transition: RolloutTransition) -> None:
        if self._finalized:
            raise BufferValidationError("cannot append after finalization")
        if not isinstance(transition, RolloutTransition):
            raise BufferValidationError("transition must be RolloutTransition")
        transition.validate_layout(self.layout)
        factor_enabled = self.factor_normalization_epsilon is not None
        if (transition.factor_credit is not None) != factor_enabled:
            raise BufferValidationError(
                "rollout factor-credit mode and transition payload disagree"
            )
        if self._transitions:
            previous = self._transitions[-1]
            if not previous.terminated and previous.next_observation != transition.observation:
                raise BufferValidationError("nonterminal transitions must form a contiguous decision sequence")
        self._transitions.append(transition)

    def record_episode_totals(self, totals: EpisodeTotals) -> None:
        if self._finalized:
            raise BufferValidationError("cannot record episode totals after finalization")
        if not isinstance(totals, EpisodeTotals):
            raise BufferValidationError("totals must be EpisodeTotals")
        totals.validate_layout(self.layout)
        self._episode_totals.append(totals)

    def clear(self) -> None:
        self._transitions.clear()
        self._episode_totals.clear()
        self._finalized = False

    def finalize(self, bootstrap_values: ValueOutput | None = None) -> PreparedRollout:
        if self._finalized:
            raise BufferValidationError("rollout buffer is already finalized")
        transitions = tuple(self._transitions)
        if not transitions:
            if bootstrap_values is not None:
                raise BufferValidationError("zero-decision rollout must not receive bootstrap values")
            rewards = torch.empty(0, dtype=torch.float32)
            residuals = torch.empty((0, self.layout.constraint_count), dtype=torch.float32)
            terminated = torch.empty(0, dtype=torch.bool)
            spans = torch.empty(0, dtype=torch.int64)
            old_reward = torch.empty(0, dtype=torch.float32)
            old_constraints = torch.empty((0, self.layout.constraint_count), dtype=torch.float32)
            reward_advantages = reward_returns = torch.empty(0, dtype=torch.float32)
            constraint_advantages = constraint_returns = torch.empty((0, self.layout.constraint_count), dtype=torch.float32)
        elif transitions[-1].terminated:
            if bootstrap_values is not None:
                raise BufferValidationError("terminal rollout must not receive bootstrap values")
            bootstrap_reward = 0.0
            bootstrap_constraints = torch.zeros(self.layout.constraint_count, dtype=torch.float32)
        else:
            if not isinstance(bootstrap_values, ValueOutput):
                raise BufferValidationError("nonterminal rollout requires ValueOutput bootstrap values")
            expected = ((bootstrap_values.reward_value, (1,)),
                        (bootstrap_values.sensing_sla_values, (1, self.layout.tenant_count)),
                        (bootstrap_values.communication_qos_values, (1, self.layout.communication_count)))
            for tensor, shape in expected:
                if not isinstance(tensor, torch.Tensor) or tensor.dtype is not torch.float32 or tensor.device.type != "cpu" or tuple(tensor.shape) != shape:
                    raise BufferValidationError("bootstrap values must be CPU float32 with exact ConstraintLayout dimensions")
                if tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
                    raise BufferValidationError("bootstrap values must be finite and detached")
            bootstrap_reward = float(bootstrap_values.reward_value[0])
            bootstrap_constraints = torch.cat((bootstrap_values.sensing_sla_values[0], bootstrap_values.communication_qos_values[0]))
        if transitions:
            rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32)
            residuals = torch.tensor([item.tenant_residuals + item.communication_residuals for item in transitions], dtype=torch.float32)
            terminated = torch.tensor([item.terminated for item in transitions], dtype=torch.bool)
            spans = torch.tensor([item.physical_slot_span for item in transitions], dtype=torch.int64)
            old_reward = torch.tensor([item.old_reward_value for item in transitions], dtype=torch.float32)
            old_constraints = torch.tensor(
                [item.old_tenant_values + item.old_communication_values for item in transitions], dtype=torch.float32)
            next_reward = torch.empty_like(old_reward)
            next_constraints = torch.empty_like(old_constraints)
            for index, item in enumerate(transitions):
                if item.terminated:
                    next_reward[index] = 0.0
                    next_constraints[index].zero_()
                elif index + 1 < len(transitions):
                    next_reward[index] = old_reward[index + 1]
                    next_constraints[index] = old_constraints[index + 1]
                else:
                    next_reward[index] = bootstrap_reward
                    next_constraints[index] = bootstrap_constraints
            from isac_ssc.algorithms.losses import generalized_advantage_estimate

            reward_advantages, reward_returns = generalized_advantage_estimate(
                rewards, old_reward, next_reward, terminated, spans, self.discount, self.gae_lambda,
            )
            constraint_advantages, constraint_returns = generalized_advantage_estimate(
                residuals, old_constraints, next_constraints, terminated, spans, self.discount, self.gae_lambda,
            )
        factor_credit = None
        if self.factor_normalization_epsilon is not None:
            if not transitions:
                factor_credit = PreparedFactorCredit.empty(
                    self.layout.constraint_count,
                    self.factor_normalization_epsilon,
                )
            else:
                payloads = tuple(item.factor_credit for item in transitions)
                if any(item is None for item in payloads):
                    raise BufferValidationError(
                        "factor-credit rollout contains a missing transition payload"
                    )
                values = tuple(item for item in payloads if item is not None)
                old_factor_logs = torch.tensor([
                    (
                        item.old_action_type_log_probability,
                        item.old_merge_session_log_probability,
                        item.old_profile_log_probability,
                    )
                    for item in values
                ], dtype=torch.float32)
                applicability = torch.tensor([
                    (
                        True,
                        item.merge_session_applicable,
                        item.profile_applicable,
                    )
                    for item in values
                ], dtype=torch.bool)
                merge_selected = torch.tensor([
                    _ACTION_TYPES[transition.action.action_type_index]
                    is ActionType.MERGE
                    for transition in transitions
                ], dtype=torch.bool)
                old_type_reward = torch.tensor(
                    [item.old_action_type_reward_value for item in values],
                    dtype=torch.float32,
                )
                old_type_constraints = torch.tensor(
                    [item.old_action_type_constraint_values for item in values],
                    dtype=torch.float32,
                )
                old_session_reward = torch.tensor(
                    [item.old_merge_session_reward_value for item in values],
                    dtype=torch.float32,
                )
                old_session_constraints = torch.tensor(
                    [item.old_merge_session_constraint_values for item in values],
                    dtype=torch.float32,
                )

                from isac_ssc.algorithms.losses import (
                    build_frozen_factor_advantages,
                )

                frozen = build_frozen_factor_advantages(
                    reward_advantages=reward_advantages,
                    constraint_advantages=constraint_advantages,
                    reward_returns=reward_returns,
                    constraint_returns=constraint_returns,
                    old_reward_values=old_reward,
                    old_constraint_values=old_constraints,
                    old_action_type_reward_values=old_type_reward,
                    old_action_type_constraint_values=old_type_constraints,
                    old_merge_session_reward_values=old_session_reward,
                    old_merge_session_constraint_values=old_session_constraints,
                    merge_selected=merge_selected,
                    merge_session_applicable=applicability[:, 1],
                    profile_applicable=applicability[:, 2],
                    epsilon=self.factor_normalization_epsilon,
                )
                factor_credit = PreparedFactorCredit(
                    old_factor_logs, applicability,
                    old_type_reward, old_type_constraints,
                    old_session_reward, old_session_constraints,
                    frozen.reward, frozen.constraints,
                    frozen.normalized_reward, frozen.normalized_constraints,
                    self.factor_normalization_epsilon,
                )

        episode_rewards = torch.tensor([item.reward_total for item in self._episode_totals], dtype=torch.float32)
        episode_constraints = torch.tensor(
            [item.tenant_residual_totals + item.communication_residual_totals for item in self._episode_totals], dtype=torch.float32,
        ).reshape(len(self._episode_totals), self.layout.constraint_count)
        episode_slots = torch.tensor([item.physical_slot_count for item in self._episode_totals], dtype=torch.int64)
        self._finalized = True
        return PreparedRollout(
            self.layout, tuple(item.observation for item in transitions), tuple(item.next_observation for item in transitions),
            tuple(item.action for item in transitions),
            torch.tensor([item.old_log_probability for item in transitions], dtype=torch.float32), rewards, residuals,
            terminated, spans, old_reward, old_constraints, reward_advantages, constraint_advantages,
            reward_returns, constraint_returns, episode_rewards,
            episode_constraints, episode_slots, factor_credit,
        )