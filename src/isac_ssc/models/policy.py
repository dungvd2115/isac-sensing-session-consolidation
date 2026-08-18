"""Candidate-conditioned factorized policy for the edge-free set reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from isac_ssc.core.entities import EntityId
from isac_ssc.envs.action_space import (
    ActionType, EnvironmentAction, identifier_key,
)
from isac_ssc.envs.observation import ObservationSnapshot
from isac_ssc.models.set_encoder import (
    EdgeFreeSetEncoder, FeatureLayout, SetEncoderInput, SetEncoderOutput,
)
from isac_ssc.models.value import (
    HierarchicalPrefixValueHead, MultiConstraintValueHead, PrefixValueOutput, ValueOutput,
)
from isac_ssc.utils.config import CanonicalConfig, ConstrainedPPOConfig


class PolicyValidationError(ValueError):
    """Raised when policy inputs, masks, or selected actions violate the public contract."""


_ACTION_TYPES = (
    ActionType.MERGE, ActionType.CREATE, ActionType.DEFER, ActionType.REJECT,
)
_ACTION_INDEX = {
    item: index for index, item in enumerate(_ACTION_TYPES)
}


@dataclass(frozen=True, slots=True)
class FactorizedPolicyBatch:
    layout: FeatureLayout
    encoder_input: SetEncoderInput
    action_type_mask: torch.Tensor
    merge_session_mask: torch.Tensor
    merge_profile_mask: torch.Tensor
    create_profile_mask: torch.Tensor
    session_ids: tuple[tuple[EntityId, ...], ...]
    profile_ids: tuple[str, ...]
    feasible_actions: tuple[tuple[EnvironmentAction, ...], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.layout, FeatureLayout)
            or not isinstance(self.encoder_input, SetEncoderInput)
        ):
            raise PolicyValidationError("layout and encoder_input are required")
        masks = (
            self.action_type_mask, self.merge_session_mask,
            self.merge_profile_mask, self.create_profile_mask,
        )
        if any(
            not isinstance(item, torch.Tensor) or item.dtype is not torch.bool
            for item in masks
        ):
            raise PolicyValidationError(
                "policy masks must be torch.bool tensors"
            )
        batch = self.encoder_input.request_features.shape[0]
        sessions = self.encoder_input.session_features.shape[1]
        profiles = len(self.profile_ids)
        if (
            self.action_type_mask.shape != (batch, 4)
            or self.merge_session_mask.shape != (batch, sessions)
            or self.merge_profile_mask.shape != (batch, sessions, profiles)
            or self.create_profile_mask.shape != (batch, profiles)
        ):
            raise PolicyValidationError(
                "factorized policy mask shapes are inconsistent"
            )
        if any(
            item.device != self.encoder_input.request_features.device
            for item in masks
        ):
            raise PolicyValidationError(
                "policy masks and features must use one device"
            )
        if self.profile_ids != self.layout.profile_ids:
            raise PolicyValidationError(
                "batch profile order must match FeatureLayout"
            )
        if (
            len(self.session_ids) != batch
            or len(self.feasible_actions) != batch
        ):
            raise PolicyValidationError(
                "policy metadata batch dimensions disagree"
            )
        for index, values in enumerate(self.session_ids):
            if len(values) > sessions:
                raise PolicyValidationError(
                    "session metadata exceeds padded tensor width"
                )
            if len({
                identifier_key(item) for item in values
            }) != len(values):
                raise PolicyValidationError(
                    "session identifiers must be unique under typed identity"
                )
            if bool(self.merge_session_mask[index, len(values):].any()):
                raise PolicyValidationError(
                    "padded sessions must be masked infeasible"
                )
            if bool(self.merge_profile_mask[index, len(values):].any()):
                raise PolicyValidationError(
                    "padded session profiles must be masked infeasible"
                )
        if bool((~self.action_type_mask).all(dim=1).any()):
            raise PolicyValidationError("every focal request requires at least one feasible action type")
        merge_enabled = self.action_type_mask[:, _ACTION_INDEX[ActionType.MERGE]]
        create_enabled = self.action_type_mask[:, _ACTION_INDEX[ActionType.CREATE]]
        if not torch.equal(merge_enabled, self.merge_session_mask.any(dim=1)):
            raise PolicyValidationError("MERGE feasibility must match the merge-session mask")
        if bool((self.merge_session_mask & ~self.merge_profile_mask.any(dim=2)).any()):
            raise PolicyValidationError("every feasible merge session requires a feasible profile")
        if not torch.equal(create_enabled, self.create_profile_mask.any(dim=1)):
            raise PolicyValidationError("CREATE feasibility must match the create-profile mask")


@dataclass(frozen=True, slots=True)
class FactorizedActionIndices:
    action_type: torch.Tensor
    merge_session: torch.Tensor
    profile: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.action_type, self.merge_session, self.profile,
        )
        if any(
            not isinstance(item, torch.Tensor)
            or item.dtype is not torch.int64
            for item in tensors
        ):
            raise PolicyValidationError(
                "factorized action indices must be torch.int64 tensors"
            )
        if (
            any(item.ndim != 1 for item in tensors)
            or len({item.shape for item in tensors}) != 1
        ):
            raise PolicyValidationError(
                "factorized action indices must share shape [B]"
            )
        if len({item.device for item in tensors}) != 1:
            raise PolicyValidationError(
                "factorized action indices must use one device"
            )


@dataclass(frozen=True, slots=True)
class FactorizedPolicyLogits:
    action_type_logits: torch.Tensor
    merge_session_logits: torch.Tensor
    merge_profile_logits: torch.Tensor
    create_profile_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class FactorizedLogProbabilities:
    action_type: torch.Tensor
    merge_session: torch.Tensor
    profile: torch.Tensor
    merge_session_applicable: torch.Tensor
    profile_applicable: torch.Tensor

    @property
    def joint(self) -> torch.Tensor:
        zero = torch.zeros_like(self.action_type)
        session = torch.where(self.merge_session_applicable, self.merge_session, zero)
        profile = torch.where(self.profile_applicable, self.profile, zero)
        return self.action_type + session + profile


@dataclass(frozen=True, slots=True)
class PolicySelection:
    indices: FactorizedActionIndices
    actions: tuple[EnvironmentAction, ...]
    log_probability: torch.Tensor
    entropy: torch.Tensor
    factor_log_probabilities: FactorizedLogProbabilities


def _observation_values(
    observations: ObservationSnapshot | Iterable[ObservationSnapshot],
) -> tuple[ObservationSnapshot, ...]:
    values = (
        (observations,)
        if isinstance(observations, ObservationSnapshot)
        else tuple(observations)
    )
    if (
        not values
        or any(not isinstance(item, ObservationSnapshot) for item in values)
    ):
        raise PolicyValidationError(
            "observations must contain ObservationSnapshot values"
        )
    return values


def build_policy_batch(
    observations: ObservationSnapshot | Iterable[ObservationSnapshot],
    *,
    device: str | torch.device = "cpu",
) -> FactorizedPolicyBatch:
    """Pad edge-free public observations without encoding relational identifiers."""
    values = _observation_values(observations)
    target_device = torch.device(device)
    views = tuple(item.set_view for item in values)
    layouts = tuple(FeatureLayout.from_view(item) for item in views)
    layout = layouts[0]
    if any(item != layout for item in layouts[1:]):
        raise PolicyValidationError(
            "all observations in a batch must share one feature schema"
        )
    batch = len(views)
    max_requests = max(
        len(item.request_table.rows) for item in views
    )
    max_sessions = max(
        len(item.session_table.rows) for item in views
    )
    profiles = len(layout.profile_ids)
    request_features = torch.zeros(
        (batch, max_requests, layout.request_width),
        dtype=torch.float32, device=target_device,
    )
    session_features = torch.zeros(
        (batch, max_sessions, layout.session_width),
        dtype=torch.float32, device=target_device,
    )
    global_features = torch.zeros(
        (batch, layout.global_width),
        dtype=torch.float32, device=target_device,
    )
    request_padding = torch.ones(
        (batch, max_requests),
        dtype=torch.bool, device=target_device,
    )
    session_padding = torch.ones(
        (batch, max_sessions),
        dtype=torch.bool, device=target_device,
    )
    focal_index = torch.empty(
        batch, dtype=torch.int64, device=target_device,
    )
    action_type_mask = torch.zeros(
        (batch, 4), dtype=torch.bool, device=target_device,
    )
    merge_session_mask = torch.zeros(
        (batch, max_sessions),
        dtype=torch.bool, device=target_device,
    )
    merge_profile_mask = torch.zeros(
        (batch, max_sessions, profiles),
        dtype=torch.bool, device=target_device,
    )
    create_profile_mask = torch.zeros(
        (batch, profiles),
        dtype=torch.bool, device=target_device,
    )
    session_metadata, feasible_metadata = [], []
    for batch_index, view in enumerate(views):
        request_rows = view.request_table.rows
        session_rows = view.session_table.rows
        if not request_rows:
            raise PolicyValidationError(
                "each policy observation requires a focal request row"
            )
        request_count = len(request_rows)
        session_count = len(session_rows)
        request_features[batch_index, :request_count] = torch.tensor(
            request_rows, dtype=torch.float32, device=target_device,
        )
        if session_count:
            session_features[batch_index, :session_count] = torch.tensor(
                session_rows, dtype=torch.float32, device=target_device,
            )
        global_features[batch_index] = torch.tensor(
            view.global_features,
            dtype=torch.float32, device=target_device,
        )
        request_padding[batch_index, :request_count] = False
        session_padding[batch_index, :session_count] = False
        focal_key = identifier_key(
            view.action_masks.focal_request_id
        )
        matches = tuple(
            index for index, key in enumerate(view.request_table.keys)
            if identifier_key(key.request_id) == focal_key
        )
        if len(matches) != 1:
            raise PolicyValidationError(
                "focal relational key must identify exactly one request row"
            )
        focal_index[batch_index] = matches[0]
        typed_mask = view.action_masks.action_type_mask
        if tuple(item[0] for item in typed_mask) != _ACTION_TYPES:
            raise PolicyValidationError(
                "public action-type order does not match the learning contract"
            )
        action_type_mask[batch_index] = torch.tensor(
            tuple(item[1] for item in typed_mask),
            dtype=torch.bool, device=target_device,
        )
        session_ids = tuple(
            key.session_id for key in view.session_table.keys
        )
        public_session_mask = view.action_masks.merge_session_mask
        if tuple(
            item[0] for item in public_session_mask
        ) != session_ids:
            raise PolicyValidationError(
                "session rows and public merge-session mask disagree"
            )
        if session_count:
            merge_session_mask[
                batch_index, :session_count
            ] = torch.tensor(
                tuple(item[1] for item in public_session_mask),
                dtype=torch.bool, device=target_device,
            )
        public_merge_profiles = (
            view.action_masks.merge_profile_mask
        )
        if tuple(
            item[0] for item in public_merge_profiles
        ) != session_ids:
            raise PolicyValidationError(
                "session rows and public merge-profile mask disagree"
            )
        for session_index, (_, profile_mask) in enumerate(
            public_merge_profiles
        ):
            if tuple(
                item[0] for item in profile_mask
            ) != layout.profile_ids:
                raise PolicyValidationError(
                    "public merge-profile order does not match FeatureLayout"
                )
            merge_profile_mask[
                batch_index, session_index
            ] = torch.tensor(
                tuple(item[1] for item in profile_mask),
                dtype=torch.bool, device=target_device,
            )
        public_create = view.action_masks.create_profile_mask
        if tuple(
            item[0] for item in public_create
        ) != layout.profile_ids:
            raise PolicyValidationError(
                "public create-profile order does not match FeatureLayout"
            )
        create_profile_mask[batch_index] = torch.tensor(
            tuple(item[1] for item in public_create),
            dtype=torch.bool, device=target_device,
        )
        session_metadata.append(session_ids)
        feasible_metadata.append(
            view.action_masks.feasible_actions
        )
    encoder_input = SetEncoderInput(
        request_features, request_padding,
        session_features, session_padding,
        global_features, focal_index,
    )
    return FactorizedPolicyBatch(
        layout, encoder_input, action_type_mask,
        merge_session_mask, merge_profile_mask,
        create_profile_mask, tuple(session_metadata),
        layout.profile_ids, tuple(feasible_metadata),
    )


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(~mask, -torch.inf)
    log_probabilities = F.log_softmax(masked, dim=-1)
    probabilities = log_probabilities.exp()
    safe_logs = torch.where(mask, log_probabilities, torch.zeros_like(log_probabilities))
    return log_probabilities, probabilities, -(probabilities * safe_logs).sum()


def _action_type_logits_with_merge_options(
    logits: FactorizedPolicyLogits, batch: FactorizedPolicyBatch, batch_index: int,
) -> torch.Tensor:
    """Add a candidate-count-normalized joint option score only to MERGE."""
    action_logits = logits.action_type_logits[batch_index]
    if not bool(batch.action_type_mask[batch_index, _ACTION_INDEX[ActionType.MERGE]]):
        return action_logits
    pair_mask = batch.merge_profile_mask[batch_index]
    pair_logits = logits.merge_session_logits[batch_index].unsqueeze(-1) + logits.merge_profile_logits[batch_index]
    valid_pair_logits = pair_logits.masked_select(pair_mask)
    if not valid_pair_logits.numel():
        raise PolicyValidationError("MERGE requires at least one feasible session-profile pair")
    merge_option_score = torch.logsumexp(valid_pair_logits, dim=0) - valid_pair_logits.new_tensor(valid_pair_logits.numel()).log()
    adjustment = torch.zeros_like(action_logits)
    adjustment[_ACTION_INDEX[ActionType.MERGE]] = merge_option_score
    return action_logits + adjustment


def _draw(
    probabilities: torch.Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None,
) -> int:
    if deterministic:
        return int(torch.argmax(probabilities).item())
    return int(torch.multinomial(
        probabilities, 1, generator=generator,
    ).item())


def _initialize_scorer(module: nn.Sequential) -> None:
    first, last = module[0], module[2]
    nn.init.orthogonal_(
        first.weight, gain=nn.init.calculate_gain("tanh"),
    )
    nn.init.zeros_(first.bias)
    nn.init.orthogonal_(last.weight, gain=0.01)
    nn.init.zeros_(last.bias)


class FactorizedPolicyHead(nn.Module):
    """Score action type, merge candidate, and conditional profile branches."""

    def __init__(self, hidden_dim: int, profile_count: int, profile_embedding_dim: int) -> None:
        super().__init__()
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in (
                hidden_dim, profile_count, profile_embedding_dim,
            )
        ):
            raise PolicyValidationError(
                "policy dimensions must be positive integers"
            )
        self.hidden_dim = hidden_dim
        self.profile_count = profile_count
        self.profile_embeddings = nn.Embedding(
            profile_count, profile_embedding_dim,
        )
        self.action_type_head = nn.Linear(hidden_dim, 4)
        self.merge_session_head = nn.Linear(hidden_dim, 1)
        self.create_profile_head = nn.Sequential(
            nn.Linear(
                hidden_dim + profile_embedding_dim, hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.merge_profile_head = nn.Sequential(
            nn.Linear(
                hidden_dim + profile_embedding_dim, hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.orthogonal_(self.profile_embeddings.weight, gain=1.0)
        for head in (self.action_type_head, self.merge_session_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        _initialize_scorer(self.create_profile_head)
        _initialize_scorer(self.merge_profile_head)

    def forward(self, encoded: SetEncoderOutput) -> FactorizedPolicyLogits:
        decision = encoded.decision_embedding
        candidates = encoded.merge_candidate_embeddings
        batch, sessions = candidates.shape[:2]
        profiles = self.profile_embeddings.weight
        create_profiles = profiles.unsqueeze(0).expand(batch, -1, -1)
        create_decision = decision.unsqueeze(1).expand(-1, self.profile_count, -1)
        create_logits = self.create_profile_head(torch.cat((create_decision, create_profiles), dim=-1)).squeeze(-1)
        merge_candidates = candidates.unsqueeze(2).expand(-1, -1, self.profile_count, -1)
        merge_profiles = profiles.view(1, 1, self.profile_count, -1).expand(batch, sessions, -1, -1)
        return FactorizedPolicyLogits(
            self.action_type_head(decision), self.merge_session_head(candidates).squeeze(-1),
            self.merge_profile_head(torch.cat((merge_candidates, merge_profiles), dim=-1)).squeeze(-1),
            create_logits,
        )

    def entropy(
        self,
        logits: FactorizedPolicyLogits,
        batch: FactorizedPolicyBatch,
    ) -> torch.Tensor:
        values = []
        for index in range(
            logits.action_type_logits.shape[0]
        ):
            _, type_probabilities, total = _masked_categorical(
                _action_type_logits_with_merge_options(logits, batch, index),
                batch.action_type_mask[index]
            )
            if bool(batch.action_type_mask[
                index, _ACTION_INDEX[ActionType.MERGE]
            ]):
                (
                    _,
                    session_probabilities,
                    session_entropy,
                ) = _masked_categorical(
                    logits.merge_session_logits[index],
                    batch.merge_session_mask[index]
                )
                profile_entropy = torch.zeros(
                    (), dtype=total.dtype, device=total.device,
                )
                for session_index in range(
                    logits.merge_session_logits.shape[1]
                ):
                    if bool(batch.merge_session_mask[
                        index, session_index
                    ]):
                        _, _, conditional = _masked_categorical(
                            logits.merge_profile_logits[
                                index, session_index
                            ],
                            batch.merge_profile_mask[
                                index, session_index
                            ]
                        )
                        profile_entropy = (
                            profile_entropy
                            + session_probabilities[session_index]
                            * conditional
                        )
                total = (
                    total
                    + type_probabilities[
                        _ACTION_INDEX[ActionType.MERGE]
                    ]
                    * (session_entropy + profile_entropy)
                )
            if bool(batch.action_type_mask[
                index, _ACTION_INDEX[ActionType.CREATE]
            ]):
                _, _, create_entropy = _masked_categorical(
                    logits.create_profile_logits[index],
                    batch.create_profile_mask[index]
                )
                total = (
                    total
                    + type_probabilities[
                        _ACTION_INDEX[ActionType.CREATE]
                    ]
                    * create_entropy
                )
            values.append(total)
        return torch.stack(values)

    def evaluate_components(
        self, logits: FactorizedPolicyLogits, batch: FactorizedPolicyBatch,
        indices: FactorizedActionIndices,
    ) -> FactorizedLogProbabilities:
        type_values, session_values, profile_values = [], [], []
        session_applicability, profile_applicability = [], []
        for batch_index in range(logits.action_type_logits.shape[0]):
            type_index = int(indices.action_type[batch_index])
            type_logs, _, _ = _masked_categorical(
                _action_type_logits_with_merge_options(logits, batch, batch_index), batch.action_type_mask[batch_index],
            )
            type_value = type_logs[type_index]
            session_value = torch.zeros_like(type_value)
            profile_value = torch.zeros_like(type_value)
            action_type = _ACTION_TYPES[type_index]
            session_index = int(indices.merge_session[batch_index])
            profile_index = int(indices.profile[batch_index])
            session_applies = False
            profile_applies = False
            if action_type is ActionType.MERGE:
                session_mask = batch.merge_session_mask[batch_index]
                profile_mask = batch.merge_profile_mask[batch_index, session_index]
                session_logs, _, _ = _masked_categorical(
                    logits.merge_session_logits[batch_index], session_mask,
                )
                profile_logs, _, _ = _masked_categorical(
                    logits.merge_profile_logits[batch_index, session_index], profile_mask,
                )
                session_applies = int(session_mask.sum()) > 1
                profile_applies = int(profile_mask.sum()) > 1
                if session_applies:
                    session_value = session_logs[session_index]
                if profile_applies:
                    profile_value = profile_logs[profile_index]
            elif action_type is ActionType.CREATE:
                profile_mask = batch.create_profile_mask[batch_index]
                profile_logs, _, _ = _masked_categorical(
                    logits.create_profile_logits[batch_index], profile_mask,
                )
                profile_applies = int(profile_mask.sum()) > 1
                if profile_applies:
                    profile_value = profile_logs[profile_index]
            type_values.append(type_value)
            session_values.append(session_value)
            profile_values.append(profile_value)
            session_applicability.append(session_applies)
            profile_applicability.append(profile_applies)
        device = logits.action_type_logits.device
        return FactorizedLogProbabilities(
            torch.stack(type_values), torch.stack(session_values), torch.stack(profile_values),
            torch.tensor(session_applicability, dtype=torch.bool, device=device),
            torch.tensor(profile_applicability, dtype=torch.bool, device=device),
        )

    def evaluate(
        self, logits: FactorizedPolicyLogits, batch: FactorizedPolicyBatch,
        indices: FactorizedActionIndices,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        components = self.evaluate_components(logits, batch, indices)
        return components.joint, self.entropy(logits, batch)

    def select(
        self,
        logits: FactorizedPolicyLogits,
        batch: FactorizedPolicyBatch,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> PolicySelection:
        type_indices = []
        session_indices = []
        profile_indices = []
        actions = []
        for batch_index in range(
            logits.action_type_logits.shape[0]
        ):
            _, type_probabilities, _ = _masked_categorical(
                _action_type_logits_with_merge_options(logits, batch, batch_index),
                batch.action_type_mask[batch_index]
            )
            type_index = _draw(
                type_probabilities,
                deterministic=deterministic,
                generator=generator,
            )
            action_type = _ACTION_TYPES[type_index]
            session_index = -1
            profile_index = -1
            if action_type is ActionType.MERGE:
                (
                    _,
                    session_probabilities,
                    _,
                ) = _masked_categorical(
                    logits.merge_session_logits[batch_index],
                    batch.merge_session_mask[batch_index]
                )
                session_index = _draw(
                    session_probabilities,
                    deterministic=deterministic,
                    generator=generator,
                )
                (
                    _,
                    profile_probabilities,
                    _,
                ) = _masked_categorical(
                    logits.merge_profile_logits[
                        batch_index, session_index
                    ],
                    batch.merge_profile_mask[
                        batch_index, session_index
                    ]
                )
                profile_index = _draw(
                    profile_probabilities,
                    deterministic=deterministic,
                    generator=generator,
                )
                action = EnvironmentAction(
                    action_type,
                    batch.session_ids[
                        batch_index
                    ][session_index],
                    batch.profile_ids[profile_index],
                )
            elif action_type is ActionType.CREATE:
                (
                    _,
                    profile_probabilities,
                    _,
                ) = _masked_categorical(
                    logits.create_profile_logits[batch_index],
                    batch.create_profile_mask[batch_index]
                )
                profile_index = _draw(
                    profile_probabilities,
                    deterministic=deterministic,
                    generator=generator,
                )
                action = EnvironmentAction(
                    action_type,
                    profile_id=batch.profile_ids[profile_index],
                )
            else:
                action = EnvironmentAction(action_type)
            type_indices.append(type_index)
            session_indices.append(session_index)
            profile_indices.append(profile_index)
            actions.append(action)
        device = logits.action_type_logits.device
        indices = FactorizedActionIndices(
            torch.tensor(
                type_indices, dtype=torch.int64, device=device,
            ),
            torch.tensor(
                session_indices, dtype=torch.int64, device=device,
            ),
            torch.tensor(
                profile_indices, dtype=torch.int64, device=device,
            ),
        )
        components = self.evaluate_components(logits, batch, indices)
        entropy = self.entropy(logits, batch)
        return PolicySelection(
            indices, tuple(actions), components.joint, entropy, components,
        )


class EdgeFreeSetActorCritic(nn.Module):
    """Share one edge-free set encoder across factorized actor and all critics."""

    def __init__(
        self,
        layout: FeatureLayout,
        algorithm: ConstrainedPPOConfig,
        environment: CanonicalConfig,
    ) -> None:
        super().__init__()
        if (
            not isinstance(algorithm, ConstrainedPPOConfig)
            or not isinstance(environment, CanonicalConfig)
        ):
            raise PolicyValidationError(
                "validated algorithm and environment configs are required"
            )
        self.layout = layout
        self.encoder = EdgeFreeSetEncoder(
            layout, algorithm.model,
        )
        self.policy = FactorizedPolicyHead(
            algorithm.model.hidden_dim, len(layout.profile_ids), algorithm.model.profile_embedding_dim,
        )
        self.value = MultiConstraintValueHead(
            algorithm.model.hidden_dim, len(environment.tenants), environment.population["communication_users"],
        )

    def forward(
        self,
        batch: FactorizedPolicyBatch,
    ) -> tuple[
        SetEncoderOutput,
        FactorizedPolicyLogits,
        ValueOutput,
    ]:
        encoded = self.encoder(batch.encoder_input)
        return (
            encoded,
            self.policy(encoded),
            self.value(encoded.decision_embedding),
        )

    def select(
        self,
        batch: FactorizedPolicyBatch,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[PolicySelection, ValueOutput]:
        _, logits, values = self.forward(batch)
        return (
            self.policy.select(
                logits, batch,
                deterministic=deterministic,
                generator=generator,
            ),
            values,
        )


class CommonTraceActorCritic(EdgeFreeSetActorCritic):
    """Add detached hierarchical prefix critics without changing the shared actor-critic."""

    def __init__(
        self, layout: FeatureLayout, algorithm: ConstrainedPPOConfig, environment: CanonicalConfig,
    ) -> None:
        super().__init__(layout, algorithm, environment)
        constraint_count = len(environment.tenants) + environment.population["communication_users"]
        self.prefix_value = HierarchicalPrefixValueHead(algorithm.model.hidden_dim, constraint_count)

    def forward(
        self, batch: FactorizedPolicyBatch,
    ) -> tuple[SetEncoderOutput, FactorizedPolicyLogits, ValueOutput, PrefixValueOutput]:
        encoded, logits, values = super().forward(batch)
        prefix_values = self.prefix_value(
            encoded.decision_embedding, encoded.merge_candidate_embeddings,
            values,
        )
        return encoded, logits, values, prefix_values

    def select(
        self, batch: FactorizedPolicyBatch, *, deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[PolicySelection, ValueOutput, PrefixValueOutput]:
        _, logits, values, prefix_values = self.forward(batch)
        selection = self.policy.select(
            logits, batch, deterministic=deterministic, generator=generator,
        )
        return selection, values, prefix_values