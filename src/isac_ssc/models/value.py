"""Reward and per-constraint value heads for constrained PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ValueValidationError(ValueError):
    """Raised when critic inputs or dimensions violate the locked contract."""


@dataclass(frozen=True, slots=True)
class ValueOutput:
    reward_value: torch.Tensor
    sensing_sla_values: torch.Tensor
    communication_qos_values: torch.Tensor


class MultiConstraintValueHead(nn.Module):
    """Predict reward, per-tenant SLA, and per-user communication returns."""

    def __init__(self, hidden_dim: int, number_of_tenants: int, number_of_communication_users: int) -> None:
        super().__init__()
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in (
                hidden_dim,
                number_of_tenants,
                number_of_communication_users,
            )
        ):
            raise ValueValidationError(
                "critic dimensions must be positive integers"
            )
        self.hidden_dim = hidden_dim
        self.number_of_tenants = number_of_tenants
        self.number_of_communication_users = (
            number_of_communication_users
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.sensing_sla_head = nn.Linear(
            hidden_dim, number_of_tenants,
        )
        self.communication_qos_head = nn.Linear(
            hidden_dim, number_of_communication_users,
        )
        nn.init.orthogonal_(self.trunk[0].weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(self.trunk[0].bias)
        for head in (self.reward_head, self.sensing_sla_head, self.communication_qos_head):
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)

    def forward(self, decision_embedding: torch.Tensor) -> ValueOutput:
        hidden = self.trunk(decision_embedding)
        return ValueOutput(
            self.reward_head(hidden).squeeze(-1), self.sensing_sla_head(hidden),
            self.communication_qos_head(hidden),
        )


@dataclass(frozen=True, slots=True)
class PrefixValueOutput:
    type_reward_values: torch.Tensor
    type_constraint_values: torch.Tensor
    merge_session_reward_values: torch.Tensor
    merge_session_constraint_values: torch.Tensor


class HierarchicalPrefixValueHead(nn.Module):
    """Predict selected-prefix returns for type and MERGE-session credit assignment."""

    def __init__(self, hidden_dim: int, constraint_count: int) -> None:
        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (hidden_dim, constraint_count)
        ):
            raise ValueValidationError("prefix critic dimensions must be positive integers")
        self.hidden_dim = hidden_dim
        self.constraint_count = constraint_count
        self.type_trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.type_reward_head = nn.Linear(hidden_dim, 4)
        self.type_constraint_head = nn.Linear(hidden_dim, 4 * constraint_count)
        self.merge_session_trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.merge_session_reward_head = nn.Linear(hidden_dim, 1)
        self.merge_session_constraint_head = nn.Linear(hidden_dim, constraint_count)
        for trunk in (self.type_trunk, self.merge_session_trunk):
            nn.init.orthogonal_(trunk[0].weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(trunk[0].bias)
        for head in (
            self.type_reward_head, self.type_constraint_head,
            self.merge_session_reward_head, self.merge_session_constraint_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, decision_embedding: torch.Tensor,
        merge_candidate_embeddings: torch.Tensor,
        base_values: ValueOutput,
    ) -> PrefixValueOutput:
        if not isinstance(base_values, ValueOutput):
            raise ValueValidationError("prefix critics require global base values")
        type_hidden = self.type_trunk(decision_embedding.detach())
        session_hidden = self.merge_session_trunk(merge_candidate_embeddings.detach())
        batch = decision_embedding.shape[0]
        base_constraints = torch.cat(
            (base_values.sensing_sla_values, base_values.communication_qos_values),
            dim=1,
        ).detach()
        type_reward = (
            base_values.reward_value.detach().unsqueeze(1)
            + self.type_reward_head(type_hidden)
        )
        type_constraints = (
            base_constraints.unsqueeze(1)
            + self.type_constraint_head(type_hidden).reshape(
                batch, 4, self.constraint_count,
            )
        )
        merge_reward = type_reward[:, 0].detach().unsqueeze(1)
        merge_constraints = type_constraints[:, 0].detach().unsqueeze(1)
        return PrefixValueOutput(
            type_reward,
            type_constraints,
            merge_reward + self.merge_session_reward_head(session_hidden).squeeze(-1),
            merge_constraints + self.merge_session_constraint_head(session_hidden),
        )