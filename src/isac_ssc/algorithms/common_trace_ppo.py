"""Hierarchical factor-credit PPO with rollout-frozen prefix advantages."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import torch

from isac_ssc.algorithms.buffers import PreparedRollout
from isac_ssc.algorithms.constrained_ppo import (
    ConstrainedPPO, PPOValidationError, _assign_parameter_gradients,
    _clipped_objective_gradients,
)
from isac_ssc.algorithms.losses import (
    FactorizedPolicySurrogates, build_factorized_policy_surrogates, clipped_value_loss,
    masked_clipped_value_loss, signal_scales,
)
from isac_ssc.models.policy import FactorizedActionIndices, CommonTraceActorCritic, build_policy_batch
from isac_ssc.utils.config import ConstrainedPPOConfig


def _cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    result = float(value.detach().cpu())
    if not isfinite(result):
        raise PPOValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CommonTracePPOUpdateMetrics:
    transitions: int
    epochs_completed: int
    minibatches_completed: int
    optimizer_steps: int
    early_stopped_for_kl: bool
    mean_total_loss: float
    mean_actor_loss: float
    mean_reward_surrogate: float
    mean_reward_surrogates_by_factor: torch.Tensor
    mean_constraint_surrogates: torch.Tensor
    mean_constraint_surrogates_by_factor: torch.Tensor
    mean_reward_value_loss: float
    mean_constraint_value_loss: float
    mean_tenant_value_loss: float
    mean_communication_value_loss: float
    mean_type_reward_value_loss: float
    mean_type_constraint_value_loss: float
    mean_session_reward_value_loss: float
    mean_session_constraint_value_loss: float
    mean_entropy: float
    mean_joint_approximate_kl: float
    max_minibatch_joint_approximate_kl: float
    mean_joint_clip_fraction: float
    mean_factor_clip_fractions: torch.Tensor
    joint_ratio_quantiles: torch.Tensor
    minimum_joint_ratio: float
    maximum_joint_ratio: float
    nonfinite_joint_ratio_count: int
    mean_common_gradient_norm_before_clip: float
    max_common_gradient_norm_before_clip: float
    mean_prefix_gradient_norm_before_clip: float
    max_prefix_gradient_norm_before_clip: float
    mean_actor_gradient_norm_before_clip: float
    max_actor_gradient_norm_before_clip: float
    mean_global_critic_gradient_norm_before_clip: float
    max_global_critic_gradient_norm_before_clip: float
    mean_type_prefix_gradient_norm_before_clip: float
    max_type_prefix_gradient_norm_before_clip: float
    mean_session_prefix_gradient_norm_before_clip: float
    max_session_prefix_gradient_norm_before_clip: float
    reward_advantage_scale: float
    constraint_advantage_scales: torch.Tensor
    reward_return_scale: float
    constraint_return_scales: torch.Tensor
    mean_normalized_reward_advantages_by_factor: torch.Tensor
    positive_normalized_reward_advantage_fractions: torch.Tensor
    merge_transition_count: int
    profile_transition_count: int
    single_session_merge_transition_count: int
    multi_session_merge_transition_count: int
    session_prefix_reward_target_variance: float
    session_prefix_constraint_target_variance: torch.Tensor
    dual_values_used: torch.Tensor

    @property
    def mean_approximate_kl(self) -> float:
        return self.mean_joint_approximate_kl

    @property
    def mean_clip_fraction(self) -> float:
        return self.mean_joint_clip_fraction

    @property
    def max_minibatch_approximate_kl(self) -> float:
        return self.max_minibatch_joint_approximate_kl

    @property
    def mean_gradient_norm_before_clip(self) -> float:
        return self.mean_common_gradient_norm_before_clip

    @property
    def max_gradient_norm_before_clip(self) -> float:
        return self.max_common_gradient_norm_before_clip


@dataclass(frozen=True, slots=True)
class _CommonTraceLoss:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    reward_surrogate: torch.Tensor
    reward_surrogates_by_factor: torch.Tensor
    constraint_surrogates: torch.Tensor
    constraint_surrogates_by_factor: torch.Tensor
    entropy: torch.Tensor
    reward_value_loss: torch.Tensor
    constraint_value_loss: torch.Tensor
    tenant_value_loss: torch.Tensor
    communication_value_loss: torch.Tensor
    type_reward_value_loss: torch.Tensor
    type_constraint_value_loss: torch.Tensor
    session_reward_value_loss: torch.Tensor
    session_constraint_value_loss: torch.Tensor
    policy_surrogates: FactorizedPolicySurrogates


class CommonTracePPO(ConstrainedPPO):
    """Optimize factor-specific actor surrogates while retaining the original CMDP returns."""

    def __init__(self, model: CommonTraceActorCritic, config: ConstrainedPPOConfig) -> None:
        if not isinstance(model, CommonTraceActorCritic):
            raise PPOValidationError("CommonTracePPO requires CommonTraceActorCritic")
        super().__init__(model, config)
        prefix = model.prefix_value
        self.type_prefix_parameters = tuple(
            parameter for module in (
                prefix.type_trunk, prefix.type_reward_head, prefix.type_constraint_head,
            ) for parameter in module.parameters()
        )
        self.session_prefix_parameters = tuple(
            parameter for module in (
                prefix.merge_session_trunk, prefix.merge_session_reward_head,
                prefix.merge_session_constraint_head,
            ) for parameter in module.parameters()
        )
        self.prefix_parameters = (
            *self.type_prefix_parameters, *self.session_prefix_parameters,
        )
        prefix_ids = {id(parameter) for parameter in self.prefix_parameters}
        self.common_parameters = tuple(
            parameter for parameter in model.parameters()
            if id(parameter) not in prefix_ids
        )
        if (
            not self.common_parameters or not self.type_prefix_parameters
            or not self.session_prefix_parameters
        ):
            raise PPOValidationError(
                "common-trace optimizer requires common, type-prefix, and session-prefix parameters"
            )
        grouped = (*self.common_parameters, *self.prefix_parameters)
        if (
            len({id(parameter) for parameter in grouped}) != len(grouped)
            or len(grouped) != len(tuple(model.parameters()))
        ):
            raise PPOValidationError(
                "common-trace optimizer parameter groups must be disjoint and complete"
            )
        self.optimizer = torch.optim.Adam(
            ({"params": self.common_parameters}, {"params": self.prefix_parameters}),
            lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon,
        )

    def _validate_rollout(self, rollout: PreparedRollout) -> None:
        super()._validate_rollout(rollout)
        if rollout.factor_credit is None:
            raise PPOValidationError("CommonTracePPO requires rollout factor-credit state")
        if rollout.factor_credit.normalization_epsilon != self.config.normalization.epsilon:
            raise PPOValidationError("factor-credit normalization epsilon does not match algorithm config")

    @staticmethod
    def _selected_prefix_values(prefix_values, actions: FactorizedActionIndices, constraint_count: int):
        batch = actions.action_type.shape[0]
        rows = torch.arange(batch, device=actions.action_type.device)
        type_reward = prefix_values.type_reward_values[rows, actions.action_type]
        type_constraints = prefix_values.type_constraint_values[rows, actions.action_type]
        session_width = prefix_values.merge_session_reward_values.shape[1]
        if session_width == 0:
            session_reward = torch.zeros(batch, dtype=torch.float32, device=actions.action_type.device)
            session_constraints = torch.zeros(batch, constraint_count, dtype=torch.float32, device=actions.action_type.device)
        else:
            sessions = actions.merge_session.clamp_min(0)
            session_reward = prefix_values.merge_session_reward_values[rows, sessions]
            session_constraints = prefix_values.merge_session_constraint_values[rows, sessions]
        return type_reward, type_constraints, session_reward, session_constraints

    def _minibatch_loss(
        self, rollout: PreparedRollout, indices: torch.Tensor,
        actor_duals: torch.Tensor, reward_return_scale: torch.Tensor,
        constraint_return_scales: torch.Tensor,
    ) -> tuple[_CommonTraceLoss, float, float, float, float, float, float]:
        factor_state = rollout.factor_credit
        if factor_state is None:
            raise PPOValidationError("factor-credit minibatch state is absent")
        minibatch = rollout.minibatch(indices, rollout.reward_advantages, rollout.constraint_advantages)
        factor = factor_state.minibatch(indices)
        policy_batch = build_policy_batch(minibatch.observations, device=self.device)
        actions = FactorizedActionIndices(
            minibatch.actions.action_type.to(self.device),
            minibatch.actions.merge_session.to(self.device),
            minibatch.actions.profile.to(self.device),
        )
        _, logits, values, prefix_values = self.model(policy_batch)
        components = self.model.policy.evaluate_components(logits, policy_batch, actions)
        new_factor_logs = torch.stack(
            (components.action_type, components.merge_session, components.profile), dim=1,
        )
        applicability = factor.applicability.to(self.device)
        current_applicability = torch.stack((
            torch.ones_like(components.merge_session_applicable),
            components.merge_session_applicable,
            components.profile_applicable,
        ), dim=1)
        if not torch.equal(applicability, current_applicability):
            raise PPOValidationError("current factor applicability does not match frozen rollout actions")
        common_reward = factor.normalized_reward_advantages[:, 0].to(self.device)
        common_reward_by_factor = common_reward.unsqueeze(1) * applicability.to(torch.float32)
        surrogates = build_factorized_policy_surrogates(
            new_log_probabilities=new_factor_logs,
            old_log_probabilities=factor.old_log_probabilities.to(self.device),
            applicability=applicability,
            reward_advantages=common_reward_by_factor,
            constraint_advantages=factor.normalized_constraint_advantages.to(self.device),
            clip_ratio=self.config.ppo.clip_ratio,
        )
        reward_surrogate = surrogates.reward_by_factor.sum()
        constraint_surrogates = surrogates.constraints_by_factor.sum(dim=0)
        entropy = self.model.policy.entropy(logits, policy_batch).mean()
        actor_loss = (
            -reward_surrogate + torch.dot(actor_duals.detach(), constraint_surrogates)
            - self.config.ppo.entropy_coefficient * entropy
        )

        constraint_values = torch.cat(
            (values.sensing_sla_values, values.communication_qos_values), dim=1,
        )
        old_constraints = minibatch.old_constraint_values.to(self.device)
        constraint_returns = minibatch.constraint_returns.to(self.device)
        reward_value_loss = clipped_value_loss(
            values.reward_value, minibatch.old_reward_values.to(self.device),
            minibatch.reward_returns.to(self.device), self.config.ppo.value_clip_ratio,
            reward_return_scale,
        )
        constraint_value_loss = clipped_value_loss(
            constraint_values, old_constraints, constraint_returns,
            self.config.ppo.value_clip_ratio, constraint_return_scales,
        )
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        tenant_count = rollout.layout.tenant_count
        tenant_value_loss = clipped_value_loss(
            constraint_values[:, :tenant_count], old_constraints[:, :tenant_count],
            constraint_returns[:, :tenant_count], self.config.ppo.value_clip_ratio,
            constraint_return_scales[:tenant_count],
        ) if tenant_count else zero
        communication_value_loss = clipped_value_loss(
            constraint_values[:, tenant_count:], old_constraints[:, tenant_count:],
            constraint_returns[:, tenant_count:], self.config.ppo.value_clip_ratio,
            constraint_return_scales[tenant_count:],
        ) if tenant_count < self.constraint_count else zero

        type_reward, type_constraints, session_reward, session_constraints = self._selected_prefix_values(
            prefix_values, actions, self.constraint_count,
        )
        reward_returns = minibatch.reward_returns.to(self.device)
        type_reward_value_loss = clipped_value_loss(
            type_reward, factor.old_action_type_reward_values.to(self.device),
            reward_returns, self.config.ppo.value_clip_ratio, reward_return_scale,
        )
        type_constraint_value_loss = clipped_value_loss(
            type_constraints, factor.old_action_type_constraint_values.to(self.device),
            constraint_returns, self.config.ppo.value_clip_ratio,
            constraint_return_scales,
        )
        session_mask = actions.merge_session >= 0
        session_reward_value_loss = masked_clipped_value_loss(
            session_reward, factor.old_merge_session_reward_values.to(self.device),
            reward_returns, session_mask, self.config.ppo.value_clip_ratio,
            reward_return_scale,
        )
        session_constraint_value_loss = masked_clipped_value_loss(
            session_constraints, factor.old_merge_session_constraint_values.to(self.device),
            constraint_returns, session_mask, self.config.ppo.value_clip_ratio,
            constraint_return_scales,
        )
        global_critic_loss = (
            self.config.ppo.reward_value_coefficient * reward_value_loss
            + self.config.ppo.constraint_value_coefficient * constraint_value_loss
        )
        type_prefix_critic_loss = (
            self.config.ppo.reward_value_coefficient * type_reward_value_loss
            + self.config.ppo.constraint_value_coefficient * type_constraint_value_loss
        )
        session_prefix_critic_loss = (
            self.config.ppo.reward_value_coefficient * session_reward_value_loss
            + self.config.ppo.constraint_value_coefficient * session_constraint_value_loss
        )
        total_loss = actor_loss + global_critic_loss + type_prefix_critic_loss + session_prefix_critic_loss
        loss = _CommonTraceLoss(
            total_loss, actor_loss, reward_surrogate, surrogates.reward_by_factor,
            constraint_surrogates, surrogates.constraints_by_factor, entropy,
            reward_value_loss, constraint_value_loss, tenant_value_loss,
            communication_value_loss, type_reward_value_loss, type_constraint_value_loss,
            session_reward_value_loss, session_constraint_value_loss, surrogates,
        )
        if not bool(torch.isfinite(total_loss)):
            raise PPOValidationError("total common-trace PPO loss is non-finite")
        actor_parameters = (*self.encoder_parameters, *self.policy_parameters)
        global_critic_parameters = (*self.encoder_parameters, *self.value_parameters)
        actor_gradients, actor_norm = _clipped_objective_gradients(
            actor_loss, actor_parameters, self.config.ppo.max_gradient_norm,
            retain_graph=True,
        )
        global_critic_gradients, global_critic_norm = _clipped_objective_gradients(
            global_critic_loss, global_critic_parameters,
            self.config.ppo.max_gradient_norm, retain_graph=True,
        )
        type_prefix_gradients, type_prefix_norm = _clipped_objective_gradients(
            type_prefix_critic_loss, self.type_prefix_parameters,
            self.config.ppo.max_gradient_norm, retain_graph=True,
        )
        session_prefix_gradients, session_prefix_norm = _clipped_objective_gradients(
            session_prefix_critic_loss, self.session_prefix_parameters,
            self.config.ppo.max_gradient_norm, retain_graph=False,
        )
        encoder_count = len(self.encoder_parameters)
        self.optimizer.zero_grad(set_to_none=True)
        _assign_parameter_gradients(
            self.encoder_parameters,
            actor_gradients[:encoder_count],
            global_critic_gradients[:encoder_count],
        )
        _assign_parameter_gradients(
            self.policy_parameters, actor_gradients[encoder_count:],
        )
        _assign_parameter_gradients(
            self.value_parameters, global_critic_gradients[encoder_count:],
        )
        _assign_parameter_gradients(
            self.type_prefix_parameters, type_prefix_gradients,
        )
        _assign_parameter_gradients(
            self.session_prefix_parameters, session_prefix_gradients,
        )
        all_gradients = tuple(
            parameter.grad for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if not all_gradients or any(
            not bool(torch.isfinite(gradient).all()) for gradient in all_gradients
        ):
            raise PPOValidationError("composed common-trace gradients are missing or non-finite")
        composed_norm = torch.nn.utils.clip_grad_norm_(
            self.common_parameters, self.config.ppo.max_gradient_norm,
        )
        composed_norm_value = _finite_scalar(
            composed_norm, "composed common gradient norm",
        )
        torch.nn.utils.clip_grad_norm_(
            self.type_prefix_parameters, self.config.ppo.max_gradient_norm,
        )
        torch.nn.utils.clip_grad_norm_(
            self.session_prefix_parameters, self.config.ppo.max_gradient_norm,
        )
        if any(not bool(torch.isfinite(gradient).all()) for gradient in all_gradients):
            raise PPOValidationError(
                "final common-trace clipping produced non-finite gradients"
            )
        self.optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
            raise PPOValidationError("common-trace optimizer step produced non-finite parameters")
        self.optimizer_step_count += 1
        prefix_norm = sqrt(type_prefix_norm ** 2 + session_prefix_norm ** 2)
        return (
            loss, composed_norm_value, prefix_norm, actor_norm, global_critic_norm,
            type_prefix_norm, session_prefix_norm,
        )

    @staticmethod
    def _rollout_structure(
        rollout: PreparedRollout,
    ) -> tuple[int, int, int, int, float, torch.Tensor]:
        factor = rollout.factor_credit
        if factor is None:
            raise PPOValidationError("factor-credit rollout state is absent")
        merge_mask = torch.tensor([
            action.merge_session_index >= 0 for action in rollout.stored_actions
        ], dtype=torch.bool)
        profile_mask = factor.applicability[:, 2]
        merge_count = int(merge_mask.sum())
        profile_count = int(profile_mask.sum())
        single = multi = 0
        for observation, is_merge in zip(
            rollout.observations, merge_mask.tolist(), strict=True,
        ):
            if not is_merge:
                continue
            feasible = sum(
                value for _, value
                in observation.set_view.action_masks.merge_session_mask
            )
            single += int(feasible == 1)
            multi += int(feasible > 1)
        reward_variance = (
            float(rollout.reward_returns[merge_mask].var(unbiased=False))
            if merge_count else 0.0
        )
        constraint_variance = (
            rollout.constraint_returns[merge_mask].var(dim=0, unbiased=False)
            if merge_count
            else torch.zeros(
                rollout.layout.constraint_count, dtype=torch.float32,
            )
        )
        return (
            merge_count, profile_count, single, multi,
            reward_variance, constraint_variance,
        )

    @staticmethod
    def _factor_advantage_diagnostics(
        factor_state,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means = torch.zeros(3, dtype=torch.float32)
        positive = torch.zeros(3, dtype=torch.float32)
        common = factor_state.normalized_reward_advantages[:, 0]
        for index in range(3):
            mask = factor_state.applicability[:, index]
            if bool(mask.any()):
                values = common[mask]
                means[index] = values.mean()
                positive[index] = values.gt(0.0).to(torch.float32).mean()
        return means, positive

    def optimize_rollout(
        self, rollout: PreparedRollout, *,
        generator: torch.Generator | None = None,
    ) -> CommonTracePPOUpdateMetrics:
        self._validate_rollout(rollout)
        factor_state = rollout.factor_credit
        if factor_state is None:
            raise PPOValidationError("factor-credit rollout state is absent")
        factor_advantage_means, factor_positive_fractions = (
            self._factor_advantage_diagnostics(factor_state)
        )
        advantage_scales = signal_scales(
            rollout.reward_advantages, rollout.constraint_advantages,
            self.config.normalization.epsilon,
        )
        return_scales = signal_scales(
            rollout.reward_returns, rollout.constraint_returns,
            self.config.normalization.epsilon,
        )
        raw_duals = self.dual_values.detach().clone()
        actor_duals = raw_duals * (
            advantage_scales.constraints.to(self.device)
            / advantage_scales.reward.to(self.device)
        )
        reward_return_scale = return_scales.reward.to(self.device)
        constraint_return_scales = return_scales.constraints.to(self.device)
        n = rollout.transition_count
        q = self.constraint_count
        scalar_names = (
            "total", "actor", "reward", "reward_value", "constraint_value",
            "tenant_value", "communication_value", "type_reward_value",
            "type_constraint_value", "session_reward_value",
            "session_constraint_value", "entropy", "joint_kl", "joint_clip",
            "common_gradient", "prefix_gradient", "actor_gradient",
            "global_critic_gradient", "type_prefix_gradient",
            "session_prefix_gradient",
        )
        weighted = {name: 0.0 for name in scalar_names}
        reward_factor_sum = torch.zeros(3, dtype=torch.float64)
        constraint_sum = torch.zeros(q, dtype=torch.float64)
        constraint_factor_sum = torch.zeros(3, q, dtype=torch.float64)
        factor_clip_counts = torch.zeros(3, dtype=torch.float64)
        factor_applicable_counts = torch.zeros(3, dtype=torch.float64)
        joint_ratios: list[torch.Tensor] = []
        total_samples = 0
        minibatches = 0
        epochs_completed = 0
        optimizer_steps = 0
        max_common = 0.0
        max_prefix = 0.0
        max_actor = 0.0
        max_global_critic = 0.0
        max_type_prefix = 0.0
        max_session_prefix = 0.0
        maximum_minibatch_kl = 0.0
        early_stopped = False
        for epoch in range(self.config.ppo.epochs_per_rollout):
            permutation = torch.randperm(n, generator=generator)
            epoch_kl_sum = 0.0
            epoch_samples = 0
            for start in range(0, n, self.config.ppo.minibatch_decisions):
                indices = permutation[start:start + self.config.ppo.minibatch_decisions]
                (
                    loss, common_norm, prefix_norm, actor_norm,
                    global_critic_norm, type_prefix_norm, session_prefix_norm,
                ) = self._minibatch_loss(
                    rollout, indices, actor_duals,
                    reward_return_scale, constraint_return_scales,
                )
                size = int(indices.numel())
                diagnostics = {
                    "total": loss.total_loss,
                    "actor": loss.actor_loss,
                    "reward": loss.reward_surrogate,
                    "reward_value": loss.reward_value_loss,
                    "constraint_value": loss.constraint_value_loss,
                    "tenant_value": loss.tenant_value_loss,
                    "communication_value": loss.communication_value_loss,
                    "type_reward_value": loss.type_reward_value_loss,
                    "type_constraint_value": loss.type_constraint_value_loss,
                    "session_reward_value": loss.session_reward_value_loss,
                    "session_constraint_value": loss.session_constraint_value_loss,
                    "entropy": loss.entropy,
                    "joint_kl": loss.policy_surrogates.approximate_joint_kl,
                    "joint_clip": loss.policy_surrogates.joint_clip_fraction,
                }
                for name, value in diagnostics.items():
                    weighted[name] += _finite_scalar(value, name) * size
                for name, value in (
                    ("common_gradient", common_norm),
                    ("prefix_gradient", prefix_norm),
                    ("actor_gradient", actor_norm),
                    ("global_critic_gradient", global_critic_norm),
                    ("type_prefix_gradient", type_prefix_norm),
                    ("session_prefix_gradient", session_prefix_norm),
                ):
                    weighted[name] += value * size
                reward_factor_sum += (
                    loss.reward_surrogates_by_factor.detach()
                    .cpu().to(torch.float64) * size
                )
                constraint_sum += (
                    loss.constraint_surrogates.detach()
                    .cpu().to(torch.float64) * size
                )
                constraint_factor_sum += (
                    loss.constraint_surrogates_by_factor.detach()
                    .cpu().to(torch.float64) * size
                )
                factor = factor_state.minibatch(indices)
                ratios = loss.policy_surrogates.ratios.detach().cpu()
                applicable = factor.applicability
                clipped = (
                    ((ratios - 1.0).abs() > self.config.ppo.clip_ratio)
                    & applicable
                )
                factor_clip_counts += clipped.sum(dim=0).to(torch.float64)
                factor_applicable_counts += applicable.sum(dim=0).to(torch.float64)
                joint_ratios.append(
                    loss.policy_surrogates.joint_ratio.detach().cpu(),
                )
                minibatch_kl = _finite_scalar(
                    loss.policy_surrogates.approximate_joint_kl,
                    "joint approximate KL",
                )
                epoch_kl_sum += minibatch_kl * size
                maximum_minibatch_kl = max(maximum_minibatch_kl, minibatch_kl)
                epoch_samples += size
                total_samples += size
                minibatches += 1
                optimizer_steps += 1
                max_common = max(max_common, common_norm)
                max_prefix = max(max_prefix, prefix_norm)
                max_actor = max(max_actor, actor_norm)
                max_global_critic = max(max_global_critic, global_critic_norm)
                max_type_prefix = max(max_type_prefix, type_prefix_norm)
                max_session_prefix = max(max_session_prefix, session_prefix_norm)
            epochs_completed += 1
            if (
                epoch_kl_sum / epoch_samples > self.config.ppo.target_kl
                and epoch + 1 < self.config.ppo.epochs_per_rollout
            ):
                early_stopped = True
                break
        if total_samples < 1 or not joint_ratios:
            raise PPOValidationError(
                "common-trace PPO update completed no optimization samples"
            )
        ratio_values = torch.cat(joint_ratios)
        quantiles = torch.quantile(
            ratio_values, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]),
        )
        factor_clip_fractions = torch.where(
            factor_applicable_counts > 0,
            factor_clip_counts / factor_applicable_counts.clamp_min(1.0),
            torch.zeros(3, dtype=torch.float64),
        ).to(torch.float32)
        (
            merge_count, profile_count, single, multi,
            target_variance, constraint_target_variance,
        ) = self._rollout_structure(rollout)
        return CommonTracePPOUpdateMetrics(
            n, epochs_completed, minibatches, optimizer_steps, early_stopped,
            weighted["total"] / total_samples,
            weighted["actor"] / total_samples,
            weighted["reward"] / total_samples,
            (reward_factor_sum / total_samples).to(torch.float32),
            (constraint_sum / total_samples).to(torch.float32),
            (constraint_factor_sum / total_samples).to(torch.float32),
            weighted["reward_value"] / total_samples,
            weighted["constraint_value"] / total_samples,
            weighted["tenant_value"] / total_samples,
            weighted["communication_value"] / total_samples,
            weighted["type_reward_value"] / total_samples,
            weighted["type_constraint_value"] / total_samples,
            weighted["session_reward_value"] / total_samples,
            weighted["session_constraint_value"] / total_samples,
            weighted["entropy"] / total_samples,
            weighted["joint_kl"] / total_samples,
            maximum_minibatch_kl,
            weighted["joint_clip"] / total_samples,
            factor_clip_fractions,
            quantiles.to(torch.float32),
            float(ratio_values.min()),
            float(ratio_values.max()),
            0,
            weighted["common_gradient"] / total_samples,
            max_common,
            weighted["prefix_gradient"] / total_samples,
            max_prefix,
            weighted["actor_gradient"] / total_samples,
            max_actor,
            weighted["global_critic_gradient"] / total_samples,
            max_global_critic,
            weighted["type_prefix_gradient"] / total_samples,
            max_type_prefix,
            weighted["session_prefix_gradient"] / total_samples,
            max_session_prefix,
            float(advantage_scales.reward),
            _cpu_float(advantage_scales.constraints),
            float(return_scales.reward),
            _cpu_float(return_scales.constraints),
            factor_advantage_means,
            factor_positive_fractions,
            merge_count,
            profile_count,
            single,
            multi,
            target_variance,
            constraint_target_variance.to(torch.float32),
            _cpu_float(actor_duals),
        )