"""Primal-dual optimization core for the joint-credit PPO baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from isac_ssc.algorithms.buffers import PreparedRollout
from isac_ssc.algorithms.losses import ConstrainedPPOLoss, build_constrained_ppo_loss, normalize_advantages, signal_scales
from isac_ssc.models.policy import EdgeFreeSetActorCritic, FactorizedActionIndices, build_policy_batch
from isac_ssc.utils.config import ConstrainedPPOConfig


class PPOValidationError(ValueError):
    """Raised when constrained-PPO optimization cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    transitions: int
    epochs_completed: int
    minibatches_completed: int
    optimizer_steps: int
    early_stopped_for_kl: bool
    mean_total_loss: float
    mean_actor_loss: float
    mean_reward_surrogate: float
    mean_constraint_surrogates: torch.Tensor
    mean_reward_value_loss: float
    mean_constraint_value_loss: float
    mean_tenant_value_loss: float
    mean_communication_value_loss: float
    mean_entropy: float
    mean_approximate_kl: float
    max_minibatch_approximate_kl: float
    mean_clip_fraction: float
    mean_gradient_norm_before_clip: float
    max_gradient_norm_before_clip: float
    mean_actor_gradient_norm_before_clip: float
    max_actor_gradient_norm_before_clip: float
    mean_critic_gradient_norm_before_clip: float
    max_critic_gradient_norm_before_clip: float
    reward_advantage_scale: float
    constraint_advantage_scales: torch.Tensor
    reward_return_scale: float
    constraint_return_scales: torch.Tensor
    dual_values_used: torch.Tensor


@dataclass(frozen=True, slots=True)
class DualUpdateMetrics:
    completed_episodes: int
    mean_episode_residuals: torch.Tensor
    dual_values_before: torch.Tensor
    dual_values_after: torch.Tensor


def _cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    result = float(value.detach().cpu())
    if not isfinite(result):
        raise PPOValidationError(f"{name} must be finite")
    return result


def _clipped_objective_gradients(
    loss: torch.Tensor, parameters: tuple[nn.Parameter, ...],
    maximum_norm: float, *, retain_graph: bool,
) -> tuple[tuple[torch.Tensor | None, ...], float]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters), 0.0
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True,
    )
    present = tuple(gradient for gradient in gradients if gradient is not None)
    if not present:
        return gradients, 0.0
    if any(not bool(torch.isfinite(gradient).all()) for gradient in present):
        raise PPOValidationError("objective gradients are non-finite")
    norm = torch.sqrt(sum(gradient.square().sum() for gradient in present))
    norm_value = _finite_scalar(norm, "objective gradient norm")
    scale = min(1.0, maximum_norm / max(norm_value, 1e-12))
    return tuple(
        None if gradient is None else gradient.detach() * scale
        for gradient in gradients
    ), norm_value


def _assign_parameter_gradients(
    parameters: tuple[nn.Parameter, ...],
    *gradient_groups: tuple[torch.Tensor | None, ...],
) -> None:
    if any(len(group) != len(parameters) for group in gradient_groups):
        raise PPOValidationError("gradient groups do not match parameters")
    for index, parameter in enumerate(parameters):
        contributions = [
            group[index] for group in gradient_groups if group[index] is not None
        ]
        parameter.grad = None if not contributions else sum(contributions[1:], contributions[0].clone())


class ConstrainedPPO:
    """Optimize one shared actor-critic while holding duals fixed per rollout update."""

    def __init__(self, model: EdgeFreeSetActorCritic, config: ConstrainedPPOConfig) -> None:
        if not isinstance(model, EdgeFreeSetActorCritic) or not isinstance(config, ConstrainedPPOConfig):
            raise PPOValidationError("validated edge-free actor-critic and algorithm config are required")
        parameters = tuple(model.parameters())
        if not parameters:
            raise PPOValidationError("model must contain trainable parameters")
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1 or next(iter(devices)).type != config.device:
            raise PPOValidationError("model device must match algorithm runtime device")
        if any(parameter.dtype is not torch.float32 for parameter in parameters):
            raise PPOValidationError("model parameters must use float32")
        self.model, self.config = model, config
        self.device = next(iter(devices))
        self.constraint_count = model.value.number_of_tenants + model.value.number_of_communication_users
        self.encoder_parameters = tuple(model.encoder.parameters())
        self.policy_parameters = tuple(model.policy.parameters())
        self.value_parameters = tuple(model.value.parameters())
        core = (*self.encoder_parameters, *self.policy_parameters, *self.value_parameters)
        if (
            not all((self.encoder_parameters, self.policy_parameters, self.value_parameters))
            or len({id(parameter) for parameter in core}) != len(core)
        ):
            raise PPOValidationError("actor-critic parameter groups must be non-empty and disjoint")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
        self.dual_values = torch.full((self.constraint_count,), config.dual.initial_value, dtype=torch.float32, device=self.device)
        self.optimizer_step_count = 0

    def _validate_rollout(self, rollout: PreparedRollout) -> None:
        if not isinstance(rollout, PreparedRollout) or rollout.transition_count < 1:
            raise PPOValidationError("optimize_rollout requires a non-empty PreparedRollout")
        if rollout.layout.constraint_count != self.constraint_count:
            raise PPOValidationError("rollout constraint dimension does not match model critics")
        if rollout.layout.tenant_count != self.model.value.number_of_tenants:
            raise PPOValidationError("rollout tenant dimension does not match model critics")
        if bool((self.dual_values < 0.0).any()) or not bool(torch.isfinite(self.dual_values).all()):
            raise PPOValidationError("dual values must be finite and non-negative")

    def _minibatch_loss(
        self, rollout: PreparedRollout, indices: torch.Tensor,
        reward_advantages: torch.Tensor, constraint_advantages: torch.Tensor,
        actor_duals: torch.Tensor, reward_return_scale: torch.Tensor,
        constraint_return_scales: torch.Tensor,
    ) -> tuple[ConstrainedPPOLoss, float, float, float]:
        minibatch = rollout.minibatch(indices, reward_advantages, constraint_advantages)
        policy_batch = build_policy_batch(minibatch.observations, device=self.device)
        actions = FactorizedActionIndices(
            minibatch.actions.action_type.to(self.device),
            minibatch.actions.merge_session.to(self.device),
            minibatch.actions.profile.to(self.device),
        )
        _, logits, values = self.model(policy_batch)
        new_log_probabilities, entropy = self.model.policy.evaluate(logits, policy_batch, actions)
        constraint_values = torch.cat((values.sensing_sla_values, values.communication_qos_values), dim=1)
        loss = build_constrained_ppo_loss(
            new_log_probabilities=new_log_probabilities,
            old_log_probabilities=minibatch.old_log_probabilities.to(self.device),
            reward_advantages=minibatch.reward_advantages.to(self.device),
            constraint_advantages=minibatch.constraint_advantages.to(self.device),
            entropy=entropy, new_reward_values=values.reward_value,
            old_reward_values=minibatch.old_reward_values.to(self.device),
            reward_returns=minibatch.reward_returns.to(self.device),
            new_constraint_values=constraint_values,
            old_constraint_values=minibatch.old_constraint_values.to(self.device),
            constraint_returns=minibatch.constraint_returns.to(self.device),
            dual_values=actor_duals, tenant_count=rollout.layout.tenant_count,
            reward_return_scale=reward_return_scale,
            constraint_return_scales=constraint_return_scales,
            clip_ratio=self.config.ppo.clip_ratio,
            value_clip_ratio=self.config.ppo.value_clip_ratio,
            entropy_coefficient=self.config.ppo.entropy_coefficient,
            reward_value_coefficient=self.config.ppo.reward_value_coefficient,
            constraint_value_coefficient=self.config.ppo.constraint_value_coefficient,
        )
        if not bool(torch.isfinite(loss.total_loss)):
            raise PPOValidationError("total PPO loss is non-finite")
        actor_parameters = (*self.encoder_parameters, *self.policy_parameters)
        critic_parameters = (*self.encoder_parameters, *self.value_parameters)
        critic_loss = (
            self.config.ppo.reward_value_coefficient * loss.reward_value_loss
            + self.config.ppo.constraint_value_coefficient * loss.constraint_value_loss
        )
        actor_gradients, actor_norm = _clipped_objective_gradients(
            loss.actor_loss, actor_parameters, self.config.ppo.max_gradient_norm,
            retain_graph=True,
        )
        critic_gradients, critic_norm = _clipped_objective_gradients(
            critic_loss, critic_parameters, self.config.ppo.max_gradient_norm,
            retain_graph=False,
        )
        encoder_count = len(self.encoder_parameters)
        self.optimizer.zero_grad(set_to_none=True)
        _assign_parameter_gradients(
            self.encoder_parameters,
            actor_gradients[:encoder_count], critic_gradients[:encoder_count],
        )
        _assign_parameter_gradients(
            self.policy_parameters, actor_gradients[encoder_count:],
        )
        _assign_parameter_gradients(
            self.value_parameters, critic_gradients[encoder_count:],
        )
        gradients = tuple(
            parameter.grad for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise PPOValidationError("composed actor-critic gradients are missing or non-finite")
        composed_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.ppo.max_gradient_norm,
        )
        composed_norm_value = _finite_scalar(
            composed_norm, "composed actor-critic gradient norm",
        )
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise PPOValidationError("final actor-critic clipping produced non-finite gradients")
        self.optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
            raise PPOValidationError("optimizer step produced non-finite parameters")
        self.optimizer_step_count += 1
        return loss, composed_norm_value, actor_norm, critic_norm

    def optimize_rollout(
        self, rollout: PreparedRollout, *, generator: torch.Generator | None = None,
    ) -> PPOUpdateMetrics:
        self._validate_rollout(rollout)
        normalized = normalize_advantages(
            rollout.reward_advantages, rollout.constraint_advantages,
            self.config.normalization.epsilon,
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
        n, q = rollout.transition_count, self.constraint_count
        weighted = {name: 0.0 for name in (
            "total", "actor", "reward_surrogate", "reward_value", "constraint_value",
            "tenant_value", "communication_value", "entropy", "kl", "clip_fraction",
            "gradient_norm", "actor_gradient_norm", "critic_gradient_norm",
        )}
        constraint_sum = torch.zeros(q, dtype=torch.float64)
        total_samples = minibatches = epochs_completed = optimizer_steps = 0
        maximum_gradient_norm = 0.0
        maximum_actor_gradient_norm = 0.0
        maximum_critic_gradient_norm = 0.0
        maximum_minibatch_kl = 0.0
        early_stopped = False
        reward_return_scale = return_scales.reward.to(self.device)
        constraint_return_scales = return_scales.constraints.to(self.device)
        for epoch in range(self.config.ppo.epochs_per_rollout):
            permutation = torch.randperm(n, generator=generator)
            epoch_kl_sum, epoch_samples = 0.0, 0
            for start in range(0, n, self.config.ppo.minibatch_decisions):
                indices = permutation[start:start + self.config.ppo.minibatch_decisions]
                loss, gradient_norm, actor_gradient_norm, critic_gradient_norm = self._minibatch_loss(
                    rollout, indices, normalized.reward, normalized.constraints,
                    actor_duals, reward_return_scale, constraint_return_scales,
                )
                size = int(indices.numel())
                diagnostics = {
                    "total": loss.total_loss, "actor": loss.actor_loss,
                    "reward_surrogate": loss.reward_surrogate,
                    "reward_value": loss.reward_value_loss,
                    "constraint_value": loss.constraint_value_loss,
                    "tenant_value": loss.tenant_value_loss,
                    "communication_value": loss.communication_value_loss,
                    "entropy": loss.entropy, "kl": loss.approximate_kl,
                    "clip_fraction": loss.clip_fraction,
                }
                for name, value in diagnostics.items():
                    weighted[name] += _finite_scalar(value, name) * size
                weighted["gradient_norm"] += gradient_norm * size
                weighted["actor_gradient_norm"] += actor_gradient_norm * size
                weighted["critic_gradient_norm"] += critic_gradient_norm * size
                constraint_sum += loss.constraint_surrogates.detach().to(
                    dtype=torch.float64, device="cpu",
                ) * size
                minibatch_kl = _finite_scalar(loss.approximate_kl, "approximate KL")
                epoch_kl_sum += minibatch_kl * size
                maximum_minibatch_kl = max(maximum_minibatch_kl, minibatch_kl)
                epoch_samples += size
                total_samples += size
                minibatches += 1
                optimizer_steps += 1
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                maximum_actor_gradient_norm = max(maximum_actor_gradient_norm, actor_gradient_norm)
                maximum_critic_gradient_norm = max(maximum_critic_gradient_norm, critic_gradient_norm)
            epochs_completed += 1
            epoch_mean_kl = epoch_kl_sum / epoch_samples
            if (
                epoch_mean_kl > self.config.ppo.target_kl
                and epoch + 1 < self.config.ppo.epochs_per_rollout
            ):
                early_stopped = True
                break
        if total_samples < 1:
            raise PPOValidationError("PPO update completed no optimization samples")
        return PPOUpdateMetrics(
            n, epochs_completed, minibatches, optimizer_steps, early_stopped,
            weighted["total"] / total_samples, weighted["actor"] / total_samples,
            weighted["reward_surrogate"] / total_samples,
            (constraint_sum / total_samples).to(torch.float32),
            weighted["reward_value"] / total_samples,
            weighted["constraint_value"] / total_samples,
            weighted["tenant_value"] / total_samples,
            weighted["communication_value"] / total_samples,
            weighted["entropy"] / total_samples,
            weighted["kl"] / total_samples,
            maximum_minibatch_kl,
            weighted["clip_fraction"] / total_samples,
            weighted["gradient_norm"] / total_samples,
            maximum_gradient_norm,
            weighted["actor_gradient_norm"] / total_samples,
            maximum_actor_gradient_norm,
            weighted["critic_gradient_norm"] / total_samples,
            maximum_critic_gradient_norm,
            float(advantage_scales.reward),
            _cpu_float(advantage_scales.constraints),
            float(return_scales.reward),
            _cpu_float(return_scales.constraints),
            _cpu_float(actor_duals),
        )

    def update_duals(self, episode_constraint_totals: torch.Tensor) -> DualUpdateMetrics:
        if not isinstance(episode_constraint_totals, torch.Tensor) or episode_constraint_totals.dtype is not torch.float32:
            raise PPOValidationError("episode_constraint_totals must be a float32 tensor")
        if (
            episode_constraint_totals.ndim != 2
            or episode_constraint_totals.shape[0] < 1
            or episode_constraint_totals.shape[1] != self.constraint_count
        ):
            raise PPOValidationError("episode_constraint_totals must be non-empty [E, Q]")
        if episode_constraint_totals.requires_grad or not bool(torch.isfinite(episode_constraint_totals).all()):
            raise PPOValidationError("episode_constraint_totals must be detached and finite")
        mean_residuals = episode_constraint_totals.to(self.device).mean(dim=0)
        before = self.dual_values.detach().clone()
        after = (
            before + self.config.dual.learning_rate * mean_residuals
        ).clamp(0.0, self.config.dual.maximum)
        if not bool(torch.isfinite(after).all()):
            raise PPOValidationError("dual update produced non-finite values")
        self.dual_values.copy_(after)
        return DualUpdateMetrics(
            int(episode_constraint_totals.shape[0]),
            _cpu_float(mean_residuals), _cpu_float(before), _cpu_float(after),
        )