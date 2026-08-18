"""Pure constrained-PPO mathematics with validation at rollout boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


class LossValidationError(ValueError):
    """Raised when rollout-boundary tensors cannot define PPO targets."""


def _float_tensor(tensor: torch.Tensor, name: str, *, rank: int | None = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype is not torch.float32:
        raise LossValidationError(f"{name} must be a float32 tensor")
    if rank is not None and tensor.ndim != rank:
        raise LossValidationError(f"{name} must be rank-{rank}")
    if not bool(torch.isfinite(tensor).all()):
        raise LossValidationError(f"{name} must be finite")
    return tensor


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise LossValidationError(f"{name} must be finite")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise LossValidationError(f"{name} must lie in [0, 1]")
    return result


FACTOR_NAMES = ("action_type", "merge_session", "profile")


@dataclass(frozen=True, slots=True)
class NormalizedAdvantages:
    reward: torch.Tensor
    constraints: torch.Tensor


@dataclass(frozen=True, slots=True)
class SignalScales:
    reward: torch.Tensor
    constraints: torch.Tensor


@dataclass(frozen=True, slots=True)
class FrozenFactorAdvantages:
    reward: torch.Tensor
    constraints: torch.Tensor
    normalized_reward: torch.Tensor
    normalized_constraints: torch.Tensor


@dataclass(frozen=True, slots=True)
class FactorizedPolicySurrogates:
    ratios: torch.Tensor
    joint_log_ratio: torch.Tensor
    joint_ratio: torch.Tensor
    reward_by_factor: torch.Tensor
    constraints_by_factor: torch.Tensor
    approximate_joint_kl: torch.Tensor
    joint_clip_fraction: torch.Tensor
    factor_clip_fractions: torch.Tensor


@dataclass(frozen=True, slots=True)
class ConstrainedPPOLoss:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    reward_surrogate: torch.Tensor
    constraint_surrogates: torch.Tensor
    dual_penalty: torch.Tensor
    entropy: torch.Tensor
    reward_value_loss: torch.Tensor
    constraint_value_loss: torch.Tensor
    tenant_value_loss: torch.Tensor
    communication_value_loss: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    probability_ratio_mean: torch.Tensor


def generalized_advantage_estimate(
    signals: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, terminated: torch.Tensor,
    physical_slot_spans: torch.Tensor, discount: float, gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    signals = _float_tensor(signals, "signals")
    values = _float_tensor(values, "values")
    next_values = _float_tensor(next_values, "next_values")
    if signals.ndim not in (1, 2) or signals.shape[0] < 1 or values.shape != signals.shape or next_values.shape != signals.shape:
        raise LossValidationError("signals, values, and next_values must share non-empty shape [N] or [N, Q]")
    if not isinstance(terminated, torch.Tensor) or terminated.dtype is not torch.bool or terminated.shape != (signals.shape[0],):
        raise LossValidationError("terminated must be bool [N]")
    if not isinstance(physical_slot_spans, torch.Tensor) or physical_slot_spans.dtype is not torch.int64 or physical_slot_spans.shape != (signals.shape[0],):
        raise LossValidationError("physical_slot_spans must be int64 [N]")
    if bool((physical_slot_spans < 1).any()):
        raise LossValidationError("physical_slot_spans must be positive")
    gamma, trace = _probability(discount, "discount"), _probability(gae_lambda, "gae_lambda")
    advantages = torch.zeros_like(signals)
    carry = torch.zeros_like(signals[0])
    for index in range(signals.shape[0] - 1, -1, -1):
        span = int(physical_slot_spans[index])
        alive = 0.0 if bool(terminated[index]) else 1.0
        delta = signals[index] + (gamma ** span) * alive * next_values[index] - values[index]
        carry = delta + ((gamma * trace) ** span) * alive * carry
        advantages[index] = carry
    returns = advantages + values
    if not bool(torch.isfinite(advantages).all()) or not bool(torch.isfinite(returns).all()):
        raise LossValidationError("GAE produced non-finite tensors")
    return advantages, returns


def _normalize(values: torch.Tensor, epsilon: float, dimension: int) -> torch.Tensor:
    centered = values - values.mean(dim=dimension, keepdim=True)
    standard_deviation = values.var(dim=dimension, unbiased=False, keepdim=True).sqrt()
    return torch.where(standard_deviation <= epsilon, centered, centered / (standard_deviation + epsilon))


def signal_scales(
    reward_values: torch.Tensor, constraint_values: torch.Tensor, epsilon: float,
) -> SignalScales:
    reward = _float_tensor(reward_values, "reward_values", rank=1)
    constraints = _float_tensor(constraint_values, "constraint_values", rank=2)
    if constraints.shape[0] != reward.shape[0] or reward.numel() < 1:
        raise LossValidationError("reward and constraint signals must share non-empty transition dimension")
    epsilon = _positive_finite(epsilon, "normalization epsilon")
    reward_standard_deviation = reward.var(unbiased=False).sqrt()
    constraint_standard_deviations = constraints.var(dim=0, unbiased=False).sqrt()
    reward_scale = torch.where(
        reward_standard_deviation <= epsilon,
        torch.ones_like(reward_standard_deviation),
        reward_standard_deviation + epsilon,
    )
    constraint_scales = torch.where(
        constraint_standard_deviations <= epsilon,
        torch.ones_like(constraint_standard_deviations),
        constraint_standard_deviations + epsilon,
    )
    return SignalScales(reward_scale, constraint_scales)


def normalize_advantages(
    reward_advantages: torch.Tensor, constraint_advantages: torch.Tensor, epsilon: float,
) -> NormalizedAdvantages:
    reward = _float_tensor(reward_advantages, "reward_advantages", rank=1)
    constraints = _float_tensor(constraint_advantages, "constraint_advantages", rank=2)
    scales = signal_scales(reward, constraints, epsilon)
    return NormalizedAdvantages(
        (reward - reward.mean()) / scales.reward,
        (constraints - constraints.mean(dim=0, keepdim=True)) / scales.constraints,
    )


def _bool_tensor(tensor: torch.Tensor, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype is not torch.bool or tuple(tensor.shape) != shape:
        raise LossValidationError(f"{name} must be bool with shape {shape}")
    return tensor


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or float(value) <= 0.0:
        raise LossValidationError(f"{name} must be positive and finite")
    return float(value)


def _normalize_return_anchored_factors(
    parts: torch.Tensor, global_values: torch.Tensor,
    applicability: torch.Tensor, epsilon: float,
) -> torch.Tensor:
    mean = global_values.mean(dim=0, keepdim=True)
    standard_deviation = global_values.var(
        dim=0, unbiased=False, keepdim=True,
    ).sqrt()
    denominator = standard_deviation + epsilon
    type_values = _normalize(global_values, epsilon, 0)
    centered_lower = parts[:, 1:] - mean.unsqueeze(1)
    lower_values = torch.where(
        standard_deviation.unsqueeze(1) <= epsilon, centered_lower,
        centered_lower / denominator.unsqueeze(1),
    )
    lower_mask = applicability[:, 1:]
    if parts.ndim == 3:
        lower_mask = lower_mask.unsqueeze(2)
    lower_values = torch.where(
        lower_mask, lower_values, torch.zeros_like(lower_values),
    )
    return torch.cat((type_values.unsqueeze(1), lower_values), dim=1)


def build_frozen_factor_advantages(
    *, reward_advantages: torch.Tensor, constraint_advantages: torch.Tensor,
    reward_returns: torch.Tensor, constraint_returns: torch.Tensor,
    old_reward_values: torch.Tensor, old_constraint_values: torch.Tensor,
    old_action_type_reward_values: torch.Tensor, old_action_type_constraint_values: torch.Tensor,
    old_merge_session_reward_values: torch.Tensor, old_merge_session_constraint_values: torch.Tensor,
    merge_selected: torch.Tensor, merge_session_applicable: torch.Tensor,
    profile_applicable: torch.Tensor, epsilon: float,
) -> FrozenFactorAdvantages:
    """Build rollout-frozen return-anchored factor advantages from old prefix values."""
    global_reward = _float_tensor(reward_advantages, "reward_advantages", rank=1)
    global_constraints = _float_tensor(
        constraint_advantages, "constraint_advantages", rank=2,
    )
    reward_returns = _float_tensor(reward_returns, "reward_returns", rank=1)
    constraints = _float_tensor(constraint_returns, "constraint_returns", rank=2)
    n, q = constraints.shape
    vectors = (
        (old_reward_values, "old_reward_values", (n,)),
        (old_action_type_reward_values, "old_action_type_reward_values", (n,)),
        (old_merge_session_reward_values, "old_merge_session_reward_values", (n,)),
        (old_constraint_values, "old_constraint_values", (n, q)),
        (old_action_type_constraint_values, "old_action_type_constraint_values", (n, q)),
        (old_merge_session_constraint_values, "old_merge_session_constraint_values", (n, q)),
    )
    if (
        reward_returns.shape != (n,) or global_reward.shape != (n,)
        or global_constraints.shape != (n, q) or n < 1
    ):
        raise LossValidationError(
            "factor advantages require non-empty aligned advantages and returns"
        )
    for tensor, name, shape in vectors:
        value = _float_tensor(tensor, name)
        if tuple(value.shape) != shape or value.requires_grad:
            raise LossValidationError(f"{name} must be detached with shape {shape}")
    if (
        global_reward.requires_grad or global_constraints.requires_grad
        or reward_returns.requires_grad or constraints.requires_grad
    ):
        raise LossValidationError("factor advantages and returns must be detached")
    merge_mask = _bool_tensor(merge_selected, "merge_selected", (n,))
    session_mask = _bool_tensor(merge_session_applicable, "merge_session_applicable", (n,))
    profile_mask = _bool_tensor(profile_applicable, "profile_applicable", (n,))
    if bool((session_mask & ~merge_mask).any()):
        raise LossValidationError("MERGE-session actor applicability requires selected MERGE")
    epsilon = _positive_finite(epsilon, "normalization epsilon")
    zero_reward = torch.zeros_like(global_reward)
    zero_constraints = torch.zeros_like(global_constraints)

    type_reward = global_reward
    session_reward = torch.where(
        session_mask, reward_returns - old_action_type_reward_values, zero_reward,
    )
    profile_reward_baseline = torch.where(
        merge_mask, old_merge_session_reward_values, old_action_type_reward_values,
    )
    profile_reward = torch.where(
        profile_mask, reward_returns - profile_reward_baseline, zero_reward,
    )

    session_expanded = session_mask.unsqueeze(1)
    profile_expanded = profile_mask.unsqueeze(1)
    profile_constraint_baseline = torch.where(
        merge_mask.unsqueeze(1), old_merge_session_constraint_values,
        old_action_type_constraint_values,
    )
    type_constraints = global_constraints
    session_constraints = torch.where(
        session_expanded, constraints - old_action_type_constraint_values,
        zero_constraints,
    )
    profile_constraints = torch.where(
        profile_expanded, constraints - profile_constraint_baseline,
        zero_constraints,
    )

    reward = torch.stack((type_reward, session_reward, profile_reward), dim=1)
    constraint_values = torch.stack(
        (type_constraints, session_constraints, profile_constraints), dim=1,
    )
    applicability = torch.stack(
        (torch.ones_like(session_mask), session_mask, profile_mask), dim=1,
    )
    normalized_reward = _normalize_return_anchored_factors(
        reward, global_reward, applicability, epsilon,
    )
    normalized_constraints = _normalize_return_anchored_factors(
        constraint_values, global_constraints, applicability, epsilon,
    )
    return FrozenFactorAdvantages(
        reward, constraint_values, normalized_reward, normalized_constraints,
    )


def build_factorized_policy_surrogates(
    *, new_log_probabilities: torch.Tensor, old_log_probabilities: torch.Tensor,
    applicability: torch.Tensor, reward_advantages: torch.Tensor,
    constraint_advantages: torch.Tensor, clip_ratio: float,
) -> FactorizedPolicySurrogates:
    new_logs = _float_tensor(new_log_probabilities, "new_log_probabilities", rank=2)
    old_logs = _float_tensor(old_log_probabilities, "old_log_probabilities", rank=2)
    reward = _float_tensor(reward_advantages, "reward_advantages", rank=2)
    constraints = _float_tensor(constraint_advantages, "constraint_advantages", rank=3)
    n = new_logs.shape[0]
    if (
        n < 1 or new_logs.shape != (n, len(FACTOR_NAMES))
        or old_logs.shape != new_logs.shape or reward.shape != new_logs.shape
    ):
        raise LossValidationError("factor policy tensors must share non-empty shape [N, 3]")
    if constraints.shape[:2] != new_logs.shape:
        raise LossValidationError("factor constraint advantages must have shape [N, 3, C]")
    mask = _bool_tensor(applicability, "applicability", tuple(new_logs.shape))
    if not bool(mask[:, 0].all()):
        raise LossValidationError("action-type factor must apply to every transition")
    if (
        bool(reward.masked_select(~mask).ne(0.0).any())
        or bool(constraints.masked_select(~mask.unsqueeze(2)).ne(0.0).any())
    ):
        raise LossValidationError("inapplicable factor advantages must be zero")

    clip_ratio = _positive_finite(clip_ratio, "clip ratio")
    if clip_ratio >= 1.0:
        raise LossValidationError("clip ratio must be smaller than one")

    log_ratios = new_logs - old_logs
    ratios = torch.exp(log_ratios)
    if not bool(torch.isfinite(ratios).all()):
        raise LossValidationError("factor probability ratios must be finite")

    clipped = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    mask_float = mask.to(torch.float32)
    reward_terms = torch.minimum(
        ratios * reward, clipped * reward,
    ) * mask_float
    constraint_terms = torch.maximum(
        ratios.unsqueeze(2) * constraints,
        clipped.unsqueeze(2) * constraints,
    ) * mask_float.unsqueeze(2)

    reward_by_factor = reward_terms.sum(dim=0) / n
    constraints_by_factor = constraint_terms.sum(dim=0) / n
    joint_log_ratio = (log_ratios * mask_float).sum(dim=1)
    joint_ratio = torch.exp(joint_log_ratio)
    if not bool(torch.isfinite(joint_ratio).all()):
        raise LossValidationError("joint probability ratios must be finite")

    factor_clipped = ((ratios - 1.0).abs() > clip_ratio) & mask
    applicable_counts = mask.sum(dim=0)
    factor_clip_fractions = torch.where(
        applicable_counts > 0,
        factor_clipped.sum(dim=0).to(torch.float32) / applicable_counts.clamp_min(1),
        torch.zeros(len(FACTOR_NAMES), dtype=torch.float32, device=ratios.device),
    )
    return FactorizedPolicySurrogates(
        ratios, joint_log_ratio, joint_ratio, reward_by_factor, constraints_by_factor,
        approximate_kl(joint_ratio, joint_log_ratio),
        clip_fraction(joint_ratio, clip_ratio), factor_clip_fractions,
    )


def masked_clipped_value_loss(
    values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
    applicable: torch.Tensor, clip_ratio: float, scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    current = _float_tensor(values, "values")
    old = _float_tensor(old_values, "old_values")
    targets = _float_tensor(returns, "returns")
    if (
        current.ndim not in (1, 2)
        or old.shape != current.shape
        or targets.shape != current.shape
    ):
        raise LossValidationError("masked value tensors must share shape [N] or [N, C]")
    mask = _bool_tensor(applicable, "applicable", (current.shape[0],))
    _positive_finite(clip_ratio, "value clip ratio")
    if not bool(mask.any()):
        return torch.zeros((), dtype=current.dtype, device=current.device)
    return clipped_value_loss(current[mask], old[mask], targets[mask], clip_ratio, scale)


def clipped_reward_surrogate(ratio: torch.Tensor, advantages: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    return torch.minimum(ratio * advantages, clipped * advantages).mean()


def conservative_constraint_surrogates(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_ratio: float,
) -> torch.Tensor:
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio).unsqueeze(1)
    return torch.maximum(ratio.unsqueeze(1) * advantages, clipped * advantages).mean(dim=0)


def clipped_value_loss(
    values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
    clip_ratio: float, scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    current = _float_tensor(values, "values")
    old = _float_tensor(old_values, "old_values")
    targets = _float_tensor(returns, "returns")
    if current.shape != old.shape or current.shape != targets.shape or current.ndim not in (1, 2):
        raise LossValidationError("value tensors must share shape [N] or [N, C]")
    _positive_finite(clip_ratio, "value clip ratio")
    scale_tensor = torch.as_tensor(scale, dtype=current.dtype, device=current.device)
    expected = () if current.ndim == 1 else (current.shape[1],)
    if tuple(scale_tensor.shape) != expected or not bool(torch.isfinite(scale_tensor).all()) or bool((scale_tensor <= 0.0).any()):
        raise LossValidationError(f"value scale must have shape {expected} with positive finite entries")
    clip_delta = clip_ratio * scale_tensor
    clipped = old + (current - old).clamp(-clip_delta, clip_delta)
    raw_error = (current - targets) / scale_tensor
    clipped_error = (clipped - targets) / scale_tensor
    return 0.5 * torch.maximum(raw_error.square(), clipped_error.square()).mean()


def approximate_kl(ratio: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
    return ((ratio - 1.0) - log_ratio).mean()


def clip_fraction(ratio: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    return ((ratio - 1.0).abs() > clip_ratio).to(torch.float32).mean()


def build_constrained_ppo_loss(
    *, new_log_probabilities: torch.Tensor, old_log_probabilities: torch.Tensor,
    reward_advantages: torch.Tensor, constraint_advantages: torch.Tensor, entropy: torch.Tensor,
    new_reward_values: torch.Tensor, old_reward_values: torch.Tensor, reward_returns: torch.Tensor,
    new_constraint_values: torch.Tensor, old_constraint_values: torch.Tensor,
    constraint_returns: torch.Tensor, dual_values: torch.Tensor, tenant_count: int,
    reward_return_scale: torch.Tensor | float = 1.0,
    constraint_return_scales: torch.Tensor | None = None,
    clip_ratio: float, value_clip_ratio: float, entropy_coefficient: float,
    reward_value_coefficient: float, constraint_value_coefficient: float,
) -> ConstrainedPPOLoss:
    if constraint_return_scales is None:
        constraint_return_scales = torch.ones(
            constraint_returns.shape[1], dtype=constraint_returns.dtype,
            device=constraint_returns.device,
        )
    log_ratio = new_log_probabilities - old_log_probabilities
    ratio = torch.exp(log_ratio)
    reward_surrogate = clipped_reward_surrogate(ratio, reward_advantages, clip_ratio)
    constraint_surrogates = conservative_constraint_surrogates(ratio, constraint_advantages, clip_ratio)
    dual_penalty = torch.dot(dual_values.detach(), constraint_surrogates)
    entropy_mean = entropy.mean()
    actor_loss = -reward_surrogate + dual_penalty - entropy_coefficient * entropy_mean
    reward_value_loss = clipped_value_loss(
        new_reward_values, old_reward_values, reward_returns,
        value_clip_ratio, reward_return_scale,
    )
    constraint_value_loss = clipped_value_loss(
        new_constraint_values, old_constraint_values, constraint_returns,
        value_clip_ratio, constraint_return_scales,
    )
    constraint_count = constraint_advantages.shape[1]
    zero = torch.zeros((), dtype=torch.float32, device=new_constraint_values.device)
    tenant_value_loss = (
        clipped_value_loss(
            new_constraint_values[:, :tenant_count], old_constraint_values[:, :tenant_count],
            constraint_returns[:, :tenant_count], value_clip_ratio,
            constraint_return_scales[:tenant_count],
        ) if tenant_count else zero
    )
    communication_value_loss = (
        clipped_value_loss(
            new_constraint_values[:, tenant_count:], old_constraint_values[:, tenant_count:],
            constraint_returns[:, tenant_count:], value_clip_ratio,
            constraint_return_scales[tenant_count:],
        ) if tenant_count < constraint_count else zero
    )
    total_loss = actor_loss + reward_value_coefficient * reward_value_loss + constraint_value_coefficient * constraint_value_loss
    return ConstrainedPPOLoss(
        total_loss, actor_loss, reward_surrogate, constraint_surrogates, dual_penalty, entropy_mean,
        reward_value_loss, constraint_value_loss, tenant_value_loss, communication_value_loss,
        approximate_kl(ratio, log_ratio), clip_fraction(ratio, clip_ratio), ratio.mean(),
    )