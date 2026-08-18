from dataclasses import replace
import inspect
from math import log
from pathlib import Path

import pytest
import torch

from isac_ssc.algorithms.buffers import (
    BufferValidationError, ConstraintLayout, EpisodeTotals, RolloutBuffer, RolloutTransition, StoredAction,
)
from isac_ssc.algorithms.constrained_ppo import ConstrainedPPO, PPOValidationError
from isac_ssc.algorithms.losses import (
    build_constrained_ppo_loss, clipped_reward_surrogate, clipped_value_loss, conservative_constraint_surrogates,
    generalized_advantage_estimate, normalize_advantages,
)
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import ISACSSCEnv
from isac_ssc.models.policy import EdgeFreeSetActorCritic, FactorizedActionIndices, build_policy_batch
from isac_ssc.models.value import ValueOutput
from isac_ssc.utils.config import load_algorithm_config, load_config

CONFIG = load_config()
ALGORITHM = load_algorithm_config()
TENANT_IDS = tuple(item.tenant_id for item in CONFIG.tenants)
USER_IDS = tuple(f"user_{index}" for index in range(1, CONFIG.population["communication_users"] + 1))
LAYOUT = ConstraintLayout(TENANT_IDS, USER_IDS)


def _observation(seed: int = 41001):
    env = ISACSSCEnv(CONFIG)
    observation = env.reset(generate_primitive_trace(CONFIG, seed, "independent"))
    assert observation is not None
    return observation


def _model():
    observation = _observation()
    batch = build_policy_batch(observation)
    return EdgeFreeSetActorCritic(batch.layout, ALGORITHM, CONFIG), observation


def _action_indices(action: StoredAction) -> FactorizedActionIndices:
    return action.to_indices()


def _old_policy_data(model, observation, action: StoredAction):
    batch = build_policy_batch(observation)
    _, logits, values = model(batch)
    log_probability, _ = model.policy.evaluate(logits, batch, _action_indices(action))
    return float(log_probability[0].detach()), values


def _transition(model, observation, action: StoredAction, reward: float, residual_scale: float = 1.0,
                *, terminated: bool = True, next_observation=None, span: int = 1) -> RolloutTransition:
    old_log_probability, values = _old_policy_data(model, observation, action)
    tenant = tuple(residual_scale * (index + 1) / 10.0 for index in range(LAYOUT.tenant_count))
    communication = tuple(-residual_scale * (index + 1) / 20.0 for index in range(LAYOUT.communication_count))
    return RolloutTransition(
        observation, action, reward, tenant, communication, next_observation, terminated, span, old_log_probability,
        float(values.reward_value[0].detach()), tuple(float(value) for value in values.sensing_sla_values[0].detach()),
        tuple(float(value) for value in values.communication_qos_values[0].detach()),
    )


def _prepared(model=None, count: int = 6, reward_scale: float = 1.0):
    model, observation = (model, _observation()) if model is not None else _model()
    actions = (StoredAction(1, -1, 2), StoredAction(2), StoredAction(3))
    buffer = RolloutBuffer(LAYOUT, ALGORITHM.ppo.discount, ALGORITHM.ppo.gae_lambda)
    for index in range(count):
        reward = reward_scale * (index + 1) * (1.0 if index % 2 == 0 else -0.4)
        transition = _transition(model, observation, actions[index % len(actions)], reward, index + 1)
        buffer.append(transition)
        buffer.record_episode_totals(EpisodeTotals(1, reward, transition.tenant_residuals, transition.communication_residuals))
    return model, buffer.finalize()


def test_constraint_layout_packs_canonical_residuals_without_merging_coordinates() -> None:
    tenant_pairs = tuple(reversed(tuple(zip(TENANT_IDS, range(1, LAYOUT.tenant_count + 1)))))
    communication_pairs = tuple(reversed(tuple(zip(USER_IDS, range(10, 10 + LAYOUT.communication_count)))))
    packed = LAYOUT.pack_residuals(tenant_pairs, communication_pairs)
    assert packed.dtype is torch.float32
    assert packed.tolist() == list(range(1, LAYOUT.tenant_count + 1)) + list(range(10, 10 + LAYOUT.communication_count))


def test_constraint_layout_rejects_duplicate_noncanonical_missing_and_extra_ids() -> None:
    with pytest.raises(BufferValidationError, match="unique"):
        ConstraintLayout(("tenant_1", "tenant_1"), USER_IDS)
    with pytest.raises(BufferValidationError, match="canonical"):
        ConstraintLayout(tuple(reversed(TENANT_IDS)), USER_IDS)
    with pytest.raises(BufferValidationError, match="do not match"):
        LAYOUT.pack_tenant_residuals(((TENANT_IDS[0], 0.0),))
    with pytest.raises(BufferValidationError, match="do not match"):
        LAYOUT.pack_communication_residuals(tuple(zip(USER_IDS, (0.0,) * len(USER_IDS))) + (("extra", 0.0),))


def test_typed_integer_and_string_identifiers_remain_distinct() -> None:
    layout = ConstraintLayout((1, "1"), ())
    assert layout.pack_tenant_residuals((("1", 2.0), (1, 1.0))).tolist() == [1.0, 2.0]


@pytest.mark.parametrize("values", ((0, 0, 0), (1, -1, 2), (2, -1, -1), (3, -1, -1)))
def test_stored_action_accepts_exact_factorized_conventions(values) -> None:
    assert StoredAction(*values).action_type_index == values[0]


@pytest.mark.parametrize("values", ((0, -1, 0), (1, 0, 0), (2, -1, 0), (3, 0, -1), (4, -1, -1)))
def test_stored_action_rejects_malformed_factorization(values) -> None:
    with pytest.raises(BufferValidationError):
        StoredAction(*values)


def test_stored_action_checks_public_masks_without_recomputing_feasibility() -> None:
    observation = _observation()
    StoredAction(1, -1, 2).validate_for_observation(observation)
    with pytest.raises(BufferValidationError, match="action type"):
        StoredAction(0, 0, 0).validate_for_observation(observation)
    create_profile_count = len(observation.set_view.action_masks.create_profile_mask)
    with pytest.raises(BufferValidationError, match="create-profile"):
        StoredAction(1, -1, create_profile_count).validate_for_observation(observation)


def test_transition_validates_finite_dimensions_termination_span_and_feasible_action() -> None:
    model, observation = _model()
    valid = _transition(model, observation, StoredAction(2), 1.0)
    valid.validate_layout(LAYOUT)
    with pytest.raises(BufferValidationError, match="terminated"):
        replace(valid, terminated=False)
    with pytest.raises(BufferValidationError, match="positive"):
        replace(valid, physical_slot_span=0)
    with pytest.raises(BufferValidationError, match="finite"):
        replace(valid, reward=float("nan"))
    with pytest.raises(BufferValidationError, match="tenant transition"):
        replace(valid, tenant_residuals=(0.0,)).validate_layout(LAYOUT)


def _different_layout_observation(observation):
    request_specs = list(observation.set_view.request_table.specs)
    request_specs[0] = replace(request_specs[0], name=f"{request_specs[0].name}_changed")
    request_table = replace(observation.set_view.request_table, specs=tuple(request_specs))
    return replace(observation, set_view=replace(observation.set_view, request_table=request_table))

def _different_state_observation(observation):
    def changed(view):
        values = view.global_features
        return replace(view, global_features=(values[0] + 1.0, *values[1:]))

    return replace(observation, set_view=changed(observation.set_view))

def test_prepared_rollout_rejects_invalid_physical_and_terminal_metadata() -> None:
    _, rollout = _prepared(count=2)
    for spans in (torch.zeros_like(rollout.physical_slot_spans), -torch.ones_like(rollout.physical_slot_spans)):
        with pytest.raises(BufferValidationError, match="physical_slot_spans"):
            replace(rollout, physical_slot_spans=spans)
    with pytest.raises(BufferValidationError, match="episode_physical_slots"):
        replace(rollout, episode_physical_slots=torch.zeros_like(rollout.episode_physical_slots))
    with pytest.raises(BufferValidationError, match="terminated"):
        replace(rollout, terminated=torch.zeros_like(rollout.terminated))


def test_prepared_rollout_rejects_next_observation_feature_layout_mismatch() -> None:
    model, observation = _model()
    mismatched = _different_layout_observation(observation)
    transition = _transition(model, observation, StoredAction(2), 1.0, terminated=False, next_observation=mismatched)
    buffer = RolloutBuffer(LAYOUT, 1.0, 0.95)
    buffer.append(transition)
    bootstrap = ValueOutput(torch.zeros(1), torch.zeros(1, LAYOUT.tenant_count), torch.zeros(1, LAYOUT.communication_count))
    with pytest.raises(BufferValidationError, match="FeatureLayout"):
        buffer.finalize(bootstrap)


def test_nonterminal_bootstrap_requires_exact_cpu_float32_tensors() -> None:
    model, observation = _model()
    buffer = RolloutBuffer(LAYOUT, 1.0, 0.95)
    buffer.append(_transition(model, observation, StoredAction(2), 1.0, terminated=False, next_observation=observation))
    wrong_dtype = ValueOutput(torch.zeros(1, dtype=torch.float64), torch.zeros(1, LAYOUT.tenant_count, dtype=torch.float64),
                              torch.zeros(1, LAYOUT.communication_count, dtype=torch.float64))
    with pytest.raises(BufferValidationError, match="CPU float32"):
        buffer.finalize(wrong_dtype)
    tracking = ValueOutput(torch.zeros(1, requires_grad=True), torch.zeros(1, LAYOUT.tenant_count),
                           torch.zeros(1, LAYOUT.communication_count))
    with pytest.raises(BufferValidationError, match="detached"):
        buffer.finalize(tracking)


def test_zero_decision_episode_preserves_raw_totals_for_dual_only_update() -> None:
    buffer = RolloutBuffer(LAYOUT, 1.0, 0.95)
    totals = EpisodeTotals(7, 3.0, (1.0,) * LAYOUT.tenant_count, (-0.5,) * LAYOUT.communication_count)
    buffer.record_episode_totals(totals)
    rollout = buffer.finalize()
    assert rollout.transition_count == 0
    assert rollout.episode_reward_totals.tolist() == [3.0]
    assert rollout.episode_physical_slots.tolist() == [7]
    assert rollout.episode_constraint_totals.shape == (1, LAYOUT.constraint_count)


def test_buffer_rejects_noncontiguous_sequence_and_mutation_after_finalize() -> None:
    model, observation = _model()
    transition = _transition(model, observation, StoredAction(2), 1.0, terminated=False, next_observation=observation)
    buffer = RolloutBuffer(LAYOUT, 1.0, 0.95)
    buffer.append(transition)
    different_observation = _different_state_observation(observation)
    with pytest.raises(BufferValidationError, match="contiguous"):
        buffer.append(_transition(model, different_observation, StoredAction(2), 1.0))
    bootstrap = ValueOutput(torch.zeros(1), torch.zeros(1, LAYOUT.tenant_count), torch.zeros(1, LAYOUT.communication_count))
    buffer.finalize(bootstrap)
    with pytest.raises(BufferValidationError, match="after finalization"):
        buffer.append(transition)


def test_buffer_finalization_keeps_raw_data_detached_and_computes_multi_span_gae() -> None:
    model, observation = _model()
    buffer = RolloutBuffer(LAYOUT, 1.0, 0.5)
    first = _transition(model, observation, StoredAction(2), 1.0, terminated=False, next_observation=observation, span=2)
    second = _transition(model, observation, StoredAction(3), 2.0, terminated=True, next_observation=None, span=1)
    first = replace(first, old_reward_value=0.0, old_tenant_values=(0.0,) * LAYOUT.tenant_count,
                    old_communication_values=(0.0,) * LAYOUT.communication_count)
    second = replace(second, old_reward_value=0.0, old_tenant_values=(0.0,) * LAYOUT.tenant_count,
                     old_communication_values=(0.0,) * LAYOUT.communication_count)
    buffer.append(first)
    buffer.append(second)
    rollout = buffer.finalize()
    assert rollout.rewards.tolist() == [1.0, 2.0]
    assert rollout.reward_advantages.tolist() == pytest.approx([1.5, 2.0])
    assert torch.equal(rollout.reward_returns, rollout.reward_advantages + rollout.old_reward_values)
    assert not any(tensor.requires_grad for tensor in (
        rollout.rewards, rollout.constraint_residuals, rollout.reward_advantages, rollout.constraint_advantages,
    ))


def test_nonterminal_final_transition_requires_exact_detached_bootstrap_values() -> None:
    model, observation = _model()
    buffer = RolloutBuffer(LAYOUT, 1.0, 0.95)
    buffer.append(_transition(model, observation, StoredAction(2), 1.0, terminated=False, next_observation=observation))
    with pytest.raises(BufferValidationError, match="requires"):
        buffer.finalize()
    values = ValueOutput(torch.tensor([2.0]), torch.ones(1, LAYOUT.tenant_count), torch.ones(1, LAYOUT.communication_count))
    rollout = buffer.finalize(values)
    assert rollout.reward_returns.item() == pytest.approx(3.0)


def test_gae_matches_terminal_bootstrap_multi_step_and_episode_reset_hand_calculations() -> None:
    signals = torch.tensor([1.0, 2.0, 4.0])
    values = torch.tensor([0.5, 0.7, 1.0])
    next_values = torch.tensor([0.7, 0.0, 0.0])
    terminated = torch.tensor([False, True, True])
    spans = torch.tensor([2, 1, 3])
    advantages, returns = generalized_advantage_estimate(signals, values, next_values, terminated, spans, 1.0, 0.5)
    second = 2.0 - 0.7
    first = 1.0 + 0.7 - 0.5 + (0.5 ** 2) * second
    assert advantages.tolist() == pytest.approx([first, second, 3.0])
    assert returns.tolist() == pytest.approx([first + 0.5, 2.0, 4.0])


def test_gae_calculates_each_constraint_coordinate_independently() -> None:
    signals = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    values = torch.zeros_like(signals)
    advantages, returns = generalized_advantage_estimate(
        signals, values, values, torch.tensor([False, True]), torch.tensor([1, 1]), 1.0, 0.5,
    )
    assert torch.allclose(advantages, torch.tensor([[2.0, -2.0], [2.0, -2.0]]))
    assert torch.equal(returns, advantages)


def test_advantage_normalization_is_global_per_signal_and_preserves_raw_inputs() -> None:
    reward = torch.tensor([1.0, 2.0, 3.0])
    constraints = torch.tensor([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    reward_before, constraints_before = reward.clone(), constraints.clone()
    normalized = normalize_advantages(reward, constraints, 1e-8)
    assert normalized.reward.mean().item() == pytest.approx(0.0, abs=1e-7)
    assert normalized.constraints[:, 0].mean().item() == pytest.approx(0.0, abs=1e-7)
    assert torch.equal(normalized.constraints[:, 1], torch.zeros(3))
    assert torch.equal(reward, reward_before) and torch.equal(constraints, constraints_before)


def test_reward_and_constraint_surrogates_use_min_and_conservative_max_clipping() -> None:
    ratio = torch.tensor([1.5, 0.5])
    reward = torch.tensor([1.0, -1.0])
    constraints = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    assert clipped_reward_surrogate(ratio, reward, 0.2).item() == pytest.approx((1.2 - 0.8) / 2)
    expected = torch.tensor([(1.5 - 0.5) / 2, (-1.2 + 0.8) / 2])
    assert torch.allclose(conservative_constraint_surrogates(ratio, constraints, 0.2), expected)


def test_clipped_value_loss_matches_manual_maximum_error() -> None:
    values = torch.tensor([2.0, -2.0])
    old = torch.zeros(2)
    returns = torch.tensor([1.0, -1.0])
    manual = 0.5 * torch.maximum((values - returns).square(), (old + values.clamp(-0.2, 0.2) - returns).square()).mean()
    assert clipped_value_loss(values, old, returns, 0.2).item() == pytest.approx(float(manual))


def test_complete_loss_matches_manual_actor_critic_kl_and_clip_diagnostics() -> None:
    new_logs = torch.log(torch.tensor([1.5, 0.5]))
    old_logs = torch.zeros(2)
    reward_advantages = torch.tensor([1.0, -1.0])
    constraint_advantages = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    output = build_constrained_ppo_loss(
        new_log_probabilities=new_logs, old_log_probabilities=old_logs, reward_advantages=reward_advantages,
        constraint_advantages=constraint_advantages, entropy=torch.tensor([2.0, 4.0]),
        new_reward_values=torch.tensor([2.0, -2.0]), old_reward_values=torch.zeros(2), reward_returns=torch.tensor([1.0, -1.0]),
        new_constraint_values=torch.tensor([[2.0, 0.0], [-2.0, 0.0]]), old_constraint_values=torch.zeros(2, 2),
        constraint_returns=torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), dual_values=torch.tensor([2.0, 0.0]), tenant_count=1,
        clip_ratio=0.2, value_clip_ratio=0.2, entropy_coefficient=0.1,
        reward_value_coefficient=0.5, constraint_value_coefficient=0.5,
    )
    reward_surrogate = (1.2 - 0.8) / 2
    constraint = torch.tensor([(1.5 - 0.5) / 2, (-1.2 + 0.8) / 2])
    assert output.reward_surrogate.item() == pytest.approx(reward_surrogate)
    assert torch.allclose(output.constraint_surrogates, constraint)
    assert output.actor_loss.item() == pytest.approx(-reward_surrogate + 2.0 * constraint[0] - 0.1 * 3.0)
    ratio = torch.tensor([1.5, 0.5])
    assert output.approximate_kl.item() == pytest.approx(float(((ratio - 1.0) - new_logs).mean()))
    assert output.clip_fraction.item() == 1.0
    assert output.constraint_value_loss.item() == pytest.approx((output.tenant_value_loss + output.communication_value_loss) / 2)


def test_zero_duals_recover_unconstrained_actor_and_positive_dual_adds_exact_pressure() -> None:
    common = dict(
        new_log_probabilities=torch.zeros(2), old_log_probabilities=torch.zeros(2), reward_advantages=torch.tensor([1.0, -1.0]),
        constraint_advantages=torch.tensor([[1.0], [1.0]]), entropy=torch.zeros(2), new_reward_values=torch.zeros(2),
        old_reward_values=torch.zeros(2), reward_returns=torch.zeros(2), new_constraint_values=torch.zeros(2, 1),
        old_constraint_values=torch.zeros(2, 1), constraint_returns=torch.zeros(2, 1), tenant_count=1,
        clip_ratio=0.2, value_clip_ratio=0.2, entropy_coefficient=0.0,
        reward_value_coefficient=0.0, constraint_value_coefficient=0.0,
    )
    zero = build_constrained_ppo_loss(**common, dual_values=torch.zeros(1))
    positive = build_constrained_ppo_loss(**common, dual_values=torch.tensor([3.0]))
    assert zero.actor_loss.item() == pytest.approx(-zero.reward_surrogate.item())
    assert positive.actor_loss.item() - zero.actor_loss.item() == pytest.approx(3.0)


def test_dual_update_uses_raw_episode_mean_once_and_projects_each_coordinate_independently() -> None:
    model, _ = _model()
    algorithm = ConstrainedPPO(model, ALGORITHM)
    algorithm.dual_values.copy_(torch.tensor([0.0, 99.999] + [1.0] * (LAYOUT.constraint_count - 2)))
    totals = torch.zeros(2, LAYOUT.constraint_count)
    totals[:, 0] = torch.tensor([1.0, 3.0])
    totals[:, 1] = 100.0
    totals[:, 2] = -200.0
    metrics = algorithm.update_duals(totals)
    assert metrics.mean_episode_residuals[:3].tolist() == [2.0, 100.0, -200.0]
    assert metrics.dual_values_after[0].item() == pytest.approx(0.02)
    assert metrics.dual_values_after[1].item() == pytest.approx(100.0)
    assert metrics.dual_values_after[2].item() == pytest.approx(0.0)
    assert torch.equal(metrics.dual_values_after[3:], metrics.dual_values_before[3:])


def test_dual_update_rejects_empty_wrong_or_grad_tracking_totals() -> None:
    algorithm = ConstrainedPPO(_model()[0], ALGORITHM)
    with pytest.raises(PPOValidationError, match="non-empty"):
        algorithm.update_duals(torch.empty(0, LAYOUT.constraint_count))
    with pytest.raises(PPOValidationError, match="float32"):
        algorithm.update_duals(torch.zeros(1, LAYOUT.constraint_count, dtype=torch.float64))
    with pytest.raises(PPOValidationError, match="detached"):
        algorithm.update_duals(torch.zeros(1, LAYOUT.constraint_count, requires_grad=True))


def test_real_actor_critic_optimizer_step_is_finite_clipped_and_does_not_mutate_rollout() -> None:
    torch.manual_seed(101)
    model, rollout = _prepared(reward_scale=100.0)
    algorithm = ConstrainedPPO(model, ALGORITHM)
    before_parameters = tuple(parameter.detach().clone() for parameter in model.parameters())
    before_rewards, before_residuals = rollout.rewards.clone(), rollout.constraint_residuals.clone()
    metrics = algorithm.optimize_rollout(rollout, generator=torch.Generator().manual_seed(5))
    assert metrics.optimizer_steps > 0 and metrics.transitions == rollout.transition_count
    assert all(isinstance(value, float) for value in (
        metrics.mean_total_loss, metrics.mean_actor_loss, metrics.mean_entropy, metrics.mean_approximate_kl,
    ))
    assert any(not torch.equal(before, after) for before, after in zip(before_parameters, model.parameters()))
    gradients = tuple(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
    norm = torch.linalg.vector_norm(torch.stack([gradient.detach().norm() for gradient in gradients]))
    assert float(norm) <= ALGORITHM.ppo.max_gradient_norm + 1e-5
    assert metrics.max_gradient_norm_before_clip >= float(norm)
    assert torch.equal(rollout.rewards, before_rewards) and torch.equal(rollout.constraint_residuals, before_residuals)
    assert not algorithm.dual_values.requires_grad and algorithm.dual_values.grad is None


def test_fixed_generator_reproduces_minibatch_order_metrics_and_parameter_update() -> None:
    torch.manual_seed(107)
    first_model, rollout = _prepared(count=7)
    second_model, _ = _model()
    second_model.load_state_dict(first_model.state_dict())
    first, second = ConstrainedPPO(first_model, ALGORITHM), ConstrainedPPO(second_model, ALGORITHM)
    first_metrics = first.optimize_rollout(rollout, generator=torch.Generator().manual_seed(9))
    second_metrics = second.optimize_rollout(rollout, generator=torch.Generator().manual_seed(9))
    for name in first_metrics.__dataclass_fields__:
        left, right = getattr(first_metrics, name), getattr(second_metrics, name)
        assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    for left, right in zip(first_model.parameters(), second_model.parameters()):
        assert torch.equal(left, right)


def test_tiny_target_kl_stops_before_all_later_epochs() -> None:
    torch.manual_seed(109)
    model, rollout = _prepared(count=8, reward_scale=20.0)
    ppo = replace(ALGORITHM.ppo, target_kl=1e-12, minibatch_decisions=2, epochs_per_rollout=6)
    config = replace(ALGORITHM, ppo=ppo)
    metrics = ConstrainedPPO(model, config).optimize_rollout(rollout, generator=torch.Generator().manual_seed(11))
    assert metrics.early_stopped_for_kl
    assert 1 <= metrics.epochs_completed < ppo.epochs_per_rollout
    assert metrics.minibatches_completed == metrics.epochs_completed * 4


def test_rollout_validation_stays_at_finalization_not_optimizer_hot_path() -> None:
    source = inspect.getsource(ConstrainedPPO._validate_rollout)
    assert "build_policy_batch" not in source
    assert "observations" not in source and "next_observations" not in source
    prepared = inspect.getsource(type(_prepared(count=2)[1]).__post_init__)
    assert "FeatureLayout.from_view" in prepared
    assert "validate_for_observation" in prepared


def test_zero_decision_rollout_is_rejected_by_primal_update_but_valid_for_duals() -> None:
    model, _ = _model()
    buffer = RolloutBuffer(LAYOUT, 1.0, 0.95)
    buffer.record_episode_totals(EpisodeTotals(3, 0.0, (1.0,) * LAYOUT.tenant_count, (0.0,) * LAYOUT.communication_count))
    rollout = buffer.finalize()
    algorithm = ConstrainedPPO(model, ALGORITHM)
    with pytest.raises(PPOValidationError, match="non-empty"):
        algorithm.optimize_rollout(rollout)
    assert algorithm.update_duals(rollout.episode_constraint_totals).completed_episodes == 1


def test_learning_sources_do_not_import_scientific_recomputation_or_future_modules() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "isac_ssc" / "algorithms"
    losses_source = (root / "losses.py").read_text(encoding="utf-8")
    ppo_source = (root / "constrained_ppo.py").read_text(encoding="utf-8")
    forbidden = ("core.sla", "core.utility", "envs.isac_ssc_env", "oracles", "baselines", "training", "checkpoint")
    assert not any(value in losses_source for value in forbidden)
    assert not any(value in ppo_source for value in forbidden)
    assert "state_snapshot(" not in ppo_source and "._current" not in ppo_source
    assert tuple(inspect.signature(ConstrainedPPO.optimize_rollout).parameters) == ("self", "rollout", "generator")