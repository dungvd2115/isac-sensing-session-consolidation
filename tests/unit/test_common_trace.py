from dataclasses import fields, replace

import pytest
import torch

from isac_ssc.algorithms.buffers import (
    BufferValidationError, ConstraintLayout, EpisodeTotals,
    FactorCreditTransition, RolloutBuffer, RolloutTransition, StoredAction,
)
from isac_ssc.algorithms.common_trace_ppo import CommonTracePPO
from isac_ssc.algorithms.losses import (
    LossValidationError, build_factorized_policy_surrogates,
    build_frozen_factor_advantages, clipped_value_loss,
    masked_clipped_value_loss, normalize_advantages,
)
from isac_ssc.baselines.ppo_common_trace import build_common_trace_agent
from isac_ssc.baselines.ppo_joint_credit import build_joint_credit_agent
from isac_ssc.envs.action_space import identifier_key
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import ISACSSCEnv
from isac_ssc.models.policy import CommonTraceActorCritic, build_policy_batch
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.training.rollout import collect_common_trace_training_rollout
from isac_ssc.utils.config import load_algorithm_config, load_config

CONFIG = load_config()
ALGORITHM = load_algorithm_config()


def _observation_and_layout(seed: int = 41001):
    trace = generate_primitive_trace(CONFIG, seed, "independent")
    env = ISACSSCEnv(CONFIG)
    observation = env.reset(trace)
    while observation is None and not env.terminated:
        observation = env.step(None).next_observation
    assert observation is not None
    tenants = tuple(sorted((item.tenant_id for item in CONFIG.tenants), key=identifier_key))
    users = tuple(sorted({item.user_id for item in trace.communication_states}, key=identifier_key))
    return observation, ConstraintLayout(tenants, users)


def _factor_transition():
    observation, layout = _observation_and_layout()
    batch = build_policy_batch(observation)
    torch.manual_seed(17)
    model = CommonTraceActorCritic(FeatureLayout.from_view(observation.set_view), ALGORITHM, CONFIG)
    with torch.no_grad():
        selection, values, prefixes = model.select(batch, deterministic=True)
    action = StoredAction.from_indices(selection.indices)
    type_index = action.action_type_index
    type_constraints = prefixes.type_constraint_values[0, type_index]
    if action.action_type_index == 0:
        session_index = action.merge_session_index
        session_reward = float(prefixes.merge_session_reward_values[0, session_index])
        session_constraints = tuple(map(float, prefixes.merge_session_constraint_values[0, session_index]))
    else:
        session_reward = 0.0
        session_constraints = (0.0,) * layout.constraint_count
    components = selection.factor_log_probabilities
    factor = FactorCreditTransition(
        float(components.action_type[0]), float(components.merge_session[0]),
        float(components.profile[0]), float(prefixes.type_reward_values[0, type_index]),
        tuple(map(float, type_constraints)), session_reward, session_constraints,
        bool(components.merge_session_applicable[0]), bool(components.profile_applicable[0]),
    )
    transition = RolloutTransition(
        observation, action, 1.25, (0.1,) * layout.tenant_count,
        (-0.05,) * layout.communication_count, None, True, 1,
        float(selection.log_probability[0]), float(values.reward_value[0]),
        tuple(map(float, values.sensing_sla_values[0])),
        tuple(map(float, values.communication_qos_values[0])), factor,
    )
    return transition, layout


def _agents(seed: int = 20, algorithm=ALGORITHM):
    observation, layout = _observation_and_layout(50001)
    feature_layout = FeatureLayout.from_view(observation.set_view)
    joint = build_joint_credit_agent(
        feature_layout, algorithm, CONFIG, model_seed=seed,
        action_seed=seed + 1, minibatch_seed=seed + 2,
    )
    common_trace = build_common_trace_agent(
        feature_layout, algorithm, CONFIG, model_seed=seed,
        action_seed=seed + 1, minibatch_seed=seed + 2,
    )
    return joint, common_trace, layout


def test_frozen_factor_advantages_keep_global_type_credit_and_return_anchored_lower_credit() -> None:
    reward_returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
    old_reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
    global_reward = reward_returns - old_reward
    old_type_reward = torch.tensor([4.0, 8.0, 0.0, 0.0])
    old_session_reward = torch.tensor([6.0, 0.0, 0.0, 0.0])
    old_constraints = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    constraint_returns = torch.tensor([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]])
    global_constraints = constraint_returns - old_constraints
    old_type_constraints = torch.tensor([[4.0, 40.0], [8.0, 80.0], [0.0, 0.0], [0.0, 0.0]])
    old_session_constraints = torch.tensor([[6.0, 60.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    merge_selected = torch.tensor([True, False, False, False])
    session = torch.tensor([True, False, False, False])
    profile = torch.tensor([True, True, False, False])
    frozen = build_frozen_factor_advantages(
        reward_advantages=global_reward, constraint_advantages=global_constraints,
        reward_returns=reward_returns, constraint_returns=constraint_returns,
        old_reward_values=old_reward, old_constraint_values=old_constraints,
        old_action_type_reward_values=old_type_reward,
        old_action_type_constraint_values=old_type_constraints,
        old_merge_session_reward_values=old_session_reward,
        old_merge_session_constraint_values=old_session_constraints,
        merge_selected=merge_selected, merge_session_applicable=session,
        profile_applicable=profile, epsilon=1.0e-8,
    )
    assert torch.equal(frozen.reward, torch.tensor([
        [9.0, 6.0, 4.0], [18.0, 0.0, 12.0],
        [27.0, 0.0, 0.0], [36.0, 0.0, 0.0],
    ]))
    assert torch.equal(frozen.constraints[:, 0], global_constraints)
    assert torch.equal(frozen.constraints[0, 1], torch.tensor([6.0, 60.0]))
    assert torch.equal(frozen.constraints[0, 2], torch.tensor([4.0, 40.0]))
    assert torch.equal(frozen.constraints[1, 2], torch.tensor([12.0, 120.0]))
    assert not bool(frozen.reward[~torch.stack((torch.ones_like(session), session, profile), dim=1)].ne(0.0).any())
    normalized = normalize_advantages(global_reward, global_constraints, 1.0e-8)
    assert torch.allclose(frozen.normalized_reward[:, 0], normalized.reward)
    assert torch.allclose(frozen.normalized_constraints[:, 0], normalized.constraints)
    assert not any(item.requires_grad for item in (
        frozen.reward, frozen.constraints, frozen.normalized_reward, frozen.normalized_constraints,
    ))


def test_factor_surrogates_use_total_transition_denominator_and_report_joint_drift() -> None:
    old_logs = torch.zeros(4, 3)
    ratios = torch.tensor([[1.2, 1.3, 0.7], [0.8, 1.0, 1.2], [1.0, 1.0, 1.0], [1.1, 1.0, 1.0]])
    new_logs = ratios.log()
    applicability = torch.tensor([
        [True, True, True], [True, False, True],
        [True, False, False], [True, False, False],
    ])
    reward = torch.tensor([
        [1.0, 2.0, -1.0], [-1.0, 0.0, 2.0],
        [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
    ])
    result = build_factorized_policy_surrogates(
        new_log_probabilities=new_logs, old_log_probabilities=old_logs,
        applicability=applicability, reward_advantages=reward,
        constraint_advantages=reward.unsqueeze(2), clip_ratio=0.2,
    )
    assert result.reward_by_factor.tolist() == pytest.approx([0.05, 0.6, 0.4])
    assert torch.allclose(result.constraints_by_factor, torch.tensor([[0.05], [0.65], [0.425]]))
    assert torch.allclose(result.joint_ratio, torch.tensor([1.2 * 1.3 * 0.7, 0.8 * 1.2, 1.0, 1.1]))
    assert result.factor_clip_fractions.tolist() == pytest.approx([0.25, 1.0, 1.0])
    assert torch.isfinite(result.approximate_joint_kl)
    assert torch.isfinite(result.joint_clip_fraction)


def test_factor_surrogates_reject_inapplicable_credit_and_missing_type_applicability() -> None:
    logs = torch.zeros(2, 3)
    applicability = torch.tensor([[True, False, False], [True, False, True]])
    reward = torch.zeros(2, 3)
    constraints = torch.zeros(2, 3, 2)
    malformed_reward = reward.clone()
    malformed_reward[0, 1] = 1.0
    with pytest.raises(LossValidationError, match="inapplicable"):
        build_factorized_policy_surrogates(
            new_log_probabilities=logs, old_log_probabilities=logs,
            applicability=applicability, reward_advantages=malformed_reward,
            constraint_advantages=constraints, clip_ratio=0.2,
        )
    with pytest.raises(LossValidationError, match="action-type"):
        build_factorized_policy_surrogates(
            new_log_probabilities=logs, old_log_probabilities=logs,
            applicability=torch.tensor([[True, False, False], [False, False, False]]),
            reward_advantages=reward, constraint_advantages=constraints, clip_ratio=0.2,
        )


def test_masked_prefix_value_loss_uses_only_applicable_rows_and_handles_no_rows() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])
    old = torch.tensor([0.5, 2.5, 2.0])
    returns = torch.tensor([1.5, 1.5, 4.0])
    mask = torch.tensor([True, False, True])
    assert masked_clipped_value_loss(values, old, returns, mask, 0.2) == clipped_value_loss(
        values[mask], old[mask], returns[mask], 0.2,
    )
    assert masked_clipped_value_loss(
        values, old, returns, torch.zeros(3, dtype=torch.bool), 0.2,
    ).item() == 0.0


def test_factor_buffer_freezes_current_factor_state_and_preserves_joint_mode() -> None:
    transition, layout = _factor_transition()
    buffer = RolloutBuffer(
        layout, ALGORITHM.ppo.discount, ALGORITHM.ppo.gae_lambda,
        factor_normalization_epsilon=ALGORITHM.normalization.epsilon,
    )
    buffer.append(transition)
    buffer.record_episode_totals(EpisodeTotals(
        1, transition.reward, transition.tenant_residuals, transition.communication_residuals,
    ))
    rollout = buffer.finalize()
    factor = rollout.factor_credit
    assert factor is not None
    assert factor.old_log_probabilities.shape == (1, 3)
    assert factor.normalization_epsilon == ALGORITHM.normalization.epsilon
    assert torch.allclose(
        (factor.old_log_probabilities * factor.applicability).sum(dim=1), rollout.old_log_probabilities,
    )
    assert torch.equal(factor.reward_advantages[:, 0], rollout.reward_advantages)
    assert torch.equal(factor.constraint_advantages[:, 0], rollout.constraint_advantages)
    assert not bool(factor.reward_advantages.masked_select(~factor.applicability).ne(0.0).any())
    assert not bool(factor.constraint_advantages.masked_select(~factor.applicability.unsqueeze(2)).ne(0.0).any())
    assert not any(tensor.requires_grad for tensor in (
        factor.reward_advantages, factor.constraint_advantages,
        factor.normalized_reward_advantages, factor.normalized_constraint_advantages,
    ))
    joint = RolloutBuffer(layout, ALGORITHM.ppo.discount, ALGORITHM.ppo.gae_lambda)
    joint.append(replace(transition, factor_credit=None))
    joint.record_episode_totals(EpisodeTotals(
        1, transition.reward, transition.tenant_residuals, transition.communication_residuals,
    ))
    assert joint.finalize().factor_credit is None


def test_factor_buffer_rejects_mode_mismatch_and_inapplicable_session_payload() -> None:
    transition, layout = _factor_transition()
    joint = RolloutBuffer(layout, 1.0, 0.95)
    with pytest.raises(BufferValidationError, match="mode"):
        joint.append(transition)
    factor = RolloutBuffer(layout, 1.0, 0.95, factor_normalization_epsilon=1.0e-8)
    with pytest.raises(BufferValidationError, match="mode"):
        factor.append(replace(transition, factor_credit=None))
    malformed = FactorCreditTransition(
        0.0, 0.1, 0.0, 0.0, (0.0,) * layout.constraint_count,
        0.0, (0.0,) * layout.constraint_count, True, False,
    )
    with pytest.raises(BufferValidationError, match="session actor factor"):
        malformed.validate(StoredAction(2), layout)


def test_common_trace_agent_preserves_common_initialization_and_uses_two_optimizer_groups() -> None:
    global_state = torch.random.get_rng_state().clone()
    joint, common_trace, _ = _agents(31)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    common_trace_state = common_trace.model.state_dict()
    for name, value in joint.model.state_dict().items():
        assert torch.equal(value, common_trace_state[name])
    assert isinstance(common_trace.algorithm, CommonTracePPO)
    assert len(common_trace.algorithm.optimizer.param_groups) == 2
    common_ids = {id(value) for value in common_trace.algorithm.common_parameters}
    prefix_ids = {id(value) for value in common_trace.algorithm.prefix_parameters}
    assert common_ids.isdisjoint(prefix_ids)
    assert common_ids | prefix_ids == {id(value) for value in common_trace.model.parameters()}


def test_common_trace_collection_reuses_primitive_trace_and_replaces_global_reward_credit() -> None:
    algorithm = replace(ALGORITHM, ppo=replace(ALGORITHM.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    _, common_trace, layout = _agents(41, algorithm)
    calls = []

    def factory(index, regime):
        calls.append((index, regime))
        return generate_primitive_trace(CONFIG, 56001 + index, regime)

    collected = collect_common_trace_training_rollout(
        ISACSSCEnv(CONFIG), common_trace, layout, factory,
        0, 201, algorithm, ("independent", "clustered"),
    )
    assert len(collected.episodes) == 2
    assert collected.episodes[0].trace_id == collected.episodes[1].trace_id
    assert collected.episodes[0].root_seed == collected.episodes[1].root_seed
    assert collected.episodes[0].arrival_regime == collected.episodes[1].arrival_regime == "independent"
    assert calls == [(0, "independent"), (0, "independent")]
    rollout = collected.rollout
    factor = rollout.factor_credit
    assert factor is not None
    assert torch.equal(factor.reward_advantages[:, 0], rollout.reward_advantages)
    assert torch.equal(factor.constraint_advantages[:, 0], rollout.constraint_advantages)
    assert bool(torch.isfinite(rollout.reward_advantages).all())
    assert not bool(factor.reward_advantages.masked_select(~factor.applicability).ne(0.0).any())


def test_common_trace_actor_diagnostics_broadcast_common_reward_to_applicable_factors() -> None:
    algorithm = replace(ALGORITHM, ppo=replace(ALGORITHM.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    _, agent, layout = _agents(45, algorithm)
    collected = collect_common_trace_training_rollout(
        ISACSSCEnv(CONFIG), agent, layout,
        lambda index, regime: generate_primitive_trace(CONFIG, 56501 + index, regime),
        0, 201, algorithm, ("independent", "clustered"),
    )
    factor = collected.rollout.factor_credit
    assert factor is not None
    means, positive = CommonTracePPO._factor_advantage_diagnostics(factor)
    common = factor.normalized_reward_advantages[:, 0]
    for index in range(3):
        mask = factor.applicability[:, index]
        if bool(mask.any()):
            assert means[index] == pytest.approx(float(common[mask].mean()))
            assert positive[index] == pytest.approx(float(common[mask].gt(0).float().mean()))
        else:
            assert means[index] == 0.0
            assert positive[index] == 0.0


def test_common_trace_update_is_finite_and_keeps_rollout_frozen() -> None:
    algorithm = replace(ALGORITHM, ppo=replace(ALGORITHM.ppo, epochs_per_rollout=2, minibatch_decisions=64))
    _, agent, layout = _agents(51, algorithm)
    collected = collect_common_trace_training_rollout(
        ISACSSCEnv(CONFIG), agent, layout,
        lambda index, regime: generate_primitive_trace(CONFIG, 57001 + index, regime),
        0, 201, algorithm, ("independent", "clustered"),
    )
    factor = collected.rollout.factor_credit
    assert factor is not None
    frozen = tuple(value.clone() for value in (
        factor.reward_advantages, factor.constraint_advantages,
        factor.normalized_reward_advantages, factor.normalized_constraint_advantages,
    ))
    metrics = agent.algorithm.optimize_rollout(collected.rollout, generator=agent.minibatch_generator)
    for before, after in zip(frozen, (
        factor.reward_advantages, factor.constraint_advantages,
        factor.normalized_reward_advantages, factor.normalized_constraint_advantages,
    ), strict=True):
        assert torch.equal(before, after)
    assert metrics.optimizer_steps > 0
    assert metrics.transitions == collected.rollout.transition_count
    assert metrics.mean_reward_surrogates_by_factor.shape == (3,)
    assert metrics.mean_constraint_surrogates_by_factor.shape == (3, layout.constraint_count)
    assert metrics.mean_factor_clip_fractions.shape == (3,)
    assert metrics.joint_ratio_quantiles.shape == (5,)
    assert bool(torch.isfinite(metrics.joint_ratio_quantiles).all())
    assert bool((metrics.joint_ratio_quantiles[1:] >= metrics.joint_ratio_quantiles[:-1]).all())
    assert metrics.minimum_joint_ratio <= metrics.maximum_joint_ratio
    assert metrics.nonfinite_joint_ratio_count == 0
    assert metrics.mean_common_gradient_norm_before_clip >= 0.0
    assert metrics.mean_prefix_gradient_norm_before_clip >= 0.0
    assert agent.is_finite()


def test_single_session_merge_is_not_an_actor_factor_but_still_trains_prefix_critic() -> None:
    algorithm = replace(ALGORITHM, ppo=replace(ALGORITHM.ppo, epochs_per_rollout=1, minibatch_decisions=512))
    _, agent, layout = _agents(61, algorithm)
    with torch.no_grad():
        agent.model.policy.action_type_head.weight.zero_()
        agent.model.policy.action_type_head.bias.copy_(torch.tensor([100.0, 50.0, -50.0, -100.0]))
    collected = collect_common_trace_training_rollout(
        ISACSSCEnv(CONFIG), agent, layout,
        lambda index, regime: generate_primitive_trace(CONFIG, 41001 + index, regime),
        0, 200, algorithm, ("independent",),
    )
    factor = collected.rollout.factor_credit
    assert factor is not None
    merge_rows = torch.tensor([action.merge_session_index >= 0 for action in collected.rollout.stored_actions])
    assert bool(merge_rows.any())
    assert not bool(factor.applicability[merge_rows, 1].any())
    before = tuple(value.detach().clone() for value in agent.algorithm.session_prefix_parameters)
    metrics = agent.algorithm.optimize_rollout(collected.rollout, generator=agent.minibatch_generator)
    after = tuple(agent.algorithm.session_prefix_parameters)
    assert metrics.merge_transition_count > 0
    assert metrics.single_session_merge_transition_count == metrics.merge_transition_count
    assert metrics.multi_session_merge_transition_count == 0
    assert any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))
    assert metrics.mean_session_reward_value_loss >= 0.0
    assert metrics.mean_session_constraint_value_loss >= 0.0


def test_common_trace_update_replays_exactly_with_fixed_generators() -> None:
    algorithm = replace(ALGORITHM, ppo=replace(ALGORITHM.ppo, epochs_per_rollout=2, minibatch_decisions=64))
    _, first, layout = _agents(71, algorithm)
    _, second, _ = _agents(71, algorithm)
    factory = lambda index, regime: generate_primitive_trace(CONFIG, 59001 + index, regime)
    first_rollout = collect_common_trace_training_rollout(
        ISACSSCEnv(CONFIG), first, layout, factory, 0, 201, algorithm, ("independent", "clustered"),
    )
    second_rollout = collect_common_trace_training_rollout(
        ISACSSCEnv(CONFIG), second, layout, factory, 0, 201, algorithm, ("independent", "clustered"),
    )
    assert first_rollout.metrics == second_rollout.metrics
    first_metrics = first.algorithm.optimize_rollout(first_rollout.rollout, generator=first.minibatch_generator)
    second_metrics = second.algorithm.optimize_rollout(second_rollout.rollout, generator=second.minibatch_generator)
    for field in fields(first_metrics):
        left = getattr(first_metrics, field.name)
        right = getattr(second_metrics, field.name)
        assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    for left, right in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.equal(left, right)