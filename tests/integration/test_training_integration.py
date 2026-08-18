from dataclasses import replace

import pytest
import torch

from isac_ssc.algorithms.buffers import ConstraintLayout
from isac_ssc.baselines.ppo_common_trace import build_common_trace_agent
from isac_ssc.baselines.ppo_joint_credit import build_joint_credit_agent
from isac_ssc.envs.action_space import identifier_key
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import ISACSSCEnv
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.training.rollout import (
    collect_common_trace_training_rollout,
    collect_training_rollout, evaluate_policy,
)
from isac_ssc.training.trainer import JointCreditPPOTrainer
from isac_ssc.utils.config import load_algorithm_config, load_config, load_experiment_config

ENV = load_config()
ALG = load_algorithm_config()
EXP = load_experiment_config()


def _first(seed=50001, regime="independent"):
    trace = generate_primitive_trace(ENV, seed, regime)
    env = ISACSSCEnv(ENV)
    observation = env.reset(trace)
    while observation is None and not env.terminated:
        observation = env.step(None).next_observation
    assert observation is not None
    return trace, observation


def _agent(seed=1, algorithm=ALG):
    trace, observation = _first()
    agent = build_joint_credit_agent(
        FeatureLayout.from_view(observation.set_view), algorithm, ENV,
        model_seed=seed, action_seed=seed + 1, minibatch_seed=seed + 2,
    )
    tenants = tuple(sorted((item.tenant_id for item in ENV.tenants), key=identifier_key))
    users = tuple(sorted({item.user_id for item in trace.communication_states}, key=identifier_key))
    return agent, ConstraintLayout(tenants, users), trace


def test_full_public_training_smoke_with_arbitrary_seed_and_both_regimes(tmp_path) -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    experiment = replace(
        EXP,
        training=replace(
            EXP.training, seed=987654, physical_slots=400,
            arrival_regimes=("independent", "clustered"), rollout_target_physical_slots=400,
        ),
        validation=replace(
            EXP.validation, enabled=True, interval_physical_slots=200, trace_seeds=(51001,),
            arrival_regimes=("independent", "clustered"), random_valid_replicates_per_trace=1,
        ),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=200),
        logging=replace(EXP.logging, progress=False, flush_every_records=1),
    )
    summary = JointCreditPPOTrainer(
        ENV, algorithm, experiment, output_root=tmp_path, run_name="smoke",
    ).run()
    assert summary.training_seed == 987654
    assert summary.requested_physical_slots == 400
    assert summary.completed_physical_slots == 400
    assert summary.valid_action_rate == 1.0
    run = tmp_path / "smoke"
    for name in (
        "latest.pt", "final.pt", "training.jsonl", "train_rollouts.csv", "train_episodes.csv",
        "train_constraints.csv", "train_tenants.csv", "train_communication_users.csv",
        "validation_summary.csv", "validation_traces.csv", "validation_constraints.csv",
        "validation_tenants.csv", "validation_communication_users.csv", "checkpoint_index.csv",
        "resume_segments.csv", "manifest.json", "effective_config.json",
        "segment_0000_provenance.json", "summary.json",
    ):
        assert (run / name).is_file()
    regimes = (run / "train_episodes.csv").read_text(encoding="utf-8")
    validation = (run / "validation_summary.csv").read_text(encoding="utf-8")
    assert "independent" in regimes and "clustered" in regimes
    assert "independent" in validation and "clustered" in validation and "overall" in validation


def test_short_ppo_update_is_finite_and_valid() -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    agent, layout, _ = _agent(20, algorithm)
    collected = collect_training_rollout(
        ISACSSCEnv(ENV), agent, layout,
        lambda index, regime: generate_primitive_trace(ENV, 56001 + index, regime),
        0, 201, algorithm, ("independent", "clustered"),
    )
    ppo = agent.algorithm.optimize_rollout(collected.rollout, generator=agent.minibatch_generator)
    dual = agent.algorithm.update_duals(collected.rollout.episode_constraint_totals)
    agent.normalizer.update(collected.focal_observations)
    assert collected.metrics.valid_action_rate == 1.0
    assert ppo.optimizer_steps > 0
    assert torch.isfinite(ppo.mean_constraint_surrogates).all()
    assert torch.isfinite(dual.dual_values_after).all()
    assert agent.is_finite()


def test_deterministic_validation_replay_is_regime_separated_with_macro_overall() -> None:
    agent, _, _ = _agent(30)
    traces = (
        generate_primitive_trace(ENV, 51001, "independent"),
        generate_primitive_trace(ENV, 51001, "clustered"),
    )
    first = evaluate_policy(ENV, agent, traces, physical_slot=100)
    second = evaluate_policy(ENV, agent, traces, physical_slot=100)
    assert first == second
    assert tuple(item.arrival_regime for item in first.regimes) == ("independent", "clustered")
    assert first.overall.arrival_regime == "overall"
    assert first.overall.std_return is None
    assert first.overall.mean_return == pytest.approx(sum(item.mean_return for item in first.regimes) / 2)
    assert first.valid_action_rate == 1.0


def test_short_common_trace_update_is_finite_and_valid() -> None:
    algorithm = replace(
        ALG,
        ppo=replace(
            ALG.ppo,
            epochs_per_rollout=1,
            minibatch_decisions=256,
        ),
    )
    trace, observation = _first()
    agent = build_common_trace_agent(
        FeatureLayout.from_view(observation.set_view),
        algorithm,
        ENV,
        model_seed=40,
        action_seed=41,
        minibatch_seed=42,
    )
    tenants = tuple(sorted(
        (item.tenant_id for item in ENV.tenants),
        key=identifier_key,
    ))
    users = tuple(sorted(
        {item.user_id for item in trace.communication_states},
        key=identifier_key,
    ))
    layout = ConstraintLayout(tenants, users)
    collected = collect_common_trace_training_rollout(
        ISACSSCEnv(ENV),
        agent,
        layout,
        lambda index, regime: generate_primitive_trace(
            ENV, 58001 + index, regime,
        ),
        0,
        201,
        algorithm,
        ("independent", "clustered"),
    )
    assert collected.rollout.factor_credit is not None
    ppo = agent.algorithm.optimize_rollout(
        collected.rollout,
        generator=agent.minibatch_generator,
    )
    dual = agent.algorithm.update_duals(
        collected.rollout.episode_constraint_totals,
    )
    agent.normalizer.update(collected.focal_observations)
    assert collected.metrics.valid_action_rate == 1.0
    assert ppo.optimizer_steps > 0
    assert torch.isfinite(
        ppo.mean_constraint_surrogates,
    ).all()
    assert torch.isfinite(
        ppo.mean_constraint_surrogates_by_factor,
    ).all()
    assert torch.isfinite(ppo.joint_ratio_quantiles).all()
    assert torch.isfinite(dual.dual_values_after).all()
    assert agent.is_finite()