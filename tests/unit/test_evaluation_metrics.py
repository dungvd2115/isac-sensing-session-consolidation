from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from scripts.evaluate import build_parser
from isac_ssc.baselines.ppo_common_trace import build_common_trace_agent
from isac_ssc.baselines.ppo_joint_credit import build_joint_credit_agent
from isac_ssc.core.entities import RequestState
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import ISACSSCEnv
from isac_ssc.evaluation import evaluator
from isac_ssc.evaluation.evaluator import (
    BASELINE_REGISTRY, EVALUATION_METHODS,
    LEARNED_POLICY_NAME, RANDOM_VALID_NAME, run_baseline_evaluation,
)
from isac_ssc.evaluation.metrics import request_state_counts, safe_ratio
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.training import rollout, trainer
from isac_ssc.training.checkpoint import (
    CheckpointMetadata, CheckpointValidationError, checkpoint_sha256,
    load_policy_checkpoint, save_checkpoint,
)
from isac_ssc.utils.config import (
    COMMON_TRACE_METHOD, credit_assignment_schema,
    load_algorithm_config, load_config,
)
from isac_ssc.utils.seeding import SeedContract


ALG = load_algorithm_config()


def _reduced_config(horizon_slots: int = 12):
    config = load_config()
    system = MappingProxyType(dict(config.system, horizon_slots=horizon_slots))
    limits = MappingProxyType(dict(config.oracle["instance_selection_limits"], horizon_slots=horizon_slots))
    oracle = MappingProxyType(dict(config.oracle, instance_selection_limits=limits))
    return replace(config, system=system, oracle=oracle)


def _first_observation(config, seed: int = 41001, regime: str = "independent"):
    trace = generate_primitive_trace(config, seed, regime)
    env = ISACSSCEnv(config)
    observation = env.reset(trace)
    while observation is None and not env.terminated:
        observation = env.step(None).next_observation
    assert observation is not None
    return trace, observation


def _agent(config, algorithm=ALG, seed: int = 1, method: str = LEARNED_POLICY_NAME):
    trace, observation = _first_observation(config)
    layout = FeatureLayout.from_view(observation.set_view)
    builder = (
        build_common_trace_agent
        if method == COMMON_TRACE_METHOD
        else build_joint_credit_agent
    )
    agent = builder(
        layout, algorithm, config, model_seed=seed, action_seed=seed+1, minibatch_seed=seed+2,
    )
    return agent, trace, observation


def _checkpoint(
    tmp_path: Path, config, *, algorithm=ALG, method: str = LEARNED_POLICY_NAME,
    state: dict | None = None, name: str = "selected.pt",
):
    agent, _, observation = _agent(config, algorithm, 11, method)
    agent.normalizer.update((observation,))
    agent.algorithm.dual_values.fill_(3.0)
    parameter = next(agent.model.parameters())
    agent.algorithm.optimizer.state[parameter].update({
        "step": torch.tensor(9.0), "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    })
    torch.rand(3, generator=agent.action_generator)
    torch.rand(3, generator=agent.minibatch_generator)
    metadata = CheckpointMetadata.current(
        method=method, credit_assignment_schema=credit_assignment_schema(method),
        training_seed=777, feature_schema_digest=agent.model.layout.schema_digest,
        architecture_signature="standalone-selected-policy-test",
        environment_semantic_digest="provenance-only-environment",
        validation_protocol_digest="provenance-only-validation",
        constraint_labels=tuple(f"constraint:{index}" for index in range(agent.algorithm.constraint_count)),
    )
    path = tmp_path/name
    save_checkpoint(path, agent, metadata, {"progress": {"completed_physical_slots": 321}} if state is None else state)
    return path, agent, metadata


def test_safe_ratio_uses_none_only_for_zero_denominator() -> None:
    assert safe_ratio(3, 2) == 1.5
    assert safe_ratio(0, 0) is None
    with pytest.raises(ValueError):
        safe_ratio(1, -1)


def test_request_state_counts_follow_canonical_lifecycle_order() -> None:
    requests = tuple(type("Request", (), {"state": state})() for state in (
        RequestState.WAITING, RequestState.COMPLETED, RequestState.COMPLETED,
    ))
    assert request_state_counts(requests) == (
        ("waiting", 1), ("active", 0), ("completed", 2), ("failed", 0),
        ("expired", 0), ("rejected", 0),
    )


def test_evaluate_cli_parses_random_valid_and_checkpoint_options() -> None:
    arguments = build_parser().parse_args([
        "--seeds", "52001", "52002", "--arrival-regimes", "independent", "clustered",
        "--baselines", *EVALUATION_METHODS, "--random-valid-root-seed", "53001",
        "--random-valid-replicates", "4", "--checkpoint", "/tmp/best.pt",
        "--algorithm-config", "configs/algorithm/constrained_ppo.yaml",
        "--bootstrap-root-seed", "54001", "--bootstrap-samples", "64",
        "--output", "/tmp/evaluation.json",
    ])
    assert arguments.seeds == [52001, 52002]
    assert tuple(arguments.baselines) == EVALUATION_METHODS
    assert arguments.random_valid_root_seed == 53001
    assert arguments.random_valid_replicates == 4
    assert arguments.checkpoint == Path("/tmp/best.pt")
    assert arguments.algorithm_config == Path("configs/algorithm/constrained_ppo.yaml")
    assert arguments.bootstrap_root_seed == 54001
    assert arguments.bootstrap_samples == 64


def test_generic_evaluator_preserves_deterministic_report_fields() -> None:
    report = run_baseline_evaluation(
        _reduced_config(), (41001,), ("independent",), ("no_consolidation",), bootstrap_samples=16,
    )
    assert tuple(BASELINE_REGISTRY) == (
        "no_consolidation", "static_compatibility_merge", "greedy_incremental_cost", "sla_aware_greedy",
    )
    assert len(report.episodes) == len(report.aggregates) == len(report.macro_aggregates) == 1
    episode, aggregate = report.episodes[0], report.aggregates[0]
    assert episode.physical_step_count == 12
    assert episode.merge_count == 0
    assert episode.replicate is None
    assert aggregate.baseline_name == "no_consolidation"
    assert aggregate.arrival_regime == "independent"
    payload = report.to_dict()["episodes"][0]
    assert payload["cumulative_reward"] == episode.cumulative_reward
    assert payload["reward_total"] == episode.cumulative_reward
    assert payload["method_name"] == "no_consolidation"


def test_random_valid_has_exact_coverage_counts_and_accounting(monkeypatch) -> None:
    original_step = rollout.ISACSSCEnv.step

    def checked_step(env, action):
        if action is not None:
            assert action in env.current_action_masks().feasible_actions
        return original_step(env, action)

    monkeypatch.setattr(rollout.ISACSSCEnv, "step", checked_step)
    seeds, regimes = (41001, 41002), ("independent", "clustered")
    report = run_baseline_evaluation(
        _reduced_config(), seeds, regimes, EVALUATION_METHODS,
        random_valid_root_seed=53001, random_valid_replicates=2,
        bootstrap_root_seed=54001, bootstrap_samples=32,
    )
    assert report.unique_primitive_trace_count == 4
    assert report.deterministic_heuristic_episode_count == 16
    assert report.random_valid_episode_count == 8
    assert report.learned_policy_episode_count == 0
    assert len(report.episodes) == 24
    assert len(report.aggregates) == 10
    assert len(report.macro_aggregates) == 5
    expected = {
        (method, seed, regime, None) for method in BASELINE_REGISTRY
        for seed in seeds for regime in regimes
    }
    expected.update({
        (RANDOM_VALID_NAME, seed, regime, replicate) for seed in seeds
        for regime in regimes for replicate in range(2)
    })
    actual = {
        (item.method_name, item.root_seed, item.arrival_regime, item.replicate)
        for item in report.episodes
    }
    assert actual == expected
    for episode in report.episodes:
        assert episode.all_finite
        assert episode.invalid_action_count == 0
        assert episode.valid_action_rate == 1.0
        assert episode.physical_step_count == episode.focal_decision_count+episode.no_request_slot_count
        assert episode.focal_decision_count == (
            episode.merge_count+episode.create_count+episode.defer_count+episode.reject_count
        )
        assert episode.focal_decision_count == episode.valid_action_count+episode.invalid_action_count
        assert episode.arrived_request_count == sum(count for _, count in episode.terminal_request_state_counts)
    random_episodes = tuple(item for item in report.episodes if item.method_name == RANDOM_VALID_NAME)
    assert {item.replicate for item in random_episodes} == {0, 1}
    assert all(item.method_category == "stochastic_sanity_baseline" for item in random_episodes)
    assert all(item.action_sequence == item.focal_sequence == () for item in random_episodes)
    for aggregate in report.aggregates:
        assert aggregate.seed_count == 2
        assert aggregate.unique_trace_count == 2
        assert aggregate.all_finite
        assert aggregate.valid_action_rate == 1.0
        assert aggregate.episode_count == (4 if aggregate.method_name == RANDOM_VALID_NAME else 2)
        assert aggregate.replicates_per_trace == (2 if aggregate.method_name == RANDOM_VALID_NAME else 1)
        intervals = {item.metric: item for item in aggregate.confidence_intervals}
        assert set(intervals) == {
            "reward_total", "completed_value_total", "normalized_completed_value", "completion_ratio",
            "acceptance_ratio", "sensing_resource_cost_total", "positive_constraint_excess",
            "requests_per_created_session",
        }
        for metric in (
            "reward_total", "completed_value_total", "completion_ratio", "acceptance_ratio",
            "sensing_resource_cost_total", "positive_constraint_excess",
        ):
            assert intervals[metric].sample_count == 2
    payload = report.to_dict()
    assert payload["counts"] == {
        "unique_primitive_traces": 4, "deterministic_heuristic_episodes": 16,
        "random_valid_episodes": 8, "learned_policy_episodes": 0, "total_episodes": 24,
        "aggregate_groups": 10, "macro_groups": 5,
    }
    sanity = payload["method_groups"]["stochastic_sanity_baselines"][0]
    assert sanity["name"] == RANDOM_VALID_NAME
    assert "not a theoretical lower bound" in sanity["interpretation"]


def test_random_valid_replay_and_method_order_are_deterministic() -> None:
    config = _reduced_config()
    first = run_baseline_evaluation(
        config, (41001, 41002), ("independent", "clustered"),
        ("no_consolidation", RANDOM_VALID_NAME), random_valid_root_seed=53001,
        random_valid_replicates=2, bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    replay = run_baseline_evaluation(
        config, (41002, 41001), ("clustered", "independent"),
        (RANDOM_VALID_NAME, "no_consolidation"), random_valid_root_seed=53001,
        random_valid_replicates=2, bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    assert first.to_dict() == replay.to_dict()


def test_random_valid_reuses_training_implementation_and_keyed_replicate_streams() -> None:
    assert evaluator.evaluate_random_valid is rollout.evaluate_random_valid
    assert trainer.evaluate_random_valid is rollout.evaluate_random_valid
    contract = SeedContract.from_config(load_config())
    first = contract.derive_uint64(53001, RANDOM_VALID_NAME, 52001, "clustered", 0)
    replay = contract.derive_uint64(53001, RANDOM_VALID_NAME, 52001, "clustered", 0)
    second = contract.derive_uint64(53001, RANDOM_VALID_NAME, 52001, "clustered", 1)
    assert first == replay
    assert first != second


def test_policy_checkpoint_loader_restores_exact_read_only_inference_state(tmp_path) -> None:
    config = _reduced_config()
    source_algorithm = replace(ALG, normalization=replace(ALG.normalization, clip=3.5, epsilon=1.0e-6))
    path, source, metadata = _checkpoint(tmp_path, config, algorithm=source_algorithm)
    target, _, _ = _agent(config, ALG, 91)
    torch_state = torch.random.get_rng_state().clone()
    action_state = target.action_generator.get_state().clone()
    minibatch_state = target.minibatch_generator.get_state().clone()
    dual_values = target.algorithm.dual_values.clone()
    restored_metadata, state = load_policy_checkpoint(path, target, expected_method=LEARNED_POLICY_NAME)
    assert restored_metadata == metadata
    assert state["progress"]["completed_physical_slots"] == 321
    assert target.normalizer.clip == source.normalizer.clip == 3.5
    assert target.normalizer.epsilon == source.normalizer.epsilon == 1.0e-6
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.equal(target.action_generator.get_state(), action_state)
    assert torch.equal(target.minibatch_generator.get_state(), minibatch_state)
    assert torch.equal(target.algorithm.dual_values, dual_values)
    assert not torch.equal(source.algorithm.dual_values, target.algorithm.dual_values)
    assert not target.algorithm.optimizer.state
    assert all(torch.equal(left, right) for left, right in zip(
        source.model.state_dict().values(), target.model.state_dict().values(), strict=True,
    ))
    assert source.normalizer.state().request_count == target.normalizer.state().request_count

    with pytest.raises(CheckpointValidationError, match="checkpoint method"):
        load_policy_checkpoint(path, target, expected_method=COMMON_TRACE_METHOD)

    legacy = torch.load(path, map_location="cpu", weights_only=True)
    legacy.pop("normalizer_configuration")
    legacy_path = tmp_path/"missing_normalizer_configuration.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(CheckpointValidationError, match="normalization state"):
        load_policy_checkpoint(legacy_path, target, expected_method=LEARNED_POLICY_NAME)


def test_selected_checkpoint_uses_shared_policy_evaluation_and_paired_traces(tmp_path, monkeypatch) -> None:
    config = _reduced_config()
    source_algorithm = replace(ALG, normalization=replace(ALG.normalization, clip=4.0, epsilon=2.0e-6))
    path, _, metadata = _checkpoint(tmp_path, config, algorithm=source_algorithm)
    original_digest = checkpoint_sha256(path)
    original_step = rollout.ISACSSCEnv.step

    def checked_step(env, action):
        if action is not None:
            assert action in env.current_action_masks().feasible_actions
        return original_step(env, action)

    monkeypatch.setattr(rollout.ISACSSCEnv, "step", checked_step)
    report = run_baseline_evaluation(
        config, (41001,), ("independent", "clustered"), ("no_consolidation", RANDOM_VALID_NAME),
        random_valid_root_seed=53001, random_valid_replicates=2,
        checkpoint_path=path, algorithm=ALG, bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    assert evaluator.evaluate_policy is rollout.evaluate_policy
    assert report.unique_primitive_trace_count == 2
    assert report.deterministic_heuristic_episode_count == 2
    assert report.random_valid_episode_count == 4
    assert report.learned_policy_episode_count == 2
    assert len(report.episodes) == 8
    assert len(report.aggregates) == 6
    assert len(report.macro_aggregates) == 3
    assert checkpoint_sha256(path) == original_digest
    checkpoint = report.checkpoint
    assert checkpoint is not None
    assert checkpoint.checkpoint_path == path.resolve().as_posix()
    assert checkpoint.checkpoint_sha256 == original_digest
    assert checkpoint.checkpoint_physical_slot == 321
    assert checkpoint.algorithm_config_path == ALG.source_path.resolve().as_posix()
    assert checkpoint.method == metadata.method
    assert checkpoint.credit_assignment_schema == metadata.credit_assignment_schema
    assert checkpoint.training_seed == 777
    assert checkpoint.environment_semantic_digest == "provenance-only-environment"
    assert checkpoint.validation_protocol_digest == "provenance-only-validation"
    assert checkpoint.normalization_clip == 4.0
    assert checkpoint.normalization_epsilon == 2.0e-6
    learned = tuple(item for item in report.episodes if item.method_name == LEARNED_POLICY_NAME)
    assert {(item.root_seed, item.arrival_regime, item.replicate) for item in learned} == {
        (41001, "independent", None), (41001, "clustered", None),
    }
    assert all(item.method_category == "learned_policy" for item in learned)
    assert all(item.invalid_action_count == 0 and item.valid_action_rate == 1.0 for item in learned)
    assert all(item.all_finite for item in learned)
    for item in learned:
        assert len(item.action_sequence) == len(item.focal_sequence) == item.physical_step_count
        assert sum(value is not None for value in item.focal_sequence) == item.focal_decision_count
        assert all(
            (action == "none") == (focal is None)
            for action, focal in zip(item.action_sequence, item.focal_sequence, strict=True)
        )
    payload = report.to_dict()
    assert payload["method_groups"]["learned_policies"] == [LEARNED_POLICY_NAME]
    assert payload["counts"] == {
        "unique_primitive_traces": 2, "deterministic_heuristic_episodes": 2,
        "random_valid_episodes": 4, "learned_policy_episodes": 2, "total_episodes": 8,
        "aggregate_groups": 6, "macro_groups": 3,
    }
    expected_traces = {(41001, "independent"), (41001, "clustered")}
    for method in ("no_consolidation", RANDOM_VALID_NAME, LEARNED_POLICY_NAME):
        assert {
            (item.root_seed, item.arrival_regime) for item in report.episodes if item.method_name == method
        } == expected_traces

    trace = generate_primitive_trace(config, 41001, "independent")
    empty = replace(trace, trace_id=trace.trace_id+"-empty", request_descriptors=())
    observation = evaluator._first_focal_observation(config, (empty, trace))
    assert FeatureLayout.from_view(observation.set_view).schema_digest == metadata.feature_schema_digest
    with pytest.raises(ValueError, match="no focal observation"):
        evaluator._first_focal_observation(config, (empty,))

    no_slot_path, _, _ = _checkpoint(tmp_path, config, algorithm=source_algorithm, state={}, name="no_slot.pt")
    policy_only = run_baseline_evaluation(
        config, (41001,), ("independent",), (), checkpoint_path=no_slot_path, algorithm=ALG,
        bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    assert policy_only.methods == (LEARNED_POLICY_NAME,)
    assert policy_only.checkpoint.checkpoint_physical_slot is None
    assert policy_only.learned_policy_episode_count == 1
    assert len(policy_only.episodes) == len(policy_only.aggregates) == len(policy_only.macro_aggregates) == 1


def test_common_trace_checkpoint_evaluates_through_the_same_endpoint(tmp_path) -> None:
    config = _reduced_config()
    path, _, metadata = _checkpoint(
        tmp_path, config, method=COMMON_TRACE_METHOD, name="common_trace.pt",
    )
    report = run_baseline_evaluation(
        config, (41001,), ("independent",), (),
        checkpoint_path=path, algorithm=ALG,
        bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    assert report.learned_policies == (COMMON_TRACE_METHOD,)
    assert report.learned_policy_episode_count == 1
    assert report.checkpoint is not None
    assert report.checkpoint.method == COMMON_TRACE_METHOD
    assert report.checkpoint.credit_assignment_schema == metadata.credit_assignment_schema
    assert report.episodes[0].method_name == COMMON_TRACE_METHOD
    assert report.episodes[0].valid_action_rate == 1.0
    assert report.episodes[0].all_finite


def test_selected_checkpoint_replay_is_order_independent(tmp_path) -> None:
    config = _reduced_config()
    path, _, _ = _checkpoint(tmp_path, config)
    first = run_baseline_evaluation(
        config, (41001, 41002), ("independent", "clustered"),
        ("no_consolidation", RANDOM_VALID_NAME), random_valid_root_seed=53001,
        random_valid_replicates=2, checkpoint_path=path, algorithm=ALG,
        bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    replay = run_baseline_evaluation(
        config, (41002, 41001), ("clustered", "independent"),
        (RANDOM_VALID_NAME, "no_consolidation"), random_valid_root_seed=53001,
        random_valid_replicates=2, checkpoint_path=path, algorithm=ALG,
        bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    assert first.to_dict() == replay.to_dict()


def test_evaluator_rejects_only_invalid_or_ambiguous_inputs() -> None:
    config = _reduced_config()
    with pytest.raises(ValueError, match="seeds must not contain duplicates"):
        run_baseline_evaluation(config, (41001, 41001), ("independent",), ("no_consolidation",))
    with pytest.raises(ValueError, match="arrival regimes must not contain duplicates"):
        run_baseline_evaluation(config, (41001,), ("independent", "independent"), ("no_consolidation",))
    with pytest.raises(ValueError, match="methods must not contain duplicates"):
        run_baseline_evaluation(config, (41001,), ("independent",), ("no_consolidation", "no_consolidation"))
    with pytest.raises(ValueError, match="random_valid requires"):
        run_baseline_evaluation(config, (41001,), ("independent",), (RANDOM_VALID_NAME,))
    with pytest.raises(ValueError, match="random-valid options require"):
        run_baseline_evaluation(
            config, (41001,), ("independent",), ("no_consolidation",),
            random_valid_root_seed=53001, random_valid_replicates=2,
        )
    with pytest.raises(ValueError, match="replicates must be positive"):
        run_baseline_evaluation(
            config, (41001,), ("independent",), (RANDOM_VALID_NAME,),
            random_valid_root_seed=53001, random_valid_replicates=0,
        )
    with pytest.raises(ValueError, match="both checkpoint_path and algorithm"):
        run_baseline_evaluation(config, (41001,), ("independent",), (), checkpoint_path="selected.pt")
    with pytest.raises(ValueError, match="at least one baseline or checkpoint"):
        run_baseline_evaluation(config, (41001,), ("independent",), ())


def test_random_valid_extension_does_not_change_heuristic_outputs() -> None:
    config = _reduced_config()
    deterministic = run_baseline_evaluation(
        config, (41001, 41002), ("independent", "clustered"), tuple(BASELINE_REGISTRY),
        bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    mixed = run_baseline_evaluation(
        config, (41001, 41002), ("independent", "clustered"), EVALUATION_METHODS,
        random_valid_root_seed=53001, random_valid_replicates=2,
        bootstrap_root_seed=54001, bootstrap_samples=16,
    )
    mixed_episodes = tuple(item for item in mixed.episodes if item.method_name in BASELINE_REGISTRY)
    mixed_aggregates = tuple(item for item in mixed.aggregates if item.method_name in BASELINE_REGISTRY)
    mixed_macro = tuple(item for item in mixed.macro_aggregates if item.method_name in BASELINE_REGISTRY)
    assert mixed_episodes == deterministic.episodes
    assert mixed_aggregates == deterministic.aggregates
    assert mixed_macro == deterministic.macro_aggregates