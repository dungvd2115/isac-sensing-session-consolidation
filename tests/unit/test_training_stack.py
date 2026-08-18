from dataclasses import FrozenInstanceError, replace
import csv
import inspect
from pathlib import Path

import pytest
import torch

from isac_ssc.algorithms.buffers import ConstraintLayout
from isac_ssc.baselines.ppo_common_trace import build_common_trace_agent
from isac_ssc.baselines.ppo_joint_credit import (
    JointCreditPPOValidationError, NormalizedEdgeFreeSetActorCritic,
    RunningFeatureNormalizer, build_joint_credit_agent,
)
from isac_ssc.envs.action_space import identifier_key
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import ISACSSCEnv
from isac_ssc.models.policy import build_policy_batch
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.training.checkpoint import CheckpointMetadata, CheckpointValidationError, load_checkpoint, save_checkpoint
from isac_ssc.training.logging import CsvTable, TrainingArtifacts, TrainingLogError, write_json
from isac_ssc.training.rollout import collect_episode, collect_training_rollout
from isac_ssc.training.trainer import JointCreditPPOTrainer, TrainerValidationError
from isac_ssc.utils.config import (
    COMMON_TRACE_METHOD, JOINT_CREDIT_METHOD,
    ConfigError, DEFAULT_EXPERIMENT_CONFIG_PATH,
    credit_assignment_schema, load_algorithm_config, load_config, load_experiment_config,
)

ENV = load_config()
ALG = load_algorithm_config()
EXP = load_experiment_config()


def _candidate(tmp_path: Path, replacements: tuple[tuple[str, str], ...]) -> Path:
    text = DEFAULT_EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new, 1)
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


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
    layout = FeatureLayout.from_view(observation.set_view)
    agent = build_joint_credit_agent(
        layout, algorithm, ENV, model_seed=seed, action_seed=seed + 1, minibatch_seed=seed + 2,
    )
    tenants = tuple(sorted((item.tenant_id for item in ENV.tenants), key=identifier_key))
    users = tuple(sorted({item.user_id for item in trace.communication_states}, key=identifier_key))
    return agent, ConstraintLayout(tenants, users), trace, observation


def test_experiment_config_is_default_not_a_value_lock(tmp_path) -> None:
    path = _candidate(tmp_path, (
        ("seed: 0", "seed: 123"),
        ("physical_slots: 100000", "physical_slots: 800"),
        ('arrival_regimes: ["independent", "clustered"]', 'arrival_regimes: ["clustered"]'),
        ("interval_physical_slots: 10000", "interval_physical_slots: 200"),
        ('best_metric: "paired_return_difference"', 'best_metric: "validation_return"'),
    ))
    config = load_experiment_config(path)
    assert config.training.seed == 123
    assert config.training.physical_slots == 800
    assert config.training.arrival_regimes == ("clustered",)
    assert config.validation.interval_physical_slots == 200
    assert config.checkpoint.best_metric == "validation_return"
    with pytest.raises(FrozenInstanceError):
        config.training.seed = 1


def test_common_trace_experiment_changes_only_method_identity() -> None:
    joint = load_experiment_config()
    common_trace_path = DEFAULT_EXPERIMENT_CONFIG_PATH.with_name("common_trace.yaml")
    common_trace = load_experiment_config(common_trace_path)
    assert joint.method == JOINT_CREDIT_METHOD
    assert common_trace.method == COMMON_TRACE_METHOD
    assert replace(common_trace, method=joint.method) == joint
    assert joint.validation.trace_seeds == tuple(range(51001, 51021))
    assert common_trace.validation.trace_seeds == tuple(range(51001, 51021))
    assert credit_assignment_schema(joint.method) == "joint_trajectory_credit_v2_scale_consistent"
    assert credit_assignment_schema(common_trace.method) == "common_trace_leave_one_out_mc_factor_credit_v1"


@pytest.mark.parametrize("old,new", (
    ("seed: 0", "seed: -1"),
    ("physical_slots: 100000", "physical_slots: 0"),
    ('arrival_regimes: ["independent", "clustered"]', 'arrival_regimes: ["unknown"]'),
    ('method: "joint_credit_constrained_ppo"', 'method: "edge_free_set_constrained_ppo"'),
    ("keep_top_k: 1", "keep_top_k: 0"),
))
def test_experiment_config_rejects_only_invalid_domains(tmp_path, old, new) -> None:
    with pytest.raises(ConfigError):
        load_experiment_config(_candidate(tmp_path, ((old, new),)))


def test_normalizer_ignores_padding_preserves_indicators_and_round_trips() -> None:
    _, first = _first(50001)
    _, second = _first(50002)
    batch = build_policy_batch((first, second))
    normalizer = RunningFeatureNormalizer(batch.layout, clip=0.5, epsilon=1e-8)
    before = normalizer.transform(batch)
    assert torch.equal(before.request_features, batch.encoder_input.request_features)
    normalizer.update((first, second))
    transformed = normalizer.transform(batch)
    request_indicator = torch.tensor([item.unit == "indicator" for item in batch.layout.request_specs])
    assert torch.equal(
        transformed.request_features[..., request_indicator],
        batch.encoder_input.request_features[..., request_indicator],
    )
    assert torch.count_nonzero(
        transformed.request_features.masked_select(batch.encoder_input.request_padding_mask.unsqueeze(-1)),
    ) == 0
    restored = RunningFeatureNormalizer(batch.layout, clip=0.5, epsilon=1e-8)
    restored.load_state_dict(normalizer.state_dict())
    for name in normalizer.state().__dataclass_fields__:
        left, right = getattr(normalizer.state(), name), getattr(restored.state(), name)
        assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    with pytest.raises(JointCreditPPOValidationError):
        restored.load_state_dict({**normalizer.state_dict(), "schema_digest": "0" * 64})


def test_normalized_model_forward_reuses_the_validated_policy_batch() -> None:
    transform = inspect.getsource(RunningFeatureNormalizer.transform)
    forward = inspect.getsource(NormalizedEdgeFreeSetActorCritic.forward)
    assert "build_policy_batch" not in transform and "FactorizedPolicyBatch(" not in transform
    assert "build_policy_batch" not in forward and "replace(" not in forward
    assert "_with_features" in transform


def test_episode_metrics_keep_reward_decomposition_resources_and_constraints() -> None:
    agent, layout, trace, _ = _agent()
    transitions, totals, metrics, observations = collect_episode(
        ISACSSCEnv(ENV), agent, trace, layout, deterministic=True,
    )
    weight = ENV.reward["sensing_resource_cost_weight"]
    assert metrics.reward_total == pytest.approx(
        metrics.completed_value_total - weight * metrics.sensing_resource_cost_total,
    )
    assert metrics.physical_slots == trace.horizon_slots
    assert sum(item.physical_slot_span for item in transitions) <= trace.horizon_slots
    assert metrics.sensing_bandwidth_hz_slot_sum >= metrics.sensing_bandwidth_hz_max >= 0.0
    assert metrics.sensing_power_w_slot_sum >= metrics.sensing_power_w_max >= 0.0
    assert metrics.arrived_request_value_total >= metrics.completed_value_total >= 0.0
    assert metrics.normalized_completed_value is None or 0.0 <= metrics.normalized_completed_value <= 1.0
    assert metrics.communication_users
    if metrics.network_mean_user_shortfall is None:
        assert all(item.active_demand_slots == 0 for item in metrics.communication_users)
        assert metrics.fraction_users_within_budget is None
    else:
        assert 0.0 <= metrics.network_mean_user_shortfall <= 1.0
        assert 0.0 <= metrics.fraction_users_within_budget <= 1.0
    assert len(metrics.tenant_residual_totals) == layout.tenant_count
    assert len(metrics.communication_residual_totals) == layout.communication_count
    assert len(observations) == metrics.focal_decisions
    assert totals.reward_total == pytest.approx(metrics.reward_total)


def test_rollout_cycles_any_requested_regime_sequence_without_updating_normalizer() -> None:
    agent, layout, _, _ = _agent()
    calls = []

    def factory(index, regime):
        calls.append((index, regime))
        return generate_primitive_trace(ENV, 55000 + index, regime)

    before = agent.normalizer.state().request_count
    collected = collect_training_rollout(
        ISACSSCEnv(ENV), agent, layout, factory, 0, 401, ALG,
        ("clustered", "independent", "clustered"),
    )
    assert [item[1] for item in calls] == ["clustered", "independent", "clustered"]
    assert collected.metrics.physical_slots == 600
    assert agent.normalizer.state().request_count == before == 0


def test_checkpoint_round_trip_and_nonfinite_save_guard(tmp_path) -> None:
    agent, _, _, observation = _agent(20)
    agent.normalizer.update((observation,))
    agent.algorithm.dual_values[0] = 0.5
    metadata = CheckpointMetadata.current(
        method=EXP.method, credit_assignment_schema=credit_assignment_schema(EXP.method),
        training_seed=123, feature_schema_digest=agent.model.layout.schema_digest,
        architecture_signature="test", environment_semantic_digest="env",
        validation_protocol_digest="validation",
        constraint_labels=tuple(f"constraint:{index}" for index in range(agent.algorithm.constraint_count)),
    )
    path = tmp_path / "state.pt"
    save_checkpoint(path, agent, metadata, {"progress": {"completed_physical_slots": 200}, "validations": []})
    restored, _, _, _ = _agent(99)
    restored_metadata = replace(metadata, feature_schema_digest=restored.model.layout.schema_digest)
    _, state = load_checkpoint(path, restored, restored_metadata)
    assert state["progress"]["completed_physical_slots"] == 200
    assert torch.equal(agent.algorithm.dual_values, restored.algorithm.dual_values)
    assert torch.equal(agent.action_generator.get_state(), restored.action_generator.get_state())
    assert torch.equal(agent.minibatch_generator.get_state(), restored.minibatch_generator.get_state())
    with pytest.raises(CheckpointValidationError, match="finite"):
        save_checkpoint(tmp_path / "bad_state.pt", agent, metadata, {"score": float("nan")})
    assert not (tmp_path / "bad_state.pt").exists()
    next(agent.model.parameters()).data.fill_(float("nan"))
    with pytest.raises(CheckpointValidationError, match="non-finite"):
        save_checkpoint(tmp_path / "broken.pt", agent, metadata, {})
    assert not (tmp_path / "broken.pt").exists()


def test_common_trace_checkpoint_round_trip_and_cross_method_rejection(tmp_path) -> None:
    trace, observation = _first()
    layout = FeatureLayout.from_view(observation.set_view)
    agent = build_common_trace_agent(
        layout, ALG, ENV, model_seed=21, action_seed=22, minibatch_seed=23,
    )
    metadata = CheckpointMetadata.current(
        method=COMMON_TRACE_METHOD,
        credit_assignment_schema=credit_assignment_schema(COMMON_TRACE_METHOD),
        training_seed=321, feature_schema_digest=layout.schema_digest,
        architecture_signature="common-trace-test", environment_semantic_digest="env",
        validation_protocol_digest="validation",
        constraint_labels=tuple(
            f"constraint:{index}" for index in range(agent.algorithm.constraint_count)
        ),
    )
    path = tmp_path / "common_trace.pt"
    save_checkpoint(path, agent, metadata, {"progress": {"completed_physical_slots": 0}})
    restored = build_common_trace_agent(
        layout, ALG, ENV, model_seed=99, action_seed=100, minibatch_seed=101,
    )
    load_checkpoint(path, restored, metadata)
    for left, right in zip(
        agent.model.state_dict().values(), restored.model.state_dict().values(), strict=True,
    ):
        assert torch.equal(left, right)
    joint, _, _, _ = _agent(30)
    with pytest.raises(CheckpointValidationError, match="checkpoint method"):
        load_checkpoint(path, joint, metadata)


def test_checkpoint_v3_is_rejected_without_silent_migration(tmp_path) -> None:
    path = tmp_path / "legacy_v3.pt"
    torch.save({"schema_version": "isac-ssc-training-checkpoint-v3"}, path)
    with pytest.raises(CheckpointValidationError, match="unsupported checkpoint schema"):
        from isac_ssc.training.checkpoint import read_checkpoint_metadata
        read_checkpoint_metadata(path)


def test_training_artifacts_create_analysis_ready_files_and_reject_nonfinite_csv(tmp_path) -> None:
    artifacts = TrainingArtifacts(tmp_path, append=False, jsonl=True, csv_enabled=True, flush_every_records=1)
    artifacts.event("rollout", {"loss": 1.0})
    artifacts.row("rollouts", {"rollout_index": 1, "end_slot": 200, "reward_total": 3.0})
    artifacts.row("episodes", {"episode_index": 0, "arrival_regime": "independent", "reward_total": 3.0})
    artifacts.row("validation", {"actual_physical_slot": 200, "arrival_regime": "overall", "policy_mean_return": 4.0})
    artifacts.row("validation_traces", {"policy": "policy", "physical_slot": 200, "trace_id": "x"})
    artifacts.row("checkpoints", {"path": "best.pt", "type": "best", "actual_physical_slot": 200})
    with pytest.raises(TrainingLogError):
        artifacts.row("rollouts", {"rollout_index": 2, "reward_total": float("inf")})
    artifacts.close()
    for name in (
        "training.jsonl", "train_rollouts.csv", "train_episodes.csv", "train_constraints.csv",
        "train_tenants.csv", "train_communication_users.csv", "validation_summary.csv",
        "validation_traces.csv", "validation_constraints.csv", "validation_tenants.csv",
        "validation_communication_users.csv", "checkpoint_index.csv", "resume_segments.csv",
    ):
        assert (tmp_path / name).is_file()
    with (tmp_path / "train_rollouts.csv").open(encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["reward_total"] == "3.0"
    with pytest.raises(TrainingLogError):
        write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_csv_schema_migration_archives_legacy_without_mixing_semantics(tmp_path) -> None:
    path = tmp_path / "validation_summary.csv"
    path.write_text("physical_slot,policy_mean_return\n200,1.5\n", encoding="utf-8")
    table = CsvTable(path, ("actual_physical_slot", "arrival_regime", "policy_mean_return"), append=True)
    try:
        assert table.migrated_from is not None
        table.write({"actual_physical_slot": 400, "arrival_regime": "overall", "policy_mean_return": 2.0})
    finally:
        table.close()
    legacy = Path(table.migrated_from)
    assert legacy.read_text(encoding="utf-8") == "physical_slot,policy_mean_return\n200,1.5\n"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"actual_physical_slot": "400", "arrival_regime": "overall", "policy_mean_return": "2.0"}]


def test_fresh_run_protects_existing_output_directory(tmp_path) -> None:
    experiment = replace(
        EXP,
        training=replace(EXP.training, seed=123, physical_slots=200, arrival_regimes=("independent",), rollout_target_physical_slots=200),
        validation=replace(EXP.validation, enabled=False),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=0),
        logging=replace(EXP.logging, progress=False),
    )
    target = tmp_path / "run"
    target.mkdir()
    (target / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(TrainerValidationError, match="not empty"):
        JointCreditPPOTrainer(
            ENV, replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1)),
            experiment, output_root=tmp_path, run_name="run",
        )