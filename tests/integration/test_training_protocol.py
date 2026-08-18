import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from isac_ssc.training.trainer import (
    BestCheckpointRecord, CommonTracePPOTrainer, JointCreditPPOTrainer,
    TrainerValidationError, TrainingSegmentRecord,
)
from isac_ssc.utils.config import (
    COMMON_TRACE_METHOD, credit_assignment_schema,
    load_algorithm_config, load_config, load_experiment_config,
)

ENV = load_config()
ALG = load_algorithm_config()
EXP = load_experiment_config()
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("isac_ssc_train_script", ROOT / "scripts" / "train.py")
assert SPEC is not None and SPEC.loader is not None
TRAIN_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SCRIPT)


def _quick_experiment(seed: int, slots: int):
    return replace(
        EXP,
        training=replace(
            EXP.training, seed=seed, physical_slots=slots,
            arrival_regimes=("independent",), rollout_target_physical_slots=200,
        ),
        validation=replace(
            EXP.validation, enabled=True, interval_physical_slots=200,
            trace_seeds=(51001,), arrival_regimes=("independent",),
            random_valid_replicates_per_trace=1,
        ),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=200, keep_top_k=1),
        logging=replace(EXP.logging, progress=False, flush_every_records=1),
    )


def _cli(monkeypatch, capsys, *arguments: str):
    monkeypatch.setattr(sys, "argv", ["train.py", *arguments])
    TRAIN_SCRIPT.main()
    output = capsys.readouterr().out.strip().splitlines()
    return json.loads(output[-1])


def _restored_state(slot: int = 200) -> dict:
    return {
        "progress": {
            "completed_physical_slots": slot, "completed_episodes": 1,
            "focal_decisions": 1, "valid_actions": 1, "invalid_actions": 0,
            "rollout_index": 1, "next_episode_index": 1,
            "next_validation_boundary": 0, "next_checkpoint_boundary": 0,
        },
        "validations": [], "best_checkpoints": [], "segments": [],
        "last_finite_checkpoint": "checkpoint.pt", "random_valid": None,
        "cumulative_elapsed_seconds": 0.0, "active_segment": None,
    }


def test_regime_balanced_ranking_uses_worst_macro_constraint_then_earlier_slot() -> None:
    trainer = object.__new__(JointCreditPPOTrainer)
    candidates = (
        BestCheckpointRecord("a.pt", 100, "paired_return_difference", 0.1, 0.9, 0.0),
        BestCheckpointRecord("b.pt", 400, "paired_return_difference", 0.2, 0.0, 0.9),
        BestCheckpointRecord("c.pt", 500, "paired_return_difference", 0.2, 0.1, 0.9),
        BestCheckpointRecord("d.pt", 600, "paired_return_difference", 0.2, 0.1, 0.1),
        BestCheckpointRecord("e.pt", 300, "paired_return_difference", 0.2, 0.1, 0.1),
    )
    ranked = sorted(candidates, key=trainer._best_key, reverse=True)
    assert tuple(Path(item.path).name for item in ranked) == ("a.pt", "e.pt", "d.pt", "c.pt", "b.pt")


def test_head_slot_uses_newest_persisted_progress(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"completed_physical_slots": 400}), encoding="utf-8",
    )
    (run / "train_rollouts.csv").write_text(
        "rollout_index,end_slot\n1,800\n", encoding="utf-8",
    )
    (run / "training.jsonl").write_text(
        json.dumps({"event": "rollout", "payload": {"row": {"end_slot": 900}}}) + "\n",
        encoding="utf-8",
    )
    for name in ("latest.pt", "final.pt", "recovery_00000700.pt"):
        (run / name).touch()
    checkpoint_slots = {
        "latest.pt": 600,
        "final.pt": 500,
        "recovery_00000700.pt": 700,
    }

    def checkpoint_context(path):
        slot = checkpoint_slots[Path(path).name]
        return None, {"progress": {"completed_physical_slots": slot}}

    monkeypatch.setattr(TRAIN_SCRIPT, "read_checkpoint_context", checkpoint_context)
    assert TRAIN_SCRIPT._head_slot(run) == 900


def test_legacy_v3_metadata_detects_changed_validation_protocol(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "parent" / "legacy.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    experiment = replace(
        _quick_experiment(126, 200),
        validation=replace(EXP.validation, enabled=False),
        logging=replace(EXP.logging, progress=False),
    )

    def restore_legacy(path, agent, expected):
        legacy = replace(expected, validation_protocol_digest="legacy-validation-protocol")
        return legacy, _restored_state()

    monkeypatch.setattr("isac_ssc.training.trainer.load_checkpoint", restore_legacy)
    trainer = JointCreditPPOTrainer(
        ENV, replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1)),
        experiment, output_root=tmp_path, run_name="legacy_branch",
        resume=checkpoint, branch_resume=True,
    )
    try:
        assert trainer._validation_reset
        assert trainer.random_valid is None
    finally:
        trainer.artifacts.close()


def test_resume_projects_duals_to_new_maximum(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "parent" / "dual.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    maximum = 0.25
    algorithm = replace(
        ALG, dual=replace(ALG.dual, maximum=maximum),
        ppo=replace(ALG.ppo, epochs_per_rollout=1),
    )
    experiment = replace(
        _quick_experiment(127, 200),
        validation=replace(EXP.validation, enabled=False),
        logging=replace(EXP.logging, progress=False),
    )

    def restore_large_duals(path, agent, expected):
        agent.algorithm.dual_values.fill_(5.0)
        return expected, _restored_state()

    monkeypatch.setattr("isac_ssc.training.trainer.load_checkpoint", restore_large_duals)
    trainer = JointCreditPPOTrainer(
        ENV, algorithm, experiment, output_root=tmp_path, run_name="dual_branch",
        resume=checkpoint, branch_resume=True,
    )
    try:
        assert float(trainer.agent.algorithm.dual_values.min()) >= 0.0
        assert float(trainer.agent.algorithm.dual_values.max()) <= maximum
    finally:
        trainer.artifacts.close()


def test_cli_is_flexible_and_resume_slots_are_explicit(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        TRAIN_SCRIPT._arguments()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--slots" in help_text and "minimum additional" in help_text and "physical slots with --resume" in help_text
    assert "--regimes" in help_text and "--learning-rate" in help_text
    assert "--experiment-config" in help_text and "--env-config" in help_text
    assert "source-archive" not in help_text
    summary = _cli(
        monkeypatch, capsys, "--seed", "777", "--slots", "200", "--regimes", "clustered",
        "--rollout-slots", "200", "--epochs", "1", "--minibatch-size", "256",
        "--disable-validation", "--checkpoint-interval", "0", "--output-root", str(tmp_path),
        "--run-name", "cli", "--quiet",
    )
    assert summary["training_seed"] == 777
    assert summary["requested_physical_slots"] == summary["actual_physical_slots"] == 200
    assert summary["segment_start_physical_slot"] == 0
    assert summary["segment_target_physical_slot"] == 200
    monkeypatch.setattr(sys, "argv", ["train.py", "--resume", str(tmp_path / "cli" / "latest.pt"), "--quiet"])
    with pytest.raises(ValueError, match="--slots is required with --resume"):
        TRAIN_SCRIPT.main()


def test_resume_inherits_resolved_custom_configuration(tmp_path, monkeypatch, capsys) -> None:
    first_summary = _cli(
        monkeypatch, capsys, "--seed", "778", "--slots", "200", "--regimes", "clustered", "--hidden-dim", "96",
        "--learning-rate", "0.0001", "--dual-learning-rate", "0.002", "--lr-schedule", "linear",
        "--rollout-slots", "200", "--epochs", "1", "--minibatch-size", "256", "--disable-validation",
        "--checkpoint-interval", "0", "--output-root", str(tmp_path), "--run-name", "inherit", "--quiet",
    )
    first = yaml.safe_load((tmp_path / "inherit" / "segment_0000_algorithm.yaml").read_text(encoding="utf-8"))
    second_summary = _cli(monkeypatch, capsys, "--resume", str(tmp_path / "inherit" / "latest.pt"), "--slots", "200", "--quiet")
    second_algorithm = yaml.safe_load((tmp_path / "inherit" / "segment_0001_algorithm.yaml").read_text(encoding="utf-8"))
    second_experiment = yaml.safe_load((tmp_path / "inherit" / "segment_0001_experiment.yaml").read_text(encoding="utf-8"))
    assert second_algorithm["model"]["hidden_dim"] == first["model"]["hidden_dim"] == 96
    assert second_algorithm["optimizer"]["learning_rate"] == first["optimizer"]["learning_rate"] == pytest.approx(1e-4)
    assert second_algorithm["dual"]["learning_rate"] == first["dual"]["learning_rate"] == pytest.approx(0.002)
    assert second_experiment["training"]["arrival_regimes"] == ["clustered"]
    assert second_experiment["training"]["learning_rate_schedule"] == "linear"
    assert second_experiment["training"]["rollout_target_physical_slots"] == 200
    assert second_summary["segments"][-1]["starting_learning_rate"] <= first_summary["segments"][-1]["ending_learning_rate"] + 1e-15


def test_non_branch_resume_rejects_changed_linear_global_horizon(tmp_path, monkeypatch) -> None:
    run = tmp_path / "horizon"
    run.mkdir()
    checkpoint = run / "latest.pt"
    checkpoint.touch()
    previous = TrainingSegmentRecord(
        0, None, 0, 200, 200, 200, 200, 0, ("independent",), 200,
        "linear", 500000, "algorithm", "validation", False, 0, 0,
        0.0, 1.0, 1.0, 3.0e-4, 1.8e-4, "completed",
    )

    def restore_with_previous_horizon(self, path):
        self.segments = [previous]

    monkeypatch.setattr(JointCreditPPOTrainer, "_restore", restore_with_previous_horizon)
    experiment = replace(
        _quick_experiment(51003, 200),
        training=replace(
            _quick_experiment(51003, 200).training,
            learning_rate_schedule="linear",
            learning_rate_schedule_horizon_physical_slots=1000000,
        ),
        validation=replace(EXP.validation, enabled=False),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=0),
    )
    with pytest.raises(TrainerValidationError, match="preserve the linear learning-rate horizon"):
        JointCreditPPOTrainer(
            ENV, replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1)),
            experiment, output_root=tmp_path, run_name="horizon", resume=checkpoint,
        )


def test_historical_checkpoint_automatically_creates_branch_run(tmp_path, monkeypatch, capsys) -> None:
    _cli(
        monkeypatch, capsys, "--seed", "779", "--slots", "400", "--regimes", "independent", "--rollout-slots", "200",
        "--epochs", "1", "--minibatch-size", "256", "--disable-validation", "--checkpoint-interval", "200",
        "--output-root", str(tmp_path), "--run-name", "parent", "--quiet",
    )
    parent_summary_before = hashlib.sha256((tmp_path / "parent" / "summary.json").read_bytes()).hexdigest()
    summary = _cli(monkeypatch, capsys, "--resume", str(tmp_path / "parent" / "recovery_00000200.pt"), "--slots", "200", "--quiet")
    branch = tmp_path / summary["run_name"]
    assert summary["run_name"].startswith("parent_branch_00000200_")
    assert summary["parent_run_name"] == "parent"
    assert summary["parent_checkpoint_path"].endswith("recovery_00000200.pt")
    assert summary["segment_start_physical_slot"] == 200
    assert summary["completed_physical_slots"] == 400
    assert branch.is_dir()
    assert hashlib.sha256((tmp_path / "parent" / "summary.json").read_bytes()).hexdigest() == parent_summary_before


def test_resume_trains_additional_slots_and_preserves_origin_files(tmp_path, monkeypatch) -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    first = JointCreditPPOTrainer(
        ENV, algorithm, _quick_experiment(51001, 200), output_root=tmp_path, run_name="resume",
    ).run()
    run = tmp_path / "resume"
    for name in ("best.pt", "latest.pt", "final.pt"):
        assert (run / name).is_file()
    manifest_before = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
    effective_before = hashlib.sha256((run / "effective_config.json").read_bytes()).hexdigest()
    first_history = len(first.validations)

    def should_not_recompute(*args, **kwargs):
        raise AssertionError("random-valid cache was recomputed during resume")

    monkeypatch.setattr("isac_ssc.training.trainer.evaluate_random_valid", should_not_recompute)
    second = JointCreditPPOTrainer(
        ENV, algorithm, _quick_experiment(51001, 200), output_root=tmp_path, run_name="resume",
        resume=run / "latest.pt",
    ).run()
    assert second.requested_physical_slots == second.actual_physical_slots == 200
    assert second.segment_start_physical_slot == 200
    assert second.segment_target_physical_slot == second.completed_physical_slots == 400
    assert len(second.validations) > first_history
    assert second.validations[:first_history] == first.validations
    assert second.cumulative_elapsed_seconds >= second.segment_elapsed_seconds > 0.0
    assert len(second.segments) >= 2
    with (run / "resume_segments.csv").open(encoding="utf-8") as handle:
        segment_rows = list(csv.DictReader(handle))
    assert len(segment_rows) >= 2
    assert {row["status"] for row in segment_rows} == {"completed"}
    assert hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest() == manifest_before
    assert hashlib.sha256((run / "effective_config.json").read_bytes()).hexdigest() == effective_before


def test_interrupted_active_segment_is_persisted_when_resume_continues(tmp_path) -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    experiment = replace(
        _quick_experiment(51002, 200), validation=replace(EXP.validation, enabled=False),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=0),
    )
    interrupted = JointCreditPPOTrainer(
        ENV, algorithm, experiment, output_root=tmp_path, run_name="interrupted",
    )
    try:
        interrupted._save_checkpoint("latest.pt", "latest")
    finally:
        interrupted.artifacts.close()
    resumed = JointCreditPPOTrainer(
        ENV, algorithm, experiment, output_root=tmp_path, run_name="interrupted",
        resume=tmp_path / "interrupted" / "latest.pt",
    ).run()
    with (tmp_path / "interrupted" / "resume_segments.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["checkpointed", "completed"]
    assert resumed.segments[0].status == "checkpointed"
    assert resumed.segments[-1].status == "completed"


def test_non_aligned_budget_finishes_at_complete_episode_boundary(tmp_path) -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    experiment = replace(
        _quick_experiment(51100, 250), validation=replace(EXP.validation, enabled=False),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=0),
    )
    summary = JointCreditPPOTrainer(ENV, algorithm, experiment, output_root=tmp_path, run_name="overshoot").run()
    assert summary.requested_physical_slots == 250
    assert summary.actual_physical_slots == 400
    assert summary.budget_overshoot_slots == 150
    assert summary.segment_target_physical_slot == 250
    assert summary.completed_physical_slots == 400


def test_changed_validation_protocol_creates_new_branch_series(tmp_path, monkeypatch, capsys) -> None:
    _cli(
        monkeypatch, capsys, "--seed", "780", "--slots", "200", "--regimes", "independent",
        "--rollout-slots", "200", "--epochs", "1", "--minibatch-size", "256",
        "--validation-seeds", "51001", "--validation-regimes", "independent",
        "--random-valid-replicates", "1", "--validation-interval", "200",
        "--checkpoint-interval", "200", "--output-root", str(tmp_path), "--run-name", "validation", "--quiet",
    )
    summary = _cli(
        monkeypatch, capsys, "--resume", str(tmp_path / "validation" / "latest.pt"), "--slots", "200",
        "--validation-seeds", "51002", "--quiet",
    )
    assert summary["run_name"].startswith("validation_validation_00000200_")
    assert summary["parent_run_name"] == "validation"
    assert summary["segment_start_physical_slot"] == 200
    assert summary["completed_physical_slots"] == 400
    assert len(summary["validations"]) >= 1


def test_changed_checkpoint_selection_resets_candidates_without_blocking_resume(tmp_path) -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    first = JointCreditPPOTrainer(ENV, algorithm, _quick_experiment(124, 200), output_root=tmp_path, run_name="selection").run()
    changed = replace(
        _quick_experiment(124, 200),
        checkpoint=replace(_quick_experiment(124, 200).checkpoint, best_metric="validation_return"),
    )
    second = JointCreditPPOTrainer(
        ENV, algorithm, changed, output_root=tmp_path, run_name="selection",
        resume=tmp_path / "selection" / "latest.pt",
    ).run()
    assert first.best_checkpoint_metric == "paired_return_difference"
    assert second.best_checkpoint_metric == "validation_return"
    assert second.best_checkpoint_slot is not None and second.best_checkpoint_slot >= 400


def test_disable_latest_reports_no_current_segment_latest(tmp_path) -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    first = JointCreditPPOTrainer(
        ENV, algorithm, replace(_quick_experiment(125, 200), validation=replace(EXP.validation, enabled=False)),
        output_root=tmp_path, run_name="latest",
    ).run()
    experiment = replace(
        _quick_experiment(125, 200), validation=replace(EXP.validation, enabled=False),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=0, save_latest_every_rollout=False),
    )
    second = JointCreditPPOTrainer(
        ENV, algorithm, experiment, output_root=tmp_path, run_name="latest",
        resume=tmp_path / "latest" / "latest.pt",
    ).run()
    assert first.latest_checkpoint_path is not None
    assert second.latest_checkpoint_path is None
    assert second.last_finite_checkpoint_path == second.final_checkpoint_path


def test_last_finite_checkpoint_is_not_overwritten_by_nonfinite_state(tmp_path) -> None:
    algorithm = replace(ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256))
    trainer = JointCreditPPOTrainer(
        ENV, algorithm,
        replace(
            _quick_experiment(123, 200), validation=replace(EXP.validation, enabled=False),
            checkpoint=replace(EXP.checkpoint, interval_physical_slots=0),
        ),
        output_root=tmp_path, run_name="finite",
    )
    trainer.run()
    latest = tmp_path / "finite" / "latest.pt"
    before = hashlib.sha256(latest.read_bytes()).hexdigest()
    next(trainer.agent.model.parameters()).data.fill_(float("nan"))
    with pytest.raises(Exception, match="non-finite"):
        trainer._save_checkpoint("latest.pt", "latest")
    assert hashlib.sha256(latest.read_bytes()).hexdigest() == before


def test_common_trace_trainer_resume_matches_uninterrupted_run(tmp_path) -> None:
    algorithm = replace(
        ALG, ppo=replace(ALG.ppo, epochs_per_rollout=1, minibatch_decisions=256),
    )
    base = replace(
        _quick_experiment(901, 400), method=COMMON_TRACE_METHOD,
        validation=replace(EXP.validation, enabled=False),
        checkpoint=replace(EXP.checkpoint, interval_physical_slots=0),
    )
    full = CommonTracePPOTrainer(
        ENV, algorithm, base, output_root=tmp_path, run_name="ct_full",
    ).run()
    first = CommonTracePPOTrainer(
        ENV, algorithm, replace(base, training=replace(base.training, physical_slots=200)),
        output_root=tmp_path, run_name="ct_split",
    ).run()
    resumed = CommonTracePPOTrainer(
        ENV, algorithm, replace(base, training=replace(base.training, physical_slots=200)),
        output_root=tmp_path, run_name="ct_split",
        resume=tmp_path / "ct_split" / "latest.pt",
    ).run()
    assert first.completed_physical_slots == 200
    assert full.completed_physical_slots == resumed.completed_physical_slots == 400
    assert resumed.credit_assignment_schema == credit_assignment_schema(COMMON_TRACE_METHOD)
    left = torch.load(full.final_checkpoint_path, map_location="cpu", weights_only=True)
    right = torch.load(resumed.final_checkpoint_path, map_location="cpu", weights_only=True)

    def equal(first_value, second_value):
        if isinstance(first_value, torch.Tensor):
            return torch.equal(first_value, second_value)
        if isinstance(first_value, dict):
            return first_value.keys() == second_value.keys() and all(
                equal(first_value[key], second_value[key]) for key in first_value
            )
        if isinstance(first_value, (tuple, list)):
            return len(first_value) == len(second_value) and all(
                equal(left_value, right_value)
                for left_value, right_value in zip(first_value, second_value, strict=True)
            )
        return first_value == second_value

    for key in (
        "metadata", "model_state", "optimizer_state", "dual_values",
        "normalizer_state", "normalizer_configuration", "torch_rng_state",
        "action_generator_state", "minibatch_generator_state", "optimizer_step_count",
    ):
        assert equal(left[key], right[key])