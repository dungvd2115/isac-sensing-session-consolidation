"""Start, continue or branch one JC-PPO or CT-PPO training run."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from isac_ssc.training.checkpoint import CheckpointMetadata, read_checkpoint_context, semantic_digest
from isac_ssc.training.trainer import CommonTracePPOTrainer, JointCreditPPOTrainer
from isac_ssc.utils.config import (
    DEFAULT_ALGORITHM_CONFIG_PATH, DEFAULT_CONFIG_PATH, DEFAULT_EXPERIMENT_CONFIG_PATH,
    COMMON_TRACE_METHOD, JOINT_CREDIT_METHOD,
    load_algorithm_config, load_config, load_experiment_config,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_argument_group("run")
    run.add_argument("--env-config", "--config", dest="env_config", type=Path)
    run.add_argument("--algorithm-config", type=Path)
    run.add_argument("--experiment-config", type=Path)
    run.add_argument("--seed", type=int)
    run.add_argument("--slots", type=int, help="Minimum fresh-run budget, or minimum additional physical slots with --resume.")
    run.add_argument("--regimes", nargs="+", choices=("independent", "clustered"))
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--threads", type=int)
    run.add_argument("--output-root", type=Path)
    run.add_argument("--run-name")
    run.add_argument("--resume", type=Path)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--progress-every", type=int)

    ppo = parser.add_argument_group("model and PPO")
    ppo.add_argument("--hidden-dim", type=int)
    ppo.add_argument("--profile-embedding-dim", type=int)
    ppo.add_argument("--learning-rate", type=float)
    ppo.add_argument("--optimizer-eps", type=float)
    ppo.add_argument("--lr-schedule", choices=("constant", "linear"))
    ppo.add_argument("--lr-schedule-horizon-slots", type=int)
    ppo.add_argument("--epochs", type=int)
    ppo.add_argument("--minibatch-size", type=int)
    ppo.add_argument("--rollout-slots", type=int)
    ppo.add_argument("--gae-lambda", type=float)
    ppo.add_argument("--clip-ratio", type=float)
    ppo.add_argument("--value-clip-ratio", type=float)
    ppo.add_argument("--entropy-coef", type=float)
    ppo.add_argument("--reward-value-coef", type=float)
    ppo.add_argument("--constraint-value-coef", type=float)
    ppo.add_argument("--target-kl", type=float)
    ppo.add_argument("--max-grad-norm", type=float)

    dual = parser.add_argument_group("dual and normalization")
    dual.add_argument("--dual-learning-rate", type=float)
    dual.add_argument("--dual-maximum", type=float)
    dual.add_argument("--normalizer-clip", type=float)
    dual.add_argument("--normalizer-epsilon", type=float)

    validation = parser.add_argument_group("validation")
    validation.add_argument("--validation-interval", type=int)
    validation.add_argument("--validation-seeds", nargs="+", type=int)
    validation.add_argument("--validation-regimes", nargs="+", choices=("independent", "clustered"))
    validation.add_argument("--random-valid-seed", type=int)
    validation.add_argument("--random-valid-replicates", type=int)
    validation.add_argument("--disable-validation", action="store_true")

    checkpoint = parser.add_argument_group("checkpoint")
    checkpoint.add_argument("--checkpoint-interval", type=int)
    checkpoint.add_argument("--best-metric", choices=("validation_return", "paired_return_difference", "constraint_lexicographic"))
    checkpoint.add_argument("--keep-top-k", type=int)
    checkpoint.add_argument("--disable-latest-checkpoint", action="store_true")
    return parser.parse_args()


def _pick(value, default):
    return default if value is None else value


def _unique(values: tuple, name: str) -> tuple:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _head_slot(directory: Path) -> int:
    slots: list[int] = []

    summary_path = directory / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "completed_physical_slots" in summary:
                slots.append(int(summary["completed_physical_slots"]))
            progress = summary.get("progress", {})
            if isinstance(progress, dict) and "completed_physical_slots" in progress:
                slots.append(int(progress["completed_physical_slots"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    rollout_path = directory / "train_rollouts.csv"
    if rollout_path.is_file():
        try:
            with rollout_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("end_slot"):
                        slots.append(int(row["end_slot"]))
        except (OSError, ValueError, TypeError, csv.Error):
            pass

    jsonl_path = directory / "training.jsonl"
    if jsonl_path.is_file():
        try:
            with jsonl_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        if record.get("event") != "rollout":
                            continue
                        row = record.get("payload", {}).get("row", {})
                        if "end_slot" in row:
                            slots.append(int(row["end_slot"]))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
        except OSError:
            pass

    for candidate in directory.glob("*.pt"):
        try:
            _, state = read_checkpoint_context(candidate)
            slots.append(int(state["progress"]["completed_physical_slots"]))
        except (KeyError, TypeError, ValueError):
            continue

    return max(slots, default=-1)


def _continuation_paths(resume: Path, state: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    inherited = state.get("continuation_config_paths")
    explicit = {
        "environment": args.env_config, "algorithm": args.algorithm_config,
        "experiment": args.experiment_config,
    }
    if inherited is None and any(value is None for value in explicit.values()):
        raise ValueError(
            "this checkpoint predates resolved continuation snapshots; provide --env-config, "
            "--algorithm-config and --experiment-config once when resuming it"
        )
    result = {}
    for name, supplied in explicit.items():
        if supplied is not None:
            result[name] = supplied.resolve()
            continue
        relative = inherited.get(name) if isinstance(inherited, dict) else None
        if not isinstance(relative, str):
            raise ValueError(f"checkpoint is missing its {name} continuation config")
        result[name] = (resume.parent / relative).resolve()
    missing = [name for name, path in result.items() if not path.is_file()]
    if missing:
        raise ValueError("continuation config files are missing: " + ", ".join(missing))
    return result


def _resume_plan(args: argparse.Namespace) -> tuple[CheckpointMetadata | None, dict[str, Any] | None, dict[str, Path], Path, str, bool]:
    if args.resume is None:
        paths = {
            "environment": (args.env_config or DEFAULT_CONFIG_PATH).resolve(),
            "algorithm": (args.algorithm_config or DEFAULT_ALGORITHM_CONFIG_PATH).resolve(),
            "experiment": (args.experiment_config or DEFAULT_EXPERIMENT_CONFIG_PATH).resolve(),
        }
        return None, None, paths, (args.output_root or Path("artifacts/training")), args.run_name or "", False

    resume = args.resume.resolve()
    if args.slots is None:
        raise ValueError("--slots is required with --resume and means minimum additional physical slots")
    metadata, state = read_checkpoint_context(resume)
    progress = state.get("progress", {})
    if not isinstance(progress, dict) or "completed_physical_slots" not in progress:
        raise ValueError("checkpoint does not contain resumable training progress")
    checkpoint_slot = int(progress["completed_physical_slots"])
    source_directory = resume.parent
    head_slot = _head_slot(source_directory)
    requested_root = (args.output_root or source_directory.parent).resolve()
    requested_name = args.run_name or source_directory.name
    requested_directory = (requested_root / requested_name).resolve()
    historical = head_slot >= 0 and checkpoint_slot < head_slot
    branch = requested_directory != source_directory or historical
    if historical and requested_directory == source_directory:
        if args.run_name is not None or args.output_root is not None:
            raise ValueError("a historical checkpoint cannot append to its parent run; choose a new --run-name")
        requested_name = f"{source_directory.name}_branch_{checkpoint_slot:08d}_{resume.stem}"
        requested_root = source_directory.parent
        branch = True
    paths = _continuation_paths(resume, state, args)
    return metadata, state, paths, requested_root, requested_name, branch


def _effective(args: argparse.Namespace, paths: dict[str, Path], resume_metadata: CheckpointMetadata | None):
    environment = load_config(paths["environment"])
    algorithm = load_algorithm_config(paths["algorithm"])
    experiment = load_experiment_config(paths["experiment"])
    if resume_metadata and experiment.method != resume_metadata.method:
        raise ValueError("resume experiment method must match the checkpoint method")
    if resume_metadata and args.seed is not None and args.seed != resume_metadata.training_seed:
        raise ValueError("--seed must match the checkpoint training seed")
    seed = resume_metadata.training_seed if resume_metadata else _pick(args.seed, experiment.training.seed)
    model = replace(
        algorithm.model, hidden_dim=_pick(args.hidden_dim, algorithm.model.hidden_dim),
        profile_embedding_dim=_pick(args.profile_embedding_dim, algorithm.model.profile_embedding_dim),
    )
    optimizer = replace(
        algorithm.optimizer, learning_rate=_pick(args.learning_rate, algorithm.optimizer.learning_rate),
        epsilon=_pick(args.optimizer_eps, algorithm.optimizer.epsilon),
    )
    ppo = replace(
        algorithm.ppo, gae_lambda=_pick(args.gae_lambda, algorithm.ppo.gae_lambda),
        clip_ratio=_pick(args.clip_ratio, algorithm.ppo.clip_ratio),
        value_clip_ratio=_pick(args.value_clip_ratio, algorithm.ppo.value_clip_ratio),
        entropy_coefficient=_pick(args.entropy_coef, algorithm.ppo.entropy_coefficient),
        reward_value_coefficient=_pick(args.reward_value_coef, algorithm.ppo.reward_value_coefficient),
        constraint_value_coefficient=_pick(args.constraint_value_coef, algorithm.ppo.constraint_value_coefficient),
        max_gradient_norm=_pick(args.max_grad_norm, algorithm.ppo.max_gradient_norm),
        epochs_per_rollout=_pick(args.epochs, algorithm.ppo.epochs_per_rollout),
        minibatch_decisions=_pick(args.minibatch_size, algorithm.ppo.minibatch_decisions),
        target_kl=_pick(args.target_kl, algorithm.ppo.target_kl),
    )
    dual = replace(
        algorithm.dual, learning_rate=_pick(args.dual_learning_rate, algorithm.dual.learning_rate),
        maximum=_pick(args.dual_maximum, algorithm.dual.maximum),
    )
    normalization = replace(
        algorithm.normalization, clip=_pick(args.normalizer_clip, algorithm.normalization.clip),
        epsilon=_pick(args.normalizer_epsilon, algorithm.normalization.epsilon),
    )
    algorithm = replace(
        algorithm, device=_pick(args.device, experiment.runtime.device), model=model,
        optimizer=optimizer, ppo=ppo, dual=dual, normalization=normalization,
    )
    runtime = replace(
        experiment.runtime, device=algorithm.device,
        torch_num_threads=_pick(args.threads, experiment.runtime.torch_num_threads),
    )
    training = replace(
        experiment.training, seed=seed, physical_slots=_pick(args.slots, experiment.training.physical_slots),
        arrival_regimes=_unique(tuple(_pick(args.regimes, experiment.training.arrival_regimes)), "training regimes"),
        rollout_target_physical_slots=_pick(args.rollout_slots, experiment.training.rollout_target_physical_slots),
        learning_rate_schedule=_pick(args.lr_schedule, experiment.training.learning_rate_schedule),
        learning_rate_schedule_horizon_physical_slots=_pick(
            args.lr_schedule_horizon_slots,
            experiment.training.learning_rate_schedule_horizon_physical_slots,
        ),
    )
    validation = replace(
        experiment.validation, enabled=False if args.disable_validation else experiment.validation.enabled,
        interval_physical_slots=_pick(args.validation_interval, experiment.validation.interval_physical_slots),
        trace_seeds=_unique(tuple(_pick(args.validation_seeds, experiment.validation.trace_seeds)), "validation seeds"),
        arrival_regimes=_unique(tuple(_pick(args.validation_regimes, experiment.validation.arrival_regimes)), "validation regimes"),
        random_valid_root_seed=_pick(args.random_valid_seed, experiment.validation.random_valid_root_seed),
        random_valid_replicates_per_trace=_pick(args.random_valid_replicates, experiment.validation.random_valid_replicates_per_trace),
    )
    checkpoint = replace(
        experiment.checkpoint, interval_physical_slots=_pick(args.checkpoint_interval, experiment.checkpoint.interval_physical_slots),
        best_metric=_pick(args.best_metric, experiment.checkpoint.best_metric),
        keep_top_k=_pick(args.keep_top_k, experiment.checkpoint.keep_top_k),
        save_latest_every_rollout=False if args.disable_latest_checkpoint else experiment.checkpoint.save_latest_every_rollout,
    )
    logging = replace(
        experiment.logging, progress=False if args.quiet else experiment.logging.progress,
        progress_every_rollouts=_pick(args.progress_every, experiment.logging.progress_every_rollouts),
    )
    experiment = replace(experiment, runtime=runtime, training=training, validation=validation, checkpoint=checkpoint, logging=logging)
    positive = {
        "slots": training.physical_slots, "rollout slots": training.rollout_target_physical_slots,
        "LR schedule horizon slots": training.learning_rate_schedule_horizon_physical_slots,
        "threads": runtime.torch_num_threads, "hidden dim": model.hidden_dim,
        "profile embedding dim": model.profile_embedding_dim, "epochs": ppo.epochs_per_rollout,
        "minibatch size": ppo.minibatch_decisions, "learning rate": optimizer.learning_rate,
        "optimizer epsilon": optimizer.epsilon, "max gradient norm": ppo.max_gradient_norm,
        "target KL": ppo.target_kl, "normalizer clip": normalization.clip,
        "normalizer epsilon": normalization.epsilon, "dual maximum": dual.maximum,
        "keep top-k": checkpoint.keep_top_k, "progress interval": logging.progress_every_rollouts,
        "random-valid replicates": validation.random_valid_replicates_per_trace,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("positive training settings must be greater than zero")
    if validation.interval_physical_slots < 0 or checkpoint.interval_physical_slots < 0:
        raise ValueError("validation and checkpoint intervals must be non-negative")
    if validation.enabled and (not validation.trace_seeds or not validation.arrival_regimes):
        raise ValueError("enabled validation requires non-empty regimes and seeds")
    if any(value < 0 or value > 2**64 - 1 for value in (seed, validation.random_valid_root_seed, *validation.trace_seeds)):
        raise ValueError("all seeds must lie in the unsigned 64-bit range")
    if not 0.0 <= ppo.gae_lambda <= 1.0 or not 0.0 < ppo.clip_ratio <= 1.0 or not 0.0 < ppo.value_clip_ratio <= 1.0:
        raise ValueError("GAE and PPO clip settings are outside their valid ranges")
    if any(value < 0.0 for value in (ppo.entropy_coefficient, ppo.reward_value_coefficient, ppo.constraint_value_coefficient)):
        raise ValueError("loss coefficients must be non-negative")
    if dual.learning_rate <= 0.0 or dual.maximum < dual.initial_value:
        raise ValueError("dual settings are outside their valid ranges")
    return environment, algorithm, experiment


def main() -> None:
    args = _arguments()
    metadata, state, paths, output_root, run_name, branch = _resume_plan(args)
    environment, algorithm, experiment = _effective(args, paths, metadata)
    if not run_name:
        run_name = f"seed_{experiment.training.seed}"
    validation_digest = semantic_digest({
        "trace_seeds": experiment.validation.trace_seeds,
        "arrival_regimes": experiment.validation.arrival_regimes,
        "random_valid_root_seed": experiment.validation.random_valid_root_seed,
        "random_valid_replicates_per_trace": experiment.validation.random_valid_replicates_per_trace,
    })
    if metadata is not None and metadata.validation_protocol_digest != validation_digest and not branch:
        checkpoint_slot = int(state["progress"]["completed_physical_slots"])
        output_root = args.resume.resolve().parent.parent
        run_name = f"{args.resume.resolve().parent.name}_validation_{checkpoint_slot:08d}_{validation_digest[:8]}"
        branch = True
    optimizer_overrides = tuple(
        name for name, enabled in (
            ("learning_rate", args.learning_rate is not None or args.algorithm_config is not None),
            ("epsilon", args.optimizer_eps is not None or args.algorithm_config is not None),
        ) if enabled
    )
    trainers = {
        JOINT_CREDIT_METHOD: JointCreditPPOTrainer,
        COMMON_TRACE_METHOD: CommonTracePPOTrainer,
    }
    trainer = trainers[experiment.method]
    summary = trainer(
        environment, algorithm, experiment, output_root=output_root, run_name=run_name,
        resume=args.resume, branch_resume=branch, config_sources=paths,
        optimizer_overrides=optimizer_overrides, overwrite=args.overwrite, quiet=args.quiet,
    ).run()
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()