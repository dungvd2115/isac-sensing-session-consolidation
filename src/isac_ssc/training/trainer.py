"""Single-seed trainer with regime-separated validation and report-ready artifacts."""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import torch
import yaml

from isac_ssc.algorithms.buffers import ConstraintLayout
from isac_ssc.baselines.ppo_common_trace import build_common_trace_agent
from isac_ssc.baselines.ppo_joint_credit import build_joint_credit_agent
from isac_ssc.envs.action_space import ActionType, identifier_key
from isac_ssc.envs.dynamics import PrimitiveTrace, generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import ISACSSCEnv
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint, semantic_digest
from isac_ssc.training.logging import RunManifest, TrainingArtifacts, write_json, write_run_summary
from isac_ssc.training.rollout import (
    CollectedRollout, CommunicationUserMetrics, EpisodeCollectionMetrics, TenantMetrics,
    ValidationEpisode, ValidationReport, ValidationSummary,
    collect_common_trace_training_rollout, collect_training_rollout,
    evaluate_policy, evaluate_random_valid,
)
from isac_ssc.utils.config import (
    DEFAULT_CONFIG_PATH, DEFAULT_EXPERIMENT_CONFIG_PATH,
    COMMON_TRACE_METHOD, JOINT_CREDIT_METHOD, CanonicalConfig, ConstrainedPPOConfig,
    TrainingExperimentConfig, credit_assignment_schema,
)
from isac_ssc.utils.seeding import SeedContract


class TrainerValidationError(ValueError):
    """Raised when a run setting or restored state is not executable."""


_ARTIFACT_SCHEMA = "isac-ssc-training-artifacts-v5"
_VALIDATION_RESULT_SCHEMA = "isac-ssc-validation-report-v2"
_SELECTION_RULE_VERSION = "macro-primary-v2"
_TRAINING_SEED_DOMAIN = "constrained_ppo_v1"


@dataclass(slots=True)
class TrainingProgress:
    completed_physical_slots: int = 0
    completed_episodes: int = 0
    focal_decisions: int = 0
    valid_actions: int = 0
    invalid_actions: int = 0
    rollout_index: int = 0
    next_episode_index: int = 0
    next_validation_boundary: int = 0
    next_checkpoint_boundary: int = 0

    @property
    def valid_action_rate(self) -> float:
        total = self.valid_actions + self.invalid_actions
        return 1.0 if total == 0 else self.valid_actions / total


@dataclass(frozen=True, slots=True)
class ValidationRegimeRecord:
    scheduled_physical_slot: int
    actual_physical_slot: int
    interval_overshoot_slots: int
    arrival_regime: str
    is_overall: bool
    is_worst_regime: bool
    policy_episode_count: int
    random_valid_episode_count: int
    policy_mean_return: float
    policy_std_return: float | None
    random_valid_mean_return: float
    random_valid_std_return: float | None
    paired_return_difference: float
    policy_mean_completed_value: float
    policy_mean_normalized_completed_value: float | None
    policy_mean_sensing_resource_cost: float
    policy_mean_positive_constraint_excess: float
    policy_mean_reward_per_slot: float
    policy_mean_completed_value_per_slot: float
    policy_mean_sensing_resource_cost_per_slot: float
    policy_mean_network_user_shortfall: float | None
    policy_mean_fraction_users_within_budget: float | None
    random_valid_mean_reward_per_slot: float
    random_valid_mean_normalized_completed_value: float | None
    random_valid_mean_network_user_shortfall: float | None
    random_valid_mean_fraction_users_within_budget: float | None
    policy_valid_action_rate: float
    random_valid_valid_action_rate: float
    best_metric: str
    best_score: float
    is_best: bool


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    scheduled_physical_slot: int
    actual_physical_slot: int
    interval_overshoot_slots: int
    regimes: tuple[ValidationRegimeRecord, ...]
    worst_regime: str
    worst_regime_paired_return: float
    macro_paired_return: float
    worst_regime_policy_return: float
    macro_policy_return: float
    macro_positive_constraint_excess: float
    best_metric: str
    best_score: float
    is_best: bool

    @property
    def physical_slots(self) -> int:
        return self.actual_physical_slot


@dataclass(frozen=True, slots=True)
class BestCheckpointRecord:
    path: str
    physical_slots: int
    metric: str
    worst_regime_score: float
    macro_score: float
    constraint_excess: float


@dataclass(frozen=True, slots=True)
class TrainingSegmentRecord:
    segment_index: int
    resume_checkpoint: str | None
    start_physical_slot: int
    requested_physical_slots: int
    target_physical_slot: int
    completed_physical_slot: int
    actual_physical_slots: int
    budget_overshoot_slots: int
    training_regimes: tuple[str, ...]
    rollout_target_physical_slots: int
    learning_rate_schedule: str
    learning_rate_schedule_horizon_physical_slots: int
    algorithm_semantic_digest: str
    validation_protocol_digest: str
    validation_enabled: bool
    validation_interval_physical_slots: int
    checkpoint_interval_physical_slots: int
    started_unix_s: float
    finished_unix_s: float
    elapsed_seconds: float
    starting_learning_rate: float
    ending_learning_rate: float
    status: str


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    schema_version: str
    artifact_schema_version: str
    method: str
    credit_assignment_schema: str
    training_seed: int
    run_name: str
    parent_run_name: str | None
    parent_checkpoint_path: str | None
    requested_physical_slots: int
    actual_physical_slots: int
    budget_overshoot_slots: int
    segment_start_physical_slot: int
    segment_target_physical_slot: int
    completed_physical_slots: int
    completed_episodes: int
    focal_decisions: int
    valid_action_rate: float
    best_checkpoint_path: str | None
    best_checkpoint_slot: int | None
    best_checkpoint_score: float | None
    best_checkpoint_worst_regime_score: float | None
    best_checkpoint_constraint_excess: float | None
    best_checkpoint_metric: str
    final_checkpoint_path: str
    latest_checkpoint_path: str | None
    last_finite_checkpoint_path: str | None
    final_dual_values: tuple[float, ...]
    validations: tuple[ValidationRecord, ...]
    segments: tuple[TrainingSegmentRecord, ...]
    artifact_paths: dict[str, str]
    segment_elapsed_seconds: float
    cumulative_elapsed_seconds: float
    all_finite: bool
    failure: str | None = None


def _next_boundary(current: int, interval: int) -> int:
    return 0 if interval <= 0 else ((current // interval) + 1) * interval


def _first_observation(config: CanonicalConfig, trace: PrimitiveTrace):
    env = ISACSSCEnv(config)
    observation = env.reset(trace)
    while observation is None and not env.terminated:
        observation = env.step(None).next_observation
    if observation is None:
        raise TrainerValidationError("the selected training trace has no focal observation")
    return observation


def _constraint_label(prefix: str, value: object) -> str:
    kind, text = identifier_key(value)
    return f"{prefix}:{'int' if kind == 0 else 'str'}:{text}"


def _constraint_identity(label: str) -> tuple[str, str]:
    family, _, entity = label.partition(":")
    return family, entity


def _environment_semantics(config: CanonicalConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version, "profile_name": config.profile_name,
        "units": config.units, "system": config.system, "geometry": config.geometry,
        "population": config.population, "tenants": config.tenants, "mobility": config.mobility,
        "communication": config.communication, "sensing": config.sensing,
        "resource_profiles": config.resource_profiles, "requests": config.requests,
        "arrivals": config.arrivals, "sharing_authorization": config.sharing_authorization,
        "target_compatibility": config.target_compatibility, "compatibility": config.compatibility,
        "sla": config.sla, "reward": config.reward, "observation": config.observation,
        "trace_generation": config.trace_generation,
    }


def _tenant_metrics(values: dict[str, Any]) -> TenantMetrics:
    return TenantMetrics(**values)


def _communication_metrics(values: dict[str, Any]) -> CommunicationUserMetrics:
    return CommunicationUserMetrics(**values)


def _episode_metrics(values: dict[str, Any]) -> EpisodeCollectionMetrics:
    data = dict(values)
    data["tenant_residual_totals"] = tuple(data["tenant_residual_totals"])
    data["communication_residual_totals"] = tuple(data["communication_residual_totals"])
    data["tenants"] = tuple(_tenant_metrics(item) for item in data["tenants"])
    data["communication_users"] = tuple(_communication_metrics(item) for item in data["communication_users"])
    data["action_counts"] = tuple(tuple(item) for item in data["action_counts"])
    return EpisodeCollectionMetrics(**data)


def _validation_episode(values: dict[str, Any]) -> ValidationEpisode:
    data = dict(values)
    data["metrics"] = _episode_metrics(data["metrics"])
    return ValidationEpisode(**data)


def _validation_summary(values: dict[str, Any]) -> ValidationSummary:
    return ValidationSummary(**values)


def _validation_report(values: dict[str, Any] | None) -> ValidationReport | None:
    if values is None:
        return None
    data = dict(values)
    data["episodes"] = tuple(_validation_episode(item) for item in data["episodes"])
    data["regimes"] = tuple(_validation_summary(item) for item in data["regimes"])
    data["overall"] = _validation_summary(data["overall"])
    return ValidationReport(**data)


def _validation_regime_record(values: dict[str, Any]) -> ValidationRegimeRecord:
    return ValidationRegimeRecord(**values)


def _validation_record(values: dict[str, Any]) -> ValidationRecord:
    data = dict(values)
    data["regimes"] = tuple(_validation_regime_record(item) for item in data["regimes"])
    return ValidationRecord(**data)


def _segment_record(values: dict[str, Any]) -> TrainingSegmentRecord:
    data = dict(values)
    start = int(data["start_physical_slot"])
    completed = int(data["completed_physical_slot"])
    target = int(data["target_physical_slot"])
    data.setdefault("actual_physical_slots", completed - start)
    data.setdefault("budget_overshoot_slots", max(0, completed - target))
    data.setdefault("learning_rate_schedule_horizon_physical_slots", max(target, 1))
    data["training_regimes"] = tuple(data["training_regimes"])
    data.pop("parent_run_name", None)
    data.pop("parent_checkpoint_path", None)
    return TrainingSegmentRecord(**data)


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _algorithm_snapshot(config: ConstrainedPPOConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version, "runtime": {"device": config.device, "dtype": config.dtype},
        "model": asdict(config.model), "optimizer": asdict(config.optimizer), "ppo": asdict(config.ppo),
        "dual": asdict(config.dual), "normalization": asdict(config.normalization),
    }


def _experiment_snapshot(config: TrainingExperimentConfig) -> dict[str, Any]:
    return _plain(asdict(config))


def _action_counts(values: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(values)


def _finite_mean(values: tuple[float, ...]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _artifact_paths(directory: Path) -> dict[str, str]:
    names = (
        "train_rollouts.csv", "train_episodes.csv", "train_constraints.csv", "train_tenants.csv",
        "train_communication_users.csv", "validation_summary.csv", "validation_traces.csv",
        "validation_constraints.csv", "validation_tenants.csv", "validation_communication_users.csv",
        "checkpoint_index.csv", "resume_segments.csv", "training.jsonl", "manifest.json", "effective_config.json",
    )
    result = {name: (directory / name).as_posix() for name in names if (directory / name).exists()}
    for pattern in ("segment_*", "*.legacy*.csv"):
        for path in sorted(directory.glob(pattern)):
            if path.is_file():
                result[path.name] = path.as_posix()
    superseded = directory / "superseded_best"
    if superseded.is_dir():
        result["superseded_best"] = superseded.as_posix()
    result["summary.json"] = (directory / "summary.json").as_posix()
    return result


class JointCreditPPOTrainer:
    """Train one method-specific PPO policy and preserve report-ready run state."""

    expected_method = JOINT_CREDIT_METHOD

    def __init__(
        self, environment: CanonicalConfig, algorithm: ConstrainedPPOConfig,
        experiment: TrainingExperimentConfig, *, output_root: str | Path,
        run_name: str | None = None, resume: str | Path | None = None,
        branch_resume: bool = False, config_sources: dict[str, Path] | None = None,
        optimizer_overrides: tuple[str, ...] = (), overwrite: bool = False, quiet: bool = False,
    ) -> None:
        self.environment, self.algorithm_config, self.experiment = environment, algorithm, experiment
        if experiment.method != self.expected_method:
            raise TrainerValidationError(
                f"{type(self).__name__} requires method {self.expected_method!r}"
            )
        self.credit_assignment_schema = credit_assignment_schema(experiment.method)
        self.training_seed = experiment.training.seed
        if self.training_seed < 0 or self.training_seed > 2**64 - 1:
            raise TrainerValidationError("training seed must lie in the unsigned 64-bit range")
        if experiment.training.physical_slots < 1 or experiment.training.rollout_target_physical_slots < 1:
            raise TrainerValidationError("training slot budgets must be positive")
        if not experiment.training.arrival_regimes:
            raise TrainerValidationError("at least one training regime is required")
        if algorithm.device == "cuda" and not torch.cuda.is_available():
            raise TrainerValidationError("CUDA was requested but is unavailable")

        self.run_name = run_name or f"seed_{self.training_seed}"
        self.output_directory = (Path(output_root) / self.run_name).resolve()
        self.resume_path = Path(resume).resolve() if resume is not None else None
        self.branch_resume = bool(branch_resume and self.resume_path is not None)
        if self.resume_path is not None and not self.resume_path.is_file():
            raise TrainerValidationError("resume checkpoint does not exist")
        if self.resume_path is not None and not self.branch_resume and self.output_directory != self.resume_path.parent:
            raise TrainerValidationError("in-place resume must use the checkpoint run directory")
        if self.output_directory.exists() and any(self.output_directory.iterdir()) and (self.resume_path is None or self.branch_resume):
            if not overwrite:
                raise TrainerValidationError("output directory is not empty; use --overwrite or a new run name")
            shutil.rmtree(self.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.parent_run_name = self.resume_path.parent.name if self.branch_resume else None
        self.parent_checkpoint_path = self.resume_path.as_posix() if self.branch_resume else None

        torch.use_deterministic_algorithms(experiment.runtime.deterministic_algorithms)
        torch.set_num_threads(experiment.runtime.torch_num_threads)
        self.quiet = quiet or not experiment.logging.progress
        self.seed_contract = SeedContract.from_config(environment)
        self._invocation_started = time.time()
        first_trace = self._trace(0, experiment.training.arrival_regimes[0])
        observation = _first_observation(environment, first_trace)
        feature_layout = FeatureLayout.from_view(observation.set_view)
        seeds = {
            name: self.seed_contract.derive_uint64(
                self.training_seed, _TRAINING_SEED_DOMAIN, name,
            )
            for name in ("model", "action", "minibatch")
        }
        if experiment.method == JOINT_CREDIT_METHOD:
            builder = build_joint_credit_agent
            self._collect_training_rollout = collect_training_rollout
        elif experiment.method == COMMON_TRACE_METHOD:
            builder = build_common_trace_agent
            self._collect_training_rollout = collect_common_trace_training_rollout
        else:
            raise TrainerValidationError("unsupported learned method")
        self.agent = builder(
            feature_layout, algorithm, environment,
            model_seed=seeds["model"], action_seed=seeds["action"],
            minibatch_seed=seeds["minibatch"],
        )
        tenants = tuple(sorted((item.tenant_id for item in environment.tenants), key=identifier_key))
        users = tuple(sorted({item.user_id for item in first_trace.communication_states}, key=identifier_key))
        self.layout = ConstraintLayout(tenants, users)
        self.constraint_labels = tuple(
            [_constraint_label("tenant", value) for value in tenants]
            + [_constraint_label("communication", value) for value in users]
        )
        self.environment_semantic_digest = semantic_digest(_environment_semantics(environment))
        self.validation_protocol_digest = semantic_digest({
            "trace_seeds": experiment.validation.trace_seeds,
            "arrival_regimes": experiment.validation.arrival_regimes,
            "random_valid_root_seed": experiment.validation.random_valid_root_seed,
            "random_valid_replicates_per_trace": experiment.validation.random_valid_replicates_per_trace,
        })
        self.selection_protocol_digest = semantic_digest({
            "version": _SELECTION_RULE_VERSION, "best_metric": experiment.checkpoint.best_metric,
            "keep_top_k": experiment.checkpoint.keep_top_k,
            "ranking": (
                "macro_score", "worst_regime_score", "lower_constraint_excess", "earlier_checkpoint",
            ),
        })
        self.architecture_signature = semantic_digest({
            "method": experiment.method,
            "credit_assignment_schema": self.credit_assignment_schema,
            "actor_critic": type(self.agent.model).__qualname__,
            "model": algorithm.model, "profiles": feature_layout.profile_ids,
            "tenant_count": self.layout.tenant_count,
            "communication_count": self.layout.communication_count,
        })
        self.metadata = CheckpointMetadata.current(
            method=experiment.method,
            credit_assignment_schema=self.credit_assignment_schema,
            training_seed=self.training_seed,
            feature_schema_digest=feature_layout.schema_digest, architecture_signature=self.architecture_signature,
            environment_semantic_digest=self.environment_semantic_digest,
            validation_protocol_digest=self.validation_protocol_digest, constraint_labels=self.constraint_labels,
        )
        self.progress = TrainingProgress(
            next_validation_boundary=_next_boundary(0, experiment.validation.interval_physical_slots),
            next_checkpoint_boundary=_next_boundary(0, experiment.checkpoint.interval_physical_slots),
        )
        self.validations: list[ValidationRecord] = []
        self.best_checkpoints: list[BestCheckpointRecord] = []
        self.segments: list[TrainingSegmentRecord] = []
        self.last_finite_checkpoint: str | None = None
        self.current_segment_latest_checkpoint: str | None = None
        self.random_valid: ValidationReport | None = None
        self._elapsed_before_segment = 0.0
        self._restored_active_segment: dict[str, Any] | None = None
        self._restored_segment_record: TrainingSegmentRecord | None = None
        self._selection_reset = False
        self._validation_reset = False
        self._validation_schema_reset = False
        if self.resume_path is not None:
            self._restore(self.resume_path)
            if not self.branch_resume and self.segments:
                previous = self.segments[-1]
                current_schedule = experiment.training.learning_rate_schedule
                current_horizon = experiment.training.learning_rate_schedule_horizon_physical_slots
                if previous.learning_rate_schedule != current_schedule:
                    raise TrainerValidationError(
                        "non-branch resume must preserve the learning-rate schedule"
                    )
                if current_schedule == "linear" and (
                    previous.learning_rate_schedule_horizon_physical_slots != current_horizon
                ):
                    raise TrainerValidationError(
                        "non-branch resume must preserve the linear learning-rate horizon"
                    )
            if "learning_rate" in optimizer_overrides:
                for group in self.agent.algorithm.optimizer.param_groups:
                    group["lr"] = algorithm.optimizer.learning_rate
            if "epsilon" in optimizer_overrides:
                for group in self.agent.algorithm.optimizer.param_groups:
                    group["eps"] = algorithm.optimizer.epsilon
            with torch.no_grad():
                self.agent.algorithm.dual_values.clamp_(0.0, algorithm.dual.maximum)
            if self.branch_resume:
                self.validations, self.best_checkpoints, self.segments = [], [], []
                if self._validation_reset or self._validation_schema_reset:
                    self.random_valid = None
                self._restored_active_segment = None
                self._restored_segment_record = None
                self.last_finite_checkpoint = self.resume_path.as_posix()
            elif self._validation_reset:
                raise TrainerValidationError("changed validation protocols must continue in a branch run")

        self.segment_index = len(self.segments)
        self.segment_start_slot = self.progress.completed_physical_slots
        self.segment_requested_slots = experiment.training.physical_slots
        self.segment_target_slot = self.segment_start_slot + self.segment_requested_slots
        algorithm_values = _algorithm_snapshot(algorithm)
        self.algorithm_semantic_digest = semantic_digest(algorithm_values)
        self.segment_started = time.time()
        self.segment_start_learning_rate = float(self.agent.algorithm.optimizer.param_groups[0]["lr"])
        self._segment_finalized = False
        append = self.resume_path is not None and not self.branch_resume
        archived_selection_series = self._archive_best_series() if append and self._selection_reset else None
        sources = config_sources or {
            "environment": DEFAULT_CONFIG_PATH, "algorithm": algorithm.source_path,
            "experiment": DEFAULT_EXPERIMENT_CONFIG_PATH,
        }
        self.config_sources = {name: Path(path).resolve() for name, path in sources.items()}
        self.continuation_config_paths = self._write_resolved_configs(self.config_sources, algorithm_values)
        parameter_count = sum(parameter.numel() for parameter in self.agent.model.parameters())
        trainable_count = sum(parameter.numel() for parameter in self.agent.model.parameters() if parameter.requires_grad)
        hardware = torch.cuda.get_device_name(self.agent.device) if self.agent.device.type == "cuda" else platform.processor() or platform.machine()
        write_json(self.output_directory / f"segment_{self.segment_index:04d}_provenance.json", {
            "artifact_schema_version": _ARTIFACT_SCHEMA, "method": experiment.method,
            "credit_assignment_schema": self.credit_assignment_schema,
            "training_seed": self.training_seed, "run_name": self.run_name,
            "parent_run_name": self.parent_run_name, "parent_checkpoint_path": self.parent_checkpoint_path,
            "environment": environment, "algorithm": algorithm_values, "experiment": experiment,
            "config_sources": {name: path.as_posix() for name, path in self.config_sources.items()},
            "resolved_config_paths": self.continuation_config_paths,
            "feature_schema_digest": feature_layout.schema_digest,
            "architecture_signature": self.architecture_signature,
            "environment_semantic_digest": self.environment_semantic_digest,
            "validation_protocol_digest": self.validation_protocol_digest,
            "selection_protocol_digest": self.selection_protocol_digest,
            "constraint_labels": self.constraint_labels, "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count, "device": str(self.agent.device),
            "python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "hardware": hardware, "started_unix_s": self._invocation_started,
        })

        self.artifacts = TrainingArtifacts(
            self.output_directory, append=append, jsonl=experiment.logging.jsonl,
            csv_enabled=experiment.logging.csv, flush_every_records=experiment.logging.flush_every_records,
        )
        if append:
            self._validate_existing_manifest(feature_layout.schema_digest)
            if self._selection_reset:
                self.artifacts.event("checkpoint_selection_reset", {
                    "physical_slot": self.progress.completed_physical_slots,
                    "selection_protocol_digest": self.selection_protocol_digest,
                    "validation_schema_reset": self._validation_schema_reset,
                    "archived_directory": archived_selection_series,
                })
        else:
            manifest = RunManifest(
                "isac-ssc-training-run-v3", _ARTIFACT_SCHEMA, experiment.method,
                self.credit_assignment_schema, self.training_seed, self.run_name,
                feature_layout.schema_digest, self.architecture_signature, self.environment_semantic_digest,
                self.validation_protocol_digest, self.selection_protocol_digest, self.constraint_labels,
                parameter_count, trainable_count, str(self.agent.device), platform.python_version(),
                str(torch.__version__), hardware, self._invocation_started,
            )
            write_json(self.output_directory / "manifest.json", manifest)
            write_json(self.output_directory / "effective_config.json", {
                "artifact_schema_version": _ARTIFACT_SCHEMA,
                "credit_assignment_schema": self.credit_assignment_schema,
                "environment": environment, "algorithm": algorithm_values, "experiment": experiment,
                "config_sources": {name: path.as_posix() for name, path in self.config_sources.items()},
                "constraint_labels": self.constraint_labels,
                "feature_schema_digest": feature_layout.schema_digest,
                "architecture_signature": self.architecture_signature,
                "environment_semantic_digest": self.environment_semantic_digest,
                "validation_protocol_digest": self.validation_protocol_digest,
                "selection_protocol_digest": self.selection_protocol_digest,
                "parent_run_name": self.parent_run_name,
                "parent_checkpoint_path": self.parent_checkpoint_path,
            })
            self.artifacts.event("manifest", manifest)
        if self.artifacts.schema_migrations:
            self.artifacts.event("artifact_schema_migration", self.artifacts.schema_migrations)
        if "segments" in self.artifacts.created_tables:
            for segment in self.segments:
                self.artifacts.row("segments", asdict(segment))
        elif self._restored_segment_record is not None:
            self.artifacts.row("segments", asdict(self._restored_segment_record))
        write_json(self.output_directory / f"segment_{self.segment_index:04d}_config.json", {
            "segment": self._active_segment(), "algorithm": algorithm_values,
            "experiment": experiment, "credit_assignment_schema": self.credit_assignment_schema,
            "constraint_labels": self.constraint_labels,
            "artifact_schema_version": _ARTIFACT_SCHEMA,
        })
        self.artifacts.event("training_segment_started", self._active_segment())

        self.validation_traces = tuple(
            generate_primitive_trace(environment, seed, regime)
            for seed in experiment.validation.trace_seeds
            for regime in experiment.validation.arrival_regimes
        ) if experiment.validation.enabled else ()
        random_valid_recomputed = False
        if self.validation_traces and self.random_valid is None:
            self.random_valid = evaluate_random_valid(
                environment, self.validation_traces,
                root_seed=experiment.validation.random_valid_root_seed,
                replicates_per_trace=experiment.validation.random_valid_replicates_per_trace,
            )
            random_valid_recomputed = True
        if self.random_valid is not None and not self.random_valid.all_finite:
            raise TrainerValidationError("random-valid evaluation produced non-finite metrics")
        if self.random_valid is not None and (not append or random_valid_recomputed or "validation_traces" in self.artifacts.created_tables):
            self._record_validation_report(self.random_valid)
        if self.random_valid is not None and (not append or random_valid_recomputed or self.artifacts.jsonl_created):
            self.artifacts.event("random_valid_cache", self.random_valid)

    def _write_resolved_configs(self, sources: dict[str, Path], algorithm_values: dict[str, Any]) -> dict[str, str]:
        names = {name: f"segment_{self.segment_index:04d}_{name}.yaml" for name in ("environment", "algorithm", "experiment")}
        environment_source = sources["environment"]
        if not environment_source.is_file():
            raise TrainerValidationError("environment config source does not exist")
        shutil.copy2(environment_source, self.output_directory / names["environment"])
        (self.output_directory / names["algorithm"]).write_text(
            yaml.safe_dump(_plain(algorithm_values), sort_keys=False, allow_unicode=True), encoding="utf-8",
        )
        (self.output_directory / names["experiment"]).write_text(
            yaml.safe_dump(_experiment_snapshot(self.experiment), sort_keys=False, allow_unicode=True), encoding="utf-8",
        )
        return names

    def _trace(self, episode_index: int, regime: str) -> PrimitiveTrace:
        seed = self.seed_contract.derive_uint64(
            self.training_seed, _TRAINING_SEED_DOMAIN, "training_trace", episode_index, regime,
        )
        return generate_primitive_trace(self.environment, seed, regime)

    def _validate_existing_manifest(self, feature_schema_digest: str) -> None:
        path = self.output_directory / "manifest.json"
        if not path.is_file():
            raise TrainerValidationError("resumed run is missing its original manifest")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainerValidationError("resumed run manifest is unreadable") from error
        expected = {
            "method": self.experiment.method,
            "credit_assignment_schema": self.credit_assignment_schema,
            "training_seed": self.training_seed, "run_name": self.run_name,
            "feature_schema_digest": feature_schema_digest,
            "architecture_signature": self.architecture_signature,
            "environment_semantic_digest": self.environment_semantic_digest,
            "constraint_labels": list(self.constraint_labels),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise TrainerValidationError("resumed run manifest is incompatible with the checkpoint")

    def _restore(self, path: Path) -> None:
        restored_metadata, state = load_checkpoint(path, self.agent, self.metadata)
        try:
            self.progress = TrainingProgress(**state["progress"])
            current_validation_schema = state.get("validation_result_schema") == _VALIDATION_RESULT_SCHEMA
            self._validation_schema_reset = not current_validation_schema
            self.validations = [_validation_record(item) for item in state.get("validations", [])] if current_validation_schema else []
            self.best_checkpoints = [BestCheckpointRecord(**item) for item in state.get("best_checkpoints", [])] if current_validation_schema else []
            self.segments = [_segment_record(item) for item in state.get("segments", [])]
            self.last_finite_checkpoint = state.get("last_finite_checkpoint", path.as_posix())
            self.random_valid = _validation_report(state.get("random_valid")) if current_validation_schema else None
            self._elapsed_before_segment = float(state.get("cumulative_elapsed_seconds", 0.0))
            self._restored_active_segment = state.get("active_segment")
            active_segment = state.get("active_segment") or {}
            previous_validation = state.get(
                "validation_protocol_digest",
                active_segment.get("validation_protocol_digest", restored_metadata.validation_protocol_digest),
            )
            self._validation_reset = previous_validation != self.validation_protocol_digest
            self._selection_reset = (
                self._validation_reset or self._validation_schema_reset
                or state.get("selection_protocol_digest") != self.selection_protocol_digest
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TrainerValidationError("checkpoint training history is malformed") from error
        if self._selection_reset:
            self.best_checkpoints = []
        if self._restored_active_segment:
            active = dict(self._restored_active_segment)
            active.update({
                "completed_physical_slot": self.progress.completed_physical_slots,
                "actual_physical_slots": self.progress.completed_physical_slots - int(active["start_physical_slot"]),
                "budget_overshoot_slots": max(0, self.progress.completed_physical_slots - int(active["target_physical_slot"])),
                "finished_unix_s": float(active["started_unix_s"]) + float(active["elapsed_seconds"]),
                "ending_learning_rate": float(self.agent.algorithm.optimizer.param_groups[0]["lr"]),
                "status": "checkpointed",
            })
            self._restored_segment_record = _segment_record(active)
            self.segments.append(self._restored_segment_record)
        self.progress.next_validation_boundary = _next_boundary(
            self.progress.completed_physical_slots, self.experiment.validation.interval_physical_slots,
        )
        self.progress.next_checkpoint_boundary = _next_boundary(
            self.progress.completed_physical_slots, self.experiment.checkpoint.interval_physical_slots,
        )

    def _archive_best_series(self) -> str | None:
        candidates = tuple(path for path in self.output_directory.glob("best*.pt") if path.is_file())
        if not candidates:
            return None
        destination = self.output_directory / "superseded_best" / f"slot_{self.progress.completed_physical_slots:08d}_{self.selection_protocol_digest[:8]}"
        destination.mkdir(parents=True, exist_ok=True)
        for path in candidates:
            archived = destination / path.name
            shutil.move(path.as_posix(), archived.as_posix())
            if self.resume_path is not None and path.resolve() == self.resume_path:
                self.resume_path = archived.resolve()
                self.last_finite_checkpoint = self.resume_path.as_posix()
        return destination.as_posix()

    def _active_segment(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "resume_checkpoint": None if self.resume_path is None else self.resume_path.as_posix(),
            "start_physical_slot": self.segment_start_slot,
            "requested_physical_slots": self.segment_requested_slots,
            "target_physical_slot": self.segment_target_slot,
            "training_regimes": self.experiment.training.arrival_regimes,
            "rollout_target_physical_slots": self.experiment.training.rollout_target_physical_slots,
            "learning_rate_schedule": self.experiment.training.learning_rate_schedule,
            "learning_rate_schedule_horizon_physical_slots": (
                self.experiment.training.learning_rate_schedule_horizon_physical_slots
            ),
            "algorithm_semantic_digest": self.algorithm_semantic_digest,
            "validation_protocol_digest": self.validation_protocol_digest,
            "validation_enabled": self.experiment.validation.enabled,
            "validation_interval_physical_slots": self.experiment.validation.interval_physical_slots,
            "checkpoint_interval_physical_slots": self.experiment.checkpoint.interval_physical_slots,
            "started_unix_s": self.segment_started, "elapsed_seconds": time.time() - self.segment_started,
            "starting_learning_rate": self.segment_start_learning_rate,
            "parent_run_name": self.parent_run_name, "parent_checkpoint_path": self.parent_checkpoint_path,
        }

    def _cumulative_elapsed(self) -> float:
        return self._elapsed_before_segment + (0.0 if self._segment_finalized else time.time() - self.segment_started)

    def _run_state(self, checkpoint_path: str) -> dict[str, object]:
        return {
            "progress": asdict(self.progress), "validations": [asdict(item) for item in self.validations],
            "best_checkpoints": [asdict(item) for item in self.best_checkpoints],
            "segments": [asdict(item) for item in self.segments],
            "active_segment": None if self._segment_finalized else self._active_segment(),
            "last_finite_checkpoint": checkpoint_path, "cumulative_elapsed_seconds": self._cumulative_elapsed(),
            "random_valid": None if self.random_valid is None else asdict(self.random_valid),
            "validation_result_schema": _VALIDATION_RESULT_SCHEMA,
            "credit_assignment_schema": self.credit_assignment_schema,
            "validation_protocol_digest": self.validation_protocol_digest,
            "selection_protocol_digest": self.selection_protocol_digest,
            "continuation_config_paths": self.continuation_config_paths,
            "parent_run_name": self.parent_run_name, "parent_checkpoint_path": self.parent_checkpoint_path,
        }

    def _save_checkpoint(
        self, name: str, kind: str, *, scheduled_physical_slot: int | None = None,
        metric: str = "", worst_regime_score: float | None = None, macro_score: float | None = None,
        constraint_excess: float | None = None, rank: int | None = None,
    ) -> tuple[str, str]:
        path = self.output_directory / name
        digest = save_checkpoint(path, self.agent, self.metadata, self._run_state(path.as_posix()))
        self.last_finite_checkpoint = path.as_posix()
        if kind == "latest":
            self.current_segment_latest_checkpoint = path.as_posix()
        actual = self.progress.completed_physical_slots
        scheduled = actual if scheduled_physical_slot is None else scheduled_physical_slot
        row = {
            "path": path.as_posix(), "type": kind, "scheduled_physical_slot": scheduled,
            "actual_physical_slot": actual, "interval_overshoot_slots": max(0, actual - scheduled),
            "metric": metric, "worst_regime_score": worst_regime_score, "macro_score": macro_score,
            "constraint_excess": constraint_excess, "rank_at_save": rank, "sha256": digest,
            "created_unix_s": time.time(),
        }
        self.artifacts.row("checkpoints", row)
        self.artifacts.event("checkpoint", row)
        return path.as_posix(), digest

    def _best_key(self, item: BestCheckpointRecord) -> tuple[float, float, float, int]:
        if item.metric == "constraint_lexicographic":
            return -item.constraint_excess, item.macro_score, item.worst_regime_score, -item.physical_slots
        return item.macro_score, item.worst_regime_score, -item.constraint_excess, -item.physical_slots

    def _primary_selection_score(self, candidate: BestCheckpointRecord) -> float:
        return -candidate.constraint_excess if candidate.metric == "constraint_lexicographic" else candidate.macro_score

    def _candidate_ranking(self, record: ValidationRecord) -> tuple[BestCheckpointRecord, list[BestCheckpointRecord], bool]:
        if self.experiment.checkpoint.best_metric == "validation_return":
            worst, macro = record.worst_regime_policy_return, record.macro_policy_return
        else:
            worst, macro = record.worst_regime_paired_return, record.macro_paired_return
        candidate = BestCheckpointRecord(
            (self.output_directory / f"best_{record.actual_physical_slot:08d}.pt").as_posix(),
            record.actual_physical_slot, self.experiment.checkpoint.best_metric,
            worst, macro, record.macro_positive_constraint_excess,
        )
        ranked = sorted((*self.best_checkpoints, candidate), key=self._best_key, reverse=True)
        keep = ranked[:self.experiment.checkpoint.keep_top_k]
        return candidate, keep, candidate in keep and keep[0] == candidate

    def _commit_best(self, candidate: BestCheckpointRecord, keep: list[BestCheckpointRecord]) -> None:
        ranked = sorted((*self.best_checkpoints, candidate), key=self._best_key, reverse=True)
        self.best_checkpoints = list(keep)
        self._save_checkpoint(
            Path(candidate.path).name, "best_candidate", metric=candidate.metric,
            worst_regime_score=candidate.worst_regime_score, macro_score=candidate.macro_score,
            constraint_excess=candidate.constraint_excess, rank=keep.index(candidate) + 1,
        )
        for old in ranked[self.experiment.checkpoint.keep_top_k:]:
            path = Path(old.path)
            if path.exists():
                path.unlink()
        self.best_checkpoints.sort(key=self._best_key, reverse=True)
        source = Path(self.best_checkpoints[0].path)
        temporary = self.output_directory / "best.pt.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(self.output_directory / "best.pt")

    def _validation_pair(
        self, policy: ValidationSummary, random_valid: ValidationSummary,
        scheduled_slot: int, actual_slot: int, *, is_overall: bool, is_worst: bool,
        best_score: float, is_best: bool,
    ) -> ValidationRegimeRecord:
        return ValidationRegimeRecord(
            scheduled_slot, actual_slot, max(0, actual_slot - scheduled_slot), policy.arrival_regime,
            is_overall, is_worst, policy.episode_count, random_valid.episode_count,
            policy.mean_return, policy.std_return, random_valid.mean_return, random_valid.std_return,
            policy.mean_return - random_valid.mean_return, policy.mean_completed_value,
            policy.mean_normalized_completed_value, policy.mean_sensing_resource_cost,
            policy.mean_positive_constraint_excess, policy.mean_reward_per_slot,
            policy.mean_completed_value_per_slot, policy.mean_sensing_resource_cost_per_slot,
            policy.mean_network_user_shortfall, policy.mean_fraction_users_within_budget,
            random_valid.mean_reward_per_slot, random_valid.mean_normalized_completed_value,
            random_valid.mean_network_user_shortfall, random_valid.mean_fraction_users_within_budget,
            policy.valid_action_rate, random_valid.valid_action_rate,
            self.experiment.checkpoint.best_metric, best_score, is_best,
        )

    def _run_validation(self, scheduled_physical_slot: int | None = None) -> ValidationRecord:
        if not self.validation_traces or self.random_valid is None:
            raise TrainerValidationError("validation is disabled")
        actual = self.progress.completed_physical_slots
        scheduled = actual if scheduled_physical_slot is None else scheduled_physical_slot
        policy = evaluate_policy(self.environment, self.agent, self.validation_traces, physical_slot=actual)
        if not policy.all_finite or not self.random_valid.all_finite:
            raise TrainerValidationError("validation produced non-finite metrics")
        policy_by_regime = {item.arrival_regime: item for item in policy.regimes}
        random_by_regime = {item.arrival_regime: item for item in self.random_valid.regimes}
        if tuple(policy_by_regime) != tuple(random_by_regime):
            raise TrainerValidationError("policy and random-valid validation regimes do not match")
        paired = {
            regime: policy_by_regime[regime].mean_return - random_by_regime[regime].mean_return
            for regime in policy_by_regime
        }
        worst_paired_regime = min(paired, key=lambda regime: (paired[regime], regime))
        worst_policy_regime = min(policy_by_regime, key=lambda regime: (policy_by_regime[regime].mean_return, regime))
        selection_worst_regime = (
            worst_policy_regime if self.experiment.checkpoint.best_metric == "validation_return" else worst_paired_regime
        )
        macro_paired = policy.overall.mean_return - self.random_valid.overall.mean_return
        worst_policy = policy_by_regime[worst_policy_regime].mean_return
        provisional = ValidationRecord(
            scheduled, actual, max(0, actual - scheduled), (), selection_worst_regime, paired[worst_paired_regime],
            macro_paired, worst_policy, policy.overall.mean_return,
            policy.overall.mean_positive_constraint_excess,
            self.experiment.checkpoint.best_metric, macro_paired, False,
        )
        candidate, keep, is_best = self._candidate_ranking(provisional)
        enters_top_k = candidate in keep
        best_score = self._primary_selection_score(candidate)
        records = tuple(
            self._validation_pair(
                policy_by_regime[regime], random_by_regime[regime], scheduled, actual,
                is_overall=False, is_worst=regime == selection_worst_regime, best_score=best_score, is_best=is_best,
            )
            for regime in policy_by_regime
        ) + (
            self._validation_pair(
                policy.overall, self.random_valid.overall, scheduled, actual,
                is_overall=True, is_worst=False, best_score=best_score, is_best=is_best,
            ),
        )
        record = ValidationRecord(
            scheduled, actual, max(0, actual - scheduled), records, selection_worst_regime,
            paired[worst_paired_regime], macro_paired, worst_policy, policy.overall.mean_return,
            policy.overall.mean_positive_constraint_excess,
            self.experiment.checkpoint.best_metric, best_score, is_best,
        )
        self.validations.append(record)
        if enters_top_k:
            self._commit_best(candidate, keep)
        for item in record.regimes:
            self.artifacts.row("validation", asdict(item))
        self._record_validation_report(policy)
        self.artifacts.event("validation", {"record": record, "policy": policy, "random_valid": self.random_valid})
        if not self.quiet:
            for item in record.regimes:
                marker = " | WORST" if item.is_worst_regime else ""
                label = "overall-macro" if item.is_overall else item.arrival_regime
                print(
                    f"[validation:{label}] slot {actual} scheduled {scheduled} overshoot {record.interval_overshoot_slots} | "
                    f"return {item.policy_mean_return:.4f} | random {item.random_valid_mean_return:.4f} | "
                    f"paired {item.paired_return_difference:+.4f} | constraint {item.policy_mean_positive_constraint_excess:.6f}{marker}",
                    flush=True,
                )
            if is_best:
                print(
                    f"[validation] NEW BEST | macro paired {record.macro_paired_return:+.4f} | "
                    f"worst {record.worst_regime} {record.worst_regime_paired_return:+.4f} | "
                    f"constraint {record.macro_positive_constraint_excess:.6f}",
                    flush=True,
                )
        return record

    def _set_learning_rate(self) -> float:
        if self.experiment.training.learning_rate_schedule == "linear":
            completed = self.progress.completed_physical_slots
            horizon = self.experiment.training.learning_rate_schedule_horizon_physical_slots
            fraction = max(0.0, 1.0 - completed / horizon)
            learning_rate = self.algorithm_config.optimizer.learning_rate * fraction
        else:
            learning_rate = self.algorithm_config.optimizer.learning_rate
        for group in self.agent.algorithm.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def _episode_row(self, episode: EpisodeCollectionMetrics, rollout_index: int) -> dict[str, Any]:
        actions = _action_counts(episode.action_counts)
        return {
            "rollout_index": rollout_index, "episode_index": episode.episode_index,
            "trace_id": episode.trace_id, "root_seed": episode.root_seed, "arrival_regime": episode.arrival_regime,
            "physical_slots": episode.physical_slots, "focal_decisions": episode.focal_decisions,
            "reward_total": episode.reward_total, "reward_per_slot": episode.reward_per_slot,
            "completed_value_total": episode.completed_value_total,
            "arrived_request_value_total": episode.arrived_request_value_total,
            "normalized_completed_value": episode.normalized_completed_value,
            "completed_value_per_slot": episode.completed_value_per_slot,
            "sensing_resource_cost_total": episode.sensing_resource_cost_total,
            "sensing_resource_cost_per_slot": episode.sensing_resource_cost_per_slot,
            "sensing_bandwidth_hz_slot_sum": episode.sensing_bandwidth_hz_slot_sum,
            "sensing_bandwidth_hz_mean": episode.sensing_bandwidth_hz_mean,
            "sensing_bandwidth_hz_max": episode.sensing_bandwidth_hz_max,
            "sensing_power_w_slot_sum": episode.sensing_power_w_slot_sum,
            "sensing_power_w_mean": episode.sensing_power_w_mean, "sensing_power_w_max": episode.sensing_power_w_max,
            "slots_with_session_update": episode.slots_with_session_update,
            "session_update_count": episode.session_update_count,
            "tracking_prediction_count": episode.tracking_prediction_count,
            "post_slot_active_session_count_mean": episode.post_slot_active_session_count_mean,
            "post_slot_active_session_count_max": episode.post_slot_active_session_count_max,
            "arrived": episode.arrived, "accepted": episode.accepted, "completed": episode.completed,
            "rejected": episode.rejected, "expired": episode.expired, "failed": episode.failed,
            "valid_outputs": episode.valid_outputs, "first_violations": episode.first_violations,
            "created_sessions": episode.created_sessions, "acceptance_ratio": episode.acceptance_ratio,
            "completion_ratio": episode.completion_ratio, "rejection_ratio": episode.rejection_ratio,
            "requests_served_per_created_session": episode.requests_served_per_created_session,
            "network_mean_user_shortfall": episode.network_mean_user_shortfall,
            "fraction_users_within_budget": episode.fraction_users_within_budget,
            "merge_actions": actions.get(ActionType.MERGE.value, 0), "create_actions": actions.get(ActionType.CREATE.value, 0),
            "defer_actions": actions.get(ActionType.DEFER.value, 0), "reject_actions": actions.get(ActionType.REJECT.value, 0),
            "merge_action_rate": actions.get(ActionType.MERGE.value, 0) / max(1, episode.focal_decisions),
            "create_action_rate": actions.get(ActionType.CREATE.value, 0) / max(1, episode.focal_decisions),
            "defer_action_rate": actions.get(ActionType.DEFER.value, 0) / max(1, episode.focal_decisions),
            "reject_action_rate": actions.get(ActionType.REJECT.value, 0) / max(1, episode.focal_decisions),
            "valid_action_rate": episode.valid_action_rate,
            "positive_constraint_excess": episode.positive_constraint_excess,
        }

    def _tenant_row(
        self, item: TenantMetrics, *, record_type: str, rollout_index: int | None,
        episode: EpisodeCollectionMetrics, policy: str | None = None, physical_slot: int | None = None,
        replicate: int | None = None,
    ) -> dict[str, Any]:
        return {
            "record_type": record_type, "rollout_index": rollout_index, "episode_index": episode.episode_index,
            "physical_slot": physical_slot, "policy": policy, "trace_id": episode.trace_id,
            "root_seed": episode.root_seed, "arrival_regime": episode.arrival_regime, "replicate": replicate,
            "tenant_id": item.tenant_id, "sla_violation_budget": item.sla_violation_budget,
            "arrived": item.arrived, "accepted": item.accepted, "completed": item.completed,
            "rejected": item.rejected, "expired": item.expired, "failed": item.failed,
            "first_violated": item.first_violated, "acceptance_ratio": item.acceptance_ratio,
            "completion_ratio": item.completion_ratio, "violation_rate": item.violation_rate,
            "residual_total": item.residual_total, "positive_residual": item.positive_residual,
        }

    def _communication_row(
        self, item: CommunicationUserMetrics, *, record_type: str, rollout_index: int | None,
        episode: EpisodeCollectionMetrics, policy: str | None = None, physical_slot: int | None = None,
        replicate: int | None = None,
    ) -> dict[str, Any]:
        return {
            "record_type": record_type, "rollout_index": rollout_index, "episode_index": episode.episode_index,
            "physical_slot": physical_slot, "policy": policy, "trace_id": episode.trace_id,
            "root_seed": episode.root_seed, "arrival_regime": episode.arrival_regime, "replicate": replicate,
            "user_id": item.user_id, "normalized_shortfall_budget": item.normalized_shortfall_budget,
            "active_demand_slots": item.active_demand_slots,
            "demand_bit_per_s_slot_sum": item.demand_bit_per_s_slot_sum,
            "mean_active_demand_bit_per_s": item.mean_active_demand_bit_per_s,
            "allocated_bandwidth_hz_slot_sum": item.allocated_bandwidth_hz_slot_sum,
            "mean_active_allocated_bandwidth_hz": item.mean_active_allocated_bandwidth_hz,
            "allocated_power_w_slot_sum": item.allocated_power_w_slot_sum,
            "mean_active_allocated_power_w": item.mean_active_allocated_power_w,
            "achievable_rate_bit_per_s_slot_sum": item.achievable_rate_bit_per_s_slot_sum,
            "mean_active_achievable_rate_bit_per_s": item.mean_active_achievable_rate_bit_per_s,
            "served_rate_bit_per_s_slot_sum": item.served_rate_bit_per_s_slot_sum,
            "mean_active_served_rate_bit_per_s": item.mean_active_served_rate_bit_per_s,
            "normalized_shortfall_sum": item.normalized_shortfall_sum,
            "mean_normalized_shortfall": item.mean_normalized_shortfall,
            "residual_total": item.residual_total, "positive_residual": item.positive_residual,
        }

    def _record_validation_report(self, report: ValidationReport) -> None:
        for episode in report.episodes:
            metrics = episode.metrics
            trace_row = self._episode_row(metrics, rollout_index=-1)
            trace_row.update({"policy": report.policy, "physical_slot": report.physical_slot, "replicate": episode.replicate})
            self.artifacts.row("validation_traces", trace_row)
            for tenant in metrics.tenants:
                self.artifacts.row("validation_tenants", self._tenant_row(
                    tenant, record_type="validation", rollout_index=None, episode=metrics,
                    policy=report.policy, physical_slot=report.physical_slot, replicate=episode.replicate,
                ))
            for user in metrics.communication_users:
                self.artifacts.row("validation_communication_users", self._communication_row(
                    user, record_type="validation", rollout_index=None, episode=metrics,
                    policy=report.policy, physical_slot=report.physical_slot, replicate=episode.replicate,
                ))
            residuals = (*metrics.tenant_residual_totals, *metrics.communication_residual_totals)
            for label, residual in zip(self.constraint_labels, residuals, strict=True):
                family, entity = _constraint_identity(label)
                self.artifacts.row("validation_constraints", {
                    "policy": report.policy, "physical_slot": report.physical_slot,
                    "trace_id": metrics.trace_id, "root_seed": metrics.root_seed,
                    "arrival_regime": metrics.arrival_regime, "replicate": episode.replicate,
                    "episode_index": metrics.episode_index, "family": family,
                    "constraint_label": label, "entity_id": entity,
                    "residual_total": residual, "positive_excess": max(0.0, residual),
                })

    def _record_rollout(
        self, collected: CollectedRollout, ppo, dual, learning_rate: float,
        elapsed: float, requested_rollout_slots: int,
    ) -> None:
        metrics = collected.metrics
        start_slot = self.progress.completed_physical_slots - metrics.physical_slots
        action_counts = _action_counts(metrics.action_counts)
        decision_denominator = max(1, metrics.focal_decisions)
        residuals = (*metrics.tenant_residual_totals, *metrics.communication_residual_totals)
        worst_index = max(range(len(residuals)), key=lambda index: residuals[index])
        dual_values = tuple(map(float, dual.dual_values_after))
        factor_reward = None if ppo is None or not hasattr(ppo, "mean_reward_surrogates_by_factor") else tuple(
            map(float, ppo.mean_reward_surrogates_by_factor)
        )
        factor_constraints = None if ppo is None or not hasattr(ppo, "mean_constraint_surrogates_by_factor") else ppo.mean_constraint_surrogates_by_factor
        factor_clips = None if ppo is None or not hasattr(ppo, "mean_factor_clip_fractions") else tuple(
            map(float, ppo.mean_factor_clip_fractions)
        )
        factor_advantage_means = None if ppo is None or not hasattr(ppo, "mean_normalized_reward_advantages_by_factor") else tuple(
            map(float, ppo.mean_normalized_reward_advantages_by_factor)
        )
        factor_positive_fractions = None if ppo is None or not hasattr(ppo, "positive_normalized_reward_advantage_fractions") else tuple(
            map(float, ppo.positive_normalized_reward_advantage_fractions)
        )
        joint_quantiles = None if ppo is None or not hasattr(ppo, "joint_ratio_quantiles") else tuple(
            map(float, ppo.joint_ratio_quantiles)
        )
        constraint_advantage_scales = None if ppo is None else tuple(
            map(float, ppo.constraint_advantage_scales)
        )
        constraint_return_scales = None if ppo is None else tuple(
            map(float, ppo.constraint_return_scales)
        )
        normalizer_state = self.agent.normalizer.state()
        segment_elapsed = time.time() - self.segment_started
        remaining = max(0, self.segment_target_slot - self.progress.completed_physical_slots)
        throughput = metrics.physical_slots / max(elapsed, 1e-9)
        eta = remaining / throughput if remaining > 0 else 0.0
        row = {
            "rollout_index": self.progress.rollout_index, "start_slot": start_slot,
            "end_slot": self.progress.completed_physical_slots, "requested_rollout_slots": requested_rollout_slots,
            "actual_rollout_slots": metrics.physical_slots,
            "rollout_overshoot_slots": max(0, metrics.physical_slots - requested_rollout_slots),
            "episodes": metrics.episodes, "focal_decisions": metrics.focal_decisions,
            "progress_fraction": min(1.0, (self.progress.completed_physical_slots - self.segment_start_slot) / self.segment_requested_slots),
            "remaining_requested_slots": remaining, "reward_total": metrics.reward_total,
            "reward_per_slot": metrics.reward_per_slot, "reward_per_episode": metrics.reward_per_episode,
            "completed_value_total": metrics.completed_value_total,
            "arrived_request_value_total": metrics.arrived_request_value_total,
            "normalized_completed_value": metrics.normalized_completed_value,
            "completed_value_per_slot": metrics.completed_value_per_slot,
            "completed_value_per_episode": metrics.completed_value_per_episode,
            "sensing_resource_cost_total": metrics.sensing_resource_cost_total,
            "sensing_resource_cost_per_slot": metrics.sensing_resource_cost_per_slot,
            "sensing_resource_cost_per_episode": metrics.sensing_resource_cost_per_episode,
            "sensing_bandwidth_hz_slot_sum": metrics.sensing_bandwidth_hz_slot_sum,
            "sensing_bandwidth_hz_mean": metrics.sensing_bandwidth_hz_mean,
            "sensing_bandwidth_hz_max": metrics.sensing_bandwidth_hz_max,
            "sensing_power_w_slot_sum": metrics.sensing_power_w_slot_sum,
            "sensing_power_w_mean": metrics.sensing_power_w_mean, "sensing_power_w_max": metrics.sensing_power_w_max,
            "slots_with_session_update": metrics.slots_with_session_update,
            "session_update_count": metrics.session_update_count,
            "tracking_prediction_count": metrics.tracking_prediction_count,
            "post_slot_active_session_count_mean": metrics.post_slot_active_session_count_mean,
            "post_slot_active_session_count_max": metrics.post_slot_active_session_count_max,
            "arrived": metrics.arrived, "accepted": metrics.accepted, "completed": metrics.completed,
            "rejected": metrics.rejected, "expired": metrics.expired, "failed": metrics.failed,
            "valid_outputs": metrics.valid_outputs, "first_violations": metrics.first_violations,
            "created_sessions": metrics.created_sessions, "acceptance_ratio": metrics.acceptance_ratio,
            "completion_ratio": metrics.completion_ratio, "rejection_ratio": metrics.rejection_ratio,
            "requests_served_per_created_session": metrics.requests_served_per_created_session,
            "network_mean_user_shortfall": metrics.network_mean_user_shortfall,
            "fraction_users_within_budget": metrics.fraction_users_within_budget,
            "merge_actions": action_counts.get(ActionType.MERGE.value, 0),
            "create_actions": action_counts.get(ActionType.CREATE.value, 0),
            "defer_actions": action_counts.get(ActionType.DEFER.value, 0),
            "reject_actions": action_counts.get(ActionType.REJECT.value, 0),
            "merge_action_rate": action_counts.get(ActionType.MERGE.value, 0) / decision_denominator,
            "create_action_rate": action_counts.get(ActionType.CREATE.value, 0) / decision_denominator,
            "defer_action_rate": action_counts.get(ActionType.DEFER.value, 0) / decision_denominator,
            "reject_action_rate": action_counts.get(ActionType.REJECT.value, 0) / decision_denominator,
            "valid_action_rate": metrics.valid_action_rate,
            "transitions": None if ppo is None else ppo.transitions,
            "epochs_completed": None if ppo is None else ppo.epochs_completed,
            "minibatches_completed": None if ppo is None else ppo.minibatches_completed,
            "optimizer_steps": None if ppo is None else ppo.optimizer_steps,
            "early_stopped_for_kl": None if ppo is None else ppo.early_stopped_for_kl,
            "total_loss": None if ppo is None else ppo.mean_total_loss,
            "actor_loss": None if ppo is None else ppo.mean_actor_loss,
            "reward_surrogate": None if ppo is None else ppo.mean_reward_surrogate,
            "type_reward_surrogate": None if factor_reward is None else factor_reward[0],
            "session_reward_surrogate": None if factor_reward is None else factor_reward[1],
            "profile_reward_surrogate": None if factor_reward is None else factor_reward[2],
            "reward_value_loss": None if ppo is None else ppo.mean_reward_value_loss,
            "constraint_value_loss": None if ppo is None else ppo.mean_constraint_value_loss,
            "tenant_value_loss": None if ppo is None else ppo.mean_tenant_value_loss,
            "communication_value_loss": None if ppo is None else ppo.mean_communication_value_loss,
            "type_reward_value_loss": None if ppo is None else getattr(ppo, "mean_type_reward_value_loss", None),
            "type_constraint_value_loss": None if ppo is None else getattr(ppo, "mean_type_constraint_value_loss", None),
            "session_reward_value_loss": None if ppo is None else getattr(ppo, "mean_session_reward_value_loss", None),
            "session_constraint_value_loss": None if ppo is None else getattr(ppo, "mean_session_constraint_value_loss", None),
            "entropy": None if ppo is None else ppo.mean_entropy,
            "approximate_kl": None if ppo is None else ppo.mean_approximate_kl,
            "max_minibatch_approximate_kl": None if ppo is None else ppo.max_minibatch_approximate_kl,
            "clip_fraction": None if ppo is None else ppo.mean_clip_fraction,
            "type_clip_fraction": None if factor_clips is None else factor_clips[0],
            "session_clip_fraction": None if factor_clips is None else factor_clips[1],
            "profile_clip_fraction": None if factor_clips is None else factor_clips[2],
            "joint_ratio_p01": None if joint_quantiles is None else joint_quantiles[0],
            "joint_ratio_p05": None if joint_quantiles is None else joint_quantiles[1],
            "joint_ratio_p50": None if joint_quantiles is None else joint_quantiles[2],
            "joint_ratio_p95": None if joint_quantiles is None else joint_quantiles[3],
            "joint_ratio_p99": None if joint_quantiles is None else joint_quantiles[4],
            "minimum_joint_ratio": None if ppo is None else getattr(ppo, "minimum_joint_ratio", None),
            "maximum_joint_ratio": None if ppo is None else getattr(ppo, "maximum_joint_ratio", None),
            "nonfinite_joint_ratio_count": None if ppo is None else getattr(ppo, "nonfinite_joint_ratio_count", None),
            "gradient_norm": None if ppo is None else ppo.mean_gradient_norm_before_clip,
            "max_gradient_norm": None if ppo is None else ppo.max_gradient_norm_before_clip,
            "actor_gradient_norm": None if ppo is None else ppo.mean_actor_gradient_norm_before_clip,
            "max_actor_gradient_norm": None if ppo is None else ppo.max_actor_gradient_norm_before_clip,
            "critic_gradient_norm": None if ppo is None else getattr(
                ppo, "mean_critic_gradient_norm_before_clip",
                getattr(ppo, "mean_global_critic_gradient_norm_before_clip", None),
            ),
            "max_critic_gradient_norm": None if ppo is None else getattr(
                ppo, "max_critic_gradient_norm_before_clip",
                getattr(ppo, "max_global_critic_gradient_norm_before_clip", None),
            ),
            "type_prefix_gradient_norm": None if ppo is None else getattr(ppo, "mean_type_prefix_gradient_norm_before_clip", None),
            "max_type_prefix_gradient_norm": None if ppo is None else getattr(ppo, "max_type_prefix_gradient_norm_before_clip", None),
            "session_prefix_gradient_norm": None if ppo is None else getattr(ppo, "mean_session_prefix_gradient_norm_before_clip", None),
            "max_session_prefix_gradient_norm": None if ppo is None else getattr(ppo, "max_session_prefix_gradient_norm_before_clip", None),
            "prefix_gradient_norm": None if ppo is None else getattr(ppo, "mean_prefix_gradient_norm_before_clip", None),
            "max_prefix_gradient_norm": None if ppo is None else getattr(ppo, "max_prefix_gradient_norm_before_clip", None),
            "merge_transition_count": None if ppo is None else getattr(ppo, "merge_transition_count", None),
            "profile_transition_count": None if ppo is None else getattr(ppo, "profile_transition_count", None),
            "single_session_merge_transition_count": None if ppo is None else getattr(ppo, "single_session_merge_transition_count", None),
            "multi_session_merge_transition_count": None if ppo is None else getattr(ppo, "multi_session_merge_transition_count", None),
            "session_prefix_reward_target_variance": None if ppo is None else getattr(ppo, "session_prefix_reward_target_variance", None),
            "session_prefix_constraint_target_variance": None if ppo is None else getattr(ppo, "session_prefix_constraint_target_variance", None),
            "type_normalized_reward_advantage_mean": None if factor_advantage_means is None else factor_advantage_means[0],
            "session_normalized_reward_advantage_mean": None if factor_advantage_means is None else factor_advantage_means[1],
            "profile_normalized_reward_advantage_mean": None if factor_advantage_means is None else factor_advantage_means[2],
            "type_positive_reward_advantage_fraction": None if factor_positive_fractions is None else factor_positive_fractions[0],
            "session_positive_reward_advantage_fraction": None if factor_positive_fractions is None else factor_positive_fractions[1],
            "profile_positive_reward_advantage_fraction": None if factor_positive_fractions is None else factor_positive_fractions[2],
            "reward_advantage_scale": None if ppo is None else ppo.reward_advantage_scale,
            "constraint_advantage_scale_min": None if constraint_advantage_scales is None else min(constraint_advantage_scales),
            "constraint_advantage_scale_mean": None if constraint_advantage_scales is None else _finite_mean(constraint_advantage_scales),
            "constraint_advantage_scale_max": None if constraint_advantage_scales is None else max(constraint_advantage_scales),
            "reward_return_scale": None if ppo is None else ppo.reward_return_scale,
            "constraint_return_scale_min": None if constraint_return_scales is None else min(constraint_return_scales),
            "constraint_return_scale_mean": None if constraint_return_scales is None else _finite_mean(constraint_return_scales),
            "constraint_return_scale_max": None if constraint_return_scales is None else max(constraint_return_scales),
            "normalizer_frozen": normalizer_state.frozen,
            "normalizer_request_count": normalizer_state.request_count,
            "normalizer_session_count": normalizer_state.session_count,
            "normalizer_global_count": normalizer_state.global_count,
            "worst_constraint_label": self.constraint_labels[worst_index],
            "worst_constraint_residual": residuals[worst_index],
            "positive_constraint_excess": metrics.positive_constraint_excess,
            "dual_min": min(dual_values), "dual_mean": _finite_mean(dual_values), "dual_max": max(dual_values),
            "learning_rate": learning_rate, "elapsed_seconds": elapsed,
            "cumulative_segment_elapsed_seconds": segment_elapsed, "slots_per_second": throughput,
            "eta_seconds": eta,
        }
        self.artifacts.row("rollouts", row)
        for episode in collected.episodes:
            self.artifacts.row("episodes", self._episode_row(episode, self.progress.rollout_index))
            for tenant in episode.tenants:
                self.artifacts.row("train_tenants", self._tenant_row(
                    tenant, record_type="training", rollout_index=self.progress.rollout_index, episode=episode,
                ))
            for user in episode.communication_users:
                self.artifacts.row("train_communication_users", self._communication_row(
                    user, record_type="training", rollout_index=self.progress.rollout_index, episode=episode,
                ))
        surrogates = (None,) * len(self.constraint_labels) if ppo is None else tuple(map(float, ppo.mean_constraint_surrogates))
        dual_used = (None,) * len(self.constraint_labels) if ppo is None else tuple(map(float, ppo.dual_values_used))
        for index, label in enumerate(self.constraint_labels):
            family, entity = _constraint_identity(label)
            self.artifacts.row("train_constraints", {
                "rollout_index": self.progress.rollout_index, "start_slot": start_slot,
                "end_slot": self.progress.completed_physical_slots, "family": family,
                "constraint_label": label, "entity_id": entity, "residual_total": residuals[index],
                "mean_episode_residual": float(dual.mean_episode_residuals[index]),
                "positive_excess": max(0.0, residuals[index]), "constraint_surrogate": surrogates[index],
                "type_constraint_surrogate": None if factor_constraints is None else float(factor_constraints[0, index]),
                "session_constraint_surrogate": None if factor_constraints is None else float(factor_constraints[1, index]),
                "profile_constraint_surrogate": None if factor_constraints is None else float(factor_constraints[2, index]),
                "actor_dual_used": dual_used[index],
                "raw_dual_before": float(dual.dual_values_before[index]),
                "raw_dual_after": float(dual.dual_values_after[index]),
                "advantage_scale": None if constraint_advantage_scales is None else constraint_advantage_scales[index],
                "return_scale": None if constraint_return_scales is None else constraint_return_scales[index],
            })
        self.artifacts.event("rollout", {"row": row, "episodes": collected.episodes, "ppo": ppo, "dual": dual})
        if not self.quiet and self.progress.rollout_index % self.experiment.logging.progress_every_rollouts == 0:
            loss_text = "N/A" if ppo is None else (
                f"actor {ppo.mean_actor_loss:.5f} | reward_v {ppo.mean_reward_value_loss:.5f} | "
                f"constraint_v {ppo.mean_constraint_value_loss:.5f} | entropy {ppo.mean_entropy:.5f} | "
                f"KL {ppo.mean_approximate_kl:.5f} | clip {ppo.mean_clip_fraction:.3f} | "
                f"grad composed/actor/critic {ppo.mean_gradient_norm_before_clip:.4f}/"
                f"{ppo.mean_actor_gradient_norm_before_clip:.4f}/"
                f"{getattr(ppo, 'mean_critic_gradient_norm_before_clip', getattr(ppo, 'mean_global_critic_gradient_norm_before_clip', 0.0)):.4f} | "
                f"epochs {ppo.epochs_completed} | steps {ppo.optimizer_steps} | KL_stop {ppo.early_stopped_for_kl}"
            )
            print(
                f"[train] slot {self.progress.completed_physical_slots}/{self.segment_target_slot} "
                f"({row['progress_fraction']:.1%}) | rollout {self.progress.rollout_index} | "
                f"{throughput:.1f} slots/s | ETA {eta:.1f}s | reward/slot {metrics.reward_per_slot:.5f} | "
                f"value/slot {metrics.completed_value_per_slot:.5f} | normalized value "
                f"{'N/A' if metrics.normalized_completed_value is None else f'{metrics.normalized_completed_value:.3f}'} | "
                f"cost/slot {metrics.sensing_resource_cost_per_slot:.5f}",
                flush=True,
            )
            accept_text = "N/A" if metrics.acceptance_ratio is None else f"{metrics.acceptance_ratio:.1%}"
            complete_text = "N/A" if metrics.completion_ratio is None else f"{metrics.completion_ratio:.1%}"
            reject_text = "N/A" if metrics.rejection_ratio is None else f"{metrics.rejection_ratio:.1%}"
            print(
                f"[opt] {loss_text} | worst {row['worst_constraint_label']}={row['worst_constraint_residual']:+.5f} | "
                f"dual min/mean/max {row['dual_min']:.4f}/{row['dual_mean']:.4f}/{row['dual_max']:.4f} | "
                f"actions M/C/D/R {row['merge_action_rate']:.1%}/{row['create_action_rate']:.1%}/"
                f"{row['defer_action_rate']:.1%}/{row['reject_action_rate']:.1%} | "
                f"accept/complete/reject {accept_text}/{complete_text}/{reject_text}",
                flush=True,
            )

    def _finalize_segment(self, status: str) -> TrainingSegmentRecord:
        if self._segment_finalized:
            return self.segments[-1]
        finished = time.time()
        elapsed = finished - self.segment_started
        actual = self.progress.completed_physical_slots - self.segment_start_slot
        overshoot = max(0, self.progress.completed_physical_slots - self.segment_target_slot)
        record = TrainingSegmentRecord(
            self.segment_index, None if self.resume_path is None else self.resume_path.as_posix(),
            self.segment_start_slot, self.segment_requested_slots, self.segment_target_slot,
            self.progress.completed_physical_slots, actual, overshoot,
            self.experiment.training.arrival_regimes, self.experiment.training.rollout_target_physical_slots,
            self.experiment.training.learning_rate_schedule,
            self.experiment.training.learning_rate_schedule_horizon_physical_slots,
            self.algorithm_semantic_digest, self.validation_protocol_digest,
            self.experiment.validation.enabled,
            self.experiment.validation.interval_physical_slots,
            self.experiment.checkpoint.interval_physical_slots, self.segment_started, finished, elapsed,
            self.segment_start_learning_rate, float(self.agent.algorithm.optimizer.param_groups[0]["lr"]), status,
        )
        self.segments.append(record)
        self._elapsed_before_segment += elapsed
        self._segment_finalized = True
        self.artifacts.row("segments", asdict(record))
        self.artifacts.event("training_segment_finished", record)
        return record

    def run(self) -> TrainingRunSummary:
        final_path = ""
        try:
            if self.experiment.validation.enabled and self.experiment.validation.run_before_training and not self.validations:
                self._run_validation(self.progress.completed_physical_slots)
                if self.experiment.checkpoint.save_latest_every_rollout:
                    self._save_checkpoint("latest.pt", "latest")
            while self.progress.completed_physical_slots < self.segment_target_slot:
                started = time.time()
                remaining = self.segment_target_slot - self.progress.completed_physical_slots
                rollout_target = min(self.experiment.training.rollout_target_physical_slots, remaining)
                collected = self._collect_training_rollout(
                    ISACSSCEnv(self.environment), self.agent, self.layout, self._trace,
                    self.progress.next_episode_index, rollout_target, self.algorithm_config,
                    self.experiment.training.arrival_regimes,
                )
                if not self.agent.normalizer.frozen:
                    self.agent.normalizer.calibrate_and_freeze(
                        collected.focal_observations, self.agent.model.encoder,
                    )
                learning_rate = self._set_learning_rate()
                ppo = self.agent.algorithm.optimize_rollout(
                    collected.rollout, generator=self.agent.minibatch_generator,
                ) if collected.rollout.transition_count else None
                dual = self.agent.algorithm.update_duals(collected.rollout.episode_constraint_totals)
                if not self.agent.is_finite():
                    raise TrainerValidationError("training update produced a non-finite state")
                metrics = collected.metrics
                self.progress.completed_physical_slots += metrics.physical_slots
                self.progress.completed_episodes += metrics.episodes
                self.progress.focal_decisions += metrics.focal_decisions
                self.progress.valid_actions += metrics.valid_actions
                self.progress.invalid_actions += metrics.invalid_actions
                self.progress.rollout_index += 1
                self.progress.next_episode_index = collected.next_episode_index
                self._record_rollout(collected, ppo, dual, learning_rate, time.time() - started, rollout_target)
                if self.experiment.checkpoint.save_latest_every_rollout:
                    self._save_checkpoint("latest.pt", "latest")
                if (
                    self.experiment.validation.enabled
                    and self.experiment.validation.interval_physical_slots > 0
                    and self.progress.completed_physical_slots >= self.progress.next_validation_boundary
                ):
                    scheduled = self.progress.next_validation_boundary
                    self._run_validation(scheduled)
                    self.progress.next_validation_boundary = _next_boundary(
                        self.progress.completed_physical_slots, self.experiment.validation.interval_physical_slots,
                    )
                    if self.experiment.checkpoint.save_latest_every_rollout:
                        self._save_checkpoint("latest.pt", "latest")
                if (
                    self.experiment.checkpoint.interval_physical_slots > 0
                    and self.progress.completed_physical_slots >= self.progress.next_checkpoint_boundary
                ):
                    scheduled = self.progress.next_checkpoint_boundary
                    self.progress.next_checkpoint_boundary = _next_boundary(
                        self.progress.completed_physical_slots, self.experiment.checkpoint.interval_physical_slots,
                    )
                    self._save_checkpoint(
                        f"recovery_{self.progress.completed_physical_slots:08d}.pt", "recovery",
                        scheduled_physical_slot=scheduled,
                    )
            if self.experiment.validation.enabled and (
                not self.validations or self.validations[-1].actual_physical_slot != self.progress.completed_physical_slots
            ):
                self._run_validation(self.progress.completed_physical_slots)
            segment = self._finalize_segment("completed")
            if self.experiment.checkpoint.save_latest_every_rollout:
                self._save_checkpoint("latest.pt", "latest")
            final_path, _ = self._save_checkpoint("final.pt", "final")
            best = self.best_checkpoints[0] if self.best_checkpoints else None
            summary = TrainingRunSummary(
                "isac-ssc-training-summary-v5", _ARTIFACT_SCHEMA,
                self.experiment.method, self.credit_assignment_schema,
                self.training_seed, self.run_name,
                self.parent_run_name, self.parent_checkpoint_path,
                self.segment_requested_slots, segment.actual_physical_slots, segment.budget_overshoot_slots,
                self.segment_start_slot, self.segment_target_slot, self.progress.completed_physical_slots,
                self.progress.completed_episodes, self.progress.focal_decisions, self.progress.valid_action_rate,
                None if best is None else (self.output_directory / "best.pt").as_posix(),
                None if best is None else best.physical_slots, None if best is None else self._primary_selection_score(best),
                None if best is None else best.worst_regime_score,
                None if best is None else best.constraint_excess,
                self.experiment.checkpoint.best_metric, final_path, self.current_segment_latest_checkpoint,
                self.last_finite_checkpoint, tuple(map(float, self.agent.algorithm.dual_values.detach().cpu())),
                tuple(self.validations), tuple(self.segments), _artifact_paths(self.output_directory),
                segment.elapsed_seconds, self._elapsed_before_segment,
                self.agent.is_finite() and self.progress.valid_action_rate == 1.0
                and all(all(isfinite(value) for value in (
                    item.worst_regime_paired_return, item.macro_paired_return,
                    item.macro_positive_constraint_excess,
                )) for item in self.validations),
            )
            self.artifacts.event("final_summary", summary)
            write_run_summary(self.output_directory / "summary.json", summary)
            return summary
        except Exception as error:
            segment = self._finalize_segment("failed")
            failure = f"{type(error).__name__}: {error}"
            failure_record = {
                "schema_version": "isac-ssc-training-summary-v5",
                "artifact_schema_version": _ARTIFACT_SCHEMA,
                "method": self.experiment.method,
                "credit_assignment_schema": self.credit_assignment_schema,
                "training_seed": self.training_seed,
                "run_name": self.run_name, "parent_run_name": self.parent_run_name,
                "parent_checkpoint_path": self.parent_checkpoint_path,
                "requested_physical_slots": self.segment_requested_slots,
                "actual_physical_slots": segment.actual_physical_slots,
                "budget_overshoot_slots": segment.budget_overshoot_slots,
                "segment_start_physical_slot": self.segment_start_slot,
                "segment_target_physical_slot": self.segment_target_slot,
                "progress": self.progress, "segments": self.segments, "failure": failure,
                "last_finite_checkpoint": self.last_finite_checkpoint,
                "artifact_paths": _artifact_paths(self.output_directory),
                "segment_elapsed_seconds": segment.elapsed_seconds,
                "cumulative_elapsed_seconds": self._elapsed_before_segment,
            }
            self.artifacts.event("failure", failure_record)
            write_json(self.output_directory / "summary.json", failure_record)
            raise
        finally:
            self.artifacts.close()


class CommonTracePPOTrainer(JointCreditPPOTrainer):
    """Train CT-PPO with the shared trainer and strict method identity."""

    expected_method = COMMON_TRACE_METHOD