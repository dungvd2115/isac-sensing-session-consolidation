"""Finite JSONL, CSV and summary artifacts for training and reporting."""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


class TrainingLogError(ValueError):
    """Raised when a record cannot be serialized as finite training data."""


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: str
    artifact_schema_version: str
    method: str
    credit_assignment_schema: str
    training_seed: int
    run_name: str
    feature_schema_digest: str
    architecture_signature: str
    environment_semantic_digest: str
    validation_protocol_digest: str
    selection_protocol_digest: str
    constraint_labels: tuple[str, ...]
    parameter_count: int
    trainable_parameter_count: int
    device: str
    python_version: str
    torch_version: str
    hardware: str
    started_unix_s: float


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TrainingLogError("training artifacts reject NaN and infinity")
        return value
    raise TrainingLogError(f"unsupported training artifact value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: str | Path, value: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(destination)
    return destination.as_posix()


class JsonlTrainingLogger:
    def __init__(self, path: str | Path, *, append: bool = False, flush_every_records: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a" if append else "w", encoding="utf-8", newline="\n")
        self.flush_every_records = max(1, int(flush_every_records))
        self._records = 0

    def log(self, event: str, payload: Any) -> None:
        record = {"event": event, "payload": json_value(payload), "wall_time_unix_s": time.time()}
        self._handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
        self._records += 1
        if self._records % self.flush_every_records == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> JsonlTrainingLogger:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class CsvTable:
    def __init__(self, path: str | Path, fieldnames: Sequence[str], *, append: bool = False, flush_every_records: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = tuple(fieldnames)
        self.migrated_from: str | None = None
        existed = self.path.exists() and self.path.stat().st_size > 0
        self.created = not existed
        if append and existed:
            with self.path.open(encoding="utf-8", newline="") as handle:
                prior_fields = tuple(csv.DictReader(handle).fieldnames or ())
            if prior_fields != self.fieldnames:
                legacy = self.path.with_name(f"{self.path.stem}.legacy{self.path.suffix}")
                counter = 1
                while legacy.exists():
                    legacy = self.path.with_name(f"{self.path.stem}.legacy_{counter}{self.path.suffix}")
                    counter += 1
                shutil.copy2(self.path, legacy)
                self.migrated_from = legacy.as_posix()
                with self.path.open("w", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore").writeheader()
                self.created = True
        self._handle = self.path.open("a" if append else "w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if not append or not existed:
            self.writer.writeheader()
        self.flush_every_records = max(1, int(flush_every_records))
        self._records = 0

    def write(self, row: Mapping[str, Any]) -> None:
        values = {}
        for key in self.writer.fieldnames:
            converted = json_value(row.get(key)) if key in row else None
            values[key] = json.dumps(converted, ensure_ascii=False, separators=(",", ":")) if isinstance(converted, (dict, list)) else converted
        self.writer.writerow(values)
        self._records += 1
        if self._records % self.flush_every_records == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


ROLL_OUT_FIELDS = (
    "rollout_index", "start_slot", "end_slot", "requested_rollout_slots", "actual_rollout_slots",
    "rollout_overshoot_slots", "episodes", "focal_decisions", "progress_fraction", "remaining_requested_slots",
    "reward_total", "reward_per_slot", "reward_per_episode", "completed_value_total", "arrived_request_value_total",
    "normalized_completed_value", "completed_value_per_slot", "completed_value_per_episode",
    "sensing_resource_cost_total", "sensing_resource_cost_per_slot",
    "sensing_resource_cost_per_episode", "sensing_bandwidth_hz_slot_sum", "sensing_bandwidth_hz_mean",
    "sensing_bandwidth_hz_max", "sensing_power_w_slot_sum", "sensing_power_w_mean", "sensing_power_w_max",
    "slots_with_session_update", "session_update_count", "tracking_prediction_count", "post_slot_active_session_count_mean",
    "post_slot_active_session_count_max", "arrived", "accepted", "completed", "rejected", "expired", "failed",
    "valid_outputs", "first_violations", "created_sessions", "acceptance_ratio", "completion_ratio",
    "rejection_ratio", "requests_served_per_created_session", "network_mean_user_shortfall",
    "fraction_users_within_budget", "merge_actions", "create_actions", "defer_actions",
    "reject_actions", "merge_action_rate", "create_action_rate", "defer_action_rate", "reject_action_rate",
    "valid_action_rate", "transitions", "epochs_completed", "minibatches_completed", "optimizer_steps",
    "early_stopped_for_kl", "total_loss", "actor_loss", "reward_surrogate",
    "type_reward_surrogate", "session_reward_surrogate", "profile_reward_surrogate",
    "reward_value_loss", "constraint_value_loss", "tenant_value_loss", "communication_value_loss",
    "type_reward_value_loss", "type_constraint_value_loss",
    "session_reward_value_loss", "session_constraint_value_loss",
    "entropy", "approximate_kl", "max_minibatch_approximate_kl", "clip_fraction",
    "type_clip_fraction", "session_clip_fraction", "profile_clip_fraction",
    "joint_ratio_p01", "joint_ratio_p05", "joint_ratio_p50", "joint_ratio_p95", "joint_ratio_p99",
    "minimum_joint_ratio", "maximum_joint_ratio", "nonfinite_joint_ratio_count",
    "gradient_norm", "max_gradient_norm",
    "actor_gradient_norm", "max_actor_gradient_norm",
    "critic_gradient_norm", "max_critic_gradient_norm",
    "type_prefix_gradient_norm", "max_type_prefix_gradient_norm",
    "session_prefix_gradient_norm", "max_session_prefix_gradient_norm",
    "prefix_gradient_norm", "max_prefix_gradient_norm",
    "merge_transition_count", "profile_transition_count",
    "single_session_merge_transition_count", "multi_session_merge_transition_count",
    "session_prefix_reward_target_variance", "session_prefix_constraint_target_variance",
    "type_normalized_reward_advantage_mean",
    "session_normalized_reward_advantage_mean",
    "profile_normalized_reward_advantage_mean",
    "type_positive_reward_advantage_fraction",
    "session_positive_reward_advantage_fraction",
    "profile_positive_reward_advantage_fraction",
    "reward_advantage_scale", "constraint_advantage_scale_min",
    "constraint_advantage_scale_mean", "constraint_advantage_scale_max",
    "reward_return_scale", "constraint_return_scale_min",
    "constraint_return_scale_mean", "constraint_return_scale_max",
    "normalizer_frozen", "normalizer_request_count",
    "normalizer_session_count", "normalizer_global_count",
    "worst_constraint_label", "worst_constraint_residual",
    "positive_constraint_excess", "dual_min", "dual_mean", "dual_max", "learning_rate", "elapsed_seconds",
    "cumulative_segment_elapsed_seconds", "slots_per_second", "eta_seconds",
)
EPISODE_FIELDS = (
    "rollout_index", "episode_index", "trace_id", "root_seed", "arrival_regime", "physical_slots",
    "focal_decisions", "reward_total", "reward_per_slot", "completed_value_total", "arrived_request_value_total",
    "normalized_completed_value", "completed_value_per_slot", "sensing_resource_cost_total",
    "sensing_resource_cost_per_slot", "sensing_bandwidth_hz_slot_sum",
    "sensing_bandwidth_hz_mean", "sensing_bandwidth_hz_max", "sensing_power_w_slot_sum", "sensing_power_w_mean",
    "sensing_power_w_max", "slots_with_session_update", "session_update_count", "tracking_prediction_count",
    "post_slot_active_session_count_mean", "post_slot_active_session_count_max", "arrived", "accepted", "completed", "rejected",
    "expired", "failed", "valid_outputs", "first_violations", "created_sessions", "acceptance_ratio",
    "completion_ratio", "rejection_ratio", "requests_served_per_created_session", "network_mean_user_shortfall",
    "fraction_users_within_budget", "merge_actions", "create_actions", "defer_actions", "reject_actions",
    "merge_action_rate", "create_action_rate", "defer_action_rate", "reject_action_rate",
    "valid_action_rate", "positive_constraint_excess",
)
TRAIN_CONSTRAINT_FIELDS = (
    "rollout_index", "start_slot", "end_slot", "family", "constraint_label", "entity_id", "residual_total",
    "mean_episode_residual", "positive_excess", "constraint_surrogate",
    "type_constraint_surrogate", "session_constraint_surrogate", "profile_constraint_surrogate",
    "actor_dual_used", "raw_dual_before", "raw_dual_after",
    "advantage_scale", "return_scale",
)
TENANT_FIELDS = (
    "record_type", "rollout_index", "episode_index", "physical_slot", "policy", "trace_id", "root_seed",
    "arrival_regime", "replicate", "tenant_id", "sla_violation_budget", "arrived", "accepted", "completed",
    "rejected", "expired", "failed", "first_violated", "acceptance_ratio", "completion_ratio",
    "violation_rate", "residual_total", "positive_residual",
)
COMMUNICATION_FIELDS = (
    "record_type", "rollout_index", "episode_index", "physical_slot", "policy", "trace_id", "root_seed",
    "arrival_regime", "replicate", "user_id", "normalized_shortfall_budget", "active_demand_slots",
    "demand_bit_per_s_slot_sum", "mean_active_demand_bit_per_s", "allocated_bandwidth_hz_slot_sum",
    "mean_active_allocated_bandwidth_hz", "allocated_power_w_slot_sum", "mean_active_allocated_power_w",
    "achievable_rate_bit_per_s_slot_sum", "mean_active_achievable_rate_bit_per_s", "served_rate_bit_per_s_slot_sum",
    "mean_active_served_rate_bit_per_s", "normalized_shortfall_sum", "mean_normalized_shortfall", "residual_total",
    "positive_residual",
)
VALIDATION_FIELDS = (
    "scheduled_physical_slot", "actual_physical_slot", "interval_overshoot_slots", "arrival_regime", "is_overall",
    "is_worst_regime", "policy_episode_count", "random_valid_episode_count", "policy_mean_return",
    "policy_std_return", "random_valid_mean_return", "random_valid_std_return", "paired_return_difference",
    "policy_mean_completed_value", "policy_mean_normalized_completed_value", "policy_mean_sensing_resource_cost",
    "policy_mean_positive_constraint_excess", "policy_mean_reward_per_slot", "policy_mean_completed_value_per_slot",
    "policy_mean_sensing_resource_cost_per_slot", "policy_mean_network_user_shortfall",
    "policy_mean_fraction_users_within_budget", "random_valid_mean_reward_per_slot",
    "random_valid_mean_normalized_completed_value", "random_valid_mean_network_user_shortfall",
    "random_valid_mean_fraction_users_within_budget", "policy_valid_action_rate",
    "random_valid_valid_action_rate", "best_metric",
    "best_score", "is_best",
)
VALIDATION_TRACE_FIELDS = (
    "policy", "physical_slot", "replicate",
    *(field for field in EPISODE_FIELDS if field != "rollout_index"),
)
VALIDATION_CONSTRAINT_FIELDS = (
    "policy", "physical_slot", "trace_id", "root_seed", "arrival_regime", "replicate", "episode_index",
    "family", "constraint_label", "entity_id", "residual_total", "positive_excess",
)
CHECKPOINT_FIELDS = (
    "path", "type", "scheduled_physical_slot", "actual_physical_slot", "interval_overshoot_slots", "metric",
    "worst_regime_score", "macro_score", "constraint_excess", "rank_at_save", "sha256", "created_unix_s",
)
SEGMENT_FIELDS = (
    "segment_index", "resume_checkpoint", "start_physical_slot", "requested_physical_slots", "target_physical_slot",
    "completed_physical_slot", "actual_physical_slots", "budget_overshoot_slots", "training_regimes",
    "rollout_target_physical_slots", "learning_rate_schedule",
    "learning_rate_schedule_horizon_physical_slots", "algorithm_semantic_digest",
    "validation_protocol_digest", "validation_enabled", "validation_interval_physical_slots",
    "checkpoint_interval_physical_slots", "started_unix_s", "finished_unix_s", "elapsed_seconds",
    "starting_learning_rate", "ending_learning_rate", "status",
)


class TrainingArtifacts:
    def __init__(self, directory: str | Path, *, append: bool, jsonl: bool, csv_enabled: bool, flush_every_records: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.directory / "training.jsonl"
        self.jsonl_created = bool(jsonl and (not jsonl_path.exists() or jsonl_path.stat().st_size == 0))
        self.jsonl = JsonlTrainingLogger(jsonl_path, append=append, flush_every_records=flush_every_records) if jsonl else None
        self.tables: dict[str, CsvTable] = {}
        if csv_enabled:
            specifications = {
                "rollouts": ("train_rollouts.csv", ROLL_OUT_FIELDS),
                "episodes": ("train_episodes.csv", EPISODE_FIELDS),
                "train_constraints": ("train_constraints.csv", TRAIN_CONSTRAINT_FIELDS),
                "train_tenants": ("train_tenants.csv", TENANT_FIELDS),
                "train_communication_users": ("train_communication_users.csv", COMMUNICATION_FIELDS),
                "validation": ("validation_summary.csv", VALIDATION_FIELDS),
                "validation_traces": ("validation_traces.csv", VALIDATION_TRACE_FIELDS),
                "validation_constraints": ("validation_constraints.csv", VALIDATION_CONSTRAINT_FIELDS),
                "validation_tenants": ("validation_tenants.csv", TENANT_FIELDS),
                "validation_communication_users": ("validation_communication_users.csv", COMMUNICATION_FIELDS),
                "checkpoints": ("checkpoint_index.csv", CHECKPOINT_FIELDS),
                "segments": ("resume_segments.csv", SEGMENT_FIELDS),
            }
            self.tables = {
                name: CsvTable(self.directory / filename, fieldnames, append=append, flush_every_records=flush_every_records)
                for name, (filename, fieldnames) in specifications.items()
            }
        self.created_tables = {name for name, table in self.tables.items() if table.created}
        self.schema_migrations = {
            name: table.migrated_from for name, table in self.tables.items() if table.migrated_from is not None
        }

    def event(self, name: str, payload: Any) -> None:
        if self.jsonl:
            self.jsonl.log(name, payload)

    def row(self, table: str, payload: Mapping[str, Any] | Any) -> None:
        if table in self.tables:
            self.tables[table].write(asdict(payload) if is_dataclass(payload) else payload)

    def close(self) -> None:
        if self.jsonl:
            self.jsonl.close()
        for table in self.tables.values():
            table.close()


def write_run_summary(path: str | Path, summary: Any) -> str:
    return write_json(path, summary)