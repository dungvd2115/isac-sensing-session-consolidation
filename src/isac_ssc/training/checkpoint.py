"""Atomic finite checkpoints for constrained-PPO training and resume."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import torch

from isac_ssc.baselines.ppo_common_trace import CommonTracePPOAgent
from isac_ssc.baselines.ppo_joint_credit import JointCreditPPOAgent
from isac_ssc.utils.config import (
    COMMON_TRACE_METHOD, JOINT_CREDIT_METHOD,
    credit_assignment_schema as method_credit_assignment_schema,
)


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint is corrupt, non-finite or incompatible."""


_SCHEMA = "isac-ssc-training-checkpoint-v5"


def _method_credit_schema(method: str) -> str:
    try:
        return method_credit_assignment_schema(method)
    except ValueError as error:
        raise CheckpointValidationError("unsupported checkpoint method") from error


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    schema_version: str
    method: str
    credit_assignment_schema: str
    training_seed: int
    feature_schema_digest: str
    architecture_signature: str
    environment_semantic_digest: str
    validation_protocol_digest: str
    constraint_labels: tuple[str, ...]
    torch_version: str
    python_version: str

    @classmethod
    def current(
        cls, *, method: str, credit_assignment_schema: str, training_seed: int,
        feature_schema_digest: str, architecture_signature: str,
        environment_semantic_digest: str, validation_protocol_digest: str,
        constraint_labels: tuple[str, ...],
    ) -> CheckpointMetadata:
        if credit_assignment_schema != _method_credit_schema(method):
            raise CheckpointValidationError(
                "checkpoint credit-assignment schema does not match method"
            )
        return cls(
            _SCHEMA, method, credit_assignment_schema, training_seed,
            feature_schema_digest, architecture_signature,
            environment_semantic_digest, validation_protocol_digest, tuple(constraint_labels),
            str(torch.__version__), platform.python_version(),
        )


def _semantic_value(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _semantic_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return {"type": type(value).__qualname__, "value": _semantic_value(value.value)}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        items = [(_semantic_value(key), _semantic_value(item)) for key, item in value.items()]
        return {"mapping": sorted(items, key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))}
    if isinstance(value, (set, frozenset)):
        items = [_semantic_value(item) for item in value]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))}
    if isinstance(value, tuple):
        return {"tuple": [_semantic_value(item) for item in value]}
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise CheckpointValidationError("semantic checkpoint data must be finite")
        return value
    raise CheckpointValidationError(f"unsupported semantic checkpoint value: {type(value).__name__}")


def semantic_digest(value: Any) -> str:
    payload = json.dumps(_semantic_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().clone()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise CheckpointValidationError("checkpoint tensors must be finite")
        return tensor
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, float):
        if not isfinite(value):
            raise CheckpointValidationError("checkpoint state must be finite")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise CheckpointValidationError(f"unsupported checkpoint value: {type(value).__name__}")


def _load_payload(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointValidationError("checkpoint cannot be loaded safely") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA:
        raise CheckpointValidationError("unsupported checkpoint schema")
    return payload


def _metadata(payload: Mapping[str, Any]) -> CheckpointMetadata:
    try:
        metadata = CheckpointMetadata(**payload["metadata"])
    except (KeyError, TypeError) as error:
        raise CheckpointValidationError("checkpoint metadata is malformed") from error
    if metadata.schema_version != _SCHEMA or not metadata.constraint_labels:
        raise CheckpointValidationError("checkpoint metadata is incomplete")
    if metadata.credit_assignment_schema != _method_credit_schema(metadata.method):
        raise CheckpointValidationError(
            "checkpoint credit-assignment schema is incompatible with its method"
        )
    return metadata


def _normalizer_configuration(payload: Mapping[str, Any], *, required: bool) -> tuple[float, float] | None:
    values = payload.get("normalizer_configuration")
    if values is None and not required:
        return None
    if not isinstance(values, Mapping):
        raise CheckpointValidationError("checkpoint is missing policy normalization state")
    clip, epsilon = values.get("clip"), values.get("epsilon")
    if isinstance(clip, bool) or isinstance(epsilon, bool):
        raise CheckpointValidationError("checkpoint policy normalization state is invalid")
    try:
        clip, epsilon = float(clip), float(epsilon)
    except (TypeError, ValueError) as error:
        raise CheckpointValidationError("checkpoint policy normalization state is invalid") from error
    if not isfinite(clip) or not isfinite(epsilon) or clip <= 0.0 or epsilon <= 0.0:
        raise CheckpointValidationError("checkpoint policy normalization state is invalid")
    return clip, epsilon


PPOAgent = JointCreditPPOAgent | CommonTracePPOAgent


def _agent_identity(agent: PPOAgent) -> tuple[str, str]:
    if isinstance(agent, JointCreditPPOAgent):
        method = JOINT_CREDIT_METHOD
    elif isinstance(agent, CommonTracePPOAgent):
        method = COMMON_TRACE_METHOD
    else:
        raise CheckpointValidationError("unsupported PPO agent type")
    return method, _method_credit_schema(method)


def _restore_normalizer_configuration(
    payload: Mapping[str, Any], agent: PPOAgent, *, required: bool,
) -> None:
    values = _normalizer_configuration(payload, required=required)
    if values is not None:
        agent.normalizer.clip, agent.normalizer.epsilon = values


def read_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    return _metadata(_load_payload(path))


def read_checkpoint_context(path: str | Path) -> tuple[CheckpointMetadata, dict[str, Any]]:
    payload = _load_payload(path)
    state = payload.get("run_state", {})
    if not isinstance(state, dict):
        raise CheckpointValidationError("checkpoint run state is malformed")
    return _metadata(payload), state


def save_checkpoint(
    path: str | Path, agent: PPOAgent, metadata: CheckpointMetadata,
    state: Mapping[str, Any],
) -> str:
    if not agent.is_finite():
        raise CheckpointValidationError("refusing to save a non-finite training state")
    expected_method, expected_credit = _agent_identity(agent)
    if metadata.method != expected_method or metadata.credit_assignment_schema != expected_credit:
        raise CheckpointValidationError("checkpoint metadata does not match the agent method")
    if metadata.feature_schema_digest != agent.model.layout.schema_digest:
        raise CheckpointValidationError("checkpoint feature layout does not match the agent")
    if len(metadata.constraint_labels) != agent.algorithm.constraint_count:
        raise CheckpointValidationError("checkpoint constraint labels do not match the agent")
    payload = {
        "schema_version": _SCHEMA, "metadata": asdict(metadata),
        "model_state": _cpu_tree(agent.model.state_dict()),
        "optimizer_state": _cpu_tree(agent.algorithm.optimizer.state_dict()),
        "dual_values": _cpu_tree(agent.algorithm.dual_values),
        "normalizer_state": _cpu_tree(agent.normalizer.state_dict()),
        "normalizer_configuration": {"clip": agent.normalizer.clip, "epsilon": agent.normalizer.epsilon},
        "torch_rng_state": torch.random.get_rng_state().clone(),
        "action_generator_state": agent.action_generator.get_state().clone(),
        "minibatch_generator_state": agent.minibatch_generator.get_state().clone(),
        "optimizer_step_count": agent.algorithm.optimizer_step_count,
        "run_state": _cpu_tree(dict(state)),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return checkpoint_sha256(destination)


def load_policy_checkpoint(
    path: str | Path, agent: PPOAgent, *, expected_method: str,
) -> tuple[CheckpointMetadata, dict[str, Any]]:
    """Restore only policy inference state for read-only deterministic evaluation."""
    payload = _load_payload(path)
    metadata = _metadata(payload)
    expected_identity = (expected_method, _method_credit_schema(expected_method))
    if (metadata.method, metadata.credit_assignment_schema) != expected_identity:
        raise CheckpointValidationError(
            f"checkpoint method {metadata.method!r} does not match {expected_method!r}"
        )
    if _agent_identity(agent) != expected_identity:
        raise CheckpointValidationError("evaluation agent does not match checkpoint method")
    try:
        _restore_normalizer_configuration(payload, agent, required=True)
        agent.model.load_state_dict(payload["model_state"], strict=True)
        agent.normalizer.load_state_dict(payload["normalizer_state"])
    except CheckpointValidationError:
        raise
    except Exception as error:
        raise CheckpointValidationError("checkpoint policy state cannot be restored") from error
    tensors = (*agent.model.parameters(), *agent.model.buffers())
    if not all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in tensors):
        raise CheckpointValidationError("checkpoint restored a non-finite policy state")
    if not agent.normalizer.is_finite():
        raise CheckpointValidationError("checkpoint restored a non-finite normalizer state")
    state = payload.get("run_state", {})
    return metadata, state if isinstance(state, dict) else {}


def load_checkpoint(
    path: str | Path, agent: PPOAgent, expected: CheckpointMetadata,
) -> tuple[CheckpointMetadata, dict[str, Any]]:
    payload = _load_payload(path)
    metadata = _metadata(payload)
    if _agent_identity(agent) != (expected.method, expected.credit_assignment_schema):
        raise CheckpointValidationError("current agent does not match expected checkpoint method")
    compatibility = {
        "method": metadata.method == expected.method,
        "credit assignment": metadata.credit_assignment_schema == expected.credit_assignment_schema,
        "training seed": metadata.training_seed == expected.training_seed,
        "feature schema": metadata.feature_schema_digest == expected.feature_schema_digest,
        "architecture": metadata.architecture_signature == expected.architecture_signature,
        "environment semantics": metadata.environment_semantic_digest == expected.environment_semantic_digest,
        "constraint layout": metadata.constraint_labels == expected.constraint_labels,
    }
    incompatible = [name for name, matches in compatibility.items() if not matches]
    if incompatible:
        raise CheckpointValidationError("checkpoint is incompatible with the current " + ", ".join(incompatible))
    try:
        _restore_normalizer_configuration(payload, agent, required=False)
        agent.model.load_state_dict(payload["model_state"], strict=True)
        agent.algorithm.optimizer.load_state_dict(payload["optimizer_state"])
        agent.algorithm.dual_values.copy_(payload["dual_values"].to(agent.device))
        agent.normalizer.load_state_dict(payload["normalizer_state"])
        torch.random.set_rng_state(payload["torch_rng_state"])
        agent.action_generator.set_state(payload["action_generator_state"])
        agent.minibatch_generator.set_state(payload["minibatch_generator_state"])
        agent.algorithm.optimizer_step_count = int(payload["optimizer_step_count"])
    except Exception as error:
        raise CheckpointValidationError("checkpoint state cannot be restored") from error
    if not agent.is_finite():
        raise CheckpointValidationError("checkpoint restored a non-finite state")
    state = payload.get("run_state", {})
    if not isinstance(state, dict):
        raise CheckpointValidationError("checkpoint run state is malformed")
    _cpu_tree(state)
    return metadata, state