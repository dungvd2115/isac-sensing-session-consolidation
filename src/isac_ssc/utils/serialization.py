"""Readable JSON persistence for primitive traces."""

from __future__ import annotations

from dataclasses import asdict
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, TypeVar

from isac_ssc.core.entities import EntityId, Task
from isac_ssc.envs.dynamics import (
    CommunicationSlotPrimitive, CommunicationTransitionInnovation, ParentEventPrimitive,
    PrimitiveTrace, RequestPrimitiveDescriptor, TargetSlotPrimitive, TargetTransitionInnovation,
)
from isac_ssc.utils.config import CanonicalConfig


class SerializationError(ValueError):
    """Raised when an external trace file is not usable."""


T = TypeVar("T")


def trace_to_dict(trace: PrimitiveTrace) -> dict[str, Any]:
    """Convert a trace to a plain JSON-compatible mapping."""
    return asdict(trace)


def _number(value: object, name: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise SerializationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and (number <= minimum if strict else number < minimum):
        operator = ">" if strict else ">="
        raise SerializationError(f"{name} must be {operator} {minimum}")
    return number


def _integer(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SerializationError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or maximum is not None and number > maximum:
        raise SerializationError(f"{name} is outside its valid range")
    return number


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SerializationError(f"{name} must be boolean")
    return value


def _entity_id(value: object, name: str) -> EntityId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SerializationError(f"{name} must be a string or integer identifier")
    if isinstance(value, str) and not value.strip() or isinstance(value, int) and value < 0:
        raise SerializationError(f"{name} is invalid")
    return value


def _identifier_key(value: EntityId) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _vector(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SerializationError(f"{name} must contain two coordinates")
    return _number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]")


def _records(values: object, name: str, decoder: Callable[[dict[str, Any]], T]) -> tuple[T, ...]:
    if not isinstance(values, list):
        raise SerializationError(f"{name} must be a JSON array")
    records = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise SerializationError(f"{name}[{index}] must be a JSON object")
        try:
            records.append(decoder(value))
        except SerializationError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise SerializationError(f"{name}[{index}] is malformed") from error
    return tuple(records)


def _target_state(item: dict[str, Any]) -> TargetSlotPrimitive:
    return TargetSlotPrimitive(
        _integer(item["slot"], "target.slot"), _entity_id(item["target_id"], "target.target_id"),
        _vector(item["position_m"], "target.position_m"),
        _vector(item["velocity_m_per_s"], "target.velocity_m_per_s"),
        _number(item["shadowing_db"], "target.shadowing_db"),
        _number(item["fading_real"], "target.fading_real"),
        _number(item["fading_imag"], "target.fading_imag"),
        _number(item["rcs_dbsm"], "target.rcs_dbsm"),
    )


def _target_innovation(item: dict[str, Any]) -> TargetTransitionInnovation:
    return TargetTransitionInnovation(
        _integer(item["slot"], "target_innovation.slot", minimum=1),
        _entity_id(item["target_id"], "target_innovation.target_id"),
        _vector(item["acceleration_m_per_s2"], "target_innovation.acceleration_m_per_s2"),
        _number(item["shadowing_innovation_db"], "target_innovation.shadowing_innovation_db"),
        _number(item["fading_innovation_real"], "target_innovation.fading_innovation_real"),
        _number(item["fading_innovation_imag"], "target_innovation.fading_innovation_imag"),
        _number(item["rcs_innovation_dbsm"], "target_innovation.rcs_innovation_dbsm"),
    )


def _communication_state(item: dict[str, Any]) -> CommunicationSlotPrimitive:
    return CommunicationSlotPrimitive(
        _integer(item["slot"], "communication.slot"),
        _entity_id(item["user_id"], "communication.user_id"),
        _vector(item["position_m"], "communication.position_m"),
        _vector(item["velocity_m_per_s"], "communication.velocity_m_per_s"),
        _boolean(item["traffic_on"], "communication.traffic_on"),
        _number(item["demand_bit_per_s"], "communication.demand_bit_per_s", minimum=0.0),
        _number(item["shadowing_db"], "communication.shadowing_db"),
        _number(item["fading_real"], "communication.fading_real"),
        _number(item["fading_imag"], "communication.fading_imag"),
    )


def _communication_innovation(item: dict[str, Any]) -> CommunicationTransitionInnovation:
    demand = item.get("demand_standard_normal")
    return CommunicationTransitionInnovation(
        _integer(item["slot"], "communication_innovation.slot", minimum=1),
        _entity_id(item["user_id"], "communication_innovation.user_id"),
        _vector(item["acceleration_m_per_s2"], "communication_innovation.acceleration_m_per_s2"),
        _number(item["shadowing_innovation_db"], "communication_innovation.shadowing_innovation_db"),
        _number(item["fading_innovation_real"], "communication_innovation.fading_innovation_real"),
        _number(item["fading_innovation_imag"], "communication_innovation.fading_innovation_imag"),
        _number(item["traffic_transition_uniform"], "communication_innovation.traffic_transition_uniform"),
        None if demand is None else _number(demand, "communication_innovation.demand_standard_normal"),
    )


def _parent(item: dict[str, Any]) -> ParentEventPrimitive:
    return ParentEventPrimitive(
        _entity_id(item["parent_id"], "parent.parent_id"),
        _integer(item["sampled_slot"], "parent.sampled_slot"),
        _entity_id(item["target_id"], "parent.target_id"), Task.parse(item["task"]),
        _integer(item["child_count"], "parent.child_count", minimum=1),
    )


def _descriptor(item: dict[str, Any]) -> RequestPrimitiveDescriptor:
    child_index = item.get("child_index")
    return RequestPrimitiveDescriptor(
        _entity_id(item["request_id"], "request.request_id"), item["source_regime"],
        _integer(item["sampled_slot"], "request.sampled_slot"),
        _integer(item["arrival_slot"], "request.arrival_slot"),
        _entity_id(item["tenant_id"], "request.tenant_id"),
        _entity_id(item["target_id"], "request.target_id"), Task.parse(item["task"]),
        _number(item["aoi_radius_m"], "request.aoi_radius_m", minimum=0.0, strict=True),
        _vector(item["aoi_displacement_m"], "request.aoi_displacement_m"),
        _integer(item["latest_start_slack_slots"], "request.latest_start_slack_slots"),
        _integer(item["valid_output_interval_slots"], "request.valid_output_interval_slots", minimum=1),
        _number(item["quality_threshold"], "request.quality_threshold", minimum=0.0, strict=True),
        _number(item["completion_value"], "request.completion_value", minimum=0.0, strict=True),
        _boolean(item["merge_permission"], "request.merge_permission"),
        None if item.get("parent_id") is None else _entity_id(item["parent_id"], "request.parent_id"),
        None if child_index is None else _integer(child_index, "request.child_index"),
        _boolean(item.get("horizon_omitted", False), "request.horizon_omitted"),
    )


def _complete_state_grid(
    records: tuple[Any, ...], id_field: str, horizon: int, count: int, name: str,
) -> set[tuple[str, str]]:
    identifiers = {_identifier_key(getattr(item, id_field)) for item in records}
    expected = {(slot, identifier) for slot in range(horizon) for identifier in identifiers}
    actual = {(item.slot, _identifier_key(getattr(item, id_field))) for item in records}
    if len(identifiers) != count or actual != expected:
        raise SerializationError(f"{name} state coverage is incomplete")
    return identifiers


def _validate_trace_structure(trace: PrimitiveTrace, config: CanonicalConfig) -> None:
    if trace.horizon_slots != config.system["horizon_slots"]:
        raise SerializationError("trace horizon does not match the environment")
    if trace.arrival_regime not in config.trace_generation["registered_arrival_regimes"]:
        raise SerializationError("trace arrival regime is not registered")
    configured_tenants = tuple(tenant.tenant_id for tenant in config.tenants)
    if trace.tenant_ids != configured_tenants:
        raise SerializationError("trace tenants do not match the environment")
    if len(trace.tenant_authorization_matrix) != len(trace.tenant_ids) or any(
        len(row) != len(trace.tenant_ids) or any(type(value) is not bool for value in row)
        for row in trace.tenant_authorization_matrix
    ):
        raise SerializationError("tenant authorization matrix must be square and boolean")

    target_ids = _complete_state_grid(
        trace.target_states, "target_id", trace.horizon_slots,
        config.population["physical_targets"], "target",
    )
    user_ids = _complete_state_grid(
        trace.communication_states, "user_id", trace.horizon_slots,
        config.population["communication_users"], "communication",
    )
    if any(not 1 <= item.slot < trace.horizon_slots or _identifier_key(item.target_id) not in target_ids
           for item in trace.target_innovations):
        raise SerializationError("target innovation reference is invalid")
    if any(not 1 <= item.slot < trace.horizon_slots or _identifier_key(item.user_id) not in user_ids
           for item in trace.communication_innovations):
        raise SerializationError("communication innovation reference is invalid")

    parent_ids = {_identifier_key(item.parent_id) for item in trace.parent_events}
    tenant_ids = {_identifier_key(item) for item in trace.tenant_ids}
    request_ids = set()
    for descriptor in trace.request_descriptors:
        request_id = _identifier_key(descriptor.request_id)
        if request_id in request_ids:
            raise SerializationError("request identifiers must be unique")
        request_ids.add(request_id)
        if _identifier_key(descriptor.tenant_id) not in tenant_ids:
            raise SerializationError("request references an unknown tenant")
        if _identifier_key(descriptor.target_id) not in target_ids:
            raise SerializationError("request references an unknown target")
        if (not 0 <= descriptor.sampled_slot < trace.horizon_slots
                or descriptor.arrival_slot < descriptor.sampled_slot):
            raise SerializationError("request slots are invalid")
        if descriptor.source_regime not in {"independent", "clustered"}:
            raise SerializationError("request source regime is invalid")
        if (descriptor.source_regime == "clustered"
                and _identifier_key(descriptor.parent_id) not in parent_ids):
            raise SerializationError("clustered request references an unknown parent")


def trace_from_dict(value: object, config: CanonicalConfig) -> PrimitiveTrace:
    """Construct and validate a primitive trace at the JSON boundary."""
    if not isinstance(value, dict):
        raise SerializationError("trace must be a JSON object")
    try:
        tenant_ids = tuple(_entity_id(item, "tenant_id") for item in value["tenant_ids"])
        matrix = tuple(tuple(_boolean(item, "tenant authorization") for item in row)
                       for row in value["tenant_authorization_matrix"])
        trace = PrimitiveTrace(
            value["trace_id"], _integer(value["root_seed"], "root_seed", maximum=2**64-1),
            value["arrival_regime"], _integer(value["horizon_slots"], "horizon_slots", minimum=1),
            tenant_ids, matrix,
            _records(value["target_states"], "target_states", _target_state),
            _records(value["target_innovations"], "target_innovations", _target_innovation),
            _records(value["communication_states"], "communication_states", _communication_state),
            _records(
                value["communication_innovations"], "communication_innovations",
                _communication_innovation,
            ),
            _records(value.get("parent_events", []), "parent_events", _parent),
            _records(value["request_descriptors"], "request_descriptors", _descriptor),
        )
    except SerializationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SerializationError("trace JSON is missing required primitive data") from error
    if not isinstance(trace.trace_id, str) or not trace.trace_id:
        raise SerializationError("trace_id must be a non-empty string")
    if not isinstance(trace.arrival_regime, str):
        raise SerializationError("arrival_regime must be a string")
    _validate_trace_structure(trace, config)
    return trace


def deserialize_trace_bytes(data: bytes, config: CanonicalConfig) -> PrimitiveTrace:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SerializationError("trace file is not valid UTF-8 JSON") from error
    return trace_from_dict(value, config)


def serialize_trace(trace: PrimitiveTrace, path: str | Path) -> None:
    """Save a trusted trace as readable UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = json.dumps(trace_to_dict(trace), ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    except (TypeError, ValueError) as error:
        raise SerializationError("trace contains values that JSON cannot represent") from error
    destination.write_text(content, encoding="utf-8")


def load_trace(path: str | Path, config: CanonicalConfig) -> PrimitiveTrace:
    return deserialize_trace_bytes(Path(path).read_bytes(), config)