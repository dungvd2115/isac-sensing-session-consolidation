"""Immutable scientific entities for the canonical ISAC-SSC model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping, TypeAlias

EntityId: TypeAlias = str | int
Vector2: TypeAlias = tuple[float, float]
Matrix4: TypeAlias = tuple[tuple[float, float, float, float], ...]


class EntityValidationError(ValueError):
    """Raised when an entity violates the scientific model."""


class Task(StrEnum):
    DETECTION = "detection"
    LOCALIZATION = "localization"
    TRACKING = "tracking"

    @classmethod
    def parse(cls, value: str | Task) -> Task:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as error:
            raise EntityValidationError(f"unsupported sensing task: {value!r}") from error


TaskDurationMap: TypeAlias = Mapping[Task | str, int]


class RequestState(StrEnum):
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    REJECTED = "rejected"


_TASK_OUTPUTS: Mapping[Task, frozenset[Task]] = MappingProxyType({
    Task.DETECTION: frozenset({Task.DETECTION}),
    Task.LOCALIZATION: frozenset({Task.DETECTION, Task.LOCALIZATION}),
    Task.TRACKING: frozenset({Task.DETECTION, Task.LOCALIZATION, Task.TRACKING}),
})
_TERMINAL_STATES = frozenset({
    RequestState.COMPLETED, RequestState.FAILED, RequestState.EXPIRED, RequestState.REJECTED,
})
_ALLOWED_TRANSITIONS: Mapping[RequestState, frozenset[RequestState]] = MappingProxyType({
    RequestState.WAITING: frozenset({RequestState.ACTIVE, RequestState.EXPIRED, RequestState.REJECTED}),
    RequestState.ACTIVE: frozenset({RequestState.COMPLETED, RequestState.FAILED}),
    RequestState.COMPLETED: frozenset(),
    RequestState.FAILED: frozenset(),
    RequestState.EXPIRED: frozenset(),
    RequestState.REJECTED: frozenset(),
})


def task_outputs(task: Task | str) -> frozenset[Task]:
    return _TASK_OUTPUTS[Task.parse(task)]


def task_service_duration_slots(task: Task | str, durations: TaskDurationMap) -> int:
    parsed = Task.parse(task)
    try:
        value = durations[parsed] if parsed in durations else durations[parsed.value]
    except KeyError as error:
        raise EntityValidationError(f"missing service duration for task {parsed.value}") from error
    return _positive_int(value, f"service_duration_slots[{parsed.value}]")


def _finite_number(
    value: object, name: str, *, minimum: float | None = None, strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise EntityValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and (number <= minimum if strict else number < minimum):
        operator = ">" if strict else ">="
        raise EntityValidationError(f"{name} must be {operator} {minimum}")
    return number


def _slot(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EntityValidationError(f"{name} must be a non-negative integer slot")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EntityValidationError(f"{name} must be a positive integer")
    return value


def _entity_id(value: object, name: str) -> EntityId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EntityValidationError(f"{name} must be a string or integer identifier")
    if isinstance(value, str) and not value.strip() or isinstance(value, int) and value < 0:
        raise EntityValidationError(f"{name} is invalid")
    return value


def _vector2(value: Iterable[float], name: str) -> Vector2:
    try:
        vector = tuple(value)
    except TypeError as error:
        raise EntityValidationError(f"{name} must contain two finite coordinates") from error
    if len(vector) != 2:
        raise EntityValidationError(f"{name} must contain two coordinates")
    return _finite_number(vector[0], f"{name}[0]"), _finite_number(vector[1], f"{name}[1]")


def _matrix4(value: Iterable[Iterable[float]], name: str) -> Matrix4:
    try:
        rows = tuple(tuple(row) for row in value)
    except TypeError as error:
        raise EntityValidationError(f"{name} must be a finite 4x4 matrix") from error
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise EntityValidationError(f"{name} must be a 4x4 matrix")
    return tuple(
        tuple(_finite_number(cell, f"{name}[{row},{column}]") for column, cell in enumerate(values))
        for row, values in enumerate(rows)
    )


def _updated(instance: object, **changes: object):
    """Copy a validated frozen entity without rerunning constructor validation."""
    result = object.__new__(type(instance))
    for field in fields(instance):
        object.__setattr__(result, field.name, changes.get(field.name, getattr(instance, field.name)))
    return result


@dataclass(frozen=True, slots=True)
class DiskAOI:
    center_m: Vector2
    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_m", _vector2(self.center_m, "center_m"))
        object.__setattr__(self, "radius_m", _finite_number(
            self.radius_m, "radius_m", minimum=0.0, strict=True,
        ))


@dataclass(frozen=True, slots=True)
class Tenant:
    tenant_id: EntityId
    permitted_tasks: frozenset[Task]
    sla_violation_budget: float
    authorization_row: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _entity_id(self.tenant_id, "tenant_id"))
        tasks = frozenset(Task.parse(task) for task in self.permitted_tasks)
        if not tasks:
            raise EntityValidationError("permitted_tasks must not be empty")
        object.__setattr__(self, "permitted_tasks", tasks)
        budget = _finite_number(self.sla_violation_budget, "sla_violation_budget", minimum=0.0)
        if budget > 1.0:
            raise EntityValidationError("sla_violation_budget must be in [0, 1]")
        object.__setattr__(self, "sla_violation_budget", budget)
        row = tuple(self.authorization_row)
        if not row or any(type(flag) is not bool for flag in row):
            raise EntityValidationError("authorization_row must be a non-empty boolean tuple")
        object.__setattr__(self, "authorization_row", row)

    def permits(self, task: Task | str) -> bool:
        return Task.parse(task) in self.permitted_tasks


@dataclass(frozen=True, slots=True)
class CommunicationUser:
    user_id: EntityId
    position_m: Vector2
    velocity_m_per_s: Vector2
    demand_bit_per_s: float
    minimum_rate_bit_per_s: float
    normalized_shortfall_budget: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _entity_id(self.user_id, "user_id"))
        object.__setattr__(self, "position_m", _vector2(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_m_per_s", _vector2(self.velocity_m_per_s, "velocity_m_per_s"))
        object.__setattr__(self, "demand_bit_per_s", _finite_number(
            self.demand_bit_per_s, "demand_bit_per_s", minimum=0.0,
        ))
        object.__setattr__(self, "minimum_rate_bit_per_s", _finite_number(
            self.minimum_rate_bit_per_s, "minimum_rate_bit_per_s", minimum=0.0, strict=True,
        ))
        object.__setattr__(self, "normalized_shortfall_budget", _finite_number(
            self.normalized_shortfall_budget, "normalized_shortfall_budget", minimum=0.0,
        ))


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    profile_id: str
    sensing_bandwidth_hz: float
    sensing_power_w: float
    update_period_slots: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise EntityValidationError("profile_id must be a non-empty string")
        object.__setattr__(self, "sensing_bandwidth_hz", _finite_number(
            self.sensing_bandwidth_hz, "sensing_bandwidth_hz", minimum=0.0, strict=True,
        ))
        object.__setattr__(self, "sensing_power_w", _finite_number(
            self.sensing_power_w, "sensing_power_w", minimum=0.0, strict=True,
        ))
        object.__setattr__(self, "update_period_slots", _positive_int(
            self.update_period_slots, "update_period_slots",
        ))


@dataclass(frozen=True, slots=True)
class SensingRequest:
    request_id: EntityId
    tenant_id: EntityId
    arrival_slot: int
    latest_start_slot: int
    aoi: DiskAOI
    target_id: EntityId
    task: Task
    quality_threshold: float
    valid_output_interval_slots: int
    completion_value: float
    merge_permission: bool
    eligible_slot: int | None = None
    state: RequestState = RequestState.WAITING
    admission_slot: int | None = None
    valid_output_age_slots: int | None = None
    valid_output_count: int = 0
    sla_violated: bool = False
    first_violation_slot: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _entity_id(self.request_id, "request_id"))
        object.__setattr__(self, "tenant_id", _entity_id(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "target_id", _entity_id(self.target_id, "target_id"))
        object.__setattr__(self, "arrival_slot", _slot(self.arrival_slot, "arrival_slot"))
        object.__setattr__(self, "latest_start_slot", _slot(self.latest_start_slot, "latest_start_slot"))
        if self.latest_start_slot < self.arrival_slot:
            raise EntityValidationError("latest_start_slot must not precede arrival_slot")
        eligible = self.arrival_slot if self.eligible_slot is None else _slot(self.eligible_slot, "eligible_slot")
        if not self.arrival_slot <= eligible <= self.latest_start_slot:
            raise EntityValidationError("eligible_slot must lie in [arrival_slot, latest_start_slot]")
        object.__setattr__(self, "eligible_slot", eligible)
        if not isinstance(self.aoi, DiskAOI):
            raise EntityValidationError("aoi must be a DiskAOI")
        task = Task.parse(self.task)
        object.__setattr__(self, "task", task)
        threshold = _finite_number(self.quality_threshold, "quality_threshold", minimum=0.0, strict=True)
        if task is Task.DETECTION and threshold > 1.0:
            raise EntityValidationError("detection quality_threshold must be in (0, 1]")
        object.__setattr__(self, "quality_threshold", threshold)
        object.__setattr__(self, "valid_output_interval_slots", _positive_int(
            self.valid_output_interval_slots, "valid_output_interval_slots",
        ))
        object.__setattr__(self, "completion_value", _finite_number(
            self.completion_value, "completion_value", minimum=0.0, strict=True,
        ))
        if type(self.merge_permission) is not bool:
            raise EntityValidationError("merge_permission must be boolean")
        try:
            state = RequestState(self.state)
        except (TypeError, ValueError) as error:
            raise EntityValidationError(f"unsupported request state: {self.state!r}") from error
        object.__setattr__(self, "state", state)
        self._validate_accounting()

    def _validate_accounting(self) -> None:
        admission = self.admission_slot
        if admission is not None:
            _slot(admission, "admission_slot")
            if not self.arrival_slot <= admission <= self.latest_start_slot:
                raise EntityValidationError("admission_slot must lie in [arrival_slot, latest_start_slot]")
        if self.state in {RequestState.WAITING, RequestState.EXPIRED, RequestState.REJECTED}:
            if admission is not None:
                raise EntityValidationError("pre-admission request states must not have admission_slot")
            if any((
                self.valid_output_age_slots is not None, self.valid_output_count != 0,
                self.sla_violated, self.first_violation_slot is not None,
            )):
                raise EntityValidationError("pre-admission request states must not carry SLA accounting")
        elif admission is None:
            raise EntityValidationError("accepted request states require admission_slot")
        _slot(self.valid_output_count, "valid_output_count")
        if self.valid_output_age_slots is not None:
            _slot(self.valid_output_age_slots, "valid_output_age_slots")
        if type(self.sla_violated) is not bool:
            raise EntityValidationError("sla_violated must be boolean")
        if self.first_violation_slot is not None:
            _slot(self.first_violation_slot, "first_violation_slot")
            if admission is None or self.first_violation_slot < admission:
                raise EntityValidationError("first_violation_slot must not precede admission")
        if self.sla_violated != (self.first_violation_slot is not None):
            raise EntityValidationError("sla_violated and first_violation_slot must agree")

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    @property
    def is_accepted(self) -> bool:
        return self.state in {RequestState.ACTIVE, RequestState.COMPLETED, RequestState.FAILED}

    def service_duration_slots(self, durations: TaskDurationMap) -> int:
        return task_service_duration_slots(self.task, durations)

    def final_service_slot(self, durations: TaskDurationMap) -> int | None:
        return None if self.admission_slot is None else self.admission_slot+self.service_duration_slots(durations)-1

    def service_slots(self, durations: TaskDurationMap) -> tuple[int, ...]:
        final_slot = self.final_service_slot(durations)
        return () if final_slot is None else tuple(range(self.admission_slot, final_slot+1))

    def transition(self, new_state: RequestState | str, *, slot: int | None = None) -> SensingRequest:
        try:
            target = RequestState(new_state)
        except (TypeError, ValueError) as error:
            raise EntityValidationError(f"unsupported request state: {new_state!r}") from error
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise EntityValidationError(f"illegal request transition: {self.state.value} -> {target.value}")
        if target is RequestState.ACTIVE:
            if slot is None:
                raise EntityValidationError("activation requires an admission slot")
            admission = _slot(slot, "admission_slot")
            if not self.eligible_slot <= admission <= self.latest_start_slot:
                raise EntityValidationError("admission slot violates request eligibility or deadline")
            return _updated(self, state=target, admission_slot=admission)
        return _updated(self, state=target)

    def defer(self, current_slot: int, cooldown_slots: int) -> SensingRequest:
        if self.state is not RequestState.WAITING:
            raise EntityValidationError("only waiting requests may be deferred")
        current = _slot(current_slot, "current_slot")
        cooldown = _positive_int(cooldown_slots, "cooldown_slots")
        if not self.eligible_slot <= current <= self.latest_start_slot:
            raise EntityValidationError("request is not eligible for deferral in the current slot")
        next_eligible = current+cooldown
        if next_eligible > self.latest_start_slot:
            raise EntityValidationError("deferral would exceed the latest service-start slot")
        return _updated(self, eligible_slot=next_eligible)

    def with_accounting(
        self, *, valid_output_age_slots: int | None, valid_output_count: int,
        sla_violated: bool, first_violation_slot: int | None,
    ) -> SensingRequest:
        if self.state is not RequestState.ACTIVE:
            raise EntityValidationError("SLA accounting may update active requests only")
        admission = self.admission_slot
        assert admission is not None
        age = None if valid_output_age_slots is None else _slot(
            valid_output_age_slots, "valid_output_age_slots",
        )
        count = _slot(valid_output_count, "valid_output_count")
        if type(sla_violated) is not bool:
            raise EntityValidationError("sla_violated must be boolean")
        violation_slot = None if first_violation_slot is None else _slot(
            first_violation_slot, "first_violation_slot",
        )
        if violation_slot is not None and violation_slot < admission:
            raise EntityValidationError("first_violation_slot must not precede admission")
        if count < self.valid_output_count:
            raise EntityValidationError("valid_output_count must be non-decreasing")
        if sla_violated != (violation_slot is not None):
            raise EntityValidationError("sla_violated and first_violation_slot must agree")
        if self.sla_violated and (not sla_violated or violation_slot != self.first_violation_slot):
            raise EntityValidationError("the first-violation flag and slot are absorbing")
        return _updated(
            self, valid_output_age_slots=age, valid_output_count=count,
            sla_violated=sla_violated, first_violation_slot=violation_slot,
        )


@dataclass(frozen=True, slots=True)
class SensingSession:
    session_id: EntityId
    creation_slot: int
    aoi: DiskAOI
    target_id: EntityId
    base_task: Task
    exposed_outputs: frozenset[Task]
    member_request_ids: tuple[EntityId, ...]
    profile: ResourceProfile
    next_update_slot: int
    final_active_slot: int
    tracking_covariance: Matrix4 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _entity_id(self.session_id, "session_id"))
        object.__setattr__(self, "creation_slot", _slot(self.creation_slot, "creation_slot"))
        if not isinstance(self.aoi, DiskAOI):
            raise EntityValidationError("aoi must be a DiskAOI")
        object.__setattr__(self, "target_id", _entity_id(self.target_id, "target_id"))
        task = Task.parse(self.base_task)
        object.__setattr__(self, "base_task", task)
        outputs = frozenset(Task.parse(item) for item in self.exposed_outputs)
        if outputs != task_outputs(task):
            raise EntityValidationError("exposed_outputs must equal the canonical task-output set")
        object.__setattr__(self, "exposed_outputs", outputs)
        members = tuple(_entity_id(item, "member_request_id") for item in self.member_request_ids)
        if not members or len(set(members)) != len(members):
            raise EntityValidationError("member_request_ids must be non-empty and unique")
        object.__setattr__(self, "member_request_ids", members)
        if not isinstance(self.profile, ResourceProfile):
            raise EntityValidationError("profile must be a ResourceProfile")
        object.__setattr__(self, "next_update_slot", _slot(self.next_update_slot, "next_update_slot"))
        object.__setattr__(self, "final_active_slot", _slot(self.final_active_slot, "final_active_slot"))
        if self.final_active_slot < self.creation_slot:
            raise EntityValidationError("final_active_slot must not precede creation_slot")
        if self.next_update_slot < self.creation_slot:
            raise EntityValidationError("next_update_slot must not precede creation_slot")
        if Task.TRACKING in outputs:
            if self.tracking_covariance is None:
                raise EntityValidationError("tracking-capable sessions require tracking_covariance")
            object.__setattr__(self, "tracking_covariance", _matrix4(
                self.tracking_covariance, "tracking_covariance",
            ))
        elif self.tracking_covariance is not None:
            raise EntityValidationError("non-tracking sessions must not carry tracking_covariance")

    @classmethod
    def create(
        cls, session_id: EntityId, request: SensingRequest, profile: ResourceProfile,
        current_slot: int, service_durations: TaskDurationMap,
        tracking_covariance: Matrix4 | None = None,
    ) -> SensingSession:
        if request.state is not RequestState.ACTIVE or request.admission_slot != current_slot:
            raise EntityValidationError("session creation requires a request activated in the current slot")
        final_slot = request.final_service_slot(service_durations)
        assert final_slot is not None
        return cls(
            session_id, current_slot, request.aoi, request.target_id, request.task,
            task_outputs(request.task), (request.request_id,), profile, current_slot,
            final_slot, tracking_covariance,
        )

    def with_member(
        self, request: SensingRequest, profile: ResourceProfile, current_slot: int,
        service_durations: TaskDurationMap,
    ) -> SensingSession:
        current = _slot(current_slot, "current_slot")
        if not isinstance(profile, ResourceProfile):
            raise EntityValidationError("profile must be a ResourceProfile")
        if request.state is not RequestState.ACTIVE or request.admission_slot != current:
            raise EntityValidationError("member admission requires a request activated in the current slot")
        if not self.creation_slot <= current <= self.final_active_slot:
            raise EntityValidationError("cannot admit a member outside the active session interval")
        if request.target_id != self.target_id:
            raise EntityValidationError("session target is immutable and must match the admitted request")
        if request.request_id in self.member_request_ids:
            raise EntityValidationError("request is already a session member")
        final_slot = request.final_service_slot(service_durations)
        assert final_slot is not None
        return _updated(
            self, member_request_ids=self.member_request_ids+(request.request_id,), profile=profile,
            next_update_slot=current, final_active_slot=max(self.final_active_slot, final_slot),
        )

    def detach_terminal_members(self, request_ids: Iterable[EntityId]) -> SensingSession | None:
        detached = {_entity_id(request_id, "terminal_request_id") for request_id in request_ids}
        unknown = detached.difference(self.member_request_ids)
        if unknown:
            raise EntityValidationError(f"cannot detach non-member requests: {sorted(unknown, key=str)}")
        remaining = tuple(request_id for request_id in self.member_request_ids if request_id not in detached)
        return None if not remaining else _updated(self, member_request_ids=remaining)

    def with_update_state(
        self, *, next_update_slot: int, tracking_covariance: Matrix4 | None = None,
    ) -> SensingSession:
        next_update = _slot(next_update_slot, "next_update_slot")
        if next_update < self.creation_slot:
            raise EntityValidationError("next_update_slot must not precede creation_slot")
        if Task.TRACKING in self.exposed_outputs:
            covariance = self.tracking_covariance if tracking_covariance is None else _matrix4(
                tracking_covariance, "tracking_covariance",
            )
        else:
            if tracking_covariance is not None:
                raise EntityValidationError("non-tracking sessions must not carry tracking_covariance")
            covariance = None
        return _updated(self, next_update_slot=next_update, tracking_covariance=covariance)