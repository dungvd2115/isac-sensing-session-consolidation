"""Load environment, algorithm, and experiment configuration."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from isac_ssc.core.entities import EntityValidationError, ResourceProfile, Task, Tenant


class ConfigError(ValueError):
    """Raised when configuration is missing or scientifically invalid."""


DEFAULT_CONFIG_PATH = (Path(__file__).resolve().parents[3] / "configs" / "env" / "default.yaml")
DEFAULT_ALGORITHM_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "algorithm" / "constrained_ppo.yaml"
)
DEFAULT_EXPERIMENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "experiment" / "joint_credit.yaml"
)
JOINT_CREDIT_METHOD = "joint_credit_constrained_ppo"
COMMON_TRACE_METHOD = "common_trace_constrained_ppo"
SUPPORTED_LEARNED_METHODS = (JOINT_CREDIT_METHOD, COMMON_TRACE_METHOD)
CREDIT_ASSIGNMENT_SCHEMAS = MappingProxyType({
    JOINT_CREDIT_METHOD: "joint_trajectory_credit_v2_scale_consistent",
    COMMON_TRACE_METHOD: "common_trace_leave_one_out_mc_factor_credit_v1",
})


def credit_assignment_schema(method: str) -> str:
    try:
        return CREDIT_ASSIGNMENT_SCHEMAS[method]
    except KeyError as error:
        raise ConfigError(f"unsupported learned method: {method!r}") from error


_FLOAT_PATTERN = re.compile(
    r"^(?:"
    r"[-+]?(?:[0-9][0-9_]*)?\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)"
    r"|[-+]?\.(?:inf|Inf|INF)"
    r"|\.(?:nan|NaN|NAN)"
    r")$"
)

class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", _FLOAT_PATTERN, list("-+0123456789."),
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _read_yaml(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read configuration: {path}") from exc

    try:
        loaded = yaml.load(content.decode("utf-8"), Loader=_StrictLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ConfigError) as exc:
        raise ConfigError(f"invalid YAML configuration: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")

    return loaded, content


def _validate_keys(candidate: Any, reference: Any, path: tuple[str, ...] = ()) -> None:
    """Validate the fixed algorithm/experiment schemas.

    Environment configuration is validated by scientific meaning in ``_validate_core_values``
    rather than by comparing every field with the default YAML.
    """
    label = ".".join(path) or "<root>"
    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            raise ConfigError(f"{label} must be a mapping")
        missing, extra = set(reference)-set(candidate), set(candidate)-set(reference)
        if missing or extra:
            raise ConfigError(
                f"invalid configuration keys at {label}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for key in reference:
            _validate_keys(candidate[key], reference[key], path+(str(key),))
        return
    if isinstance(reference, list):
        if not isinstance(candidate, list):
            raise ConfigError(f"{label} must be a list")
        if reference:
            for index, item in enumerate(candidate):
                _validate_keys(item, reference[0], path+(str(index),))
        return
    if isinstance(candidate, (dict, list)):
        raise ConfigError(f"{label} has an invalid container type")
    if isinstance(reference, bool):
        if type(candidate) is not bool:
            raise ConfigError(f"{label} must be boolean")
        return
    if isinstance(reference, int):
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ConfigError(f"{label} must be an integer")
        return
    if isinstance(reference, float):
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise ConfigError(f"{label} must be numeric")
        return
    if isinstance(reference, str) and not isinstance(candidate, str):
        raise ConfigError(f"{label} must be a string")


def _require_environment_sections(data: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "profile_name", "units", "system", "geometry", "population",
        "tenant_profiles", "mobility", "communication", "sensing", "resource_profiles",
        "requests", "arrivals", "sharing_authorization", "target_compatibility",
        "compatibility", "sla", "reward", "observation", "oracle", "trace_generation",
    }
    missing = required-set(data)
    if missing:
        raise ConfigError(f"missing environment configuration sections: {sorted(missing)}")


def _require_number(
    value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ConfigError(f"{path} must be a finite numeric scalar")

    number = float(value)

    if minimum is not None:
        invalid_minimum = (number <= minimum if strict_minimum else number < minimum)
        if invalid_minimum:
            raise ConfigError(f"{path} is below its allowed minimum")

    if maximum is not None and number > maximum:
        raise ConfigError(f"{path} exceeds its allowed maximum")

    return number


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _validate_probability_vector(values: Any, probabilities: Any, path: str) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{path}.values must be a non-empty list")

    if (not isinstance(probabilities, list) or len(values) != len(probabilities)):
        raise ConfigError(f"{path}.probabilities must match values")

    checked = [
        _require_number(probability, f"{path}.probabilities[{index}]", minimum=0.0) for index,
        probability in enumerate(probabilities)
    ]

    if abs(sum(checked) - 1.0) > 1e-12:
        raise ConfigError(f"{path}.probabilities must sum to one")


def _validate_task_contract(data: Mapping[str, Any]) -> tuple[tuple[Task, ...], dict[Task, float]]:
    task_spec = data["requests"]["task"]

    if task_spec["distribution"] != "categorical":
        raise ConfigError("requests.task.distribution must be categorical")

    _validate_probability_vector(task_spec["values"], task_spec["probabilities"], "requests.task")

    try:
        tasks = tuple(Task(value) for value in task_spec["values"])
    except ValueError as exc:
        raise ConfigError("requests.task contains an unsupported task: " f"{exc}") from exc

    if (len(set(tasks)) != len(Task) or set(tasks) != set(Task)):
        raise ConfigError("requests.task.values must contain each " "canonical task exactly once")

    probabilities = dict(zip(tasks, map(float, task_spec["probabilities"]), strict=True))

    return tasks, probabilities


def _range(
    data: Mapping[str, Any], prefix: str, minimum_key: str, maximum_key: str, *,
    positive: bool = False,
) -> None:
    minimum = _require_number(
        data[minimum_key], f"{prefix}.{minimum_key}", minimum=0.0, strict_minimum=positive,
    )
    maximum = _require_number(
        data[maximum_key], f"{prefix}.{maximum_key}", minimum=0.0, strict_minimum=positive,
    )

    if maximum < minimum:
        raise ConfigError(f"{prefix} maximum must not be below minimum")


def _probability(value: Any, path: str, *, open_interval: bool = False) -> float:
    probability = _require_number(value, path, minimum=0.0, maximum=1.0)

    if open_interval and probability in {0.0, 1.0}:
        raise ConfigError(f"{path} must lie strictly between zero and one")

    return probability


def _validate_core_values(
    data: Mapping[str, Any], tasks: tuple[Task, ...], task_probabilities: Mapping[Task, float],
) -> None:
    system = data["system"]
    positive_system_values = (
        "slot_duration_s", "carrier_frequency_hz", "propagation_speed_m_per_s",
        "total_bandwidth_hz", "total_power_w",
    )
    for key in positive_system_values:
        _require_number(system[key], f"system.{key}", minimum=0.0, strict_minimum=True)
    for key in ("communication_noise_figure_db", "sensing_noise_figure_db",
                "communication_implementation_gap_db"):
        _require_number(system[key], f"system.{key}", minimum=0.0)
    _require_int(system["horizon_slots"], "system.horizon_slots", minimum=1)

    geometry = data["geometry"]
    _require_number(
        geometry["minimum_link_distance_m"], "geometry.minimum_link_distance_m", minimum=0.0,
        strict_minimum=True,
    )

    region = geometry["simulation_region"]
    if (region["x_max_m"] <= region["x_min_m"] or region["y_max_m"] <= region["y_min_m"]):
        raise ConfigError("simulation_region bounds must be " "strictly increasing")

    for name in ("communication_user_initial_position", "target_initial_position"):
        _range(
            geometry[name], f"geometry.{name}", "minimum_radius_m", "maximum_radius_m",
            positive=True,
        )

    aoi = geometry["aoi"]
    _range(aoi, "geometry.aoi", "radius_min_m", "radius_max_m", positive=True)
    _require_number(aoi["center_offset_std_m"], "geometry.aoi.center_offset_std_m", minimum=0.0)
    _probability(
        aoi["center_offset_max_fraction_of_radius"],
        "geometry.aoi." "center_offset_max_fraction_of_radius", open_interval=True,
    )

    population = data["population"]
    for key in ("sensing_tenants", "communication_users", "physical_targets"):
        _require_int(population[key], f"population.{key}", minimum=1)

    expected_tenants = tuple(
        f"tenant_{index}"
        for index in range(1, population["sensing_tenants"] + 1)
    )
    if tuple(data["tenant_profiles"]) != expected_tenants:
        raise ConfigError("tenant_profiles must be ordered and named " "tenant_1 through tenant_L")

    for group_name in ("communication_users", "targets"):
        mobility = data["mobility"][group_name]
        _range(
            mobility, f"mobility.{group_name}", "initial_speed_min_m_per_s",
            "initial_speed_max_m_per_s",
        )
        _require_number(
            mobility["acceleration_std_m_per_s2"],
            f"mobility.{group_name}." "acceleration_std_m_per_s2", minimum=0.0,
        )

    communication = data["communication"]
    channel = communication["channel"]

    _require_number(
        channel["reference_distance_m"], "communication.channel.reference_distance_m", minimum=0.0,
        strict_minimum=True,
    )
    _require_number(
        channel["pathloss_exponent"], "communication.channel.pathloss_exponent", minimum=0.0,
        strict_minimum=True,
    )
    _require_number(
        channel["shadowing_std_db"], "communication.channel.shadowing_std_db", minimum=0.0,
    )
    _probability(channel["shadowing_correlation"], "communication.channel.shadowing_correlation")
    _probability(channel["fading_correlation"], "communication.channel.fading_correlation")

    traffic = communication["traffic"]
    for key in ("initial_on_probability", "on_to_off_probability", "off_to_on_probability"):
        _probability(traffic[key], f"communication.traffic.{key}")

    _require_number(
        traffic["demand_median_bit_per_s"], "communication.traffic.demand_median_bit_per_s",
        minimum=0.0, strict_minimum=True,
    )
    _require_number(
        traffic["demand_natural_log_std"], "communication.traffic.demand_natural_log_std",
        minimum=0.0,
    )
    _require_number(
        communication["minimum_rate_bit_per_s"], "communication.minimum_rate_bit_per_s",
        minimum=0.0, strict_minimum=True,
    )
    _require_number(
        communication["normalized_shortfall_budget"],
        "communication.normalized_shortfall_budget", minimum=0.0,
    )

    sensing = data["sensing"]
    _require_number(sensing["system_loss_db"], "sensing.system_loss_db", minimum=0.0)
    _require_number(
        sensing["effective_aperture_m"], "sensing.effective_aperture_m", minimum=0.0,
        strict_minimum=True,
    )
    _require_number(sensing["shadowing_std_db"], "sensing.shadowing_std_db", minimum=0.0)
    _probability(sensing["shadowing_correlation"], "sensing.shadowing_correlation")
    _probability(sensing["fading_correlation"], "sensing.fading_correlation")
    _probability(
        sensing["false_alarm_probability"], "sensing.false_alarm_probability", open_interval=True,
    )
    _probability(
        sensing["detection_gate_probability"], "sensing.detection_gate_probability",
        open_interval=True,
    )

    rcs = sensing["rcs"]
    _require_number(rcs["median_m2"], "sensing.rcs.median_m2", minimum=0.0, strict_minimum=True)
    _require_number(rcs["dbsm_std_db"], "sensing.rcs.dbsm_std_db", minimum=0.0)
    _probability(rcs["correlation"], "sensing.rcs.correlation")

    covariance = sensing["tracking"]["initial_covariance_diag"]
    if not isinstance(covariance, list) or len(covariance) != 4:
        raise ConfigError("sensing.tracking.initial_covariance_diag " "must contain four entries")

    for index, value in enumerate(covariance):
        _require_number(
            value, "sensing.tracking.initial_covariance_diag" f"[{index}]", minimum=0.0,
            strict_minimum=True,
        )

    profiles = data["resource_profiles"]
    if not profiles:
        raise ConfigError("resource_profiles must not be empty")

    for name, specification in profiles.items():
        try:
            ResourceProfile(
                name, specification["sensing_bandwidth_hz"], specification["sensing_power_w"],
                specification["update_period_slots"],
            )
        except EntityValidationError as exc:
            raise ConfigError(f"invalid resource profile {name}: {exc}") from exc

    matrix = data["sharing_authorization"]["tenant_pair_matrix"]
    tenant_count = population["sensing_tenants"]

    invalid_shape = (
        not isinstance(matrix, list)
        or len(matrix) != tenant_count
        or any(not isinstance(row, list) or len(row) != tenant_count for row in matrix)
    )
    if invalid_shape:
        raise ConfigError("tenant_pair_matrix must be square " "with one row per tenant")

    for row_index, row in enumerate(matrix):
        if any(type(flag) is not bool for flag in row):
            raise ConfigError("tenant_pair_matrix must contain " "booleans only")
        if not row[row_index]:
            raise ConfigError("tenant_pair_matrix diagonal must be true")
        if any(flag != matrix[column_index][row_index] for column_index, flag in enumerate(row)):
            raise ConfigError("tenant_pair_matrix must be symmetric")

    for index, (name, profile) in enumerate(data["tenant_profiles"].items()):
        try:
            permitted = {Task(value) for value in profile["permitted_tasks"]}
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name}.permitted_tasks contains " "an unsupported task") from exc

        if not permitted:
            raise ConfigError(f"{name}.permitted_tasks must not be empty")

        retained_probability = sum(task_probabilities[task] for task in permitted)
        if retained_probability <= 0.0:
            raise ConfigError(f"{name} retains zero global " "task probability mass")

        _probability(
            profile["sla_violation_budget"], f"tenant_profiles.{name}." "sla_violation_budget",
        )

        if not matrix[index][index]:
            raise ConfigError(f"authorization row for {name} is invalid")

    requests = data["requests"]
    threshold_keys = {
        Task.DETECTION: "detection_probability", Task.LOCALIZATION: "localization_peb_m",
        Task.TRACKING: "tracking_pcrb_m",
    }

    for task in tasks:
        _require_int(
            requests["service_duration_slots"][task.value],
            "requests.service_duration_slots." f"{task.value}", minimum=1,
        )

        update = requests["update_interval_slots"][task.value]
        _validate_probability_vector(
            update["values"], update["probabilities"],
            "requests.update_interval_slots." f"{task.value}",
        )

        invalid_update_values = any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in update["values"]
        )
        if invalid_update_values:
            raise ConfigError(
                "requests.update_interval_slots."
                f"{task.value}.values must be "
                "positive integers"
            )

        threshold_key = threshold_keys[task]
        threshold = requests["quality_thresholds"][threshold_key]

        threshold_minimum = _require_number(
            threshold["minimum"], "requests.quality_thresholds." f"{threshold_key}.minimum",
            minimum=0.0, strict_minimum=True,
        )
        threshold_maximum = _require_number(
            threshold["maximum"], "requests.quality_thresholds." f"{threshold_key}.maximum",
            minimum=0.0, strict_minimum=True,
        )

        invalid_threshold = (
            threshold_maximum <= threshold_minimum
            or (task is Task.DETECTION and threshold_maximum > 1.0)
        )
        if invalid_threshold:
            raise ConfigError("invalid quality threshold range for " f"{task.value}")

        value_specification = requests["completion_values"][task.value]
        value_minimum = _require_number(
            value_specification["minimum"], "requests.completion_values." f"{task.value}.minimum",
            minimum=0.0, strict_minimum=True,
        )
        value_maximum = _require_number(
            value_specification["maximum"], "requests.completion_values." f"{task.value}.maximum",
            minimum=0.0, strict_minimum=True,
        )

        if value_maximum <= value_minimum:
            raise ConfigError("completion value range for " f"{task.value} must be increasing")

    slack = requests["latest_start_slack_slots"]
    _require_int(slack["minimum"], "requests.latest_start_slack_slots.minimum")
    maximum_slack = _require_int(slack["maximum"], "requests.latest_start_slack_slots.maximum")
    if maximum_slack < slack["minimum"]:
        raise ConfigError("latest-start slack maximum must not " "be below minimum")

    _probability(
        requests["sharing_permission"]["probability_true"],
        "requests.sharing_permission.probability_true",
    )
    _require_int(requests["defer_cooldown_slots"], "requests.defer_cooldown_slots", minimum=1)

    arrivals = data["arrivals"]
    _require_number(
        arrivals["independent"]["per_tenant_rate_per_slot"],
        "arrivals.independent.per_tenant_rate_per_slot", minimum=0.0,
    )

    clustered = arrivals["clustered"]
    _require_number(
        clustered["parent_rate_per_slot"], "arrivals.clustered.parent_rate_per_slot", minimum=0.0,
    )
    _require_number(
        clustered["child_poisson_mean"], "arrivals.clustered.child_poisson_mean", minimum=0.0,
    )
    _require_int(
        clustered["temporal_offset_min_slots"], "arrivals.clustered." "temporal_offset_min_slots",
    )
    maximum_offset = _require_int(
        clustered["temporal_offset_max_slots"], "arrivals.clustered." "temporal_offset_max_slots",
    )
    if (maximum_offset < clustered["temporal_offset_min_slots"]):
        raise ConfigError("clustered temporal offset maximum " "must not be below minimum")

    for key in ("inherit_parent_target_probability", "inherit_parent_task_probability"):
        _probability(clustered[key], f"arrivals.clustered.{key}")

    _require_number(
        clustered["child_aoi_center_offset_std_m"],
        "arrivals.clustered." "child_aoi_center_offset_std_m", minimum=0.0,
    )

    _probability(
        data["compatibility"]["minimum_spatial_coverage_ratio"],
        "compatibility.minimum_spatial_coverage_ratio",
    )

    reward = data["reward"]
    if float(reward["finite_horizon_discount_factor"]) != 1.0:
        raise ConfigError("reward.finite_horizon_discount_factor " "must equal one")

    for key in (
        "sensing_resource_cost_weight", "sensing_cost_bandwidth_weight",
        "sensing_cost_power_weight",
    ):
        _require_number(reward[key], f"reward.{key}", minimum=0.0)

    component_weight_sum = (
        reward["sensing_cost_bandwidth_weight"]
        + reward["sensing_cost_power_weight"]
    )
    if abs(component_weight_sum - 1.0) > 1e-12:
        raise ConfigError("sensing resource component weights " "must sum to one")

    oracle = data["oracle"]
    oracle_limits = oracle["instance_selection_limits"]
    _require_int(
        oracle_limits["horizon_slots"], "oracle.instance_selection_limits." "horizon_slots",
        minimum=1,
    )
    _require_int(
        oracle_limits["max_requests"], "oracle.instance_selection_limits." "max_requests",
        minimum=1,
    )
    _require_number(oracle["solver_relative_gap"], "oracle.solver_relative_gap", minimum=0.0)

    if "max_sessions" in oracle_limits:
        raise ConfigError(
            "oracle.max_sessions is prohibited because "
            "selected-session count is endogenous"
        )

    trace_generation = data["trace_generation"]
    supported_regimes = tuple(trace_generation["registered_arrival_regimes"])
    if supported_regimes != ("independent", "clustered"):
        raise ConfigError("trace_generation.registered_arrival_regimes must contain independent and clustered")
    if arrivals["active_regime"] not in supported_regimes:
        raise ConfigError("arrivals.active_regime is unsupported")

    seeding = trace_generation["seeding"]
    root_minimum = _require_int(seeding["root_seed_minimum"], "trace_generation.seeding.root_seed_minimum")
    root_maximum = _require_int(seeding["root_seed_maximum"], "trace_generation.seeding.root_seed_maximum")
    if root_minimum != 0 or root_maximum != 2**64-1:
        raise ConfigError("root seeds must use the complete unsigned 64-bit domain")


@dataclass(frozen=True, slots=True)
class CanonicalConfig:
    schema_version: str
    profile_name: str
    units: Mapping[str, str]
    system: Mapping[str, Any]
    geometry: Mapping[str, Any]
    population: Mapping[str, int]
    tenants: tuple[Tenant, ...]
    mobility: Mapping[str, Any]
    communication: Mapping[str, Any]
    sensing: Mapping[str, Any]
    resource_profiles: Mapping[str, ResourceProfile]
    requests: Mapping[str, Any]
    arrivals: Mapping[str, Any]
    sharing_authorization: Mapping[str, Any]
    target_compatibility: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    sla: Mapping[str, Any]
    reward: Mapping[str, Any]
    observation: Mapping[str, Any]
    oracle: Mapping[str, Any]
    trace_generation: Mapping[str, Any]

    @property
    def task_probabilities(self) -> Mapping[Task, float]:
        values = tuple(Task(item) for item in self.requests["task"]["values"])
        probabilities = self.requests["task"]["probabilities"]

        return MappingProxyType(dict(zip(values, probabilities, strict=True)))

    @property
    def service_duration_slots(self) -> Mapping[Task, int]:
        return MappingProxyType({
            Task(key): int(value) for key, value in self.requests["service_duration_slots"].items()
        })

    def tenant(self, tenant_id: str | int) -> Tenant:
        for tenant in self.tenants:
            if tenant.tenant_id == tenant_id:
                return tenant

        raise KeyError(f"unknown tenant: {tenant_id!r}")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> CanonicalConfig:
    source = Path(path).resolve()
    data, _ = _read_yaml(source)
    _require_environment_sections(data)
    try:
        tasks, task_probabilities = _validate_task_contract(data)
        _validate_core_values(data, tasks, task_probabilities)
    except KeyError as error:
        raise ConfigError(f"missing required environment field: {error.args[0]}") from error

    matrix = data["sharing_authorization"]["tenant_pair_matrix"]
    tenants = tuple(
        Tenant(
            name, frozenset(Task(task) for task in specification["permitted_tasks"]),
            specification["sla_violation_budget"], tuple(matrix[index]),
        )
        for index, (name, specification) in enumerate(data["tenant_profiles"].items())
    )

    profiles = {
        name: ResourceProfile(
            name, specification["sensing_bandwidth_hz"], specification["sensing_power_w"],
            specification["update_period_slots"],
        )
        for name, specification in data["resource_profiles"].items()
    }

    frozen = _freeze(data)

    return CanonicalConfig(
        schema_version=data["schema_version"], profile_name=data["profile_name"],
        units=frozen["units"], system=frozen["system"], geometry=frozen["geometry"],
        population=frozen["population"], tenants=tenants, mobility=frozen["mobility"],
        communication=frozen["communication"], sensing=frozen["sensing"],
        resource_profiles=MappingProxyType(profiles),
        requests=frozen["requests"], arrivals=frozen["arrivals"],
        sharing_authorization=frozen["sharing_authorization"],
        target_compatibility=frozen["target_compatibility"], compatibility=frozen["compatibility"],
        sla=frozen["sla"], reward=frozen["reward"], observation=frozen["observation"],
        oracle=frozen["oracle"], trace_generation=frozen["trace_generation"],
    )


@dataclass(frozen=True, slots=True)
class AlgorithmModelConfig:
    hidden_dim: int
    profile_embedding_dim: int
    activation: str = field(default="tanh", init=False)
    pooling: str = field(default="masked_mean", init=False)
    dropout: float = field(default=0.0, init=False)
    orthogonal_initialization: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str = field(default="adam", init=False)
    learning_rate: float
    epsilon: float


@dataclass(frozen=True, slots=True)
class PPOConfig:
    discount: float
    gae_lambda: float
    clip_ratio: float
    value_clip_ratio: float
    entropy_coefficient: float
    reward_value_coefficient: float
    constraint_value_coefficient: float
    max_gradient_norm: float
    epochs_per_rollout: int
    minibatch_decisions: int
    target_kl: float


@dataclass(frozen=True, slots=True)
class DualConfig:
    initial_value: float
    learning_rate: float
    maximum: float


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    enabled: bool = field(default=True, init=False)
    clip: float
    epsilon: float


@dataclass(frozen=True, slots=True)
class ConstrainedPPOConfig:
    schema_version: str
    device: str
    dtype: str = field(default="float32", init=False)
    model: AlgorithmModelConfig
    optimizer: OptimizerConfig
    ppo: PPOConfig
    dual: DualConfig
    normalization: NormalizationConfig
    source_path: Path
    content_hash: str


def _algorithm_number(
    value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    return _require_number(
        value, path, minimum=minimum, maximum=maximum, strict_minimum=strict_minimum,
    )


def _algorithm_literal(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ConfigError(f"{path} must equal {expected!r}")


def _normalize_legacy_algorithm_values(data: dict[str, Any]) -> None:
    fixed = {
        ("runtime", "dtype"): "float32", ("model", "activation"): "tanh",
        ("model", "pooling"): "masked_mean", ("model", "dropout"): 0.0,
        ("model", "orthogonal_initialization"): True, ("optimizer", "name"): "adam",
        ("normalization", "enabled"): True,
    }
    for (section, key), expected in fixed.items():
        values = data.get(section)
        if isinstance(values, dict) and key in values:
            _algorithm_literal(values.pop(key), expected, f"{section}.{key}")


def _validate_algorithm_values(data: Mapping[str, Any]) -> None:
    _algorithm_literal(data["schema_version"], "isac-ssc-constrained-ppo-v1", "schema_version")
    runtime, model = data["runtime"], data["model"]
    optimizer, ppo = data["optimizer"], data["ppo"]
    dual, normalization = data["dual"], data["normalization"]
    if runtime["device"] not in ("cpu", "cuda"):
        raise ConfigError("runtime.device must be cpu or cuda")
    _require_int(model["hidden_dim"], "model.hidden_dim", minimum=1)
    _require_int(model["profile_embedding_dim"], "model.profile_embedding_dim", minimum=1)
    _algorithm_number(
        optimizer["learning_rate"], "optimizer.learning_rate", minimum=0.0,
        strict_minimum=True,
    )
    _algorithm_number(
        optimizer["epsilon"], "optimizer.epsilon", minimum=0.0, strict_minimum=True,
    )
    _algorithm_literal(ppo["discount"], 1.0, "ppo.discount")
    _algorithm_number(ppo["gae_lambda"], "ppo.gae_lambda", minimum=0.0, maximum=1.0)
    _algorithm_number(
        ppo["clip_ratio"], "ppo.clip_ratio", minimum=0.0, maximum=1.0,
        strict_minimum=True,
    )
    _algorithm_number(
        ppo["value_clip_ratio"], "ppo.value_clip_ratio", minimum=0.0, maximum=1.0,
        strict_minimum=True,
    )
    for name in (
        "entropy_coefficient", "reward_value_coefficient", "constraint_value_coefficient",
    ):
        _algorithm_number(ppo[name], f"ppo.{name}", minimum=0.0)
    _algorithm_number(
        ppo["max_gradient_norm"], "ppo.max_gradient_norm", minimum=0.0,
        strict_minimum=True,
    )
    _require_int(ppo["epochs_per_rollout"], "ppo.epochs_per_rollout", minimum=1)
    _require_int(ppo["minibatch_decisions"], "ppo.minibatch_decisions", minimum=1)
    _algorithm_number(
        ppo["target_kl"], "ppo.target_kl", minimum=0.0, strict_minimum=True,
    )
    initial = _algorithm_number(dual["initial_value"], "dual.initial_value", minimum=0.0)
    _algorithm_number(
        dual["learning_rate"], "dual.learning_rate", minimum=0.0, strict_minimum=True,
    )
    maximum = _algorithm_number(
        dual["maximum"], "dual.maximum", minimum=0.0, strict_minimum=True,
    )
    if maximum < initial:
        raise ConfigError("dual.maximum must be greater than or equal to dual.initial_value")
    _algorithm_number(
        normalization["clip"], "normalization.clip", minimum=0.0, strict_minimum=True,
    )
    _algorithm_number(
        normalization["epsilon"], "normalization.epsilon", minimum=0.0,
        strict_minimum=True,
    )


def load_algorithm_config(
    path: str | Path = DEFAULT_ALGORITHM_CONFIG_PATH,
) -> ConstrainedPPOConfig:
    source = Path(path).resolve()
    data, content = _read_yaml(source)
    reference, _ = _read_yaml(DEFAULT_ALGORITHM_CONFIG_PATH)
    _normalize_legacy_algorithm_values(data)
    _validate_keys(data, reference)
    _validate_algorithm_values(data)
    model, optimizer, ppo = data["model"], data["optimizer"], data["ppo"]
    dual, normalization = data["dual"], data["normalization"]
    return ConstrainedPPOConfig(
        schema_version=data["schema_version"], device=data["runtime"]["device"],
        model=AlgorithmModelConfig(**model), optimizer=OptimizerConfig(**optimizer),
        ppo=PPOConfig(**ppo),
        dual=DualConfig(**dual), normalization=NormalizationConfig(**normalization),
        source_path=source, content_hash=hashlib.sha256(content).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ExperimentRuntimeConfig:
    device: str
    deterministic_algorithms: bool
    torch_num_threads: int


@dataclass(frozen=True, slots=True)
class TrainingScheduleConfig:
    seed: int
    physical_slots: int
    arrival_regimes: tuple[str, ...]
    rollout_target_physical_slots: int
    learning_rate_schedule: str
    learning_rate_schedule_horizon_physical_slots: int


@dataclass(frozen=True, slots=True)
class ValidationScheduleConfig:
    enabled: bool
    run_before_training: bool
    interval_physical_slots: int
    trace_seeds: tuple[int, ...]
    arrival_regimes: tuple[str, ...]
    random_valid_root_seed: int
    random_valid_replicates_per_trace: int


@dataclass(frozen=True, slots=True)
class CheckpointScheduleConfig:
    interval_physical_slots: int
    best_metric: str
    keep_top_k: int
    save_latest_every_rollout: bool


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    progress: bool
    progress_every_rollouts: int
    jsonl: bool
    csv: bool
    flush_every_records: int


@dataclass(frozen=True, slots=True)
class TrainingExperimentConfig:
    method: str
    runtime: ExperimentRuntimeConfig
    training: TrainingScheduleConfig
    validation: ValidationScheduleConfig
    checkpoint: CheckpointScheduleConfig
    logging: LoggingConfig


def _seed(value: Any, path: str) -> int:
    result = _require_int(value, path)
    if result > 2**64 - 1:
        raise ConfigError(f"{path} must lie in the unsigned 64-bit range")
    return result


def _seed_list(values: Any, path: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{path} must be a non-empty list")
    result = tuple(_seed(value, f"{path}[{index}]") for index, value in enumerate(values))
    if len(set(result)) != len(result):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def _regimes(values: Any, path: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{path} must be a non-empty list")
    result = tuple(values)
    supported = {"independent", "clustered"}
    if any(not isinstance(value, str) or value not in supported for value in result):
        raise ConfigError(f"{path} contains an unsupported arrival regime")
    if len(set(result)) != len(result):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{path} must be boolean")
    return value


def _validate_experiment_values(data: Mapping[str, Any]) -> None:
    if data["method"] not in SUPPORTED_LEARNED_METHODS:
        raise ConfigError(
            "method must be one of " + ", ".join(SUPPORTED_LEARNED_METHODS)
        )
    runtime, training, validation = data["runtime"], data["training"], data["validation"]
    checkpoint, logging = data["checkpoint"], data["logging"]
    if runtime["device"] not in ("cpu", "cuda"):
        raise ConfigError("runtime.device must be cpu or cuda")
    _boolean(runtime["deterministic_algorithms"], "runtime.deterministic_algorithms")
    _require_int(runtime["torch_num_threads"], "runtime.torch_num_threads", minimum=1)
    _seed(training["seed"], "training.seed")
    _require_int(training["physical_slots"], "training.physical_slots", minimum=1)
    _regimes(training["arrival_regimes"], "training.arrival_regimes")
    _require_int(training["rollout_target_physical_slots"], "training.rollout_target_physical_slots", minimum=1)
    if training["learning_rate_schedule"] not in ("constant", "linear"):
        raise ConfigError("training.learning_rate_schedule must be constant or linear")
    _require_int(
        training["learning_rate_schedule_horizon_physical_slots"],
        "training.learning_rate_schedule_horizon_physical_slots", minimum=1,
    )
    _boolean(validation["enabled"], "validation.enabled")
    _boolean(validation["run_before_training"], "validation.run_before_training")
    _require_int(validation["interval_physical_slots"], "validation.interval_physical_slots", minimum=0)
    _seed_list(validation["trace_seeds"], "validation.trace_seeds")
    _regimes(validation["arrival_regimes"], "validation.arrival_regimes")
    _seed(validation["random_valid_root_seed"], "validation.random_valid_root_seed")
    _require_int(validation["random_valid_replicates_per_trace"], "validation.random_valid_replicates_per_trace", minimum=1)
    _require_int(checkpoint["interval_physical_slots"], "checkpoint.interval_physical_slots", minimum=0)
    if checkpoint["best_metric"] not in ("validation_return", "paired_return_difference", "constraint_lexicographic"):
        raise ConfigError("checkpoint.best_metric is unsupported")
    _require_int(checkpoint["keep_top_k"], "checkpoint.keep_top_k", minimum=1)
    _boolean(checkpoint["save_latest_every_rollout"], "checkpoint.save_latest_every_rollout")
    for key in ("progress", "jsonl", "csv"):
        _boolean(logging[key], f"logging.{key}")
    _require_int(logging["progress_every_rollouts"], "logging.progress_every_rollouts", minimum=1)
    _require_int(logging["flush_every_records"], "logging.flush_every_records", minimum=1)


def load_experiment_config(path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH) -> TrainingExperimentConfig:
    source = Path(path).resolve()
    data, _ = _read_yaml(source)
    reference, _ = _read_yaml(DEFAULT_EXPERIMENT_CONFIG_PATH)
    _validate_keys(data, reference)
    _validate_experiment_values(data)
    training = dict(data["training"], arrival_regimes=_regimes(data["training"]["arrival_regimes"], "training.arrival_regimes"))
    validation = dict(
        data["validation"],
        trace_seeds=_seed_list(data["validation"]["trace_seeds"], "validation.trace_seeds"),
        arrival_regimes=_regimes(data["validation"]["arrival_regimes"], "validation.arrival_regimes"),
    )
    return TrainingExperimentConfig(
        data["method"], ExperimentRuntimeConfig(**data["runtime"]), TrainingScheduleConfig(**training),
        ValidationScheduleConfig(**validation), CheckpointScheduleConfig(**data["checkpoint"]), LoggingConfig(**data["logging"]),
    )
