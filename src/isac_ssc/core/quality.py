"""Deterministic geometry, communication, sensing, localization, and tracking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, isfinite, log, pi, sin, sqrt
from numbers import Real
from typing import Any, Iterable, TypeAlias

import numpy as np
from scipy.stats import ncx2

from isac_ssc.core.entities import DiskAOI, Matrix4, SensingSession, Task

Matrix2: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


class QualityValidationError(ValueError):
    """Raised when a physical input or numerical result is invalid."""


def _finite(value: object, name: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise QualityValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and (number <= minimum if strict else number < minimum):
        operator = ">" if strict else ">="
        raise QualityValidationError(f"{name} must be {operator} {minimum}")
    return number


def _probability(value: object, name: str, *, open_interval: bool = False) -> float:
    probability = _finite(value, name, minimum=0.0)
    if probability > 1.0 or (open_interval and probability in {0.0, 1.0}):
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise QualityValidationError(f"{name} must lie in {interval}")
    return probability


def _vector2(value: Iterable[float], name: str) -> np.ndarray:
    try:
        vector = np.asarray(tuple(value), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QualityValidationError(f"{name} must contain two finite coordinates") from exc
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise QualityValidationError(f"{name} must contain two finite coordinates")
    return vector


def _matrix(value: Iterable[Iterable[float]], shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        matrix = np.asarray(tuple(tuple(row) for row in value), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QualityValidationError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix") from exc
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise QualityValidationError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def _symmetric(matrix: np.ndarray, name: str, *, positive_definite: bool = False) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12 * scale):
        raise QualityValidationError(f"{name} must be symmetric")
    if positive_definite:
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as exc:
            raise QualityValidationError(f"{name} must be positive definite") from exc
    elif float(np.linalg.eigvalsh(symmetric)[0]) < -1e-12 * scale:
        raise QualityValidationError(f"{name} must be positive semidefinite")
    return symmetric


def _matrix2_tuple(matrix: np.ndarray) -> Matrix2:
    return tuple(tuple(float(cell) for cell in row) for row in matrix)  # type: ignore[return-value]


def _matrix4_tuple(matrix: np.ndarray) -> Matrix4:
    return tuple(tuple(float(cell) for cell in row) for row in matrix)  # type: ignore[return-value]


def db_to_linear(value_db: float) -> float:
    """Convert a power ratio from dB to linear scale."""
    return float(np.float64(10.0) ** (np.float64(_finite(value_db, "value_db")) / np.float64(10.0)))


def shadowing_db_to_linear_gain(shadowing_db: float) -> float:
    """Convert a signed shadowing-loss state in dB to multiplicative power gain."""
    return db_to_linear(-_finite(shadowing_db, "shadowing_db"))


def dbm_per_hz_to_w_per_hz(value_dbm_per_hz: float) -> float:
    """Convert dBm/Hz to W/Hz."""
    value = _finite(value_dbm_per_hz, "value_dbm_per_hz")
    return float(np.float64(10.0) ** ((np.float64(value) - np.float64(30.0)) / np.float64(10.0)))


def wavelength_m(propagation_speed_m_per_s: float, carrier_frequency_hz: float) -> float:
    speed = _finite(propagation_speed_m_per_s, "propagation_speed_m_per_s", minimum=0.0, strict=True)
    frequency = _finite(carrier_frequency_hz, "carrier_frequency_hz", minimum=0.0, strict=True)
    return float(np.float64(speed) / np.float64(frequency))


@dataclass(frozen=True, slots=True)
class CommunicationParameters:
    propagation_speed_m_per_s: float
    carrier_frequency_hz: float
    minimum_link_distance_m: float
    reference_distance_m: float
    pathloss_exponent: float
    noise_psd_w_per_hz: float
    noise_factor_linear: float
    implementation_gap_linear: float

    def __post_init__(self) -> None:
        for name in (
            "propagation_speed_m_per_s", "carrier_frequency_hz", "minimum_link_distance_m",
            "reference_distance_m", "pathloss_exponent", "noise_psd_w_per_hz",
            "noise_factor_linear", "implementation_gap_linear",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0, strict=True))
        if self.implementation_gap_linear < 1.0:
            raise QualityValidationError("implementation_gap_linear must be >= 1")

    @classmethod
    def from_config(cls, config: Any) -> CommunicationParameters:
        system, channel, geometry = config.system, config.communication["channel"], config.geometry
        return cls(
            system["propagation_speed_m_per_s"], system["carrier_frequency_hz"],
            geometry["minimum_link_distance_m"], channel["reference_distance_m"],
            channel["pathloss_exponent"], dbm_per_hz_to_w_per_hz(system["noise_psd_dbm_per_hz"]),
            db_to_linear(system["communication_noise_figure_db"]),
            db_to_linear(system["communication_implementation_gap_db"]),
        )


@dataclass(frozen=True, slots=True)
class SensingParameters:
    propagation_speed_m_per_s: float
    carrier_frequency_hz: float
    minimum_link_distance_m: float
    noise_psd_w_per_hz: float
    noise_factor_linear: float
    combined_frontend_gain_linear: float
    system_loss_linear: float
    effective_aperture_m: float
    false_alarm_probability: float
    detection_gate_probability: float

    def __post_init__(self) -> None:
        for name in (
            "propagation_speed_m_per_s", "carrier_frequency_hz", "minimum_link_distance_m",
            "noise_psd_w_per_hz", "noise_factor_linear", "combined_frontend_gain_linear",
            "system_loss_linear", "effective_aperture_m",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0, strict=True))
        object.__setattr__(self, "false_alarm_probability", _probability(
            self.false_alarm_probability, "false_alarm_probability", open_interval=True,
        ))
        object.__setattr__(self, "detection_gate_probability", _probability(
            self.detection_gate_probability, "detection_gate_probability", open_interval=True,
        ))
        if self.noise_factor_linear < 1.0 or self.system_loss_linear < 1.0:
            raise QualityValidationError("noise_factor_linear and system_loss_linear must be >= 1")

    @classmethod
    def from_config(cls, config: Any) -> SensingParameters:
        system, sensing, geometry = config.system, config.sensing, config.geometry
        return cls(
            system["propagation_speed_m_per_s"], system["carrier_frequency_hz"],
            geometry["minimum_link_distance_m"], dbm_per_hz_to_w_per_hz(system["noise_psd_dbm_per_hz"]),
            db_to_linear(system["sensing_noise_figure_db"]),
            db_to_linear(sensing["combined_frontend_gain_db"]), db_to_linear(sensing["system_loss_db"]),
            sensing["effective_aperture_m"], sensing["false_alarm_probability"],
            sensing["detection_gate_probability"],
        )


@dataclass(frozen=True, slots=True)
class CommunicationQuality:
    distance_m: float
    path_gain: float
    sinr: float
    achievable_rate_bit_per_s: float
    served_rate_bit_per_s: float


@dataclass(frozen=True, slots=True)
class LocalizationQuality:
    information_valid: bool
    range_variance_m2: float | None
    bearing_variance_rad2: float | None
    position_fim: Matrix2 | None
    position_covariance_m2: Matrix2 | None
    peb_m: float | None

    @classmethod
    def invalid(cls) -> LocalizationQuality:
        return cls(False, None, None, None, None, None)


@dataclass(frozen=True, slots=True)
class TrackingQuality:
    measurement_updated: bool
    posterior_covariance: Matrix4
    pcrb_m: float


@dataclass(frozen=True, slots=True)
class SharedSensingQuality:
    target_in_session_aoi: bool
    distance_m: float
    bearing_rad: float
    echo_gain: float
    sensing_sinr: float
    detection_probability: float
    detection_gate_passed: bool
    localization: LocalizationQuality
    tracking: TrackingQuality | None

    @property
    def physical_valid(self) -> bool:
        return self.target_in_session_aoi and self.sensing_sinr > 0.0


def link_distance_m(position_m: Iterable[float], bs_position_m: Iterable[float], minimum_distance_m: float) -> float:
    position, bs = _vector2(position_m, "position_m"), _vector2(bs_position_m, "bs_position_m")
    minimum = _finite(minimum_distance_m, "minimum_distance_m", minimum=0.0, strict=True)
    return float(max(np.float64(minimum), np.linalg.norm(position - bs)))


def target_bearing_rad(target_position_m: Iterable[float], bs_position_m: Iterable[float]) -> float:
    target, bs = _vector2(target_position_m, "target_position_m"), _vector2(bs_position_m, "bs_position_m")
    return float(atan2(float(target[1] - bs[1]), float(target[0] - bs[0])))


def point_in_disk(point_m: Iterable[float], disk: DiskAOI, *, atol: float = 1e-12) -> bool:
    if not isinstance(disk, DiskAOI):
        raise QualityValidationError("disk must be a DiskAOI")
    tolerance = _finite(atol, "atol", minimum=0.0)
    point, center = _vector2(point_m, "point_m"), _vector2(disk.center_m, "disk.center_m")
    return bool(np.linalg.norm(point - center) <= np.float64(disk.radius_m + tolerance))


def disk_intersection_area_m2(first: DiskAOI, second: DiskAOI) -> float:
    if not isinstance(first, DiskAOI) or not isinstance(second, DiskAOI):
        raise QualityValidationError("disk intersection requires two DiskAOI values")
    center_distance = float(np.linalg.norm(_vector2(first.center_m, "first.center_m") - _vector2(
        second.center_m, "second.center_m",
    )))
    r1, r2 = float(first.radius_m), float(second.radius_m)
    if center_distance >= r1 + r2:
        return 0.0
    if center_distance <= abs(r1 - r2):
        return float(pi * min(r1, r2) ** 2)
    first_angle = acos(np.clip((center_distance**2 + r1**2 - r2**2) / (2.0 * center_distance * r1), -1.0, 1.0))
    second_angle = acos(np.clip((center_distance**2 + r2**2 - r1**2) / (2.0 * center_distance * r2), -1.0, 1.0))
    triangle_term = 0.5 * sqrt(max(0.0, (-center_distance + r1 + r2) * (center_distance + r1 - r2)
                                         * (center_distance - r1 + r2) * (center_distance + r1 + r2)))
    return float(r1**2 * first_angle + r2**2 * second_angle - triangle_term)


def aoi_coverage_ratio(request_aoi: DiskAOI, session_aoi: DiskAOI) -> float:
    if not isinstance(request_aoi, DiskAOI):
        raise QualityValidationError("request_aoi must be a DiskAOI")
    area = pi * request_aoi.radius_m**2
    return float(np.clip(disk_intersection_area_m2(request_aoi, session_aoi) / area, 0.0, 1.0))


def reference_free_space_gain(wavelength: float, reference_distance_m: float) -> float:
    wavelength_value = _finite(wavelength, "wavelength", minimum=0.0, strict=True)
    distance = _finite(reference_distance_m, "reference_distance_m", minimum=0.0, strict=True)
    return float((np.float64(wavelength_value) / (np.float64(4.0 * pi) * np.float64(distance))) ** 2)


def communication_path_gain(
    distance_m: float, parameters: CommunicationParameters, shadowing_db: float,
    fading_power_gain: float,
) -> float:
    if not isinstance(parameters, CommunicationParameters):
        raise QualityValidationError("parameters must be CommunicationParameters")
    distance = max(
        _finite(distance_m, "distance_m", minimum=0.0, strict=True), parameters.minimum_link_distance_m,
    )
    fading = _finite(fading_power_gain, "fading_power_gain", minimum=0.0)
    wavelength = wavelength_m(parameters.propagation_speed_m_per_s, parameters.carrier_frequency_hz)
    reference_gain = reference_free_space_gain(wavelength, parameters.reference_distance_m)
    gain = reference_gain * (distance / parameters.reference_distance_m) ** (-parameters.pathloss_exponent)
    return float(gain * shadowing_db_to_linear_gain(shadowing_db) * fading)


def communication_sinr(
    allocated_power_w: float, path_gain: float, allocated_bandwidth_hz: float,
    noise_psd_w_per_hz: float, noise_factor_linear: float,
) -> float:
    power = _finite(allocated_power_w, "allocated_power_w", minimum=0.0)
    gain = _finite(path_gain, "path_gain", minimum=0.0)
    bandwidth = _finite(allocated_bandwidth_hz, "allocated_bandwidth_hz", minimum=0.0)
    noise_psd = _finite(noise_psd_w_per_hz, "noise_psd_w_per_hz", minimum=0.0, strict=True)
    noise_factor = _finite(noise_factor_linear, "noise_factor_linear", minimum=0.0, strict=True)
    if power == 0.0 or bandwidth == 0.0 or gain == 0.0:
        return 0.0
    return float(np.float64(power) * np.float64(gain) / (
        np.float64(noise_psd) * np.float64(noise_factor) * np.float64(bandwidth)
    ))


def achievable_rate_bit_per_s(bandwidth_hz: float, sinr: float, implementation_gap_linear: float) -> float:
    bandwidth = _finite(bandwidth_hz, "bandwidth_hz", minimum=0.0)
    ratio = _finite(sinr, "sinr", minimum=0.0)
    gap = _finite(implementation_gap_linear, "implementation_gap_linear", minimum=0.0, strict=True)
    if gap < 1.0:
        raise QualityValidationError("implementation_gap_linear must be >= 1")
    if bandwidth == 0.0 or ratio == 0.0:
        return 0.0
    return float(np.float64(bandwidth) * np.log2(np.float64(1.0) + np.float64(ratio) / np.float64(gap)))


def served_rate_bit_per_s(demand_bit_per_s: float, achievable_rate: float) -> float:
    demand = _finite(demand_bit_per_s, "demand_bit_per_s", minimum=0.0)
    capacity = _finite(achievable_rate, "achievable_rate", minimum=0.0)
    return float(min(demand, capacity))


def evaluate_communication_quality(
    user_position_m: Iterable[float], bs_position_m: Iterable[float], allocated_bandwidth_hz: float,
    allocated_power_w: float, demand_bit_per_s: float, shadowing_db: float, fading_power_gain: float,
    parameters: CommunicationParameters,
) -> CommunicationQuality:
    if not isinstance(parameters, CommunicationParameters):
        raise QualityValidationError("parameters must be CommunicationParameters")
    distance = link_distance_m(user_position_m, bs_position_m, parameters.minimum_link_distance_m)
    gain = communication_path_gain(distance, parameters, shadowing_db, fading_power_gain)
    bandwidth = _finite(allocated_bandwidth_hz, "allocated_bandwidth_hz", minimum=0.0)
    power = _finite(allocated_power_w, "allocated_power_w", minimum=0.0)
    if bandwidth == 0.0 or power == 0.0:
        return CommunicationQuality(distance, gain, 0.0, 0.0, 0.0)
    sinr = communication_sinr(power, gain, bandwidth, parameters.noise_psd_w_per_hz,
                              parameters.noise_factor_linear)
    capacity = achievable_rate_bit_per_s(bandwidth, sinr, parameters.implementation_gap_linear)
    return CommunicationQuality(distance, gain, sinr, capacity, served_rate_bit_per_s(demand_bit_per_s, capacity))


def monostatic_echo_gain(
    distance_m: float, target_rcs_m2: float, shadowing_db: float, fading_power_gain: float,
    parameters: SensingParameters,
) -> float:
    if not isinstance(parameters, SensingParameters):
        raise QualityValidationError("parameters must be SensingParameters")
    distance = max(
        _finite(distance_m, "distance_m", minimum=0.0, strict=True), parameters.minimum_link_distance_m,
    )
    rcs = _finite(target_rcs_m2, "target_rcs_m2", minimum=0.0, strict=True)
    fading = _finite(fading_power_gain, "fading_power_gain", minimum=0.0)
    wavelength = wavelength_m(parameters.propagation_speed_m_per_s, parameters.carrier_frequency_hz)
    numerator = parameters.combined_frontend_gain_linear * wavelength**2 * rcs
    denominator = (4.0 * pi) ** 3 * distance**4 * parameters.system_loss_linear
    return float(numerator / denominator * shadowing_db_to_linear_gain(shadowing_db) * fading)


def sensing_sinr(
    sensing_power_w: float, echo_gain: float, sensing_bandwidth_hz: float,
    noise_psd_w_per_hz: float, noise_factor_linear: float,
) -> float:
    power = _finite(sensing_power_w, "sensing_power_w", minimum=0.0)
    gain = _finite(echo_gain, "echo_gain", minimum=0.0)
    bandwidth = _finite(sensing_bandwidth_hz, "sensing_bandwidth_hz", minimum=0.0)
    noise_psd = _finite(noise_psd_w_per_hz, "noise_psd_w_per_hz", minimum=0.0, strict=True)
    noise_factor = _finite(noise_factor_linear, "noise_factor_linear", minimum=0.0, strict=True)
    if power == 0.0 or bandwidth == 0.0 or gain == 0.0:
        return 0.0
    return float(np.float64(power) * np.float64(gain) / (
        np.float64(noise_psd) * np.float64(noise_factor) * np.float64(bandwidth)
    ))


def false_alarm_threshold(false_alarm_probability: float) -> float:
    probability = _probability(false_alarm_probability, "false_alarm_probability", open_interval=True)
    return float(-2.0 * log(probability))


def detection_probability(sensing_sinr_value: float, false_alarm_probability: float) -> float:
    ratio = _finite(sensing_sinr_value, "sensing_sinr", minimum=0.0)
    threshold = false_alarm_threshold(false_alarm_probability)
    probability = float(ncx2.sf(threshold, df=2, nc=2.0 * ratio))
    if not isfinite(probability):
        raise QualityValidationError("detection probability evaluation produced a non-finite result")
    return float(np.clip(probability, 0.0, 1.0))


def rms_bandwidth_hz(sensing_bandwidth_hz: float) -> float:
    bandwidth = _finite(sensing_bandwidth_hz, "sensing_bandwidth_hz", minimum=0.0, strict=True)
    return float(np.float64(bandwidth) / np.sqrt(np.float64(12.0)))


def range_variance_m2(propagation_speed_m_per_s: float, rms_bandwidth: float, sensing_sinr_value: float) -> float:
    speed = _finite(propagation_speed_m_per_s, "propagation_speed_m_per_s", minimum=0.0, strict=True)
    bandwidth = _finite(rms_bandwidth, "rms_bandwidth", minimum=0.0, strict=True)
    ratio = _finite(sensing_sinr_value, "sensing_sinr", minimum=0.0, strict=True)
    return float(speed**2 / (32.0 * pi**2 * bandwidth**2 * ratio))


def bearing_fisher_coefficient(effective_aperture_m: float, wavelength: float) -> float:
    aperture = _finite(effective_aperture_m, "effective_aperture_m", minimum=0.0, strict=True)
    wavelength_value = _finite(wavelength, "wavelength", minimum=0.0, strict=True)
    return float((2.0 * pi * aperture / wavelength_value) ** 2 / 12.0)


def bearing_variance_rad2(sensing_sinr_value: float, fisher_coefficient: float) -> float:
    ratio = _finite(sensing_sinr_value, "sensing_sinr", minimum=0.0, strict=True)
    coefficient = _finite(fisher_coefficient, "fisher_coefficient", minimum=0.0, strict=True)
    return float(1.0 / (2.0 * ratio * coefficient))


def polar_position_jacobian(distance_m: float, bearing_rad: float) -> Matrix2:
    distance = _finite(distance_m, "distance_m", minimum=0.0, strict=True)
    bearing = _finite(bearing_rad, "bearing_rad")
    matrix = np.array([
        [cos(bearing), sin(bearing)], [-sin(bearing) / distance, cos(bearing) / distance],
    ], dtype=np.float64)
    return _matrix2_tuple(matrix)


def position_fisher_information(
    distance_m: float, bearing_rad: float, range_variance: float, bearing_variance: float,
) -> Matrix2:
    range_value = _finite(range_variance, "range_variance", minimum=0.0, strict=True)
    bearing_value = _finite(bearing_variance, "bearing_variance", minimum=0.0, strict=True)
    jacobian = _matrix(polar_position_jacobian(distance_m, bearing_rad), (2, 2), "polar_jacobian")
    inverse_measurement_covariance = np.diag([1.0 / range_value, 1.0 / bearing_value])
    information = _symmetric(jacobian.T @ inverse_measurement_covariance @ jacobian,
                             "position_fim", positive_definite=True)
    return _matrix2_tuple(information)


def position_covariance_bound(position_fim: Iterable[Iterable[float]]) -> Matrix2:
    information = _symmetric(_matrix(position_fim, (2, 2), "position_fim"),
                             "position_fim", positive_definite=True)
    try:
        covariance = np.linalg.solve(information, np.eye(2, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise QualityValidationError("position_fim is singular") from exc
    covariance = _symmetric(covariance, "position_covariance", positive_definite=True)
    return _matrix2_tuple(covariance)


def position_error_bound_m(position_covariance_m2: Iterable[Iterable[float]]) -> float:
    covariance = _symmetric(_matrix(position_covariance_m2, (2, 2), "position_covariance_m2"),
                            "position_covariance_m2", positive_definite=True)
    value = float(np.sqrt(np.trace(covariance)))
    if not isfinite(value):
        raise QualityValidationError("PEB evaluation produced a non-finite result")
    return value


def evaluate_localization_quality(
    distance_m: float, bearing_rad: float, sensing_bandwidth_hz: float, sensing_sinr_value: float,
    detection_gate_passed: bool, target_in_session_aoi: bool, parameters: SensingParameters,
) -> LocalizationQuality:
    if type(detection_gate_passed) is not bool or type(target_in_session_aoi) is not bool:
        raise QualityValidationError("localization validity inputs must be boolean")
    if not isinstance(parameters, SensingParameters):
        raise QualityValidationError("parameters must be SensingParameters")
    ratio = _finite(sensing_sinr_value, "sensing_sinr", minimum=0.0)
    if not target_in_session_aoi or not detection_gate_passed or ratio == 0.0:
        return LocalizationQuality.invalid()
    bandwidth = rms_bandwidth_hz(sensing_bandwidth_hz)
    range_variance = range_variance_m2(parameters.propagation_speed_m_per_s, bandwidth, ratio)
    wavelength = wavelength_m(parameters.propagation_speed_m_per_s, parameters.carrier_frequency_hz)
    bearing_coefficient = bearing_fisher_coefficient(parameters.effective_aperture_m, wavelength)
    bearing_variance = bearing_variance_rad2(ratio, bearing_coefficient)
    information = position_fisher_information(distance_m, bearing_rad, range_variance, bearing_variance)
    covariance = position_covariance_bound(information)
    return LocalizationQuality(
        True, range_variance, bearing_variance, information, covariance, position_error_bound_m(covariance),
    )


def tracking_transition_matrix(slot_duration_s: float) -> Matrix4:
    duration = _finite(slot_duration_s, "slot_duration_s", minimum=0.0, strict=True)
    return _matrix4_tuple(np.array([
        [1.0, 0.0, duration, 0.0], [0.0, 1.0, 0.0, duration],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64))


def tracking_acceleration_mapping(slot_duration_s: float) -> tuple[tuple[float, float], ...]:
    duration = _finite(slot_duration_s, "slot_duration_s", minimum=0.0, strict=True)
    half_squared = duration**2 / 2.0
    matrix = np.array([
        [half_squared, 0.0], [0.0, half_squared], [duration, 0.0], [0.0, duration],
    ], dtype=np.float64)
    return tuple(tuple(float(cell) for cell in row) for row in matrix)


def tracking_process_covariance(slot_duration_s: float, acceleration_std_m_per_s2: float) -> Matrix4:
    acceleration_std = _finite(
        acceleration_std_m_per_s2, "acceleration_std_m_per_s2", minimum=0.0,
    )
    mapping = _matrix(tracking_acceleration_mapping(slot_duration_s), (4, 2), "acceleration_mapping")
    covariance = acceleration_std**2 * mapping @ mapping.T
    return _matrix4_tuple(_symmetric(covariance, "tracking_process_covariance"))


def predict_tracking_covariance(
    posterior_covariance: Iterable[Iterable[float]], slot_duration_s: float,
    acceleration_std_m_per_s2: float,
) -> Matrix4:
    posterior = _symmetric(_matrix(posterior_covariance, (4, 4), "posterior_covariance"),
                           "posterior_covariance", positive_definite=True)
    transition = _matrix(tracking_transition_matrix(slot_duration_s), (4, 4), "tracking_transition")
    process = _matrix(tracking_process_covariance(slot_duration_s, acceleration_std_m_per_s2),
                      (4, 4), "tracking_process_covariance")
    predicted = transition @ posterior @ transition.T + process
    return _matrix4_tuple(_symmetric(predicted, "predicted_covariance", positive_definite=True))


def tracking_pcrb_m(posterior_covariance: Iterable[Iterable[float]]) -> float:
    covariance = _symmetric(_matrix(posterior_covariance, (4, 4), "posterior_covariance"),
                            "posterior_covariance", positive_definite=True)
    value = float(np.sqrt(np.trace(covariance[:2, :2])))
    if not isfinite(value):
        raise QualityValidationError("PCRB evaluation produced a non-finite result")
    return value


def tracking_measurement_update(
    prior_covariance: Iterable[Iterable[float]], position_covariance_m2: Iterable[Iterable[float]] | None,
    *, measurement_valid: bool,
) -> TrackingQuality:
    if type(measurement_valid) is not bool:
        raise QualityValidationError("measurement_valid must be boolean")
    prior = _symmetric(_matrix(prior_covariance, (4, 4), "prior_covariance"),
                       "prior_covariance", positive_definite=True)
    if not measurement_valid:
        posterior = _matrix4_tuple(prior)
        return TrackingQuality(False, posterior, tracking_pcrb_m(posterior))
    if position_covariance_m2 is None:
        raise QualityValidationError("valid tracking update requires position covariance")
    measurement = _symmetric(_matrix(position_covariance_m2, (2, 2), "position_covariance_m2"),
                             "position_covariance_m2", positive_definite=True)
    selector = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    try:
        prior_information = np.linalg.solve(prior, np.eye(4, dtype=np.float64))
        measurement_information = np.linalg.solve(measurement, np.eye(2, dtype=np.float64))
        posterior = np.linalg.solve(
            prior_information + selector.T @ measurement_information @ selector,
            np.eye(4, dtype=np.float64),
        )
    except np.linalg.LinAlgError as exc:
        raise QualityValidationError("tracking information update is singular") from exc
    posterior = _symmetric(posterior, "posterior_covariance", positive_definite=True)
    posterior_tuple = _matrix4_tuple(posterior)
    return TrackingQuality(True, posterior_tuple, tracking_pcrb_m(posterior_tuple))


def evaluate_shared_sensing_quality(
    session: SensingSession, target_position_m: Iterable[float], bs_position_m: Iterable[float],
    target_rcs_m2: float, shadowing_db: float, fading_power_gain: float,
    parameters: SensingParameters, tracking_prior_covariance: Iterable[Iterable[float]] | None = None,
) -> SharedSensingQuality:
    if not isinstance(session, SensingSession):
        raise QualityValidationError("session must be a SensingSession")
    if not isinstance(parameters, SensingParameters):
        raise QualityValidationError("parameters must be SensingParameters")
    position = _vector2(target_position_m, "target_position_m")
    distance = link_distance_m(position, bs_position_m, parameters.minimum_link_distance_m)
    bearing = target_bearing_rad(position, bs_position_m)
    in_region = point_in_disk(position, session.aoi)
    echo = monostatic_echo_gain(distance, target_rcs_m2, shadowing_db, fading_power_gain, parameters)
    ratio = sensing_sinr(
        session.profile.sensing_power_w, echo, session.profile.sensing_bandwidth_hz,
        parameters.noise_psd_w_per_hz, parameters.noise_factor_linear,
    )
    probability = detection_probability(ratio, parameters.false_alarm_probability)
    gate_passed = bool(probability >= parameters.detection_gate_probability)
    localization = evaluate_localization_quality(
        distance, bearing, session.profile.sensing_bandwidth_hz, ratio, gate_passed, in_region, parameters,
    )
    tracking: TrackingQuality | None = None
    if Task.TRACKING in session.exposed_outputs:
        prior = tracking_prior_covariance if tracking_prior_covariance is not None else session.tracking_covariance
        if prior is None:
            raise QualityValidationError("tracking-capable session requires an already-defined prior")
        tracking = tracking_measurement_update(
            prior, localization.position_covariance_m2,
            measurement_valid=bool(in_region and gate_passed and localization.information_valid),
        )
    return SharedSensingQuality(
        in_region, distance, bearing, echo, ratio, probability, gate_passed,
        localization, tracking,
    )