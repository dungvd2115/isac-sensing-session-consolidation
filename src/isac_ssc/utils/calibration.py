"""Physical diagnostics for the canonical sensing model."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, log10, pi, sin
from typing import Iterable

import numpy as np

from isac_ssc.core.entities import DiskAOI, SensingSession, Task, task_outputs
from isac_ssc.core.quality import SensingParameters, SharedSensingQuality, evaluate_shared_sensing_quality
from isac_ssc.core.resources import profile_dominates
from isac_ssc.utils.config import CanonicalConfig

QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class GridDiagnostics:
    evaluated_states: int
    no_information_states: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class ProfileDiagnostics:
    profile_id: str
    detection_probability_quantiles: tuple[float, ...]
    peb_quantiles_m: tuple[float, ...]
    pcrb_quantiles_m: tuple[float, ...]
    localization_valid_count: int
    localization_invalid_count: int
    nonfinite_failure_count: int


@dataclass(frozen=True, slots=True)
class DominanceDiagnostics:
    stronger_profile_id: str
    weaker_profile_id: str
    compared_peb_samples: int
    no_information_samples: int
    violation_count: int

    @property
    def passed(self) -> bool:
        return self.violation_count == 0


@dataclass(frozen=True, slots=True)
class BoundaryDiagnostics:
    zero_fading_explicit_no_information: bool
    creation_prior_used_without_prediction: bool
    deterministic_repeat: bool

    @property
    def passed(self) -> bool:
        return all((
            self.zero_fading_explicit_no_information,
            self.creation_prior_used_without_prediction,
            self.deterministic_repeat,
        ))


@dataclass(frozen=True, slots=True)
class PhysicalCalibration:
    sample_count: int
    seed: int
    quantile_levels: tuple[float, ...]
    profiles: tuple[ProfileDiagnostics, ...]
    dominance: tuple[DominanceDiagnostics, ...]
    boundaries: BoundaryDiagnostics

    @property
    def passed(self) -> bool:
        return all((
            self.sample_count > 0, self.boundaries.passed,
            all(item.nonfinite_failure_count == 0 for item in self.profiles),
            all(item.passed for item in self.dominance),
        ))


def dominance_pairs(config: CanonicalConfig) -> tuple[tuple[str, str], ...]:
    profiles = tuple(config.resource_profiles.items())
    return tuple(
        (stronger_name, weaker_name)
        for stronger_name, stronger in profiles
        for weaker_name, weaker in profiles
        if stronger_name != weaker_name and profile_dominates(stronger, weaker)
    )


def _initial_tracking_covariance(config: CanonicalConfig) -> tuple[tuple[float, ...], ...]:
    diagonal = tuple(float(value) for value in config.sensing["tracking"]["initial_covariance_diag"])
    return tuple(tuple(diagonal[row] if row == column else 0.0 for column in range(4)) for row in range(4))


def _calibration_sessions(config: CanonicalConfig) -> tuple[SensingSession, ...]:
    bs = tuple(config.geometry["bs_position_m"])
    maximum_range = float(config.geometry["target_initial_position"]["maximum_radius_m"])
    covariance = _initial_tracking_covariance(config)
    aoi = DiskAOI(bs, maximum_range+1.0)
    return tuple(
        SensingSession(
            profile_id, 0, aoi, 0, Task.TRACKING, task_outputs(Task.TRACKING), (0,), profile,
            0, config.system["horizon_slots"]-1, covariance,
        )
        for profile_id, profile in config.resource_profiles.items()
    )


def _evaluate_state(
    session: SensingSession, position: tuple[float, float], rcs: float,
    shadowing_db: float, fading_power: float, parameters: SensingParameters,
) -> SharedSensingQuality:
    return evaluate_shared_sensing_quality(
        session, position, session.aoi.center_m, rcs, shadowing_db, fading_power,
        parameters, session.tracking_covariance,
    )


def deterministic_physical_grid(config: CanonicalConfig) -> GridDiagnostics:
    parameters = SensingParameters.from_config(config)
    sessions = _calibration_sessions(config)
    geometry = config.geometry["target_initial_position"]
    minimum, maximum = float(geometry["minimum_radius_m"]), float(geometry["maximum_radius_m"])
    distances = (minimum, 0.5*(minimum+maximum), maximum)
    bearings = (-pi, -pi/2.0, 0.0, pi/2.0)
    shadow_std = float(config.sensing["shadowing_std_db"])
    shadowing_values = (-shadow_std, 0.0, shadow_std)
    fading_values = (0.0, 0.5, 1.0)
    median = float(config.sensing["rcs"]["median_m2"])
    rcs_std = float(config.sensing["rcs"]["dbsm_std_db"])
    rcs_values = (median*10.0**(-rcs_std/10.0), median, median*10.0**(rcs_std/10.0))
    profile_names = tuple(config.resource_profiles)
    pairs = dominance_pairs(config)
    evaluated = no_information = 0
    failures: list[str] = []

    for distance in distances:
        for bearing in bearings:
            position = distance*cos(bearing), distance*sin(bearing)
            for shadowing in shadowing_values:
                for fading in fading_values:
                    for rcs in rcs_values:
                        results = tuple(
                            _evaluate_state(session, position, rcs, shadowing, fading, parameters)
                            for session in sessions
                        )
                        evaluated += len(results)
                        no_information += sum(not result.localization.information_valid for result in results)
                        if any(not isfinite(result.detection_probability) for result in results):
                            failures.append("non-finite detection probability")
                        if any(result.tracking is None or not isfinite(result.tracking.pcrb_m) for result in results):
                            failures.append("non-finite tracking PCRB")
                        if any(
                            result != _evaluate_state(session, position, rcs, shadowing, fading, parameters)
                            for session, result in zip(sessions, results, strict=True)
                        ):
                            failures.append("non-deterministic primitive result")
                        for stronger_name, weaker_name in pairs:
                            stronger = results[profile_names.index(stronger_name)]
                            weaker = results[profile_names.index(weaker_name)]
                            if stronger.detection_probability+1e-12 < weaker.detection_probability:
                                failures.append(f"PD dominance failed: {stronger_name}/{weaker_name}")
                            if weaker.localization.information_valid and not stronger.localization.information_valid:
                                failures.append(f"localization dominance failed: {stronger_name}/{weaker_name}")
                            if (
                                stronger.localization.information_valid
                                and weaker.localization.information_valid
                                and stronger.localization.peb_m > weaker.localization.peb_m+1e-10
                            ):
                                failures.append(f"PEB dominance failed: {stronger_name}/{weaker_name}")
                            if stronger.tracking.pcrb_m > weaker.tracking.pcrb_m+1e-10:
                                failures.append(f"PCRB dominance failed: {stronger_name}/{weaker_name}")
    return GridDiagnostics(evaluated, no_information, tuple(sorted(set(failures))))


def _quantiles(values: Iterable[float]) -> tuple[float, ...]:
    data = np.asarray(tuple(values), dtype=np.float64)
    return () if data.size == 0 else tuple(float(value) for value in np.quantile(data, QUANTILE_LEVELS))


def _boundary_diagnostics(config: CanonicalConfig, parameters: SensingParameters) -> BoundaryDiagnostics:
    session = _calibration_sessions(config)[0]
    position = float(config.geometry["target_initial_position"]["minimum_radius_m"]), 0.0
    zero = _evaluate_state(session, position, 1.0, 0.0, 0.0, parameters)
    valid = _evaluate_state(session, position, 1.0, 0.0, 1.0, parameters)
    repeated = _evaluate_state(session, position, 1.0, 0.0, 1.0, parameters)
    no_prediction = zero.tracking is not None and zero.tracking.posterior_covariance == session.tracking_covariance
    return BoundaryDiagnostics(
        not zero.localization.information_valid
        and zero.tracking is not None
        and not zero.tracking.measurement_updated,
        no_prediction,
        valid == repeated,
    )


def run_physical_calibration(
    config: CanonicalConfig, *, seed: int, sample_count: int = 10_000,
) -> PhysicalCalibration:
    """Evaluate primitive sensing quality under an explicit reproducible sample design."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")

    rng = np.random.default_rng(seed)
    minimum = float(config.geometry["target_initial_position"]["minimum_radius_m"])
    maximum = float(config.geometry["target_initial_position"]["maximum_radius_m"])
    distances = np.sqrt(rng.uniform(minimum**2, maximum**2, size=sample_count))
    bearings = rng.uniform(-pi, pi, size=sample_count)
    positions = np.column_stack((distances*np.cos(bearings), distances*np.sin(bearings)))
    shadowing = rng.normal(0.0, float(config.sensing["shadowing_std_db"]), size=sample_count)
    fading = rng.exponential(1.0, size=sample_count)
    median_dbsm = 10.0*log10(float(config.sensing["rcs"]["median_m2"]))
    rcs_dbsm = rng.normal(median_dbsm, float(config.sensing["rcs"]["dbsm_std_db"]), size=sample_count)
    rcs = np.power(10.0, rcs_dbsm/10.0)
    parameters = SensingParameters.from_config(config)
    sessions = _calibration_sessions(config)
    profile_names = tuple(config.resource_profiles)
    detection = {name: np.full(sample_count, np.nan) for name in profile_names}
    peb = {name: np.full(sample_count, np.nan) for name in profile_names}
    pcrb = {name: np.full(sample_count, np.nan) for name in profile_names}
    localization_valid = {name: np.zeros(sample_count, dtype=bool) for name in profile_names}
    failures = {name: 0 for name in profile_names}

    for index in range(sample_count):
        position = float(positions[index, 0]), float(positions[index, 1])
        for name, session in zip(profile_names, sessions, strict=True):
            try:
                result = _evaluate_state(
                    session, position, float(rcs[index]), float(shadowing[index]),
                    float(fading[index]), parameters,
                )
                if (
                    not isfinite(result.detection_probability)
                    or result.tracking is None
                    or not isfinite(result.tracking.pcrb_m)
                ):
                    failures[name] += 1
                    continue
                detection[name][index] = result.detection_probability
                pcrb[name][index] = result.tracking.pcrb_m
                if result.localization.information_valid:
                    localization_valid[name][index] = True
                    peb[name][index] = result.localization.peb_m
            except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                failures[name] += 1

    profiles = []
    for name in profile_names:
        profiles.append(ProfileDiagnostics(
            name,
            _quantiles(detection[name][np.isfinite(detection[name])]),
            _quantiles(peb[name][np.isfinite(peb[name])]),
            _quantiles(pcrb[name][np.isfinite(pcrb[name])]),
            int(localization_valid[name].sum()),
            int(sample_count-localization_valid[name].sum()-failures[name]),
            failures[name],
        ))

    dominance = []
    for stronger_name, weaker_name in dominance_pairs(config):
        valid_pair = np.isfinite(detection[stronger_name]) & np.isfinite(detection[weaker_name])
        violations = int(np.sum(
            detection[stronger_name][valid_pair]+1e-12 < detection[weaker_name][valid_pair]
        ))
        stronger_valid = localization_valid[stronger_name]
        weaker_valid = localization_valid[weaker_name]
        violations += int(np.sum(weaker_valid & ~stronger_valid))
        compared = stronger_valid & weaker_valid
        violations += int(np.sum(peb[stronger_name][compared] > peb[weaker_name][compared]+1e-10))
        pcrb_pair = np.isfinite(pcrb[stronger_name]) & np.isfinite(pcrb[weaker_name])
        violations += int(np.sum(pcrb[stronger_name][pcrb_pair] > pcrb[weaker_name][pcrb_pair]+1e-10))
        violations += int(sample_count-np.sum(valid_pair))+int(sample_count-np.sum(pcrb_pair))
        dominance.append(DominanceDiagnostics(
            stronger_name, weaker_name, int(np.sum(compared)), int(np.sum(~weaker_valid)), violations,
        ))

    return PhysicalCalibration(
        sample_count, seed, QUANTILE_LEVELS, tuple(profiles), tuple(dominance),
        _boundary_diagnostics(config, parameters),
    )