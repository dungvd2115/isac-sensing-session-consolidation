from __future__ import annotations

from dataclasses import replace
from math import pi, sqrt

import numpy as np
import pytest
from scipy.stats import ncx2

from isac_ssc.core.entities import DiskAOI, RequestState, ResourceProfile, SensingRequest, SensingSession, Task
from isac_ssc.core.quality import (
    CommunicationParameters, LocalizationQuality, QualityValidationError, SensingParameters,
    achievable_rate_bit_per_s, aoi_coverage_ratio, bearing_fisher_coefficient,
    bearing_variance_rad2, communication_path_gain, db_to_linear, dbm_per_hz_to_w_per_hz,
    detection_probability, disk_intersection_area_m2, evaluate_communication_quality,
    evaluate_localization_quality, evaluate_shared_sensing_quality, false_alarm_threshold,
    link_distance_m, monostatic_echo_gain, point_in_disk, polar_position_jacobian,
    position_covariance_bound, position_error_bound_m, position_fisher_information,
    predict_tracking_covariance, range_variance_m2, reference_free_space_gain, rms_bandwidth_hz,
    sensing_sinr, shadowing_db_to_linear_gain, target_bearing_rad, tracking_acceleration_mapping,
    tracking_measurement_update, tracking_pcrb_m, tracking_process_covariance,
    tracking_transition_matrix, wavelength_m,
)
from isac_ssc.utils.config import load_config

CONFIG = load_config()
COMMUNICATION = CommunicationParameters.from_config(CONFIG)
SENSING = SensingParameters.from_config(CONFIG)
BS = tuple(CONFIG.geometry["bs_position_m"])
DURATIONS = CONFIG.service_duration_slots
TRACKING_DIAG = CONFIG.sensing["tracking"]["initial_covariance_diag"]
INITIAL_TRACKING = tuple(
    tuple(float(TRACKING_DIAG[row]) if row == column else 0.0 for column in range(4))
    for row in range(4)
)


def _request(task: Task = Task.LOCALIZATION, request_id: int = 1) -> SensingRequest:
    threshold = 0.9 if task is Task.DETECTION else 5.0
    return SensingRequest(
        request_id, "tenant_1", 0, 4, DiskAOI((80.0, 0.0), 30.0), 1, task,
        threshold, 2, 1.0, True,
    ).transition(RequestState.ACTIVE, slot=0)


def _session(
    profile: ResourceProfile | None = None, task: Task = Task.LOCALIZATION,
    request_id: int = 1, session_id: int = 1,
) -> SensingSession:
    request = _request(task, request_id)
    tracking = INITIAL_TRACKING if task is Task.TRACKING else None
    return SensingSession.create(
        session_id, request, profile or CONFIG.resource_profiles["balanced"], 0, DURATIONS, tracking,
    )


def _shared(session: SensingSession, position=(80.0, 0.0), fading=1.0):
    return evaluate_shared_sensing_quality(
        session, position, BS, 1.0, 0.0, fading, SENSING,
        session.tracking_covariance,
    )


def test_unit_conversions_and_reference_gain() -> None:
    assert db_to_linear(10.0) == pytest.approx(10.0)
    assert shadowing_db_to_linear_gain(3.0) == pytest.approx(10.0 ** -0.3)
    assert dbm_per_hz_to_w_per_hz(-174.0) == pytest.approx(10.0 ** -20.4)
    wavelength = wavelength_m(299792458.0, 6.0e9)
    assert wavelength == pytest.approx(299792458.0 / 6.0e9)
    assert reference_free_space_gain(wavelength, 1.0) == pytest.approx((wavelength / (4.0 * pi)) ** 2)


def test_geometry_distance_bearing_and_disk_membership() -> None:
    assert link_distance_m((3.0, 4.0), (0.0, 0.0), 10.0) == 10.0
    assert link_distance_m((30.0, 40.0), (0.0, 0.0), 10.0) == 50.0
    assert target_bearing_rad((0.0, 2.0), (0.0, 0.0)) == pytest.approx(pi / 2.0)
    disk = DiskAOI((0.0, 0.0), 5.0)
    assert point_in_disk((0.0, 0.0), disk)
    assert point_in_disk((3.0, 4.0), disk)
    assert not point_in_disk((5.1, 0.0), disk)


def test_exact_disk_intersection_and_request_coverage() -> None:
    base = DiskAOI((0.0, 0.0), 2.0)
    assert disk_intersection_area_m2(base, DiskAOI((5.0, 0.0), 2.0)) == 0.0
    assert disk_intersection_area_m2(base, DiskAOI((0.0, 0.0), 1.0)) == pytest.approx(pi)
    partial = disk_intersection_area_m2(base, DiskAOI((2.0, 0.0), 2.0))
    expected = 2.0 * 2.0**2 * np.arccos(0.5) - 0.5 * sqrt(4.0**2 - 2.0**2) * 2.0
    assert partial == pytest.approx(expected)
    assert aoi_coverage_ratio(base, base) == pytest.approx(1.0)
    assert aoi_coverage_ratio(base, DiskAOI((0.0, 0.0), 1.0)) == pytest.approx(0.25)


def test_communication_path_gain_and_rate_decrease_with_distance() -> None:
    near = communication_path_gain(20.0, COMMUNICATION, 0.0, 1.0)
    far = communication_path_gain(40.0, COMMUNICATION, 0.0, 1.0)
    assert near > far
    assert near / far == pytest.approx(2.0 ** COMMUNICATION.pathloss_exponent)

    near_quality = evaluate_communication_quality(
        (20.0, 0.0), BS, 5.0e6, 2.0, 1.0e9, 0.0, 1.0, COMMUNICATION,
    )
    far_quality = evaluate_communication_quality(
        (40.0, 0.0), BS, 5.0e6, 2.0, 1.0e9, 0.0, 1.0, COMMUNICATION,
    )
    assert near_quality.achievable_rate_bit_per_s > far_quality.achievable_rate_bit_per_s


def test_communication_rate_equation_power_monotonicity_and_served_cap() -> None:
    low = evaluate_communication_quality(
        (50.0, 0.0), BS, 5.0e6, 1.0, 3.0e6, 0.0, 1.0, COMMUNICATION,
    )
    high = evaluate_communication_quality(
        (50.0, 0.0), BS, 5.0e6, 2.0, 3.0e6, 0.0, 1.0, COMMUNICATION,
    )
    expected = achievable_rate_bit_per_s(5.0e6, high.sinr, COMMUNICATION.implementation_gap_linear)
    assert high.achievable_rate_bit_per_s == pytest.approx(expected)
    assert high.achievable_rate_bit_per_s > low.achievable_rate_bit_per_s
    assert high.served_rate_bit_per_s == pytest.approx(min(3.0e6, expected))


def test_zero_communication_boundaries_never_create_nan_or_indeterminate_sinr() -> None:
    cases = (
        evaluate_communication_quality((50.0, 0.0), BS, 0.0, 1.0, 1.0, 0.0, 1.0, COMMUNICATION),
        evaluate_communication_quality((50.0, 0.0), BS, 1.0, 0.0, 1.0, 0.0, 1.0, COMMUNICATION),
        evaluate_communication_quality((50.0, 0.0), BS, 1.0, 1.0, 1.0, 0.0, 0.0, COMMUNICATION),
    )
    for quality in cases:
        assert quality.sinr == quality.achievable_rate_bit_per_s == quality.served_rate_bit_per_s == 0.0
        assert np.all(np.isfinite([quality.path_gain, quality.sinr, quality.achievable_rate_bit_per_s]))
    off = evaluate_communication_quality((50.0, 0.0), BS, 1.0e6, 1.0, 0.0, 0.0, 1.0, COMMUNICATION)
    assert off.served_rate_bit_per_s == 0.0


def test_monostatic_echo_obeys_inverse_fourth_power_and_sensing_sinr_equation() -> None:
    near = monostatic_echo_gain(40.0, 1.0, 0.0, 1.0, SENSING)
    far = monostatic_echo_gain(80.0, 1.0, 0.0, 1.0, SENSING)
    assert near / far == pytest.approx(16.0)
    profile = CONFIG.resource_profiles["balanced"]
    ratio = sensing_sinr(
        profile.sensing_power_w, far, profile.sensing_bandwidth_hz,
        SENSING.noise_psd_w_per_hz, SENSING.noise_factor_linear,
    )
    expected = profile.sensing_power_w * far / (
        SENSING.noise_psd_w_per_hz * SENSING.noise_factor_linear * profile.sensing_bandwidth_hz
    )
    assert ratio == pytest.approx(expected)


def test_detection_uses_noncentral_chi_square_survival_function_and_is_monotone() -> None:
    assert false_alarm_threshold(1.0e-4) == pytest.approx(-2.0 * np.log(1.0e-4))
    ratio = 3.0
    expected = ncx2.sf(false_alarm_threshold(1.0e-4), df=2, nc=2.0 * ratio)
    assert detection_probability(ratio, 1.0e-4) == pytest.approx(expected)
    assert detection_probability(10.0, 1.0e-4) > detection_probability(1.0, 1.0e-4)


def test_localization_primitives_match_declared_equations() -> None:
    bandwidth = 4.0e6
    rms = rms_bandwidth_hz(bandwidth)
    assert rms == pytest.approx(bandwidth / sqrt(12.0))
    ratio = 5.0
    range_variance = range_variance_m2(SENSING.propagation_speed_m_per_s, rms, ratio)
    assert range_variance == pytest.approx(
        SENSING.propagation_speed_m_per_s**2 / (32.0 * pi**2 * rms**2 * ratio)
    )
    wavelength = wavelength_m(SENSING.propagation_speed_m_per_s, SENSING.carrier_frequency_hz)
    coefficient = bearing_fisher_coefficient(SENSING.effective_aperture_m, wavelength)
    assert coefficient == pytest.approx((2.0 * pi * SENSING.effective_aperture_m / wavelength) ** 2 / 12.0)
    assert bearing_variance_rad2(ratio, coefficient) == pytest.approx(1.0 / (2.0 * ratio * coefficient))


def test_position_fim_covariance_and_peb_are_symmetric_finite_and_consistent() -> None:
    jacobian = np.asarray(polar_position_jacobian(100.0, 0.4))
    assert jacobian.shape == (2, 2)
    information = np.asarray(position_fisher_information(100.0, 0.4, 4.0, 0.01))
    covariance = np.asarray(position_covariance_bound(information))
    assert np.allclose(information, information.T)
    assert np.allclose(covariance, covariance.T)
    assert np.allclose(information @ covariance, np.eye(2), rtol=1e-10, atol=1e-10)
    assert position_error_bound_m(covariance) == pytest.approx(sqrt(np.trace(covariance)))


def test_localization_invalid_boundaries_use_explicit_none_not_nan() -> None:
    outside = evaluate_localization_quality(80.0, 0.0, 4.0e6, 10.0, True, False, SENSING)
    zero = evaluate_localization_quality(80.0, 0.0, 4.0e6, 0.0, False, True, SENSING)
    for quality in (outside, zero):
        assert quality == LocalizationQuality.invalid()
        assert quality.peb_m is None and quality.position_fim is None


def test_peb_improves_with_sensing_information() -> None:
    low = evaluate_localization_quality(80.0, 0.2, 4.0e6, 2.0, True, True, SENSING)
    high = evaluate_localization_quality(80.0, 0.2, 4.0e6, 8.0, True, True, SENSING)
    assert low.information_valid and high.information_valid
    assert high.peb_m < low.peb_m


def test_tracking_transition_mapping_process_and_one_slot_prediction() -> None:
    duration, acceleration_std = 0.1, 1.0
    transition = np.asarray(tracking_transition_matrix(duration))
    mapping = np.asarray(tracking_acceleration_mapping(duration))
    expected_transition = np.array([
        [1.0, 0.0, duration, 0.0], [0.0, 1.0, 0.0, duration],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ])
    expected_mapping = np.array([
        [duration**2 / 2.0, 0.0], [0.0, duration**2 / 2.0],
        [duration, 0.0], [0.0, duration],
    ])
    assert np.allclose(transition, expected_transition)
    assert np.allclose(mapping, expected_mapping)
    process = np.asarray(tracking_process_covariance(duration, acceleration_std))
    assert np.allclose(process, acceleration_std**2 * expected_mapping @ expected_mapping.T)
    predicted = np.asarray(predict_tracking_covariance(INITIAL_TRACKING, duration, acceleration_std))
    assert np.allclose(predicted, transition @ np.asarray(INITIAL_TRACKING) @ transition.T + process)


def test_tracking_valid_update_improves_bound_and_invalid_update_preserves_prior() -> None:
    prior = np.asarray(INITIAL_TRACKING)
    measurement = ((1.0, 0.0), (0.0, 1.0))
    valid = tracking_measurement_update(prior, measurement, measurement_valid=True)
    invalid = tracking_measurement_update(prior, None, measurement_valid=False)
    assert valid.measurement_updated
    assert valid.pcrb_m < tracking_pcrb_m(prior)
    assert np.all(np.asarray(valid.posterior_covariance)[:2, :2] <= prior[:2, :2] + 1e-12)
    assert not invalid.measurement_updated
    assert np.array_equal(np.asarray(invalid.posterior_covariance), prior)
    assert np.isfinite(invalid.pcrb_m)


def test_high_level_tracking_uses_creation_prior_without_prediction() -> None:
    session = _session(task=Task.TRACKING)
    result = _shared(session)
    direct = tracking_measurement_update(
        INITIAL_TRACKING, result.localization.position_covariance_m2,
        measurement_valid=result.localization.information_valid,
    )
    assert result.tracking == direct
    predicted = predict_tracking_covariance(INITIAL_TRACKING, CONFIG.system["slot_duration_s"],
                                            CONFIG.mobility["targets"]["acceleration_std_m_per_s2"])
    assert result.tracking.posterior_covariance != tracking_measurement_update(
        predicted, result.localization.position_covariance_m2,
        measurement_valid=result.localization.information_valid,
    ).posterior_covariance


def test_target_outside_fixed_session_aoi_produces_no_valid_localization_or_tracking_update() -> None:
    session = _session(task=Task.TRACKING)
    result = _shared(session, position=(140.0, 0.0))
    assert not result.target_in_session_aoi and not result.physical_valid
    assert not result.localization.information_valid
    assert result.tracking is not None and not result.tracking.measurement_updated
    assert result.tracking.posterior_covariance == INITIAL_TRACKING


def test_target_motion_changes_echo_bearing_fim_and_derived_quality() -> None:
    session = _session(task=Task.LOCALIZATION)
    first = _shared(session, position=(80.0, 0.0))
    moved = _shared(session, position=(75.0, 15.0))
    assert first.distance_m != moved.distance_m
    assert first.bearing_rad != moved.bearing_rad
    assert first.echo_gain != moved.echo_gain
    assert first.localization.position_fim != moved.localization.position_fim
    assert first.localization.peb_m != moved.localization.peb_m


def test_shared_result_is_deterministic_and_independent_of_member_identity() -> None:
    first = _session(request_id=1, session_id=1)
    second = replace(first, session_id=2, member_request_ids=(7, 8))
    first_result = _shared(first)
    assert first_result == _shared(first)
    assert first_result == _shared(second)


def test_comparable_profiles_preserve_detection_localization_and_tracking_dominance() -> None:
    comparable = (("balanced", "economical"), ("precision", "economical"),
                  ("rapid", "balanced"), ("rapid", "economical"))
    for stronger_name, weaker_name in comparable:
        stronger = _shared(_session(CONFIG.resource_profiles[stronger_name], Task.TRACKING))
        weaker = _shared(_session(CONFIG.resource_profiles[weaker_name], Task.TRACKING))
        assert stronger.detection_probability >= weaker.detection_probability
        assert stronger.localization.peb_m <= weaker.localization.peb_m
        assert stronger.tracking.pcrb_m <= weaker.tracking.pcrb_m


def test_unexpected_nonfinite_or_singular_inputs_fail_loudly() -> None:
    with pytest.raises(QualityValidationError):
        position_covariance_bound(((1.0, 0.0), (0.0, 0.0)))
    with pytest.raises(QualityValidationError):
        tracking_measurement_update(((float("nan"),) * 4,) * 4, None, measurement_valid=False)