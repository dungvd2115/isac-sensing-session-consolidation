from __future__ import annotations

from pathlib import Path

from isac_ssc.utils.calibration import (
    QUANTILE_LEVELS, deterministic_physical_grid, dominance_pairs, run_physical_calibration,
)
from isac_ssc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/"configs/env/default.yaml"
CONFIG = load_config(DEFAULT_CONFIG)


def test_canonical_physical_calibration_is_finite_and_dominant() -> None:
    report = run_physical_calibration(CONFIG, seed=41, sample_count=10_000)
    assert report.sample_count == 10_000 and report.quantile_levels == QUANTILE_LEVELS
    grid = deterministic_physical_grid(CONFIG)
    assert report.passed and grid.passed and grid.evaluated_states > 0
    assert 0 < grid.no_information_states < grid.evaluated_states
    assert dominance_pairs(CONFIG) == (
        ("balanced", "economical"), ("precision", "economical"),
        ("rapid", "economical"), ("rapid", "balanced"),
    )
    for profile in report.profiles:
        assert profile.localization_valid_count+profile.localization_invalid_count == report.sample_count
        assert profile.nonfinite_failure_count == 0
        assert len(profile.detection_probability_quantiles) == len(QUANTILE_LEVELS)
        assert len(profile.pcrb_quantiles_m) == len(QUANTILE_LEVELS)
        assert tuple(sorted(profile.detection_probability_quantiles)) == profile.detection_probability_quantiles
        assert tuple(sorted(profile.peb_quantiles_m)) == profile.peb_quantiles_m
        assert tuple(sorted(profile.pcrb_quantiles_m)) == profile.pcrb_quantiles_m
    assert all(item.passed for item in report.dominance)


def test_calibration_is_reproducible_for_an_explicit_sample_design() -> None:
    first = run_physical_calibration(CONFIG, seed=42, sample_count=256)
    second = run_physical_calibration(CONFIG, seed=42, sample_count=256)
    different = run_physical_calibration(CONFIG, seed=43, sample_count=256)
    assert first == second
    assert different.sample_count == first.sample_count and different.seed != first.seed
    assert tuple(item.profile_id for item in different.profiles) == tuple(item.profile_id for item in first.profiles)
    assert different.passed