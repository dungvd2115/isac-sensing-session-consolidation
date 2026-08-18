from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from isac_ssc.core.entities import Task
from isac_ssc.utils.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/"configs/env/default.yaml"


def _data() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path/"candidate.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _reject(tmp_path: Path, mutate) -> None:
    data = deepcopy(_data())
    mutate(data)
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, data))


def test_default_config_loads_as_immutable_scientific_input() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert tuple(config.resource_profiles) == ("economical", "balanced", "precision", "rapid")
    assert config.service_duration_slots[Task.TRACKING] == 8
    assert set(config.tenant("tenant_1").permitted_tasks) == set(Task)
    assert sum(config.task_probabilities.values()) == pytest.approx(1.0)
    assert not hasattr(config, "implementation_limits")
    assert not hasattr(config, "calibration")
    assert not hasattr(config, "validation")
    assert not hasattr(config, "task_output_contract")
    assert not hasattr(config, "profile_dominance")
    assert not hasattr(config, "source_path")
    assert not hasattr(config, "content_hash")
    assert "trace_format_version" not in config.trace_generation
    assert "generator_version" not in config.trace_generation
    with pytest.raises(TypeError):
        config.system["horizon_slots"] = 1


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path/"duplicate.yaml"
    path.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8")+"\nschema_version: duplicate\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_config(path)


def test_missing_required_section_or_field_is_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, lambda data: data.pop("reward"))
    _reject(tmp_path, lambda data: data["system"].pop("total_bandwidth_hz"))


def test_descriptive_metadata_does_not_lock_the_environment_to_default_yaml(tmp_path: Path) -> None:
    data = _data()
    data["profile_name"] = "same_science_custom_experiment"
    data["notes"] = {"owner": "researcher", "purpose": "local sweep"}
    config = load_config(_write(tmp_path, data))
    assert config.profile_name == "same_science_custom_experiment"


def test_probability_and_positive_value_contracts_are_enforced(tmp_path: Path) -> None:
    _reject(tmp_path, lambda data: data["requests"]["task"].__setitem__("probabilities", [0.5, 0.5, 0.5]))
    _reject(tmp_path, lambda data: data["requests"]["update_interval_slots"]["tracking"].__setitem__("values", [0, 1]))
    _reject(tmp_path, lambda data: data["geometry"].__setitem__("minimum_link_distance_m", -1.0))
    _reject(tmp_path, lambda data: data["resource_profiles"]["balanced"].__setitem__("sensing_power_w", 0.0))


def test_tenant_tasks_and_authorization_matrix_are_validated(tmp_path: Path) -> None:
    _reject(tmp_path, lambda data: data["tenant_profiles"]["tenant_1"].__setitem__("permitted_tasks", []))
    _reject(tmp_path, lambda data: data["tenant_profiles"]["tenant_1"].__setitem__("permitted_tasks", ["unknown"]))
    _reject(tmp_path, lambda data: data["sharing_authorization"]["tenant_pair_matrix"][0].__setitem__(1, False))
    _reject(tmp_path, lambda data: data["sharing_authorization"]["tenant_pair_matrix"][0].__setitem__(0, False))


def test_tenant_rows_keep_unambiguous_order(tmp_path: Path) -> None:
    def reorder(data: dict) -> None:
        profiles = data["tenant_profiles"]
        data["tenant_profiles"] = {
            "tenant_2": profiles["tenant_2"], "tenant_1": profiles["tenant_1"],
            "tenant_3": profiles["tenant_3"], "tenant_4": profiles["tenant_4"],
        }
    _reject(tmp_path, reorder)


def test_reward_and_oracle_semantics_remain_scientifically_constrained(tmp_path: Path) -> None:
    _reject(tmp_path, lambda data: data["reward"].__setitem__("finite_horizon_discount_factor", 0.99))
    _reject(tmp_path, lambda data: data["reward"].__setitem__("sensing_cost_power_weight", 0.4))
    _reject(tmp_path, lambda data: data["oracle"]["instance_selection_limits"].__setitem__("max_sessions", 4))


def test_supported_arrival_regimes_and_seed_domain_are_validated(tmp_path: Path) -> None:
    _reject(tmp_path, lambda data: data["trace_generation"].__setitem__("registered_arrival_regimes", ["independent"]))
    _reject(tmp_path, lambda data: data["arrivals"].__setitem__("active_regime", "unsupported"))
    _reject(tmp_path, lambda data: data["trace_generation"]["seeding"].__setitem__("root_seed_maximum", 100))
