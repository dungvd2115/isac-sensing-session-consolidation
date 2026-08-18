from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
from pathlib import Path

import pytest

from isac_ssc.training.checkpoint import semantic_digest
from isac_ssc.utils.config import ConfigError, DEFAULT_ALGORITHM_CONFIG_PATH, DEFAULT_CONFIG_PATH, load_algorithm_config


def _candidate(tmp_path: Path, replace: tuple[str, str] | None = None, suffix: str = "") -> Path:
    text = DEFAULT_ALGORITHM_CONFIG_PATH.read_text(encoding="utf-8")
    if replace is not None:
        old, new = replace
        assert old in text
        text = text.replace(old, new, 1)
    path = tmp_path / "algorithm.yaml"
    path.write_text(text + suffix, encoding="utf-8")
    return path


def test_canonical_algorithm_config_is_typed_frozen_and_hashed() -> None:
    config = load_algorithm_config()
    content = DEFAULT_ALGORITHM_CONFIG_PATH.read_bytes()
    assert config.source_path == DEFAULT_ALGORITHM_CONFIG_PATH.resolve()
    assert config.content_hash == hashlib.sha256(content).hexdigest()
    assert config.device == "cpu" and config.dtype == "float32"
    assert config.model.hidden_dim == 128 and config.model.activation == "tanh"
    assert config.model.pooling == "masked_mean" and config.model.dropout == 0.0
    assert config.model.orthogonal_initialization is True
    assert config.optimizer.name == "adam" and config.normalization.enabled is True
    assert config.ppo.discount == 1.0
    with pytest.raises(FrozenInstanceError):
        config.model.hidden_dim = 64
    with pytest.raises((TypeError, ValueError)):
        replace(config.model, activation="relu")


def test_fixed_implementation_choices_are_not_exposed_as_fake_config_keys() -> None:
    text = DEFAULT_ALGORITHM_CONFIG_PATH.read_text(encoding="utf-8")
    for key in ("dtype:", "activation:", "pooling:", "dropout:", "orthogonal_initialization:", "name:", "enabled:"):
        assert key not in text


def test_runtime_device_remains_a_real_configuration_choice(tmp_path) -> None:
    config = load_algorithm_config(_candidate(tmp_path, ('device: "cpu"', 'device: "cuda"')))
    assert config.device == "cuda"


def test_model_config_semantic_digest_remains_checkpoint_compatible() -> None:
    @dataclass(frozen=True, slots=True)
    class LegacyModelConfig:
        hidden_dim: int
        profile_embedding_dim: int
        activation: str
        pooling: str
        dropout: float
        orthogonal_initialization: bool

    model = load_algorithm_config().model
    legacy = LegacyModelConfig(
        model.hidden_dim, model.profile_embedding_dim, "tanh", "masked_mean", 0.0, True,
    )
    assert semantic_digest(model) == semantic_digest(legacy)


@pytest.mark.parametrize("replace,suffix", (
    (('schema_version: "isac-ssc-constrained-ppo-v1"\n', ""), ""),
    (None, "\nunknown_key: 1\n"),
    (("hidden_dim: 128", 'hidden_dim: "128"'), ""),
    (("learning_rate: 3.0e-4", "learning_rate: .nan"), ""),
    (('device: "cpu"', 'device: "tpu"'), ""),
    (("discount: 1.0", "discount: 0.99"), ""),
    (("hidden_dim: 128", "hidden_dim: 0"), ""),
    (("clip_ratio: 0.20", "clip_ratio: 1.20"), ""),
    (("maximum: 100.0", "maximum: 0.0"), ""),
))
def test_invalid_algorithm_configuration_is_rejected(tmp_path, replace, suffix) -> None:
    with pytest.raises(ConfigError):
        load_algorithm_config(_candidate(tmp_path, replace, suffix))


def test_legacy_fixed_keys_are_accepted_only_for_compatible_snapshots(tmp_path) -> None:
    text = DEFAULT_ALGORITHM_CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace('  device: "cpu"', '  device: "cpu"\n  dtype: "float32"', 1)
    text = text.replace(
        "  profile_embedding_dim: 32",
        '  profile_embedding_dim: 32\n  activation: "tanh"\n  pooling: "masked_mean"\n  dropout: 0.0\n  orthogonal_initialization: true', 1,
    )
    text = text.replace("optimizer:\n", 'optimizer:\n  name: "adam"\n', 1)
    text = text.replace("normalization:\n", "normalization:\n  enabled: true\n", 1)
    path = tmp_path / "legacy.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_algorithm_config(path)
    assert config.dtype == "float32" and config.optimizer.name == "adam"
    path.write_text(text.replace('activation: "tanh"', 'activation: "relu"'), encoding="utf-8")
    with pytest.raises(ConfigError, match="model.activation"):
        load_algorithm_config(path)


def test_duplicate_algorithm_key_is_rejected(tmp_path) -> None:
    path = _candidate(tmp_path, suffix='\nruntime:\n  device: "cpu"\n')
    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_algorithm_config(path)


def test_scientific_or_environment_keys_cannot_enter_algorithm_config(tmp_path) -> None:
    path = _candidate(tmp_path, suffix="\nreward:\n  sensing_resource_cost_weight: 0.2\n")
    with pytest.raises(ConfigError, match="invalid configuration keys"):
        load_algorithm_config(path)


def test_environment_configuration_is_untouched_by_algorithm_loading() -> None:
    before = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
    load_algorithm_config()
    assert hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest() == before