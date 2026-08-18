from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.utils.config import DEFAULT_CONFIG_PATH, load_config
from isac_ssc.utils.serialization import (
    SerializationError, deserialize_trace_bytes,
    load_trace, serialize_trace, trace_to_dict,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config()


@pytest.fixture(scope="module")
def independent_trace():
    return generate_primitive_trace(
        CONFIG, 41001, "independent",
    )


@pytest.fixture(scope="module")
def clustered_trace():
    return generate_primitive_trace(
        CONFIG, 41001, "clustered",
    )


def _bytes(value, *, allow_nan=False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=allow_nan,
    ).encode("utf-8")


def _cli_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT/"src")
    return environment


def test_same_seed_and_regime_produce_identical_primitive_trace(
    independent_trace,
) -> None:
    assert generate_primitive_trace(
        CONFIG, 41001, "independent",
    ) == independent_trace


def test_different_regimes_produce_distinct_primitive_traces(
    independent_trace, clustered_trace,
) -> None:
    assert independent_trace != clustered_trace


@pytest.mark.parametrize(
    "regime",
    ("independent", "clustered"),
)
def test_readable_json_roundtrip_preserves_all_primitive_values(
    tmp_path: Path, regime: str,
) -> None:
    trace = generate_primitive_trace(
        CONFIG, 41001, regime,
    )
    path = tmp_path/f"{regime}.json"
    serialize_trace(trace, path)
    payload = json.loads(
        path.read_text(encoding="utf-8"),
    )
    assert "payload_sha256" not in payload
    assert "config_hash" not in payload
    assert payload["target_states"]
    assert payload["communication_states"]
    assert load_trace(path, CONFIG) == trace


def test_integer_and_string_identifiers_roundtrip_distinctly(
    independent_trace,
) -> None:
    mapping = {
        "target_1": 1,
        "target_2": "1",
    }
    trace = replace(
        independent_trace,
        target_states=tuple(
            replace(
                item,
                target_id=mapping.get(item.target_id, item.target_id),
            )
            for item in independent_trace.target_states
        ),
        target_innovations=tuple(
            replace(
                item,
                target_id=mapping.get(item.target_id, item.target_id),
            )
            for item in independent_trace.target_innovations
        ),
        request_descriptors=tuple(
            replace(
                item,
                target_id=mapping.get(item.target_id, item.target_id),
            )
            for item in independent_trace.request_descriptors
        ),
    )
    loaded = deserialize_trace_bytes(
        _bytes(trace_to_dict(trace)),
        CONFIG,
    )
    ids = {
        item.target_id
        for item in loaded.target_states
        if item.slot == 0
    }
    assert any(
        type(item) is int and item == 1
        for item in ids
    )
    assert any(
        type(item) is str and item == "1"
        for item in ids
    )


def test_experiment_metadata_change_does_not_reject_scientific_trace(
    tmp_path: Path, independent_trace,
) -> None:
    data = yaml.safe_load(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"),
    )
    data["profile_name"] = "different_experiment_metadata"
    data["arrivals"]["independent"]["per_tenant_rate_per_slot"] = 0.09
    path = tmp_path/"config.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    loaded = deserialize_trace_bytes(
        _bytes(trace_to_dict(independent_trace)),
        load_config(path),
    )
    assert loaded.target_states == independent_trace.target_states
    assert (
        loaded.request_descriptors
        == independent_trace.request_descriptors
    )


def test_invalid_utf8_or_missing_required_data_is_rejected(
    independent_trace,
) -> None:
    with pytest.raises(
        SerializationError,
        match="UTF-8 JSON",
    ):
        deserialize_trace_bytes(b"\xff", CONFIG)

    payload = trace_to_dict(independent_trace)
    payload.pop("target_states")
    with pytest.raises(SerializationError, match="required primitive data"):
        deserialize_trace_bytes(_bytes(payload), CONFIG)

    payload = trace_to_dict(independent_trace)
    payload["target_states"] = payload["target_states"][1:]
    with pytest.raises(SerializationError, match="coverage"):
        deserialize_trace_bytes(_bytes(payload), CONFIG)


def test_invalid_numeric_primitive_is_rejected(
    independent_trace,
) -> None:
    payload = trace_to_dict(independent_trace)
    payload["target_states"][0]["position_m"] = (
        "1.0",
        payload["target_states"][0]["position_m"][1],
    )
    with pytest.raises(
        SerializationError,
        match="finite number",
    ):
        deserialize_trace_bytes(_bytes(payload), CONFIG)

    payload = trace_to_dict(independent_trace)
    payload["target_states"][0]["shadowing_db"] = float("nan")
    with pytest.raises(
        SerializationError,
        match="finite number",
    ):
        deserialize_trace_bytes(
            _bytes(payload, allow_nan=True),
            CONFIG,
        )


def test_cli_generates_reloadable_trace_and_scientific_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path/"trace.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_traces.py",
            "--seed",
            "41001",
            "--arrival-regime",
            "clustered",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=_cli_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert set(summary) == {
        "trace_id", "seed", "arrival_regime",
        "horizon_slots", "target_count",
        "communication_user_count",
        "materialized_request_count",
        "horizon_omitted_count",
        "parent_event_count",
        "pending_descriptor_count",
        "output_path",
    }
    assert summary["seed"] == 41001
    assert summary["arrival_regime"] == "clustered"
    assert load_trace(
        output, CONFIG,
    ).request_descriptors