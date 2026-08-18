from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from isac_ssc.utils.config import load_config
from isac_ssc.utils.seeding import SeedContract, SeedError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config()
CONTRACT = SeedContract.from_config(CONFIG)


def test_seed_contract_uses_configured_namespace_and_uint64_domain() -> None:
    assert CONTRACT.namespace == "isac-ssc-primitive-trace-v1"
    assert CONTRACT.root_seed_minimum == 0
    assert CONTRACT.root_seed_maximum == 2**64-1


def test_seed_derivation_has_a_frozen_cross_platform_digest() -> None:
    words = CONTRACT.derive_seed_words(41001, "target", 1, ("slot", 3))
    assert words == (
        4237553397, 547613334, 2799284357, 2234318720,
        1109805167, 1505755749, 1565589203, 3769266562,
    )
    assert CONTRACT.derive_uint64(41001, "target", 1, ("slot", 3)) == 18200153255716317846


def test_typed_tokens_and_container_shapes_do_not_collide() -> None:
    keys = (
        CONTRACT.derive_seed_words(41001, 1), CONTRACT.derive_seed_words(41001, "1"),
        CONTRACT.derive_seed_words(41001, True), CONTRACT.derive_seed_words(41001, 1.0),
        CONTRACT.derive_seed_words(41001, [1, 2]), CONTRACT.derive_seed_words(41001, (1, 2)),
    )
    assert len(set(keys)) == len(keys)


def test_mapping_encoding_is_order_independent_but_token_order_is_not() -> None:
    left = CONTRACT.derive_seed_words(41001, {"target": 3, "slot": 2})
    right = CONTRACT.derive_seed_words(41001, {"slot": 2, "target": 3})
    assert left == right
    assert CONTRACT.derive_seed_words(41001, "target", 3) != CONTRACT.derive_seed_words(41001, 3, "target")


def test_keyed_substreams_are_independent_of_iteration_order() -> None:
    identifiers = (1, "1", "target_2", 7)
    forward = {identifier: CONTRACT.rng(41001, "target", identifier).normal(size=6) for identifier in identifiers}
    reverse = {
        identifier: CONTRACT.rng(41001, "target", identifier).normal(size=6)
        for identifier in reversed(identifiers)
    }
    for identifier in identifiers:
        assert np.array_equal(forward[identifier], reverse[identifier])


def test_repeated_rng_construction_replays_exactly() -> None:
    first = CONTRACT.rng(41001, "communication", "user_1", "slot", 12).lognormal(size=16)
    second = CONTRACT.rng(41001, "communication", "user_1", "slot", 12).lognormal(size=16)
    assert np.array_equal(first, second)


def test_different_root_seed_changes_the_keyed_substream() -> None:
    first = CONTRACT.rng(41001, "target", 1).normal(size=16)
    second = CONTRACT.rng(41002, "target", 1).normal(size=16)
    assert not np.array_equal(first, second)


def test_seed_derivation_is_identical_in_a_fresh_python_process() -> None:
    code = """
import json
from isac_ssc.utils.config import load_config
from isac_ssc.utils.seeding import SeedContract
contract = SeedContract.from_config(load_config())
print(json.dumps(contract.derive_seed_words(41001, 'target', 1, ('slot', 3))))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert tuple(json.loads(completed.stdout)) == CONTRACT.derive_seed_words(
        41001, "target", 1, ("slot", 3),
    )


@pytest.mark.parametrize("seed", (-1, 2**64, True, 1.5, "41001"))
def test_invalid_root_seeds_are_rejected(seed) -> None:
    with pytest.raises(SeedError, match="root seed"):
        CONTRACT.rng(seed, "target")


@pytest.mark.parametrize("token", (float("nan"), float("inf"), object()))
def test_invalid_substream_tokens_are_rejected(token) -> None:
    with pytest.raises(SeedError):
        CONTRACT.derive_seed_words(41001, token)