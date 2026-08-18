"""Stable keyed NumPy substreams for primitive trace generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from isac_ssc.utils.config import CanonicalConfig

_DERIVATION = "sha256_typed_token_v1"
_BIT_GENERATOR = "PCG64DXSM"


class SeedError(ValueError):
    """Raised when a root seed or substream token is invalid."""


def _typed_token(value: Any) -> list[Any]:
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return ["int", str(int(value))]
    if isinstance(value, Real) and not isinstance(value, Integral):
        number = float(value)
        if not isfinite(number):
            raise SeedError("floating-point seed tokens must be finite")
        return ["float", number.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, tuple):
        return ["tuple", [_typed_token(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_typed_token(item) for item in value]]
    if isinstance(value, Mapping):
        encoded = [
            (_typed_token(key), _typed_token(item))
            for key, item in value.items()
        ]
        encoded.sort(
            key=lambda pair: json.dumps(
                pair[0], ensure_ascii=False, separators=(",", ":"),
            ),
        )
        return ["mapping", [[key, item] for key, item in encoded]]
    raise SeedError(f"unsupported seed token type: {type(value).__name__}")


def _root_seed(value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SeedError("root seed must be an integer")
    seed = int(value)
    if not minimum <= seed <= maximum:
        raise SeedError(f"root seed must lie in [{minimum}, {maximum}]")
    return seed


@dataclass(frozen=True, slots=True)
class SeedContract:
    namespace: str
    root_seed_minimum: int = 0
    root_seed_maximum: int = 2**64-1

    @classmethod
    def from_config(cls, config: CanonicalConfig) -> SeedContract:
        specification = config.trace_generation["seeding"]
        if specification["derivation"] != _DERIVATION:
            raise SeedError(
                f"unsupported seed derivation: {specification['derivation']!r}",
            )
        if specification["numpy_bit_generator"] != _BIT_GENERATOR:
            raise SeedError(
                "unsupported NumPy bit generator: "
                f"{specification['numpy_bit_generator']!r}",
            )
        if specification["substream_contract"] != "keyed_order_independent":
            raise SeedError("unsupported substream contract")
        minimum = specification["root_seed_minimum"]
        maximum = specification["root_seed_maximum"]
        if minimum != 0 or maximum != 2**64-1:
            raise SeedError(
                "root seed domain must be the complete unsigned 64-bit interval",
            )
        namespace = specification["namespace"]
        if not isinstance(namespace, str) or not namespace:
            raise SeedError("seed namespace must be a non-empty string")
        return cls(namespace, minimum, maximum)

    def canonical_material(self, root_seed: int, *tokens: Any) -> bytes:
        seed = _root_seed(
            root_seed, self.root_seed_minimum, self.root_seed_maximum,
        )
        payload = [
            ["namespace", self.namespace],
            ["derivation", _DERIVATION],
            ["root_seed", ["int", str(seed)]],
            ["tokens", [_typed_token(token) for token in tokens]],
        ]
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")

    def derive_seed_words(
        self, root_seed: int, *tokens: Any,
    ) -> tuple[int, ...]:
        digest = hashlib.sha256(
            self.canonical_material(root_seed, *tokens),
        ).digest()
        return tuple(
            int.from_bytes(digest[offset:offset+4], "big")
            for offset in range(0, 32, 4)
        )

    def derive_uint64(self, root_seed: int, *tokens: Any) -> int:
        digest = hashlib.sha256(
            self.canonical_material(root_seed, *tokens),
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def rng(self, root_seed: int, *tokens: Any) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            self.derive_seed_words(root_seed, *tokens),
        )
        return np.random.Generator(np.random.PCG64DXSM(sequence))