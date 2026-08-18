"""Permutation-consistent edge-free set encoder for constrained PPO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn

from isac_ssc.envs.observation import FeatureSpec, ObservationView
from isac_ssc.utils.config import AlgorithmModelConfig


class SetEncoderValidationError(ValueError):
    """Raised when edge-free set inputs violate the locked tensor contract."""


def _spec_payload(spec: FeatureSpec) -> dict[str, str]:
    return {
        "name": spec.name, "unit": spec.unit, "normalization": spec.normalization,
        "canonical_source": spec.canonical_source,
    }


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    request_specs: tuple[FeatureSpec, ...]
    session_specs: tuple[FeatureSpec, ...]
    global_specs: tuple[FeatureSpec, ...]
    profile_ids: tuple[str, ...]
    schema_digest: str

    def __post_init__(self) -> None:
        for name in ("request_specs", "session_specs", "global_specs"):
            values = getattr(self, name)
            if not values or any(not isinstance(item, FeatureSpec) for item in values):
                raise SetEncoderValidationError(f"{name} must contain FeatureSpec values")
            feature_names = tuple(item.name for item in values)
            if len(set(feature_names)) != len(feature_names):
                raise SetEncoderValidationError(f"{name} names must be unique")
        if (
            not self.profile_ids or len(set(self.profile_ids)) != len(self.profile_ids)
            or tuple(sorted(self.profile_ids)) != self.profile_ids
            or any(not isinstance(item, str) or not item for item in self.profile_ids)
        ):
            raise SetEncoderValidationError(
                "profile_ids must be non-empty, unique, and sorted"
            )
        if not isinstance(self.schema_digest, str) or len(self.schema_digest) != 64:
            raise SetEncoderValidationError("schema_digest must be a SHA-256 hex digest")

    @classmethod
    def from_view(cls, view: ObservationView) -> FeatureLayout:
        if not isinstance(view, ObservationView):
            raise SetEncoderValidationError("FeatureLayout requires an ObservationView")
        profile_ids = view.action_masks.feasibility.profile_ids
        payload = {
            "request": [_spec_payload(item) for item in view.request_table.specs],
            "session": [_spec_payload(item) for item in view.session_table.specs],
            "global": [_spec_payload(item) for item in view.global_specs],
            "profile_ids": list(profile_ids),
        }
        digest = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        return cls(
            view.request_table.specs, view.session_table.specs, view.global_specs,
            profile_ids, digest,
        )

    @property
    def request_width(self) -> int:
        return len(self.request_specs)

    @property
    def session_width(self) -> int:
        return len(self.session_specs)

    @property
    def global_width(self) -> int:
        return len(self.global_specs)


@dataclass(frozen=True, slots=True)
class SetEncoderInput:
    request_features: torch.Tensor
    request_padding_mask: torch.Tensor
    session_features: torch.Tensor
    session_padding_mask: torch.Tensor
    global_features: torch.Tensor
    focal_request_index: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.request_features, self.request_padding_mask, self.session_features,
            self.session_padding_mask, self.global_features, self.focal_request_index,
        )
        if any(not isinstance(item, torch.Tensor) for item in tensors):
            raise SetEncoderValidationError("set encoder inputs must be torch tensors")
        if self.request_features.ndim != 3 or self.session_features.ndim != 3:
            raise SetEncoderValidationError(
                "request and session features must be rank-3"
            )
        if self.global_features.ndim != 2:
            raise SetEncoderValidationError("global features must be rank-2")
        if (
            self.request_padding_mask.ndim != 2
            or self.session_padding_mask.ndim != 2
        ):
            raise SetEncoderValidationError("padding masks must be rank-2")
        if self.focal_request_index.ndim != 1:
            raise SetEncoderValidationError("focal_request_index must be rank-1")
        batch = self.request_features.shape[0]
        if batch < 1 or self.request_features.shape[1] < 1:
            raise SetEncoderValidationError(
                "every batch requires at least one request row"
            )
        if (
            self.session_features.shape[0] != batch
            or self.global_features.shape[0] != batch
            or self.request_padding_mask.shape != self.request_features.shape[:2]
            or self.session_padding_mask.shape != self.session_features.shape[:2]
            or self.focal_request_index.shape[0] != batch
        ):
            raise SetEncoderValidationError("set encoder batch dimensions disagree")
        if (
            self.request_features.dtype is not torch.float32
            or self.session_features.dtype is not torch.float32
        ):
            raise SetEncoderValidationError("node features must use torch.float32")
        if self.global_features.dtype is not torch.float32:
            raise SetEncoderValidationError("global features must use torch.float32")
        if (
            self.request_padding_mask.dtype is not torch.bool
            or self.session_padding_mask.dtype is not torch.bool
        ):
            raise SetEncoderValidationError("padding masks must use torch.bool")
        if self.focal_request_index.dtype is not torch.int64:
            raise SetEncoderValidationError(
                "focal_request_index must use torch.int64"
            )
        devices = {item.device for item in tensors}
        if len(devices) != 1:
            raise SetEncoderValidationError(
                "all set encoder inputs must use one device"
            )
        for name, tensor in (
            ("request_features", self.request_features),
            ("session_features", self.session_features),
            ("global_features", self.global_features),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise SetEncoderValidationError(f"{name} must be finite")
        if bool(self.request_padding_mask.all(dim=1).any()):
            raise SetEncoderValidationError(
                "every sample requires a non-padding request"
            )
        rows = self.request_features.shape[1]
        if bool(
            (
                (self.focal_request_index < 0)
                | (self.focal_request_index >= rows)
            ).any()
        ):
            raise SetEncoderValidationError("focal_request_index is out of range")
        focal_is_padding = self.request_padding_mask.gather(
            1, self.focal_request_index[:, None],
        ).squeeze(1)
        if bool(focal_is_padding.any()):
            raise SetEncoderValidationError(
                "focal_request_index must select a real request row"
            )
        if bool(torch.count_nonzero(self.request_features.masked_select(
            self.request_padding_mask.unsqueeze(-1),
        ))):
            raise SetEncoderValidationError(
                "padded request feature rows must be exact zero"
            )
        if bool(torch.count_nonzero(self.session_features.masked_select(
            self.session_padding_mask.unsqueeze(-1),
        ))):
            raise SetEncoderValidationError(
                "padded session feature rows must be exact zero"
            )

    def _with_features(
        self, request_features: torch.Tensor, session_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> SetEncoderInput:
        """Reuse validated masks and indices after deterministic feature normalization."""
        result = object.__new__(type(self))
        for name, value in (
            ("request_features", request_features), ("request_padding_mask", self.request_padding_mask),
            ("session_features", session_features), ("session_padding_mask", self.session_padding_mask),
            ("global_features", global_features), ("focal_request_index", self.focal_request_index),
        ):
            object.__setattr__(result, name, value)
        return result


@dataclass(frozen=True, slots=True)
class SetEncoderOutput:
    request_embeddings: torch.Tensor
    session_embeddings: torch.Tensor
    focal_embedding: torch.Tensor
    request_pool: torch.Tensor
    session_pool: torch.Tensor
    global_embedding: torch.Tensor
    decision_embedding: torch.Tensor
    merge_candidate_embeddings: torch.Tensor


def _orthogonal_initialize(module: nn.Module, gain: float) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def _masked_mean(
    values: torch.Tensor, padding_mask: torch.Tensor,
) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1).to(values.dtype)
    total = (values * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp_min(1.0)
    return total / count


class EdgeFreeSetEncoder(nn.Module):
    """Encode request/session sets without compatibility edges or cross-set messages."""

    def __init__(
        self, layout: FeatureLayout, config: AlgorithmModelConfig,
    ) -> None:
        super().__init__()
        if not isinstance(layout, FeatureLayout) or not isinstance(config, AlgorithmModelConfig):
            raise SetEncoderValidationError("layout and model config are required")
        self.layout, self.hidden_dim = layout, config.hidden_dim
        hidden, gain = config.hidden_dim, nn.init.calculate_gain("tanh")
        self.request_encoder = nn.Sequential(
            nn.Linear(layout.request_width, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.session_encoder = nn.Sequential(
            nn.Linear(layout.session_width, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(layout.global_width, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.decision_encoder = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.merge_candidate_encoder = nn.Sequential(
            nn.Linear(5 * hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        _orthogonal_initialize(self, gain)

    def forward(self, inputs: SetEncoderInput) -> SetEncoderOutput:
        request = self.request_encoder(inputs.request_features)
        session = self.session_encoder(inputs.session_features)
        request = request.masked_fill(
            inputs.request_padding_mask.unsqueeze(-1), 0.0,
        )
        session = session.masked_fill(
            inputs.session_padding_mask.unsqueeze(-1), 0.0,
        )
        request_pool = _masked_mean(request, inputs.request_padding_mask)
        session_pool = _masked_mean(session, inputs.session_padding_mask)
        global_embedding = self.global_encoder(inputs.global_features)
        batch_index = torch.arange(request.shape[0], device=request.device)
        focal = request[batch_index, inputs.focal_request_index]
        decision = self.decision_encoder(torch.cat(
            (focal, request_pool, session_pool, global_embedding), dim=-1,
        ))
        sessions = session.shape[1]
        expanded = tuple(
            item.unsqueeze(1).expand(-1, sessions, -1)
            for item in (
                focal, request_pool, session_pool, global_embedding,
            )
        )
        candidates = self.merge_candidate_encoder(torch.cat(
            (
                expanded[0], session, expanded[1],
                expanded[2], expanded[3],
            ),
            dim=-1,
        ))
        candidates = candidates.masked_fill(
            inputs.session_padding_mask.unsqueeze(-1), 0.0,
        )
        return SetEncoderOutput(
            request, session, focal, request_pool, session_pool,
            global_embedding, decision, candidates,
        )