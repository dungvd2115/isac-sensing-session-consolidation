"""Joint-credit PPO agent with lightweight running feature normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from isac_ssc.algorithms.constrained_ppo import ConstrainedPPO
from isac_ssc.envs.observation import ObservationSnapshot
from isac_ssc.models.policy import EdgeFreeSetActorCritic, FactorizedPolicyBatch, PolicySelection, build_policy_batch
from isac_ssc.models.set_encoder import EdgeFreeSetEncoder, FeatureLayout, SetEncoderInput
from isac_ssc.models.value import ValueOutput
from isac_ssc.utils.config import CanonicalConfig, ConstrainedPPOConfig


class JointCreditPPOValidationError(ValueError):
    """Raised for incompatible normalizer or agent state."""


@dataclass(frozen=True, slots=True)
class RunningNormalizerState:
    schema_digest: str
    request_count: int
    request_mean: torch.Tensor
    request_m2: torch.Tensor
    session_count: int
    session_mean: torch.Tensor
    session_m2: torch.Tensor
    global_count: int
    global_mean: torch.Tensor
    global_m2: torch.Tensor
    frozen: bool


class RunningFeatureNormalizer:
    """Independent Welford moments for request, session and global feature rows."""

    def __init__(self, layout: FeatureLayout, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.layout, self.clip, self.epsilon = layout, float(clip), float(epsilon)
        self._indicator = {
            "request": torch.tensor([item.unit == "indicator" for item in layout.request_specs], dtype=torch.bool),
            "session": torch.tensor([item.unit == "indicator" for item in layout.session_specs], dtype=torch.bool),
            "global": torch.tensor([item.unit == "indicator" for item in layout.global_specs], dtype=torch.bool),
        }
        for name, width in (("request", layout.request_width), ("session", layout.session_width), ("global", layout.global_width)):
            setattr(self, f"_{name}_count", 0)
            setattr(self, f"_{name}_mean", torch.zeros(width, dtype=torch.float64))
            setattr(self, f"_{name}_m2", torch.zeros(width, dtype=torch.float64))
        self._frozen = False

    @staticmethod
    def _merge(count: int, mean: torch.Tensor, m2: torch.Tensor, rows: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        if rows.shape[0] == 0:
            return count, mean, m2
        values = rows.detach().to(device="cpu", dtype=torch.float64)
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_m2 = (values - batch_mean).square().sum(dim=0)
        if count == 0:
            return batch_count, batch_mean, batch_m2
        total = count + batch_count
        delta = batch_mean - mean
        return total, mean + delta * (batch_count / total), m2 + batch_m2 + delta.square() * (count * batch_count / total)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def update(self, observations: Iterable[ObservationSnapshot]) -> None:
        if self._frozen:
            raise JointCreditPPOValidationError("frozen normalizer cannot be updated")
        observations = tuple(observations)
        if not observations:
            return
        batch = build_policy_batch(observations)
        if batch.layout != self.layout:
            raise JointCreditPPOValidationError("normalizer feature layout mismatch")
        values = batch.encoder_input
        rows = {
            "request": values.request_features[~values.request_padding_mask],
            "session": values.session_features[~values.session_padding_mask],
            "global": values.global_features.reshape(-1, self.layout.global_width),
        }
        for name, group in rows.items():
            merged = self._merge(getattr(self, f"_{name}_count"), getattr(self, f"_{name}_mean"), getattr(self, f"_{name}_m2"), group)
            setattr(self, f"_{name}_count", merged[0])
            setattr(self, f"_{name}_mean", merged[1])
            setattr(self, f"_{name}_m2", merged[2])

    def _statistics(self, name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        count = getattr(self, f"_{name}_count")
        if count < 1:
            raise JointCreditPPOValidationError("normalizer statistics are unavailable")
        mean = getattr(self, f"_{name}_mean").to(device=device, dtype=torch.float32)
        variance = getattr(self, f"_{name}_m2") / count
        scale = (
            torch.ones_like(mean) if count == 1
            else variance.to(device=device, dtype=torch.float32).sqrt().clamp_min(self.epsilon)
        )
        indicator = self._indicator[name].to(device)
        return torch.where(indicator, torch.zeros_like(mean), mean), torch.where(
            indicator, torch.ones_like(scale), scale,
        )

    def _transform(self, values: torch.Tensor, name: str, padding: torch.Tensor | None = None) -> torch.Tensor:
        count = getattr(self, f"_{name}_count")
        if count == 0:
            result = values.clone()
        else:
            mean, scale = self._statistics(name, values.device)
            normalized = ((values - mean) / scale).clamp(-self.clip, self.clip)
            result = torch.where(self._indicator[name].to(values.device), values, normalized)
        return result.masked_fill(padding.unsqueeze(-1), 0.0) if padding is not None else result

    def calibrate_and_freeze(
        self, observations: Iterable[ObservationSnapshot], encoder: EdgeFreeSetEncoder,
    ) -> None:
        if self._frozen or any(getattr(self, f"_{name}_count") for name in ("request", "session", "global")):
            raise JointCreditPPOValidationError("normalizer calibration requires an empty unfrozen state")
        if not isinstance(encoder, EdgeFreeSetEncoder) or encoder.layout != self.layout:
            raise JointCreditPPOValidationError("normalizer calibration encoder mismatch")
        observations = tuple(observations)
        if not observations:
            raise JointCreditPPOValidationError("normalizer calibration requires observations")
        batch = build_policy_batch(observations)
        values = batch.encoder_input
        raw_rows = {
            "request": values.request_features[~values.request_padding_mask],
            "session": values.session_features[~values.session_padding_mask],
            "global": values.global_features.reshape(-1, self.layout.global_width),
        }
        layers = {
            "request": encoder.request_encoder[0],
            "session": encoder.session_encoder[0],
            "global": encoder.global_encoder[0],
        }
        normalizer_backup = self.state_dict()
        parameter_backup = {
            name: (layer.weight.detach().clone(), layer.bias.detach().clone())
            for name, layer in layers.items()
        }
        try:
            before = {
                name: layer(rows.to(layer.weight.device)).detach().clone()
                for name, (layer, rows) in (
                    (name, (layers[name], raw_rows[name])) for name in layers
                )
            }
            self.update(observations)
            for name, rows in raw_rows.items():
                count = getattr(self, f"_{name}_count")
                if count == 0:
                    setattr(self, f"_{name}_count", 1)
                    setattr(
                        self, f"_{name}_mean",
                        torch.zeros(rows.shape[1], dtype=torch.float64),
                    )
                    setattr(
                        self, f"_{name}_m2",
                        torch.zeros(rows.shape[1], dtype=torch.float64),
                    )
                    continue
                mean = getattr(self, f"_{name}_mean")
                variance = getattr(self, f"_{name}_m2") / count
                standard_deviation = (
                    torch.ones_like(mean) if count == 1
                    else variance.sqrt().clamp_min(self.epsilon)
                )
                maximum_deviation = (
                    rows.detach().to(dtype=torch.float64).sub(mean).abs().amax(dim=0)
                    if rows.shape[0] else torch.zeros_like(mean)
                )
                minimum_unclipped_scale = maximum_deviation / self.clip * (1.0 + 1e-6)
                effective_scale = torch.maximum(
                    standard_deviation, minimum_unclipped_scale,
                )
                indicator = self._indicator[name]
                effective_scale = torch.where(
                    indicator, torch.ones_like(effective_scale), effective_scale,
                )
                setattr(self, f"_{name}_m2", effective_scale.square() * count)
            with torch.no_grad():
                for name, layer in layers.items():
                    mean, scale = self._statistics(name, layer.weight.device)
                    old_weight = layer.weight.detach().clone()
                    layer.weight.mul_(scale.unsqueeze(0))
                    layer.bias.add_(old_weight @ mean)
            transformed = self.transform(batch)
            normalized_rows = {
                "request": transformed.request_features[~transformed.request_padding_mask],
                "session": transformed.session_features[~transformed.session_padding_mask],
                "global": transformed.global_features.reshape(-1, self.layout.global_width),
            }
            for name, layer in layers.items():
                after = layer(normalized_rows[name].to(layer.weight.device))
                if not torch.allclose(before[name], after, rtol=5e-5, atol=5e-5):
                    raise JointCreditPPOValidationError(
                        f"{name} normalizer calibration failed function-preservation check"
                    )
            self._frozen = True
        except Exception:
            self.load_state_dict(normalizer_backup)
            with torch.no_grad():
                for name, layer in layers.items():
                    weight, bias = parameter_backup[name]
                    layer.weight.copy_(weight)
                    layer.bias.copy_(bias)
            raise

    def transform(self, batch: FactorizedPolicyBatch) -> SetEncoderInput:
        if batch.layout != self.layout:
            raise JointCreditPPOValidationError("policy batch feature layout mismatch")
        values = batch.encoder_input
        return values._with_features(
            self._transform(values.request_features, "request", values.request_padding_mask),
            self._transform(values.session_features, "session", values.session_padding_mask),
            self._transform(values.global_features, "global"),
        )

    def state(self) -> RunningNormalizerState:
        return RunningNormalizerState(
            self.layout.schema_digest,
            self._request_count, self._request_mean.clone(), self._request_m2.clone(),
            self._session_count, self._session_mean.clone(), self._session_m2.clone(),
            self._global_count, self._global_mean.clone(), self._global_m2.clone(),
            self._frozen,
        )

    def state_dict(self) -> dict[str, object]:
        state = self.state()
        return {name: getattr(state, name) for name in state.__dataclass_fields__}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("schema_digest") != self.layout.schema_digest:
            raise JointCreditPPOValidationError("normalizer feature layout mismatch")
        frozen = state.get("frozen")
        if type(frozen) is not bool:
            raise JointCreditPPOValidationError("normalizer frozen state is missing or invalid")
        for name, width in (("request", self.layout.request_width), ("session", self.layout.session_width), ("global", self.layout.global_width)):
            count, mean, m2 = state[f"{name}_count"], state[f"{name}_mean"], state[f"{name}_m2"]
            if not isinstance(count, int) or count < 0 or mean.shape != (width,) or m2.shape != (width,):
                raise JointCreditPPOValidationError(f"invalid {name} normalizer state")
            setattr(self, f"_{name}_count", count)
            setattr(self, f"_{name}_mean", mean.detach().cpu().to(torch.float64).clone())
            setattr(self, f"_{name}_m2", m2.detach().cpu().to(torch.float64).clamp_min(0.0).clone())
        self._frozen = frozen

    def is_finite(self) -> bool:
        return all(bool(torch.isfinite(getattr(self, f"_{name}_{field}")).all()) for name in ("request", "session", "global") for field in ("mean", "m2"))


class NormalizedEdgeFreeSetActorCritic(EdgeFreeSetActorCritic):
    """Apply frozen running moments before the shared edge-free actor-critic."""

    def __init__(self, layout: FeatureLayout, algorithm: ConstrainedPPOConfig, environment: CanonicalConfig, normalizer: RunningFeatureNormalizer) -> None:
        self.normalizer = normalizer
        super().__init__(layout, algorithm, environment)

    def forward(self, batch: FactorizedPolicyBatch):
        encoded = self.encoder(self.normalizer.transform(batch))
        return encoded, self.policy(encoded), self.value(encoded.decision_embedding)


@dataclass(slots=True)
class JointCreditPPOAgent:
    normalizer: RunningFeatureNormalizer
    model: NormalizedEdgeFreeSetActorCritic
    algorithm: ConstrainedPPO
    action_generator: torch.Generator
    minibatch_generator: torch.Generator

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def select(self, observation: ObservationSnapshot, *, deterministic: bool = False) -> tuple[PolicySelection, ValueOutput]:
        batch = build_policy_batch(observation, device=self.device)
        generator = None if deterministic else self.action_generator
        return self.model.select(batch, deterministic=deterministic, generator=generator)

    def is_finite(self) -> bool:
        tensors = list(self.model.parameters()) + [self.algorithm.dual_values]
        for state in self.algorithm.optimizer.state.values():
            tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
        return all(bool(torch.isfinite(value).all()) for value in tensors) and self.normalizer.is_finite()


def build_joint_credit_agent(
    layout: FeatureLayout, algorithm: ConstrainedPPOConfig, environment: CanonicalConfig,
    *, model_seed: int, action_seed: int, minibatch_seed: int,
) -> JointCreditPPOAgent:
    device = torch.device(algorithm.device)
    global_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(model_seed)
        model = NormalizedEdgeFreeSetActorCritic(
            layout, algorithm, environment,
            RunningFeatureNormalizer(layout, algorithm.normalization.clip, algorithm.normalization.epsilon),
        ).to(device)
    finally:
        torch.random.set_rng_state(global_state)
    action_generator = torch.Generator(device=device).manual_seed(action_seed)
    minibatch_generator = torch.Generator(device="cpu").manual_seed(minibatch_seed)
    algorithm_instance = ConstrainedPPO(model, algorithm)
    return JointCreditPPOAgent(model.normalizer, model, algorithm_instance, action_generator, minibatch_generator)