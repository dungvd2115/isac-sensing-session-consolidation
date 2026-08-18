"""Common-Trace PPO agent on the locked edge-free set representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isac_ssc.algorithms.common_trace_ppo import CommonTracePPO
from isac_ssc.baselines.ppo_joint_credit import RunningFeatureNormalizer
from isac_ssc.envs.observation import ObservationSnapshot
from isac_ssc.models.policy import (
    CommonTraceActorCritic, FactorizedPolicyBatch,
    PolicySelection, build_policy_batch,
)
from isac_ssc.models.set_encoder import FeatureLayout
from isac_ssc.models.value import PrefixValueOutput, ValueOutput
from isac_ssc.utils.config import CanonicalConfig, ConstrainedPPOConfig


class NormalizedCommonTraceActorCritic(CommonTraceActorCritic):
    """Apply the shared running normalizer before the common-trace actor-critic."""

    def __init__(
        self, layout: FeatureLayout, algorithm: ConstrainedPPOConfig,
        environment: CanonicalConfig, normalizer: RunningFeatureNormalizer,
    ) -> None:
        self.normalizer = normalizer
        super().__init__(layout, algorithm, environment)

    def forward(self, batch: FactorizedPolicyBatch):
        encoded = self.encoder(self.normalizer.transform(batch))
        logits = self.policy(encoded)
        values = self.value(encoded.decision_embedding)
        prefixes = self.prefix_value(
            encoded.decision_embedding,
            encoded.merge_candidate_embeddings,
            values,
        )
        return encoded, logits, values, prefixes


@dataclass(slots=True)
class CommonTracePPOAgent:
    normalizer: RunningFeatureNormalizer
    model: NormalizedCommonTraceActorCritic
    algorithm: CommonTracePPO
    action_generator: torch.Generator
    minibatch_generator: torch.Generator

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def select(
        self, observation: ObservationSnapshot, *,
        deterministic: bool = False,
    ) -> tuple[PolicySelection, ValueOutput, PrefixValueOutput]:
        batch = build_policy_batch(observation, device=self.device)
        generator = None if deterministic else self.action_generator
        return self.model.select(
            batch, deterministic=deterministic, generator=generator,
        )

    def is_finite(self) -> bool:
        tensors = list(self.model.parameters()) + [self.algorithm.dual_values]
        for state in self.algorithm.optimizer.state.values():
            tensors.extend(
                value for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        return (
            all(bool(torch.isfinite(value).all()) for value in tensors)
            and self.normalizer.is_finite()
        )


def build_common_trace_agent(
    layout: FeatureLayout, algorithm: ConstrainedPPOConfig,
    environment: CanonicalConfig, *, model_seed: int,
    action_seed: int, minibatch_seed: int,
) -> CommonTracePPOAgent:
    device = torch.device(algorithm.device)
    global_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(model_seed)
        model = NormalizedCommonTraceActorCritic(
            layout, algorithm, environment,
            RunningFeatureNormalizer(
                layout, algorithm.normalization.clip,
                algorithm.normalization.epsilon,
            ),
        ).to(device)
    finally:
        torch.random.set_rng_state(global_state)
    action_generator = torch.Generator(device=device).manual_seed(action_seed)
    minibatch_generator = torch.Generator(
        device="cpu",
    ).manual_seed(minibatch_seed)
    algorithm_instance = CommonTracePPO(model, algorithm)
    return CommonTracePPOAgent(
        model.normalizer, model, algorithm_instance,
        action_generator, minibatch_generator,
    )