import inspect

import pytest
import torch

from isac_ssc.envs.observation import FeatureSpec
from isac_ssc.models.set_encoder import (
    EdgeFreeSetEncoder, FeatureLayout,
    SetEncoderInput, SetEncoderValidationError,
)
from isac_ssc.utils.config import AlgorithmModelConfig


def _spec(
    prefix: str, count: int,
) -> tuple[FeatureSpec, ...]:
    return tuple(
        FeatureSpec(
            f"{prefix}_{index}", "ratio", "none", prefix,
        )
        for index in range(count)
    )


def _layout() -> FeatureLayout:
    return FeatureLayout(
        _spec("request", 3),
        _spec("session", 2),
        _spec("global", 2),
        ("a", "b"),
        "0" * 64,
    )


def _config() -> AlgorithmModelConfig:
    return AlgorithmModelConfig(8, 4)


def _inputs() -> SetEncoderInput:
    requests = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [7.0, 8.0, 9.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ])
    sessions = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [0.0, 0.0]],
    ])
    return SetEncoderInput(
        requests,
        torch.tensor([
            [False, False, True],
            [False, True, True],
        ]),
        sessions,
        torch.tensor([
            [False, False],
            [False, True],
        ]),
        torch.tensor([
            [0.5, 1.0],
            [1.5, 2.0],
        ]),
        torch.tensor([1, 0]),
    )


def _permute(
    inputs: SetEncoderInput,
    request_order,
    session_order,
    focal,
) -> SetEncoderInput:
    return SetEncoderInput(
        inputs.request_features[:, request_order],
        inputs.request_padding_mask[:, request_order],
        inputs.session_features[:, session_order],
        inputs.session_padding_mask[:, session_order],
        inputs.global_features,
        torch.tensor(focal, dtype=torch.int64),
    )


def test_encoder_shapes_finiteness_zero_padding_and_gradients() -> None:
    torch.manual_seed(7)
    encoder = EdgeFreeSetEncoder(_layout(), _config())
    inputs = _inputs()
    output = encoder(inputs)
    assert output.request_embeddings.shape == (2, 3, 8)
    assert output.session_embeddings.shape == (2, 2, 8)
    assert output.decision_embedding.shape == (2, 8)
    assert output.merge_candidate_embeddings.shape == (
        2, 2, 8,
    )
    assert (
        torch.count_nonzero(
            output.request_embeddings[
                inputs.request_padding_mask
            ]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            output.session_embeddings[
                inputs.session_padding_mask
            ]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            output.merge_candidate_embeddings[
                inputs.session_padding_mask
            ]
        )
        == 0
    )
    assert all(
        torch.isfinite(value).all()
        for value in (
            output.request_embeddings,
            output.session_embeddings,
            output.focal_embedding,
            output.request_pool,
            output.session_pool,
            output.global_embedding,
            output.decision_embedding,
            output.merge_candidate_embeddings,
        )
    )
    output.decision_embedding.sum().backward()
    assert (
        encoder.request_encoder[0].weight.grad
        is not None
    )
    assert torch.count_nonzero(
        encoder.request_encoder[0].weight.grad
    ) > 0


def test_request_and_session_permutations_preserve_or_permute_outputs_exactly() -> None:
    torch.manual_seed(11)
    encoder = EdgeFreeSetEncoder(_layout(), _config())
    inputs = _inputs()
    original = encoder(inputs)
    permuted = encoder(_permute(
        inputs,
        [1, 0, 2],
        [1, 0],
        [0, 1],
    ))
    assert torch.allclose(
        permuted.request_embeddings,
        original.request_embeddings[:, [1, 0, 2]],
    )
    assert torch.allclose(
        permuted.session_embeddings,
        original.session_embeddings[:, [1, 0]],
    )
    assert torch.allclose(
        permuted.request_pool, original.request_pool,
    )
    assert torch.allclose(
        permuted.session_pool, original.session_pool,
    )
    assert torch.allclose(
        permuted.decision_embedding,
        original.decision_embedding,
    )
    assert torch.allclose(
        permuted.merge_candidate_embeddings,
        original.merge_candidate_embeddings[:, [1, 0]],
    )


def test_padding_rows_do_not_change_real_outputs_or_inputs() -> None:
    torch.manual_seed(13)
    encoder = EdgeFreeSetEncoder(_layout(), _config())
    inputs = _inputs()
    request_before = inputs.request_features.clone()
    session_before = inputs.session_features.clone()
    original = encoder(inputs)
    padded = SetEncoderInput(
        torch.cat((
            inputs.request_features,
            torch.zeros(2, 2, 3),
        ), dim=1),
        torch.cat((
            inputs.request_padding_mask,
            torch.ones(2, 2, dtype=torch.bool),
        ), dim=1),
        torch.cat((
            inputs.session_features,
            torch.zeros(2, 2, 2),
        ), dim=1),
        torch.cat((
            inputs.session_padding_mask,
            torch.ones(2, 2, dtype=torch.bool),
        ), dim=1),
        inputs.global_features,
        inputs.focal_request_index,
    )
    expanded = encoder(padded)
    assert torch.allclose(
        expanded.request_embeddings[:, :3],
        original.request_embeddings,
    )
    assert torch.allclose(
        expanded.session_embeddings[:, :2],
        original.session_embeddings,
    )
    assert torch.allclose(
        expanded.decision_embedding,
        original.decision_embedding,
    )
    assert torch.equal(
        inputs.request_features, request_before,
    )
    assert torch.equal(
        inputs.session_features, session_before,
    )


def test_empty_session_set_is_finite_and_has_exact_zero_pool() -> None:
    inputs = _inputs()
    empty = SetEncoderInput(
        inputs.request_features,
        inputs.request_padding_mask,
        torch.zeros(2, 0, 2),
        torch.ones(2, 0, dtype=torch.bool),
        inputs.global_features,
        inputs.focal_request_index,
    )
    output = EdgeFreeSetEncoder(
        _layout(), _config(),
    )(empty)
    assert output.session_embeddings.shape == (2, 0, 8)
    assert output.merge_candidate_embeddings.shape == (
        2, 0, 8,
    )
    assert torch.equal(
        output.session_pool,
        torch.zeros_like(output.session_pool),
    )
    assert torch.isfinite(
        output.decision_embedding
    ).all()


def test_single_and_batched_forward_are_consistent() -> None:
    torch.manual_seed(17)
    encoder = EdgeFreeSetEncoder(_layout(), _config())
    inputs = _inputs()
    batched = encoder(inputs)
    single = SetEncoderInput(
        inputs.request_features[:1],
        inputs.request_padding_mask[:1],
        inputs.session_features[:1],
        inputs.session_padding_mask[:1],
        inputs.global_features[:1],
        inputs.focal_request_index[:1],
    )
    output = encoder(single)
    assert torch.allclose(
        output.decision_embedding[0],
        batched.decision_embedding[0],
    )
    assert torch.allclose(
        output.merge_candidate_embeddings[0],
        batched.merge_candidate_embeddings[0],
    )


def test_invalid_dtypes_padding_and_focal_rows_are_rejected() -> None:
    inputs = _inputs()
    with pytest.raises(SetEncoderValidationError):
        SetEncoderInput(
            inputs.request_features.double(),
            inputs.request_padding_mask,
            inputs.session_features,
            inputs.session_padding_mask,
            inputs.global_features,
            inputs.focal_request_index,
        )
    bad = inputs.request_features.clone()
    bad[0, 2, 0] = 1.0
    with pytest.raises(
        SetEncoderValidationError,
        match="padded request",
    ):
        SetEncoderInput(
            bad,
            inputs.request_padding_mask,
            inputs.session_features,
            inputs.session_padding_mask,
            inputs.global_features,
            inputs.focal_request_index,
        )
    with pytest.raises(
        SetEncoderValidationError,
        match="real request",
    ):
        SetEncoderInput(
            inputs.request_features,
            inputs.request_padding_mask,
            inputs.session_features,
            inputs.session_padding_mask,
            inputs.global_features,
            torch.tensor([2, 0]),
        )


def test_encoder_signature_has_no_edge_input_or_graph_dependency() -> None:
    assert tuple(inspect.signature(EdgeFreeSetEncoder.forward).parameters) == ("self", "inputs")
    source = inspect.getsource(inspect.getmodule(EdgeFreeSetEncoder))
    forward = inspect.getsource(EdgeFreeSetEncoder.forward)
    assert "graph_encoder" not in source and "edge_features" not in source
    assert "isfinite" not in forward and "FeatureLayout" not in forward