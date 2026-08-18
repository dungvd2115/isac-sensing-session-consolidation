from dataclasses import replace
from math import log
from unittest.mock import patch

import pytest
import torch

from isac_ssc.core.entities import (
    DiskAOI, RequestState, SensingRequest,
    SensingSession, Task,
)
from isac_ssc.envs.action_masks import (
    build_action_masks, build_current_feasibility,
)
from isac_ssc.envs.action_space import (
    ActionType, EnvironmentAction,
)
from isac_ssc.envs.dynamics import (
    CommunicationSlotPrimitive, TargetSlotPrimitive,
)
from isac_ssc.envs.observation import (
    CommunicationAccountingState,
    TenantAccountingState,
    build_observation,
)
from isac_ssc.models.policy import (
    EdgeFreeSetActorCritic, CommonTraceActorCritic,
    FactorizedActionIndices,
    FactorizedPolicyHead,
    FactorizedPolicyLogits,
    PolicyValidationError, _action_type_logits_with_merge_options,
    build_policy_batch,
)
from isac_ssc.models.set_encoder import SetEncoderOutput
from isac_ssc.models.value import HierarchicalPrefixValueHead, ValueOutput
from isac_ssc.utils.config import (
    load_algorithm_config, load_config,
)

CONFIG = load_config()
ALGORITHM = load_algorithm_config()
DURATIONS = CONFIG.service_duration_slots
AOI = DiskAOI((80.0, 0.0), 30.0)
PRIOR = tuple(tuple(
    float(
        CONFIG.sensing[
            "tracking"
        ]["initial_covariance_diag"][row]
    )
    if row == column
    else 0.0
    for column in range(4)
) for row in range(4))


def _request(
    request_id,
    *,
    state=RequestState.WAITING,
    tenant="tenant_1",
    target=7,
):
    request = SensingRequest(
        request_id,
        tenant,
        1 if state is RequestState.WAITING else 0,
        8,
        AOI,
        target,
        Task.DETECTION,
        0.1,
        2,
        1.0,
        True,
    )
    return (
        request
        if state is RequestState.WAITING
        else request.transition(state, slot=0)
    )


def _observation(
    *,
    focal_id="focal",
    session_id=1,
):
    creator = _request(
        f"creator_{session_id}",
        state=RequestState.ACTIVE,
    )
    session = SensingSession.create(
        session_id,
        creator,
        CONFIG.resource_profiles["balanced"],
        0,
        DURATIONS,
        None,
    )
    focal = _request(focal_id)
    requests = (creator, focal)
    sessions = (session,)
    targets = (
        TargetSlotPrimitive(
            1,
            7,
            (80.0, 0.0),
            (0.0, 0.0),
            0.0,
            1.0,
            0.0,
            0.0,
        ),
    )
    communication = tuple(
        CommunicationSlotPrimitive(
            1,
            f"user_{index}",
            (40.0 + index, 5.0),
            (0.0, 0.0),
            True,
            5.0e6,
            0.0,
            1.0,
            0.0,
        )
        for index in range(
            1,
            CONFIG.population[
                "communication_users"
            ] + 1,
        )
    )
    feasibility = build_current_feasibility(
        1,
        requests,
        sessions,
        targets,
        99,
        CONFIG,
    )
    masks = build_action_masks(
        focal,
        requests,
        sessions,
        feasibility,
        CONFIG,
    )
    tenant = tuple(
        TenantAccountingState(
            item.tenant_id,
            index,
            index // 2,
            index // 3,
            index // 2
            - item.sla_violation_budget * index,
        )
        for index, item in enumerate(
            CONFIG.tenants, start=1,
        )
    )
    communication_accounting = tuple(
        CommunicationAccountingState(
            item.user_id,
            index,
            0.1 * index,
            0.1 * index
            - CONFIG.communication[
                "normalized_shortfall_budget"
            ] * index,
        )
        for index, item in enumerate(
            communication, start=1,
        )
    )
    return build_observation(
        1,
        focal,
        requests,
        sessions,
        targets,
        communication,
        feasibility,
        masks,
        tenant,
        communication_accounting,
        0.0,
        0.0,
        CONFIG,
    )


def _zero_logits(
    batch,
) -> FactorizedPolicyLogits:
    batch_size = batch.action_type_mask.shape[0]
    sessions = batch.merge_session_mask.shape[1]
    profiles = len(batch.profile_ids)
    return FactorizedPolicyLogits(
        torch.zeros(batch_size, 4),
        torch.zeros(batch_size, sessions),
        torch.zeros(
            batch_size, sessions, profiles,
        ),
        torch.zeros(batch_size, profiles),
    )


def test_policy_batch_uses_public_set_view_masks_and_never_encodes_ids() -> None:
    observation = _observation()
    batch = build_policy_batch(observation)
    view = observation.set_view
    assert (
        batch.encoder_input.request_features.shape
        == (1, 1, 56)
    )
    assert (
        batch.encoder_input.session_features.shape
        == (1, 1, 39)
    )
    assert (
        batch.encoder_input.global_features.shape
        == (1, 27)
    )
    assert batch.profile_ids == (
        "balanced",
        "economical",
        "precision",
        "rapid",
    )
    assert batch.session_ids == ((1,),)
    assert (
        batch.action_type_mask.tolist()[0]
        == [
            item[1]
            for item in view.action_masks.action_type_mask
        ]
    )
    assert (
        batch.merge_session_mask.tolist()[0]
        == [
            item[1]
            for item in view.action_masks.merge_session_mask
        ]
    )
    assert (
        batch.create_profile_mask.tolist()[0]
        == [
            item[1]
            for item in view.action_masks.create_profile_mask
        ]
    )
    assert all(
        tensor.dtype in {
            torch.float32,
            torch.bool,
            torch.int64,
        }
        for tensor in (
            batch.encoder_input.request_features,
            batch.encoder_input.request_padding_mask,
            batch.encoder_input.focal_request_index,
        )
    )


def test_opaque_identifier_renaming_does_not_change_neural_tensors() -> None:
    first = build_policy_batch(_observation(
        focal_id="focal_a",
        session_id=1,
    ))
    second = build_policy_batch(_observation(
        focal_id="focal_z",
        session_id="session_z",
    ))
    for left, right in (
        (
            first.encoder_input.request_features,
            second.encoder_input.request_features,
        ),
        (
            first.encoder_input.session_features,
            second.encoder_input.session_features,
        ),
        (
            first.encoder_input.global_features,
            second.encoder_input.global_features,
        ),
        (
            first.action_type_mask,
            second.action_type_mask,
        ),
        (
            first.merge_session_mask,
            second.merge_session_mask,
        ),
        (
            first.merge_profile_mask,
            second.merge_profile_mask,
        ),
        (
            first.create_profile_mask,
            second.create_profile_mask,
        ),
    ):
        assert torch.equal(left, right)


def test_actor_critic_shapes_finiteness_shared_encoder_and_gradients() -> None:
    torch.manual_seed(23)
    batch = build_policy_batch(_observation())
    model = EdgeFreeSetActorCritic(
        batch.layout, ALGORITHM, CONFIG,
    )
    with patch.object(
        model.encoder,
        "forward",
        wraps=model.encoder.forward,
    ) as forward:
        encoded, logits, values = model(batch)
    assert forward.call_count == 1
    assert logits.action_type_logits.shape == (
        1, 4,
    )
    assert logits.merge_session_logits.shape == (
        1, 1,
    )
    assert logits.merge_profile_logits.shape == (
        1, 1, 4,
    )
    assert logits.create_profile_logits.shape == (
        1, 4,
    )
    assert values.reward_value.shape == (1,)
    assert values.sensing_sla_values.shape == (
        1, len(CONFIG.tenants),
    )
    assert values.communication_qos_values.shape == (
        1,
        CONFIG.population[
            "communication_users"
        ],
    )
    loss = (
        logits.action_type_logits.sum()
        + logits.merge_session_logits.sum()
        + logits.merge_profile_logits.sum()
        + logits.create_profile_logits.sum()
        + values.reward_value.sum()
        + values.sensing_sla_values.sum()
        + values.communication_qos_values.sum()
    )
    loss.backward()
    assert (
        model.encoder.request_encoder[
            0
        ].weight.grad
        is not None
    )
    assert (
        model.policy.action_type_head.weight.grad
        is not None
    )
    assert (
        model.value.reward_head.weight.grad
        is not None
    )
    assert torch.isfinite(
        encoded.decision_embedding
    ).all()


def test_candidate_conditioned_decoder_scores_each_session_separately() -> None:
    head = FactorizedPolicyHead(2, 2, 1)
    with torch.no_grad():
        head.merge_session_head.weight.copy_(
            torch.tensor([[1.0, 0.0]])
        )
        head.merge_session_head.bias.zero_()
    zero = torch.zeros(1, 2)
    encoded = SetEncoderOutput(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2, 2),
        zero,
        zero,
        zero,
        zero,
        zero,
        torch.tensor([
            [
                [1.0, 0.0],
                [3.0, 0.0],
            ],
        ]),
    )
    logits = head(encoded)
    assert logits.merge_session_logits.tolist() == [
        [1.0, 3.0],
    ]


def test_zero_logits_have_exact_factorized_log_probability_and_entropy() -> None:
    batch = build_policy_batch(_observation())
    head = FactorizedPolicyHead(
        8, len(batch.profile_ids), 4,
    )
    logits = _zero_logits(batch)
    action_types = int(
        batch.action_type_mask[0].sum()
    )
    merge_sessions = int(
        batch.merge_session_mask[0].sum()
    )
    session_index = int(torch.nonzero(
        batch.merge_session_mask[0],
        as_tuple=False,
    )[0].item())
    merge_profiles = int(
        batch.merge_profile_mask[
            0, session_index
        ].sum()
    )
    profile_index = int(torch.nonzero(
        batch.merge_profile_mask[
            0, session_index
        ],
        as_tuple=False,
    )[0].item())
    indices = FactorizedActionIndices(
        torch.tensor([0]),
        torch.tensor([session_index]),
        torch.tensor([profile_index]),
    )
    log_probability, entropy = head.evaluate(
        logits, batch, indices,
    )
    assert log_probability.item() == pytest.approx(
        -log(action_types)
        - log(merge_sessions)
        - log(merge_profiles)
    )
    type_probability = 1.0 / action_types
    expected = log(action_types)
    expected += type_probability * (
        log(merge_sessions)
        + sum(
            (1.0 / merge_sessions)
            * log(int(
                batch.merge_profile_mask[
                    0, index
                ].sum()
            ))
            for index in range(
                batch.merge_session_mask.shape[1]
            )
            if batch.merge_session_mask[
                0, index
            ]
        )
    )
    if batch.action_type_mask[0, 1]:
        expected += type_probability * log(int(
            batch.create_profile_mask[0].sum()
        ))
    assert entropy.item() == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "action_type",
    (
        ActionType.CREATE,
        ActionType.DEFER,
        ActionType.REJECT,
    ),
)
def test_non_merge_branch_log_probabilities_follow_exact_factorization(
    action_type,
) -> None:
    batch = build_policy_batch(_observation())
    head = FactorizedPolicyHead(
        8, len(batch.profile_ids), 4,
    )
    logits = _zero_logits(batch)
    index = tuple(ActionType).index(
        action_type
    )
    if not batch.action_type_mask[0, index]:
        pytest.skip(
            f"{action_type.value} is infeasible "
            "in this public state"
        )
    if action_type is ActionType.CREATE:
        profile = int(torch.nonzero(
            batch.create_profile_mask[0],
            as_tuple=False,
        )[0].item())
        expected = (
            -log(int(
                batch.action_type_mask[0].sum()
            ))
            - log(int(
                batch.create_profile_mask[0].sum()
            ))
        )
    else:
        profile = -1
        expected = -log(int(
            batch.action_type_mask[0].sum()
        ))
    log_probability, _ = head.evaluate(
        logits,
        batch,
        FactorizedActionIndices(
            torch.tensor([index]),
            torch.tensor([-1]),
            torch.tensor([profile]),
        ),
    )
    assert log_probability.item() == pytest.approx(
        expected
    )


def test_stochastic_and_deterministic_selection_always_decode_public_feasible_actions() -> None:
    torch.manual_seed(29)
    batch = build_policy_batch(_observation())
    model = EdgeFreeSetActorCritic(
        batch.layout, ALGORITHM, CONFIG,
    )
    _, logits, _ = model(batch)
    deterministic = model.policy.select(
        logits, batch, deterministic=True,
    )
    assert (
        deterministic.actions[0]
        in batch.feasible_actions[0]
    )
    generator = torch.Generator().manual_seed(
        31
    )
    for _ in range(100):
        selected = model.policy.select(
            logits,
            batch,
            generator=generator,
        )
        assert (
            selected.actions[0]
            in batch.feasible_actions[0]
        )
        assert torch.isfinite(
            selected.log_probability
        ).all()
        assert torch.isfinite(
            selected.entropy
        ).all()


def test_deterministic_zero_logit_tie_uses_first_canonical_feasible_indices() -> None:
    batch = build_policy_batch(_observation())
    head = FactorizedPolicyHead(
        8, len(batch.profile_ids), 4,
    )
    selection = head.select(
        _zero_logits(batch),
        batch,
        deterministic=True,
    )
    expected_type = int(torch.nonzero(
        batch.action_type_mask[0],
        as_tuple=False,
    )[0].item())
    assert (
        selection.indices.action_type.item()
        == expected_type
    )
    if expected_type == 0:
        expected_session = int(torch.nonzero(
            batch.merge_session_mask[0],
            as_tuple=False,
        )[0].item())
        expected_profile = int(torch.nonzero(
            batch.merge_profile_mask[
                0, expected_session
            ],
            as_tuple=False,
        )[0].item())
        assert (
            selection.indices.merge_session.item()
            == expected_session
        )
        assert (
            selection.indices.profile.item()
            == expected_profile
        )


def test_conditional_masks_are_validated_once_at_batch_construction() -> None:
    batch = build_policy_batch(_observation())
    with pytest.raises(PolicyValidationError, match="MERGE feasibility"):
        replace(
            batch, action_type_mask=torch.tensor([[True, False, False, False]]),
            merge_session_mask=torch.zeros_like(batch.merge_session_mask),
            merge_profile_mask=torch.zeros_like(batch.merge_profile_mask),
        )
    with pytest.raises(PolicyValidationError, match="CREATE feasibility"):
        replace(
            batch, action_type_mask=torch.tensor([[False, True, False, False]]),
            merge_session_mask=torch.zeros_like(batch.merge_session_mask),
            merge_profile_mask=torch.zeros_like(batch.merge_profile_mask),
            create_profile_mask=torch.zeros_like(batch.create_profile_mask),
        )


def test_selected_typed_session_identifier_is_preserved_exactly() -> None:
    observation = _observation(
        session_id="1",
    )
    batch = build_policy_batch(observation)
    head = FactorizedPolicyHead(
        8, 4, 4,
    )
    logits = _zero_logits(batch)
    if not batch.action_type_mask[0, 0]:
        pytest.skip(
            "MERGE is not feasible in this public state"
        )
    selection = head.select(
        logits,
        batch,
        deterministic=True,
    )
    assert (
        selection.actions[0].action_type
        is ActionType.MERGE
    )
    assert (
        type(selection.actions[0].session_id)
        is str
    )
    assert selection.actions[0].session_id == "1"
    assert (
        selection.actions[0]
        in observation.set_view.action_masks.feasible_actions
    )


def test_hot_policy_paths_do_not_repeat_boundary_validation() -> None:
    import inspect

    forward = inspect.getsource(FactorizedPolicyHead.forward)
    evaluate = inspect.getsource(FactorizedPolicyHead.evaluate)
    select = inspect.getsource(FactorizedPolicyHead.select)
    assert "isfinite" not in forward and "PolicyValidationError" not in evaluate
    assert "feasible_actions" not in select


def test_shared_merge_option_score_is_masked_normalized_and_non_merge_logits_are_unchanged() -> None:
    original = build_policy_batch(_observation())
    merge_profile_mask = torch.tensor([[[True, False, True, False]]])
    feasible_actions = tuple(
        action
        for action in original.feasible_actions[0]
        if action.action_type is not ActionType.MERGE
        or action.profile_id in (original.profile_ids[0], original.profile_ids[2])
    )
    batch = replace(
        original,
        merge_profile_mask=merge_profile_mask,
        feasible_actions=(feasible_actions,),
    )
    raw_action_logits = torch.tensor([[0.25, -0.5, 0.75, -1.25]])
    merge_session_logits = torch.tensor([[0.4]])
    merge_profile_logits = torch.tensor([[[0.1, 1000.0, 0.7, -1000.0]]])
    logits = FactorizedPolicyLogits(
        raw_action_logits,
        merge_session_logits,
        merge_profile_logits,
        torch.zeros(1, len(batch.profile_ids)),
    )

    effective = _action_type_logits_with_merge_options(logits, batch, 0)
    pair_logits = merge_session_logits[0].unsqueeze(-1) + merge_profile_logits[0]
    valid_pair_logits = pair_logits.masked_select(batch.merge_profile_mask[0])
    expected = torch.logsumexp(valid_pair_logits, dim=0) - valid_pair_logits.new_tensor(valid_pair_logits.numel()).log()
    assert effective[0] == pytest.approx(raw_action_logits[0, 0] + expected)
    assert torch.equal(effective[1:], raw_action_logits[0, 1:])

    zero_logits = _zero_logits(batch)
    assert torch.equal(
        _action_type_logits_with_merge_options(zero_logits, batch, 0),
        zero_logits.action_type_logits[0],
    )


def test_duplicate_identical_merge_candidates_do_not_create_count_bonus() -> None:
    single = build_policy_batch(_observation())
    duplicate_session_id = "duplicate_session"
    doubled_input = replace(
        single.encoder_input,
        session_features=single.encoder_input.session_features.repeat(1, 2, 1),
        session_padding_mask=single.encoder_input.session_padding_mask.repeat(1, 2),
    )
    duplicated_merge_actions = tuple(
        EnvironmentAction(ActionType.MERGE, duplicate_session_id, action.profile_id)
        for action in single.feasible_actions[0]
        if action.action_type is ActionType.MERGE
    )
    doubled = replace(
        single,
        encoder_input=doubled_input,
        merge_session_mask=single.merge_session_mask.repeat(1, 2),
        merge_profile_mask=single.merge_profile_mask.repeat(1, 2, 1),
        session_ids=((single.session_ids[0][0], duplicate_session_id),),
        feasible_actions=(single.feasible_actions[0] + duplicated_merge_actions,),
    )
    profile_logits = torch.tensor([[[0.2, -0.4, 0.8, -0.1]]])
    single_logits = FactorizedPolicyLogits(
        torch.tensor([[0.3, -0.2, 0.1, -0.4]]),
        torch.tensor([[0.6]]),
        profile_logits,
        torch.zeros(1, len(single.profile_ids)),
    )
    doubled_logits = FactorizedPolicyLogits(
        single_logits.action_type_logits.clone(),
        single_logits.merge_session_logits.repeat(1, 2),
        profile_logits.repeat(1, 2, 1),
        single_logits.create_profile_logits.clone(),
    )
    assert torch.allclose(
        _action_type_logits_with_merge_options(single_logits, single, 0),
        _action_type_logits_with_merge_options(doubled_logits, doubled, 0),
    )


def test_shared_merge_option_score_is_used_by_select_evaluate_and_entropy() -> None:
    batch = build_policy_batch(_observation())
    head = FactorizedPolicyHead(8, len(batch.profile_ids), 4)
    logits = FactorizedPolicyLogits(
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[4.0]]),
        torch.zeros(1, 1, len(batch.profile_ids)),
        torch.zeros(1, len(batch.profile_ids)),
    )
    effective = _action_type_logits_with_merge_options(logits, batch, 0)
    selection = head.select(logits, batch, deterministic=True)
    assert selection.actions[0].action_type is ActionType.MERGE

    evaluated_log_probability, evaluated_entropy = head.evaluate(logits, batch, selection.indices)
    type_logs = torch.log_softmax(effective, dim=-1)
    type_probabilities = type_logs.exp()
    merge_profiles = int(batch.merge_profile_mask[0, 0].sum())
    create_profiles = int(batch.create_profile_mask[0].sum())
    expected_log_probability = type_logs[0]-log(merge_profiles)
    type_entropy = -(type_probabilities*type_logs).sum()
    expected_entropy = (
        type_entropy
        + type_probabilities[0]*log(merge_profiles)
        + type_probabilities[1]*log(create_profiles)
    )
    assert selection.log_probability.item() == pytest.approx(expected_log_probability)
    assert evaluated_log_probability.item() == pytest.approx(expected_log_probability)
    assert selection.entropy.item() == pytest.approx(expected_entropy)
    assert evaluated_entropy.item() == pytest.approx(expected_entropy)


def test_factor_log_probabilities_preserve_joint_policy_for_every_action_type() -> None:
    batch = build_policy_batch(_observation())
    head = FactorizedPolicyHead(8, len(batch.profile_ids), 4)
    logits = FactorizedPolicyLogits(
        torch.tensor([[0.3, -0.2, 0.4, -0.1]]), torch.tensor([[0.7]]),
        torch.tensor([[[0.2, -0.4, 0.8, -0.6]]]),
        torch.tensor([[0.5, -0.3, 0.1, -0.7]]),
    )
    merge_profile = int(torch.nonzero(batch.merge_profile_mask[0, 0], as_tuple=False)[0])
    create_profile = int(torch.nonzero(batch.create_profile_mask[0], as_tuple=False)[0])
    merge_session_applies = int(batch.merge_session_mask[0].sum()) > 1
    merge_profile_applies = int(batch.merge_profile_mask[0, 0].sum()) > 1
    create_profile_applies = int(batch.create_profile_mask[0].sum()) > 1
    cases = (
        (0, 0, merge_profile, merge_session_applies, merge_profile_applies),
        (1, -1, create_profile, False, create_profile_applies),
        (2, -1, -1, False, False),
        (3, -1, -1, False, False),
    )
    for action_type, session, profile, session_applies, profile_applies in cases:
        indices = FactorizedActionIndices(
            torch.tensor([action_type]), torch.tensor([session]), torch.tensor([profile]),
        )
        components = head.evaluate_components(logits, batch, indices)
        joint, entropy = head.evaluate(logits, batch, indices)
        assert torch.equal(components.joint, joint)
        assert components.merge_session_applicable.tolist() == [session_applies]
        assert components.profile_applicable.tolist() == [profile_applies]
        assert torch.isfinite(torch.stack((
            components.action_type[0], components.merge_session[0],
            components.profile[0], joint[0], entropy[0],
        ))).all()
        if not session_applies:
            assert components.merge_session.item() == 0.0
        if not profile_applies:
            assert components.profile.item() == 0.0


def test_policy_selection_exposes_exact_component_log_probabilities() -> None:
    batch = build_policy_batch(_observation())
    head = FactorizedPolicyHead(8, len(batch.profile_ids), 4)
    logits = _zero_logits(batch)
    selection = head.select(logits, batch, deterministic=True)
    evaluated = head.evaluate_components(logits, batch, selection.indices)
    assert torch.equal(selection.log_probability, evaluated.joint)
    selected = selection.factor_log_probabilities
    for left, right in (
        (selected.action_type, evaluated.action_type),
        (selected.merge_session, evaluated.merge_session),
        (selected.profile, evaluated.profile),
        (selected.merge_session_applicable, evaluated.merge_session_applicable),
        (selected.profile_applicable, evaluated.profile_applicable),
    ):
        assert torch.equal(left, right)


def test_hierarchical_prefix_critic_is_zero_residual_on_global_base_and_detaches_inputs() -> None:
    head = HierarchicalPrefixValueHead(8, 3)
    decision = torch.randn(2, 8, requires_grad=True)
    candidates = torch.randn(2, 5, 8, requires_grad=True)
    base = ValueOutput(
        torch.randn(2, requires_grad=True),
        torch.randn(2, 1, requires_grad=True),
        torch.randn(2, 2, requires_grad=True),
    )
    values = head(decision, candidates, base)
    assert values.type_reward_values.shape == (2, 4)
    assert values.type_constraint_values.shape == (2, 4, 3)
    assert values.merge_session_reward_values.shape == (2, 5)
    assert values.merge_session_constraint_values.shape == (2, 5, 3)
    base_constraints = torch.cat((base.sensing_sla_values, base.communication_qos_values), dim=1)
    assert torch.allclose(values.type_reward_values, base.reward_value.detach().unsqueeze(1).expand(-1, 4))
    assert torch.allclose(values.type_constraint_values, base_constraints.detach().unsqueeze(1).expand(-1, 4, -1))
    assert torch.allclose(values.merge_session_reward_values, base.reward_value.detach().unsqueeze(1).expand(-1, 5))
    assert torch.allclose(values.merge_session_constraint_values, base_constraints.detach().unsqueeze(1).expand(-1, 5, -1))
    loss = sum(item.sum() for item in (
        values.type_reward_values, values.type_constraint_values,
        values.merge_session_reward_values, values.merge_session_constraint_values,
    ))
    loss.backward()
    assert decision.grad is None and candidates.grad is None
    assert base.reward_value.grad is None
    assert base.sensing_sla_values.grad is None
    assert base.communication_qos_values.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_common_trace_actor_critic_preserves_common_initialization_and_outputs() -> None:
    batch = build_policy_batch(_observation())
    torch.manual_seed(73)
    joint = EdgeFreeSetActorCritic(batch.layout, ALGORITHM, CONFIG)
    torch.manual_seed(73)
    common_trace = CommonTraceActorCritic(batch.layout, ALGORITHM, CONFIG)
    common_trace_state = common_trace.state_dict()
    for name, value in joint.state_dict().items():
        assert torch.equal(value, common_trace_state[name])

    joint_encoded, joint_logits, joint_values = joint(batch)
    ct_encoded, ct_logits, ct_values, prefix_values = common_trace(batch)
    for left, right in (
        (joint_encoded.decision_embedding, ct_encoded.decision_embedding),
        (joint_encoded.merge_candidate_embeddings, ct_encoded.merge_candidate_embeddings),
        (joint_logits.action_type_logits, ct_logits.action_type_logits),
        (joint_logits.merge_session_logits, ct_logits.merge_session_logits),
        (joint_logits.merge_profile_logits, ct_logits.merge_profile_logits),
        (joint_logits.create_profile_logits, ct_logits.create_profile_logits),
        (joint_values.reward_value, ct_values.reward_value),
        (joint_values.sensing_sla_values, ct_values.sensing_sla_values),
        (joint_values.communication_qos_values, ct_values.communication_qos_values),
    ):
        assert torch.equal(left, right)

    constraint_count = len(CONFIG.tenants) + CONFIG.population["communication_users"]
    assert prefix_values.type_reward_values.shape == (1, 4)
    assert prefix_values.type_constraint_values.shape == (1, 4, constraint_count)
    assert prefix_values.merge_session_reward_values.shape == (1, 1)
    assert prefix_values.merge_session_constraint_values.shape == (1, 1, constraint_count)

    hidden = ALGORITHM.model.hidden_dim
    expected_extra = 2 * (hidden * hidden + hidden) + 5 * (constraint_count + 1) * (hidden + 1)
    assert sum(parameter.numel() for parameter in common_trace.prefix_value.parameters()) == expected_extra
    assert all(torch.isfinite(parameter).all() for parameter in common_trace.parameters())