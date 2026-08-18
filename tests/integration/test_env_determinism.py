from __future__ import annotations

import random

import numpy as np
import pytest

from isac_ssc.envs.action_space import ActionType, EnvironmentAction
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import EnvironmentValidationError, ISACSSCEnv
from isac_ssc.utils.config import load_config
from isac_ssc.utils.serialization import trace_to_dict

CONFIG = load_config()


def _policy(env: ISACSSCEnv):
    masks = env.current_action_masks()
    if masks is None:
        return None
    return next((
        action for action in masks.feasible_actions
        if action.action_type in {ActionType.MERGE, ActionType.CREATE}
    ), masks.feasible_actions[-1])


def _rollout(trace):
    env = ISACSSCEnv(CONFIG)
    initial = env.reset(trace)
    results = []
    while not env.terminated:
        results.append(env.step(_policy(env)))
    return initial, tuple(results), env.state_snapshot()


def test_two_environments_and_repeated_reset_replay_exactly() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "clustered")
    original = trace_to_dict(trace)
    first = _rollout(trace)
    second = _rollout(trace)
    assert first == second
    assert trace_to_dict(trace) == original

    env = ISACSSCEnv(CONFIG)
    initial_one = env.reset(trace)
    results_one = []
    while not env.terminated:
        results_one.append(env.step(_policy(env)))
    final_one = env.state_snapshot()
    initial_two = env.reset(trace)
    results_two = []
    while not env.terminated:
        results_two.append(env.step(_policy(env)))
    assert initial_one == initial_two
    assert tuple(results_one) == tuple(results_two)
    assert final_one == env.state_snapshot()


def test_python_and_numpy_global_rng_state_do_not_change_replay() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    random.seed(1)
    np.random.seed(1)
    first = _rollout(trace)
    random.seed(987654)
    np.random.seed(987654)
    second = _rollout(trace)
    assert first == second


def test_invalid_or_masked_action_is_rejected_before_mutation() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    while env.current_action_masks() is None:
        env.step(None)
    before = env.state_snapshot()
    with pytest.raises(EnvironmentValidationError, match="catalogue"):
        env.step(EnvironmentAction(ActionType.CREATE, profile_id="missing"))
    assert env.state_snapshot() == before
    masked = next((entry.action for entry in env.current_action_masks().entries if not entry.feasible), None)
    if masked is not None:
        with pytest.raises(EnvironmentValidationError, match="masked"):
            env.step(masked)
        assert env.state_snapshot() == before
    with pytest.raises(EnvironmentValidationError, match="requires an EnvironmentAction"):
        env.step(None)
    assert env.state_snapshot() == before


def test_session_counter_is_consumed_only_by_successful_create() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    while env.current_action_masks() is None:
        env.step(None)
    assert env.state_snapshot().next_session_counter == 0
    env.step(EnvironmentAction(ActionType.REJECT))
    assert env.state_snapshot().next_session_counter == 0
    while not env.terminated:
        masks = env.current_action_masks()
        if masks is None:
            env.step(None)
            continue
        create = next((
            action for action in masks.feasible_actions if action.action_type is ActionType.CREATE
        ), None)
        if create is not None:
            prospective = masks.prospective_session_id
            env.step(create)
            assert env.state_snapshot().next_session_counter == 1
            assert any(session.session_id == prospective for session in env.state_snapshot().active_sessions)
            break
        env.step(EnvironmentAction(ActionType.REJECT))
    else:
        pytest.fail("registered trace did not expose a feasible CREATE action")