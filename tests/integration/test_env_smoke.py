from __future__ import annotations

import numpy as np
import pytest

from isac_ssc.envs.action_space import ActionType
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import EnvironmentValidationError, ISACSSCEnv
from isac_ssc.utils.config import load_config

CONFIG = load_config()


def _action(env: ISACSSCEnv):
    masks = env.current_action_masks()
    if masks is None:
        return None
    return next((
        action for action in masks.feasible_actions
        if action.action_type in {ActionType.MERGE, ActionType.CREATE}
    ), masks.feasible_actions[-1])


def _run(env: ISACSSCEnv):
    results = []
    while not env.terminated:
        results.append(env.step(_action(env)))
    return tuple(results)


@pytest.mark.parametrize("regime", ("independent", "clustered"))
def test_registered_trace_runs_exactly_one_step_per_physical_slot(regime: str) -> None:
    trace = generate_primitive_trace(CONFIG, 41001, regime)
    env = ISACSSCEnv(CONFIG)
    initial = env.reset(trace)
    assert initial is env.current_observation()
    results = _run(env)
    assert len(results) == trace.horizon_slots == 200
    assert tuple(item.processed_slot for item in results) == tuple(range(trace.horizon_slots))
    assert env.terminated and env.current_slot == trace.horizon_slots
    assert env.current_action_masks() is None and env.current_observation() is None
    assert results[-1].terminated and results[-1].next_observation is None


def test_no_focal_slot_requires_none_and_still_executes_physical_service() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    observed_no_request = False
    while not env.terminated:
        masks = env.current_action_masks()
        if masks is None:
            result = env.step(None)
            assert result.action is None and result.focal_request_id is None
            assert len(result.communication_service) == CONFIG.population["communication_users"]
            observed_no_request = True
        else:
            result = env.step(_action(env))
    assert observed_no_request


def test_clustered_pending_children_match_current_full_markov_state() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "clustered")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    for _ in range(40):
        state = env.state_snapshot()
        assert state.pending_clustered_children == trace.pending_children_at(state.current_slot)
        env.step(_action(env))


def test_environment_execution_does_not_use_numpy_rng(monkeypatch) -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "clustered")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)

    def forbidden(*args, **kwargs):
        raise AssertionError("environment execution must not draw randomness")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "random", forbidden)
    _run(env)


def test_reset_clears_all_episode_state_and_reproduces_initial_decision() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env = ISACSSCEnv(CONFIG)
    first = env.reset(trace)
    first_state = env.state_snapshot()
    for _ in range(12):
        env.step(_action(env))
    second = env.reset(trace)
    assert second == first
    assert env.state_snapshot() == first_state
    assert env.current_slot == 0 and not env.terminated


def test_public_api_rejects_use_before_reset_and_after_termination() -> None:
    env = ISACSSCEnv(CONFIG)
    with pytest.raises(EnvironmentValidationError, match="reset"):
        env.current_observation()
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env.reset(trace)
    _run(env)
    with pytest.raises(EnvironmentValidationError, match="terminated"):
        env.step(None)