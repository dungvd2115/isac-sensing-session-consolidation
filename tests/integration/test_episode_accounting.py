from __future__ import annotations

from dataclasses import replace
from math import isclose

import pytest

import isac_ssc.envs.isac_ssc_env as environment_module
from isac_ssc.core.entities import RequestState, Task
from isac_ssc.core.resources import normalized_sensing_resource_cost
from isac_ssc.core.sla import summarize_tenant_requests, tenant_episode_sla_residual
from isac_ssc.core.utility import finite_horizon_return
from isac_ssc.envs.action_space import ActionType, EnvironmentAction, identifier_key
from isac_ssc.envs.dynamics import generate_primitive_trace
from isac_ssc.envs.isac_ssc_env import EnvironmentValidationError, ISACSSCEnv
from isac_ssc.utils.config import load_config

CONFIG = load_config()


def _immediate(env: ISACSSCEnv):
    masks = env.current_action_masks()
    if masks is None:
        return None
    return next((
        action for action in masks.feasible_actions
        if action.action_type in {ActionType.MERGE, ActionType.CREATE}
    ), masks.feasible_actions[-1])


def test_focal_queue_arrivals_expiration_and_defer_cooldown_are_canonical() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    deferred = False
    while not env.terminated:
        state = env.state_snapshot()
        eligible = tuple(
            item for item in state.requests
            if item.state is RequestState.WAITING and item.eligible_slot <= state.current_slot
        )
        expected = min(
            eligible,
            key=lambda item: (item.eligible_slot, item.arrival_slot, identifier_key(item.request_id)),
            default=None,
        )
        assert state.focal_request_id == (None if expected is None else expected.request_id)
        if expected is not None and not deferred and env.current_action_masks().defer_feasible:
            before_slot = state.current_slot
            request_id = expected.request_id
            result = env.step(EnvironmentAction(ActionType.DEFER))
            updated = next(item for item in env.state_snapshot().requests if item.request_id == request_id)
            assert updated.eligible_slot == before_slot+CONFIG.requests["defer_cooldown_slots"]
            assert not result.accepted_request_ids and not result.rejected_request_ids
            deferred = True
        else:
            env.step(_immediate(env))
    assert deferred


def test_retained_quality_is_reused_and_one_session_update_serves_all_members() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "clustered")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    observed_reuse = observed_shared_members = False
    while not env.terminated:
        masks = env.current_action_masks()
        action = _immediate(env)
        retained = session_id = None
        if action is not None and action.action_type in {ActionType.MERGE, ActionType.CREATE}:
            entry = masks.entry_for(action)
            if action.action_type is ActionType.MERGE:
                retained = entry.merge_assessment.shared_quality
                session_id = action.session_id
            else:
                retained = entry.create_assessment.shared_quality
                session_id = entry.create_assessment.candidate_session.session_id
        result = env.step(action)
        if retained is not None:
            record = next(item for item in result.session_updates if item.session_id == session_id)
            assert record.shared_quality is retained
            assert session_id in result.sensing_resource_usage.updating_session_ids
            observed_reuse = True
        for record in result.session_updates:
            assert sum(item.session_id == record.session_id for item in result.session_updates) == 1
            if len(record.valid_request_ids) > 1:
                observed_shared_members = True
        assert len(result.session_updates) == len(result.sensing_resource_usage.updating_session_ids)
    assert observed_reuse
    assert observed_shared_members


def test_tracking_prediction_occurs_once_per_later_active_slot_and_never_creation_slot(monkeypatch) -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "clustered")
    calls = []
    original = environment_module.predict_tracking_covariance

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(environment_module, "predict_tracking_covariance", counted)
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    reported = 0
    while not env.terminated:
        state = env.state_snapshot()
        session_index = {session.session_id: session for session in state.active_sessions}
        expected = tuple(
            session.session_id for session in state.active_sessions
            if Task.TRACKING in session.exposed_outputs and session.creation_slot < state.current_slot
        )
        result = env.step(_immediate(env))
        assert set(result.tracking_prediction_session_ids) == set(expected)
        for record in result.session_updates:
            tracking = record.shared_quality.tracking
            if tracking is not None and not tracking.measurement_updated:
                assert tracking.posterior_covariance == session_index[record.session_id].tracking_covariance
        assert all(
            next(session for session in state.active_sessions if session.session_id == item).creation_slot
            < result.processed_slot for item in result.tracking_prediction_session_ids
        )
        reported += len(result.tracking_prediction_session_ids)
    assert len(calls) == reported


def test_reward_and_constraint_accounting_reconstruct_exactly() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    results = []
    while not env.terminated:
        result = env.step(_immediate(env))
        expected_cost = normalized_sensing_resource_cost(
            result.sensing_resource_usage, CONFIG.system["total_bandwidth_hz"],
            CONFIG.system["total_power_w"], CONFIG.reward["sensing_cost_bandwidth_weight"],
            CONFIG.reward["sensing_cost_power_weight"],
        )
        assert result.sensing_resource_cost == pytest.approx(expected_cost)
        assert result.reward == pytest.approx(
            result.completed_value-CONFIG.reward["sensing_resource_cost_weight"]*expected_cost
        )
        for record in result.communication_service:
            expected = 0.0 if record.demand_bit_per_s == 0.0 else (
                record.normalized_shortfall-CONFIG.communication["normalized_shortfall_budget"]
            )
            assert record.residual == pytest.approx(expected)
        results.append(result)
    state = env.state_snapshot()
    assert state.cumulative_reward == pytest.approx(finite_horizon_return(item.reward for item in results))
    assert state.cumulative_completed_value == pytest.approx(sum(item.completed_value for item in results))
    assert state.cumulative_sensing_resource_cost == pytest.approx(
        sum(item.sensing_resource_cost for item in results)
    )
    requests = state.requests
    for tenant, accounting in zip(CONFIG.tenants, state.tenant_accounting, strict=True):
        summary = summarize_tenant_requests(requests, tenant.tenant_id)
        assert accounting.accepted_count == summary.accepted_count
        assert accounting.first_violated_count == summary.first_violated_count
        assert accounting.completed_count == summary.completed_count
        assert accounting.residual == pytest.approx(
            tenant_episode_sla_residual(summary, tenant.sla_violation_budget)
        )
    for accounting in state.communication_accounting:
        assert accounting.residual_sum == pytest.approx(
            accounting.shortfall_sum
            - CONFIG.communication["normalized_shortfall_budget"]*accounting.active_demand_slots
        )
    for tenant in CONFIG.tenants:
        assert sum(dict(item.tenant_sla_residuals)[tenant.tenant_id] for item in results) == pytest.approx(
            next(item.residual for item in state.tenant_accounting if item.tenant_id == tenant.tenant_id)
        )


def test_valid_output_freshness_terminal_outcome_and_next_slot_detachment() -> None:
    trace = generate_primitive_trace(CONFIG, 41001, "independent")
    env = ISACSSCEnv(CONFIG)
    env.reset(trace)
    observed_terminal = observed_valid = observed_first_violation = False
    while not env.terminated:
        result = env.step(_immediate(env))
        observed_valid |= bool(result.valid_output_request_ids)
        observed_first_violation |= bool(result.first_violation_request_ids)
        terminal_ids = set(result.completed_request_ids) | set(result.failed_request_ids)
        if terminal_ids and not result.terminated:
            member_ids = {
                item for session in env.state_snapshot().active_sessions for item in session.member_request_ids
            }
            assert terminal_ids.isdisjoint(member_ids)
            observed_terminal = True
    assert observed_valid and observed_first_violation and observed_terminal
    for request in env.state_snapshot().requests:
        if request.state is RequestState.COMPLETED:
            assert request.valid_output_count >= 1 and not request.sla_violated
        if request.state is RequestState.FAILED:
            assert request.sla_violated
