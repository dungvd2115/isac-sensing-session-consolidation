from __future__ import annotations

import pytest

from isac_ssc.core.entities import DiskAOI, RequestState, SensingRequest, Task
from isac_ssc.core.sla import (
    CommunicationQosSummary, SlaValidationError, communication_qos_slot,
    effective_communication_target_bit_per_s, initialize_admitted_request,
    normalized_communication_shortfall, summarize_communication_qos,
    summarize_tenant_requests, tenant_episode_sla_residual, tenant_slot_sla_residual,
    update_request_sla,
)
from isac_ssc.utils.config import load_config

CONFIG = load_config()
DURATIONS = CONFIG.service_duration_slots


def _waiting(request_id: int, task: Task = Task.TRACKING, tenant_id: str = "tenant_1",
             interval: int = 2, completion_value: float = 2.0) -> SensingRequest:
    threshold = 0.9 if task is Task.DETECTION else 4.0
    return SensingRequest(
        request_id=request_id, tenant_id=tenant_id, arrival_slot=0, latest_start_slot=6,
        aoi=DiskAOI((80.0, 0.0), 25.0), target_id=1, task=task,
        quality_threshold=threshold, valid_output_interval_slots=interval,
        completion_value=completion_value, merge_permission=True,
    )


def _active(request_id: int, task: Task = Task.TRACKING, tenant_id: str = "tenant_1",
            interval: int = 2, admission_slot: int = 0) -> SensingRequest:
    return _waiting(request_id, task, tenant_id, interval).transition(
        RequestState.ACTIVE, slot=admission_slot,
    )


def test_admission_initialization_does_not_grant_freshness() -> None:
    initialized = initialize_admitted_request(_active(1))
    assert initialized.valid_output_age_slots == initialized.valid_output_interval_slots + 1
    assert initialized.valid_output_count == 0
    assert not initialized.sla_violated


def test_valid_output_resets_age_and_increments_count() -> None:
    request = initialize_admitted_request(_active(1))
    result = update_request_sla(request, 0, DURATIONS, valid_output=True)
    assert result.valid_output_event and not result.first_violation_event
    assert result.updated_request.valid_output_age_slots == 0
    assert result.updated_request.valid_output_count == 1
    assert result.updated_request.state is RequestState.ACTIVE


def test_invalid_output_after_admission_emits_one_absorbing_first_violation() -> None:
    request = initialize_admitted_request(_active(1))
    first = update_request_sla(request, 0, DURATIONS, valid_output=False)
    assert first.first_violation_event
    assert first.updated_request.sla_violated
    assert first.updated_request.first_violation_slot == 0
    assert first.updated_request.state is RequestState.ACTIVE

    repeated = update_request_sla(first.updated_request, 1, DURATIONS, valid_output=False)
    assert not repeated.first_violation_event
    assert repeated.updated_request.first_violation_slot == 0
    assert repeated.updated_request.state is RequestState.ACTIVE


def test_fresh_output_delays_violation_until_age_exceeds_interval() -> None:
    request = initialize_admitted_request(_active(1, interval=2))
    request = update_request_sla(request, 0, DURATIONS, valid_output=True).updated_request
    request = update_request_sla(request, 1, DURATIONS, valid_output=False).updated_request
    request = update_request_sla(request, 2, DURATIONS, valid_output=False).updated_request
    assert not request.sla_violated and request.valid_output_age_slots == 2
    result = update_request_sla(request, 3, DURATIONS, valid_output=False)
    assert result.first_violation_event and result.updated_request.valid_output_age_slots == 3


def test_first_violation_does_not_terminate_before_final_service_slot() -> None:
    request = initialize_admitted_request(_active(1, Task.TRACKING))
    result = update_request_sla(request, 0, DURATIONS, valid_output=False)
    assert result.updated_request.state is RequestState.ACTIVE
    assert not result.completed_event and not result.failed_event
    assert result.updated_request.final_service_slot(DURATIONS) == 7


def test_successful_final_slot_completion_requires_valid_output_and_no_violation() -> None:
    request = initialize_admitted_request(_active(1, Task.DETECTION))
    request = update_request_sla(request, 0, DURATIONS, valid_output=True).updated_request
    request = update_request_sla(request, 1, DURATIONS, valid_output=False).updated_request
    result = update_request_sla(request, 2, DURATIONS, valid_output=True)
    assert result.completed_event and not result.failed_event
    assert result.updated_request.state is RequestState.COMPLETED
    assert result.updated_request.valid_output_count == 2


def test_violated_or_never_valid_request_fails_at_final_slot_without_duplicate_event() -> None:
    request = initialize_admitted_request(_active(1, Task.DETECTION))
    first = update_request_sla(request, 0, DURATIONS, valid_output=False)
    middle = update_request_sla(first.updated_request, 1, DURATIONS, valid_output=True)
    final = update_request_sla(middle.updated_request, 2, DURATIONS, valid_output=True)
    assert final.failed_event and not final.first_violation_event
    assert final.updated_request.state is RequestState.FAILED

    never_valid = initialize_admitted_request(_active(2, Task.DETECTION))
    never_valid = update_request_sla(never_valid, 0, DURATIONS, valid_output=False).updated_request
    never_valid = update_request_sla(never_valid, 1, DURATIONS, valid_output=False).updated_request
    final_never = update_request_sla(never_valid, 2, DURATIONS, valid_output=False)
    assert final_never.failed_event and final_never.updated_request.valid_output_count == 0
    assert not final_never.first_violation_event


def test_tenant_counts_violation_rate_and_episode_residual_reconstruct_events() -> None:
    completed = initialize_admitted_request(_active(1, Task.DETECTION))
    completed = update_request_sla(completed, 0, DURATIONS, valid_output=True).updated_request
    completed = update_request_sla(completed, 1, DURATIONS, valid_output=True).updated_request
    completed = update_request_sla(completed, 2, DURATIONS, valid_output=True).updated_request

    failed = initialize_admitted_request(_active(2, Task.DETECTION))
    failed = update_request_sla(failed, 0, DURATIONS, valid_output=False).updated_request
    failed = update_request_sla(failed, 1, DURATIONS, valid_output=False).updated_request
    failed = update_request_sla(failed, 2, DURATIONS, valid_output=False).updated_request
    rejected = _waiting(3).transition(RequestState.REJECTED)
    expired = _waiting(4).transition(RequestState.EXPIRED)

    summary = summarize_tenant_requests((completed, failed, rejected, expired), "tenant_1")
    assert summary.arrived_count == 4 and summary.accepted_count == 2
    assert summary.first_violated_count == 1 and summary.completed_count == 1
    assert summary.failed_count == summary.rejected_count == summary.expired_count == 1
    assert summary.violation_rate == pytest.approx(0.5)

    budget = 0.05
    slot_residuals = (
        tenant_slot_sla_residual(2, 0, budget),
        tenant_slot_sla_residual(0, 1, budget),
    )
    assert sum(slot_residuals) == pytest.approx(tenant_episode_sla_residual(summary, budget))
    assert tenant_episode_sla_residual(summary, budget) == pytest.approx(1.0 - 2.0 * budget)


def test_zero_accepted_tenant_reports_not_applicable_violation_rate() -> None:
    requests = (_waiting(1).transition(RequestState.REJECTED), _waiting(2).transition(RequestState.EXPIRED))
    summary = summarize_tenant_requests(requests, "tenant_1")
    assert summary.accepted_count == 0 and summary.completed_count == 0
    assert summary.violation_rate is None


def test_communication_effective_target_shortfall_and_slot_residual() -> None:
    assert effective_communication_target_bit_per_s(1.0e6, 2.0e6) == 1.0e6
    assert effective_communication_target_bit_per_s(5.0e6, 2.0e6) == 2.0e6
    assert normalized_communication_shortfall(5.0e6, 2.0e6, 1.5e6) == pytest.approx(0.25)
    slot = communication_qos_slot(5.0e6, 2.0e6, 1.5e6, 0.05)
    assert slot.active_demand and slot.normalized_shortfall == pytest.approx(0.25)
    assert slot.residual == pytest.approx(0.20)


def test_zero_demand_has_zero_shortfall_and_zero_residual() -> None:
    slot = communication_qos_slot(0.0, 2.0e6, 0.0, 0.05)
    assert not slot.active_demand
    assert slot.effective_target_bit_per_s == slot.normalized_shortfall == slot.residual == 0.0


def test_communication_episode_summary_uses_na_for_zero_active_demand() -> None:
    empty = summarize_communication_qos((communication_qos_slot(0.0, 2.0e6, 0.0, 0.05),))
    assert empty == CommunicationQosSummary(0, 0.0, 0.0)
    assert empty.mean_shortfall is None

    slots = (
        communication_qos_slot(5.0e6, 2.0e6, 1.5e6, 0.05),
        communication_qos_slot(0.0, 2.0e6, 0.0, 0.05),
        communication_qos_slot(1.0e6, 2.0e6, 1.0e6, 0.05),
    )
    summary = summarize_communication_qos(slots)
    assert summary.active_demand_slots == 2
    assert summary.shortfall_sum == pytest.approx(0.25)
    assert summary.residual_sum == pytest.approx(0.15)
    assert summary.mean_shortfall == pytest.approx(0.125)


def test_sla_update_rejects_slots_outside_service_interval() -> None:
    request = initialize_admitted_request(_active(1, Task.DETECTION))
    with pytest.raises(SlaValidationError):
        update_request_sla(request, 3, DURATIONS, valid_output=True)