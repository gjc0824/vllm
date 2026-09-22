# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.layered_prefill import (
    LayeredFrontier,
    LayeredPrefillConfig,
    LayeredPrefillPlan,
    LayeredPrefillPolicy,
    LayeredPrefillStateStore,
    make_layer_group_ranges,
    make_pp_aligned_layer_group_ranges,
    select_num_groups,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import (
    _LAYERED_STALL_RECOVERY_STEPS,
    Scheduler,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus


def _config(
    *,
    enabled: bool = True,
    layers: int = 8,
    tensor_parallel_size: int = 1,
    use_sequence_parallel_moe: bool = False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            num_hidden_layers=layers,
            enforce_eager=True,
        ),
        additional_config={
            "enable_dsa_cp": False,
            "scheduler_config": {
                "layered_prefill_config": {"enabled": enabled}
            }
        },
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
        ),
    )


def test_layer_group_ranges_cover_each_layer_once():
    ranges = make_layer_group_ranges(10, 4)

    assert [(item.start, item.end) for item in ranges] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 10),
    ]
    assert ranges[-1].end == 10
    assert sum(item.end - item.start for item in ranges) == 10


def test_pp_layer_group_ranges_do_not_cross_stage_boundaries():
    ranges = make_pp_aligned_layer_group_ranges(8, 4, 2)

    assert [(item.start, item.end) for item in ranges] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]


def test_pp_layer_group_ranges_follow_custom_partition(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,6")
    ranges = make_pp_aligned_layer_group_ranges(8, 4, 2)

    assert [(item.start, item.end) for item in ranges] == [
        (0, 1),
        (1, 2),
        (2, 5),
        (5, 8),
    ]


@pytest.mark.parametrize(
    "prompt_tokens, expected_groups",
    [(1, 1), (512, 1), (513, 2), (2048, 4), (10000, 16)],
)
def test_select_num_groups_uses_stable_layout_buckets(prompt_tokens, expected_groups):
    assert select_num_groups(prompt_tokens, 32) == expected_groups


def test_plan_separates_query_and_logical_commit():
    plan = LayeredPrefillPlan(
        version=1,
        cohort_id=3,
        group_id=0,
        num_groups=2,
        group_start=0,
        group_end=4,
        prefill_req_ids=("req",),
        query_tokens={"req": 16},
        commit_tokens={"req": 0},
    )

    assert plan.is_final_group is False
    assert plan.query_tokens["req"] == 16
    assert plan.commit_tokens["req"] == 0

    final_plan = LayeredPrefillPlan(
        version=1,
        cohort_id=3,
        group_id=1,
        num_groups=2,
        group_start=4,
        group_end=8,
        prefill_req_ids=("req",),
        query_tokens={"req": 16},
        commit_tokens={"req": 16},
    )
    assert final_plan.is_final_group is True


def test_intermediate_plan_rejects_partial_commit():
    with pytest.raises(ValueError, match="intermediate group"):
        LayeredPrefillPlan(
            version=1,
            cohort_id=0,
            group_id=0,
            num_groups=2,
            group_start=0,
            group_end=1,
            prefill_req_ids=("req",),
            query_tokens={"req": 2},
            commit_tokens={"req": 1},
        )


def test_policy_initializes_request_and_advances_groups():
    policy = LayeredPrefillPolicy(_config(layers=8))
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=513,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_num_groups == 2
    assert request.layered_prefill_group_id == 0

    first = policy.make_plan(request)
    assert first.group_id == 0
    assert first.commit_tokens["req"] == 0
    assert first.reuse_kv_blocks is False
    request.layered_prefill_group_id += 1
    request.layered_prefill_kv_reserved = True
    second = policy.make_plan(request)
    assert second.group_id == 1
    assert second.commit_tokens["req"] == 513
    assert second.reuse_kv_blocks is True


def test_policy_aligns_pp_groups_with_stage_partitions():
    config = _config(layers=8)
    config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_num_groups == 2
    plan = policy.make_plan(request)
    assert (plan.group_start, plan.group_end) == (0, 4)


def test_policy_keeps_unaligned_tail_in_layered_query():
    config = _config(layers=8, tensor_parallel_size=8)
    config.model_config.hf_text_config = SimpleNamespace(index_topk=2048)
    config.additional_config["enable_dsa_cp"] = True
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=65540,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_query_tokens == 65540

    plan = policy.make_plan(request)
    assert plan.query_tokens["req"] == 65540


def test_frontier_store_is_keyed_by_request_id():
    store = LayeredPrefillStateStore()
    frontier = LayeredFrontier(
        req_id="req",
        group_id=1,
        query_len=2,
        hidden_states="hidden",
        residual="residual",
    )
    store.put(frontier)
    assert store.get("req") is frontier
    assert store.pop("req") is frontier
    assert store.get("req") is None


def test_disabled_config_does_not_require_model_layer_count():
    config = SimpleNamespace(
        model_config=SimpleNamespace(),
        additional_config={"scheduler_config": {}},
    )
    policy = LayeredPrefillPolicy(config)
    assert policy.enabled is False
    assert policy.num_hidden_layers == 0
    assert LayeredPrefillConfig().enabled is False


def test_config_parses_phase_one_scheduler_options():
    config = LayeredPrefillConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                "scheduler_config": {
                    "layered_prefill_config": {
                        "enabled": True,
                        "group_token_target": 1024,
                        "allowed_num_groups": [1, 2, 4],
                        "max_groups_per_step": 1,
                    }
                }
            }
        )
    )

    assert config.enabled is True
    assert config.group_token_target == 1024
    assert config.allowed_num_groups == (1, 2, 4)


def test_config_rejects_multiple_groups_per_step_in_phase_one():
    with pytest.raises(ValueError, match="max_groups_per_step=1"):
        LayeredPrefillConfig(enabled=True, max_groups_per_step=2)


_LAYERED_ASYNC_ADDITIONAL_CONFIG = {
    "scheduler_config": {
        "layered_prefill_config": {
            "enabled": True,
            # The test model is not forced eager and the cohort runs without
            # Decode work, so relax the phase-one eligibility knobs.
            "require_eager": False,
            "require_pd_mixed": False,
        }
    }
}


def _run_async_step(scheduler: Scheduler, sched_output: SchedulerOutput) -> None:
    """Emulate the worker for one scheduler step.

    Only the final layer group of the final chunk samples a token; every
    other scheduled row (regular Decode or an intermediate layer group)
    returns none, mirroring the layered worker protocol.
    """
    plan = sched_output.layered_prefill_plan
    req_ids = list(sched_output.num_scheduled_tokens)
    sampled_token_ids = [
        [0]
        if plan is None or plan.is_sampling_step or req_id not in plan.query_tokens
        else []
        for req_id in req_ids
    ]
    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(sched_output, model_runner_output)


@pytest.mark.cpu_test
def test_async_scheduler_layered_prefill_placeholder_protocol():
    """The layered request's first sampled token must be accounted as an
    async output placeholder when the final layer group is scheduled, and
    cleared once the worker delivers the token."""
    scheduler = create_scheduler(
        async_scheduling=True,
        additional_config=_LAYERED_ASYNC_ADDITIONAL_CONFIG,
        cudagraph_mode="NONE",
        enforce_eager=True,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=520, max_tokens=2, ignore_eos=True
    )
    scheduler.add_request(request)

    sched_output = scheduler.schedule()
    plan = sched_output.layered_prefill_plan
    assert plan is not None and plan.num_groups == 2
    # Intermediate layer group: the request executes but samples nothing.
    assert not plan.is_sampling_step
    assert request.num_output_placeholders == 0
    _run_async_step(scheduler, sched_output)
    assert request.num_output_placeholders == 0

    sched_output = scheduler.schedule()
    plan = sched_output.layered_prefill_plan
    assert plan is not None and plan.is_sampling_step
    # The step samples the request's first output token, so the async
    # protocol requires one placeholder at schedule time.
    assert request.num_output_placeholders == 1
    _run_async_step(scheduler, sched_output)
    assert request.num_output_placeholders == 0

    # The prompt finished; the request graduates to regular async decode.
    sched_output = scheduler.schedule()
    assert sched_output.layered_prefill_plan is None
    assert request.num_output_placeholders == 1
    _run_async_step(scheduler, sched_output)
    assert request.num_output_placeholders == 0
    assert request.num_output_tokens == 2
    assert scheduler.get_num_unfinished_requests() == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
def test_async_scheduler_layered_prefill_first_decode_step_has_no_draft_slots(
    num_speculative_tokens: int,
):
    """The layered request's first decode step schedules no draft slots.

    The async placeholder loop in ``_update_after_schedule`` runs before the
    layered append, so the request's spec_token_ids stay empty when the final
    layer group samples.  Draft slots only appear from the step after its
    first decode step.  This is the contract that lets the worker skip the
    layered P subbatch's propose under async scheduling without losing any
    consumer.
    """
    scheduler = create_scheduler(
        async_scheduling=True,
        additional_config=_LAYERED_ASYNC_ADDITIONAL_CONFIG,
        cudagraph_mode="NONE",
        enforce_eager=True,
        num_speculative_tokens=num_speculative_tokens,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=520, max_tokens=6, ignore_eos=True
    )
    scheduler.add_request(request)
    req_id = request.request_id

    # Intermediate group: no sampling, no draft slots.
    sched_output = scheduler.schedule()
    assert sched_output.layered_prefill_plan is not None
    assert req_id not in sched_output.scheduled_spec_decode_tokens
    _run_async_step(scheduler, sched_output)

    # Final group samples the first token; still no draft slots.
    sched_output = scheduler.schedule()
    assert sched_output.layered_prefill_plan is not None
    assert sched_output.layered_prefill_plan.is_sampling_step
    assert req_id not in sched_output.scheduled_spec_decode_tokens
    _run_async_step(scheduler, sched_output)

    # First decode step: exactly one token, no draft slots.
    sched_output = scheduler.schedule()
    assert sched_output.layered_prefill_plan is None
    assert req_id not in sched_output.scheduled_spec_decode_tokens
    assert sched_output.num_scheduled_tokens[req_id] == 1
    _run_async_step(scheduler, sched_output)

    # From the next step on, regular async spec decode provides the slots.
    sched_output = scheduler.schedule()
    assert sched_output.layered_prefill_plan is None
    draft_slots = sched_output.scheduled_spec_decode_tokens[req_id]
    assert len(draft_slots) == num_speculative_tokens
    assert sched_output.num_scheduled_tokens[req_id] == 1 + num_speculative_tokens


@pytest.mark.cpu_test
def test_async_scheduler_layered_prefill_gate_falls_back_for_pp2():
    """async scheduling with PP>1 stays on regular token scheduling."""
    scheduler = create_scheduler(
        async_scheduling=True,
        additional_config=_LAYERED_ASYNC_ADDITIONAL_CONFIG,
        cudagraph_mode="NONE",
        enforce_eager=True,
    )
    # Emulate PP>1 at the scheduler level; constructing with
    # pipeline_parallel_size>1 requires that many visible GPUs.
    scheduler.parallel_config.pipeline_parallel_size = 2
    assert not scheduler._layered_prefill_supported_for_scheduler()

    (request,) = create_requests(
        num_requests=1, num_tokens=520, max_tokens=2, ignore_eos=True
    )
    scheduler.add_request(request)
    sched_output = scheduler.schedule()
    assert sched_output.layered_prefill_plan is None
    assert not request.layered_prefill_enabled


@pytest.mark.cpu_test
def test_layered_admission_failure_does_not_starve_decode():
    """When the waiting candidate's full-prompt reservation fails, the step
    must fall back to regular scheduling with the full token budget so the
    running request's Decode rows keep executing, instead of the doomed
    candidate's query budget starving them into zero-token steps."""
    scheduler = create_scheduler(
        async_scheduling=True,
        additional_config=_LAYERED_ASYNC_ADDITIONAL_CONFIG,
        cudagraph_mode="NONE",
        enforce_eager=True,
    )
    first, blocked = create_requests(
        num_requests=2, num_tokens=520, max_tokens=8, ignore_eos=True
    )
    scheduler.add_request(first)
    scheduler.add_request(blocked)

    # The first request runs through its two layer groups and samples.
    for _ in range(2):
        sched_output = scheduler.schedule()
        assert sched_output.total_num_scheduled_tokens > 0
        assert scheduler._layered_stall_steps == 0
        _run_async_step(scheduler, sched_output)

    # `first` is now a decode request. Fail every full-prompt reservation
    # (the pool cannot fit a second request) while small decode allocations
    # keep succeeding.
    kv_cache_manager = scheduler.kv_cache_manager
    original_allocate_slots = kv_cache_manager.allocate_slots

    def fail_full_prompt_reservations(request, num_new_tokens, *args, **kwargs):
        if num_new_tokens > 100:
            return None
        return original_allocate_slots(request, num_new_tokens, *args, **kwargs)

    kv_cache_manager.allocate_slots = fail_full_prompt_reservations
    try:
        for _ in range(3):
            sched_output = scheduler.schedule()
            # The decode row survives the failed layered admission.
            assert sched_output.total_num_scheduled_tokens == 1
            assert first.status == RequestStatus.RUNNING
            assert blocked.status == RequestStatus.WAITING
            assert scheduler._layered_stall_steps == 0
            _run_async_step(scheduler, sched_output)
    finally:
        kv_cache_manager.allocate_slots = original_allocate_slots

    # With the reservation restorable the waiting request is admitted again.
    sched_output = scheduler.schedule()
    assert blocked.status == RequestStatus.RUNNING
    assert sched_output.total_num_scheduled_tokens > 0


@pytest.mark.cpu_test
def test_layered_stall_watchdog_preempts_unschedulable_running_request():
    """The stall watchdog is the backstop for zero-token steps that regular
    preemption cannot resolve, e.g. a running request whose decode rows are
    deferred indefinitely with no waiting request to drive admission."""
    scheduler = create_scheduler(
        async_scheduling=True,
        additional_config=_LAYERED_ASYNC_ADDITIONAL_CONFIG,
        cudagraph_mode="NONE",
        enforce_eager=True,
    )
    (wedged,) = create_requests(
        num_requests=1, num_tokens=520, max_tokens=4, ignore_eos=True
    )
    scheduler.add_request(wedged)

    for _ in range(2):
        sched_output = scheduler.schedule()
        assert sched_output.total_num_scheduled_tokens > 0
        _run_async_step(scheduler, sched_output)

    wedged.next_decode_eligible_step = 10**9
    for step in range(_LAYERED_STALL_RECOVERY_STEPS):
        assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert step == _LAYERED_STALL_RECOVERY_STEPS - 1

    assert wedged.status == RequestStatus.PREEMPTED
    assert wedged not in scheduler.running
    assert wedged in scheduler.waiting
    assert scheduler._layered_stall_steps == 0

    sched_output = scheduler.schedule()
    assert sched_output.total_num_scheduled_tokens > 0
    assert wedged.status == RequestStatus.RUNNING
