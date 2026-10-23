# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts using the actual scheduler methods without accelerator imports."""

import __future__

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vllm.v1.core.layered_prefill import (
    LayeredPrefillPlan,
    reset_layered_prefill_request,
)


class RequestState(NS):
    __hash__ = object.__hash__


@pytest.fixture
def scheduler_class():
    root = Path(__file__).parents[3] / "vllm/v1/core/sched"
    wanted = {
        "_update_after_schedule",
        "_update_after_scheduled_decode",
        "_update_after_layered_schedule",
        "_update_after_boundary_schedule",
        "_update_after_layered_sample",
        "_attach_running_decodes_to_layered_output",
        "_is_layered_decode_request",
        "_can_queue_layered_group",
        "_layered_prefill_supported_for_scheduler",
        "_schedule_boundary_prefill",
        "_free_request_blocks",
        "_drain_deferred_frees",
    }
    namespace = dict(
        Request=RequestState,
        SchedulerOutput=NS,
        reset_layered_prefill_request=reset_layered_prefill_request,
    )
    for filename, classname in (
        ("scheduler.py", "Scheduler"),
        ("async_scheduler.py", "AsyncScheduler"),
    ):
        path = root / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname
        )
        if classname == "Scheduler":
            cls.bases = []
        cls.body = [
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted
        ]
        module = ast.Module(body=[cls], type_ignores=[])
        # Deferred annotations keep the extraction independent of device types.
        exec(
            compile(
                module, str(path), "exec", flags=__future__.annotations.compiler_flag
            ),
            namespace,
        )
    return namespace["AsyncScheduler"]


def make_scheduler(cls):
    scheduler = cls()
    scheduler.use_v2_model_runner = True
    scheduler.use_pp = True
    scheduler.parallel_config = NS(pipeline_parallel_size=2, data_parallel_size=1)
    scheduler.pp_size = 2
    scheduler.current_step = 10
    scheduler.num_spec_tokens = 0
    scheduler.num_sampled_tokens_per_step = 1
    scheduler._spec_token_placeholders = []
    scheduler.scheduler_config = NS(async_scheduling=True)
    scheduler.vllm_config = NS(kv_transfer_config=None)
    scheduler.layered_prefill_policy = NS(
        config=NS(require_eager=False, max_prefill_batch_size=2)
    )
    scheduler.connector = scheduler.ec_connector = None
    scheduler.cache_config = NS(enable_prefix_caching=False)
    scheduler.defer_block_free = True
    scheduler.sched_step_seq = 3
    scheduler.processed_step_seq = 0
    scheduler.deferred_frees = deque()
    scheduler._inflight_prefills = set()
    scheduler._boundary_event = Mock()
    scheduler.enable_return_routed_experts = False
    scheduler.kv_cache_manager = Mock()
    return scheduler


def request(rid, prompt=4, computed=0, outputs=0, placeholders=0):
    return RequestState(
        request_id=rid,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        num_in_flight_tokens=0,
        num_output_placeholders=placeholders,
        num_tokens=prompt + outputs,
        num_tokens_with_spec=prompt + outputs,
        output_token_ids=[1000] * outputs,
        spec_token_ids=[],
        max_tokens=32,
        is_prefill_chunk=not outputs,
        use_structured_output=False,
        next_decode_eligible_step=0,
        num_stale_output_tokens=0,
        layered_prefill_enabled=True,
        layered_prefill_kv_reserved=True,
        layered_prefill_group_id=1,
        layered_prefill_num_groups=4,
        last_sched_seq=0,
    )


def output(plan=None, counts=None):
    return NS(
        layered_prefill_plan=plan,
        num_scheduled_tokens=counts or {},
        total_num_scheduled_tokens=sum((counts or {}).values()),
        scheduled_spec_decode_tokens={},
        num_spec_tokens_to_schedule=0,
        pending_structured_output_tokens=False,
        has_structured_output_requests=False,
        scheduled_cached_reqs=NS(),
        kv_cache_block_copies=None,
    )


def boundary_plan(*, final, sampling=("short",)):
    return LayeredPrefillPlan(
        version=1,
        cohort_id=1,
        group_id=3 if final else 1,
        num_groups=4,
        group_start=6 if final else 2,
        group_end=8 if final else 4,
        prefill_req_ids=("short", "long"),
        query_tokens={"short": 4, "long": 4},
        commit_tokens={"short": 4 if final else 0, "long": 4 if final else 0},
        boundary_mode="isolated",
        group_stop=4 if final else 2,
        model_num_layers=8,
        is_final_chunk=False,
        sampling_req_ids=sampling if final else (),
    )


@pytest.mark.parametrize("entry", ["regular", "single", "batch"])
@pytest.mark.parametrize("final", [False, True])
def test_boundary_sample_is_accounted_once_for_every_entry(
    scheduler_class, entry, final
):
    scheduler = make_scheduler(scheduler_class)
    short, long = request("short"), request("long", prompt=8)
    scheduler.requests = {r.request_id: r for r in (short, long)}
    step = output(boundary_plan(final=final), {"short": 4, "long": 4})
    if entry == "regular":
        scheduler._update_after_schedule(step)
    else:
        method = (
            scheduler._update_after_layered_schedule
            if entry == "single"
            else scheduler._update_after_boundary_schedule
        )
        for req in (short, long):
            method(req, step)
    assert short.num_output_placeholders == int(final)
    assert long.num_output_placeholders == 0
    assert short.next_decode_eligible_step == (12 if final else 0)
    assert long.next_decode_eligible_step == 0
    assert short.spec_token_ids == long.spec_token_ids == []
    assert short.num_in_flight_tokens == long.num_in_flight_tokens == 4
    assert short.num_computed_tokens == long.num_computed_tokens == (4 if final else 0)
    assert short.last_sched_seq == long.last_sched_seq == 3


@pytest.mark.parametrize("spec_tokens", [0, 2])
def test_late_decode_gets_async_sample_and_cadence_bookkeeping(
    scheduler_class, spec_tokens
):
    scheduler = make_scheduler(scheduler_class)
    p = request("p")
    d = request("d", prompt=8, computed=9, outputs=1, placeholders=1)
    d.use_structured_output = True
    d.spec_token_ids = [-1] * spec_tokens
    d.num_tokens_with_spec += spec_tokens
    scheduler._spec_token_placeholders = [-1] * spec_tokens
    scheduler.running = [p, d]
    scheduler.max_num_scheduled_tokens = 16
    scheduler.max_model_len = 128
    scheduler.num_lookahead_tokens = spec_tokens
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._is_layered_prefill_pending = lambda req: req is p
    scheduler.kv_cache_manager.allocate_slots.return_value = NS()
    scheduler.kv_cache_manager.take_kv_cache_block_copies.return_value = ([], [])
    scheduler._make_cached_request_data = Mock(return_value=NS())
    scheduler._append_cached_request_data = Mock()
    step = output(counts={"p": 4})
    scheduler._attach_running_decodes_to_layered_output(step, p)
    assert step.num_scheduled_tokens == {"p": 4, "d": 1 + spec_tokens}
    assert d.num_output_placeholders == 2 + spec_tokens
    assert d.num_computed_tokens == 10 + spec_tokens
    assert d.num_in_flight_tokens == 1 + spec_tokens
    assert d.next_decode_eligible_step == 12
    assert d.last_sched_seq == 3
    assert step.has_structured_output_requests
    assert step.pending_structured_output_tokens
    assert len(step.scheduled_spec_decode_tokens.get("d", [])) == spec_tokens
    assert p.num_in_flight_tokens == p.num_output_placeholders == 0


def test_late_decode_does_not_repeat_last_sample(scheduler_class):
    scheduler = make_scheduler(scheduler_class)
    p = request("p")
    d = request("d", prompt=8, computed=9, outputs=1, placeholders=1)
    d.max_tokens = 2
    scheduler.running = [p, d]
    scheduler.max_num_scheduled_tokens = 16
    scheduler._is_layered_prefill_pending = lambda req: req is p
    step = output(counts={"p": 4})
    scheduler._attach_running_decodes_to_layered_output(step, p)
    assert step.num_scheduled_tokens == {"p": 4}
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()


def test_async_boundary_abort_keeps_blocks_until_last_interval_retires(scheduler_class):
    scheduler = make_scheduler(scheduler_class)
    short = request("short")
    step = output(boundary_plan(final=False), {"short": 4, "long": 4})
    scheduler._update_after_boundary_schedule(short, step)
    scheduler.sched_step_seq += 1
    scheduler._update_after_boundary_schedule(short, step)
    blocks = [object()]
    scheduler.kv_cache_manager.pop_blocks_for_free.return_value = blocks
    scheduler._free_request_blocks(short)
    assert list(scheduler.deferred_frees) == [(4, blocks)]
    scheduler.processed_step_seq = 3
    scheduler._drain_deferred_frees()
    scheduler.kv_cache_manager.block_pool.free_blocks.assert_not_called()
    scheduler.processed_step_seq = 4
    scheduler._drain_deferred_frees()
    free_blocks = scheduler.kv_cache_manager.block_pool.free_blocks
    free_blocks.assert_called_once()
    assert list(free_blocks.call_args.args[0]) == blocks
    assert not scheduler.deferred_frees


def test_async_pp_can_queue_only_same_chunk_continuations(scheduler_class):
    scheduler = make_scheduler(scheduler_class)
    req = request("p")
    assert scheduler._can_queue_layered_group(req)
    req.layered_prefill_group_id = 0
    assert not scheduler._can_queue_layered_group(req)
    req.layered_prefill_group_id = 1
    req.num_stale_output_tokens = 4
    assert not scheduler._can_queue_layered_group(req)


def test_async_pp_gate_requires_mrv2_and_boundary_still_rejects_spec(scheduler_class):
    scheduler = make_scheduler(scheduler_class)
    assert scheduler._layered_prefill_supported_for_scheduler()
    sentinel = object()
    scheduler._schedule_boundary_batch = lambda throttle: sentinel
    assert scheduler._schedule_boundary_prefill() is sentinel
    scheduler.num_spec_tokens = 1
    with pytest.raises(ValueError, match="no speculative decoding"):
        scheduler._schedule_boundary_prefill()
    scheduler.use_v2_model_runner = False
    assert not scheduler._layered_prefill_supported_for_scheduler()
