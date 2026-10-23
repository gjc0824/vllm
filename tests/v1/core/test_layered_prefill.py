# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

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

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler


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
            "scheduler_config": {"layered_prefill_config": {"enabled": enabled}},
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
    assert plan.is_sampling_step is False
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
    assert final_plan.is_final_chunk is True
    assert final_plan.is_sampling_step is True

    last_group_non_final_chunk = LayeredPrefillPlan(
        version=1,
        cohort_id=3,
        group_id=1,
        num_groups=2,
        group_start=4,
        group_end=8,
        prefill_req_ids=("req",),
        query_tokens={"req": 16},
        commit_tokens={"req": 16},
        is_final_chunk=False,
    )
    assert last_group_non_final_chunk.is_final_group is True
    assert last_group_non_final_chunk.is_sampling_step is False


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
    assert first.is_sampling_step is False
    assert first.reuse_kv_blocks is False
    assert first.cached_tokens["req"] == 0
    assert request.layered_prefill_cached_tokens == 0
    request.layered_prefill_group_id += 1
    request.layered_prefill_kv_reserved = True
    second = policy.make_plan(request)
    assert second.group_id == 1
    assert second.commit_tokens["req"] == 513
    assert second.reuse_kv_blocks is True
    assert second.is_sampling_step is True


def test_plan_carries_prefix_cached_tokens():
    policy = LayeredPrefillPolicy(_config(layers=8))
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=64,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    request.layered_prefill_cached_tokens = 16
    request.layered_prefill_query_tokens = 48
    plan = policy.make_plan(request)
    assert plan.query_tokens["req"] == 48
    assert plan.cached_tokens["req"] == 16


def test_reset_clears_cached_tokens():
    from vllm.v1.core.layered_prefill import reset_layered_prefill_request

    request = SimpleNamespace(
        layered_prefill_enabled=True,
        layered_prefill_cohort_id=3,
        layered_prefill_group_id=1,
        layered_prefill_num_groups=2,
        layered_prefill_query_tokens=48,
        layered_prefill_kv_reserved=True,
        layered_prefill_cached_tokens=16,
    )
    reset_layered_prefill_request(request)
    assert request.layered_prefill_enabled is False
    assert request.layered_prefill_cached_tokens == 0
    assert request.layered_prefill_query_tokens == 0


def test_eligibility_allows_prefix_hit_on_layered_request():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    sampling = SimpleNamespace(
        logprobs=None, prompt_logprobs=None, logprob_token_ids=None
    )
    waiting = SimpleNamespace(
        status=RequestStatus.WAITING,
        pooling_params=None,
        num_prompt_tokens=64,
        num_computed_tokens=0,
        output_token_ids=[],
        sampling_params=sampling,
        use_structured_output=False,
        layered_prefill_enabled=False,
    )
    assert Scheduler._is_layered_request_eligible(waiting)

    running_hit = SimpleNamespace(
        status=RequestStatus.RUNNING,
        pooling_params=None,
        num_prompt_tokens=64,
        num_computed_tokens=16,
        output_token_ids=[],
        sampling_params=sampling,
        use_structured_output=False,
        layered_prefill_enabled=True,
    )
    assert Scheduler._is_layered_request_eligible(running_hit)

    waiting_hit_not_layered = SimpleNamespace(
        status=RequestStatus.WAITING,
        pooling_params=None,
        num_prompt_tokens=64,
        num_computed_tokens=16,
        output_token_ids=[],
        sampling_params=sampling,
        use_structured_output=False,
        layered_prefill_enabled=False,
    )
    assert not Scheduler._is_layered_request_eligible(waiting_hit_not_layered)


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


def test_policy_caches_pp_ranges_per_instance(monkeypatch):
    import vllm.v1.core.layered_prefill as layered_prefill

    config = _config(layers=8)
    config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    original_make_ranges = layered_prefill.make_pp_aligned_layer_group_ranges
    calls = []

    def counted_make_ranges(*args):
        calls.append(args)
        return original_make_ranges(*args)

    monkeypatch.setattr(
        layered_prefill,
        "make_pp_aligned_layer_group_ranges",
        counted_make_ranges,
    )

    def make_request(req_id):
        return SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=1024,
            num_computed_tokens=0,
            layered_prefill_enabled=False,
            layered_prefill_group_id=0,
            layered_prefill_num_groups=0,
            layered_prefill_query_tokens=0,
            layered_prefill_cohort_id=-1,
            layered_prefill_kv_reserved=False,
            layered_prefill_cached_tokens=0,
        )

    policy = LayeredPrefillPolicy(config)
    first = make_request("first")
    policy.initialize_request(first)
    first_group = policy.make_plan(first)
    first.layered_prefill_group_id = 1
    second_group = policy.make_plan(first)
    policy.plan_chunk(first, 512)
    policy.make_plan(first)
    assert (first_group.group_start, first_group.group_end) == (0, 4)
    assert (second_group.group_start, second_group.group_end) == (4, 8)
    assert len(calls) == 1

    second_policy = LayeredPrefillPolicy(config)
    second = make_request("second")
    second_policy.initialize_request(second)
    second_policy.make_plan(second)
    assert len(calls) == 2


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


def test_plan_chunk_only_samples_on_last_chunk():
    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_query_tokens == 512
    first = policy.make_plan(request)
    assert first.is_final_chunk is False
    assert first.is_sampling_step is False

    request.layered_prefill_group_id = request.layered_prefill_num_groups - 1
    last_of_first_chunk = policy.make_plan(request)
    assert last_of_first_chunk.is_final_group is True
    assert last_of_first_chunk.is_sampling_step is False

    request.num_computed_tokens = 512
    policy.plan_chunk(request, 512)
    last_chunk = policy.make_plan(request)
    assert last_chunk.is_final_chunk is True
    request.layered_prefill_group_id = request.layered_prefill_num_groups - 1
    last = policy.make_plan(request)
    assert last.is_final_chunk is True
    assert last.is_sampling_step is True


def test_plan_chunk_respects_max_chunk_tokens():
    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=16392)
    config.additional_config["scheduler_config"]["layered_prefill_config"][
        "max_chunk_tokens"
    ] = 4096
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=16392,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    assert request.layered_prefill_query_tokens == 4096


def test_plan_chunk_caps_query_after_prefix_hit():
    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    policy.plan_chunk(request, 16)
    assert request.layered_prefill_cached_tokens == 16
    assert request.layered_prefill_query_tokens == 512
    plan = policy.make_plan(request)
    assert plan.cached_tokens["req"] == 16
    assert plan.is_final_chunk is False


def test_v2_prefix_hit_uses_plan_chunk_not_full_remainder():
    from vllm.v1.core.sched.scheduler import Scheduler

    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    sched = Scheduler.__new__(Scheduler)
    sched.use_v2_model_runner = True
    sched.layered_prefill_policy = policy
    sched._apply_layered_prefix_hit(request, 16)
    assert request.layered_prefill_query_tokens == 512
    assert request.layered_prefill_cached_tokens == 16


def test_v1_prefix_hit_keeps_full_remainder():
    from vllm.v1.core.sched.scheduler import Scheduler

    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    sched = Scheduler.__new__(Scheduler)
    sched.use_v2_model_runner = False
    sched.layered_prefill_policy = policy
    sched._apply_layered_prefix_hit(request, 16)
    assert request.layered_prefill_query_tokens == 1008
    assert request.layered_prefill_cached_tokens == 16


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
    assert config.fuse_mixed_batch is False


@pytest.mark.parametrize(
    "field",
    ["single_forward", "single_forward_decode_graph", "single_forward_compile"],
)
def test_config_parses_single_forward_controls(field: str):
    assert getattr(LayeredPrefillConfig(), field) is True
    config = LayeredPrefillConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                "scheduler_config": {
                    "layered_prefill_config": {field: False},
                }
            }
        )
    )
    assert getattr(config, field) is False


def test_config_rejects_multiple_groups_per_step_in_phase_one():
    with pytest.raises(ValueError, match="max_groups_per_step=1"):
        LayeredPrefillConfig(enabled=True, max_groups_per_step=2)


def test_decode_first_and_prefill_batching_are_opt_in():
    config = LayeredPrefillConfig()

    assert config.decode_first is False
    assert config.max_prefill_batch_size == 1


@pytest.mark.parametrize(
    "options, message",
    [
        ({"decode_first": True}, "decode_first requires pp_transition"),
        (
            {"boundary_policy": "pp_transition", "max_prefill_batch_size": 2},
            "multi-prefill admission requires decode_first",
        ),
    ],
)
def test_decode_first_config_rejects_unsupported_activation(options, message):
    with pytest.raises(ValueError, match=message):
        LayeredPrefillConfig(enabled=True, **options)


def test_merge_consecutive_groups_on_pp1_is_a_wider_layer_range():
    from vllm.v1.core.layered_prefill import merge_consecutive_layer_groups

    ranges = make_layer_group_ranges(8, 4)
    merged = merge_consecutive_layer_groups(ranges, 0, 2)
    assert (merged.start, merged.end) == (0, 4)
    assert merged.consumed == 2
    assert merged.includes_final is False

    tail = merge_consecutive_layer_groups(ranges, 2, 4)
    assert (tail.start, tail.end) == (4, 8)
    assert tail.consumed == 2
    assert tail.includes_final is True
    assert tail.group_id == 2


def test_merge_consecutive_groups_does_not_cross_pp_stage():
    from vllm.v1.core.layered_prefill import merge_consecutive_layer_groups

    ranges = make_pp_aligned_layer_group_ranges(8, 4, 2)
    # [(0,2), (2,4), (4,6), (6,8)] — stage boundary at 4.
    same_stage = merge_consecutive_layer_groups(
        ranges, 0, 4, pipeline_parallel_size=2, num_hidden_layers=8
    )
    assert (same_stage.start, same_stage.end) == (0, 4)
    assert same_stage.consumed == 2

    across = merge_consecutive_layer_groups(
        ranges, 1, 2, pipeline_parallel_size=2, num_hidden_layers=8
    )
    assert (across.start, across.end) == (2, 4)
    assert across.consumed == 1


def test_merge_single_group_skips_pp_owner_lookup(monkeypatch):
    import vllm.distributed.utils as distributed_utils
    from vllm.v1.core.layered_prefill import merge_consecutive_layer_groups

    def unexpected_owner_lookup(*_args, **_kwargs):
        raise AssertionError("single-group merge must not query PP ownership")

    monkeypatch.setattr(
        distributed_utils,
        "get_pp_indices",
        unexpected_owner_lookup,
    )
    ranges = make_layer_group_ranges(8, 4)
    merged = merge_consecutive_layer_groups(
        ranges,
        1,
        1,
        pipeline_parallel_size=2,
        num_hidden_layers=8,
    )

    assert (merged.start, merged.end) == (2, 4)
    assert merged.group_id == merged.last_group_id == 1
    assert merged.consumed == 1
    with pytest.raises(ValueError, match="max_groups must be >= 1"):
        merge_consecutive_layer_groups(ranges, 0, 0)
    with pytest.raises(ValueError, match="ranges must be non-empty"):
        merge_consecutive_layer_groups([], 0, 1)
    with pytest.raises(ValueError, match="group_id out of range"):
        merge_consecutive_layer_groups(ranges, len(ranges), 1)


def test_mixed_compaction_visits_p_only_in_active_groups():
    import numpy as np

    rng = np.random.default_rng(0)
    n_d, n_p, hidden, layers = 2, 5, 8, 6
    gs, ge = 2, 4
    weights = [
        rng.normal(size=(hidden, hidden)).astype(np.float64) for _ in range(layers)
    ]
    d_in = rng.normal(size=(n_d, hidden)).astype(np.float64)
    p_embed = rng.normal(size=(n_p, hidden)).astype(np.float64)

    def apply(h, start, end):
        for i in range(start, end):
            h = h @ weights[i]
        return h

    p_frontier = apply(p_embed.copy(), 0, gs)
    two_call_d = apply(d_in.copy(), 0, layers)
    two_call_p = apply(p_frontier.copy(), gs, ge)

    h = np.concatenate([d_in.copy(), p_frontier.copy()], axis=0)
    p_visits = np.zeros(layers, dtype=int)
    d_visits = np.zeros(layers, dtype=int)
    for i in range(layers):
        if gs <= i < ge:
            h = h @ weights[i]
            d_visits[i] += 1
            p_visits[i] += 1
        else:
            h[:n_d] = h[:n_d] @ weights[i]
            d_visits[i] += 1
    mixed_d, mixed_p = h[:n_d], h[n_d:]

    assert np.allclose(mixed_d, two_call_d)
    assert np.allclose(mixed_p, two_call_p)
    assert d_visits.tolist() == [1] * layers
    assert p_visits.tolist() == [1 if gs <= i < ge else 0 for i in range(layers)]

    naive = np.concatenate([d_in.copy(), p_frontier.copy()], axis=0)
    naive_p_visits = np.zeros(layers, dtype=int)
    for i in range(layers):
        naive = naive @ weights[i]
        naive_p_visits[i] += 1
    assert naive_p_visits.tolist() == [1] * layers
    assert not np.allclose(naive[n_d:], two_call_p)


def _enable_layered_scheduler(scheduler, **layered_cfg):
    from vllm.v1.core.layered_prefill import LayeredPrefillPolicy

    scheduler.vllm_config.model_config.enforce_eager = True
    cfg = {
        "enabled": True,
        "mode": "one_group",
        "require_eager": True,
        "require_pd_mixed": False,
        "allowed_num_groups": [2],
        "group_token_target": 1,
        "max_groups_per_step": 1,
    }
    cfg.update(layered_cfg)
    scheduler.vllm_config.additional_config = {
        "scheduler_config": {"layered_prefill_config": cfg}
    }
    scheduler.layered_prefill_policy = LayeredPrefillPolicy(scheduler.vllm_config)
    assert scheduler.layered_prefill_policy.enabled


def _local_scheduler(**kwargs):
    from contextlib import nullcontext
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from tests.v1.core.utils import create_scheduler
    from transformers import OPTConfig

    opts = dict(
        skip_tokenizer_init=True,
        max_num_seqs=2,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=32768,
    )
    layers = kwargs.pop("num_hidden_layers", 32)
    opts.update(kwargs)
    # Scheduler contracts need model metadata, never weights or a remote path.
    with TemporaryDirectory() as model_dir:
        OPTConfig(
            architectures=["OPTForCausalLM"],
            num_hidden_layers=layers,
            hidden_size=64,
            ffn_dim=128,
            num_attention_heads=2,
            vocab_size=50272,
            max_position_embeddings=32768,
        ).save_pretrained(model_dir)
        opts.setdefault("model", model_dir)
        use_v2 = opts.get("use_v2_model_runner")
        runner_env = (
            patch.dict("os.environ", {"VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2))})
            if use_v2 is not None
            else nullcontext()
        )
        # VllmConfig validates the runner before create_scheduler overrides it.
        with runner_env:
            return create_scheduler(**opts)


def _empty_runner_output(req_ids: list[str]):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _sampled_runner_output(req_id: str, token_id: int = 1000):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[token_id]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_in_flight_layer_group_does_not_steal_waiting_prompt():
    """In-flight group: hold waiters; do not start their regular prefill."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler()
    _enable_layered_scheduler(scheduler)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan is not None
    assert first.request_id in first_g0.num_scheduled_tokens
    assert second.request_id not in first_g0.num_scheduled_tokens

    # Do not update_from_output: the first group is still in flight, as in PP.
    held = scheduler.schedule()
    assert held.layered_prefill_plan is None
    assert second.request_id not in held.num_scheduled_tokens
    assert first.request_id not in held.num_scheduled_tokens
    assert second.status.name in ("WAITING", "PREEMPTED")


def test_async_in_flight_holds_waiting_prompt(monkeypatch: pytest.MonkeyPatch):
    from tests.v1.core.utils import create_requests

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    scheduler = _local_scheduler(
        async_scheduling=True,
        use_v2_model_runner=True,
    )
    _enable_layered_scheduler(scheduler)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan is not None
    assert second.request_id not in first_g0.num_scheduled_tokens

    continued = scheduler.schedule()
    assert continued.layered_prefill_plan.group_id == 1
    assert continued.layered_prefill_plan.prefill_req_ids == (first.request_id,)
    assert second.request_id not in continued.num_scheduled_tokens
    held = scheduler.schedule()
    assert held.layered_prefill_plan is None
    assert second.request_id not in held.num_scheduled_tokens


def test_pp2_admits_independent_prompt_without_advancing_inflight_frontier():
    """Legacy runner waits for group completion while staggering cohorts."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(max_num_seqs=3, use_v2_model_runner=False)
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.use_pp = True
    _enable_layered_scheduler(scheduler, allowed_num_groups=[4])
    first, second, third = create_requests(num_requests=3, num_tokens=8, max_tokens=8)
    for request in (first, second, third):
        scheduler.add_request(request)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan.prefill_req_ids == (first.request_id,)
    stage0_held = scheduler.schedule()
    assert not stage0_held.num_scheduled_tokens
    assert second.status.name == "WAITING"
    assert first.layered_prefill_group_id == 1
    assert first.num_computed_tokens == 0

    scheduler.update_from_output(first_g0, _empty_runner_output([first.request_id]))
    first_g1 = scheduler.schedule()
    stage0_held = scheduler.schedule()
    assert not stage0_held.num_scheduled_tokens
    scheduler.update_from_output(first_g1, _empty_runner_output([first.request_id]))
    first_g2 = scheduler.schedule()
    second_g0 = scheduler.schedule()
    assert first_g2.layered_prefill_plan.group_id == 2
    assert second_g0.layered_prefill_plan.prefill_req_ids == (second.request_id,)
    assert second_g0.layered_prefill_plan.group_id == 0

    full = scheduler.schedule()
    assert not full.num_scheduled_tokens
    assert third.status.name == "WAITING"

    scheduler.update_from_output(first_g2, _empty_runner_output([first.request_id]))
    first_final = scheduler.schedule()
    assert first_final.layered_prefill_plan.is_final_group
    assert first.num_computed_tokens == first.num_prompt_tokens
    assert second.layered_prefill_group_id == 1

    # A final group still consumes a slot until its sample comes back.
    full = scheduler.schedule()
    assert third.request_id not in full.num_scheduled_tokens
    scheduler.update_from_output(second_g0, _empty_runner_output([second.request_id]))
    second_g1 = scheduler.schedule()
    assert second_g1.layered_prefill_plan.prefill_req_ids == (second.request_id,)
    assert second_g1.layered_prefill_plan.group_id == 1
    scheduler.update_from_output(first_final, _sampled_runner_output(first.request_id))

    # A's slot is free, but B is still on stage 0. Drain B1 before starting C.
    drain = scheduler.schedule()
    assert drain.layered_prefill_plan is None
    assert second.request_id not in drain.num_scheduled_tokens
    assert third.request_id not in drain.num_scheduled_tokens
    assert second.layered_prefill_group_id == 2
    scheduler.update_from_output(second_g1, _empty_runner_output([second.request_id]))
    second_g2 = scheduler.schedule()
    third_g0 = scheduler.schedule()
    assert third_g0.layered_prefill_plan.prefill_req_ids == (third.request_id,)
    # Adjacent in-flight plans can now put P on rank 1 and then rank 0.
    assert second_g2.layered_prefill_plan.group_start == (
        second_g1.layered_prefill_plan.group_end
    )
    assert third_g0.layered_prefill_plan.group_start == 0


def test_pp2_waiting_prompt_does_not_fall_back_while_decode_is_pending():
    """No D identity yet: retain PB until PA's first sample is available."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler()
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.use_pp = True
    _enable_layered_scheduler(scheduler)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)
    first_g0 = scheduler.schedule()
    scheduler.update_from_output(first_g0, _empty_runner_output([first.request_id]))
    scheduler.schedule()  # Final group is in flight on stage 1.
    _enable_layered_scheduler(scheduler, require_pd_mixed=True)

    held = scheduler.schedule()
    assert not held.num_scheduled_tokens
    assert second.status.name == "WAITING"
    assert second.layered_prefill_enabled
    assert second.layered_prefill_group_id == 0


@pytest.mark.parametrize("prompt_tokens", [4, 8])
def test_pp2_final_group_admits_prompt_after_chunk_cursor_reset(prompt_tokens):
    """Both a next-chunk reset and a terminal final still run on stage 1."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler()
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.use_pp = True
    _enable_layered_scheduler(scheduler, allowed_num_groups=[4], max_chunk_tokens=4)
    first, second = create_requests(
        num_requests=2, num_tokens=prompt_tokens, max_tokens=8
    )
    scheduler.add_request(first)
    for group_id in range(3):
        step = scheduler.schedule()
        assert step.layered_prefill_plan.group_id == group_id
        scheduler.update_from_output(step, _empty_runner_output([first.request_id]))
    final = scheduler.schedule()
    assert final.layered_prefill_plan.is_final_group
    assert first.layered_prefill_group_id == 0
    assert first.num_in_flight_tokens == 4
    assert first.num_computed_tokens == 4
    assert first.layered_prefill_enabled == (prompt_tokens > 4)

    scheduler.add_request(second)
    next_prompt = scheduler.schedule()
    assert next_prompt.layered_prefill_plan.prefill_req_ids == (second.request_id,)
    assert next_prompt.layered_prefill_plan.group_id == 0
    assert first.request_id not in next_prompt.num_scheduled_tokens


def test_first_sample_mixes_decode_with_next_layered_prompt():
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler()
    _enable_layered_scheduler(scheduler)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan is not None
    scheduler.update_from_output(first_g0, _empty_runner_output([first.request_id]))

    first_g1 = scheduler.schedule()
    assert first_g1.layered_prefill_plan is not None
    assert first_g1.layered_prefill_plan.is_final_group
    assert second.request_id not in first_g1.num_scheduled_tokens

    # Sample has not landed yet: hold the waiter instead of starting a P-only step.
    held = scheduler.schedule()
    assert second.request_id not in held.num_scheduled_tokens

    scheduler.update_from_output(
        first_g1, _sampled_runner_output(first.request_id, token_id=1000)
    )

    mixed = scheduler.schedule()
    assert mixed.layered_prefill_plan is not None
    assert mixed.layered_prefill_plan.group_id == 0
    assert second.request_id in mixed.num_scheduled_tokens
    assert first.request_id in mixed.num_scheduled_tokens

    from vllm.v1.outputs import ModelRunnerOutput

    mixed_ids = list(mixed.num_scheduled_tokens)
    scheduler.update_from_output(
        mixed,
        ModelRunnerOutput(
            req_ids=mixed_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(mixed_ids)},
            sampled_token_ids=[
                [1001] if req_id == first.request_id else [] for req_id in mixed_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    final_mix = scheduler.schedule()
    assert final_mix.layered_prefill_plan is not None
    assert final_mix.layered_prefill_plan.is_final_group
    assert second.request_id in final_mix.num_scheduled_tokens
    assert first.request_id in final_mix.num_scheduled_tokens


def _drive_layered_until_sample(scheduler, token_for):
    """Run layer groups until the sampling step, then apply sampled tokens."""
    from vllm.v1.outputs import ModelRunnerOutput

    step = scheduler.schedule()
    while (
        step.layered_prefill_plan is not None
        and not step.layered_prefill_plan.is_sampling_step
    ):
        scheduler.update_from_output(
            step, _empty_runner_output(list(step.num_scheduled_tokens))
        )
        step = scheduler.schedule()
    assert step.layered_prefill_plan is not None
    assert step.layered_prefill_plan.is_sampling_step
    req_ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[token_for(req_id)] for req_id in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return step


def _pp2_v2_layered_scheduler(groups=2):
    scheduler = _local_scheduler(
        max_num_batched_tokens=4096,
        max_model_len=16384,
        use_v2_model_runner=True,
    )
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.parallel_config.tensor_parallel_size = 2
    scheduler.use_pp = True
    _enable_layered_scheduler(
        scheduler,
        allowed_num_groups=[groups],
        max_chunk_tokens=4096,
    )
    return scheduler


def _add_ready_decode(scheduler):
    from tests.v1.core.utils import create_requests

    (request,) = create_requests(
        num_requests=1,
        num_tokens=8,
        max_tokens=256,
        req_ids=["decode"],
    )
    scheduler.add_request(request)
    _drive_layered_until_sample(scheduler, lambda _req_id: 1000)
    return request


def _boundary_scheduler(groups=8):
    scheduler = _pp2_v2_layered_scheduler(groups)
    decode = _add_ready_decode(scheduler)
    _enable_layered_scheduler(
        scheduler, allowed_num_groups=[groups], boundary_policy="pp_transition"
    )
    # The helper swaps the policy after initialization; production sets this
    # fence when constructing the scheduler with boundary_policy enabled.
    scheduler.defer_block_free = True
    return scheduler, decode


@pytest.mark.parametrize("groups", [4, 8, 16])
def test_boundary_tail_crosses_stages_in_one_step_and_commits_once(groups):
    from tests.v1.core.utils import create_requests

    scheduler, _ = _boundary_scheduler(groups)
    (prefill,) = create_requests(
        num_requests=1, num_tokens=8, req_ids=["boundary"], max_tokens=4
    )
    scheduler.add_request(prefill)
    plans = []
    while True:
        output = scheduler.schedule()
        plan = output.layered_prefill_plan
        assert plan is not None
        plans.append(plan)
        assert plan.query_tokens[prefill.request_id] == 8
        assert prefill.num_computed_tokens == (8 if plan.is_final_group else 0)
        runner_output = _empty_runner_output(list(output.num_scheduled_tokens))
        if plan.is_final_group:
            runner_output.sampled_token_ids = [[1000] for _ in runner_output.req_ids]
        scheduler.update_from_output(output, runner_output)
        if plan.is_final_group:
            break
    assert (plans[0].group_start, plans[0].group_end) == (0, 16 + 32 // groups)
    assert plans[0].resume_group_id == groups // 2 + 1
    assert [
        layer for p in plans for layer in range(p.group_start, p.group_end)
    ] == list(range(32))
    assert sum(p.commit_tokens[prefill.request_id] for p in plans) == 8
    assert not prefill.layered_prefill_enabled
    assert scheduler.processed_step_seq == scheduler.sched_step_seq


def test_boundary_head_exits_across_stages_with_successor():
    from tests.v1.core.utils import create_requests

    scheduler, decode = _boundary_scheduler()
    pure_decode = scheduler.schedule()
    assert pure_decode.layered_prefill_plan is None
    scheduler.update_from_output(pure_decode, _sampled_runner_output(decode.request_id))
    requests = create_requests(
        num_requests=2, num_tokens=8, req_ids=["head", "successor"], max_tokens=4
    )
    for request in requests:
        scheduler.add_request(request)
    intervals = []
    while True:
        output = scheduler.schedule()
        plan = output.layered_prefill_plan
        assert plan is not None and plan.boundary_mode == "head"
        assert "successor" not in output.num_scheduled_tokens
        intervals.append((plan.group_start, plan.group_end))
        scheduler.update_from_output(
            output, _empty_runner_output(list(output.num_scheduled_tokens))
        )
        if plan.is_final_group:
            break
    assert intervals == [(0, 4), (4, 8), (8, 12), (12, 32)]


@pytest.mark.parametrize("groups", [4, 8, 16])
@pytest.mark.parametrize("arrive_before_stage_end", [False, True])
def test_boundary_isolated_rechecks_successor_at_stage_end_at_depth_two(
    groups, arrive_before_stage_end
):
    """A newly visible P can shorten isolated traversal before PP0 exits."""
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler, _ = _boundary_scheduler(groups)
    pure_decode = scheduler.schedule()
    scheduler.update_from_output(pure_decode, _wavefront_output(pure_decode))
    active, successor = create_requests(
        num_requests=2, num_tokens=8, req_ids=["isolated", "late_p"], max_tokens=4
    )
    scheduler.add_request(active)
    queue = deque()
    plans = []
    added_successor = False
    while True:
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        assert plan is not None and plan.boundary_mode == "isolated"
        assert successor.request_id not in step.num_scheduled_tokens
        assert plan.query_tokens == {active.request_id: 8}
        plans.append(plan)
        queue.append(step)
        if len(queue) == 2:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
        if not added_successor and (arrive_before_stage_end or plan.group_end == 16):
            scheduler.max_num_running_reqs = 3
            scheduler.add_request(successor)
            added_successor = True
        if plan.is_final_group:
            break
    assert added_successor
    assert len(plans) == (groups // 2 if arrive_before_stage_end else groups)
    assert [
        layer for plan in plans for layer in range(plan.group_start, plan.group_end)
    ] == list(range(32))
    assert sum(plan.commit_tokens[active.request_id] for plan in plans) == 8
    if arrive_before_stage_end:
        assert plans[-1].group_start < 16 < plans[-1].group_end
    else:
        assert all(plan.group_end <= 16 or plan.group_start >= 16 for plan in plans)

    final_seq = scheduler._boundary_final_step_seq
    assert scheduler.processed_step_seq == final_seq - 1
    assert active.num_in_flight_tokens == 8
    native = scheduler.schedule()
    queue.append(native)
    assert len(queue) == 2
    assert native.layered_prefill_plan is None
    assert native.num_scheduled_tokens[successor.request_id] == 8
    assert active.request_id not in native.num_scheduled_tokens
    assert scheduler._boundary_request_id == active.request_id
    assert scheduler._boundary_mode == "isolated"
    assert scheduler._boundary_final_step_seq == final_seq
    assert scheduler.processed_step_seq < final_seq
    while queue:
        completed = queue.popleft()
        scheduler.update_from_output(completed, _wavefront_output(completed))
    assert active.num_computed_tokens == successor.num_computed_tokens == 8
    assert active.num_in_flight_tokens == successor.num_in_flight_tokens == 0
    assert active.num_output_tokens == successor.num_output_tokens == 1
    assert scheduler.processed_step_seq == scheduler.sched_step_seq


def test_boundary_steady_chunk_uses_regular_admission():
    from tests.v1.core.utils import create_requests

    scheduler, _ = _boundary_scheduler()
    (prefill,) = create_requests(
        num_requests=1, num_tokens=8192, req_ids=["steady"], max_tokens=4
    )
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.layered_prefill_plan is None
    assert prefill.num_computed_tokens == output.num_scheduled_tokens["steady"]
    assert not prefill.layered_prefill_enabled
    assert scheduler.boundary_stats["regular_prefill"] == 1


def test_boundary_can_start_after_retired_native_token_chunk():
    from tests.v1.core.utils import create_requests

    scheduler, _ = _boundary_scheduler()
    (prefill,) = create_requests(
        num_requests=1, num_tokens=5000, req_ids=["chunked"], max_tokens=4
    )
    scheduler.add_request(prefill)
    native = scheduler.schedule()
    assert native.layered_prefill_plan is None
    origin = prefill.num_computed_tokens
    scheduler.update_from_output(
        native, _empty_runner_output(list(native.num_scheduled_tokens))
    )
    boundary = scheduler.schedule()
    plan = boundary.layered_prefill_plan
    assert plan is not None and plan.boundary_mode == "tail"
    assert plan.cached_tokens == {"chunked": origin}
    assert plan.query_tokens == {"chunked": 5000 - origin}
    assert prefill.num_computed_tokens == origin


def test_boundary_queue_depth_two_keeps_inflight_native_tail_native():
    """Submit both native chunks before retiring the oldest, like EngineCore."""
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler, decode = _boundary_scheduler()
    (prefill,) = create_requests(
        num_requests=1, num_tokens=5000, req_ids=["queued_native"], max_tokens=4
    )
    scheduler.add_request(prefill)
    queue = deque(maxlen=2)
    first = scheduler.schedule()
    queue.append(first)
    origin = prefill.num_computed_tokens
    assert 0 < origin < prefill.num_prompt_tokens
    assert prefill.num_in_flight_tokens == origin
    first_data = next(
        req for req in first.scheduled_new_reqs if req.req_id == prefill.request_id
    )
    # The ring carries every batch row, including nonsampling placeholders.
    # Only the per-request cursor predicate determines whether P can sample.
    assert (
        first_data.num_computed_tokens + first.num_scheduled_tokens[prefill.request_id]
        < prefill.num_prompt_tokens
    )

    # Do not consume first's result before scheduling the tail: this is the
    # queue-depth-two case that deliberately misses the boundary optimization.
    tail = scheduler.schedule()
    queue.append(tail)
    assert len(queue) == 2
    assert first.layered_prefill_plan is None
    assert tail.layered_prefill_plan is None
    assert tail.num_scheduled_tokens[prefill.request_id] == 5000 - origin
    cached = tail.scheduled_cached_reqs
    assert (
        cached.num_computed_tokens[cached.req_ids.index(prefill.request_id)] == origin
    )
    assert origin + tail.num_scheduled_tokens[prefill.request_id] == 5000
    assert scheduler.boundary_stats["fallback_request_or_inflight"] == 1
    assert prefill.num_computed_tokens == prefill.num_in_flight_tokens == 5000
    assert scheduler._boundary_request_id is None

    completed = queue.popleft()
    result = _empty_runner_output(list(completed.num_scheduled_tokens))
    result.sampled_token_ids = [
        [1001] if rid == decode.request_id else [] for rid in result.req_ids
    ]
    scheduler.update_from_output(completed, result)
    assert prefill.num_computed_tokens == 5000
    assert prefill.num_in_flight_tokens == 5000 - origin
    assert prefill.num_output_tokens == 0
    assert not prefill.layered_prefill_enabled

    completed = queue.popleft()
    result = _empty_runner_output(list(completed.num_scheduled_tokens))
    result.sampled_token_ids = [[1002] for _ in result.req_ids]
    scheduler.update_from_output(completed, result)
    assert prefill.num_computed_tokens == 5000
    assert prefill.num_in_flight_tokens == 0
    assert prefill.num_output_tokens == 1
    assert scheduler.processed_step_seq == scheduler.sched_step_seq


def test_boundary_no_decode_and_empty_steps_do_not_create_head_history():
    from tests.v1.core.utils import create_requests

    scheduler = _pp2_v2_layered_scheduler()
    _enable_layered_scheduler(scheduler, boundary_policy="pp_transition")
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert not scheduler._boundary_decode_only
    (prefill,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(prefill)
    assert scheduler.schedule().layered_prefill_plan is None
    assert scheduler.boundary_stats["fallback_no_decode"] == 1


def test_boundary_prefix_cache_fails_explicitly():
    scheduler, _ = _boundary_scheduler()
    scheduler.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="prefix caching disabled"):
        scheduler.schedule()


def test_boundary_final_inflight_chunk_allows_independent_native_prefill():
    from tests.v1.core.utils import create_requests

    scheduler, _ = _boundary_scheduler(4)
    first, second = create_requests(
        num_requests=2, num_tokens=8, req_ids=["first", "second"], max_tokens=4
    )
    scheduler.add_request(first)
    initial = scheduler.schedule()
    assert initial.layered_prefill_plan is not None
    final = scheduler.schedule()
    assert final.layered_prefill_plan.is_final_group
    scheduler.max_num_running_reqs = 3
    scheduler.add_request(second)
    native = scheduler.schedule()
    assert native.num_scheduled_tokens["second"] == 8
    assert native.layered_prefill_plan is None
    assert "first" not in native.num_scheduled_tokens
    assert scheduler._boundary_request_id == "first"
    assert scheduler._boundary_final_step_seq > scheduler.processed_step_seq
    assert not first.layered_prefill_enabled
    assert first.num_in_flight_tokens == 16


@pytest.mark.parametrize("abort_after_final", [False, True])
@pytest.mark.parametrize("trace_enabled", [False, True])
def test_boundary_head_exit_queues_independent_native_at_depth_two(
    abort_after_final, trace_enabled, monkeypatch
):
    """Fill PP0 after HEAD's final interval while PP1 still owns that chunk."""
    from collections import deque
    from dataclasses import replace

    from tests.v1.core.utils import create_requests

    from vllm.v1.request import RequestStatus

    scheduler, decode = _boundary_scheduler()
    pure_decode = scheduler.schedule()
    scheduler.update_from_output(pure_decode, _sampled_runner_output(decode.request_id))
    (active,) = create_requests(
        num_requests=1, num_tokens=5000, req_ids=["head_active"], max_tokens=4
    )
    scheduler.add_request(active)
    queue = deque()
    intervals = []
    for _ in range(4):
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        assert plan is not None and plan.boundary_mode == "head"
        intervals.append((plan.group_start, plan.group_end))
        queue.append(step)
        if len(queue) == 2:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
    assert intervals == [(0, 4), (4, 8), (8, 12), (12, 32)]
    assert len(queue) == 1
    final = queue[0]
    plan = final.layered_prefill_plan
    assert plan.is_final_group and not plan.is_final_chunk
    chunk_tokens = plan.query_tokens[active.request_id]
    assert active.num_computed_tokens == chunk_tokens < active.num_prompt_tokens
    assert active.num_in_flight_tokens == chunk_tokens
    final_seq = scheduler._boundary_final_step_seq
    assert final_seq == scheduler.sched_step_seq
    assert scheduler.processed_step_seq == final_seq - 1

    if abort_after_final:
        scheduler.finish_requests(active.request_id, RequestStatus.FINISHED_ABORTED)
        assert scheduler.deferred_frees
    scheduler.max_num_running_reqs = 3
    (independent,) = create_requests(
        num_requests=1, num_tokens=8, req_ids=["native_next"], max_tokens=4
    )
    scheduler.add_request(independent)
    scheduler.layered_prefill_policy.config = replace(
        scheduler.layered_prefill_policy.config, boundary_trace=trace_enabled
    )
    events = []
    boundary_event = scheduler._boundary_event

    def record_event(event, **details):
        events.append((event, details))
        boundary_event(event, **details)

    monkeypatch.setattr(scheduler, "_boundary_event", record_event)
    native = scheduler.schedule()
    queue.append(native)
    assert len(queue) == 2
    assert native.layered_prefill_plan is None
    assert native.num_scheduled_tokens[independent.request_id] == 8
    assert active.request_id not in native.num_scheduled_tokens
    assert scheduler._boundary_request_id == active.request_id
    assert scheduler._boundary_mode == "head"
    assert scheduler._boundary_final_step_seq == final_seq
    assert not scheduler._boundary_hold_req_ids
    assert scheduler.processed_step_seq == final_seq - 1
    assert scheduler.sched_step_seq == final_seq + 1
    assert events == (
        [
            (
                "final_draining_admit",
                {
                    "active_req": active.request_id,
                    "final_seq": final_seq,
                    "processed_seq": final_seq - 1,
                    "step_seq": final_seq + 1,
                    "prefills": {independent.request_id: 8},
                },
            )
        ]
        if trace_enabled
        else []
    )
    if not abort_after_final:
        assert active.num_computed_tokens == chunk_tokens
        assert active.num_in_flight_tokens == chunk_tokens

    completed = queue.popleft()
    scheduler.update_from_output(completed, _wavefront_output(completed))
    assert scheduler.processed_step_seq == final_seq
    assert not scheduler.deferred_frees
    if not abort_after_final:
        assert active.num_in_flight_tokens == 0
        assert active.num_computed_tokens == chunk_tokens
        assert active.num_output_tokens == 0
    completed = queue.popleft()
    scheduler.update_from_output(completed, _wavefront_output(completed))
    assert independent.num_output_tokens == 1
    assert independent.num_in_flight_tokens == 0
    assert scheduler.processed_step_seq == scheduler.sched_step_seq


def test_boundary_multi_prefill_keeps_native_batch():
    from tests.v1.core.utils import create_requests

    scheduler, _ = _boundary_scheduler()
    scheduler.max_num_running_reqs = 4
    requests = create_requests(
        num_requests=2, num_tokens=8, req_ids=["p0", "p1"], max_tokens=4
    )
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert scheduler.layered_prefill_policy.config.decode_first is False
    assert scheduler.layered_prefill_policy.config.max_prefill_batch_size == 1
    assert output.layered_prefill_plan is None
    assert all(output.num_scheduled_tokens[r.request_id] == 8 for r in requests)
    assert all(r.num_computed_tokens == 8 for r in requests)
    assert scheduler.boundary_stats["fallback_multi_p"] == 1


def test_boundary_budget_fallback_does_not_shrink_native_query():
    from tests.v1.core.utils import create_requests

    scheduler, _ = _boundary_scheduler()
    scheduler._layered_decode_reserve_tokens = 64
    (prefill,) = create_requests(
        num_requests=1, num_tokens=4095, req_ids=["wide"], max_tokens=4
    )
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.layered_prefill_plan is None
    assert output.num_scheduled_tokens["wide"] == 4095
    assert prefill.num_computed_tokens == 4095
    assert scheduler.boundary_stats["fallback_budget"] == 1


@pytest.mark.parametrize("groups", [4, 8, 16])
def test_boundary_head_reserves_inflight_decode_budget_at_depth_two(
    groups, monkeypatch
):
    """Shrink saturated HEAD once so D retiring after admission still fits."""
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler, first_decode = _boundary_scheduler(groups)
    scheduler.max_num_running_reqs = 3
    (second_decode,) = create_requests(
        num_requests=1, num_tokens=8, req_ids=["second_decode"], max_tokens=256
    )
    scheduler.add_request(second_decode)
    while not second_decode.output_token_ids:
        step = scheduler.schedule()
        scheduler.update_from_output(step, _wavefront_output(step))
    decodes = (first_decode, second_decode)
    assert all(not request.num_in_flight_tokens for request in decodes)

    # A real D-only submission remains queued; the other D is still ready.
    scheduler.max_num_scheduled_tokens = 1
    queued_decode = scheduler.schedule()
    assert queued_decode.total_num_scheduled_tokens == 1
    assert queued_decode.layered_prefill_plan is None
    scheduler.max_num_scheduled_tokens = 4096
    (prefill,) = create_requests(
        num_requests=1, num_tokens=8192, req_ids=["saturated_head"], max_tokens=4
    )
    scheduler.add_request(prefill)
    admitted = []
    finalize = scheduler._finalize_boundary_schedule

    def record_admission(output):
        if prefill.request_id in output.num_scheduled_tokens:
            admitted.append(output.num_scheduled_tokens[prefill.request_id])
        finalize(output)

    monkeypatch.setattr(scheduler, "_finalize_boundary_schedule", record_admission)
    queue = deque([queued_decode])
    plans = []
    while True:
        ready_decodes = {
            request.request_id
            for request in decodes
            if not request.num_in_flight_tokens
        }
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        assert plan is not None and plan.boundary_mode == "head"
        plans.append(plan)
        assert plan.query_tokens == {prefill.request_id: 4094}
        assert ready_decodes <= step.num_scheduled_tokens.keys()
        assert step.total_num_scheduled_tokens == sum(
            step.num_scheduled_tokens.values()
        )
        # Even if both D retire, the frozen P plus all D and TP padding fits.
        assert (4094 + len(decodes) + 1) // 2 * 2 <= 4096
        assert prefill.num_computed_tokens == (4094 if plan.is_final_group else 0)
        queue.append(step)
        assert len(queue) == 2
        completed = queue.popleft()
        scheduler.update_from_output(completed, _wavefront_output(completed))
        if plan.is_final_group:
            break
    assert admitted == [4095]
    assert scheduler.boundary_stats["query_shrunk"] == 1
    assert scheduler.boundary_stats["fallback_budget"] == 0
    assert len(plans) == groups // 2
    assert sum(plan.commit_tokens[prefill.request_id] for plan in plans) == 4094
    assert prefill.num_in_flight_tokens == 4094
    assert not plans[-1].is_final_chunk
    completed = queue.popleft()
    scheduler.update_from_output(completed, _wavefront_output(completed))
    assert prefill.num_in_flight_tokens == 0
    assert prefill.num_computed_tokens == 4094
    assert prefill.num_output_tokens == 0
    assert scheduler.processed_step_seq == scheduler.sched_step_seq


def test_boundary_abort_defers_block_reuse_until_interval_retires():
    from tests.v1.core.utils import create_requests

    from vllm.v1.request import RequestStatus

    scheduler, _ = _boundary_scheduler()
    (prefill,) = create_requests(
        num_requests=1, num_tokens=8, req_ids=["aborted"], max_tokens=4
    )
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.layered_prefill_plan is not None
    scheduler.finish_requests("aborted", RequestStatus.FINISHED_ABORTED)
    assert "aborted" not in scheduler.requests
    assert scheduler.deferred_frees
    assert scheduler.processed_step_seq < scheduler.sched_step_seq
    scheduler.update_from_output(
        output, _empty_runner_output(list(output.num_scheduled_tokens))
    )
    assert not scheduler.deferred_frees
    assert scheduler.processed_step_seq == scheduler.sched_step_seq


@pytest.mark.parametrize("groups", [4, 8, 16])
@pytest.mark.parametrize("arrive_before_stage_end", [False, True])
def test_boundary_isolated_policy_exits_only_before_pp0_completion(
    groups, arrive_before_stage_end
):
    config = _config(layers=43)
    config.parallel_config.pipeline_parallel_size = 2
    config.additional_config["scheduler_config"]["layered_prefill_config"].update(
        allowed_num_groups=[groups], boundary_policy="pp_transition"
    )
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="p",
        num_prompt_tokens=16,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
    )
    policy.initialize_request(request)
    plans = []
    arrival_group = 1 if arrive_before_stage_end else groups // 2
    while request.layered_prefill_group_id < groups:
        plan = policy.make_boundary_plan(
            request,
            "isolated",
            has_successor=request.layered_prefill_group_id >= arrival_group,
        )
        plans.append(plan)
        assert plan.boundary_mode == "isolated"
        request.layered_prefill_group_id = plan.resume_group_id
    assert len(plans) == (groups // 2 if arrive_before_stage_end else groups)
    assert [
        layer for plan in plans for layer in range(plan.group_start, plan.group_end)
    ] == list(range(43))
    assert sum(plan.commit_tokens["p"] for plan in plans) == 16
    assert plans[-1].is_final_group and plans[-1].resume_group_id == groups
    if arrive_before_stage_end:
        assert plans[-1].group_start < 22 < plans[-1].group_end
    else:
        assert all(plan.group_end <= 22 or plan.group_start >= 22 for plan in plans)


@pytest.mark.parametrize("groups", [4, 8, 16])
@pytest.mark.parametrize("partition", ["16,16", "10,22", "23,9"])
def test_boundary_interval_protocol_covers_custom_partition_once(
    groups, partition, monkeypatch
):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    config = _config(layers=32)
    config.parallel_config.pipeline_parallel_size = 2
    config.additional_config["scheduler_config"]["layered_prefill_config"].update(
        allowed_num_groups=[groups], boundary_policy="pp_transition"
    )
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="p",
        num_prompt_tokens=16,
        num_computed_tokens=4,
        layered_prefill_enabled=False,
    )
    policy.initialize_request(request)
    covered = []
    while request.layered_prefill_group_id < request.layered_prefill_num_groups:
        plan = policy.make_boundary_plan(request, "tail")
        covered.extend(range(plan.group_start, plan.group_end))
        request.layered_prefill_group_id = plan.resume_group_id
    assert covered == list(range(32))
    assert plan.is_sampling_step
    assert plan.cached_tokens == {"p": 4}


def test_pp2_v2_fresh_chunk_reserves_tp_aligned_decode_budget():
    """A full-MBT P chunk leaves room for D and keeps one width per group."""
    from tests.v1.core.utils import create_requests

    scheduler = _pp2_v2_layered_scheduler()
    decode = _add_ready_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=16380,
        max_tokens=256,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill)

    first_group = scheduler.schedule()
    first_plan = first_group.layered_prefill_plan
    assert first_plan is not None
    assert first_plan.group_id == 0
    assert first_plan.query_tokens[prefill.request_id] == 4094
    assert first_group.num_scheduled_tokens[decode.request_id] == 1
    assert first_group.num_scheduled_tokens[prefill.request_id] == 4094

    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(first_group.num_scheduled_tokens)
    scheduler.update_from_output(
        first_group,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [1001] if req_id == decode.request_id else [] for req_id in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    final_group = scheduler.schedule()
    final_plan = final_group.layered_prefill_plan
    assert final_plan is not None
    assert final_plan.is_final_group
    assert final_plan.query_tokens[prefill.request_id] == 4094
    assert final_plan.commit_tokens[prefill.request_id] == 4094

    # The next chunk is planned while this final group is still in flight.
    # Its full width must remain untouched until that chunk actually starts.
    assert prefill.layered_prefill_group_id == 0
    assert prefill.num_in_flight_tokens == 4094
    assert prefill.layered_prefill_query_tokens == 4096
    assert scheduler._reserve_layered_decode_budget(prefill) == 4096


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
def test_pp2_v2_fresh_chunk_without_decode_keeps_full_width(groups):
    from tests.v1.core.utils import create_requests

    scheduler = _pp2_v2_layered_scheduler(groups)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=16380,
        max_tokens=256,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill)

    output = scheduler.schedule()
    plan = output.layered_prefill_plan
    assert plan is not None
    assert plan.query_tokens[prefill.request_id] == 4096
    assert output.num_scheduled_tokens[prefill.request_id] == 4096


@pytest.mark.parametrize(
    "eligible_offset, expected_query, decode_scheduled",
    [(0, 4094, True), (1, 4094, False)],
)
def test_pp2_v2_decode_reservation_uses_prospective_step_eligibility(
    eligible_offset, expected_query, decode_scheduled
):
    from tests.v1.core.utils import create_requests

    scheduler = _pp2_v2_layered_scheduler()
    decode = _add_ready_decode(scheduler)
    prospective_step = scheduler.current_step + 1
    decode.next_decode_eligible_step = prospective_step + eligible_offset
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=16380,
        max_tokens=256,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill)

    output = scheduler.schedule()
    plan = output.layered_prefill_plan
    assert plan is not None
    assert plan.query_tokens[prefill.request_id] == expected_query
    assert (decode.request_id in output.num_scheduled_tokens) is decode_scheduled


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
    from vllm.v1.outputs import ModelRunnerOutput

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
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(
        async_scheduling=True,
        use_v2_model_runner=False,
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
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(
        async_scheduling=True,
        use_v2_model_runner=False,
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
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(
        async_scheduling=True,
        use_v2_model_runner=False,
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
    from tests.v1.core.utils import create_requests

    from vllm.v1.request import RequestStatus

    scheduler = _local_scheduler(
        async_scheduling=True,
        use_v2_model_runner=False,
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
    from tests.v1.core.utils import create_requests

    from vllm.v1.core.sched.scheduler import _LAYERED_STALL_RECOVERY_STEPS
    from vllm.v1.request import RequestStatus

    scheduler = _local_scheduler(
        async_scheduling=True,
        use_v2_model_runner=False,
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


def _wavefront_scheduler(groups=2, layers=32, *, async_scheduling=False, **kwargs):
    scheduler = _local_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=16,
        use_v2_model_runner=True,
        async_scheduling=async_scheduling,
        num_hidden_layers=layers,
        **kwargs,
    )
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.pp_size = 2
    scheduler.use_pp = True
    _enable_layered_scheduler(
        scheduler,
        allowed_num_groups=[groups],
        max_chunk_tokens=4,
    )
    return scheduler


@pytest.mark.cpu_test
@pytest.mark.parametrize("groups", [2, 4, 16])
@pytest.mark.parametrize("with_decode", [False, True])
def test_async_pp2_wave_overlaps_cohorts_at_engine_queue_depth(groups, with_decode):
    """PP+1 queued RPCs retain wave overlap and drain before the next chunk."""
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler(groups, layers=43, async_scheduling=True)
    if with_decode:
        _add_ready_decode(scheduler)
    first, second = create_requests(
        num_requests=2,
        num_tokens=8,
        max_tokens=1,
        req_ids=["async_wave_a", "async_wave_b"],
    )
    for req in (first, second):
        scheduler.add_request(req)
    queue = deque()
    sequence = []
    frontiers = {}
    first_stage1 = None
    overlapping_admission = False
    for _ in range(8 * groups + 32):
        before_flight = {r.request_id: r.num_in_flight_tokens for r in (first, second)}
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        if plan is not None:
            rid = plan.prefill_req_ids[0]
            chunk = plan.cached_tokens[rid]
            key = (rid, chunk)
            assert plan.group_id == frontiers.get(key, 0)
            frontiers[key] = plan.group_id + 1
            if plan.group_id == 0:
                assert before_flight[rid] == 0
            sequence.append((rid, plan.group_id, chunk))
            if (
                rid == first.request_id
                and chunk == 0
                and plan.group_start >= 22
                and first_stage1 is None
            ):
                first_stage1 = len(sequence) - 1
            if rid == second.request_id and chunk == 0 and plan.group_id == 0:
                assert first_stage1 is not None
                assert len(sequence) - 1 == first_stage1 + 1
                assert before_flight[first.request_id] > 0
                overlapping_admission = True
        queue.append(step)
        if len(queue) == 3:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
        assert all(r.num_in_flight_tokens >= 0 for r in (first, second))
        if first.num_output_tokens == second.num_output_tokens == 1:
            break
    while queue:
        completed = queue.popleft()
        scheduler.update_from_output(completed, _wavefront_output(completed))
    assert overlapping_admission
    assert len(sequence) == 4 * groups
    assert set(frontiers.values()) == {groups}
    for req in (first, second):
        assert req.num_computed_tokens == 8
        assert req.num_in_flight_tokens == req.num_output_placeholders == 0
        assert req.num_output_tokens == 1


@pytest.mark.cpu_test
@pytest.mark.parametrize("action", ["abort", "preempt"])
def test_async_pp2_wave_interrupt_preserves_other_cohort_and_stale_fence(action):
    from collections import deque

    from tests.v1.core.utils import create_requests
    from vllm.v1.request import RequestStatus

    scheduler = _wavefront_scheduler(4, async_scheduling=True)
    scheduler.defer_block_free = True
    first, second = create_requests(
        num_requests=2,
        num_tokens=8,
        max_tokens=1,
        req_ids=["interrupt_a", "interrupt_b"],
    )
    for req in (first, second):
        scheduler.add_request(req)
    queue = deque()
    for _ in range(4):
        step = scheduler.schedule()
        queue.append(step)
        if len(queue) == 3:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
    assert queue[-1].layered_prefill_plan.prefill_req_ids == (second.request_id,)
    assert first.num_in_flight_tokens == second.num_in_flight_tokens == 4
    if action == "abort":
        scheduler.finish_requests(first.request_id, RequestStatus.FINISHED_ABORTED)
    else:
        scheduler.running.remove(first)
        scheduler._preempt_request(first, 0.0)
        assert first.num_stale_output_tokens == 4
    # The saved PP1 tail belongs to an interrupted generation. Its pending
    # completion must drain; meanwhile the other cohort can still advance.
    step = scheduler.schedule()
    assert first.request_id not in step.num_scheduled_tokens
    assert step.layered_prefill_plan.prefill_req_ids == (second.request_id,)
    queue.append(step)
    assert scheduler.deferred_frees
    for _ in range(128):
        if queue:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
        if not scheduler.get_num_unfinished_requests() and not queue:
            break
        if scheduler.get_num_unfinished_requests():
            queue.append(scheduler.schedule())
    assert not scheduler.deferred_frees
    if action == "abort":
        assert first.request_id not in scheduler.requests
        assert first not in scheduler.running
    else:
        assert first.num_stale_output_tokens == first.num_in_flight_tokens == 0
    assert second.num_output_tokens == 1
    assert first.num_output_tokens == (0 if action == "abort" else 1)
    assert first.num_output_placeholders == second.num_output_placeholders == 0
    assert scheduler.get_num_unfinished_requests() == 0


def _ordinary_async_scheduler(pp_size, *, groups=4, prefix_cache=False):
    scheduler = _local_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        use_v2_model_runner=True,
        async_scheduling=True,
        enable_prefix_caching=prefix_cache,
    )
    scheduler.parallel_config.pipeline_parallel_size = pp_size
    scheduler.pp_size = pp_size
    scheduler.use_pp = pp_size > 1
    scheduler.defer_block_free = True
    _enable_layered_scheduler(
        scheduler,
        allowed_num_groups=[groups],
        boundary_policy="off",
        max_chunk_tokens=4,
    )
    return scheduler


@pytest.mark.cpu_test
@pytest.mark.parametrize("pp_size", [1, 2, 3, 4])
@pytest.mark.parametrize("queue_depth", [1, None], ids=["serial", "engine_queue"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("max_tokens", [1, 5])
def test_ordinary_async_layered_chunks_finish_at_queue_depth(
    pp_size, queue_depth, mixed, max_tokens
):
    """Actual schedule/retire cycles preserve chunk coverage and sample counts."""
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler = _ordinary_async_scheduler(pp_size)
    if queue_depth is None:
        # Match EngineCore's MRV2 async queue: PP ranks plus one overlap slot.
        # Larger queues would also exceed the sampled-token ring's lifetime.
        queue_depth = pp_size + 1
    decodes = []
    if mixed:
        (decode,) = create_requests(
            num_requests=1,
            num_tokens=2,
            max_tokens=20,
            ignore_eos=True,
            req_ids=["decode"],
        )
        scheduler.add_request(decode)
        for _ in range(4):
            step = scheduler.schedule()
            scheduler.update_from_output(step, _wavefront_output(step))
        assert decode.num_output_tokens == 1
        decodes.append(decode)
    prefills = create_requests(
        num_requests=2,
        num_tokens=9,
        max_tokens=max_tokens,
        ignore_eos=True,
        req_ids=["p0", "p1"],
    )
    for request in prefills:
        scheduler.add_request(request)
    queue = deque()
    coverage = {}
    mixed_steps = 0
    for _ in range(256):
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        if plan is not None:
            assert plan.boundary_mode is None
            rid = plan.prefill_req_ids[0]
            key = (rid, plan.cached_tokens[rid])
            coverage.setdefault(key, []).extend(range(plan.group_start, plan.group_end))
            mixed_steps += bool(set(step.num_scheduled_tokens) - {rid})
        queue.append(step)
        if len(queue) >= queue_depth:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
        for request in prefills + decodes:
            assert request.num_output_placeholders >= 0
            assert request.num_preemptions == 0
            assert request.num_output_tokens <= request.max_tokens
        if not scheduler.get_num_unfinished_requests():
            break
    else:
        pytest.fail("ordinary async layered requests failed to finish")
    while queue:
        completed = queue.popleft()
        scheduler.update_from_output(completed, _wavefront_output(completed))
    for request in prefills:
        assert request.num_output_tokens == max_tokens
        assert request.num_output_placeholders == 0
        assert request.num_in_flight_tokens == 0
        assert [coverage[(request.request_id, start)] for start in (0, 4, 8)] == [
            list(range(32))
        ] * 3
    if mixed:
        assert mixed_steps > 0
    assert scheduler.sched_step_seq == scheduler.processed_step_seq
    assert not scheduler.deferred_frees


@pytest.mark.cpu_test
@pytest.mark.parametrize("pp_size", [1, 2, 3, 4])
def test_ordinary_async_queues_same_chunk_but_drains_before_next_chunk(pp_size):
    from tests.v1.core.utils import create_requests

    scheduler = _ordinary_async_scheduler(pp_size)
    (request,) = create_requests(num_requests=1, num_tokens=8, max_tokens=2)
    scheduler.add_request(request)
    queued = [scheduler.schedule() for _ in range(4)]
    assert [step.layered_prefill_plan.group_id for step in queued] == list(range(4))
    assert request.num_in_flight_tokens == 16
    assert request.num_computed_tokens == 4
    assert request.num_output_placeholders == 0
    assert request.last_sched_seq == scheduler.sched_step_seq == 4
    for step in queued:
        assert scheduler.schedule().total_num_scheduled_tokens == 0
        scheduler.update_from_output(step, _wavefront_output(step))
    next_chunk = scheduler.schedule().layered_prefill_plan
    assert next_chunk is not None and next_chunk.group_id == 0
    assert next_chunk.cached_tokens[request.request_id] == 4


@pytest.mark.cpu_test
@pytest.mark.parametrize("pp_size", [1, 2, 3, 4])
@pytest.mark.parametrize("action", ["abort", "preempt"])
def test_ordinary_async_queued_groups_keep_kv_until_abort_or_preempt_retires(
    pp_size, action
):
    from vllm.v1.request import RequestStatus

    from tests.v1.core.utils import create_requests

    scheduler = _ordinary_async_scheduler(pp_size)
    (request,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=2, ignore_eos=True
    )
    scheduler.add_request(request)
    queued = [scheduler.schedule(), scheduler.schedule()]
    assert request.last_sched_seq == 2
    if action == "abort":
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    else:
        scheduler.running.remove(request)
        scheduler._preempt_request(request, 0.0)
        assert request.num_stale_output_tokens == 8
        # Layered admission must respect the same stale-generation drain as
        # regular admission; otherwise it can restart on freed KV too early.
        assert scheduler.schedule().total_num_scheduled_tokens == 0
        assert request.status == RequestStatus.PREEMPTED
    assert scheduler.deferred_frees
    scheduler.update_from_output(queued[0], _wavefront_output(queued[0]))
    assert scheduler.deferred_frees
    scheduler.update_from_output(queued[1], _wavefront_output(queued[1]))
    assert not scheduler.deferred_frees
    if action == "abort":
        assert request.request_id not in scheduler.requests
        return
    assert request.num_stale_output_tokens == request.num_in_flight_tokens == 0
    resumed = scheduler.schedule()
    assert resumed.layered_prefill_plan.group_id == 0
    assert resumed.scheduled_new_reqs[0].req_id == request.request_id
    scheduler.update_from_output(resumed, _wavefront_output(resumed))
    for _ in range(64):
        if not scheduler.get_num_unfinished_requests():
            break
        step = scheduler.schedule()
        scheduler.update_from_output(step, _wavefront_output(step))
    assert request.num_output_tokens == 2
    assert request.num_output_placeholders == 0
    assert scheduler.get_num_unfinished_requests() == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("pp_size", [1, 2, 3, 4])
def test_ordinary_async_preempted_final_sample_drains_old_generation(pp_size):
    from tests.v1.core.utils import create_requests

    scheduler = _ordinary_async_scheduler(pp_size)
    (request,) = create_requests(
        num_requests=1, num_tokens=4, max_tokens=3, ignore_eos=True
    )
    scheduler.add_request(request)
    queued = [scheduler.schedule() for _ in range(4)]
    assert queued[-1].layered_prefill_plan.is_sampling_step
    assert request.num_output_placeholders == 1
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.num_output_placeholders == 0
    for step in queued:
        assert scheduler.schedule().total_num_scheduled_tokens == 0
        scheduler.update_from_output(step, _wavefront_output(step))
        assert request.num_output_placeholders == 0
    # The old generation's real sample is delivered exactly once without
    # consuming a placeholder from the resumed generation.
    assert request.num_output_tokens == 1
    assert request.num_stale_output_tokens == request.num_in_flight_tokens == 0
    assert not scheduler.deferred_frees
    for _ in range(32):
        if not scheduler.get_num_unfinished_requests():
            break
        step = scheduler.schedule()
        scheduler.update_from_output(step, _wavefront_output(step))
    assert request.num_output_tokens == 3
    assert request.num_output_placeholders == 0
    assert scheduler.get_num_unfinished_requests() == 0


@pytest.mark.cpu_test
def test_ordinary_async_prefix_cache_waits_for_final_group_retirement(monkeypatch):
    from tests.v1.core.utils import create_requests

    scheduler = _ordinary_async_scheduler(1, groups=2, prefix_cache=True)
    (request,) = create_requests(num_requests=1, num_tokens=4, max_tokens=4)
    scheduler.add_request(request)
    published = []
    cache_blocks = scheduler.kv_cache_manager.cache_blocks

    def record_cache(req, computed):
        published.append(computed)
        return cache_blocks(req, computed)

    monkeypatch.setattr(scheduler.kv_cache_manager, "cache_blocks", record_cache)
    first, final = scheduler.schedule(), scheduler.schedule()
    assert not published
    decode = scheduler.schedule()
    assert decode.layered_prefill_plan is None
    assert request.num_computed_tokens == 5
    scheduler.update_from_output(first, _wavefront_output(first))
    assert not published
    scheduler.update_from_output(final, _wavefront_output(final))
    assert published and max(published) == 4
    scheduler.update_from_output(decode, _wavefront_output(decode))


def _wavefront_output(step):
    """Sample D and only the P plan's final chunk, using its saved snapshot."""
    result = _empty_runner_output(list(step.num_scheduled_tokens))
    plan = step.layered_prefill_plan
    for index, req_id in enumerate(result.req_ids):
        if plan is None or req_id not in plan.prefill_req_ids or plan.is_sampling_step:
            result.sampled_token_ids[index] = [1000]
    return result


def _drain_blocked_layered_wave(scheduler, queue, step):
    """Consume a required drain, never excuse an idle runnable P group."""
    assert step.layered_prefill_plan is None
    assert queue
    assert not scheduler.waiting and not scheduler.skipped_waiting
    pending = [r for r in scheduler.running if not r.output_token_ids]
    assert pending
    assert all(
        r.num_in_flight_tokens > 0
        and (not r.layered_prefill_enabled or r.layered_prefill_group_id == 0)
        for r in pending
    ), [
        (r.request_id, r.layered_prefill_group_id, r.num_in_flight_tokens)
        for r in pending
    ]
    # A D-only step still occupies one of EngineCore's two RPC slots.
    if step.total_num_scheduled_tokens:
        queue.append(step)
    assert len(queue) <= 2
    completed = queue.popleft()
    scheduler.update_from_output(completed, _wavefront_output(completed))
    return completed.layered_prefill_plan


@pytest.mark.parametrize("floor, expected_query", [(0, 2), (2, 2)])
def test_pp2_v2_decode_floor_covers_tokens_ready_after_chunk_opens(
    monkeypatch, floor, expected_query
):
    """A D RPC and B's final P RPC retire after C's width has frozen."""
    from tests.v1.core.utils import create_requests

    monkeypatch.setenv("VLLM_LAYERED_DECODE_RESERVE_TOKENS", str(floor))
    scheduler = _local_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=4,
        use_v2_model_runner=True,
        async_scheduling=False,
    )
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.parallel_config.tensor_parallel_size = 2
    scheduler.use_pp = True
    _enable_layered_scheduler(scheduler, allowed_num_groups=[2], max_chunk_tokens=4)
    assert scheduler._layered_decode_reserve_tokens == floor
    # Initialization snapshots the experiment; later environment changes do
    # not alter an existing scheduler, including midway through a P chunk.
    monkeypatch.setenv("VLLM_LAYERED_DECODE_RESERVE_TOKENS", "99")
    decode = _add_ready_decode(scheduler)
    (b,) = create_requests(
        num_requests=1, num_tokens=3, max_tokens=8, req_ids=["floor_b"]
    )
    scheduler.add_request(b)
    for _ in range(16):
        b_final = scheduler.schedule()
        plan = b_final.layered_prefill_plan
        assert plan is not None and plan.prefill_req_ids == (b.request_id,)
        if plan.is_sampling_step:
            break
        scheduler.update_from_output(b_final, _wavefront_output(b_final))
    else:
        pytest.fail("B did not reach its final prompt group")
    assert b_final.num_scheduled_tokens[decode.request_id] == 1
    assert decode.num_in_flight_tokens == 1
    assert b.num_in_flight_tokens > 0 and b.num_output_tokens == 0

    (c,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=8, req_ids=["floor_c"]
    )
    scheduler.add_request(c)
    c_first = scheduler.schedule()  # B-final and C0 now occupy the two RPC slots.
    first_plan = c_first.layered_prefill_plan
    assert first_plan is not None and first_plan.group_id == 0
    assert first_plan.prefill_req_ids == (c.request_id,)
    assert first_plan.query_tokens[c.request_id] == expected_query
    assert decode.request_id not in c_first.num_scheduled_tokens
    assert b.request_id not in c_first.num_scheduled_tokens

    scheduler.update_from_output(b_final, _wavefront_output(b_final))
    assert decode.num_tokens_with_spec - decode.num_computed_tokens == 1
    assert b.num_tokens_with_spec - b.num_computed_tokens == 1
    c_final = scheduler.schedule()  # C0 is still outstanding: never resize C1.
    assert c_final.layered_prefill_plan.group_id == 1
    assert c_final.layered_prefill_plan.query_tokens[c.request_id] == expected_query
    for req in (decode, b):
        assert c_final.num_scheduled_tokens.get(req.request_id, 0) == 1
    assert c_final.total_num_scheduled_tokens <= 4
    for step in (c_first, c_final):
        scheduler.update_from_output(step, _wavefront_output(step))
    assert all(req.num_in_flight_tokens == 0 for req in (decode, b, c))


@pytest.mark.parametrize("value", ["-1", "abc", "1.5"])
def test_layered_decode_floor_invalid_environment_fails_at_init(monkeypatch, value):
    monkeypatch.setenv("VLLM_LAYERED_DECODE_RESERVE_TOKENS", value)
    with pytest.raises(ValueError, match="VLLM_LAYERED_DECODE_RESERVE_TOKENS"):
        _local_scheduler(use_v2_model_runner=True)


@pytest.mark.parametrize("with_decode", [False, True])
def test_pp2_v2_public_queue_submits_successive_groups_without_bubbles(with_decode):
    """Match EngineCore: enqueue a second RPC before consuming the oldest."""
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler()
    decode = _add_ready_decode(scheduler) if with_decode else None
    first, second = create_requests(
        num_requests=2,
        num_tokens=8,
        max_tokens=1,
        req_ids=["wave_a", "wave_b"],
    )
    for request in (first, second):
        scheduler.add_request(request)
    queue = deque()
    sequence = []
    for _ in range(8):
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        assert plan is not None
        req_id = plan.prefill_req_ids[0]
        sequence.append((req_id, plan.group_id, plan.cached_tokens[req_id]))
        assert plan.query_tokens[req_id] == 4
        assert plan.commit_tokens[req_id] == (4 if plan.is_final_group else 0)
        if decode is not None:
            assert step.num_scheduled_tokens.get(decode.request_id, 0) <= 1
        queue.append(step)
        if len(queue) == 2:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
        assert all(request.num_in_flight_tokens >= 0 for request in (first, second))
    while queue:
        completed = queue.popleft()
        scheduler.update_from_output(completed, _wavefront_output(completed))
    assert sequence == [
        (request.request_id, group, chunk)
        for chunk in (0, 4)
        for request in (first, second)
        for group in (0, 1)
    ]
    for request in (first, second):
        assert request.num_computed_tokens == 8
        assert request.num_in_flight_tokens == 0
        assert request.num_output_tokens == 1


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
def test_pp2_v2_wavefront_does_not_submit_next_chunk_before_prior_outputs(groups):
    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler(groups)
    (request,) = create_requests(num_requests=1, num_tokens=8, max_tokens=1)
    scheduler.add_request(request)
    for group in range(groups - 2):
        completed = scheduler.schedule()
        assert completed.layered_prefill_plan.group_id == group
        scheduler.update_from_output(completed, _wavefront_output(completed))
    first, final = scheduler.schedule(), scheduler.schedule()
    assert [
        first.layered_prefill_plan.group_id,
        final.layered_prefill_plan.group_id,
    ] == [
        groups - 2,
        groups - 1,
    ]
    assert request.num_computed_tokens == 4
    assert request.num_in_flight_tokens == 8
    assert not scheduler.schedule().num_scheduled_tokens
    scheduler.update_from_output(first, _wavefront_output(first))
    assert request.num_in_flight_tokens == 4
    assert not scheduler.schedule().num_scheduled_tokens
    scheduler.update_from_output(final, _wavefront_output(final))
    next_chunk = scheduler.schedule().layered_prefill_plan
    assert next_chunk.group_id == 0
    assert next_chunk.cached_tokens[request.request_id] == 4
    assert first.layered_prefill_plan.cached_tokens[request.request_id] == 0
    assert final.layered_prefill_plan.cached_tokens[request.request_id] == 0


@pytest.mark.parametrize("interrupt", ["abort", "preempt"])
@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
def test_pp2_v2_wavefront_drains_both_outputs_after_interrupt(interrupt, groups):
    from tests.v1.core.utils import create_requests

    from vllm.v1.request import RequestStatus

    scheduler = _wavefront_scheduler(groups)
    (request,) = create_requests(num_requests=1, num_tokens=8, max_tokens=1)
    scheduler.add_request(request)
    for _ in range(groups - 2):
        completed = scheduler.schedule()
        scheduler.update_from_output(completed, _wavefront_output(completed))
    queued = [scheduler.schedule(), scheduler.schedule()]
    assert request.num_in_flight_tokens == 8
    if interrupt == "abort":
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    else:
        scheduler.reset_prefix_cache(reset_running_requests=True)
        assert request.num_computed_tokens == 0
        assert request.num_stale_output_tokens == 8
    for step in queued:
        scheduler.update_from_output(step, _wavefront_output(step))
    assert request.num_output_tokens == 0
    if interrupt == "abort":
        assert request.request_id not in scheduler.requests
        assert request not in scheduler.running
    else:
        assert request.num_in_flight_tokens == 0
        assert request.num_stale_output_tokens == 0
        resumed = scheduler.schedule().layered_prefill_plan
        assert resumed.group_id == 0
        assert resumed.cached_tokens[request.request_id] == 0


def _assert_pp2_wave_interleaves_owners_with_independent_frontiers(
    request,
    monkeypatch,
    with_decode,
    groups,
    layers,
    partition,
):
    """Two RPC slots alternate A's PP1 tail and B's PP0 work without early samples."""
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler(groups, layers)
    if partition is not None:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    scheduler._layered_pp_group_ranges.cache_clear()
    # Clearing after this test prevents a custom partition leaking into the
    # cached (layers, groups, pp_size) lookup in the next parameter case.
    request.addfinalizer(scheduler._layered_pp_group_ranges.cache_clear)
    decode = _add_ready_decode(scheduler) if with_decode else None
    first, second = create_requests(
        num_requests=2,
        num_tokens=8,
        max_tokens=1,
        req_ids=["a", "b"],
    )
    for request in (first, second):
        scheduler.add_request(request)
    queue = deque()
    sequence = []
    frontiers = {}
    samples = []
    concurrent_owners = 0
    owners = []
    from vllm.distributed.utils import get_pp_indices

    stage0_end = get_pp_indices(layers, 0, 2)[1]
    for _ in range(4 * groups + 4):
        before_flight = {r.request_id: r.num_in_flight_tokens for r in (first, second)}
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        if plan is None:
            completed_plan = _drain_blocked_layered_wave(scheduler, queue, step)
            if completed_plan is not None and completed_plan.is_sampling_step:
                samples.append(completed_plan.prefill_req_ids[0])
            continue
        req_id = plan.prefill_req_ids[0]
        chunk = plan.cached_tokens[req_id]
        key = (req_id, plan.cohort_id, chunk)
        assert plan.group_id == frontiers.get(key, 0)
        frontiers[key] = plan.group_id + 1
        if plan.group_id == 0:
            assert before_flight[req_id] == 0
        sequence.append((req_id, plan.group_id, chunk))
        owners.append(int(plan.group_start >= stage0_end))
        assert plan.query_tokens[req_id] == 4
        assert plan.commit_tokens[req_id] == (4 if plan.is_final_group else 0)
        if decode is not None:
            assert step.num_scheduled_tokens.get(decode.request_id, 0) <= 1
        queue.append(step)
        if len(queue) == 2:
            older_plan = queue[0].layered_prefill_plan
            if (
                older_plan is not None
                and older_plan.prefill_req_ids != plan.prefill_req_ids
                and (older_plan.group_start >= stage0_end)
                != (plan.group_start >= stage0_end)
            ):
                concurrent_owners += 1
                assert all(r.num_in_flight_tokens > 0 for r in (first, second))
            completed = queue.popleft()
            result = _wavefront_output(completed)
            completed_plan = completed.layered_prefill_plan
            if completed_plan is not None and completed_plan.is_sampling_step:
                samples.append(completed_plan.prefill_req_ids[0])
            scheduler.update_from_output(completed, result)
        assert all(r.num_in_flight_tokens >= 0 for r in (first, second))
        if len(sequence) == 4 * groups:
            break
    assert len(sequence) == 4 * groups
    while queue:
        completed = queue.popleft()
        completed_plan = completed.layered_prefill_plan
        if completed_plan is not None and completed_plan.is_sampling_step:
            samples.append(completed_plan.prefill_req_ids[0])
        scheduler.update_from_output(completed, _wavefront_output(completed))
    if groups == 4 and partition is None:
        assert sequence == [
            ("a", 0, 0),
            ("a", 1, 0),
            ("a", 2, 0),
            ("b", 0, 0),
            ("a", 3, 0),
            ("b", 1, 0),
            ("b", 2, 0),
            ("a", 0, 4),
            ("b", 3, 0),
            ("a", 1, 4),
            ("a", 2, 4),
            ("b", 0, 4),
            ("a", 3, 4),
            ("b", 1, 4),
            ("b", 2, 4),
            ("b", 3, 4),
        ]
    # After the first PP1 submission, another request must fill PP0 while
    # the first is still in flight; all subsequent PP1 groups yield again.
    first_stage1 = owners.index(1)
    assert sequence[first_stage1 + 1] == ("b", 0, 0)
    first_tail = [
        index
        for index, (req_id, _, chunk) in enumerate(sequence)
        if req_id == "a" and chunk == 0 and owners[index] == 1
    ]
    assert all(
        sequence[index + 1][0] == "b" and owners[index + 1] == 0
        for index in first_tail[:first_stage1]
    )
    assert concurrent_owners >= min(first_stage1, len(first_tail))
    assert set(frontiers.values()) == {groups}
    assert samples == ["a", "b"]
    for request in (first, second):
        assert request.num_computed_tokens == 8
        assert request.num_in_flight_tokens == 0
        assert request.num_output_tokens == 1


@pytest.mark.parametrize("with_decode", [False, True])
@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize(
    "layers, partition",
    [(32, None), (43, None), (43, "3,40"), (43, "40,3")],
)
def test_pp2_wave_interleaves_owners_with_independent_frontiers(
    request,
    monkeypatch,
    with_decode,
    groups,
    layers,
    partition,
):
    _assert_pp2_wave_interleaves_owners_with_independent_frontiers(
        request,
        monkeypatch,
        with_decode,
        groups,
        layers,
        partition,
    )


@pytest.mark.parametrize("groups", range(2, 44))
def test_pp2_wave_interleaves_every_legal_group_count(request, monkeypatch, groups):
    _assert_pp2_wave_interleaves_owners_with_independent_frontiers(
        request,
        monkeypatch,
        False,
        groups,
        43,
        None,
    )


def test_pp2_group4_wave_never_opens_a_third_prefill_cohort():
    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler(4)
    requests = create_requests(num_requests=3, num_tokens=8, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)
    for group in (0, 1):
        step = scheduler.schedule()
        assert step.layered_prefill_plan.group_id == group
        scheduler.update_from_output(step, _wavefront_output(step))
    a2, b0 = scheduler.schedule(), scheduler.schedule()
    assert b0.layered_prefill_plan.prefill_req_ids == (requests[1].request_id,)
    scheduler.update_from_output(a2, _wavefront_output(a2))
    a3 = scheduler.schedule()
    scheduler.update_from_output(b0, _wavefront_output(b0))
    b1 = scheduler.schedule()
    scheduler.update_from_output(a3, _wavefront_output(a3))
    b2 = scheduler.schedule()
    assert b2.layered_prefill_plan.group_id == 2
    scheduler.update_from_output(b1, _wavefront_output(b1))
    next_chunk = scheduler.schedule().layered_prefill_plan
    assert next_chunk.prefill_req_ids == (requests[0].request_id,)
    assert next_chunk.cached_tokens[requests[0].request_id] == 4
    assert requests[2].status.name == "WAITING"
    assert len(scheduler.running) == 2


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
def test_pp2_wave_bounds_cohorts_and_eventually_serves_all_waiters(groups):
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler(groups, layers=43)
    requests = create_requests(num_requests=3, num_tokens=8, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)
    queue = deque()
    submitted = {request.request_id: 0 for request in requests}
    for _ in range(6 * groups + 6):
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        if plan is None:
            _drain_blocked_layered_wave(scheduler, queue, step)
            continue
        submitted[plan.prefill_req_ids[0]] += 1
        assert len(scheduler.running) <= 2
        queue.append(step)
        assert len(queue) <= 2
        if len(queue) == 2:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
        if sum(submitted.values()) == 6 * groups:
            break
    while queue:
        completed = queue.popleft()
        scheduler.update_from_output(completed, _wavefront_output(completed))
    assert set(submitted.values()) == {2 * groups}
    assert all(request.num_output_tokens == 1 for request in requests)
    assert all(request.num_in_flight_tokens == 0 for request in requests)
    assert not scheduler.has_unfinished_requests()


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
def test_pp2_wave_prefix_hit_preserves_chunk_width_and_group_order(groups):
    from collections import deque

    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler(groups, enable_prefix_caching=True)
    first, second = create_requests(
        num_requests=2,
        num_tokens=32,
        max_tokens=1,
        same_prompt=True,
    )
    scheduler.add_request(first)
    queue = deque()
    for _ in range(8 * groups):
        step = scheduler.schedule()
        # A single cohort must drain its final group before its next chunk.
        if step.layered_prefill_plan is not None:
            queue.append(step)
        if queue:
            completed = queue.popleft()
            scheduler.update_from_output(completed, _wavefront_output(completed))
    assert first.num_output_tokens == 1
    scheduler.add_request(second)
    for group in range(groups):
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        assert plan is not None and plan.group_id == group
        assert plan.cached_tokens[second.request_id] == 16
        assert plan.query_tokens[second.request_id] == 4
        assert not plan.is_sampling_step
        scheduler.update_from_output(step, _wavefront_output(step))
    assert second.num_computed_tokens == 20
    assert second.num_in_flight_tokens == 0
    assert second.num_output_tokens == 0


@pytest.mark.parametrize("groups", [4, 5, 8, 16])
def test_pp2_wave_admission_failure_does_not_starve_tail(monkeypatch, groups):
    from tests.v1.core.utils import create_requests

    scheduler = _wavefront_scheduler(groups)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=1)
    scheduler.add_request(first)
    scheduler.add_request(second)
    while True:
        step = scheduler.schedule()
        if scheduler._layered_pp_group_owner(first) == 1:
            scheduler.update_from_output(step, _wavefront_output(step))
            break
        scheduler.update_from_output(step, _wavefront_output(step))
    stage1 = scheduler.schedule()
    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", lambda *a, **kw: None
    )
    failed = scheduler.schedule()
    assert second.request_id not in failed.num_scheduled_tokens
    tail = scheduler.schedule().layered_prefill_plan
    assert tail is not None and tail.prefill_req_ids == (first.request_id,)
    assert tail.group_id == stage1.layered_prefill_plan.group_id + 1


def _pp_reserve_scheduler(groups):
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=32,
        use_v2_model_runner=True,
        async_scheduling=False,
    )
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.parallel_config.tensor_parallel_size = 2
    scheduler.use_pp = True
    decodes = create_requests(
        num_requests=3,
        num_tokens=2,
        max_tokens=32,
        req_ids=["d0", "d1", "d2"],
    )
    for request in decodes:
        scheduler.add_request(request)
    prompt = scheduler.schedule()
    scheduler.update_from_output(prompt, _wavefront_output(prompt))
    assert all(request.num_output_tokens == 1 for request in decodes)
    scheduler.max_num_scheduled_tokens = 8
    _enable_layered_scheduler(
        scheduler,
        allowed_num_groups=[groups],
        max_chunk_tokens=8,
    )
    return scheduler, decodes


@pytest.mark.parametrize("groups", [2, 3, 4, 5, 8, 16])
def test_pp2_reserve_admits_decode_after_chunk_width_freezes(groups):
    """Fresh group zero keeps room for D that becomes ready at the next group."""
    from tests.v1.core.utils import create_requests

    scheduler, decodes = _pp_reserve_scheduler(groups)
    pending = scheduler.schedule()
    (prefill,) = create_requests(num_requests=1, num_tokens=8, max_tokens=1)
    scheduler.add_request(prefill)
    first = scheduler.schedule()
    assert first.layered_prefill_plan.query_tokens[prefill.request_id] == 4
    assert set(first.num_scheduled_tokens) == {prefill.request_id}
    scheduler.update_from_output(pending, _wavefront_output(pending))
    following = scheduler.schedule()
    assert following.layered_prefill_plan.group_id == 1
    assert following.layered_prefill_plan.query_tokens[prefill.request_id] == 4
    assert {r.request_id for r in decodes} <= set(following.num_scheduled_tokens)
    for step in (first, following):
        scheduler.update_from_output(step, _wavefront_output(step))
    assert all(r.num_in_flight_tokens == 0 for r in (*decodes, prefill))


@pytest.mark.parametrize("flag", ["same_layer_batch", "fuse_mixed_batch"])
def test_unsupported_mixed_execution_modes_fail_closed(flag):
    config = _config()
    config.additional_config["scheduler_config"]["layered_prefill_config"][flag] = True
    with pytest.raises(ValueError, match="not supported"):
        LayeredPrefillConfig.from_vllm_config(config)


def test_ring_step_plan_survives_scheduler_output_copy_and_transport():
    import pickle
    from dataclasses import replace

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.pp_utils import RingStepPlan

    plan = RingStepPlan(
        step_id=7,
        collective_required=True,
        payload_rows=1,
        sampled_request_order=("decode",),
        sample_width=1,
        plan_hash="step-7",
    )
    output = SchedulerOutput.make_empty()
    output.ring_step_plan = plan
    assert replace(output).ring_step_plan is plan
    assert pickle.loads(pickle.dumps(output)).ring_step_plan == plan


def _decode_first_boundary_scheduler(
    *,
    groups=4,
    budget=8,
    num_decodes=1,
    max_num_seqs=6,
    chunk_threshold=0,
    async_scheduling=False,
):
    """Bootstrap ready D rows, then enable the opt-in boundary cohort policy."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        max_model_len=16384,
        use_v2_model_runner=True,
        async_scheduling=async_scheduling,
    )
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.parallel_config.tensor_parallel_size = 2
    scheduler.use_pp = True
    if async_scheduling:
        scheduler.pp_size = 2
    decodes = create_requests(
        num_requests=num_decodes,
        num_tokens=2,
        max_tokens=32,
        req_ids=[f"decode_{i}" for i in range(num_decodes)],
    )
    for request in decodes:
        scheduler.add_request(request)
    bootstrap = scheduler.schedule()
    scheduler.update_from_output(bootstrap, _wavefront_output(bootstrap))
    assert all(request.num_output_tokens == 1 for request in decodes)
    if chunk_threshold:
        scheduler.scheduler_config.long_prefill_token_threshold = chunk_threshold
    _enable_layered_scheduler(
        scheduler,
        allowed_num_groups=[groups],
        boundary_policy="pp_transition",
        decode_first=True,
        max_prefill_batch_size=2,
        max_chunk_tokens=budget,
    )
    scheduler.defer_block_free = True
    # Boundary admission deliberately needs retired D-only history. Enabling
    # the policy after bootstrap must therefore observe one policy-owned D step.
    decode_only = scheduler.schedule()
    if async_scheduling and not decode_only.total_num_scheduled_tokens:
        # PP2's first decode waits for the broadcast ring's second slot.
        scheduler.update_from_output(decode_only, _cohort_wavefront_output(decode_only))
        decode_only = scheduler.schedule()
    assert decode_only.layered_prefill_plan is None
    scheduler.update_from_output(decode_only, _cohort_wavefront_output(decode_only))
    return scheduler, decodes


@pytest.mark.cpu_test
@pytest.mark.parametrize("max_prefill_batch_size", [1, 2])
def test_async_boundary_samples_subset_once_at_queue_depth_two(max_prefill_batch_size):
    from collections import deque
    from dataclasses import replace

    from tests.v1.core.utils import create_requests

    scheduler, _ = _decode_first_boundary_scheduler(
        groups=4, budget=8, async_scheduling=True
    )
    scheduler.layered_prefill_policy.config = replace(
        scheduler.layered_prefill_policy.config,
        max_prefill_batch_size=max_prefill_batch_size,
    )
    (short,) = create_requests(
        num_requests=1, num_tokens=2, max_tokens=1, req_ids=["short"]
    )
    scheduler.add_request(short)
    long = None
    if max_prefill_batch_size == 2:
        (long,) = create_requests(
            num_requests=1, num_tokens=8, max_tokens=4, req_ids=["long"]
        )
        scheduler.add_request(long)

    queue = deque()
    plans = []
    for _ in range(20):
        step = scheduler.schedule()
        queue.append(step)
        plan = step.layered_prefill_plan
        if plan is not None:
            plans.append(plan)
            if plan.is_final_group:
                assert short.num_output_placeholders == 1
                assert short.next_decode_eligible_step == scheduler.current_step + 2
                if long is not None:
                    assert plan.sampling_req_ids == (short.request_id,)
                    assert long.num_output_placeholders == 0
            else:
                assert short.num_output_placeholders == 0
        if len(queue) < 2:
            continue
        completed = queue.popleft()
        scheduler.update_from_output(completed, _cohort_wavefront_output(completed))
        completed_plan = completed.layered_prefill_plan
        if completed_plan is not None and completed_plan.is_final_group:
            break
    else:
        pytest.fail("boundary final interval did not retire")
    assert [
        layer for plan in plans for layer in range(plan.group_start, plan.group_end)
    ] == list(range(32))
    assert short.num_output_tokens == 1
    assert short.num_output_placeholders == 0
    if long is not None:
        assert long.num_output_tokens == 0
        assert long.num_computed_tokens == 4


@pytest.mark.cpu_test
def test_async_boundary_partial_abort_preserves_cohort_fence():
    from vllm.v1.request import RequestStatus

    from tests.v1.core.utils import create_requests

    scheduler, _ = _decode_first_boundary_scheduler(
        groups=8, budget=12, chunk_threshold=4, async_scheduling=True
    )
    aborted, survivor = create_requests(
        num_requests=2, num_tokens=9, max_tokens=4, req_ids=["abort", "survivor"]
    )
    scheduler.add_request(aborted)
    scheduler.add_request(survivor)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert first.layered_prefill_plan is not None
    assert second.layered_prefill_plan is not None
    scheduler.finish_requests(aborted.request_id, RequestStatus.FINISHED_ABORTED)
    assert scheduler.deferred_frees
    scheduler.update_from_output(first, _cohort_wavefront_output(first))
    assert scheduler.deferred_frees
    continued = scheduler.schedule()
    assert continued.layered_prefill_plan.prefill_req_ids == (survivor.request_id,)
    scheduler.update_from_output(second, _cohort_wavefront_output(second))
    assert not scheduler.deferred_frees
    assert aborted.num_output_placeholders == survivor.num_output_placeholders == 0


def _cohort_wavefront_output(step):
    """Sample D plus only P rows explicitly eligible in this interval."""
    result = _empty_runner_output(list(step.num_scheduled_tokens))
    plan = step.layered_prefill_plan
    prefill_ids = set() if plan is None else set(plan.prefill_req_ids)
    if plan is None:
        sampling_prefills = set()
    elif plan.sampling_req_ids is None:
        sampling_prefills = prefill_ids if plan.is_sampling_step else set()
    else:
        sampling_prefills = set(plan.sampling_req_ids) if plan.is_final_group else set()
    for index, req_id in enumerate(result.req_ids):
        if req_id not in prefill_ids or req_id in sampling_prefills:
            result.sampled_token_ids[index] = [1000]
    return result


def test_boundary_batch_decode_first_reserves_all_decode_rows_at_depth_two():
    from tests.v1.core.utils import create_requests

    scheduler, decodes = _decode_first_boundary_scheduler(
        groups=4, budget=8, num_decodes=2
    )
    prefills = create_requests(
        num_requests=2, num_tokens=3, max_tokens=4, req_ids=["p0", "p1"]
    )
    for request in prefills:
        scheduler.add_request(request)

    first = scheduler.schedule()
    first_plan = first.layered_prefill_plan
    assert list(first.num_scheduled_tokens)[:2] == [r.request_id for r in decodes]
    assert first_plan is not None
    assert first_plan.prefill_req_ids == tuple(r.request_id for r in prefills)
    assert sum(first_plan.query_tokens.values()) == 6
    assert first.total_num_scheduled_tokens == 8

    second = scheduler.schedule()
    second_plan = second.layered_prefill_plan
    assert second_plan is not None
    assert second_plan.cohort_id == first_plan.cohort_id
    assert second_plan.prefill_req_ids == first_plan.prefill_req_ids
    assert second_plan.query_tokens == first_plan.query_tokens
    assert second.total_num_scheduled_tokens <= 6

    scheduler.update_from_output(first, _cohort_wavefront_output(first))
    third = scheduler.schedule()
    third_plan = third.layered_prefill_plan
    assert third_plan is not None
    assert third_plan.cohort_id == first_plan.cohort_id
    assert third_plan.prefill_req_ids == first_plan.prefill_req_ids
    assert third_plan.query_tokens == first_plan.query_tokens
    assert all(third.num_scheduled_tokens[r.request_id] == 1 for r in decodes)
    assert third.total_num_scheduled_tokens <= 8


def test_boundary_batch_samples_only_prefills_finishing_this_chunk():
    from tests.v1.core.utils import create_requests

    scheduler, _ = _decode_first_boundary_scheduler(groups=4, budget=8)
    (short,) = create_requests(
        num_requests=1, num_tokens=2, max_tokens=4, req_ids=["short"]
    )
    (long,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(short)
    scheduler.add_request(long)

    plans = []
    while True:
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        assert plan is not None
        plans.append(plan)
        assert plan.prefill_req_ids == (short.request_id, long.request_id)
        assert plan.query_tokens == {short.request_id: 2, long.request_id: 4}
        if not plan.is_final_group:
            assert not any(plan.commit_tokens.values())
        scheduler.update_from_output(step, _cohort_wavefront_output(step))
        if plan.is_final_group:
            break

    final = plans[-1]
    assert final.commit_tokens == final.query_tokens
    assert final.is_final_chunk is False
    assert final.sampling_req_ids == (short.request_id,)
    assert final.is_sampling_step
    assert short.num_output_tokens == 1
    assert long.num_output_tokens == 0
    assert long.num_computed_tokens == 4


def test_boundary_batch_continuation_stays_in_lockstep_and_holds_next_chunk():
    from tests.v1.core.utils import create_requests

    scheduler, _ = _decode_first_boundary_scheduler(
        groups=4, budget=12, chunk_threshold=4
    )
    prefills = create_requests(
        num_requests=2, num_tokens=9, max_tokens=4, req_ids=["lock0", "lock1"]
    )
    for request in prefills:
        scheduler.add_request(request)

    first = scheduler.schedule()
    second = scheduler.schedule()
    first_plan = first.layered_prefill_plan
    second_plan = second.layered_prefill_plan
    assert first_plan is not None and second_plan is not None
    assert (
        first_plan.prefill_req_ids
        == second_plan.prefill_req_ids
        == (
            prefills[0].request_id,
            prefills[1].request_id,
        )
    )
    assert (
        first_plan.query_tokens
        == second_plan.query_tokens
        == {
            prefills[0].request_id: 4,
            prefills[1].request_id: 4,
        }
    )
    assert second_plan.cohort_id == first_plan.cohort_id
    assert second_plan.group_start == first_plan.group_end

    from collections import deque

    queue = deque([first, second])
    plans = [first_plan, second_plan]
    while not plans[-1].is_final_group:
        completed = queue.popleft()
        scheduler.update_from_output(completed, _cohort_wavefront_output(completed))
        step = scheduler.schedule()
        plan = step.layered_prefill_plan
        if plan is not None:
            assert plan.cohort_id == first_plan.cohort_id
            assert plan.prefill_req_ids == first_plan.prefill_req_ids
            assert plan.query_tokens == first_plan.query_tokens
            plans.append(plan)
        if step.total_num_scheduled_tokens:
            queue.append(step)
        assert len(queue) <= 2

    assert [
        layer for plan in plans for layer in range(plan.group_start, plan.group_end)
    ] == list(range(32))
    assert plans[-1].resume_group_id == plans[-1].num_groups == 4
    # The cohort's next chunk cannot enter while its final interval is pending.
    final_retired = False
    while queue:
        completed = queue.popleft()
        was_final = (
            completed.layered_prefill_plan is not None
            and completed.layered_prefill_plan.is_final_group
        )
        scheduler.update_from_output(completed, _cohort_wavefront_output(completed))
        final_retired |= was_final
        if not final_retired:
            held = scheduler.schedule()
            assert not set(first_plan.prefill_req_ids) & set(held.num_scheduled_tokens)
            if held.total_num_scheduled_tokens:
                queue.append(held)

    next_chunk = scheduler.schedule()
    assert all(req.request_id in next_chunk.num_scheduled_tokens for req in prefills)


def test_boundary_batch_partial_abort_continues_only_survivor():
    from tests.v1.core.utils import create_requests

    from vllm.v1.request import RequestStatus

    scheduler, _ = _decode_first_boundary_scheduler(
        groups=8, budget=12, chunk_threshold=4
    )
    aborted, survivor = create_requests(
        num_requests=2, num_tokens=9, max_tokens=4, req_ids=["abort", "survivor"]
    )
    for request in (aborted, survivor):
        scheduler.add_request(request)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert first.layered_prefill_plan.group_id == 0
    assert second.layered_prefill_plan.group_id == 1

    scheduler.finish_requests(aborted.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(first, _cohort_wavefront_output(first))
    continued = scheduler.schedule()
    plan = continued.layered_prefill_plan
    assert plan is not None
    assert plan.group_id > second.layered_prefill_plan.group_id
    assert plan.prefill_req_ids == (survivor.request_id,)
    assert aborted.request_id not in scheduler.requests


def test_boundary_batch_all_aborted_keeps_last_submission_fence():
    from tests.v1.core.utils import create_requests

    from vllm.v1.request import RequestStatus

    scheduler, _ = _decode_first_boundary_scheduler(
        groups=8, budget=12, chunk_threshold=4
    )
    requests = create_requests(
        num_requests=2, num_tokens=9, max_tokens=4, req_ids=["abort0", "abort1"]
    )
    for request in requests:
        scheduler.add_request(request)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert second.layered_prefill_plan is not None
    assert not second.layered_prefill_plan.is_final_group
    scheduler.finish_requests(
        [request.request_id for request in requests], RequestStatus.FINISHED_ABORTED
    )
    assert scheduler.deferred_frees

    scheduler.update_from_output(first, _cohort_wavefront_output(first))
    assert scheduler.deferred_frees
    scheduler.update_from_output(second, _cohort_wavefront_output(second))
    assert not scheduler.deferred_frees
    assert scheduler.processed_step_seq == scheduler.sched_step_seq


def test_boundary_batch_respects_max_num_seqs_without_falling_back():
    from tests.v1.core.utils import create_requests

    scheduler, decodes = _decode_first_boundary_scheduler(
        groups=4, budget=8, num_decodes=2, max_num_seqs=3
    )
    first, second = create_requests(
        num_requests=2, num_tokens=3, max_tokens=4, req_ids=["slot0", "slot1"]
    )
    scheduler.add_request(first)
    scheduler.add_request(second)

    output = scheduler.schedule()
    plan = output.layered_prefill_plan
    assert plan is not None
    assert plan.prefill_req_ids == (first.request_id,)
    assert second.request_id not in output.num_scheduled_tokens
    assert len(scheduler.running) == len(decodes) + 1


def test_boundary_batch_pause_all_stops_and_pause_new_resumes_existing_cohort():
    from tests.v1.core.utils import create_requests

    from vllm.v1.core.sched.interface import PauseState

    scheduler, _ = _decode_first_boundary_scheduler(
        groups=8, budget=12, chunk_threshold=4
    )
    active = create_requests(
        num_requests=2, num_tokens=9, max_tokens=4, req_ids=["active0", "active1"]
    )
    for request in active:
        scheduler.add_request(request)
    first = scheduler.schedule()
    first_plan = first.layered_prefill_plan
    assert first_plan is not None

    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    stopped = scheduler.schedule()
    assert stopped.layered_prefill_plan is None
    assert not set(first_plan.prefill_req_ids) & set(stopped.num_scheduled_tokens)

    (newcomer,) = create_requests(
        num_requests=1, num_tokens=3, max_tokens=4, req_ids=["newcomer"]
    )
    scheduler.add_request(newcomer)
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    resumed = scheduler.schedule()
    resumed_plan = resumed.layered_prefill_plan
    assert resumed_plan is not None
    assert resumed_plan.cohort_id == first_plan.cohort_id
    assert resumed_plan.prefill_req_ids == first_plan.prefill_req_ids
    assert resumed_plan.group_start == first_plan.group_end
    assert newcomer.request_id not in resumed.num_scheduled_tokens


def _running_prefill_ahead_of_ready_decode_scheduler():
    """Leave an older partial P before a ready D in the running queue."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=4,
        max_model_len=16384,
        long_prefill_token_threshold=2,
        use_v2_model_runner=True,
        async_scheduling=False,
    )
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.parallel_config.tensor_parallel_size = 2
    scheduler.use_pp = True
    (prefill,) = create_requests(
        num_requests=1, num_tokens=8, max_tokens=4, req_ids=["older_prefill"]
    )
    (decode,) = create_requests(
        num_requests=1, num_tokens=2, max_tokens=32, req_ids=["ready_decode"]
    )
    scheduler.add_request(prefill)
    scheduler.add_request(decode)
    native = scheduler.schedule()
    assert native.num_scheduled_tokens == {
        prefill.request_id: 2,
        decode.request_id: 2,
    }
    result = _empty_runner_output(list(native.num_scheduled_tokens))
    result.sampled_token_ids[result.req_id_to_index[decode.request_id]] = [1000]
    scheduler.update_from_output(native, result)
    assert prefill.num_computed_tokens == 2 and prefill.num_output_tokens == 0
    assert decode.num_output_tokens == 1

    scheduler.scheduler_config.long_prefill_token_threshold = 0
    _enable_layered_scheduler(
        scheduler,
        allowed_num_groups=[4],
        boundary_policy="pp_transition",
        decode_first=True,
        max_prefill_batch_size=1,
        max_chunk_tokens=4,
    )
    scheduler.defer_block_free = True
    return scheduler, prefill, decode


def test_decode_first_reorders_ready_decode_ahead_of_older_running_prefill():
    scheduler, prefill, decode = _running_prefill_ahead_of_ready_decode_scheduler()

    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens)[:2] == [
        decode.request_id,
        prefill.request_id,
    ]
    assert output.num_scheduled_tokens[decode.request_id] == 1
    assert output.num_scheduled_tokens[prefill.request_id] == 2
    assert output.total_num_scheduled_tokens == 3


def test_decode_first_priority_kv_failure_does_not_evict_selected_decode(monkeypatch):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy

    scheduler, prefill, decode = _running_prefill_ahead_of_ready_decode_scheduler()
    scheduler.policy = SchedulingPolicy.PRIORITY
    prefill.priority = -10
    decode.priority = 10
    allocate_slots = scheduler.kv_cache_manager.allocate_slots

    def fail_prefill_allocation(request, *args, **kwargs):
        if request is prefill:
            return None
        return allocate_slots(request, *args, **kwargs)

    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", fail_prefill_allocation
    )
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {decode.request_id: 1}
    assert decode.request_id not in output.preempted_req_ids
    assert prefill.request_id in output.preempted_req_ids
    assert decode in scheduler.running
