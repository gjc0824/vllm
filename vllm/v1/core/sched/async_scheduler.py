# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            plan = scheduler_output.layered_prefill_plan
            if (
                plan is not None
                and plan.boundary_mode is not None
                and req_id in plan.prefill_req_ids
            ):
                # The boundary hook runs inside the base scheduler for first
                # admission, and directly for later cohort intervals.
                continue
            self._update_after_scheduled_decode(request, scheduler_output)

    def _update_after_scheduled_decode(
        self, request: Request, scheduler_output: SchedulerOutput
    ) -> None:
        # Layered scheduling can append Decode rows after _schedule_regular
        # has run this bookkeeping for the original rows. Apply it exactly
        # once to each appended row, without advancing the original rows again.
        if request.is_prefill_chunk:
            return
        scheduler_output.pending_structured_output_tokens |= (
            request.use_structured_output and request.num_output_placeholders > 0
        )
        # Diffusion has no AR bonus token, only the canvas/spec tokens.
        cur_num_spec_tokens = len(
            scheduler_output.scheduled_spec_decode_tokens.get(request.request_id, ())
        )
        request.num_output_placeholders += (
            self.num_sampled_tokens_per_step + cur_num_spec_tokens
        )
        # The worker replaces these with the actual draft token ids.
        request.spec_token_ids = self._spec_token_placeholders

        if self.use_v2_model_runner:
            request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_after_layered_schedule(
        self, request: Request, scheduler_output: SchedulerOutput
    ) -> None:
        super()._update_after_layered_schedule(request, scheduler_output)
        plan = scheduler_output.layered_prefill_plan
        if plan is not None and plan.boundary_mode is not None:
            # The base method dispatched to our boundary hook already.
            return
        self._update_after_layered_sample(request, scheduler_output)

    def _update_after_boundary_schedule(
        self, request: Request, scheduler_output: SchedulerOutput
    ) -> None:
        super()._update_after_boundary_schedule(request, scheduler_output)
        self._update_after_layered_sample(request, scheduler_output)

    def _update_after_layered_sample(
        self, request: Request, scheduler_output: SchedulerOutput
    ) -> None:
        plan = scheduler_output.layered_prefill_plan
        if (
            plan is None
            or not plan.is_sampling_step
            or (
                plan.sampling_req_ids is not None
                and request.request_id not in plan.sampling_req_ids
            )
        ):
            # Intermediate layer groups deliver no output tokens.
            return
        # The layered request is appended to num_scheduled_tokens after
        # _update_after_schedule ran inside _schedule_regular, so the
        # placeholder bookkeeping there never sees it. Account for the first
        # sampled token here, matching the async protocol, or
        # update_from_output would drive num_output_placeholders below zero.
        request.num_output_placeholders += self.num_sampled_tokens_per_step
        if self.use_v2_model_runner:
            request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
