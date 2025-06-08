from abc import ABC
from enum import Enum
from typing import Optional


class BudgetType(Enum):
    """Enum for budget type."""
    
    TOKEN = "token"
    CL = "computational_load"
    TIME = "time"  # 基于执行时间的预算


class TokenBudget(ABC):
    """ Unified management of the token budget
    
    Supports three budget types:
    - TOKEN: Based on number of tokens
    - CL: Based on computational load (FLOPs)
    - TIME: Based on execution time (requires EKF calibration)
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

        self.type = scheduler.budget_type
        self.pp_size = scheduler.parallel_config.pipeline_parallel_size

        self.max_num_scheduled_tokens = scheduler.max_num_scheduled_tokens
        self.token_budget = self.max_num_scheduled_tokens

        self.attn_estimator = scheduler.attn_estimator

        # FLOPs budget
        if self.attn_estimator is not None:
            self.max_num_scheduled_flops = self.attn_estimator.calculate_current_flops(
                chunk_size=self.max_num_scheduled_tokens,
                hist_seq_len=0
            )
        else:
            self.max_num_scheduled_flops = 0
        self.flops_budget = self.max_num_scheduled_flops

        # Time budget (seconds)
        # 初始时间预算设为0，EKF校准后由scheduler设置
        self.max_time_budget: float = 0.0
        self.time_budget: float = 0.0

    def set_max_time_budget(self, max_time: float) -> None:
        """Set the maximum time budget (called after EKF calibration)."""
        self.max_time_budget = max_time
        self.time_budget = max_time

    def update(self):
        """Reset budgets at the start of each scheduling step."""
        if self.type == BudgetType.TOKEN.value:
            self.token_budget = self.max_num_scheduled_tokens
        elif self.type == BudgetType.CL.value:
            self.flops_budget = self.max_num_scheduled_flops
            self.token_budget = self.max_num_scheduled_tokens
        elif self.type == BudgetType.TIME.value:
            self.time_budget = self.max_time_budget
            self.token_budget = self.max_num_scheduled_tokens
            # 也重置flops_budget作为fallback
            self.flops_budget = self.max_num_scheduled_flops

    def has_running(self):
        """Check if there's remaining budget for running requests."""
        if self.type == BudgetType.TOKEN.value:
            return self.token_budget > 0
        elif self.type == BudgetType.CL.value:
            return self.flops_budget > 0 and self.token_budget > 0
        elif self.type == BudgetType.TIME.value:
            return self.token_budget > 0
        return self.token_budget > 0

    def has_waiting(self):
        """Check if there's remaining budget for waiting requests."""
        if self.type == BudgetType.TOKEN.value:
            return self.token_budget > 0
        elif self.type == BudgetType.CL.value:
            return self.flops_budget > 0 and self.token_budget > 0
        elif self.type == BudgetType.TIME.value:
            return self.token_budget > 0
        return self.token_budget > 0

    def get(self, computed_prompt: Optional[bool] = False):
        """Get remaining token budget."""
        return self.token_budget

    def get_flops(self):
        """Get remaining FLOPs budget."""
        return self.flops_budget

    def get_max_flops(self):
        """Get maximum FLOPs budget."""
        return self.max_num_scheduled_flops

    def get_time(self) -> float:
        """Get remaining time budget (seconds)."""
        return self.time_budget

    def get_max_time(self) -> float:
        """Get maximum time budget (seconds)."""
        return self.max_time_budget

    def get_consumed_time(self) -> float:
        """Get consumed time in this scheduling step (seconds)."""
        return self.max_time_budget - self.time_budget

    def consume(self, num_new_tokens: int, num_computed_tokens: int, 
                computed_prompt: Optional[bool] = False):
        """Consume budget for scheduling tokens."""
        if self.type == BudgetType.TOKEN.value:
            self.token_budget -= num_new_tokens
        elif self.type == BudgetType.CL.value:
            if self.attn_estimator is not None:
                num_new_flops = self.attn_estimator.calculate_current_flops(
                    chunk_size=num_new_tokens,
                    hist_seq_len=num_computed_tokens
                )
                self.flops_budget -= num_new_flops
            self.token_budget -= num_new_tokens
        elif self.type == BudgetType.TIME.value:
            if self.attn_estimator is not None:
                # 使用EKF校准后的参数计算预估时间
                estimated_time = self.attn_estimator.compute_time_from_params(
                    num_new_tokens, num_computed_tokens
                )
                self.time_budget -= estimated_time
                # 同时更新flops_budget作为fallback
                num_new_flops = self.attn_estimator.calculate_current_flops(
                    chunk_size=num_new_tokens,
                    hist_seq_len=num_computed_tokens
                )
                self.flops_budget -= num_new_flops
            self.token_budget -= num_new_tokens

    def rollback(self, num_new_tokens: int, num_computed_tokens: int, 
                 computed_prompt: Optional[bool] = False):
        """Rollback budget when a request cannot be scheduled."""
        if self.type == BudgetType.TOKEN.value:
            self.token_budget += num_new_tokens
        elif self.type == BudgetType.CL.value:
            if self.attn_estimator is not None:
                num_new_flops = self.attn_estimator.calculate_current_flops(
                    chunk_size=num_new_tokens,
                    hist_seq_len=num_computed_tokens
                )
                self.flops_budget += num_new_flops
            self.token_budget += num_new_tokens
        elif self.type == BudgetType.TIME.value:
            if self.attn_estimator is not None:
                estimated_time = self.attn_estimator.compute_time_from_params(
                    num_new_tokens, num_computed_tokens
                )
                self.time_budget += estimated_time
                num_new_flops = self.attn_estimator.calculate_current_flops(
                    chunk_size=num_new_tokens,
                    hist_seq_len=num_computed_tokens
                )
                self.flops_budget += num_new_flops
            self.token_budget += num_new_tokens

    def verify(self):
        """Verify budget constraints are satisfied."""
        if self.type == BudgetType.TOKEN.value:
            assert self.token_budget >= 0
        elif self.type == BudgetType.CL.value:
            assert self.token_budget >= 0
        elif self.type == BudgetType.TIME.value:
            assert self.token_budget >= 0
            # time_budget可以略微为负（由于预估误差），不做严格断言

    def is_time_budget_mode(self) -> bool:
        """Check if currently using time budget mode."""
        return (self.type == BudgetType.TIME.value and 
                self.max_time_budget > 0 and
                self.attn_estimator is not None and
                getattr(self.attn_estimator, '_ekf_initialized', False))
