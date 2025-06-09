# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Profiling-based Dynamic Chunk Size Predictor.

This module implements a dynamic chunk sizing strategy based on profiling prefill
latency and fitting a quadratic model. Inspired by SGLang's PP + Dynamic Chunk.

The approach:
1. Profile: Run forward passes with different chunk sizes to measure latency
2. Fit: Use quadratic model f(l) = a*l^2 + b*l + c to fit the latency data
3. Predict: Given current history_len, solve for chunk size that achieves target latency
"""

import math
import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c
    
    Given a target latency T and current history length L, predicts next chunk size x
    such that: f(L+x) - f(L) = T
    
    This expands to the quadratic equation: a*x^2 + (2aL+b)*x - T = 0
    """

    def __init__(self, smooth_factor: float = 0.8):
        # seq_len fit
        self.quadratic_coeff_a: float = 0.0
        self.linear_coeff_b: float = 0.0
        self.constant_coeff_c: float = 0.0
        # chunked fit
        self.quadratic_chunk_a: float = 0.0
        self.linear_chunk_b: float = 0.0
        self.constant_chunk_c: float = 0.0

        self.target_latency: Optional[float] = None
        self.is_ready: bool = False
        self.with_history_ready: bool = False
        self.smooth_factor = smooth_factor
        self.min_chunk = 8192

        self.history_fitted = False

    def fit(self, seq_lens: List[int], latencies: List[float]) -> bool:
        """
        Fit quadratic coefficients f(l) = al^2 + bl + c from data points.
        
        Returns:
            True if fitting succeeded, False otherwise
        """
        L = np.array(seq_lens, dtype=np.float64)
        T = np.array(latencies, dtype=np.float64)

        if len(L) < 8:
            logger.warning(
                "Not enough data points for quadratic fitting (%d < 8)", len(L)
            )
            return False

        # Build design matrix: [l^2, l, 1]
        X = np.column_stack([L * L, L, np.ones_like(L)])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, T, rcond=None)
            fitted_a = float(coeffs[0])
            fitted_b = float(coeffs[1])
            fitted_c = float(coeffs[2])
        except np.linalg.LinAlgError as e:
            logger.warning("Failed to fit quadratic model: %s", e)
            return False

        # Validate: 'a' should be positive for O(n^2) attention
        if fitted_a < 0:
            logger.warning(
                "Fitted a=%.2e is not positive. Setting a=1e-9.", fitted_a
            )
            fitted_a = 1e-9

        if fitted_b < 0:
            logger.warning(
                "Fitted b=%.2e is not positive. Setting b=0.0.", fitted_b
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            "[ProfilingChunk] Fitted: a=%.2e, b=%.2e, c=%.2e",
            fitted_a, fitted_b, fitted_c
        )
        return True
    
    def fit_chunk(self, chunked_data) -> bool:
        """
        Fit time with chunks, f(C,H) = a*C(C+H) + b*C + c*H from data points.
        
        Returns:
            True if fitting succeeded, False otherwise
        """
        L = len(chunked_data)
        if L < 5:
            logger.warning(
                "Not enough data points for chunked data fitting (%d < 5)", len(L)
            )
            return False
        if L > 30:
            self.history_fitted = True
            return False
        chunked_data_array = np.array(chunked_data)
        execute_time = chunked_data_array[:, -1]
        input_x = chunked_data_array[:, :-1]

        try:
            params, _, _, _ = np.linalg.lstsq(input_x, execute_time, rcond=None)
            fitted_a = float(params[0])
            fitted_b = float(params[1])
            fitted_c = float(params[2])
        except np.linalg.LinAlgError as e:
            logger.warning("Failed to fit quadratic model: %s", e)
            return False

        if fitted_a < 0:
            logger.warning(
                "Fitted a=%.2e is not positive. Setting a=1e-9.", fitted_a
            )
            fitted_a = 1e-9

        if fitted_b < 0:
            logger.warning(
                "Fitted b=%.2e is not positive. Setting b=0.0.", fitted_b
            )
            fitted_b = 0.0
        self.quadratic_chunk_a = fitted_a
        self.linear_chunk_b = fitted_b
        self.constant_chunk_c = fitted_c

        logger.info(
            "[ProfilingChunk With History] Fitted: a=%.2e, b=%.2e, c=%.2e",
            fitted_a, fitted_b, fitted_c
        )
        return True
        

    def set_target_latency(self, base_chunk_size: int) -> None:
        """Set target latency based on base chunk size."""
        def f(l: float) -> float:
            return (
                self.quadratic_coeff_a * l * l
                + self.linear_coeff_b * l
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)
        if self.target_latency <= 0:
            self.target_latency = 1.0  # Fallback to 1ms

        logger.info(
            "[ProfilingChunk] Target latency: %.2f ms (base_chunk=%d)",
            self.target_latency, base_chunk_size
        )

    def predict(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(L+x) - f(L) = target_latency.
        """
        if not self.is_ready or self.target_latency is None:
            return None

        if self.quadratic_coeff_a <= 0:
            return None

        # Solve: a*x^2 + (2aL+b)*x - T = 0
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        x = (-B + sqrt_disc) / (2 * A)

        if x <= 0:
            return None

        # Apply smoothing
        smoothed = base_chunk_size + self.smooth_factor * (x - base_chunk_size)
        chunk_size = max(int(smoothed), self.min_chunk)

        # Align to page size
        align = max(page_size, 64)
        chunk_size = (chunk_size // align) * align
        if chunk_size < align:
            chunk_size = align

        # Apply max constraint
        max_allowed = context_len - history_len
        if max_chunk_size:
            max_allowed = min(max_allowed, max_chunk_size)
        chunk_size = min(chunk_size, max_allowed)
        print(">>>>>>>>>>>>>>> chunk_size from profiling", chunk_size)

        # Re-align
        chunk_size = (chunk_size // align) * align
        return chunk_size if chunk_size >= align else None
    
    def predict_with_history(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(C,H) = target_latency.
        """
        if not self.is_ready or self.target_latency is None:
            return None
        
        if not self.with_history_ready:
            return None

        if self.quadratic_chunk_a <= 0:
            return None

        # Solve: a*C(C+H) + b*C + c*H - T = 0
        # a*C^2 + (a*H + b)*C + c*H - T = 0
        # (a*(C+H)*C + b*(C+H) + c - T = 0
        # a*C^2 + (a*H + b)*C + b*H + c - T = 0
        A = self.quadratic_chunk_a
        B = self.quadratic_chunk_a * history_len + self.linear_chunk_b
        C = self.linear_chunk_b * history_len + self.constant_chunk_c - self.target_latency

        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        x = (-B + sqrt_disc) / (2 * A)

        if x <= 0:
            return None

        # Apply smoothing
        print(">>>>>>>>>>>>>>> chunk_size from history profiling", x)
        smoothed = base_chunk_size + self.smooth_factor * (x - base_chunk_size)
        chunk_size = max(int(smoothed), self.min_chunk)

        # Align to page size
        align = max(page_size, 64)
        chunk_size = (chunk_size // align) * align
        if chunk_size < align:
            chunk_size = align

        # Apply max constraint
        max_allowed = context_len - history_len
        if max_chunk_size:
            max_allowed = min(max_allowed, max_chunk_size)
        chunk_size = min(chunk_size, max_allowed)

        # Re-align
        chunk_size = (chunk_size // align) * align
        return chunk_size if chunk_size >= align else None


class ProfilingChunkManager:
    """
    Manager for profiling-based dynamic chunk sizing.
    
    Handles the profiling process and maintains the ChunkSizePredictor.
    """

    def __init__(
        self,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_prefill_tokens: Optional[int] = None,
    ):
        self.base_chunk_size = base_chunk_size
        self.page_size = page_size
        self.context_len = context_len
        self.max_prefill_tokens = max_prefill_tokens
        self.chunked_fit_data = []
        
        smooth_factor = float(os.environ.get(
            "VLLM_PROFILING_CHUNK_SMOOTH_FACTOR", "1.0"
        ))
        self.predictor = ChunkSizePredictor(smooth_factor)
        self._profiling_done = False

    @property
    def is_ready(self) -> bool:
        return self._profiling_done and self.predictor.is_ready
    
    @property
    def history_ready(self) -> bool:
        return self.is_ready and self.predictor.with_history_ready

    def run_profiling(
        self,
        forward_fn: Callable[[int], None],
        sync_fn: Callable[[], None],
        num_samples: int = 64,
    ) -> bool:
        """
        Run profiling by executing forward passes with different chunk sizes.
        
        Args:
            forward_fn: Function that runs forward pass with given num_tokens
            sync_fn: Function to synchronize device (e.g., torch.cuda.synchronize)
            num_samples: Number of samples to collect
            
        Returns:
            True if profiling succeeded
        """
        logger.info("[ProfilingChunk] Starting profiling with %d samples...", num_samples)
        
        seq_lens: List[int] = []
        latencies: List[float] = []

        for i in range(num_samples):
            chunk_size = int(
                self.base_chunk_size * 1.25
                - i * (self.base_chunk_size * 1.25 / num_samples)
            )
            if chunk_size <= 0:
                break

            sync_fn()
            start = time.perf_counter()
            
            try:
                forward_fn(chunk_size)
            except Exception as e:
                logger.debug("Forward failed for chunk=%d: %s", chunk_size, e)
                continue
            
            sync_fn()
            latency_ms = (time.perf_counter() - start) * 1000

            seq_lens.append(chunk_size)
            latencies.append(latency_ms)

        if len(seq_lens) < 8:
            logger.warning(
                "[ProfilingChunk] Profiling failed: only %d samples collected",
                len(seq_lens)
            )
            return False

        logger.info(
            "[ProfilingChunk] Collected %d samples. Latency range: [%.2f, %.2f] ms",
            len(seq_lens), min(latencies), max(latencies)
        )

        # Fit model
        if not self.predictor.fit(seq_lens, latencies):
            return False

        self.predictor.set_target_latency(self.base_chunk_size)
        self.predictor.is_ready = True
        self._profiling_done = True
        return True

    def predict_chunk_size(self, history_len: int) -> Optional[int]:
        """Predict optimal chunk size for given history length."""
        if not self.is_ready:
            return None

        if history_len == 0 or not self.history_ready:
            predict_func = self.predictor.predict
        else:
            predict_func = self.predictor.predict_with_history
        return predict_func(
            history_len=history_len,
            base_chunk_size=self.base_chunk_size,
            page_size=self.page_size,
            context_len=self.context_len,
            max_chunk_size=self.max_prefill_tokens,
        )

    def record_batch_execution_time(self, request_chunks: list, elapsed_time: float):
        # T = sum(a*(C+H)*C + b*C + d*H) T = sum(a*(C+H)*C + b*(C+H) + d) 
        # x1 = sum((C+H) * C)
        # x2 = sum(C) sum(C+H)
        # x3 = sum(H) sum(1)
        # T = a*X1 + b*X2 + d*X3
        x1 = x2 = x3 = 0
        for C, H in request_chunks:
            x1 += (C + H) * C
            x2 += C + H
            x3 += 1
        self.chunked_fit_data.append([x1, x2, x3, elapsed_time *  1000])
        print(">>>>>>>>>>>>> chunk fit data", self.chunked_fit_data)
        # Fit model
        if not self.predictor.fit_chunk(self.chunked_fit_data):
            return False

        self.predictor.with_history_ready = True
        return True
        

    def broadcast_and_init(
        self,
        seq_lens: List[int],
        latencies: List[float],
        src_rank: int = 0,
    ) -> bool:
        """
        Broadcast profiling data from src_rank and initialize predictor.
        Used in distributed settings.
        """
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                data = [seq_lens, latencies]
                dist.broadcast_object_list(data, src=src_rank)
                seq_lens, latencies = data[0], data[1]
        except Exception:
            pass

        if len(seq_lens) < 8:
            return False

        if not self.predictor.fit(seq_lens, latencies):
            return False

        self.predictor.set_target_latency(self.base_chunk_size)
        self.predictor.is_ready = True
        self._profiling_done = True
        return True

