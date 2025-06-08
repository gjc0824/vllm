"""
Model Configuration P-Value Estimator
Estimate non-attention and attention calculation ratio for different model architectures

Supports dynamic calibration using Extended Kalman Filter (EKF) to update
FLOPs estimation parameters based on actual execution time.
"""

import json
import argparse
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import logging
import numpy as np


# Global logger: prefer vllm logger if available, else fallback to std logging
try:
    from vllm.logger import init_logger as _vllm_init_logger  # type: ignore
    logger = _vllm_init_logger(__name__)
except Exception:  # noqa: BLE001
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class BaseModelEstimator(ABC):
    """Abstract base class for model p-value estimators"""
    
    def __init__(self, config: Dict):
        """
        Initialize estimator with model config
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        # Predefine attributes that may be populated by subclasses
        self.effective_kv_dim = None
        self.effective_q_dim = None
        self._parse_common_config()
        self._parse_architecture_specific_config()
        self._validate_config()
        # Baseline and attention-equivalent tokens configured from scheduler
        self.attn_equiv_baseline_c: int = 0
        self.attn_equiv_tokens: int = 0
        
        # FLOPs estimation parameters: FLOPs = a·(C+H) + b·C·(C+H) + d
        self.a = 0.0
        self.b = 0.0
        self.d = 0.0
        
        # Time calibration: cost_time = k * (a·(C+H) + b·C·(C+H) + d)
        self.k = 1e-12  # Initial scale factor (FLOPs to seconds)
        
        # Extended Kalman Filter state
        self._ekf_initialized = False
        self._ekf_state = None  # [a, b, d, k]
        self._ekf_covariance = None  # 4x4 covariance matrix
        
        # EKF tuning parameters
        self._ekf_process_noise_q = 1e-6  # Process noise variance
        self._ekf_measurement_noise_r = 1e-4  # Measurement noise variance
        self._ekf_min_samples_for_init = 1  # Min samples for k initialization
        self._ekf_sample_buffer: List[Tuple[int, int, float]] = []  # (C, H, time)
    
    def _parse_common_config(self):
        """Parse common configuration parameters"""
        self.model_type = self.config.get('model_type', 'unknown')
        self.hidden_size = self.config.get('hidden_size')
        # Some architectures may override or refine this later.
        self.intermediate_size = self.config.get('intermediate_size')
        self.num_attention_heads = self.config.get('num_attention_heads')
        self.num_key_value_heads = self.config.get('num_key_value_heads', self.num_attention_heads)
        self.num_hidden_layers = self.config.get('num_hidden_layers')
        self.vocab_size = self.config.get('vocab_size', 32000)
        # Optional calibration factor to account for softmax/RoPE and kernel overheads
        self.attn_flops_scale = float(self.config.get('attn_flops_scale', 1.0))
        
        # Calculate common derived values
        if self.hidden_size and self.num_attention_heads:
            self.head_dim = self.hidden_size // self.num_attention_heads
            self.gqa_ratio = self.num_attention_heads // self.num_key_value_heads
    
    @abstractmethod
    def _parse_architecture_specific_config(self):
        """Parse architecture-specific configuration parameters"""
        raise NotImplementedError
    
    def _validate_config(self):
        """Validate required configuration parameters"""
        if not all([self.hidden_size, self.num_attention_heads, self.num_hidden_layers]):
            raise ValueError(f"Missing required parameters for {self.model_type} model")
    
    @abstractmethod
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """
        Calculate attention FLOPs
        
        Args:
            chunk_size: Input chunk size
            seq_len: Total sequence length
            
        Returns:
            (projection_flops, attention_computation_flops)
        """
        raise NotImplementedError
    
    @abstractmethod
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """
        Calculate FFN FLOPs
        
        Args:
            chunk_size: Input chunk size  
            layer_idx: Layer index (for models with varying layer types)
            
        Returns:
            FFN FLOPs
        """
        raise NotImplementedError
    
    def calculate_other_flops(self, chunk_size: int) -> int:
        """Calculate other FLOPs (LayerNorm, residual connections, etc.)"""
        # Standard: 2 LayerNorms per layer
        norm_flops = chunk_size * self.hidden_size * 2
        
        # Residual connections
        residual_flops = chunk_size * self.hidden_size * 2
        
        return norm_flops + residual_flops

    # -------------------------
    # Shared FLOPs helpers
    # -------------------------
    def _standard_attention_flops(self, 
                                  chunk_size: int, 
                                  seq_len: int, 
                                  *, 
                                  kv_dim: Optional[int] = None, 
                                  effective_seq_len: Optional[int] = None
                                  ) -> Tuple[int, int]:
        """
        Standard scaled dot-product attention FLOPs with GQA.

        Args:
            chunk_size: Number of tokens processed this step.
            seq_len: Total attention context length (KV cache + chunk).
            kv_dim: Dimension used for K/V projections per token.
                    Defaults to hidden_size // gqa_ratio.
            effective_seq_len: Effective attention length if using sliding
                               window or other sparsity; defaults to seq_len.
        Returns:
            (projection_flops, attention_compute_flops)
        """
        kv_dim = kv_dim if kv_dim is not None else (self.hidden_size // self.gqa_ratio)
        eff_len = effective_seq_len if effective_seq_len is not None else seq_len

        q_flops = chunk_size * self.hidden_size * self.hidden_size
        kv_flops = chunk_size * self.hidden_size * kv_dim * 2
        score_flops = chunk_size * eff_len * self.hidden_size
        value_flops = chunk_size * eff_len * self.hidden_size
        out_flops = chunk_size * self.hidden_size * self.hidden_size

        return q_flops + kv_flops + out_flops, score_flops + value_flops

    def _ffn_swiglu_like_flops(self, chunk_size: int, *, intermediate_size: Optional[int] = None) -> int:
        """
        FLOPs for FFN with gate+up and down projections (SwiGLU/GeGLU).
        The activation doesn't change matmul FLOPs.
        """
        inter_size = intermediate_size if intermediate_size is not None else self.intermediate_size
        # Support configs that use `ffn_dim` (e.g., OPT) without guessing.
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')
        if inter_size is None or self.hidden_size is None:
            raise ValueError("Missing FFN dimension: expected 'intermediate_size' or 'ffn_dim' in config")
        up_gate_flops = chunk_size * self.hidden_size * inter_size * 2
        down_flops = chunk_size * inter_size * self.hidden_size
        return up_gate_flops + down_flops

    # -------------------------
    # Dynamic chunk sizing helper (DCPP-style)
    # -------------------------
    def configure_baseline_from_scheduler(self, scheduler_config) -> None:
        """Configure baseline chunk and attention-equivalent tokens using scheduler config."""
        baseline_c = getattr(scheduler_config, 'long_prefill_token_threshold', None)
        if baseline_c is None or baseline_c <= 0:
            baseline_c = 2048
        self.attn_equiv_baseline_c = int(baseline_c)
        self.attn_equiv_tokens = int(self.estimate_flops_ratio_scaled(self.attn_equiv_baseline_c, 0))

    def compute_chunk_size_with_overhead(self,
                                         hist_seq_len: int,
                                         seq_len: int,
                                         chunk_size: int,
                                         block_size: int) -> tuple[int, int]:
        """
        Compute shortened chunk to offset long-KV attention overhead.

        Returns (dcpp_chunk, scheduled_chunk):
        - dcpp_chunk: actual reduced chunk length to run this step (block-aligned)
        - scheduled_chunk: the baseline chunk length used for budgeting/accounting
        """
        # Must be configured once by the scheduler
        assert self.attn_equiv_tokens > 0, "Call configure_baseline_from_scheduler() before chunk computation."

        target_time = chunk_size * (self.attn_equiv_tokens + chunk_size)
        discriminant = (self.attn_equiv_tokens + hist_seq_len) ** 2 + 4 * target_time
        dcpp_chunk = int((-(self.attn_equiv_tokens + hist_seq_len) + discriminant ** 0.5) / 2)
        # align to block size
        dcpp_chunk = (dcpp_chunk + block_size - 1) // block_size * block_size
        scheduled_chunk = dcpp_chunk
        # do not exceed remaining tokens
        dcpp_chunk = min(dcpp_chunk, seq_len - hist_seq_len)
        return dcpp_chunk, scheduled_chunk
    
    def calculate_current_flops(self, chunk_size:int, hist_seq_len:int, layer_idx: Optional[int] = None) -> int:
        seq_len = hist_seq_len + chunk_size
        
        proj_flops, attn_flops = self.calculate_attention_flops(chunk_size, seq_len)
        # Apply global attention calibration to better match wall-clock
        attn_flops = attn_flops * self.attn_flops_scale
        ffn_flops = self.calculate_ffn_flops(chunk_size, layer_idx)
        other_flops = self.calculate_other_flops(chunk_size)

        total_flops = ffn_flops + other_flops + proj_flops + attn_flops
        return total_flops

    def compute_chunk_size_with_flops(self,
                                      hist_seq_len: int,
                                      target_flops: int,
                                      block_size: int,
                                      layer_idx: Optional[int] = None) -> int:
        """                            
        FLOPs = a·(C+H) + b·C·(C+H) + d
        C = chunk_size
        H = hist_seq_len
        a = 线性项系数(与序列长度相关)
        b = 注意力二次项系数
        d = 常数项(固定开销)
        
        通过联立三元一次方程求解a, b, d参数
        
        Args:
            hist_seq_len: KV缓存长度
            target_flops: 目标FLOPs
            block_size: 块大小对齐
            layer_idx: 特定层索引
            
        Returns:
            chunk_size: 估算的chunk大小
            scheduled_flops: 实际消耗的flops
        """
        if self.a == 0 and self.b == 0 and self.d == 0:
            self._init_abd_params(hist_seq_len, block_size, layer_idx)
    
        a = self.a
        b = self.b
        d = self.d
        H = hist_seq_len

        if b <= 0 or a <= 0:
            # a和b应该为正，如果为负说明模型不合适，使用线性近似
            test_c1 = block_size * 10
            test_c2 = block_size * 20
            
            f1 = self.calculate_current_flops(test_c1, hist_seq_len, layer_idx)
            f2 = self.calculate_current_flops(test_c2, hist_seq_len, layer_idx)
            
            linear_coeff = (f2 - f1) / (test_c2 - test_c1)
            chunk_size = target_flops / linear_coeff if linear_coeff > 0 else block_size
        else:
            # 求解: a·(C+H) + b·C·(C+H) + d = target_flops
            # 展开: a·C + a·H + b·C² + b·C·H + d = target_flops
            # 整理: b·C² + (a + b·H)·C + (a·H + d - target_flops) = 0
            A = b
            B = a + b * H
            C_coeff = d - target_flops
            
            discriminant = B**2 - 4 * A * C_coeff
            if discriminant < 0:
                # 无解，使用线性近似
                chunk_size = (target_flops - d) / (a + b * H) if (a + b * H) > 0 else block_size
            else:
                chunk_size = (-B + discriminant**0.5) / (2 * A)
        print(f">>>>>>>>>>>> flop schedule chunk_size {chunk_size} H {H}, a {a} b {b} d {d}, target_flops {target_flops}")
        chunk_size = max(block_size, int(chunk_size))
        chunk_size = (chunk_size + block_size - 1) // block_size * block_size
        
        return chunk_size, None
    
    def _init_abd_params(self, hist_seq_len: int, block_size: int, 
                         layer_idx: Optional[int] = None) -> None:
        """
        通过联立三元一次方程求解 a, b, d 参数
        
        FLOPs = a·(C+H) + b·C·(C+H) + d
        
        使用三个测试点 (C1, C2, C3) 建立方程组：
        F1 = a·(C1+H) + b·C1·(C1+H) + d
        F2 = a·(C2+H) + b·C2·(C2+H) + d
        F3 = a·(C3+H) + b·C3·(C3+H) + d
        """
        # 选择三个不同的chunk大小作为测试点
        test_c1 = block_size * 8
        test_c2 = block_size * 16
        test_c3 = block_size * 32
        
        H = hist_seq_len
        
        # 计算三个测试点的FLOPs
        f1 = self.calculate_current_flops(test_c1, H, layer_idx)
        f2 = self.calculate_current_flops(test_c2, H, layer_idx)
        f3 = self.calculate_current_flops(test_c3, H, layer_idx)
        
        # 构建方程组系数矩阵
        # | S1  P1  1 |   | a |   | F1 |
        # | S2  P2  1 | × | b | = | F2 |
        # | S3  P3  1 |   | d |   | F3 |
        # 其中 Si = Ci + H, Pi = Ci * Si
        
        L1, L2, L3 = test_c1, test_c2, test_c3
        P1, P2, P3 = test_c1 * (test_c1 + H), test_c2 * (test_c2 + H), test_c3 * (test_c3 + H)
        
        # 使用克莱默法则求解
        # 计算主行列式 det(M)
        det_M = (L1 * (P2 - P3) - P1 * (L2 - L3) + (L2 * P3 - L3 * P2))
        
        if abs(det_M) < 1e-10:
            # 行列式接近0，方程组可能无解或有无穷解，使用简化的二元方程求解
            logger.warning("三元方程组行列式接近0，使用简化的二元求解")
            self._init_ab_params_fallback(hist_seq_len, block_size, layer_idx)
            return
        
        # det_a: 将第一列替换为 [F1, F2, F3]
        det_a = (f1 * (P2 - P3) - P1 * (f2 - f3) + (f2 * P3 - f3 * P2))
        
        # det_b: 将第二列替换为 [F1, F2, F3]  
        det_b = (L1 * (f2 - f3) - f1 * (L2 - L3) + (L2 * f3 - L3 * f2))
        
        # det_d: 将第三列替换为 [F1, F2, F3]
        det_d = (L1 * (P2 * f3 - P3 * f2) - P1 * (L2 * f3 - L3 * f2) + 
                 f1 * (L2 * P3 - L3 * P2))
        
        self.a = det_a / det_M
        self.b = det_b / det_M
        self.d = det_d / det_M
        
        # 存储测试数据用于调试
        self._init_test_data = {
            'test_c': (test_c1, test_c2, test_c3),
            'flops': (f1, f2, f3),
            'hist_seq_len': H
        }
        
        logger.debug(f"初始化FLOPs参数: a={self.a:.2e}, b={self.b:.2e}, d={self.d:.2e}")
    
    def _init_ab_params_fallback(self, hist_seq_len: int, block_size: int,
                                  layer_idx: Optional[int] = None) -> None:
        """二元方程组求解的fallback方案（忽略常数项d）"""
        test_c1 = block_size * 10
        test_c2 = block_size * 20
        
        f1 = self.calculate_current_flops(test_c1, hist_seq_len, layer_idx)
        f2 = self.calculate_current_flops(test_c2, hist_seq_len, layer_idx)
        
        H = hist_seq_len
        S1, S2 = test_c1 + H, test_c2 + H
        P1, P2 = test_c1 * S1, test_c2 * S2
        
        # 简化方程: a·S + b·P ≈ F (忽略d)
        det = S1 * P2 - S2 * P1
        if abs(det) < 1e-10:
            # 仍然无法求解，使用默认值
            self.a = f1 / S1 if S1 > 0 else 1.0
            self.b = 1e-6
            self.d = 0.0
        else:
            self.a = (f1 * P2 - f2 * P1) / det
            self.b = (S1 * f2 - S2 * f1) / det
            self.d = 0.0
    
    # -------------------------
    # 时间校准和EKF动态更新
    # -------------------------
    
    def compute_flops_from_params(self, chunk_size: int, hist_seq_len: int) -> float:
        """
        根据当前参数计算FLOPs
        FLOPs = a·(C+H) + b·C·(C+H) + d
        """
        C, H = chunk_size, hist_seq_len
        return self.a * C + self.b * C * (C + H) + self.d
    
    def compute_time_from_params(self, chunk_size: int, hist_seq_len: int) -> float:
        """
        根据当前参数计算预估执行时间
        cost_time = k * (a·(C+H) + b·C·(C+H) + d)
        """
        flops = self.compute_flops_from_params(chunk_size, hist_seq_len)
        return self.k * flops
    
    def compute_chunk_size_from_target_time(self, 
                                            hist_seq_len: int,
                                            target_time: float,
                                            base_chunk_size: int,
                                            block_size: int) -> int:
        """
        根据目标执行时间反解chunk_size
        cost_time = k * (a·(C+H) + b·C·(C+H) + d)
        
        target_time / k = a·(C+H) + b·C·(C+H) + d
        展开: b·C² + (a + b·H)·C + (a·H + d - target_time/k) = 0
        """
        if self.k <= 0:
            return block_size
            
        target_flops = target_time / self.k
        H = hist_seq_len
        a, b, d = self.a, self.b, self.d
        
        if b <= 0:
            # 线性近似
            chunk_size = (target_flops - d) / (a + b * H) if (a + b * H) > 0 else block_size
        else:
            # 二次方程求解
            A = b
            B = a + b * H
            C_coeff = d - target_flops
            
            discriminant = B**2 - 4 * A * C_coeff
            if discriminant < 0:
                chunk_size = (target_flops - d) / (a + b * H) if (a + b * H) > 0 else block_size
            else:
                chunk_size = (-B + discriminant**0.5) / (2 * A)

        chunk_size = base_chunk_size + 1.0 * (chunk_size - base_chunk_size)
        chunk_size = max(block_size, int(chunk_size))
        chunk_size = (chunk_size + block_size - 1) // block_size * block_size
        print(f">>>>>>>>>>>>>>>> time scheduler chunk_size {chunk_size}, a {self.a}, b {self.b} d {self.d} k {self.k}, target_flops {target_flops}")
        
        return chunk_size
    
    def record_execution_time(self, chunk_size: int, hist_seq_len: int, 
                               elapsed_time: float) -> None:
        """
        记录一次执行时间（单请求模式），并使用EKF更新参数
        
        注意：当一次调度多条请求时，应使用 record_batch_execution_time() 方法
        
        Args:
            chunk_size: 本次执行的chunk大小
            hist_seq_len: KV缓存长度
            elapsed_time: 实际执行时间(秒)
        """
        # 转换为batch格式，单请求作为一个元素的batch
        self.record_batch_execution_time(
            request_chunks=[(chunk_size, hist_seq_len)],
            elapsed_time=elapsed_time
        )
    
    def record_batch_execution_time(self, 
                                     request_chunks: List[Tuple[int, int]], 
                                     elapsed_time: float) -> None:
        """
        记录一次batch执行时间，并使用EKF更新参数
        
        当一次调度多条请求时，使用此方法记录总耗时。
        EKF观测方程变为batch级别：
        z_batch = Σ [ k * (a*(C_i+H_i) + b*C_i*(C_i+H_i) + d) ]
        
        Args:
            request_chunks: 请求列表，每个元素为 (chunk_size, hist_seq_len)
            elapsed_time: batch的实际总执行时间(秒)
        """
        if not request_chunks or elapsed_time <= 0:
            return
        
        # 计算总chunk_size用于样本收集
        total_chunk_size = sum(c for c, h in request_chunks)
        if total_chunk_size <= 0:
            return
        
        # 如果a,b,d还未初始化，先收集样本
        # 使用加权平均作为近似（仅用于初始化阶段）
        if self.a == 0 and self.b == 0 and self.d == 0:
            avg_hist = sum(h for c, h in request_chunks) // len(request_chunks) if request_chunks else 0
            self._ekf_sample_buffer.append((total_chunk_size, avg_hist, elapsed_time))
            if len(self._ekf_sample_buffer) >= self._ekf_min_samples_for_init:
                self._init_params_from_samples()
            return
        
        # 如果k还未初始化，根据预估FLOPs和实际时间估算k
        if not self._ekf_initialized:
            # 计算batch总预估FLOPs
            total_flops = sum(
                self.compute_flops_from_params(c, h) 
                for c, h in request_chunks
            )
            if total_flops > 0:
                self._ekf_sample_buffer.append((total_chunk_size, 0, elapsed_time))
                # 直接用总flops和总时间估算k
                k_estimate = elapsed_time / total_flops
                self._ekf_sample_buffer_k = getattr(self, '_ekf_sample_buffer_k', [])
                self._ekf_sample_buffer_k.append(k_estimate)
                
                if len(self._ekf_sample_buffer_k) >= self._ekf_min_samples_for_init:
                    # 使用中位数作为k的初始估计
                    self._ekf_sample_buffer_k.sort()
                    mid = len(self._ekf_sample_buffer_k) // 2
                    self.k = self._ekf_sample_buffer_k[mid]
                    logger.debug(f"初始化时间校准系数: k={self.k:.2e}")
                    self._ekf_sample_buffer.clear()
                    self._ekf_sample_buffer_k.clear()
                    self._init_ekf_state()
            return
        
        # EKF batch更新
        self._ekf_batch_update(request_chunks, elapsed_time)
    
    def _init_params_from_samples(self) -> None:
        """从收集的样本初始化a, b, d参数"""
        if len(self._ekf_sample_buffer) < 3:
            return
        
        # 使用前三个样本建立方程组
        samples = self._ekf_sample_buffer[:3]
        
        C1, H1, t1 = samples[0]
        C2, H2, t2 = samples[1]
        C3, H3, t3 = samples[2]
        
        # 暂时假设 k*FLOPs = time，使用相对值来估算a, b, d
        # 先用默认block_size进行初始化
        block_size = 16  # 默认值
        self._init_abd_params(H1, block_size)
    
    def _init_k_from_samples(self) -> None:
        """从收集的样本初始化k参数"""
        if len(self._ekf_sample_buffer) < self._ekf_min_samples_for_init:
            return
        
        k_estimates = []
        for chunk_size, hist_seq_len, elapsed_time in self._ekf_sample_buffer:
            flops = self.compute_flops_from_params(chunk_size, hist_seq_len)
            if flops > 0:
                k_estimates.append(elapsed_time / flops)
        
        if k_estimates:
            # 使用中位数作为k的初始估计（更鲁棒）
            k_estimates.sort()
            mid = len(k_estimates) // 2
            self.k = k_estimates[mid]
            logger.debug(f"初始化时间校准系数: k={self.k:.2e}")
        
        # 清空样本缓冲区
        self._ekf_sample_buffer.clear()
    
    def _init_ekf_state(self) -> None:
        """初始化EKF状态"""
        
        # 状态向量: [a, b, d, k]
        self._ekf_state = np.array([self.a, self.b, self.d, self.k], dtype=np.float64)
        
        # 初始协方差矩阵 (对角矩阵，较大的初始不确定性)
        self._ekf_covariance = np.diag([
            (self.a * 0.1)**2 if self.a > 0 else 1e10,  # a的方差
            (self.b * 0.1)**2 if self.b > 0 else 1e-12,  # b的方差
            (abs(self.d) * 0.1)**2 if self.d != 0 else 1e10,  # d的方差
            (self.k * 0.1)**2 if self.k > 0 else 1e-24,  # k的方差
        ])
        
        self._ekf_initialized = True
        logger.debug("EKF已初始化")
    
    def _ekf_update(self, chunk_size: int, hist_seq_len: int, 
                    elapsed_time: float) -> None:
        """
        扩展卡尔曼滤波更新步骤（单请求）
        
        观测方程: z = k * (a*(C+H) + b*C*(C+H) + d) + v
        其中 v ~ N(0, R) 是测量噪声
        
        状态向量: x = [a, b, d, k]^T
        """
        # 转换为batch格式
        self._ekf_batch_update([(chunk_size, hist_seq_len)], elapsed_time)
    
    def _ekf_batch_update(self, request_chunks: List[Tuple[int, int]], 
                          elapsed_time: float) -> None:
        """
        扩展卡尔曼滤波batch更新步骤
        
        观测方程 (batch级别): 
        z_batch = Σ_i [ k * (a*(C_i+H_i) + b*C_i*(C_i+H_i) + d) ] + v
        
        雅可比矩阵:
        dz/da = k * Σ_i (C_i + H_i)
        dz/db = k * Σ_i C_i*(C_i + H_i)
        dz/dd = k * N  (N是请求数量)
        dz/dk = Σ_i (a*(C_i+H_i) + b*C_i*(C_i+H_i) + d) = total_flops
        
        状态向量: x = [a, b, d, k]^T
        """
        if self._ekf_state is None or self._ekf_covariance is None:
            return
        
        if not request_chunks:
            return
        
        a, b, d, k = self._ekf_state
        N = len(request_chunks)
        
        # 1. 预测步骤 (假设参数缓慢变化，状态转移矩阵为单位矩阵)
        Q = np.eye(4) * self._ekf_process_noise_q
        P_pred = self._ekf_covariance + Q
        
        # 2. 计算batch级别的预测值和雅可比矩阵
        sum_C = 0.0    # Σ (C_i + H_i)
        sum_P = 0.0    # Σ C_i*(C_i + H_i)
        total_flops = 0.0
        
        for C, H in request_chunks:
            P_i = C * (C + H)
            flops_i = a * C + b * P_i + d
            
            sum_C += C
            sum_P += P_i
            total_flops += flops_i
        
        # 预测的batch总耗时
        z_pred = k * total_flops
        
        # 计算观测雅可比矩阵 H_jac = dz/dx
        H_jac = np.array([
            k * sum_C,       # dz/da
            k * sum_P,       # dz/db
            k * N,           # dz/dd
            total_flops      # dz/dk
        ]).reshape(1, 4)
        
        # 计算卡尔曼增益
        R = self._ekf_measurement_noise_r * elapsed_time**2
        S_innov = H_jac @ P_pred @ H_jac.T + R
        K = P_pred @ H_jac.T / S_innov[0, 0]
        
        # 更新状态
        innovation = elapsed_time - z_pred
        self._ekf_state = self._ekf_state + K.flatten() * innovation
        
        # 更新协方差
        I = np.eye(4)
        self._ekf_covariance = (I - K @ H_jac) @ P_pred
        
        # 参数约束：确保物理意义的合理性
        self._ekf_state[0] = max(1e-10, self._ekf_state[0])  # a > 0
        self._ekf_state[1] = max(1e-20, self._ekf_state[1])  # b > 0
        self._ekf_state[3] = max(1e-20, self._ekf_state[3])  # k > 0
        
        # 更新实例变量
        self.a, self.b, self.d, self.k = self._ekf_state
        
        # 调试日志
        # if logger.isEnabledFor(logging.DEBUG):
        total_chunk = sum(c for c, h in request_chunks)
        logger.info(
            f"EKF batch更新: N={N}, total_chunk={total_chunk}, "
            f"预测时间={z_pred:.6f}s, 实际时间={elapsed_time:.6f}s, "
            f"残差={innovation:.6f}s ({abs(innovation/elapsed_time*100):.1f}%)"
        )
        logger.info(
            f"EKF参数: a={self.a:.2e}, b={self.b:.2e}, "
            f"d={self.d:.2e}, k={self.k:.2e}"
        )
    
    def get_ekf_state_info(self) -> Dict:
        """获取EKF状态信息用于调试"""
        info = {
            'initialized': self._ekf_initialized,
            'params': {'a': self.a, 'b': self.b, 'd': self.d, 'k': self.k},
            'sample_buffer_size': len(self._ekf_sample_buffer),
        }
        
        if self._ekf_covariance is not None:
            # 提取标准差
            std = np.sqrt(np.diag(self._ekf_covariance))
            info['std'] = {'a': std[0], 'b': std[1], 'd': std[2], 'k': std[3]}
        
        return info
    
    def estimate_flops_ratio(self, chunk_size: int, kv_cache_len: int, layer_idx: Optional[int] = None) -> float:
        """
        Estimate FLOPs ratio (non-attention / attention)
        
        Args:
            chunk_size: Current chunk size
            kv_cache_len: KV cache length (number of processed tokens)
            layer_idx: Specific layer index
        
        Returns:
            non-attention FLOPs / attention FLOPs
        """
        seq_len = kv_cache_len + chunk_size
        
        proj_flops, attn_flops = self.calculate_attention_flops(chunk_size, seq_len)
        # Apply global attention calibration to better match wall-clock
        attn_flops = attn_flops * self.attn_flops_scale
        ffn_flops = self.calculate_ffn_flops(chunk_size, layer_idx)
        other_flops = self.calculate_other_flops(chunk_size)
        
        non_attn_flops = ffn_flops + other_flops + proj_flops
        
        if attn_flops == 0:
            return float('inf')
        
        return non_attn_flops / attn_flops
    
    def estimate_flops_ratio_average(self, chunk_size: int, kv_cache_len: int) -> float:
        """Average FLOPs ratio across layers (default: uniform layers)."""
        return self.estimate_flops_ratio(chunk_size, kv_cache_len)
    
    def estimate_flops_ratio_scaled(self, chunk_size: int, kv_cache_len: int) -> float:
        """Estimate FLOPs ratio scaled by chunk size (ratio * chunk_size)."""
        return self.estimate_flops_ratio_average(chunk_size, kv_cache_len) * chunk_size

    # Backward-compatible aliases
    def estimate_p(self, chunk_size: int, kv_cache_len: int, layer_idx: Optional[int] = None) -> float:
        return self.estimate_flops_ratio(chunk_size, kv_cache_len, layer_idx)

    def estimate_p_average(self, chunk_size: int, kv_cache_len: int) -> float:
        return self.estimate_flops_ratio_average(chunk_size, kv_cache_len)

    def estimate_p_times_c(self, chunk_size: int, kv_cache_len: int) -> float:
        return self.estimate_flops_ratio_scaled(chunk_size, kv_cache_len)
    
    def print_model_info(self):
        """Print model information (common + subclass extras)."""
        logger.info("Model Type: %s", self.model_type)
        logger.info("Hidden Size: %s", self.hidden_size)
        logger.info("Attention Heads: %s", self.num_attention_heads)
        logger.info("KV Heads: %s (GQA ratio: %s:1)", self.num_key_value_heads, self.gqa_ratio)
        logger.info("Layers: %s", self.num_hidden_layers)
        logger.info("Head Dimension: %s", self.head_dim)
        # Subclass-provided extra details
        for line in self._extra_model_info():
            logger.info("%s", line)
    
    def get_flops_ratio_table(self, chunk_sizes: List[int] = None, 
                              kv_cache_lens: List[int] = None) -> Dict:
        """Generate FLOPs ratio table for different chunk sizes and cache lengths"""
        if chunk_sizes is None:
            chunk_sizes = [512, 1024, 2048, 4096]
        
        if kv_cache_lens is None:
            kv_cache_lens = [0, 2048, 8192, 16384, 32768, 65536, 98304]
        
        table = {}
        for chunk_size in chunk_sizes:
            table[chunk_size] = {}
            for kv_len in kv_cache_lens:
                ratio = self.estimate_flops_ratio_average(chunk_size, kv_len)
                table[chunk_size][kv_len] = ratio
        
        return table
    
    def print_flops_ratio_table(self, chunk_sizes: List[int] = None, 
                                kv_cache_lens: List[int] = None):
        """Print FLOPs ratio table"""
        table = self.get_flops_ratio_table(chunk_sizes, kv_cache_lens)
        
        logger.info("\nFLOPs Ratio Table (non-attention / attention)")
        logger.info("%s", "=" * 80)
        header = ["KV Cache"] + [f"C={cs:<6}" for cs in sorted(table.keys())]
        logger.info("%s", "".join(f"{h:<10}" for h in header))
        logger.info("%s", "-" * 80)
        
        all_kv_lens = sorted(next(iter(table.values())).keys())
        for kv_len in all_kv_lens:
            row = [f"{kv_len:<10}"]
            for chunk_size in sorted(table.keys()):
                ratio = table[chunk_size][kv_len]
                row.append(f"{ratio:8.2f}")
            logger.info("%s", "".join(row))
        # Allow subclasses to append extra table-related info
        self._extra_p_table_info()

    # Backward-compatible aliases
    def get_p_table(self, chunk_sizes: List[int] = None, 
                    kv_cache_lens: List[int] = None) -> Dict:
        return self.get_flops_ratio_table(chunk_sizes, kv_cache_lens)

    def print_p_table(self, chunk_sizes: List[int] = None, 
                      kv_cache_lens: List[int] = None):
        return self.print_flops_ratio_table(chunk_sizes, kv_cache_lens)

    # -------------------------
    # Subclass hooks
    # -------------------------
    def _extra_model_info(self) -> List[str]:
        """Override to append subclass-specific info lines."""
        return []

    def _extra_p_table_info(self) -> None:
        """Override to print additional info after p-table (e.g., layer analysis)."""
        return None


class LlamaEstimator(BaseModelEstimator):
    """FLOP estimator for Llama models (standard transformer architecture)"""
    
    def _parse_architecture_specific_config(self):
        """Parse Llama-specific configuration"""
        inter_size = self.config.get('intermediate_size')
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')  # e.g., OPT family
        if inter_size is None:
            raise ValueError(
                "Missing FFN size in config: expected 'intermediate_size' or 'ffn_dim'")
        self.intermediate_size = inter_size
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.hidden_size else 0
        
        # Llama uses SwiGLU activation (gate + up projections)
        self.has_gate_proj = True
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Standard transformer attention FLOPs (Llama)."""
        return self._standard_attention_flops(chunk_size, seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """FFN FLOPs for Llama (SwiGLU)."""
        return self._ffn_swiglu_like_flops(chunk_size)
    
    def _extra_model_info(self) -> List[str]:
        return [
            "Architecture: Standard Transformer (Llama)",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            "Activation: SwiGLU",
        ]


class QwenEstimator(BaseModelEstimator):
    """FLOP estimator for Qwen models"""
    
    def _parse_architecture_specific_config(self):
        """Parse Qwen-specific configuration"""
        inter_size = self.config.get('intermediate_size')
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')
        if inter_size is None:
            raise ValueError(
                "Missing FFN size in config: expected 'intermediate_size' or 'ffn_dim'")
        self.intermediate_size = inter_size
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
        
        # Qwen also uses SwiGLU but may have different parameter names
        self.has_gate_proj = True
        
        # Qwen-specific parameters
        self.max_position_embeddings = self.config.get('max_position_embeddings', 32768)
        self.use_sliding_window = self.config.get('use_sliding_window', False)
        self.sliding_window = self.config.get('sliding_window')
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Qwen attention FLOPs with optional sliding window sparsity."""
        effective_seq_len = (min(seq_len, self.sliding_window)
                             if self.use_sliding_window and self.sliding_window
                             else seq_len)
        return self._standard_attention_flops(
            chunk_size, seq_len, effective_seq_len=effective_seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """FFN FLOPs for Qwen (SwiGLU)."""
        return self._ffn_swiglu_like_flops(chunk_size)
    
    def _extra_model_info(self) -> List[str]:
        lines = [
            "Architecture: Qwen Transformer",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            f"Max Position Embeddings: {self.max_position_embeddings}",
        ]
        if self.use_sliding_window:
            lines.append(f"Sliding Window: {self.sliding_window}")
        return lines


class DeepSeekEstimator(BaseModelEstimator):
    """FLOP estimator for DeepSeek models (supports MLA + MoE)"""
    
    def _parse_architecture_specific_config(self):
        """Parse DeepSeek-specific configuration"""
        # Standard FFN parameters
        self.intermediate_size = self.config.get('intermediate_size')
        
        # MLA (Multi-head Latent Attention) parameters
        self.kv_lora_rank = self.config.get('kv_lora_rank')
        self.q_lora_rank = self.config.get('q_lora_rank')
        self.qk_rope_head_dim = self.config.get('qk_rope_head_dim', 64)
        
        # MoE parameters
        self.moe_intermediate_size = self.config.get('moe_intermediate_size')
        self.first_k_dense_replace = self.config.get('first_k_dense_replace', 0)
        self.n_routed_experts = self.config.get('n_routed_experts', 0)
        self.num_experts_per_tok = self.config.get('num_experts_per_tok', 1)
        self.n_shared_experts = self.config.get('n_shared_experts', 0)
        
        # Architecture flags
        self.is_mla = self._is_mla_architecture()
        self.is_moe = self._is_moe_architecture()
        
        # Calculate effective dimensions
        self._calculate_effective_dims()
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
    
    def _is_mla_architecture(self) -> bool:
        """Check if model uses MLA (Multi-head Latent Attention)"""
        kv = self.kv_lora_rank
        q = self.q_lora_rank
        return (isinstance(kv, int) and kv > 0 and
                isinstance(q, int) and q > 0)
    
    def _is_moe_architecture(self) -> bool:
        """Check if model uses MoE (Mixture of Experts)"""
        return (self.moe_intermediate_size is not None and 
                self.n_routed_experts > 0)
    
    def _calculate_effective_dims(self):
        """Calculate effective dimensions for MLA"""
        if self.is_mla:
            # MLA uses compressed K/V representations. Do not add RoPE dim here
            # to avoid double-counting; RoPE cost is captured separately.
            self.effective_kv_dim = self.kv_lora_rank
            self.effective_q_dim = self.q_lora_rank
        else:
            # Standard dimensions
            self.effective_kv_dim = self.hidden_size // self.gqa_ratio
            self.effective_q_dim = self.hidden_size
    
    def _is_dense_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer is dense (not MoE)"""
        return layer_idx < self.first_k_dense_replace
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate attention FLOPs with MLA support"""
        if self.is_mla:
            return self._calculate_mla_attention_flops(chunk_size, seq_len)
        else:
            return self._calculate_standard_attention_flops(chunk_size, seq_len)
    
    def _calculate_mla_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate FLOPs for MLA (Multi-head Latent Attention)"""
        # Q projections: q_a (to LoRA rank) + q_b (LoRA to heads)
        q_a_flops = chunk_size * self.hidden_size * self.q_lora_rank
        q_b_flops = chunk_size * self.q_lora_rank * (self.num_attention_heads * self.head_dim)
        
        # KV projections: kv_a (to LoRA rank) + kv_b (LoRA to compressed KV)
        kv_a_flops = chunk_size * self.hidden_size * self.effective_kv_dim
        kv_b_flops = chunk_size * self.effective_kv_dim * (self.num_key_value_heads * self.head_dim * 2)
        
        # Attention computation using compressed dimensions
        # MLA uses two components: compressed KV and RoPE
        # Score computation: Q^T @ K^C (compressed) + Q^TR @ K^R (RoPE)
        # The effective computation is with compressed dimensions, not full hidden_size
        compressed_score_flops = chunk_size * seq_len * self.effective_kv_dim
        rope_score_flops = chunk_size * seq_len * self.qk_rope_head_dim * self.num_attention_heads
        
        # Value computation: attention_weights @ V^C (compressed)
        value_flops = chunk_size * seq_len * self.effective_kv_dim
        
        # Output projection
        out_flops = chunk_size * self.hidden_size * self.hidden_size
        
        proj_flops = q_a_flops + q_b_flops + kv_a_flops + kv_b_flops + out_flops
        attn_flops = compressed_score_flops + rope_score_flops + value_flops
        
        return proj_flops, attn_flops
    
    def _calculate_standard_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate FLOPs for standard transformer attention"""
        return self._standard_attention_flops(chunk_size, seq_len, kv_dim=self.effective_kv_dim)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """Calculate FFN FLOPs with MoE support"""
        if self.is_moe and (layer_idx is None or not self._is_dense_layer(layer_idx)):
            return self._calculate_moe_ffn_flops(chunk_size)
        else:
            return self._calculate_dense_ffn_flops(chunk_size)
    
    def _calculate_dense_ffn_flops(self, chunk_size: int) -> int:
        """Calculate dense FFN FLOPs"""
        # Gate and Up projections (SwiGLU)
        up_gate_flops = chunk_size * self.hidden_size * self.intermediate_size * 2
        
        # Down projection
        down_flops = chunk_size * self.intermediate_size * self.hidden_size
        
        return up_gate_flops + down_flops
    
    def _calculate_moe_ffn_flops(self, chunk_size: int) -> int:
        """Calculate MoE FFN FLOPs"""
        # Shared experts (if exist)
        shared_flops = 0
        if self.n_shared_experts > 0:
            shared_flops = self._calculate_single_expert_flops(chunk_size, self.moe_intermediate_size)
        
        # Routed experts (only active ones)
        routed_flops = (self._calculate_single_expert_flops(chunk_size, self.moe_intermediate_size) * 
                       self.num_experts_per_tok)
        
        # Gating network
        gate_flops = chunk_size * self.hidden_size * self.n_routed_experts
        
        return shared_flops + routed_flops + gate_flops
    
    def _calculate_single_expert_flops(self, chunk_size: int, expert_size: int) -> int:
        """Calculate FLOPs for a single expert"""
        # Gate and Up projections
        up_gate_flops = chunk_size * self.hidden_size * expert_size * 2
        
        # Down projection  
        down_flops = chunk_size * expert_size * self.hidden_size
        
        return up_gate_flops + down_flops
    
    def calculate_other_flops(self, chunk_size: int) -> int:
        """Calculate other FLOPs with MLA-specific normalization"""
        # MLA has additional normalization layers
        norm_multiplier = 6 if self.is_mla else 2
        norm_flops = chunk_size * self.hidden_size * norm_multiplier
        
        # Residual connections
        residual_flops = chunk_size * self.hidden_size * 2
        
        return norm_flops + residual_flops
    
    def estimate_flops_ratio_average(self, chunk_size: int, kv_cache_len: int) -> float:
        """Average FLOPs ratio across layers (MoE uses per-layer averaging)."""
        if not self.is_moe:
            return super().estimate_flops_ratio_average(chunk_size, kv_cache_len)
        total_ratio = 0.0
        for layer_idx in range(self.num_hidden_layers):
            layer_ratio = self.estimate_flops_ratio(chunk_size, kv_cache_len, layer_idx)
            total_ratio += layer_ratio
        return total_ratio / self.num_hidden_layers
    
    def get_layer_analysis(self, chunk_size: int = 2048, kv_cache_len: int = 16384) -> Dict:
        """Analyze p values for each layer type (useful for MoE models)"""
        analysis = {
            'dense_layers': [],
            'moe_layers': [],
            'layer_p_values': {}
        }
        
        for layer_idx in range(self.num_hidden_layers):
            is_dense = self._is_dense_layer(layer_idx)
            p_value = self.estimate_p(chunk_size, kv_cache_len, layer_idx)
            
            analysis['layer_p_values'][layer_idx] = {
                'p_value': p_value,
                'is_dense': is_dense,
                'layer_type': 'dense' if is_dense else 'moe'
            }
            
            if is_dense:
                analysis['dense_layers'].append(layer_idx)
            else:
                analysis['moe_layers'].append(layer_idx)
        
        return analysis
    
    def _extra_model_info(self) -> List[str]:
        architecture_type = f"{'MLA' if self.is_mla else 'Standard'} + {'MoE' if self.is_moe else 'Dense'}"
        lines: List[str] = [f"Architecture: {architecture_type}"]
        if self.is_mla:
            lines += [
                f"Q LoRA Rank: {self.q_lora_rank}",
                f"KV LoRA Rank: {self.kv_lora_rank}",
                f"Effective KV Dim: {self.effective_kv_dim}",
            ]
        if self.is_moe:
            lines += [
                f"MoE Intermediate Size: {self.moe_intermediate_size}",
                f"Routed Experts: {self.n_routed_experts}",
                f"Experts per Token: {self.num_experts_per_tok}",
                f"First K Dense Layers: {self.first_k_dense_replace}",
            ]
            if self.n_shared_experts:
                lines.append(f"Shared Experts: {self.n_shared_experts}")
        else:
            lines.append(
                f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)")
        return lines
    
    def _extra_p_table_info(self) -> None:
        if self.is_moe:
            logger.info("\nLayer Analysis:")
            logger.info("%s", "-" * 50)
            analysis = self.get_layer_analysis()
            logger.info("Dense layers (0-%d): %d", self.first_k_dense_replace - 1, len(analysis['dense_layers']))
            logger.info("MoE layers (%d-%d): %d", self.first_k_dense_replace, self.num_hidden_layers - 1, len(analysis['moe_layers']))


class MixtralEstimator(BaseModelEstimator):
    """FLOP estimator for Mixtral models (MoE architecture)"""
    
    def _parse_architecture_specific_config(self):
        """Parse Mixtral-specific configuration"""
        self.intermediate_size = self.config.get('intermediate_size')
        self.num_local_experts = self.config.get('num_local_experts', 8)
        self.num_experts_per_tok = self.config.get('num_experts_per_tok', 2)
        
        # Mixtral specific parameters
        self.max_position_embeddings = self.config.get('max_position_embeddings', 32768)
        self.sliding_window = self.config.get('sliding_window')
        
        # Architecture flags
        self.is_moe = True
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Attention FLOPs for Mixtral with optional sliding window."""
        effective_seq_len = min(seq_len, self.sliding_window) if self.sliding_window else seq_len
        return self._standard_attention_flops(
            chunk_size, seq_len, effective_seq_len=effective_seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """Calculate MoE FFN FLOPs for Mixtral"""
        # Only active experts are computed
        active_expert_flops = self._calculate_single_expert_flops(chunk_size) * self.num_experts_per_tok
        
        # Gating network
        gate_flops = chunk_size * self.hidden_size * self.num_local_experts
        
        return active_expert_flops + gate_flops
    
    def _calculate_single_expert_flops(self, chunk_size: int) -> int:
        """Calculate FLOPs for a single Mixtral expert"""
        # Gate, Up and Down projections (SwiGLU)
        up_gate_flops = chunk_size * self.hidden_size * self.intermediate_size * 2
        down_flops = chunk_size * self.intermediate_size * self.hidden_size
        
        return up_gate_flops + down_flops
    
    def _extra_model_info(self) -> List[str]:
        lines = [
            "Architecture: Mixtral (MoE)",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            f"Local Experts: {self.num_local_experts}",
            f"Experts per Token: {self.num_experts_per_tok}",
        ]
        if self.sliding_window:
            lines.append(f"Sliding Window: {self.sliding_window}")
        return lines


class GemmaEstimator(BaseModelEstimator):
    """FLOP estimator for Gemma models"""
    
    def _parse_architecture_specific_config(self):
        """Parse Gemma-specific configuration"""
        inter_size = self.config.get('intermediate_size')
        if inter_size is None:
            inter_size = self.config.get('ffn_dim')
        if inter_size is None:
            raise ValueError(
                "Missing FFN size in config: expected 'intermediate_size' or 'ffn_dim'")
        self.intermediate_size = inter_size
        self.ffn_expansion = self.intermediate_size / self.hidden_size if self.intermediate_size else 0
        
        # Gemma specific parameters
        self.head_dim = self.config.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 8192)
        
        # Gemma uses GeGLU activation instead of SwiGLU
        self.activation_type = 'gelu'
    
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> Tuple[int, int]:
        """Calculate attention FLOPs for Gemma"""
        return self._standard_attention_flops(chunk_size, seq_len)
    
    def calculate_ffn_flops(self, chunk_size: int, layer_idx: Optional[int] = None) -> int:
        """FFN FLOPs for Gemma (GeGLU)."""
        return self._ffn_swiglu_like_flops(chunk_size)
    
    def _extra_model_info(self) -> List[str]:
        return [
            "Architecture: Gemma Transformer",
            f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)",
            "Activation: GeGLU",
            f"Max Position Embeddings: {self.max_position_embeddings}",
        ]


# Registry of model estimators
MODEL_ESTIMATOR_REGISTRY = {
    # Llama family
    'llama': LlamaEstimator,
    'llama2': LlamaEstimator,
    'llama3': LlamaEstimator,
    'llama4': LlamaEstimator,
    'code_llama': LlamaEstimator,
    
    # Qwen family  
    'qwen': QwenEstimator,
    'qwen2': QwenEstimator,
    'qwen2_moe': QwenEstimator,
    'qwen2.5': QwenEstimator,
    
    # DeepSeek family
    'deepseek': DeepSeekEstimator,
    'deepseek_v2': DeepSeekEstimator,
    'deepseek_v3': DeepSeekEstimator,
    'deepseek_mtp': DeepSeekEstimator,  # Unified model type for DeepSeek
    
    # Mixtral family
    'mixtral': MixtralEstimator,
    'mixtral_8x7b': MixtralEstimator,
    'mixtral_8x22b': MixtralEstimator,
    
    # Gemma family
    'gemma': GemmaEstimator,
    'gemma2': GemmaEstimator,
    
    # Add other models as needed
    # 'phi': PhiEstimator,
    # 'chatglm': ChatGLMEstimator,
    # etc.
}


def create_estimator(config_path: str = None, config_dict: Dict = None) -> BaseModelEstimator:
    """
    Factory function to create appropriate model estimator based on model type
    
    Args:
        config_path: Path to config file
        config_dict: Config dictionary
        
    Returns:
        Appropriate model estimator instance
    """
    # Load config
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif config_dict:
        config = config_dict
    else:
        raise ValueError("Must provide either config_path or config_dict")
    
    # Determine model type
    model_type = config.get('model_type', 'unknown').lower()
    
    # Handle special cases and mappings
    if model_type == 'deepseek_v3':
        model_type = 'deepseek_mtp'  # Use unified DeepSeek estimator
    
    # Find appropriate estimator class
    estimator_class = MODEL_ESTIMATOR_REGISTRY.get(model_type)
    
    if estimator_class is None:
        # Fallback to Llama estimator for unknown models (most are transformer-based)
        logger.warning("Unknown model type '%s', falling back to Llama estimator",
                       model_type)
        estimator_class = LlamaEstimator
    
    return estimator_class(config)


# Backward compatibility: keep the original ModelPEstimator as an alias
class ModelPEstimator:
    """Backward compatibility wrapper - use create_estimator() for new code"""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        # Create appropriate estimator and delegate to it
        self._estimator = create_estimator(config_path, config_dict)
        
        # Copy attributes for backward compatibility
        self.__dict__.update(self._estimator.__dict__)
    
    def __getattr__(self, name):
        """Delegate any missing methods to the underlying estimator"""
        return getattr(self._estimator, name)


def create_estimator_safely(
    hf_config_obj: object | None = None,
    config_path: str | None = None,
    scheduler_config: object | None = None,
):
    """Best-effort estimator factory.

    - Prefer building from an in-memory HF config object (with to_dict()).
    - Fallback to reading a JSON config from disk.
    - If provided, configure baseline from scheduler config.

    Returns an estimator instance or None on failure.
    """
    estimator = None
    # Try HF config object first
    if hf_config_obj is not None:
        try:
            to_dict_fn = getattr(hf_config_obj, "to_dict", None)
            cfg_dict = to_dict_fn() if callable(to_dict_fn) else dict(
                getattr(hf_config_obj, "__dict__", {}))
            estimator = create_estimator(config_dict=cfg_dict)
        except (AttributeError, TypeError, ValueError):
            estimator = None

    # Fallback to disk json
    if estimator is None and config_path:
        try:
            estimator = create_estimator(config_path=config_path)
        except (OSError, json.JSONDecodeError, ValueError):
            estimator = None

    # Configure baseline if possible
    if estimator is not None and scheduler_config is not None:
        try:
            estimator.configure_baseline_from_scheduler(scheduler_config)
        except AttributeError:
            # Non-fatal; leave estimator as-is
            pass

    return estimator


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Estimate p-values from model config')
    parser.add_argument('--config-path', type=str, help='Path to model config.json')
    parser.add_argument('--chunk-size', type=int, default=2048, 
                       help='Chunk size for estimation')
    parser.add_argument('--kv-cache-len', type=int, default=0,
                       help='KV cache length')
    parser.add_argument('--table', action='store_true',
                       help='Print p-value table')
    parser.add_argument('--estimator-type', type=str, 
                       help='Force specific estimator type (llama, qwen, deepseek)')
    
    args = parser.parse_args()
    
    # Create estimator
    if args.estimator_type:
        # Override model type for testing
        config = json.load(open(args.config_path, 'r', encoding='utf-8'))
        config['model_type'] = args.estimator_type
        estimator = create_estimator(config_dict=config)
    else:
        # Auto-detect from config
        estimator = create_estimator(config_path=args.config_path)
    
    # Print info and calculate ratios
    estimator.print_model_info()
    
    ratio = estimator.estimate_flops_ratio(args.chunk_size, args.kv_cache_len)
    ratio_scaled = estimator.estimate_flops_ratio_scaled(args.chunk_size, args.kv_cache_len)
    
    print("\nEstimated values:")
    print(f"flops_ratio = {ratio:.4f}")
    print(f"flops_ratio × chunk_size = {ratio_scaled:.4f}")
    
    # Print ratio table if requested
    if args.table:
        estimator.print_flops_ratio_table()
    
    estimator.configure_baseline_from_scheduler(None)
    seq_len = 128000
    chunk_size = 2048
    hist_seq_len = 0
    block_size = 16
    while(hist_seq_len < seq_len):
        chunk_size_with_overhead = estimator.compute_chunk_size_with_overhead(
            hist_seq_len, seq_len, chunk_size, block_size)
        print(f"chunk_size_with_overhead = {chunk_size_with_overhead}")
        hist_seq_len += chunk_size_with_overhead[0]

if __name__ == "__main__":
    main()