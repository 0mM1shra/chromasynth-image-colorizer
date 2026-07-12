"""
metrics.py — SDE Profiling & DSML Analytics Framework
=====================================================
Dual-purpose evaluation module satisfying both Software Engineering (SDE)
telemetry requirements and Data Science / Machine Learning (DSML) validation
standards for production colorization inference.

SDE Telemetry:
    - InferenceProfiler:  Context manager capturing wall-clock latency,
                          RAM delta, and VRAM delta per inference request.
    - SystemSnapshot:     Frozen dataclass of system resource utilization.
    - get_system_snapshot: Factory for current system state.

DSML Validation:
    - compute_psnr:         Peak Signal-to-Noise Ratio between image domains.
    - compute_ssim:         Structural Similarity Index between image domains.
    - validate_colorization: Aggregate quality assessment with classification.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import psutil
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


logger = logging.getLogger("colorization.metrics")


# ═══════════════════════════════════════════════════════════════════════════
# SDE TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SystemSnapshot:
    """Immutable snapshot of system resource utilization.

    Attributes:
        cpu_percent:      CPU utilization as a percentage [0, 100].
        ram_used_mb:      Process-level resident set size in megabytes.
        ram_total_mb:     Total system RAM in megabytes.
        gpu_allocated_mb: CUDA memory currently allocated (None if no GPU).
        gpu_reserved_mb:  CUDA memory reserved by caching allocator.
        gpu_name:         GPU device name string.
    """

    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    gpu_name: Optional[str] = None


def get_system_snapshot() -> SystemSnapshot:
    """Capture a point-in-time snapshot of system resources.

    Returns:
        SystemSnapshot with CPU, RAM, and GPU utilization metrics.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()

    gpu_alloc = None
    gpu_reserved = None
    gpu_name = None

    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        gpu_name = torch.cuda.get_device_name(0)

    return SystemSnapshot(
        cpu_percent=process.cpu_percent(interval=None),
        ram_used_mb=mem_info.rss / (1024 ** 2),
        ram_total_mb=vm.total / (1024 ** 2),
        gpu_allocated_mb=gpu_alloc,
        gpu_reserved_mb=gpu_reserved,
        gpu_name=gpu_name,
    )


@dataclass
class ProfileResult:
    """Container for inference profiling measurements.

    Attributes:
        latency_ms:     Wall-clock inference time in milliseconds.
        ram_delta_mb:   Change in process RSS during inference.
        vram_delta_mb:  Change in CUDA allocated memory during inference.
    """

    latency_ms: float = 0.0
    ram_delta_mb: float = 0.0
    vram_delta_mb: float = 0.0

    def to_headers(self) -> dict[str, str]:
        """Serialize as HTTP response header key-value pairs."""
        return {
            "X-Inference-Latency-Ms": f"{self.latency_ms:.2f}",
            "X-RAM-Delta-MB": f"{self.ram_delta_mb:.2f}",
            "X-VRAM-Delta-MB": f"{self.vram_delta_mb:.2f}",
        }


class InferenceProfiler:
    """Context manager for profiling a single inference pass.

    Captures high-resolution wall-clock time, process RSS delta,
    and CUDA memory allocation delta around the wrapped code block.

    Usage:
        profiler = InferenceProfiler()
        with profiler:
            out_ab = model(input_tensor)
        print(profiler.result.latency_ms)
    """

    def __init__(self) -> None:
        self.result = ProfileResult()
        self._start_time: float = 0.0
        self._start_rss: int = 0
        self._start_vram: int = 0

    def __enter__(self) -> "InferenceProfiler":
        # Synchronize CUDA before timing to drain any pending ops
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_vram = torch.cuda.memory_allocated()

        self._start_rss = psutil.Process().memory_info().rss
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Synchronize CUDA after inference to ensure accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_vram = torch.cuda.memory_allocated()
            self.result.vram_delta_mb = (end_vram - self._start_vram) / (1024 ** 2)

        end_time = time.perf_counter()
        end_rss = psutil.Process().memory_info().rss

        self.result.latency_ms = (end_time - self._start_time) * 1000.0
        self.result.ram_delta_mb = (end_rss - self._start_rss) / (1024 ** 2)

        logger.info(
            "Inference completed — latency=%.2fms | RAM Δ=%.2fMB | VRAM Δ=%.2fMB",
            self.result.latency_ms,
            self.result.ram_delta_mb,
            self.result.vram_delta_mb,
        )


# ═══════════════════════════════════════════════════════════════════════════
# DSML VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════


def compute_psnr(
    original: np.ndarray,
    colorized: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Higher values indicate better reconstruction fidelity.
    Typical ranges: 20–25 dB (acceptable), 25–30 dB (good), 30+ dB (excellent).

    Args:
        original:   Reference RGB image as numpy array, range [0, 1].
        colorized:  Predicted RGB image as numpy array, range [0, 1].
        data_range: Dynamic range of the input (1.0 for float images).

    Returns:
        PSNR value in decibels.
    """
    return float(peak_signal_noise_ratio(original, colorized, data_range=data_range))


def compute_ssim(
    original: np.ndarray,
    colorized: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index between two images.

    Values range from -1 to 1, where 1 indicates perfect similarity.
    Typical thresholds: >0.9 (excellent), 0.8–0.9 (good), <0.8 (fair/poor).

    Args:
        original:   Reference RGB image as numpy array [H, W, 3], range [0, 1].
        colorized:  Predicted RGB image as numpy array [H, W, 3], range [0, 1].
        data_range: Dynamic range of the input.

    Returns:
        SSIM value as a float in [-1, 1].
    """
    # Determine the minimum dimension for win_size constraint
    min_dim = min(original.shape[0], original.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(3, win_size)  # Ensure minimum window of 3

    return float(
        structural_similarity(
            original,
            colorized,
            data_range=data_range,
            channel_axis=2,
            win_size=win_size,
        )
    )


@dataclass(frozen=True)
class QualityReport:
    """Aggregate quality assessment for a colorization result.

    Attributes:
        psnr_db:         Peak Signal-to-Noise Ratio in decibels.
        ssim:            Structural Similarity Index [-1, 1].
        classification:  Human-readable quality label.
    """

    psnr_db: float
    ssim: float
    classification: str

    def to_dict(self) -> dict:
        return {
            "psnr_db": round(self.psnr_db, 2),
            "ssim": round(self.ssim, 4),
            "classification": self.classification,
        }


def validate_colorization(
    original_rgb: np.ndarray,
    colorized_rgb: np.ndarray,
) -> QualityReport:
    """Run full DSML validation suite on a colorization result.

    Computes PSNR, SSIM, and classifies overall quality. Both images
    must be the same shape and in [0, 1] float range.

    Classification thresholds:
        Excellent: SSIM ≥ 0.90 and PSNR ≥ 30 dB
        Good:      SSIM ≥ 0.80 and PSNR ≥ 25 dB
        Fair:      SSIM ≥ 0.65 and PSNR ≥ 20 dB
        Poor:      Below Fair thresholds

    Args:
        original_rgb:  Ground-truth RGB image [H, W, 3], range [0, 1].
        colorized_rgb: Predicted RGB image [H, W, 3], range [0, 1].

    Returns:
        QualityReport with PSNR, SSIM, and quality classification.
    """
    psnr = compute_psnr(original_rgb, colorized_rgb)
    ssim = compute_ssim(original_rgb, colorized_rgb)

    if ssim >= 0.90 and psnr >= 30.0:
        classification = "Excellent"
    elif ssim >= 0.80 and psnr >= 25.0:
        classification = "Good"
    elif ssim >= 0.65 and psnr >= 20.0:
        classification = "Fair"
    else:
        classification = "Poor"

    report = QualityReport(psnr_db=psnr, ssim=ssim, classification=classification)

    logger.info(
        "Quality report — PSNR=%.2f dB | SSIM=%.4f | Classification=%s",
        report.psnr_db,
        report.ssim,
        report.classification,
    )

    return report
