"""Correctness + speed test for the torch bilateral filters.

Compares `algo_torch.{bilateral_filter_torch, joint_bilateral_filter_torch}`
against the reference numpy implementations in `algo.py`. Asserts numerical
equivalence on small images, then benchmarks speed on a larger image.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from models.algo import bilateral_filter, joint_bilateral_filter
from models.algo_torch import bilateral_filter_torch, joint_bilateral_filter_torch


def _rand_image(h: int, w: int, c: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w, c), dtype=np.float64)


def correctness(sigma_d: float = 2.0, sigma_r: float = 0.1, h: int = 48, w: int = 48):
    print(f"\n[correctness] image={h}x{w}x3, sigma_d={sigma_d}, sigma_r={sigma_r}")
    A = _rand_image(h, w, seed=1)
    F = _rand_image(h, w, seed=2)

    # bilateral
    t0 = time.perf_counter()
    ref_b = bilateral_filter(A, sigma_d=sigma_d, sigma_r=sigma_r)
    t_np_b = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_b_64 = bilateral_filter_torch(A, sigma_d=sigma_d, sigma_r=sigma_r,
                                      dtype=torch.float64)
    t_pt_b64 = time.perf_counter() - t0

    out_b_32 = bilateral_filter_torch(A, sigma_d=sigma_d, sigma_r=sigma_r,
                                      dtype=torch.float32)

    diff_b64 = np.abs(out_b_64 - ref_b).max()
    diff_b32 = np.abs(out_b_32 - ref_b).max()
    print(f"  bilateral       np ref:        {t_np_b*1000:7.1f} ms")
    print(f"  bilateral       torch (f64):   {t_pt_b64*1000:7.1f} ms"
          f"   max|diff|={diff_b64:.2e}")
    print(f"  bilateral       torch (f32):                   "
          f"   max|diff|={diff_b32:.2e}")

    # joint bilateral
    t0 = time.perf_counter()
    ref_j = joint_bilateral_filter(A, F, sigma_d=sigma_d, sigma_r=sigma_r)
    t_np_j = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_j_64 = joint_bilateral_filter_torch(A, F, sigma_d=sigma_d, sigma_r=sigma_r,
                                            dtype=torch.float64)
    t_pt_j64 = time.perf_counter() - t0

    out_j_32 = joint_bilateral_filter_torch(A, F, sigma_d=sigma_d, sigma_r=sigma_r,
                                            dtype=torch.float32)

    diff_j64 = np.abs(out_j_64 - ref_j).max()
    diff_j32 = np.abs(out_j_32 - ref_j).max()
    print(f"  joint bilateral np ref:        {t_np_j*1000:7.1f} ms")
    print(f"  joint bilateral torch (f64):   {t_pt_j64*1000:7.1f} ms"
          f"   max|diff|={diff_j64:.2e}")
    print(f"  joint bilateral torch (f32):                   "
          f"   max|diff|={diff_j32:.2e}")

    # Tight tolerance for f64 (just float rounding); looser for f32.
    assert diff_b64 < 1e-10, f"bilateral f64 mismatch: {diff_b64}"
    assert diff_j64 < 1e-10, f"joint bilateral f64 mismatch: {diff_j64}"
    assert diff_b32 < 1e-4,  f"bilateral f32 too loose: {diff_b32}"
    assert diff_j32 < 1e-4,  f"joint bilateral f32 too loose: {diff_j32}"
    print("  OK — torch outputs match numpy reference.")


def speed(sigma_d: float = 3.0, sigma_r: float = 0.08, h: int = 256, w: int = 256):
    print(f"\n[speed] image={h}x{w}x3, sigma_d={sigma_d}, sigma_r={sigma_r}")
    A = _rand_image(h, w, seed=3)
    F = _rand_image(h, w, seed=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch device: {device}")

    # Warm-up (CUDA kernel compile + cuDNN pick).
    _ = bilateral_filter_torch(A, sigma_d=sigma_d, sigma_r=sigma_r, dtype=torch.float32)
    if device == "cuda":
        torch.cuda.synchronize()

    # numpy — only run once; with default radius it's slow.
    t0 = time.perf_counter()
    ref_b = bilateral_filter(A, sigma_d=sigma_d, sigma_r=sigma_r)
    t_np_b = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_j = joint_bilateral_filter(A, F, sigma_d=sigma_d, sigma_r=sigma_r)
    t_np_j = time.perf_counter() - t0

    # torch — average over a few runs (after warm-up).
    def _time(fn, runs=3):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / runs, out

    t_pt_b32, out_b32 = _time(
        lambda: bilateral_filter_torch(A, sigma_d=sigma_d, sigma_r=sigma_r,
                                       dtype=torch.float32))
    t_pt_b64, out_b64 = _time(
        lambda: bilateral_filter_torch(A, sigma_d=sigma_d, sigma_r=sigma_r,
                                       dtype=torch.float64))
    t_pt_j32, out_j32 = _time(
        lambda: joint_bilateral_filter_torch(A, F, sigma_d=sigma_d, sigma_r=sigma_r,
                                             dtype=torch.float32))
    t_pt_j64, out_j64 = _time(
        lambda: joint_bilateral_filter_torch(A, F, sigma_d=sigma_d, sigma_r=sigma_r,
                                             dtype=torch.float64))

    print(f"  bilateral       np ref:      {t_np_b*1000:8.1f} ms"
          f"   (max|diff f32|={np.abs(out_b32-ref_b).max():.2e})")
    print(f"  bilateral       torch f64:   {t_pt_b64*1000:8.1f} ms"
          f"   speedup x{t_np_b/t_pt_b64:7.1f}")
    print(f"  bilateral       torch f32:   {t_pt_b32*1000:8.1f} ms"
          f"   speedup x{t_np_b/t_pt_b32:7.1f}")
    print(f"  joint bilateral np ref:      {t_np_j*1000:8.1f} ms"
          f"   (max|diff f32|={np.abs(out_j32-ref_j).max():.2e})")
    print(f"  joint bilateral torch f64:   {t_pt_j64*1000:8.1f} ms"
          f"   speedup x{t_np_j/t_pt_j64:7.1f}")
    print(f"  joint bilateral torch f32:   {t_pt_j32*1000:8.1f} ms"
          f"   speedup x{t_np_j/t_pt_j32:7.1f}")


if __name__ == "__main__":
    correctness()
    speed()
