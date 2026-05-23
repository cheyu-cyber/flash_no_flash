"""Small DistributedDataParallel helper used by ``model/train.py``.

When the script is launched with ``torchrun --nproc_per_node=N -m model.train``
PyTorch sets ``LOCAL_RANK``, ``RANK``, and ``WORLD_SIZE`` in each spawned
worker's environment. ``init_ddp()`` picks those up and initialises the
``nccl`` process group. When the script is launched plainly
(``python -m model.train``) those env vars are unset and every helper here
falls back to single-GPU values so the legacy code path is unaffected.

All public functions are safe to call whether or not DDP is initialised —
they simply return the no-DDP defaults when ``WORLD_SIZE`` is 1 or unset.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Init / teardown
# ---------------------------------------------------------------------------

def _has_torchrun_env() -> bool:
    """True if the process was spawned by ``torchrun`` (or another launcher
    that sets ``WORLD_SIZE``)."""
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def init_ddp(backend: str = "nccl") -> None:
    """Initialise the distributed process group if a torchrun-style launcher
    set the env vars. A no-op otherwise."""
    if not _has_torchrun_env():
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank())


def cleanup_ddp() -> None:
    """Tear down the distributed process group. Safe to call unconditionally."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Per-rank info
# ---------------------------------------------------------------------------

def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    """Index of the current process within its node. Used to pick which
    CUDA device the process should bind to."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    """True only on the rank-0 process. Use to gate logging, checkpoint
    saves, csv writes, tqdm bars, etc."""
    return rank() == 0


# ---------------------------------------------------------------------------
# Collective ops
# ---------------------------------------------------------------------------

def barrier() -> None:
    """Synchronise all ranks. No-op in single-GPU mode."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float | torch.Tensor, device: Optional[torch.device] = None) -> float:
    """All-reduce ``value`` across ranks with mean reduction.

    Accepts a Python float or a 0-d tensor and returns a Python float.
    In single-GPU mode this is a no-op (just casts to float).
    """
    ws = world_size()
    if ws <= 1:
        return float(value.item() if isinstance(value, torch.Tensor) else value)

    if not isinstance(value, torch.Tensor):
        if device is None:
            device = torch.device(f"cuda:{local_rank()}")
        value = torch.tensor(float(value), device=device, dtype=torch.float64)
    else:
        value = value.detach().to(dtype=torch.float64)

    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= ws
    return float(value.item())
