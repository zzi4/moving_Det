from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size != 2
        ):
            raise ValueError("distributed world size must be exactly 2")
        for name, value in (
            ("rank", self.rank),
            ("local rank", self.local_rank),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.world_size
            ):
                raise ValueError(
                    f"distributed {name} must be in [0, world_size)"
                )
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("distributed backend must be a non-empty string")

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _collective_device(context: DistributedContext) -> torch.device:
    if context.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL collectives require CUDA")
        return torch.device("cuda", context.local_rank)
    return torch.device("cpu")


def _validate_process_group(context: DistributedContext) -> None:
    if not isinstance(context, DistributedContext):
        raise ValueError("context must be a DistributedContext")
    if not dist.is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    if dist.get_rank() != context.rank:
        raise RuntimeError("distributed process-group rank differs from context")
    if dist.get_world_size() != context.world_size:
        raise RuntimeError(
            "distributed process-group world size differs from context"
        )


def distributed_sum_count(
    local_sum: float,
    local_count: int,
    context: DistributedContext,
) -> tuple[float, int]:
    _validate_process_group(context)
    if (
        isinstance(local_sum, bool)
        or not isinstance(local_sum, (int, float))
        or not math.isfinite(float(local_sum))
    ):
        raise ValueError("local sum must be finite")
    if (
        isinstance(local_count, bool)
        or not isinstance(local_count, int)
        or local_count < 0
    ):
        raise ValueError("local count must be a non-negative integer")
    values = torch.tensor(
        [float(local_sum), float(local_count)],
        dtype=torch.float64,
        device=_collective_device(context),
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count_value = float(values[1].item())
    if not count_value.is_integer():
        raise RuntimeError("distributed count reduction is not integral")
    return float(values[0].item()), int(count_value)


def distributed_mean(
    local_value: float,
    context: DistributedContext,
) -> float:
    total, count = distributed_sum_count(local_value, 1, context)
    if count != context.world_size:
        raise RuntimeError("distributed mean did not receive every rank")
    return total / count


def gather_rank_objects(
    value: Any,
    context: DistributedContext,
) -> tuple[Any, ...] | None:
    _validate_process_group(context)
    gathered: list[Any] | None = (
        [None] * context.world_size if context.is_primary else None
    )
    dist.gather_object(value, gathered, dst=0)
    return None if gathered is None else tuple(gathered)


def broadcast_metric_pair(
    metrics: tuple[float, float] | None,
    context: DistributedContext,
) -> tuple[float, float]:
    _validate_process_group(context)
    if context.is_primary:
        if (
            not isinstance(metrics, tuple)
            or len(metrics) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics
            )
        ):
            raise ValueError("primary metrics must contain two finite values")
        values = torch.tensor(
            metrics,
            dtype=torch.float64,
            device=_collective_device(context),
        )
    else:
        if metrics is not None:
            raise ValueError("non-primary metrics must be None")
        values = torch.zeros(
            2,
            dtype=torch.float64,
            device=_collective_device(context),
        )
    dist.broadcast(values, src=0)
    return float(values[0].item()), float(values[1].item())
