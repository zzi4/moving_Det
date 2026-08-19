from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _distributed_api():
    try:
        from moving_det.ml.distributed import (
            DistributedContext,
            broadcast_metric_pair,
            distributed_mean,
            distributed_sum_count,
            gather_rank_objects,
        )
    except ModuleNotFoundError:
        pytest.fail("moving_det.ml.distributed is missing")
    return (
        DistributedContext,
        broadcast_metric_pair,
        distributed_mean,
        distributed_sum_count,
        gather_rank_objects,
    )


def _collective_worker(
    rank: int,
    init_file: str,
    output_dir: str,
) -> None:
    (
        DistributedContext,
        broadcast_metric_pair,
        distributed_mean,
        distributed_sum_count,
        gather_rank_objects,
    ) = _distributed_api()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        context = DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=2,
            backend="gloo",
        )
        local_sum = 2.0 if rank == 0 else 6.0
        reduced_sum, reduced_count = distributed_sum_count(
            local_sum,
            2,
            context,
        )
        mean = distributed_mean(local_sum, context)
        gathered = gather_rank_objects({"rank": rank}, context)
        metrics = broadcast_metric_pair(
            (0.25, 0.75) if context.is_primary else None,
            context,
        )
        torch.save(
            {
                "sum": reduced_sum,
                "count": reduced_count,
                "mean": mean,
                "gathered": gathered,
                "metrics": metrics,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("rank", "local_rank", "world_size", "backend"),
    [
        (-1, 0, 2, "gloo"),
        (2, 0, 2, "gloo"),
        (0, -1, 2, "gloo"),
        (0, 0, 1, "gloo"),
        (0, 0, 2, ""),
    ],
)
def test_distributed_context_rejects_invalid_topology(
    rank,
    local_rank,
    world_size,
    backend,
):
    DistributedContext, *_ = _distributed_api()

    with pytest.raises(ValueError):
        DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
        )


def test_two_rank_collectives_reduce_gather_and_broadcast(tmp_path):
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _collective_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    rank_zero = torch.load(
        tmp_path / "rank-0.pt",
        map_location="cpu",
        weights_only=False,
    )
    rank_one = torch.load(
        tmp_path / "rank-1.pt",
        map_location="cpu",
        weights_only=False,
    )
    for result in (rank_zero, rank_one):
        assert result["sum"] == pytest.approx(8.0)
        assert result["count"] == 4
        assert result["mean"] == pytest.approx(4.0)
        assert result["metrics"] == pytest.approx((0.25, 0.75))
    assert rank_zero["gathered"] == ({"rank": 0}, {"rank": 1})
    assert rank_one["gathered"] is None


def test_distributed_worker_passes_temporal_scope_to_trainer(tmp_path):
    from moving_det.distributed_train import build_parser, run_worker
    from moving_det.ml.distributed import DistributedContext

    context = DistributedContext(
        rank=1,
        local_rank=1,
        world_size=2,
        backend="nccl",
    )
    captured = {}

    def trainer(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    args = build_parser().parse_args(
        [
            "--model",
            "mg_vtod",
            "--config",
            "config.yaml",
            "--manifest",
            "manifest",
            "--output",
            str(tmp_path / "checkpoints"),
            "--train-scope",
            "temporal",
        ]
    )

    assert run_worker(
        args,
        config_loader=lambda _path: object(),
        trainer=trainer,
        context_initializer=lambda: context,
        process_group_destroyer=lambda: None,
        validator=lambda *_args, **_kwargs: {},
    ) == 0
    assert captured["train_scope"] == "temporal"


def test_distributed_worker_forwards_warm_start_checkpoint(tmp_path):
    from moving_det.distributed_train import build_parser, run_worker
    from moving_det.ml.distributed import DistributedContext

    context = DistributedContext(
        rank=1,
        local_rank=1,
        world_size=2,
        backend="nccl",
    )
    captured = {}
    warm_start = tmp_path / "last.pt"

    def trainer(*_args, **kwargs):
        captured.update(kwargs)
        return object()

    args = build_parser().parse_args(
        [
            "--model",
            "mg_vtod",
            "--config",
            "config.yaml",
            "--manifest",
            "manifest",
            "--output",
            str(tmp_path / "checkpoints"),
            "--warm-start-checkpoint",
            str(warm_start),
            "--train-scope",
            "full",
        ]
    )

    assert run_worker(
        args,
        config_loader=lambda _path: object(),
        trainer=trainer,
        context_initializer=lambda: context,
        process_group_destroyer=lambda: None,
        validator=lambda *_args, **_kwargs: {},
    ) == 0
    assert captured["warm_start_checkpoint"] == warm_start
    assert captured["train_scope"] == "full"
