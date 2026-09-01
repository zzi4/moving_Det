from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from moving_det.ml.distributed import DistributedContext


_MODEL_NAMES = ("baseline", "mg_vtod", "mg_vtod_8class", "lstfe")


def _positive_integer(value: str) -> int:
    try:
        converted = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc
    if converted <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moving_det.distributed_train",
        description="Internal two-process VRUD OBB training worker",
    )
    parser.add_argument("--model", choices=_MODEL_NAMES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--alignment-cache", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=_positive_integer)
    parser.add_argument(
        "--train-scope",
        choices=("full", "temporal"),
        default="full",
    )
    return parser


def _environment_integer(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"torchrun environment is missing {name}")
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"torchrun environment has invalid {name}"
        ) from exc


def initialize_distributed_context() -> DistributedContext:
    context = DistributedContext(
        rank=_environment_integer("RANK"),
        local_rank=_environment_integer("LOCAL_RANK"),
        world_size=_environment_integer("WORLD_SIZE"),
        backend="nccl",
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("distributed worker requires two CUDA devices")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    torch.cuda.set_device(context.local_rank)
    try:
        dist.init_process_group(backend=context.backend, init_method="env://")
    except BaseException:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    return context


def _destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run_worker(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], Any] | None = None,
    trainer: Callable[..., Any] | None = None,
    context_initializer: Callable[[], DistributedContext] | None = None,
    process_group_destroyer: Callable[[], None] | None = None,
    validator: Callable[..., dict[str, float]] | None = None,
) -> int:
    if config_loader is None:
        from moving_det.temporal_config import load_temporal_config

        config_loader = load_temporal_config
    if trainer is None:
        from moving_det.ml.training import train_model

        trainer = train_model
    if validator is None:
        from moving_det.vru_cli import _loader_task11_metrics

        validator = _loader_task11_metrics
    selected_initializer = (
        context_initializer or initialize_distributed_context
    )
    selected_destroyer = process_group_destroyer or _destroy_process_group

    context: DistributedContext | None = None
    try:
        context = selected_initializer()
        if not isinstance(context, DistributedContext):
            raise RuntimeError(
                "distributed initializer returned an invalid context"
            )
        cfg = config_loader(Path(args.config))
        if args.weights is not None:
            cfg = replace(cfg, pretrained_weights=str(args.weights))
        if args.alignment_cache is not None:
            cfg = replace(
                cfg,
                output_root=Path(args.alignment_cache).parent,
            )

        from moving_det.ml.training import TrainingHooks

        hooks = TrainingHooks(
            validator=lambda model, loader, device: validator(
                model,
                loader,
                device,
                cfg,
                distributed_context=context,
            )
        )
        trainer(
            args.model,
            cfg,
            Path(args.manifest),
            Path(args.output),
            max_steps=args.max_steps,
            train_scope=args.train_scope,
            init_checkpoint=args.init_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
            warm_start_checkpoint=args.warm_start_checkpoint,
            hooks=hooks,
            distributed_context=context,
        )
        return 0
    finally:
        if context is not None:
            selected_destroyer()


def main(argv: Sequence[str] | None = None) -> int:
    return run_worker(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
