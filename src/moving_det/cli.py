from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from moving_det.config import load_config
from moving_det.data.labelme import load_sequence, summarize_sequence
from moving_det.experiment import (
    calibrate,
    evaluate,
    run_method,
    write_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moving-det")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect-data")
    inspect_parser.add_argument("--config", type=Path, required=True)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--sequence",
        choices=("calibration", "evaluation"),
        required=True,
    )
    run_parser.add_argument(
        "--method",
        choices=(
            "frame_diff",
            "mog2",
            "temporal_median",
            "multiscale",
            "multiscale_tubelet",
        ),
        required=True,
    )
    run_parser.add_argument("--scale", type=float, required=True)
    run_parser.add_argument("--threshold", type=float, required=True)
    run_parser.add_argument("--frame-start", type=int)
    run_parser.add_argument("--frame-end", type=int)
    run_parser.add_argument("--output", type=Path, required=True)

    calibrate_parser = commands.add_parser("calibrate")
    calibrate_parser.add_argument("--config", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--calibration", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)

    report_parser = commands.add_parser("report")
    report_parser.add_argument("--metrics", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    return parser


def _inspect(config_path: Path) -> int:
    config = load_config(config_path)
    for sequence_name in (
        config.calibration_sequence,
        config.evaluation_sequence,
    ):
        sequence = load_sequence(
            config.data_root / sequence_name,
            fps=config.fps,
        )
        print(
            json.dumps(
                {
                    "sequence_id": sequence.sequence_id,
                    **summarize_sequence(sequence),
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
    return 0


def _selected_sequence(config, selection: str):
    sequence_name = (
        config.calibration_sequence
        if selection == "calibration"
        else config.evaluation_sequence
    )
    return load_sequence(config.data_root / sequence_name, fps=config.fps)


def _frame_subset(sequence, frame_start: int | None, frame_end: int | None):
    if frame_start is None and frame_end is None:
        return sequence
    if frame_start is None or frame_end is None:
        raise ValueError("--frame-start and --frame-end must be used together")
    if frame_start > frame_end:
        raise ValueError("--frame-start must not exceed --frame-end")
    selected = tuple(
        frame
        for frame in sequence.frames
        if frame_start <= frame.frame_index <= frame_end
    )
    if not selected:
        raise ValueError("requested frame range does not exist in the sequence")
    if (
        selected[0].frame_index != frame_start
        or selected[-1].frame_index != frame_end
        or len(selected) != frame_end - frame_start + 1
    ):
        raise ValueError("requested frame range must be complete and consecutive")
    return replace(sequence, frames=selected)


def _dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command == "inspect-data":
        return _inspect(arguments.config)
    if arguments.command == "run":
        config = load_config(arguments.config)
        sequence = _frame_subset(
            _selected_sequence(config, arguments.sequence),
            arguments.frame_start,
            arguments.frame_end,
        )
        artifacts = run_method(
            config=config,
            sequence=sequence,
            method_name=arguments.method,
            scale=arguments.scale,
            thresholds=(arguments.threshold,),
            output_dir=arguments.output,
        )
        print(artifacts.root)
        return 0
    if arguments.command == "calibrate":
        config = load_config(arguments.config)
        print(calibrate(config, arguments.output))
        return 0
    if arguments.command == "evaluate":
        config = load_config(arguments.config)
        print(evaluate(config, arguments.calibration, arguments.output))
        return 0
    if arguments.command == "report":
        print(write_report(arguments.metrics, arguments.output))
        return 0
    raise AssertionError(f"unhandled command {arguments.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        return _dispatch(arguments)
    except (OSError, TypeError, ValueError) as exc:
        print(f"moving-det: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
