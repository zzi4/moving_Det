from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from numbers import Real
from pathlib import Path

import cv2
import numpy as np
import scipy
import shapely
import yaml
from PIL import __version__ as pillow_version

from moving_det.config import ExperimentConfig
from moving_det.evaluation.metrics import (
    CalibrationCandidate,
    CalibrationChoice,
    EvaluationReport,
    evaluate_sequence,
    select_calibration_result,
)
from moving_det.models import Component, MotionEvidence, Proposal, SequenceData
from moving_det.motion.masks import (
    clean_binary_mask,
    extract_components,
    threshold_and_clean,
)
from moving_det.motion.methods import create_method
from moving_det.motion.tubelets import (
    link_tubelets,
    proposals_for_frame,
    proposals_from_components,
)


_METHOD_NAMES = (
    "frame_diff",
    "mog2",
    "temporal_median",
    "multiscale",
    "multiscale_tubelet",
)
_Z_THRESHOLDS = (3.0, 4.0, 5.0, 6.0)
_MOG2_VAR_THRESHOLDS = (9.0, 16.0, 25.0)
_DEFAULT_MOVING_THRESHOLD = 3.0
_PREVIEW_MAX_WIDTH = 960
_PREVIEW_MAX_HEIGHT = 540
_SCHEMA_VERSION = 1
_CALIBRATION_CONFIG_SCHEMA = (
    ("random_seed", int, False),
    ("fps", int, False),
    ("window_radius", int, False),
    ("offsets", int, True),
    ("scale_factors", float, True),
    ("mad_floor", float, False),
    ("mad_clip", float, False),
    ("threshold_candidates", float, True),
    ("mog2_history", int, False),
    ("mog2_var_threshold_candidates", float, True),
    ("ecc_min_correlation", float, False),
    ("ecc_max_translation", float, False),
    ("ecc_max_rotation_degrees", float, False),
    ("close_kernel", int, False),
    ("min_component_area", int, False),
    ("tubelet_link_radius", int, False),
    ("tubelet_min_frames", int, False),
    ("obb_padding_factor", float, False),
    ("moving_displacement_frames", int, False),
    ("moving_thresholds", float, True),
    ("primary_iou_thresholds", float, True),
    ("max_false_proposals_per_100_gt", float, False),
)
_GATE_LABELS = {
    "tubelet_recall_improvement": (
        "Recall@rIoU 0.25 improvement over best single baseline"
    ),
    "native_center_in_gt_recall": "Native Center-in-GT recall",
    "native_recall_025": "Native Recall@rIoU 0.25",
    "scale_recall_drop": "Native-to-0.7 Recall@rIoU 0.25 drop",
    "moving_frame_track_coverage": "Moving-frame track coverage",
    "mean_extra_fragments": "Mean extra fragments per GT track",
}
_PER_FRAME_FIELDS = (
    "frame_index",
    "is_primary",
    "moving_gt_count",
    "matched_gt_count_025",
    "recall_025",
    "matched_gt_count_050",
    "recall_050",
    "center_in_gt_count",
    "center_in_gt_recall",
    "mask_coverage_mean",
    "difficult_moving_gt_count",
    "proposal_count",
    "false_proposal_count",
)
_PER_TRACK_FIELDS = (
    "track_id",
    "first_moving_frame",
    "first_detection_frame",
    "first_detection_delay_frames",
    "moving_frame_count",
    "detected_moving_frame_count",
    "moving_frame_coverage",
    "extra_tubelet_fragments",
)


@dataclass(frozen=True)
class RunArtifacts:
    root: Path
    config_path: Path
    metrics_path: Path
    per_frame_path: Path
    per_track_path: Path
    run_metadata_path: Path
    proposals_path: Path
    frame_cache_dir: Path


@dataclass
class _ThresholdState:
    threshold: float
    mask_dir: Path
    preview_mask_dir: Path
    score_dir: Path
    components_by_frame: dict[int, tuple[Component, ...]] | None
    proposals_by_method: dict[str, dict[int, tuple[Proposal, ...]]]


@dataclass(frozen=True)
class _CandidateResult:
    threshold: float
    report: EvaluationReport
    sensitivity: Mapping[str, Mapping[str, object]]
    proposals_by_frame: Mapping[int, tuple[Proposal, ...]]
    state: _ThresholdState


@dataclass(frozen=True)
class _MethodRunResult:
    artifacts: RunArtifacts
    candidates: tuple[_CandidateResult, ...]
    choice: CalibrationChoice


class _DiskMaskMapping(Mapping[int, np.ndarray]):
    def __init__(
        self,
        root: Path,
        frame_indices: Sequence[int],
    ) -> None:
        self._root = root
        self._frame_indices = tuple(frame_indices)
        self._frame_index_set = frozenset(self._frame_indices)

    def __len__(self) -> int:
        return len(self._frame_indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self._frame_indices)

    def __getitem__(self, frame_index: int) -> np.ndarray:
        if frame_index not in self._frame_index_set:
            raise KeyError(frame_index)
        path = self._root / f"{frame_index:06d}.npz"
        with np.load(path, allow_pickle=False) as stored:
            shape = tuple(int(value) for value in stored["shape"])
            count = shape[0] * shape[1]
            unpacked = np.unpackbits(stored["packed"], count=count)
        return unpacked.reshape(shape).astype(np.uint8, copy=False)


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            raise ValueError("NaN cannot be serialized as strict JSON")
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        converted = float(value)
        if (
            math.isfinite(converted)
            and bool(type(value)(converted) == value)
        ):
            return converted
        return np.format_float_scientific(
            value,
            unique=True,
            trim="k",
        )
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Real):
        converted = float(value)
        if math.isnan(converted):
            raise ValueError("NaN cannot be serialized as strict JSON")
        if converted == math.inf:
            return "Infinity"
        if converted == -math.inf:
            raise ValueError("-Infinity cannot be serialized as strict JSON")
        return converted
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_ready(value),
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_strict_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(
            f"calibration must be strict JSON; found nonstandard {value}"
        )

    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration must be strict JSON: {exc}") from exc


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            _json_ready(value),
            stream,
            sort_keys=False,
        )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Real) and not isinstance(value, bool):
        converted = float(value)
        if converted == math.inf:
            return "inf"
        if converted == -math.inf:
            return "-inf"
        if math.isnan(converted):
            raise ValueError("NaN cannot be serialized to CSV")
    return value


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    field_names: Sequence[str],
) -> None:
    normalized_fields = tuple(field_names)
    expected_fields = set(normalized_fields)
    if (
        not normalized_fields
        or len(expected_fields) != len(normalized_fields)
    ):
        raise ValueError("CSV field schema must be non-empty and unique")
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError(
                "CSV row schema does not match the fixed artifact schema"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=normalized_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key, ""))
                    for key in normalized_fields
                }
            )


def _write_proposals(
    path: Path,
    proposals_by_frame: Mapping[int, Sequence[Proposal]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for frame_index in sorted(proposals_by_frame):
            for candidate in proposals_by_frame[frame_index]:
                payload = {
                    "frame_index": candidate.frame_index,
                    "obb": candidate.obb,
                    "motion_score": candidate.motion_score,
                    "tubelet_id": candidate.tubelet_id,
                }
                json.dump(
                    _json_ready(payload),
                    stream,
                    allow_nan=False,
                    sort_keys=True,
                )
                stream.write("\n")


def _config_values(config: ExperimentConfig) -> dict[str, object]:
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
    }


def _package_version(distribution: str, fallback: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "shapely": shapely.__version__,
        "pillow": pillow_version,
        "moving-det": _package_version("moving-det", "0.1.0"),
    }


def _metadata(
    config: ExperimentConfig,
    sequence: SequenceData,
    *,
    method: str,
    scale: float | None,
    threshold: float | None,
    threshold_source: Path | None = None,
) -> dict[str, object]:
    frame_indices = tuple(frame.frame_index for frame in sequence.frames)
    input_path = (
        sequence.frames[0].image_path.parent.resolve()
        if sequence.frames
        else None
    )
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": method,
        "scale": scale,
        "threshold": threshold,
        "sequence_id": sequence.sequence_id,
        "input_path": input_path,
        "frame_range": (
            [min(frame_indices), max(frame_indices)]
            if frame_indices
            else []
        ),
        "random_seed": config.random_seed,
        "determinism": {
            "random_seed": config.random_seed,
            "opencv_threads": 1,
            "streaming_evidence": True,
        },
        "versions": _versions(),
    }
    if threshold_source is not None:
        payload["threshold_source"] = threshold_source.resolve()
    return payload


def _prepare_determinism(config: ExperimentConfig) -> None:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    cv2.setNumThreads(1)


def _validate_fixed_candidates(config: ExperimentConfig) -> None:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if (
        not isinstance(config.threshold_candidates, tuple)
        or config.threshold_candidates != _Z_THRESHOLDS
        or not isinstance(config.mog2_var_threshold_candidates, tuple)
        or config.mog2_var_threshold_candidates != _MOG2_VAR_THRESHOLDS
    ):
        raise ValueError(
            "configuration must use the fixed POC candidates "
            f"{_Z_THRESHOLDS} and {_MOG2_VAR_THRESHOLDS}"
        )


def _validate_run_inputs(
    config: ExperimentConfig,
    sequence: SequenceData,
    method_name: str,
    scale: float,
    thresholds: Sequence[float],
) -> tuple[float, tuple[float, ...]]:
    _validate_fixed_candidates(config)
    if not isinstance(sequence, SequenceData):
        raise TypeError("sequence must be a SequenceData")
    if method_name not in _METHOD_NAMES:
        raise ValueError(f"unknown method {method_name!r}")
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise ValueError("scale must be a configured finite scale")
    normalized_scale = float(scale)
    if (
        not math.isfinite(normalized_scale)
        or normalized_scale not in config.scale_factors
    ):
        raise ValueError("scale must be a configured finite scale")

    allowed = (
        _MOG2_VAR_THRESHOLDS
        if method_name == "mog2"
        else _Z_THRESHOLDS
    )
    normalized = []
    for threshold in thresholds:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("threshold must be a configured finite value")
        converted = float(threshold)
        if not math.isfinite(converted) or converted not in allowed:
            raise ValueError(
                f"threshold for {method_name} must be one of {allowed}"
            )
        normalized.append(converted)
    if not normalized:
        raise ValueError("at least one threshold is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must not contain duplicates")
    return normalized_scale, tuple(normalized)


def _validate_processing_sequence(
    sequence: SequenceData,
    processing_sequence: SequenceData | None,
) -> SequenceData:
    if processing_sequence is None:
        return sequence
    if not isinstance(processing_sequence, SequenceData):
        raise TypeError("processing_sequence must be a SequenceData")
    if (
        processing_sequence.sequence_id != sequence.sequence_id
        or processing_sequence.width != sequence.width
        or processing_sequence.height != sequence.height
        or processing_sequence.fps != sequence.fps
    ):
        raise ValueError(
            "processing_sequence must describe the same source sequence"
        )
    processing_by_index = {
        frame.frame_index: frame
        for frame in processing_sequence.frames
    }
    if len(processing_by_index) != len(processing_sequence.frames):
        raise ValueError("processing_sequence frame indices must be unique")
    processing_indices = tuple(processing_by_index)
    if any(
        current != previous + 1
        for previous, current in zip(
            processing_indices,
            processing_indices[1:],
        )
    ):
        raise ValueError(
            "processing_sequence frames must be consecutive and ordered"
        )
    for frame in sequence.frames:
        if processing_by_index.get(frame.frame_index) != frame:
            raise ValueError(
                "processing_sequence must contain every output frame unchanged"
            )
    return processing_sequence


def _preview(array: np.ndarray, interpolation: int) -> np.ndarray:
    height, width = array.shape
    ratio = min(
        1.0,
        _PREVIEW_MAX_WIDTH / width,
        _PREVIEW_MAX_HEIGHT / height,
    )
    if ratio == 1.0:
        return np.asarray(array).copy()
    target = (
        max(1, int(round(width * ratio))),
        max(1, int(round(height * ratio))),
    )
    return cv2.resize(array, target, interpolation=interpolation)


def _preview_score(score: np.ndarray) -> np.ndarray:
    resized = _preview(score, cv2.INTER_AREA)
    return np.clip(np.rint(resized * 255.0), 0, 255).astype(np.uint8)


def _preview_mask(mask: np.ndarray) -> np.ndarray:
    resized = _preview(mask, cv2.INTER_NEAREST)
    return np.not_equal(resized, 0).astype(np.uint8)


def _store_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed=np.packbits(mask, axis=None),
        shape=np.asarray(mask.shape, dtype=np.int64),
    )


def _store_preview(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def _scaled_ignore_polygons(
    polygons: Sequence[Sequence[Sequence[float]]],
    scale: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        tuple(
            (float(x) * scale, float(y) * scale)
            for x, y in polygon
        )
        for polygon in polygons
    )


def _method_stream(
    method: object,
    sequence: SequenceData,
    scale: float,
    center_indices: Sequence[int],
) -> Iterator[MotionEvidence]:
    iterator = getattr(method, "iter_run", None)
    if not callable(iterator):
        raise RuntimeError(
            "motion method does not provide bounded-memory iter_run()"
        )
    yield from iterator(
        sequence,
        scale,
        center_indices=center_indices,
    )


def _new_state(
    cache_root: Path,
    threshold: float,
    *,
    needs_components: bool,
    method_names: Sequence[str],
    score_dir: Path,
) -> _ThresholdState:
    key = f"{threshold:g}".replace(".", "_")
    return _ThresholdState(
        threshold=threshold,
        mask_dir=cache_root / f"masks-{key}",
        preview_mask_dir=cache_root / f"preview-masks-{key}",
        score_dir=score_dir,
        components_by_frame={} if needs_components else None,
        proposals_by_method={
            method_name: {}
            for method_name in method_names
            if method_name != "multiscale_tubelet"
        },
    )


def _consume_stream(
    *,
    config: ExperimentConfig,
    sequence: SequenceData,
    processing_sequence: SequenceData,
    scale: float,
    method_names: Sequence[str],
    thresholds_by_method: Mapping[str, tuple[float, ...]],
    cache_root: Path,
) -> dict[float, _ThresholdState]:
    frame_by_index = {
        frame.frame_index: frame
        for frame in sequence.frames
    }
    expected_indices = tuple(frame_by_index)
    evidence_method = (
        "multiscale"
        if set(method_names) <= {"multiscale", "multiscale_tubelet"}
        else method_names[0]
    )
    unique_thresholds = tuple(
        dict.fromkeys(
            threshold
            for method_name in method_names
            for threshold in thresholds_by_method[method_name]
        )
    )
    states: dict[float, _ThresholdState] = {}

    if evidence_method == "mog2":
        for threshold in unique_thresholds:
            score_dir = cache_root / f"scores-{threshold:g}"
            states[threshold] = _new_state(
                cache_root,
                threshold,
                needs_components=False,
                method_names=method_names,
                score_dir=score_dir,
            )
            method = create_method(
                "mog2",
                config,
                var_threshold=threshold,
            )
            observed = []
            for evidence in _method_stream(
                method,
                processing_sequence,
                scale,
                expected_indices,
            ):
                observed.append(evidence.frame_index)
                mask = clean_binary_mask(
                    np.not_equal(evidence.fused_z, 0).astype(np.uint8),
                    config,
                )
                _consume_evidence(
                    config,
                    evidence,
                    mask,
                    scale,
                    frame_by_index,
                    states[threshold],
                    method_names,
                    thresholds_by_method,
                )
            if tuple(observed) != expected_indices:
                raise ValueError("motion method did not yield every frame in order")
        return states

    score_dir = cache_root / "scores"
    needs_components = "multiscale_tubelet" in method_names
    for threshold in unique_thresholds:
        states[threshold] = _new_state(
            cache_root,
            threshold,
            needs_components=(
                needs_components
                and threshold
                in thresholds_by_method.get("multiscale_tubelet", ())
            ),
            method_names=method_names,
            score_dir=score_dir,
        )
    method = create_method(evidence_method, config)
    observed = []
    for evidence in _method_stream(
        method,
        processing_sequence,
        scale,
        expected_indices,
    ):
        observed.append(evidence.frame_index)
        score_path = score_dir / f"{evidence.frame_index:06d}.npy"
        _store_preview(score_path, _preview_score(evidence.fused_score))
        for threshold, state in states.items():
            mask = threshold_and_clean(
                evidence.fused_z,
                threshold,
                config,
            )
            _consume_evidence(
                config,
                evidence,
                mask,
                scale,
                frame_by_index,
                state,
                method_names,
                thresholds_by_method,
                score_already_stored=True,
            )
    if tuple(observed) != expected_indices:
        raise ValueError("motion method did not yield every frame in order")
    return states


def _consume_evidence(
    config: ExperimentConfig,
    evidence: MotionEvidence,
    mask: np.ndarray,
    scale: float,
    frame_by_index: Mapping[int, object],
    state: _ThresholdState,
    method_names: Sequence[str],
    thresholds_by_method: Mapping[str, tuple[float, ...]],
    *,
    score_already_stored: bool = False,
) -> None:
    frame_index = evidence.frame_index
    if frame_index not in frame_by_index:
        raise ValueError(f"motion method yielded unknown frame {frame_index}")
    if not score_already_stored:
        _store_preview(
            state.score_dir / f"{frame_index:06d}.npy",
            _preview_score(evidence.fused_score),
        )
    _store_mask(state.mask_dir / f"{frame_index:06d}.npz", mask)
    _store_preview(
        state.preview_mask_dir / f"{frame_index:06d}.npy",
        _preview_mask(mask),
    )
    components = extract_components(
        frame_index,
        mask,
        evidence.fused_score,
        config,
    )
    if state.components_by_frame is not None:
        state.components_by_frame[frame_index] = components

    frame = frame_by_index[frame_index]
    ignore_polygons = _scaled_ignore_polygons(
        getattr(frame, "ignore_polygons"),
        scale,
    )
    for method_name in method_names:
        if (
            method_name == "multiscale_tubelet"
            or state.threshold not in thresholds_by_method[method_name]
        ):
            continue
        state.proposals_by_method[method_name][frame_index] = (
            proposals_from_components(
                frame_index,
                components,
                ignore_polygons,
                config,
            )
        )


def _finish_tubelets(
    config: ExperimentConfig,
    sequence: SequenceData,
    scale: float,
    states: Mapping[float, _ThresholdState],
    thresholds: Sequence[float],
) -> None:
    frame_by_index = {
        frame.frame_index: frame
        for frame in sequence.frames
    }
    for threshold in thresholds:
        state = states[threshold]
        if state.components_by_frame is None:
            raise RuntimeError("tubelet threshold did not retain components")
        tubelets = link_tubelets(state.components_by_frame, config)
        proposals = {}
        for frame_index, frame in frame_by_index.items():
            proposals[frame_index] = proposals_for_frame(
                frame_index,
                tubelets,
                _scaled_ignore_polygons(frame.ignore_polygons, scale),
                config,
            )
        state.proposals_by_method["multiscale_tubelet"] = proposals


def _sensitivity_reports(
    config: ExperimentConfig,
    sequence: SequenceData,
    proposals_by_frame: Mapping[int, tuple[Proposal, ...]],
    masks: Mapping[int, np.ndarray],
    scale: float,
) -> tuple[EvaluationReport, dict[str, Mapping[str, object]]]:
    reports = {}
    primary_report = None
    for moving_threshold in config.moving_thresholds:
        report = evaluate_sequence(
            sequence,
            proposals_by_frame,
            masks,
            moving_threshold=float(moving_threshold),
            iou_thresholds=config.primary_iou_thresholds,
            scale=scale,
        )
        reports[f"{float(moving_threshold):g}"] = report.aggregate
        if float(moving_threshold) == _DEFAULT_MOVING_THRESHOLD:
            primary_report = report
    if primary_report is None:
        raise ValueError("moving_thresholds must include the fixed default 3.0")
    return primary_report, reports


def _artifacts(root: Path) -> RunArtifacts:
    return RunArtifacts(
        root=root,
        config_path=root / "config.yaml",
        metrics_path=root / "metrics.json",
        per_frame_path=root / "per_frame.csv",
        per_track_path=root / "per_track.csv",
        run_metadata_path=root / "run.json",
        proposals_path=root / "proposals.jsonl",
        frame_cache_dir=root / "frames",
    )


def _candidate_payload(candidate: _CandidateResult) -> dict[str, object]:
    return {
        "threshold": candidate.threshold,
        "aggregate": candidate.report.aggregate,
        "boundary": candidate.report.boundary,
        "strata": candidate.report.strata,
        "moving_threshold_sensitivity": candidate.sensitivity,
    }


def _select_choice(
    config: ExperimentConfig,
    method_name: str,
    candidates: Sequence[_CandidateResult],
) -> CalibrationChoice:
    parameter_name = (
        "varThreshold"
        if method_name == "mog2"
        else "z_threshold"
    )
    values = tuple(
        CalibrationCandidate(
            parameter_name=parameter_name,
            parameter_value=candidate.threshold,
            recall_025=float(candidate.report.aggregate["recall_025"]),
            fp_per_100_gt=float(
                candidate.report.aggregate[
                    "false_proposals_per_100_moving_gt"
                ]
            ),
        )
        for candidate in candidates
    )
    if len(values) == 1:
        false_positive_rate = values[0].fp_per_100_gt
        satisfied = (
            math.isfinite(false_positive_rate)
            and false_positive_rate
            <= config.max_false_proposals_per_100_gt
        )
        return CalibrationChoice(values[0], satisfied)
    return select_calibration_result(
        values,
        max_fp_per_100_gt=config.max_false_proposals_per_100_gt,
    )


def _write_frames(
    artifacts: RunArtifacts,
    sequence: SequenceData,
    state: _ThresholdState,
) -> None:
    artifacts.frame_cache_dir.mkdir(parents=True)
    for frame in sequence.frames:
        frame_index = frame.frame_index
        score = np.load(
            state.score_dir / f"{frame_index:06d}.npy",
            allow_pickle=False,
        )
        mask = np.load(
            state.preview_mask_dir / f"{frame_index:06d}.npy",
            allow_pickle=False,
        )
        np.savez_compressed(
            artifacts.frame_cache_dir / f"{frame_index:06d}.npz",
            preview_score=score.astype(np.uint8, copy=False),
            preview_mask=mask.astype(np.uint8, copy=False),
        )


def _write_method_artifacts(
    *,
    config: ExperimentConfig,
    sequence: SequenceData,
    method_name: str,
    scale: float,
    output_dir: Path,
    candidates: tuple[_CandidateResult, ...],
) -> _MethodRunResult:
    choice = _select_choice(config, method_name, candidates)
    selected = next(
        candidate
        for candidate in candidates
        if candidate.threshold == float(choice.candidate.parameter_value)
    )
    artifacts = _artifacts(output_dir)
    artifacts.root.mkdir(parents=True)

    parameter_name = choice.candidate.parameter_name
    _write_yaml(
        artifacts.config_path,
        {
            **_config_values(config),
            "sequence_id": sequence.sequence_id,
            "method": method_name,
            "scale": scale,
            "threshold_parameter": parameter_name,
            "threshold": selected.threshold,
        },
    )
    _write_json(
        artifacts.metrics_path,
        {
            "schema_version": _SCHEMA_VERSION,
            "method": method_name,
            "sequence_id": sequence.sequence_id,
            "scale": scale,
            "threshold_parameter": parameter_name,
            "threshold": selected.threshold,
            "constraint_satisfied": choice.constraint_satisfied,
            "aggregate": selected.report.aggregate,
            "boundary": selected.report.boundary,
            "strata": selected.report.strata,
            "moving_threshold_sensitivity": selected.sensitivity,
            "candidates": [
                _candidate_payload(candidate)
                for candidate in candidates
            ],
            "gate_passed": None,
            "gates": {},
        },
    )
    _write_csv(
        artifacts.per_frame_path,
        selected.report.per_frame,
        _PER_FRAME_FIELDS,
    )
    _write_csv(
        artifacts.per_track_path,
        selected.report.per_track,
        _PER_TRACK_FIELDS,
    )
    _write_proposals(
        artifacts.proposals_path,
        selected.proposals_by_frame,
    )
    _write_frames(artifacts, sequence, selected.state)
    _write_json(
        artifacts.run_metadata_path,
        _metadata(
            config,
            sequence,
            method=method_name,
            scale=scale,
            threshold=selected.threshold,
        ),
    )
    return _MethodRunResult(
        artifacts=artifacts,
        candidates=candidates,
        choice=choice,
    )


def _compute_group(
    *,
    config: ExperimentConfig,
    sequence: SequenceData,
    method_names: tuple[str, ...],
    scale: float,
    thresholds_by_method: Mapping[str, tuple[float, ...]],
    output_dirs: Mapping[str, Path],
    work_dir: Path,
    processing_sequence: SequenceData | None = None,
) -> dict[str, _MethodRunResult]:
    _prepare_determinism(config)
    work_dir.mkdir(parents=True)
    processing_sequence = (
        sequence
        if processing_sequence is None
        else processing_sequence
    )
    states = _consume_stream(
        config=config,
        sequence=sequence,
        processing_sequence=processing_sequence,
        scale=scale,
        method_names=method_names,
        thresholds_by_method=thresholds_by_method,
        cache_root=work_dir,
    )
    if "multiscale_tubelet" in method_names:
        _finish_tubelets(
            config,
            sequence,
            scale,
            states,
            thresholds_by_method["multiscale_tubelet"],
        )

    frame_indices = tuple(frame.frame_index for frame in sequence.frames)
    results = {}
    for method_name in method_names:
        candidates = []
        for threshold in thresholds_by_method[method_name]:
            state = states[threshold]
            proposals = state.proposals_by_method[method_name]
            masks = _DiskMaskMapping(state.mask_dir, frame_indices)
            report, sensitivity = _sensitivity_reports(
                config,
                sequence,
                proposals,
                masks,
                scale,
            )
            candidates.append(
                _CandidateResult(
                    threshold=threshold,
                    report=report,
                    sensitivity=sensitivity,
                    proposals_by_frame=proposals,
                    state=state,
                )
            )
        results[method_name] = _write_method_artifacts(
            config=config,
            sequence=sequence,
            method_name=method_name,
            scale=scale,
            output_dir=output_dirs[method_name],
            candidates=tuple(candidates),
        )
    return results


def _new_work_root(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )


def run_method(
    config: ExperimentConfig,
    sequence: SequenceData,
    method_name: str,
    scale: float,
    thresholds: Sequence[float],
    output_dir: Path,
    processing_sequence: SequenceData | None = None,
) -> RunArtifacts:
    scale, normalized_thresholds = _validate_run_inputs(
        config,
        sequence,
        method_name,
        scale,
        thresholds,
    )
    processing_sequence = _validate_processing_sequence(
        sequence,
        processing_sequence,
    )
    output_dir = Path(output_dir)
    work_root = _new_work_root(output_dir)
    staged_output = work_root / "artifact"
    try:
        _compute_group(
            config=config,
            sequence=sequence,
            method_names=(method_name,),
            scale=scale,
            thresholds_by_method={
                method_name: normalized_thresholds,
            },
            output_dirs={method_name: staged_output},
            work_dir=work_root / "cache",
            processing_sequence=processing_sequence,
        )
        shutil.rmtree(work_root / "cache")
        os.replace(staged_output, output_dir)
        return _artifacts(output_dir)
    except BaseException:
        shutil.rmtree(work_root, ignore_errors=True)
        raise
    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


def _scale_key(scale: float) -> str:
    return str(float(scale))


def _artifact_relative_path(method_name: str, scale: float) -> Path:
    return Path(method_name) / f"scale-{_scale_key(scale)}"


def _calibration_entry(
    result: _MethodRunResult,
) -> dict[str, object]:
    return {
        "parameter_name": result.choice.candidate.parameter_name,
        "candidates": [
            CalibrationCandidate(
                parameter_name=result.choice.candidate.parameter_name,
                parameter_value=candidate.threshold,
                recall_025=float(candidate.report.aggregate["recall_025"]),
                fp_per_100_gt=float(
                    candidate.report.aggregate[
                        "false_proposals_per_100_moving_gt"
                    ]
                ),
            )
            for candidate in result.candidates
        ],
        "selected": result.choice.candidate,
        "constraint_satisfied": result.choice.constraint_satisfied,
    }


def _calibration_config_fingerprint(
    config: ExperimentConfig,
) -> dict[str, object]:
    fingerprint: dict[str, object] = {}
    for field_name, item_type, repeated in _CALIBRATION_CONFIG_SCHEMA:
        value = getattr(config, field_name)
        if repeated:
            if type(value) is not tuple or any(
                type(item) is not item_type
                or (item_type is float and not math.isfinite(item))
                for item in value
            ):
                raise ValueError(
                    f"configuration fingerprint field {field_name} "
                    "must use its native tuple type"
                )
            fingerprint[field_name] = list(value)
        else:
            if (
                type(value) is not item_type
                or (item_type is float and not math.isfinite(value))
            ):
                raise ValueError(
                    f"configuration fingerprint field {field_name} "
                    "must use its native scalar type"
                )
            fingerprint[field_name] = value
    return fingerprint


def calibrate(
    config: ExperimentConfig,
    output_dir: Path,
) -> Path:
    _validate_fixed_candidates(config)
    from moving_det.data.labelme import load_sequence

    sequence_path = config.data_root / config.calibration_sequence
    sequence = load_sequence(sequence_path, fps=config.fps)
    output_dir = Path(output_dir)
    work_root = _new_work_root(output_dir)
    stage = work_root / "artifact"
    stage.mkdir()
    collected: dict[str, dict[str, _MethodRunResult]] = {
        method_name: {}
        for method_name in _METHOD_NAMES
    }
    try:
        for scale in config.scale_factors:
            for method_name in ("frame_diff", "temporal_median"):
                result = _compute_group(
                    config=config,
                    sequence=sequence,
                    method_names=(method_name,),
                    scale=scale,
                    thresholds_by_method={
                        method_name: _Z_THRESHOLDS,
                    },
                    output_dirs={
                        method_name: stage
                        / _artifact_relative_path(method_name, scale),
                    },
                    work_dir=work_root
                    / f"cache-{method_name}-{_scale_key(scale)}",
                )
                collected[method_name][_scale_key(scale)] = result[method_name]
            result = _compute_group(
                config=config,
                sequence=sequence,
                method_names=("mog2",),
                scale=scale,
                thresholds_by_method={
                    "mog2": _MOG2_VAR_THRESHOLDS,
                },
                output_dirs={
                    "mog2": stage / _artifact_relative_path("mog2", scale),
                },
                work_dir=work_root / f"cache-mog2-{_scale_key(scale)}",
            )
            collected["mog2"][_scale_key(scale)] = result["mog2"]
            result = _compute_group(
                config=config,
                sequence=sequence,
                method_names=("multiscale", "multiscale_tubelet"),
                scale=scale,
                thresholds_by_method={
                    "multiscale": _Z_THRESHOLDS,
                    "multiscale_tubelet": _Z_THRESHOLDS,
                },
                output_dirs={
                    method_name: stage
                    / _artifact_relative_path(method_name, scale)
                    for method_name in ("multiscale", "multiscale_tubelet")
                },
                work_dir=work_root / f"cache-multiscale-{_scale_key(scale)}",
            )
            for method_name in ("multiscale", "multiscale_tubelet"):
                collected[method_name][_scale_key(scale)] = result[method_name]

        calibration_path = stage / "calibration.json"
        _write_json(
            calibration_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "sequence_id": sequence.sequence_id,
                "input_path": sequence_path.resolve(),
                "config_fingerprint": _calibration_config_fingerprint(config),
                "methods": {
                    method_name: {
                        scale_key: _calibration_entry(result)
                        for scale_key, result in by_scale.items()
                    }
                    for method_name, by_scale in collected.items()
                },
            },
        )
        for cache_dir in work_root.glob("cache-*"):
            shutil.rmtree(cache_dir)
        os.replace(stage, output_dir)
        return output_dir / "calibration.json"
    except BaseException:
        shutil.rmtree(work_root, ignore_errors=True)
        raise
    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"{context} fields are invalid: {', '.join(details)}")


def _frozen_number(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    allow_positive_infinity_string: bool = False,
) -> float:
    if allow_positive_infinity_string and value == "Infinity":
        return math.inf
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"calibration {field_name} must use its expected numeric form"
        )
    converted = float(value)
    if converted < minimum or (
        maximum is not None and converted > maximum
    ):
        raise ValueError(f"calibration {field_name} is outside its valid range")
    return converted


def _validate_frozen_candidate(
    candidate: object,
    expected_parameter: str,
    allowed_values: tuple[float, ...],
) -> tuple[float, float, float]:
    if not isinstance(candidate, dict):
        raise ValueError("calibration candidate must be an object")
    _require_exact_keys(
        candidate,
        {
            "parameter_name",
            "parameter_value",
            "recall_025",
            "fp_per_100_gt",
        },
        "calibration candidate",
    )
    if candidate["parameter_name"] != expected_parameter:
        raise ValueError("calibration candidate parameter_name is invalid")
    parameter_value = _frozen_number(
        candidate["parameter_value"],
        "parameter_value",
        minimum=0.0,
    )
    if parameter_value not in allowed_values:
        raise ValueError("calibration candidate value is not configured")
    recall = _frozen_number(
        candidate["recall_025"],
        "recall_025",
        minimum=0.0,
        maximum=1.0,
    )
    false_positive_rate = _frozen_number(
        candidate["fp_per_100_gt"],
        "fp_per_100_gt",
        minimum=0.0,
        allow_positive_infinity_string=True,
    )
    return parameter_value, recall, false_positive_rate


def _same_json_record(
    left: Mapping[str, object],
    right: object,
) -> bool:
    if not isinstance(right, dict) or set(left) != set(right):
        return False
    return all(
        type(left[key]) is type(right[key])
        and left[key] == right[key]
        for key in left
    )


def _validate_frozen_config_fingerprint(
    value: object,
    config: ExperimentConfig,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("calibration config_fingerprint must be an object")
    expected = _calibration_config_fingerprint(config)
    _require_exact_keys(
        value,
        set(expected),
        "calibration config_fingerprint",
    )
    for field_name, expected_value in expected.items():
        actual_value = value[field_name]
        if type(actual_value) is not type(expected_value):
            raise ValueError(
                "calibration config_fingerprint native type does not match "
                f"configuration for {field_name}"
            )
        if isinstance(expected_value, list):
            if len(actual_value) != len(expected_value) or any(
                type(actual_item) is not type(expected_item)
                or actual_item != expected_item
                for actual_item, expected_item in zip(
                    actual_value,
                    expected_value,
                    strict=True,
                )
            ):
                raise ValueError(
                    "calibration config_fingerprint does not match "
                    f"configuration for {field_name}"
                )
        elif actual_value != expected_value:
            raise ValueError(
                "calibration config_fingerprint does not match "
                f"configuration for {field_name}"
            )


def _frozen_selections(
    config: ExperimentConfig,
    calibration_path: Path,
) -> dict[str, dict[str, float]]:
    _validate_fixed_candidates(config)
    payload = _load_strict_json(calibration_path)
    if not isinstance(payload, dict):
        raise ValueError("calibration must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "sequence_id",
            "input_path",
            "config_fingerprint",
            "methods",
        },
        "calibration top level",
    )
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _SCHEMA_VERSION
    ):
        raise ValueError("unsupported calibration schema_version")
    if payload["sequence_id"] != config.calibration_sequence:
        raise ValueError("calibration sequence does not match configuration")
    expected_input_path = str(
        (config.data_root / config.calibration_sequence).resolve()
    )
    if payload["input_path"] != expected_input_path:
        raise ValueError("calibration input_path does not match configuration")
    _validate_frozen_config_fingerprint(
        payload["config_fingerprint"],
        config,
    )
    methods = payload.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(_METHOD_NAMES):
        raise ValueError("calibration must contain every configured method")

    selections: dict[str, dict[str, float]] = {}
    for method_name in _METHOD_NAMES:
        by_scale = methods[method_name]
        expected_scale_keys = {
            _scale_key(scale)
            for scale in config.scale_factors
        }
        if not isinstance(by_scale, dict) or set(by_scale) != expected_scale_keys:
            raise ValueError(
                f"calibration for {method_name} must contain every scale"
            )
        selections[method_name] = {}
        allowed = (
            _MOG2_VAR_THRESHOLDS
            if method_name == "mog2"
            else _Z_THRESHOLDS
        )
        expected_parameter = (
            "varThreshold" if method_name == "mog2" else "z_threshold"
        )
        for scale_key, entry in by_scale.items():
            if not isinstance(entry, dict):
                raise ValueError("calibration method/scale entry must be an object")
            _require_exact_keys(
                entry,
                {
                    "parameter_name",
                    "candidates",
                    "selected",
                    "constraint_satisfied",
                },
                "calibration method/scale entry",
            )
            if entry["parameter_name"] != expected_parameter:
                raise ValueError("calibration parameter_name is invalid")
            candidates = entry["candidates"]
            if not isinstance(candidates, list):
                raise ValueError("calibration candidates must be a list")
            validated_candidates = tuple(
                _validate_frozen_candidate(
                    candidate,
                    expected_parameter,
                    allowed,
                )
                for candidate in candidates
            )
            if tuple(
                candidate[0]
                for candidate in validated_candidates
            ) != allowed:
                raise ValueError(
                    "calibration candidates must contain the fixed values "
                    "in deterministic order"
                )
            selected = entry["selected"]
            if not isinstance(selected, dict):
                raise ValueError("calibration selected value is missing")
            selected_record = _validate_frozen_candidate(
                selected,
                expected_parameter,
                allowed,
            )
            if not any(
                _same_json_record(selected, candidate)
                for candidate in candidates
            ):
                raise ValueError(
                    "calibration selected candidate is not present exactly "
                    "in candidates"
                )
            constraint_satisfied = entry["constraint_satisfied"]
            if not isinstance(constraint_satisfied, bool):
                raise ValueError(
                    "calibration constraint_satisfied must be a boolean"
                )
            selected_fp = selected_record[2]
            expected_constraint = (
                math.isfinite(selected_fp)
                and selected_fp <= config.max_false_proposals_per_100_gt
            )
            if constraint_satisfied != expected_constraint:
                raise ValueError(
                    "calibration constraint_satisfied conflicts with selected "
                    "candidate"
                )
            selections[method_name][scale_key] = selected_record[0]
    return selections


def _run_payload(result: _MethodRunResult) -> dict[str, object]:
    selected_value = float(result.choice.candidate.parameter_value)
    selected = next(
        candidate
        for candidate in result.candidates
        if candidate.threshold == selected_value
    )
    return {
        "threshold": selected.threshold,
        "aggregate": selected.report.aggregate,
        "boundary": selected.report.boundary,
        "strata": selected.report.strata,
        "moving_threshold_sensitivity": selected.sensitivity,
    }


def _gate(
    measured_value: float,
    required: str,
    passed: bool,
) -> dict[str, object]:
    return {
        "measured_value": measured_value,
        "required": required,
        "passed": bool(passed),
    }


def _evaluation_gates(
    runs: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> tuple[dict[str, dict[str, object]], bool]:
    native = "1.0"
    reduced = "0.7"

    def aggregate(method_name: str, scale_key: str) -> Mapping[str, object]:
        value = runs[method_name][scale_key]["aggregate"]
        if not isinstance(value, Mapping):
            raise ValueError("evaluation run aggregate is invalid")
        return value

    tube_native = aggregate("multiscale_tubelet", native)
    tube_reduced = aggregate("multiscale_tubelet", reduced)
    tube_recall = float(tube_native["recall_025"])
    baseline_recall = max(
        float(aggregate(method_name, native)["recall_025"])
        for method_name in ("frame_diff", "mog2", "temporal_median")
    )
    improvement = tube_recall - baseline_recall
    center_recall = float(tube_native["center_in_gt_recall"])
    reduced_recall = float(tube_reduced["recall_025"])
    scale_drop = tube_recall - reduced_recall
    track_coverage = float(tube_native["mean_moving_frame_coverage"])
    fragments = float(tube_native["mean_extra_tubelet_fragments"])
    gates = {
        "tubelet_recall_improvement": _gate(
            improvement,
            ">= 0.05",
            improvement >= 0.05,
        ),
        "native_center_in_gt_recall": _gate(
            center_recall,
            ">= 0.95",
            center_recall >= 0.95,
        ),
        "native_recall_025": _gate(
            tube_recall,
            ">= 0.90",
            tube_recall >= 0.90,
        ),
        "scale_recall_drop": _gate(
            scale_drop,
            "<= 0.10",
            scale_drop <= 0.10,
        ),
        "moving_frame_track_coverage": _gate(
            track_coverage,
            ">= 0.90",
            track_coverage >= 0.90,
        ),
        "mean_extra_fragments": _gate(
            fragments,
            "<= 0.20",
            fragments <= 0.20,
        ),
    }
    return gates, all(bool(gate["passed"]) for gate in gates.values())


def evaluate(
    config: ExperimentConfig,
    calibration_path: Path,
    output_dir: Path,
) -> Path:
    _validate_fixed_candidates(config)
    from moving_det.data.labelme import load_sequence

    calibration_path = Path(calibration_path)
    selections = _frozen_selections(config, calibration_path)
    sequence_path = config.data_root / config.evaluation_sequence
    sequence = load_sequence(sequence_path, fps=config.fps)
    output_dir = Path(output_dir)
    work_root = _new_work_root(output_dir)
    stage = work_root / "artifact"
    stage.mkdir()
    collected: dict[str, dict[str, _MethodRunResult]] = {
        method_name: {}
        for method_name in _METHOD_NAMES
    }
    try:
        for scale in config.scale_factors:
            scale_key = _scale_key(scale)
            for method_name in ("frame_diff", "temporal_median"):
                result = _compute_group(
                    config=config,
                    sequence=sequence,
                    method_names=(method_name,),
                    scale=scale,
                    thresholds_by_method={
                        method_name: (selections[method_name][scale_key],),
                    },
                    output_dirs={
                        method_name: stage
                        / _artifact_relative_path(method_name, scale),
                    },
                    work_dir=work_root
                    / f"cache-{method_name}-{scale_key}",
                )
                collected[method_name][scale_key] = result[method_name]
            result = _compute_group(
                config=config,
                sequence=sequence,
                method_names=("mog2",),
                scale=scale,
                thresholds_by_method={
                    "mog2": (selections["mog2"][scale_key],),
                },
                output_dirs={
                    "mog2": stage / _artifact_relative_path("mog2", scale),
                },
                work_dir=work_root / f"cache-mog2-{scale_key}",
            )
            collected["mog2"][scale_key] = result["mog2"]
            result = _compute_group(
                config=config,
                sequence=sequence,
                method_names=("multiscale", "multiscale_tubelet"),
                scale=scale,
                thresholds_by_method={
                    method_name: (selections[method_name][scale_key],)
                    for method_name in ("multiscale", "multiscale_tubelet")
                },
                output_dirs={
                    method_name: stage
                    / _artifact_relative_path(method_name, scale)
                    for method_name in ("multiscale", "multiscale_tubelet")
                },
                work_dir=work_root / f"cache-multiscale-{scale_key}",
            )
            for method_name in ("multiscale", "multiscale_tubelet"):
                collected[method_name][scale_key] = result[method_name]

        runs = {
            method_name: {
                scale_key: _run_payload(result)
                for scale_key, result in by_scale.items()
            }
            for method_name, by_scale in collected.items()
        }
        gates, gate_passed = _evaluation_gates(runs)
        metrics_path = stage / "metrics.json"
        _write_json(
            metrics_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "sequence_id": sequence.sequence_id,
                "threshold_source": calibration_path.resolve(),
                "runs": runs,
                "gates": gates,
                "gate_passed": gate_passed,
            },
        )
        _write_json(
            stage / "run.json",
            _metadata(
                config,
                sequence,
                method="frozen_evaluation",
                scale=None,
                threshold=None,
                threshold_source=calibration_path,
            ),
        )
        for cache_dir in work_root.glob("cache-*"):
            shutil.rmtree(cache_dir)
        os.replace(stage, output_dir)
        return output_dir / "metrics.json"
    except BaseException:
        shutil.rmtree(work_root, ignore_errors=True)
        raise
    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


def write_report(metrics_path: Path, output_path: Path) -> Path:
    payload = _load_strict_json(metrics_path)
    if not isinstance(payload, dict):
        raise ValueError("metrics must be a JSON object")
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("metrics does not contain gate details")
    lines = [
        "# Motion evidence evaluation report",
        "",
        f"- Sequence: `{payload.get('sequence_id', 'unknown')}`",
        f"- gate_passed: `{payload.get('gate_passed')}`",
        "",
        "## Design gates",
        "",
        "| Gate | Measured value | Requirement | Passed |",
        "|---|---:|---:|:---:|",
    ]
    for name, gate in gates.items():
        if not isinstance(gate, dict):
            raise ValueError("gate entry must be an object")
        lines.append(
            "| "
            f"{_GATE_LABELS.get(name, name)} | "
            f"{gate.get('measured_value')} | "
            f"{gate.get('required')} | {gate.get('passed')} |"
        )
    output_path = Path(output_path)
    _write_text(output_path, "\n".join(lines) + "\n")
    return output_path


__all__ = [
    "RunArtifacts",
    "calibrate",
    "evaluate",
    "run_method",
    "write_report",
]
