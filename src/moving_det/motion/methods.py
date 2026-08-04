from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from moving_det.config import ExperimentConfig
from moving_det.models import FrameSample, MotionEvidence, SequenceData
from moving_det.motion.alignment import (
    AlignmentResult,
    estimate_euclidean_ecc,
    warp_to_reference,
)
from moving_det.motion.evidence import compute_motion_evidence, robust_z


_METHOD_NAMES = (
    "frame_diff",
    "mog2",
    "temporal_median",
    "multiscale",
    "multiscale_tubelet",
)
_SCALE_FACTORS = (1.0, 0.7)
_MOG2_HISTORY = 60
_MOG2_VAR_THRESHOLDS = (9.0, 16.0, 25.0)
_MOG2_DEFAULT_VAR_THRESHOLD = 16.0
_PRELIMINARY_Z_THRESHOLD = 5.0
_FRAME_CACHE_SIZE = 31
_TEMPORAL_STACK_TARGET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class AlignmentPassDiagnostic:
    correlation: float
    used_fallback: bool
    reason: str | None


@dataclass(frozen=True)
class AlignmentDiagnostic:
    reference_index: int
    support_index: int
    mode: str
    first_pass: AlignmentPassDiagnostic | None
    second_pass: AlignmentPassDiagnostic | None


@runtime_checkable
class MotionMethod(Protocol):
    def run(
        self,
        sequence: SequenceData,
        scale: float,
    ) -> Mapping[int, MotionEvidence]: ...


def _immutable_array(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(
        contiguous.tobytes(),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)


def _pass_diagnostic(result: AlignmentResult) -> AlignmentPassDiagnostic:
    return AlignmentPassDiagnostic(
        correlation=result.correlation,
        used_fallback=result.used_fallback,
        reason=result.reason,
    )


def _validate_scale(scale: float, cfg: ExperimentConfig) -> float:
    if (
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not np.isfinite(scale)
    ):
        raise ValueError("scale must be one of 1.0 or 0.7")
    scale = float(scale)
    configured_scales = tuple(float(value) for value in cfg.scale_factors)
    if scale not in _SCALE_FACTORS or scale not in configured_scales:
        raise ValueError("scale must be one of the configured 1.0 or 0.7 values")
    return scale


def _validate_mog2_factory_config(cfg: ExperimentConfig) -> None:
    if cfg.mog2_history != _MOG2_HISTORY:
        raise ValueError("mog2_history must be 60")
    if (
        not isinstance(cfg.mog2_var_threshold_candidates, tuple)
        or cfg.mog2_var_threshold_candidates != _MOG2_VAR_THRESHOLDS
    ):
        raise ValueError(
            "mog2_var_threshold_candidates must be (9.0, 16.0, 25.0)"
        )


def _validate_poc_config(cfg: ExperimentConfig) -> None:
    checks = (
        ("window_radius", cfg.window_radius == 15),
        (
            "offsets",
            isinstance(cfg.offsets, tuple)
            and cfg.offsets == (1, 3, 7, 15),
        ),
        (
            "scale_factors",
            isinstance(cfg.scale_factors, tuple)
            and cfg.scale_factors == _SCALE_FACTORS,
        ),
        ("mad_floor", cfg.mad_floor == 2.0),
        ("mad_clip", cfg.mad_clip == 6.0),
        ("mog2_history", cfg.mog2_history == _MOG2_HISTORY),
        (
            "mog2_var_threshold_candidates",
            isinstance(cfg.mog2_var_threshold_candidates, tuple)
            and cfg.mog2_var_threshold_candidates
            == _MOG2_VAR_THRESHOLDS,
        ),
        ("ecc_min_correlation", cfg.ecc_min_correlation == 0.8),
        ("ecc_max_translation", cfg.ecc_max_translation == 20.0),
        (
            "ecc_max_rotation_degrees",
            cfg.ecc_max_rotation_degrees == 2.0,
        ),
    )
    for field_name, is_valid in checks:
        if not is_valid:
            raise ValueError(f"{field_name} does not match the fixed POC config")


def _ordered_samples(sequence: SequenceData) -> tuple[FrameSample, ...]:
    if not isinstance(sequence, SequenceData):
        raise TypeError("sequence must be a SequenceData")
    samples = sequence.frames
    if any(not isinstance(sample, FrameSample) for sample in samples):
        raise TypeError("sequence frames must be FrameSample instances")
    if any(
        isinstance(sample.frame_index, bool)
        or not isinstance(sample.frame_index, Integral)
        for sample in samples
    ):
        raise TypeError("frame indices must be integers")
    ordered = tuple(sorted(samples, key=lambda sample: int(sample.frame_index)))
    frame_indices = tuple(int(sample.frame_index) for sample in ordered)
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("frame indices must be unique")
    if any(
        current != previous + 1
        for previous, current in zip(frame_indices, frame_indices[1:])
    ):
        raise ValueError("frame indices must be consecutive integers")
    return ordered


def _center_samples(
    samples: tuple[FrameSample, ...],
    center_indices: Sequence[int] | None,
) -> tuple[FrameSample, ...]:
    if center_indices is None:
        return samples
    try:
        requested = tuple(center_indices)
    except TypeError as exc:
        raise TypeError("center indices must be an integer sequence") from exc
    if not requested:
        raise ValueError("at least one center index is required")
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in requested
    ):
        raise TypeError("center indices must be integers")
    normalized = tuple(int(index) for index in requested)
    if len(set(normalized)) != len(normalized):
        raise ValueError("center indices must be unique")
    if normalized != tuple(sorted(normalized)):
        raise ValueError("center indices must be ordered")
    if any(
        current != previous + 1
        for previous, current in zip(normalized, normalized[1:])
    ):
        raise ValueError("center indices must be consecutive")
    by_index = {
        int(sample.frame_index): sample
        for sample in samples
    }
    if any(index not in by_index for index in normalized):
        raise ValueError("center indices must exist in the processing sequence")
    return tuple(by_index[index] for index in normalized)


class _ImageReader:
    def __init__(
        self,
        sequence: SequenceData,
        scale: float,
    ) -> None:
        self._sequence = sequence
        self._scale = scale
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def read(self, sample: FrameSample) -> np.ndarray:
        frame_index = int(sample.frame_index)
        cached = self._cache.pop(frame_index, None)
        if cached is not None:
            self._cache[frame_index] = cached
            return cached

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(
                f"unable to read frame {frame_index} from {sample.image_path}"
            )
        expected_shape = (self._sequence.height, self._sequence.width)
        if image.shape != expected_shape:
            raise ValueError(
                f"frame {frame_index} shape {image.shape} does not match "
                f"sequence shape {expected_shape}"
            )
        if self._scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=self._scale,
                fy=self._scale,
                interpolation=cv2.INTER_AREA,
            )

        self._cache[frame_index] = image
        if len(self._cache) > _FRAME_CACHE_SIZE:
            self._cache.popitem(last=False)
        return image


def _ignored_mask(
    sample: FrameSample,
    shape: tuple[int, int],
    scale: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in sample.ignore_polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 3
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                f"frame {sample.frame_index} has an invalid ignored polygon"
            )
        scaled_points = np.rint(points * scale).astype(np.int32)
        cv2.fillPoly(mask, [scaled_points], color=1)
    return mask.astype(bool)


def _two_pass_align(
    reference_sample: FrameSample,
    support_sample: FrameSample,
    reference: np.ndarray,
    support: np.ndarray,
    cfg: ExperimentConfig,
    scale: float,
) -> tuple[np.ndarray, AlignmentDiagnostic]:
    base_exclusion = _ignored_mask(
        support_sample,
        reference.shape,
        scale,
    )
    first_result = estimate_euclidean_ecc(
        reference,
        support,
        cfg,
        exclude_mask=base_exclusion,
    )
    first_aligned = warp_to_reference(support, first_result)
    preliminary_delta = cv2.absdiff(reference, first_aligned)
    preliminary_z = robust_z(
        preliminary_delta,
        floor=cfg.mad_floor,
        clip=cfg.mad_clip,
    )
    moving_exclusion = cv2.warpAffine(
        np.greater_equal(
            preliminary_z,
            _PRELIMINARY_Z_THRESHOLD,
        ).astype(np.uint8),
        np.asarray(first_result.matrix, dtype=np.float32),
        (support.shape[1], support.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    second_exclusion = np.logical_or(
        base_exclusion,
        moving_exclusion,
    )
    second_result = estimate_euclidean_ecc(
        reference,
        support,
        cfg,
        exclude_mask=second_exclusion,
    )
    aligned = warp_to_reference(support, second_result)
    diagnostic = AlignmentDiagnostic(
        reference_index=int(reference_sample.frame_index),
        support_index=int(support_sample.frame_index),
        mode="ecc_two_pass",
        first_pass=_pass_diagnostic(first_result),
        second_pass=_pass_diagnostic(second_result),
    )
    return aligned, diagnostic


def _single_channel_evidence(
    frame_index: int,
    channel_name: str,
    channel_z: np.ndarray,
    support_indices: tuple[int, ...],
    cfg: ExperimentConfig,
) -> MotionEvidence:
    immutable_channel = _immutable_array(channel_z)
    fused_z = _immutable_array(channel_z)
    fused_score = _immutable_array(channel_z / cfg.mad_clip)
    return MotionEvidence(
        frame_index=frame_index,
        channel_z=MappingProxyType({channel_name: immutable_channel}),
        fused_z=fused_z,
        fused_score=fused_score,
        support_indices=support_indices,
    )


def _temporal_background_difference(
    center: np.ndarray,
    aligned: Mapping[int, np.ndarray],
    support_indices: tuple[int, ...],
) -> np.ndarray:
    height, width = center.shape
    bytes_per_row = len(support_indices) * width * center.dtype.itemsize
    rows_per_chunk = max(
        1,
        _TEMPORAL_STACK_TARGET_BYTES // bytes_per_row,
    )
    difference = np.empty(center.shape, dtype=np.float32)
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        stack = np.stack(
            [
                aligned[index][start:stop]
                for index in support_indices
            ],
            axis=0,
        )
        background = np.median(stack, axis=0, overwrite_input=True)
        difference[start:stop] = np.abs(
            center[start:stop].astype(np.float32) - background,
        )
    return difference


class _AlignedMethod:
    def __init__(self, name: str, cfg: ExperimentConfig) -> None:
        self._name = name
        self._cfg = cfg
        self._diagnostics: Mapping[
            int,
            tuple[AlignmentDiagnostic, ...],
        ] = MappingProxyType({})

    @property
    def diagnostics(
        self,
    ) -> Mapping[int, tuple[AlignmentDiagnostic, ...]]:
        return self._diagnostics

    def _aligned_support(
        self,
        center_sample: FrameSample,
        support_samples: tuple[FrameSample, ...],
        reader: _ImageReader,
        scale: float,
    ) -> tuple[
        dict[int, np.ndarray],
        tuple[AlignmentDiagnostic, ...],
    ]:
        center_index = int(center_sample.frame_index)
        center = reader.read(center_sample)
        aligned = {center_index: center}
        diagnostics = []
        for support_sample in support_samples:
            support_index = int(support_sample.frame_index)
            if support_index == center_index:
                continue
            support, diagnostic = _two_pass_align(
                center_sample,
                support_sample,
                center,
                reader.read(support_sample),
                self._cfg,
                scale,
            )
            aligned[support_index] = support
            diagnostics.append(diagnostic)
        return aligned, tuple(diagnostics)

    def _iter_frame_diff(
        self,
        samples: tuple[FrameSample, ...],
        center_samples: tuple[FrameSample, ...],
        reader: _ImageReader,
        scale: float,
    ) -> Iterator[MotionEvidence]:
        by_index = {int(sample.frame_index): sample for sample in samples}
        all_diagnostics: dict[
            int,
            tuple[AlignmentDiagnostic, ...],
        ] = {}
        self._diagnostics = MappingProxyType(all_diagnostics)
        for center_sample in center_samples:
            center_index = int(center_sample.frame_index)
            previous_sample = by_index.get(center_index - 1)
            center = reader.read(center_sample)
            if previous_sample is None:
                z = np.zeros(center.shape, dtype=np.float32)
                support_indices = (center_index,)
                diagnostics = ()
            else:
                aligned, diagnostic = _two_pass_align(
                    center_sample,
                    previous_sample,
                    center,
                    reader.read(previous_sample),
                    self._cfg,
                    scale,
                )
                delta = cv2.absdiff(center, aligned)
                z = robust_z(
                    delta,
                    floor=self._cfg.mad_floor,
                    clip=self._cfg.mad_clip,
                )
                support_indices = (center_index - 1, center_index)
                diagnostics = (diagnostic,)
            evidence = _single_channel_evidence(
                center_index,
                "d1",
                z,
                support_indices,
                self._cfg,
            )
            all_diagnostics[center_index] = diagnostics
            yield evidence

    def _iter_windowed(
        self,
        samples: tuple[FrameSample, ...],
        center_samples: tuple[FrameSample, ...],
        reader: _ImageReader,
        scale: float,
    ) -> Iterator[MotionEvidence]:
        all_diagnostics: dict[
            int,
            tuple[AlignmentDiagnostic, ...],
        ] = {}
        self._diagnostics = MappingProxyType(all_diagnostics)
        for center_sample in center_samples:
            center_index = int(center_sample.frame_index)
            support_samples = tuple(
                sample
                for sample in samples
                if abs(int(sample.frame_index) - center_index)
                <= self._cfg.window_radius
            )
            aligned, diagnostics = self._aligned_support(
                center_sample,
                support_samples,
                reader,
                scale,
            )
            if self._name == "temporal_median":
                support_indices = tuple(sorted(aligned))
                delta = _temporal_background_difference(
                    aligned[center_index],
                    aligned,
                    support_indices,
                )
                z = robust_z(
                    delta,
                    floor=self._cfg.mad_floor,
                    clip=self._cfg.mad_clip,
                )
                evidence = _single_channel_evidence(
                    center_index,
                    "dbg",
                    z,
                    support_indices,
                    self._cfg,
                )
            else:
                evidence = compute_motion_evidence(
                    center_index,
                    aligned,
                    self._cfg,
                )
            all_diagnostics[center_index] = diagnostics
            yield evidence

    def iter_run(
        self,
        sequence: SequenceData,
        scale: float,
        *,
        center_indices: Sequence[int] | None = None,
    ) -> Iterator[MotionEvidence]:
        _validate_poc_config(self._cfg)
        scale = _validate_scale(scale, self._cfg)
        samples = _ordered_samples(sequence)
        centers = _center_samples(samples, center_indices)
        reader = _ImageReader(sequence, scale)
        if self._name == "frame_diff":
            yield from self._iter_frame_diff(
                samples,
                centers,
                reader,
                scale,
            )
        else:
            yield from self._iter_windowed(
                samples,
                centers,
                reader,
                scale,
            )

    def run(
        self,
        sequence: SequenceData,
        scale: float,
    ) -> Mapping[int, MotionEvidence]:
        results = {
            evidence.frame_index: evidence
            for evidence in self.iter_run(sequence, scale)
        }
        return MappingProxyType(results)


class _MOG2Method:
    def __init__(
        self,
        cfg: ExperimentConfig,
        var_threshold: float,
    ) -> None:
        self._cfg = cfg
        self._var_threshold = var_threshold
        self._diagnostics: Mapping[
            int,
            tuple[AlignmentDiagnostic, ...],
        ] = MappingProxyType({})

    @property
    def diagnostics(
        self,
    ) -> Mapping[int, tuple[AlignmentDiagnostic, ...]]:
        return self._diagnostics

    @property
    def var_threshold(self) -> float:
        return self._var_threshold

    def run(
        self,
        sequence: SequenceData,
        scale: float,
    ) -> Mapping[int, MotionEvidence]:
        results = {
            evidence.frame_index: evidence
            for evidence in self.iter_run(sequence, scale)
        }
        return MappingProxyType(results)

    def iter_run(
        self,
        sequence: SequenceData,
        scale: float,
        *,
        center_indices: Sequence[int] | None = None,
    ) -> Iterator[MotionEvidence]:
        _validate_poc_config(self._cfg)
        scale = _validate_scale(scale, self._cfg)
        samples = _ordered_samples(sequence)
        centers = _center_samples(samples, center_indices)
        selected_indices = {
            int(sample.frame_index)
            for sample in centers
        }
        final_selected_index = max(selected_indices)
        reader = _ImageReader(sequence, scale)
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=_MOG2_HISTORY,
            varThreshold=self._var_threshold,
            detectShadows=False,
        )
        for sample in samples[:_MOG2_HISTORY]:
            subtractor.apply(reader.read(sample))

        diagnostics: dict[
            int,
            tuple[AlignmentDiagnostic, ...],
        ] = {}
        self._diagnostics = MappingProxyType(diagnostics)
        for sample in samples:
            frame_index = int(sample.frame_index)
            if frame_index > final_selected_index:
                break
            foreground_mask = subtractor.apply(reader.read(sample))
            if frame_index not in selected_indices:
                continue
            z = (
                np.not_equal(foreground_mask, 0).astype(np.float32)
                * self._cfg.mad_clip
            )
            evidence = _single_channel_evidence(
                frame_index,
                "foreground",
                z,
                (frame_index,),
                self._cfg,
            )
            diagnostics[frame_index] = (
                AlignmentDiagnostic(
                    reference_index=frame_index,
                    support_index=frame_index,
                    mode="identity",
                    first_pass=None,
                    second_pass=None,
                ),
            )
            yield evidence


def create_method(
    name: str,
    cfg: ExperimentConfig,
    var_threshold: float | None = None,
) -> MotionMethod:
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError("cfg must be an ExperimentConfig")
    if not isinstance(name, str) or name not in _METHOD_NAMES:
        raise ValueError(
            f"unknown motion method {name!r}; expected one of {_METHOD_NAMES}"
        )
    if name != "mog2":
        if var_threshold is not None:
            raise ValueError("var_threshold is only valid for the mog2 method")
        aligned_name = (
            "multiscale"
            if name == "multiscale_tubelet"
            else name
        )
        return _AlignedMethod(aligned_name, cfg)

    _validate_mog2_factory_config(cfg)
    if var_threshold is None:
        var_threshold = _MOG2_DEFAULT_VAR_THRESHOLD
    if (
        isinstance(var_threshold, bool)
        or not isinstance(var_threshold, Real)
        or not np.isfinite(var_threshold)
        or float(var_threshold)
        not in _MOG2_VAR_THRESHOLDS
    ):
        raise ValueError(
            "var_threshold must be one of the configured MOG2 candidates"
        )
    return _MOG2Method(cfg, float(var_threshold))
