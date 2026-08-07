from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor


def _resolve_layer_input(
    source: int | Sequence[int],
    current: Any,
    saved: list[Any],
) -> Any:
    if isinstance(source, int):
        if source == -1:
            return current
        return saved[source]
    return [current if index == -1 else saved[index] for index in source]


def _require_compatible_override(
    layer_index: int,
    natural: Any,
    replacement: Any,
) -> None:
    if not isinstance(natural, Tensor) or not isinstance(replacement, Tensor):
        raise ValueError(
            f"layer {layer_index} override must replace a tensor with a tensor"
        )
    if replacement.shape != natural.shape:
        raise ValueError(
            f"layer {layer_index} override shape {tuple(replacement.shape)} "
            f"does not match {tuple(natural.shape)}"
        )
    if replacement.dtype != natural.dtype:
        raise ValueError(
            f"layer {layer_index} override dtype {replacement.dtype} "
            f"does not match {natural.dtype}"
        )
    if replacement.device != natural.device:
        raise ValueError(
            f"layer {layer_index} override device {replacement.device} "
            f"does not match {natural.device}"
        )


def _run_yolo_graph(
    model: Any,
    image: Tensor,
    *,
    overrides: Mapping[int, Tensor] | None = None,
    capture: frozenset[int] = frozenset(),
    stop_after: int | None = None,
) -> tuple[Any, dict[int, Tensor]]:
    replacements = {} if overrides is None else dict(overrides)
    layer_indices = {int(layer.i) for layer in model.model}
    unknown = set(replacements).difference(layer_indices)
    if unknown:
        raise ValueError(f"unknown YOLO layer overrides: {sorted(unknown)}")

    saved: list[Any] = []
    captured: dict[int, Tensor] = {}
    value: Any = image
    for layer in model.model:
        value = _resolve_layer_input(layer.f, value, saved)
        natural = layer(value)
        if layer.i in replacements:
            replacement = replacements[layer.i]
            _require_compatible_override(layer.i, natural, replacement)
            value = replacement
        else:
            value = natural

        if layer.i in capture:
            if not isinstance(value, Tensor):
                raise ValueError(f"YOLO layer {layer.i} does not output a tensor")
            captured[layer.i] = value
        saved.append(value if layer.i in model.save else None)
        if layer.i == stop_after:
            break
    return value, captured


def execute_yolo_graph(
    model: Any,
    image: Tensor,
    overrides: Mapping[int, Tensor] | None = None,
) -> Any:
    """Execute an Ultralytics graph while optionally replacing layer outputs."""
    output, _ = _run_yolo_graph(model, image, overrides=overrides)
    return output


def extract_backbone_features(
    model: Any,
    image: Tensor,
    indices: Sequence[int] = (2, 4),
) -> dict[int, Tensor]:
    """Return selected tensor outputs from the detector backbone."""
    requested = tuple(indices)
    if len(set(requested)) != len(requested):
        raise ValueError("backbone feature indices must be unique")
    invalid = [index for index in requested if index < 0 or index > 10]
    if invalid:
        raise ValueError(f"backbone feature indices must be in [0, 10]: {invalid}")
    if not requested:
        return {}

    _, captured = _run_yolo_graph(
        model,
        image,
        capture=frozenset(requested),
        stop_after=max(requested),
    )
    return {index: captured[index] for index in requested}
