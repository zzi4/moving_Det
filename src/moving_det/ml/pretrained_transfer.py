from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickletools
import random
import shutil
import stat
import struct
import tempfile
from types import MappingProxyType
from typing import Any, BinaryIO, Iterator
import zipfile

import numpy as np
import torch
from torch import Tensor


APPROVED_UNIVERSAL_SHA256 = (
    "114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7"
)
APPROVED_UNIVERSAL_PATH = Path(
    "/home/stu1/Projects/moving_Det/models/best_vru_universal.pt"
)
_ARTIFACT_KIND = "universal_p2_initialization"
_SCHEMA_VERSION = 1
_EXPECTED_NC = 4
_EXPECTED_TRANSFER_COUNT = 427
_EXPECTED_TARGET_COUNT = 859
_FROZEN_CHILDREN = frozenset(
    {"p2-init.pt", "transfer_report.json", "run.json"}
)
_P2_MODEL_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "models"
    / "yolo11m-p2-obb.yaml"
)
_PAYLOAD_FIELDS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "seed",
        "nc",
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "model_state_sha256",
        "model_state",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "seed",
        "nc",
        "source_weights_sha256",
        "target_config_sha256",
        "source_count",
        "target_count",
        "loaded_count",
        "missing_in_source_count",
        "shape_mismatch_count",
        "unused_source_count",
        "transfer_names_sha256",
        "loaded_tensors_sha256",
        "model_state_sha256",
        "loaded",
        "missing_in_source",
        "shape_mismatch",
        "unused_source",
    }
)
_RUN_FIELDS = frozenset({"artifact_kind", "schema_version", "artifacts"})
_PICKLE_PROBE_LIMIT = 16 * 1024 * 1024
_LEGACY_HEADER_LIMIT = 64 * 1024
_CHECKPOINT_PROBE_FILE_LIMIT = 8 * 1024 * 1024 * 1024
_ZIP_MEMBER_LIMIT = 4096
_ZIP_CENTRAL_DIRECTORY_LIMIT = 4 * 1024 * 1024
_ZIP_PICKLE_COMPRESSED_LIMIT = 16 * 1024 * 1024
_ZIP_EOCD_SEARCH_LIMIT = 22 + 65535
_LEGACY_TORCH_MAGIC = 0x1950A86A20F9469CFC6C
_LEGACY_TORCH_PROTOCOL = 1001
_PICKLE_MARK = object()
_PICKLE_OPAQUE = object()
_PICKLE_SEQUENCE = object()
_ARCHIVE_NOT_TORCH = object()
_ARCHIVE_INDETERMINATE = object()
_ZIP_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_ZIP_EOCD_HEADER = struct.Struct("<4s4H2LH")
_ZIP64_EOCD_HEADER = struct.Struct("<4sQ2H2L4Q")
_ZIP64_EOCD_LOCATOR = struct.Struct("<4sLQL")
_SNAPSHOT_COPY_CHUNK = 1024 * 1024


class _ProbeDict(dict[object, object]):
    pass


class _CheckpointMissingError(ValueError):
    pass


class _CheckpointUnsafeError(ValueError):
    pass


@dataclass(frozen=True)
class _CheckpointSnapshot:
    source_path: Path
    path: Path
    stream: BinaryIO
    file_size: int
    sha256: str


class _ValidatedZipView:
    def __init__(self, stream: BinaryIO, eocd_offset: int) -> None:
        self._stream = stream
        self._eocd_offset = eocd_offset
        self._size = eocd_offset + _ZIP_EOCD_HEADER.size
        self._position = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError("invalid ZIP seek mode")
        if position < 0:
            raise ValueError("negative ZIP seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        available = self._size - self._position
        read_size = available if size is None or size < 0 else min(size, available)
        start = self._position
        self._stream.seek(start)
        content = self._stream.read(read_size)
        self._position += len(content)
        comment_length_offset = self._eocd_offset + 20
        overlap_start = max(start, comment_length_offset)
        overlap_end = min(
            start + len(content),
            comment_length_offset + 2,
        )
        if overlap_start < overlap_end:
            normalized = bytearray(content)
            normalized[
                overlap_start - start : overlap_end - start
            ] = b"\x00" * (overlap_end - overlap_start)
            return bytes(normalized)
        return content


def _validated_state(
    state: Mapping[str, Tensor],
    *,
    label: str,
) -> dict[str, Tensor]:
    validated: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} state names must be non-empty strings")
        if not isinstance(value, Tensor):
            raise ValueError(f"{label} state values must be tensors")
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError(f"{label} state tensors must be dense strided tensors")
        if value.device.type == "meta":
            raise ValueError(f"{label} state tensors cannot use the meta device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} state tensors must be finite")
        validated[name] = value
    return validated


def compatible_state(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Clone source tensors whose names and shapes exactly match a target."""
    source_state = _validated_state(source, label="source")
    target_state = _validated_state(target, label="target")
    return {
        name: source_state[name].detach().clone()
        for name in sorted(target_state)
        if name in source_state
        and source_state[name].shape == target_state[name].shape
    }


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match schema")


def _require_plain_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _require_nonnegative_plain_int(value: object, *, label: str) -> int:
    converted = _require_plain_int(value, label=label)
    if converted < 0:
        raise ValueError(f"{label} must be non-negative")
    return converted


def _require_identity_fields(
    value: Mapping[str, object],
    *,
    label: str,
    include_seed_and_nc: bool,
) -> None:
    artifact_kind = value["artifact_kind"]
    if not isinstance(artifact_kind, str) or artifact_kind != _ARTIFACT_KIND:
        raise ValueError(f"{label} artifact_kind is invalid")
    schema_version = _require_plain_int(
        value["schema_version"],
        label=f"{label} schema_version",
    )
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version is unsupported")
    if include_seed_and_nc:
        seed = _require_plain_int(value["seed"], label=f"{label} seed")
        nc = _require_plain_int(value["nc"], label=f"{label} nc")
        if not 0 <= seed < 2**63:
            raise ValueError(f"{label} seed is invalid")
        if nc != _EXPECTED_NC:
            raise ValueError(f"{label} nc must equal 4")


def _require_shape(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} shape must be a list")
    if any(
        type(dimension) is not int
        or dimension < 0
        or dimension >= 2**63
        for dimension in value
    ):
        raise ValueError(
            f"{label} shape dimensions must be plain non-negative integers"
        )
    signed_storage_max = 2**63 - 1
    trailing_stride = 1
    for dimension in reversed(value[1:]):
        stride_factor = max(dimension, 1)
        if trailing_stride > signed_storage_max // stride_factor:
            raise ValueError(f"{label} shape stride calculation overflows")
        trailing_stride *= stride_factor
    if value and 0 not in value:
        if trailing_stride > signed_storage_max // value[0]:
            raise ValueError(f"{label} shape storage size overflows")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact contains an invalid JSON value") from exc


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains non-finite constant: {value}")


def _load_canonical_json_bytes(content: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if _canonical_json_bytes(value) != content:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    validated = _validated_state(state, label="hashed")
    digest = hashlib.sha256()
    for name in sorted(validated):
        tensor = validated[name]
        _update_framed(digest, name.encode("utf-8"))
        _update_framed(digest, str(tensor.dtype).encode("ascii"))
        digest.update(len(tensor.shape).to_bytes(8, "big"))
        for dimension in tensor.shape:
            digest.update(struct.pack(">q", int(dimension)))
        _update_framed(digest, _tensor_bytes(tensor))
    return digest.hexdigest()


def _transfer_names_sha256(names: list[str]) -> str:
    content = json.dumps(
        names,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise ValueError(f"path contains a symlink: {path}")
        if current == current.parent:
            return
        current = current.parent


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("checkpoint snapshot write made no progress")
        offset += written


@contextmanager
def _open_checkpoint_snapshot(
    path: Path,
    *,
    label: str,
) -> Iterator[_CheckpointSnapshot]:
    source = Path(path)
    if ".." in source.parts:
        raise _CheckpointUnsafeError(f"{label} path traversal is forbidden")
    try:
        _reject_symlink_components(source)
    except ValueError as exc:
        raise _CheckpointUnsafeError(str(exc)) from exc
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise _CheckpointUnsafeError(
            f"{label} requires O_NOFOLLOW support"
        )
    flags = (
        os.O_RDONLY
        | nofollow
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        source_descriptor = os.open(source, flags)
    except FileNotFoundError as exc:
        raise _CheckpointMissingError(f"{label} does not exist: {source}") from exc
    except OSError as exc:
        raise _CheckpointUnsafeError(
            f"{label} cannot be opened safely: {source}"
        ) from exc
    try:
        initial_metadata = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(initial_metadata.st_mode)
            or initial_metadata.st_size < 0
            or initial_metadata.st_size > _CHECKPOINT_PROBE_FILE_LIMIT
        ):
            raise _CheckpointUnsafeError(
                f"{label} must be a bounded regular file: {source}"
            )
        file_size = initial_metadata.st_size
        digest = hashlib.sha256()
        with tempfile.TemporaryDirectory(
            prefix="moving-det-checkpoint-"
        ) as temporary_root:
            snapshot_path = Path(temporary_root) / "checkpoint.pt"
            snapshot_flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
            )
            snapshot_descriptor = os.open(
                snapshot_path,
                snapshot_flags,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            snapshot_stream: BinaryIO | None = None
            try:
                remaining = file_size
                while remaining:
                    chunk = os.read(
                        source_descriptor,
                        min(remaining, _SNAPSHOT_COPY_CHUNK),
                    )
                    if not chunk:
                        raise _CheckpointUnsafeError(
                            f"{label} changed while snapshotting"
                        )
                    digest.update(chunk)
                    _write_all(snapshot_descriptor, chunk)
                    remaining -= len(chunk)
                if os.read(source_descriptor, 1):
                    raise _CheckpointUnsafeError(
                        f"{label} changed while snapshotting"
                    )
                final_metadata = os.fstat(source_descriptor)
                if _file_identity(final_metadata) != _file_identity(
                    initial_metadata
                ):
                    raise _CheckpointUnsafeError(
                        f"{label} changed while snapshotting"
                    )
                snapshot_metadata = os.fstat(snapshot_descriptor)
                if (
                    not stat.S_ISREG(snapshot_metadata.st_mode)
                    or snapshot_metadata.st_size != file_size
                ):
                    raise _CheckpointUnsafeError(
                        f"{label} snapshot is incomplete"
                    )
                os.fchmod(snapshot_descriptor, stat.S_IRUSR)
                os.lseek(snapshot_descriptor, 0, os.SEEK_SET)
                snapshot_stream = os.fdopen(
                    snapshot_descriptor,
                    "rb",
                    closefd=True,
                )
                snapshot_descriptor = -1
                os.unlink(snapshot_path)
                yield _CheckpointSnapshot(
                    source_path=source,
                    path=snapshot_path,
                    stream=snapshot_stream,
                    file_size=file_size,
                    sha256=digest.hexdigest(),
                )
            finally:
                if snapshot_stream is not None:
                    snapshot_stream.close()
                elif snapshot_descriptor >= 0:
                    os.close(snapshot_descriptor)
    finally:
        os.close(source_descriptor)


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    source = Path(path)
    if ".." in source.parts:
        raise ValueError(f"{label} path traversal is forbidden")
    _reject_symlink_components(source)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")
    return source.read_bytes()


def _sha256_regular_file(path: Path, *, label: str) -> str:
    return hashlib.sha256(_regular_file_bytes(path, label=label)).hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = left.resolve(strict=False)
    resolved_right = right.resolve(strict=False)
    return (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    )


def _validated_freeze_paths(source_weights: Path, output: Path) -> tuple[Path, Path]:
    source = Path(source_weights)
    destination = Path(output)
    if ".." in source.parts:
        raise ValueError("source weights path traversal is forbidden")
    _reject_symlink_components(source)
    if not destination.name or ".." in destination.parts:
        raise ValueError("output path traversal is forbidden")
    _reject_symlink_components(destination)
    if _paths_overlap(source, destination):
        raise ValueError("output overlaps source weights")
    if destination.exists() or destination.is_symlink():
        raise ValueError("output directory must not already exist")
    return source.resolve(strict=False), destination.resolve(strict=False)


@contextmanager
def _scoped_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: list[int] = []
    if torch.cuda.is_available():
        cuda_devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _load_ultralytics_state(stream: BinaryIO) -> Mapping[str, Tensor]:
    from ultralytics.nn import tasks as ultralytics_tasks

    stream.seek(0)
    with ultralytics_tasks.temporary_modules(
        modules={
            "ultralytics.yolo.utils": "ultralytics.utils",
            "ultralytics.yolo.v8": "ultralytics.models.yolo",
            "ultralytics.yolo.data": "ultralytics.data",
        },
        attributes={
            "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",
            "ultralytics.nn.tasks.YOLOv10DetectionModel": (
                "ultralytics.nn.tasks.DetectionModel"
            ),
            "ultralytics.utils.loss.v10DetectLoss": (
                "ultralytics.utils.loss.E2EDetectLoss"
            ),
            **(
                {"pathlib.PosixPath": "pathlib.WindowsPath"}
                if ultralytics_tasks.WINDOWS
                else {"pathlib.WindowsPath": "pathlib.PosixPath"}
            ),
        },
    ):
        checkpoint = ultralytics_tasks.torch_load(
            stream,
            map_location="cpu",
        )
    if not isinstance(checkpoint, dict):
        checkpoint = {"model": getattr(checkpoint, "model", None)}
    candidate = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(candidate, torch.nn.Module):
        raise ValueError("Ultralytics checkpoint does not contain a model")
    return candidate.float().state_dict()


def _load_universal_state(stream: BinaryIO) -> Mapping[str, Tensor]:
    return _load_ultralytics_state(stream)


def _build_p2_target(nc: int) -> object:
    from moving_det.ml.models.baseline import create_p2_obb_detector

    return create_p2_obb_detector(weights=None, nc=nc)


def _target_config_sha256() -> str:
    return _sha256_regular_file(_P2_MODEL_CONFIG, label="P2 target config")


def _marked_items(stack: list[object]) -> tuple[int, list[object]] | None:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is _PICKLE_MARK:
            return index, stack[index + 1 :]
    return None


def _probe_pickle_top_level_marker(content: bytes) -> bool | None:
    stream_end = _pickle_stream_end(content, 0)
    if stream_end is None:
        return None
    stack: list[object] = []
    memo: dict[int, object] = {}
    unicode_ops = {
        "STRING",
        "UNICODE",
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE8",
    }
    scalar_ops = {
        "NONE",
        "NEWTRUE",
        "NEWFALSE",
        "INT",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG",
        "LONG1",
        "LONG4",
        "FLOAT",
        "BINFLOAT",
        "BINSTRING",
        "SHORT_BINSTRING",
        "BINBYTES",
        "SHORT_BINBYTES",
        "BINBYTES8",
        "BYTEARRAY8",
        "PERSID",
        "NEXT_BUFFER",
    }
    memo_put_ops = {"PUT", "BINPUT", "LONG_BINPUT"}
    memo_get_ops = {"GET", "BINGET", "LONG_BINGET"}
    try:
        operations = pickletools.genops(content[:stream_end])
        for opcode, argument, _position in operations:
            name = opcode.name
            if name in {"PROTO", "FRAME"}:
                continue
            if name == "MARK":
                stack.append(_PICKLE_MARK)
            elif name in unicode_ops:
                stack.append(argument if isinstance(argument, str) else _PICKLE_OPAQUE)
            elif name in scalar_ops:
                stack.append(_PICKLE_OPAQUE)
            elif name == "EMPTY_DICT":
                stack.append(_ProbeDict())
            elif name in {"EMPTY_LIST", "EMPTY_TUPLE", "EMPTY_SET"}:
                stack.append(_PICKLE_SEQUENCE)
            elif name == "DICT":
                marked = _marked_items(stack)
                if marked is None:
                    return None
                mark_index, items = marked
                del stack[mark_index:]
                mapping = _ProbeDict()
                if len(items) % 2:
                    return None
                for index in range(0, len(items), 2):
                    mapping[items[index]] = items[index + 1]
                stack.append(mapping)
            elif name == "SETITEM":
                if len(stack) < 3:
                    return None
                value = stack.pop()
                key = stack.pop()
                mapping = stack[-1]
                if mapping is _PICKLE_OPAQUE:
                    mapping = _ProbeDict()
                    stack[-1] = mapping
                if isinstance(mapping, _ProbeDict):
                    mapping[key] = value
            elif name == "SETITEMS":
                marked = _marked_items(stack)
                if marked is None:
                    return None
                mark_index, items = marked
                if mark_index == 0 or len(items) % 2:
                    return None
                mapping = stack[mark_index - 1]
                del stack[mark_index:]
                if mapping is _PICKLE_OPAQUE:
                    mapping = _ProbeDict()
                    stack[mark_index - 1] = mapping
                if isinstance(mapping, _ProbeDict):
                    for index in range(0, len(items), 2):
                        mapping[items[index]] = items[index + 1]
            elif name in {"LIST", "TUPLE", "FROZENSET"}:
                marked = _marked_items(stack)
                if marked is None:
                    return None
                mark_index, _items = marked
                del stack[mark_index:]
                stack.append(_PICKLE_SEQUENCE)
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                item_count = int(name[-1])
                if len(stack) < item_count:
                    return None
                del stack[-item_count:]
                stack.append(_PICKLE_SEQUENCE)
            elif name in {"APPEND", "ADDITEM"}:
                if len(stack) < 2:
                    return None
                stack.pop()
            elif name in {"APPENDS", "ADDITEMS"}:
                marked = _marked_items(stack)
                if marked is None or marked[0] == 0:
                    return None
                del stack[marked[0]:]
            elif name in memo_put_ops:
                if not stack:
                    return None
                memo[int(argument)] = stack[-1]
            elif name == "MEMOIZE":
                if not stack:
                    return None
                memo[len(memo)] = stack[-1]
            elif name in memo_get_ops:
                memo_index = int(argument)
                if memo_index not in memo:
                    return None
                stack.append(memo[memo_index])
            elif name == "POP":
                if not stack:
                    return None
                stack.pop()
            elif name == "POP_MARK":
                marked = _marked_items(stack)
                if marked is None:
                    return None
                del stack[marked[0]:]
            elif name == "DUP":
                if not stack:
                    return None
                stack.append(stack[-1])
            elif name in {"GLOBAL", "EXT1", "EXT2", "EXT4"}:
                stack.append(_PICKLE_OPAQUE)
            elif name == "STACK_GLOBAL":
                if len(stack) < 2:
                    return None
                del stack[-2:]
                stack.append(_PICKLE_OPAQUE)
            elif name in {"REDUCE", "NEWOBJ"}:
                if len(stack) < 2:
                    return None
                del stack[-2:]
                stack.append(_PICKLE_OPAQUE)
            elif name == "NEWOBJ_EX":
                if len(stack) < 3:
                    return None
                del stack[-3:]
                stack.append(_PICKLE_OPAQUE)
            elif name == "BUILD":
                if len(stack) < 2:
                    return None
                stack.pop()
            elif name == "BINPERSID":
                if not stack:
                    return None
                stack[-1] = _PICKLE_OPAQUE
            elif name in {"OBJ", "INST"}:
                marked = _marked_items(stack)
                if marked is None:
                    return None
                del stack[marked[0]:]
                stack.append(_PICKLE_OPAQUE)
            elif name == "READONLY_BUFFER":
                if not stack:
                    return None
            elif name == "STOP":
                if not stack:
                    return None
                root = stack[-1]
                if root is _PICKLE_SEQUENCE:
                    return False
                if not isinstance(root, _ProbeDict):
                    return None
                return root.get("artifact_kind") == _ARTIFACT_KIND
            else:
                return None
    except (ValueError, OverflowError):
        return None
    return None


def _legacy_scalar_pickle(
    content: bytes,
    offset: int,
) -> tuple[int, int] | None:
    stream_end = _pickle_stream_end(content, offset)
    if stream_end is None:
        return None
    try:
        operations = iter(pickletools.genops(content[offset:stream_end]))
        first, first_argument, _first_position = next(operations)
        if first.name == "PROTO":
            value, value_argument, _value_position = next(operations)
            while value.name == "FRAME":
                value, value_argument, _value_position = next(operations)
        else:
            value, value_argument = first, first_argument
        stop, _stop_argument, stop_position = next(operations)
    except (StopIteration, ValueError, OverflowError):
        return None
    if (
        value.name
        not in {
            "INT",
            "BININT",
            "BININT1",
            "BININT2",
            "LONG",
            "LONG1",
            "LONG4",
        }
        or type(value_argument) is not int
        or stop.name != "STOP"
    ):
        return None
    return value_argument, offset + stop_position + 1


def _pickle_stream_end(content: bytes, offset: int) -> int | None:
    active_frame_end: int | None = None
    try:
        for opcode, _argument, position in pickletools.genops(content[offset:]):
            absolute_position = offset + position
            if active_frame_end is not None:
                if absolute_position > active_frame_end:
                    return None
                if absolute_position == active_frame_end:
                    active_frame_end = None
            if opcode.name == "FRAME":
                if active_frame_end is not None:
                    return None
                frame_header_end = absolute_position + 9
                if frame_header_end > len(content):
                    return None
                frame_length = struct.unpack_from(
                    "<Q",
                    content,
                    absolute_position + 1,
                )[0]
                if (
                    type(_argument) is not int
                    or _argument != frame_length
                    or frame_length > _PICKLE_PROBE_LIMIT
                ):
                    return None
                frame_end = frame_header_end + frame_length
                if frame_end > len(content):
                    return None
                active_frame_end = frame_end
            elif opcode.name == "STOP":
                stream_end = absolute_position + 1
                if (
                    active_frame_end is not None
                    and stream_end != active_frame_end
                ):
                    return None
                return stream_end
    except (ValueError, OverflowError, struct.error):
        return None
    return None


def _legacy_torch_pickle(snapshot: _CheckpointSnapshot) -> bytes | object:
    try:
        snapshot.stream.seek(0)
        content = snapshot.stream.read(
            _PICKLE_PROBE_LIMIT + _LEGACY_HEADER_LIMIT + 1
        )
    except (OSError, ValueError):
        return _ARCHIVE_INDETERMINATE
    magic = _legacy_scalar_pickle(content, 0)
    if magic is None:
        if content.startswith(b"\x80"):
            return _ARCHIVE_INDETERMINATE
        return _ARCHIVE_NOT_TORCH
    if magic[0] != _LEGACY_TORCH_MAGIC:
        return _ARCHIVE_NOT_TORCH
    protocol = _legacy_scalar_pickle(content, magic[1])
    if protocol is None:
        return _ARCHIVE_INDETERMINATE
    if protocol[0] != _LEGACY_TORCH_PROTOCOL:
        return _ARCHIVE_NOT_TORCH
    system_info_end = _pickle_stream_end(content, protocol[1])
    if system_info_end is None or system_info_end > _LEGACY_HEADER_LIMIT:
        return _ARCHIVE_INDETERMINATE
    payload_end = _pickle_stream_end(content, system_info_end)
    if payload_end is None:
        return _ARCHIVE_INDETERMINATE
    if payload_end - system_info_end > _PICKLE_PROBE_LIMIT:
        return _ARCHIVE_INDETERMINATE
    return content[system_info_end:payload_end]


def _valid_zip64_bridge(
    stream: BinaryIO,
    *,
    directory_end: int,
    eocd_offset: int,
    disk_entries: int,
    total_entries: int,
    directory_size: int,
    directory_offset: int,
) -> bool:
    if directory_end == eocd_offset:
        return True
    if eocd_offset - directory_end != (
        _ZIP64_EOCD_HEADER.size + _ZIP64_EOCD_LOCATOR.size
    ):
        return False
    try:
        stream.seek(directory_end)
        zip64_header = stream.read(_ZIP64_EOCD_HEADER.size)
        locator_header = stream.read(_ZIP64_EOCD_LOCATOR.size)
        if (
            len(zip64_header) != _ZIP64_EOCD_HEADER.size
            or len(locator_header) != _ZIP64_EOCD_LOCATOR.size
        ):
            return False
        zip64_fields = _ZIP64_EOCD_HEADER.unpack(zip64_header)
        locator_fields = _ZIP64_EOCD_LOCATOR.unpack(locator_header)
    except (OSError, ValueError, struct.error):
        return False
    return (
        zip64_fields[0] == b"PK\x06\x06"
        and zip64_fields[1] == _ZIP64_EOCD_HEADER.size - 12
        and zip64_fields[4] == 0
        and zip64_fields[5] == 0
        and zip64_fields[6] == disk_entries
        and zip64_fields[7] == total_entries
        and zip64_fields[8] == directory_size
        and zip64_fields[9] == directory_offset
        and locator_fields[0] == b"PK\x06\x07"
        and locator_fields[1] == 0
        and locator_fields[2] == directory_end
        and locator_fields[3] == 1
    )


def _valid_central_directory(
    stream: BinaryIO,
    *,
    eocd_offset: int,
    disk_entries: int,
    total_entries: int,
    directory_size: int,
    directory_offset: int,
) -> bool:
    directory_end = directory_offset + directory_size
    if not _valid_zip64_bridge(
        stream,
        directory_end=directory_end,
        eocd_offset=eocd_offset,
        disk_entries=disk_entries,
        total_entries=total_entries,
        directory_size=directory_size,
        directory_offset=directory_offset,
    ):
        return False
    entry_count = 0
    consumed = 0
    try:
        stream.seek(directory_offset)
        while consumed < directory_size:
            header = stream.read(_ZIP_CENTRAL_HEADER.size)
            if len(header) != _ZIP_CENTRAL_HEADER.size:
                return False
            fields = _ZIP_CENTRAL_HEADER.unpack(header)
            if (
                fields[0] != b"PK\x01\x02"
                or fields[13] != 0
                or fields[16] >= directory_offset
            ):
                return False
            variable_size = fields[10] + fields[11] + fields[12]
            entry_size = _ZIP_CENTRAL_HEADER.size + variable_size
            if entry_size > directory_size - consumed:
                return False
            stream.seek(variable_size, os.SEEK_CUR)
            consumed += entry_size
            entry_count += 1
            if entry_count > _ZIP_MEMBER_LIMIT:
                return False
    except (OSError, ValueError, struct.error):
        return False
    return consumed == directory_size and entry_count == total_entries


def _bounded_zip_directory(
    snapshot: _CheckpointSnapshot,
) -> tuple[int, int, int] | None:
    file_size = snapshot.file_size
    read_size = min(file_size, _ZIP_EOCD_SEARCH_LIMIT)
    if read_size < 22:
        return None
    try:
        snapshot.stream.seek(file_size - read_size)
        tail = snapshot.stream.read(read_size)
    except (OSError, ValueError):
        return None
    if len(tail) != read_size:
        return None
    candidates: list[tuple[int, int, int]] = []
    search_offset = 0
    tail_absolute_offset = file_size - read_size
    while True:
        relative_offset = tail.find(b"PK\x05\x06", search_offset)
        if relative_offset < 0:
            break
        search_offset = relative_offset + 1
        if relative_offset + _ZIP_EOCD_HEADER.size > len(tail):
            continue
        try:
            (
                signature,
                disk_number,
                directory_disk,
                disk_entries,
                total_entries,
                directory_size,
                directory_offset,
                comment_size,
            ) = _ZIP_EOCD_HEADER.unpack_from(tail, relative_offset)
        except struct.error:
            continue
        absolute_offset = tail_absolute_offset + relative_offset
        if (
            signature != b"PK\x05\x06"
            or absolute_offset + _ZIP_EOCD_HEADER.size + comment_size
            != file_size
            or disk_number != 0
            or directory_disk != 0
            or disk_entries != total_entries
            or total_entries > _ZIP_MEMBER_LIMIT
            or directory_size > _ZIP_CENTRAL_DIRECTORY_LIMIT
            or directory_offset > absolute_offset
            or directory_size > absolute_offset - directory_offset
        ):
            continue
        if not _valid_central_directory(
            snapshot.stream,
            eocd_offset=absolute_offset,
            disk_entries=disk_entries,
            total_entries=total_entries,
            directory_size=directory_size,
            directory_offset=directory_offset,
        ):
            continue
        candidates.append(
            (total_entries, directory_size, absolute_offset)
        )
        if len(candidates) > 1:
            return None
    if len(candidates) != 1:
        return None
    return candidates[0]


def _torch_archive_pickle(snapshot: _CheckpointSnapshot) -> bytes | object:
    try:
        snapshot.stream.seek(0)
        signature = snapshot.stream.read(4)
    except (OSError, ValueError):
        return _ARCHIVE_INDETERMINATE
    if signature != b"PK\x03\x04":
        if signature.startswith(b"PK"):
            return _ARCHIVE_INDETERMINATE
        return _legacy_torch_pickle(snapshot)
    directory = _bounded_zip_directory(snapshot)
    if directory is None:
        return _ARCHIVE_INDETERMINATE
    expected_members, _directory_size, eocd_offset = directory
    try:
        archive_view = _ValidatedZipView(snapshot.stream, eocd_offset)
        with zipfile.ZipFile(archive_view) as archive:
            members = archive.infolist()
            if (
                len(members) != expected_members
                or len(members) > _ZIP_MEMBER_LIMIT
            ):
                return _ARCHIVE_INDETERMINATE
            candidate = None
            for info in members:
                if info.filename == "data.pkl" or info.filename.endswith(
                    "/data.pkl"
                ):
                    if candidate is not None:
                        return _ARCHIVE_INDETERMINATE
                    candidate = info
            if candidate is None:
                return _ARCHIVE_INDETERMINATE
            if (
                type(candidate.file_size) is not int
                or candidate.file_size < 0
                or candidate.file_size > _PICKLE_PROBE_LIMIT
                or type(candidate.compress_size) is not int
                or candidate.compress_size < 0
                or candidate.compress_size > _ZIP_PICKLE_COMPRESSED_LIMIT
            ):
                return _ARCHIVE_INDETERMINATE
            with archive.open(candidate) as stream:
                content = stream.read(_PICKLE_PROBE_LIMIT + 1)
            if (
                len(content) > _PICKLE_PROBE_LIMIT
                or len(content) != candidate.file_size
            ):
                return _ARCHIVE_INDETERMINATE
            return content
    except zipfile.BadZipFile:
        return _ARCHIVE_INDETERMINATE
    except (OSError, ValueError, EOFError, NotImplementedError, RuntimeError):
        return _ARCHIVE_INDETERMINATE


def _static_frozen_marker_evidence_from_snapshot(
    snapshot: _CheckpointSnapshot,
) -> bool | None:
    content = _torch_archive_pickle(snapshot)
    if content is _ARCHIVE_NOT_TORCH:
        return False
    if content is _ARCHIVE_INDETERMINATE:
        return None
    assert isinstance(content, bytes)
    probed = _probe_pickle_top_level_marker(content)
    return probed


def _static_frozen_marker_evidence(path: Path) -> bool | None:
    try:
        with _open_checkpoint_snapshot(path, label="checkpoint") as snapshot:
            return _static_frozen_marker_evidence_from_snapshot(snapshot)
    except _CheckpointMissingError:
        return False
    except _CheckpointUnsafeError:
        return None


def _is_frozen_p2_initialization(path: Path) -> bool:
    marker_evidence = _static_frozen_marker_evidence(path)
    return marker_evidence is not False


def _shape(value: Tensor) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def _transfer_report_entries(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    loaded = []
    missing = []
    mismatched = []
    unused = []
    for name in sorted(target):
        if name not in source:
            missing.append({"name": name, "shape": _shape(target[name])})
        elif source[name].shape == target[name].shape:
            loaded.append({"name": name, "shape": _shape(target[name])})
        else:
            mismatched.append(
                {
                    "name": name,
                    "source_shape": _shape(source[name]),
                    "target_shape": _shape(target[name]),
                }
            )
    for name in sorted(set(source) - set(target)):
        unused.append({"name": name, "shape": _shape(source[name])})
    return loaded, missing, mismatched, unused


def _torch_payload_bytes(payload: Mapping[str, object]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_artifacts(
    output: Path,
    *,
    checkpoint: bytes,
    report: bytes,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("output directory must not already exist")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    try:
        checkpoint_path = staging / "p2-init.pt"
        report_path = staging / "transfer_report.json"
        _write_file(checkpoint_path, checkpoint)
        _write_file(report_path, report)
        run = {
            "artifact_kind": _ARTIFACT_KIND,
            "schema_version": _SCHEMA_VERSION,
            "artifacts": {
                "p2-init.pt": {
                    "sha256": hashlib.sha256(checkpoint).hexdigest()
                },
                "transfer_report.json": {
                    "sha256": hashlib.sha256(report).hexdigest()
                },
            },
        }
        _write_file(staging / "run.json", _canonical_json_bytes(run))
        _fsync_directory(staging)
        if destination.exists() or destination.is_symlink():
            raise ValueError("output directory must not already exist")
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return destination / "p2-init.pt"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def freeze_p2_initialization(
    source_weights: Path,
    output: Path,
    seed: int = 20260806,
    nc: int = 4,
) -> Path:
    """Freeze the approved Universal checkpoint into a deterministic P2 artifact."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(nc) is not int or nc != _EXPECTED_NC:
        raise ValueError("Universal P2 initialization requires integer nc=4")
    requested_source = Path(source_weights)
    approved_lexical_path = Path(APPROVED_UNIVERSAL_PATH)
    if not requested_source.is_absolute():
        raise ValueError(
            "source weights must use the absolute approved Universal path: "
            f"{approved_lexical_path}"
        )
    _reject_symlink_components(requested_source)
    if (
        not approved_lexical_path.is_absolute()
        or requested_source != approved_lexical_path
    ):
        raise ValueError(
            "source weights must use the absolute approved Universal path: "
            f"{approved_lexical_path}"
        )
    source_path, output_path = _validated_freeze_paths(source_weights, output)
    approved_path = approved_lexical_path.resolve(strict=False)
    if source_path != approved_path:
        raise ValueError(
            f"source weights must use the approved Universal path: {approved_path}"
        )
    with _open_checkpoint_snapshot(
        source_path,
        label="source weights",
    ) as source_snapshot:
        source_sha256 = source_snapshot.sha256
        if source_sha256 != APPROVED_UNIVERSAL_SHA256:
            raise ValueError(
                "source weights SHA-256 is not the approved Universal hash"
            )
        with _scoped_rng(seed):
            loaded_source = _load_universal_state(source_snapshot.stream)
    source_state = _validated_state(loaded_source, label="source")
    with _scoped_rng(seed):
        target = _build_p2_target(nc)
    target_state = _validated_state(target.state_dict(), label="target")
    if len(target_state) != _EXPECTED_TARGET_COUNT:
        raise ValueError(
            f"P2 target must contain exactly {_EXPECTED_TARGET_COUNT} tensors"
        )
    transferred = compatible_state(source_state, target_state)
    if len(transferred) != _EXPECTED_TRANSFER_COUNT:
        raise ValueError(
            "Universal-to-P2 transfer must contain exactly "
            f"{_EXPECTED_TRANSFER_COUNT} tensors"
        )
    target.load_state_dict(transferred, strict=False)
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in sorted(
            _validated_state(target.state_dict(), label="final target").items()
        )
    }
    if len(model_state) != _EXPECTED_TARGET_COUNT:
        raise ValueError("final P2 target tensor count changed during transfer")

    loaded, missing, mismatched, unused = _transfer_report_entries(
        source_state,
        target_state,
    )
    loaded_names = [entry["name"] for entry in loaded]
    transfer_names_sha256 = _transfer_names_sha256(loaded_names)
    loaded_tensors_sha256 = _state_sha256(
        {name: model_state[name] for name in loaded_names}
    )
    model_state_sha256 = _state_sha256(model_state)
    target_config_sha256 = _target_config_sha256()
    report = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "nc": nc,
        "source_weights_sha256": source_sha256,
        "target_config_sha256": target_config_sha256,
        "source_count": len(source_state),
        "target_count": len(target_state),
        "loaded_count": len(loaded),
        "missing_in_source_count": len(missing),
        "shape_mismatch_count": len(mismatched),
        "unused_source_count": len(unused),
        "transfer_names_sha256": transfer_names_sha256,
        "loaded_tensors_sha256": loaded_tensors_sha256,
        "model_state_sha256": model_state_sha256,
        "loaded": loaded,
        "missing_in_source": missing,
        "shape_mismatch": mismatched,
        "unused_source": unused,
    }
    payload = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "nc": nc,
        "source_weights_sha256": source_sha256,
        "target_config_sha256": target_config_sha256,
        "transfer_names_sha256": transfer_names_sha256,
        "model_state_sha256": model_state_sha256,
        "model_state": model_state,
    }
    return _publish_artifacts(
        output_path,
        checkpoint=_torch_payload_bytes(payload),
        report=_canonical_json_bytes(report),
    )


def _artifact_bytes(root: Path, name: str) -> bytes:
    path = root / name
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"frozen artifact is missing or unsafe: {name}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"frozen artifact escapes its root: {name}")
    return path.read_bytes()


def _entry_list(report: Mapping[str, object], name: str) -> list[dict[str, object]]:
    value = report[name]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"transfer report {name} must be a list of objects")
    entries = [dict(item) for item in value]
    names = [entry.get("name") for entry in entries]
    if any(not isinstance(item, str) or not item for item in names):
        raise ValueError(f"transfer report {name} has an invalid tensor name")
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(f"transfer report {name} names must be unique and sorted")
    return entries


def _validate_report(
    report: Mapping[str, object],
    payload: Mapping[str, object],
    model_state: Mapping[str, Tensor],
) -> dict[str, object]:
    _require_exact_fields(report, _REPORT_FIELDS, label="transfer report")
    _require_identity_fields(
        report,
        label="transfer report",
        include_seed_and_nc=True,
    )
    for hash_name in (
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "loaded_tensors_sha256",
        "model_state_sha256",
    ):
        if not _is_sha256(report[hash_name]):
            raise ValueError(f"transfer report {hash_name} is invalid")
    loaded = _entry_list(report, "loaded")
    missing = _entry_list(report, "missing_in_source")
    mismatched = _entry_list(report, "shape_mismatch")
    unused = _entry_list(report, "unused_source")
    expected_entry_fields = {
        "loaded": frozenset({"name", "shape"}),
        "missing_in_source": frozenset({"name", "shape"}),
        "shape_mismatch": frozenset({"name", "source_shape", "target_shape"}),
        "unused_source": frozenset({"name", "shape"}),
    }
    for label, entries in (
        ("loaded", loaded),
        ("missing_in_source", missing),
        ("shape_mismatch", mismatched),
        ("unused_source", unused),
    ):
        for entry in entries:
            if set(entry) != expected_entry_fields[label]:
                raise ValueError(f"transfer report {label} entry fields are invalid")
            if label == "shape_mismatch":
                source_shape = _require_shape(
                    entry["source_shape"],
                    label="transfer report shape_mismatch source",
                )
                target_shape = _require_shape(
                    entry["target_shape"],
                    label="transfer report shape_mismatch target",
                )
                if source_shape == target_shape:
                    raise ValueError(
                        "transfer report shape_mismatch shapes must differ"
                    )
            else:
                _require_shape(
                    entry["shape"],
                    label=f"transfer report {label}",
                )

    count_lists = {
        "loaded_count": loaded,
        "missing_in_source_count": missing,
        "shape_mismatch_count": mismatched,
        "unused_source_count": unused,
    }
    for count_name, entries in count_lists.items():
        if _require_nonnegative_plain_int(
            report[count_name],
            label=f"transfer report {count_name}",
        ) != len(entries):
            label = count_name.removesuffix("_count").replace("_", " ")
            raise ValueError(f"transfer report {label} count is inconsistent")
    source_count = _require_nonnegative_plain_int(
        report["source_count"],
        label="transfer report source_count",
    )
    target_count = _require_nonnegative_plain_int(
        report["target_count"],
        label="transfer report target_count",
    )
    if source_count != len(loaded) + len(mismatched) + len(unused):
        raise ValueError("transfer report source count is inconsistent")
    if target_count != len(loaded) + len(mismatched) + len(missing):
        raise ValueError("transfer report target count is inconsistent")
    if len(loaded) != _EXPECTED_TRANSFER_COUNT:
        raise ValueError("transfer report loaded count is not production-compatible")
    if target_count != _EXPECTED_TARGET_COUNT or len(model_state) != target_count:
        raise ValueError("transfer report target count is not production-compatible")

    loaded_names = [str(entry["name"]) for entry in loaded]
    missing_names = [str(entry["name"]) for entry in missing]
    mismatched_names = [str(entry["name"]) for entry in mismatched]
    unused_names = [str(entry["name"]) for entry in unused]
    target_partitions = (
        set(loaded_names),
        set(missing_names),
        set(mismatched_names),
    )
    if (
        target_partitions[0] & target_partitions[1]
        or target_partitions[0] & target_partitions[2]
        or target_partitions[1] & target_partitions[2]
        or set().union(*target_partitions) != set(model_state)
        or set(unused_names) & set(model_state)
    ):
        raise ValueError("transfer report tensor categories are not an exact partition")
    names_sha256 = _transfer_names_sha256(loaded_names)
    if report["transfer_names_sha256"] != names_sha256:
        raise ValueError("transfer report loaded-name hash is inconsistent")
    loaded_state: dict[str, Tensor] = {}
    for entry in loaded:
        name = str(entry["name"])
        if name not in model_state or entry["shape"] != _shape(model_state[name]):
            raise ValueError("transfer report loaded tensor shape is inconsistent")
        loaded_state[name] = model_state[name]
    if report["loaded_tensors_sha256"] != _state_sha256(loaded_state):
        raise ValueError("transfer report loaded tensor hash is inconsistent")
    for entry in missing:
        name = str(entry["name"])
        if name not in model_state or entry["shape"] != _shape(model_state[name]):
            raise ValueError("transfer report missing tensor shape is inconsistent")
    for entry in mismatched:
        name = str(entry["name"])
        if (
            name not in model_state
            or entry["target_shape"] != _shape(model_state[name])
        ):
            raise ValueError("transfer report shape mismatch entry is inconsistent")

    shared_fields = (
        "artifact_kind",
        "schema_version",
        "seed",
        "nc",
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "model_state_sha256",
    )
    if report["source_weights_sha256"] != payload["source_weights_sha256"]:
        raise ValueError("checkpoint and transfer report source SHA disagree")
    if any(report[name] != payload[name] for name in shared_fields):
        raise ValueError("checkpoint and transfer report provenance disagree")
    if report["model_state_sha256"] != _state_sha256(model_state):
        raise ValueError("frozen model-state tensor hash is inconsistent")
    return dict(report)


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _load_frozen_p2_initialization_snapshot(
    path: Path,
    snapshot: _CheckpointSnapshot,
) -> tuple[dict[str, Tensor], Mapping[str, object]]:
    artifact = Path(path)
    if artifact.name != "p2-init.pt":
        raise ValueError("frozen initialization path must name p2-init.pt")
    root = artifact.parent.resolve(strict=True)
    _reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen initialization root must be a regular directory")
    children = {child.name for child in root.iterdir()}
    if children != _FROZEN_CHILDREN:
        raise ValueError("frozen initialization children do not match schema")

    report_content = _artifact_bytes(root, "transfer_report.json")
    run_content = _artifact_bytes(root, "run.json")
    report = _load_canonical_json_bytes(report_content, label="transfer report")
    run = _load_canonical_json_bytes(run_content, label="run metadata")
    _require_exact_fields(run, _RUN_FIELDS, label="run metadata")
    _require_identity_fields(
        run,
        label="run metadata",
        include_seed_and_nc=False,
    )
    artifacts = run["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "p2-init.pt",
        "transfer_report.json",
    }:
        raise ValueError("run metadata artifact set is invalid")
    digests = {
        "p2-init.pt": snapshot.sha256,
        "transfer_report.json": hashlib.sha256(report_content).hexdigest(),
    }
    for name, actual_digest in digests.items():
        entry = artifacts[name]
        if not isinstance(entry, dict) or set(entry) != {"sha256"}:
            raise ValueError("run metadata artifact fingerprint is invalid")
        expected = entry["sha256"]
        if not _is_sha256(expected) or expected != actual_digest:
            raise ValueError(f"frozen artifact SHA-256 mismatch: {name}")

    try:
        snapshot.stream.seek(0)
        payload = torch.load(
            snapshot.stream,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError("frozen checkpoint payload is malformed") from exc
    if not isinstance(payload, dict):
        raise ValueError("frozen checkpoint payload must be a mapping")
    _require_exact_fields(payload, _PAYLOAD_FIELDS, label="frozen checkpoint")
    _require_identity_fields(
        payload,
        label="frozen checkpoint",
        include_seed_and_nc=True,
    )
    for name in (
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "model_state_sha256",
    ):
        if not _is_sha256(payload[name]):
            raise ValueError(f"frozen checkpoint {name} is invalid")
    if payload["source_weights_sha256"] != APPROVED_UNIVERSAL_SHA256:
        raise ValueError("frozen checkpoint source SHA is not approved")
    if payload["target_config_sha256"] != _target_config_sha256():
        raise ValueError("frozen checkpoint target config hash is unexpected")
    raw_state = payload["model_state"]
    if not isinstance(raw_state, Mapping):
        raise ValueError("frozen checkpoint model_state must be a mapping")
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in sorted(
            _validated_state(raw_state, label="frozen model").items()
        )
    }
    if tuple(raw_state) != tuple(sorted(raw_state)):
        raise ValueError("frozen checkpoint model_state names must be sorted")
    validated_report = _validate_report(report, payload, model_state)
    provenance = {
        **validated_report,
        "initialization_kind": "frozen_p2",
        "transferred_tensors": validated_report["loaded_count"],
        "target_tensors": validated_report["target_count"],
    }
    return model_state, _deep_freeze(provenance)


def load_frozen_p2_initialization(
    path: Path,
) -> tuple[dict[str, Tensor], Mapping[str, object]]:
    """Strictly verify and load a frozen Universal-P2 initialization artifact."""
    artifact = Path(path)
    with _open_checkpoint_snapshot(
        artifact,
        label="frozen checkpoint",
    ) as snapshot:
        return _load_frozen_p2_initialization_snapshot(artifact, snapshot)
