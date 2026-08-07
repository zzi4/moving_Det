from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from moving_det.motion.alignment import AlignmentResult
from moving_det.vrud.tiling import Tile


_CACHE_SCHEMA_VERSION = 1
_SAFE_KEY_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARTIFACT_NAME = re.compile(r"^[0-9a-f]{64}-[0-9a-f]{64}\.npz$")
_NPZ_FIELDS = {
    "matrix",
    "correlation",
    "used_fallback",
    "reason_present",
    "reason",
}


def _validate_key_part(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or not _SAFE_KEY_PART.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ValueError(f"{field} must be a traversal-safe identifier")


@dataclass(frozen=True)
class AlignmentKey:
    site: str
    sequence: str
    center_frame: int
    support_frame: int

    def __post_init__(self) -> None:
        _validate_key_part(self.site, "site")
        _validate_key_part(self.sequence, "sequence")
        for value, field in (
            (self.center_frame, "center_frame"),
            (self.support_frame, "support_frame"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")


def _validate_affine(matrix: object) -> np.ndarray:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.dtype(np.float32)
        or matrix.shape != (2, 3)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("alignment matrix must be a finite float32 2x3 array")
    return matrix


def localize_affine(global_matrix: np.ndarray, tile: Tile) -> np.ndarray:
    matrix = _validate_affine(global_matrix)
    if not isinstance(tile, Tile):
        raise ValueError("tile must be a Tile")

    global_h = np.eye(3, dtype=np.float64)
    global_h[:2] = matrix.astype(np.float64)
    crop = np.asarray(
        [
            [1.0, 0.0, float(tile.x)],
            [0.0, 1.0, float(tile.y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    localized = np.linalg.inv(crop) @ global_h @ crop
    return localized[:2].astype(np.float32)


def _key_payload(key: AlignmentKey) -> dict[str, object]:
    return {
        "site": key.site,
        "sequence": key.sequence,
        "center_frame": key.center_frame,
        "support_frame": key.support_frame,
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _key_digest(key: AlignmentKey) -> str:
    return hashlib.sha256(_canonical_json(_key_payload(key))).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_result(result: AlignmentResult) -> None:
    if not isinstance(result, AlignmentResult):
        raise ValueError("cache result must be an AlignmentResult")
    _validate_affine(result.matrix)
    if (
        isinstance(result.correlation, bool)
        or not isinstance(result.correlation, (float, np.floating))
        or not np.isfinite(result.correlation)
    ):
        raise ValueError("alignment correlation must be finite")
    if not isinstance(result.used_fallback, bool):
        raise ValueError("used_fallback must be a boolean")
    if result.used_fallback:
        if (
            not isinstance(result.reason, str)
            or not result.reason
            or len(result.reason) > 1024
            or "\x00" in result.reason
        ):
            raise ValueError("fallback alignment must have a valid reason")
    elif result.reason is not None:
        raise ValueError("successful alignment reason must be None")


class AlignmentCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"

    def _empty_index(self) -> dict[str, object]:
        return {"schema_version": _CACHE_SCHEMA_VERSION, "entries": {}}

    def _read_index(self) -> dict[str, object]:
        if not self.index_path.exists():
            return self._empty_index()
        if self.index_path.is_symlink() or not self.index_path.is_file():
            raise ValueError("alignment cache index is not a regular file")
        try:
            with self.index_path.open("r", encoding="utf-8") as stream:
                index = json.load(stream, object_pairs_hook=_strict_json_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("alignment cache index is malformed") from exc
        self._validate_index(index)
        return index

    def _validate_index(self, index: object) -> None:
        if not isinstance(index, dict) or set(index) != {
            "schema_version",
            "entries",
        }:
            raise ValueError("alignment cache index has invalid fields")
        if (
            type(index["schema_version"]) is not int
            or index["schema_version"] != _CACHE_SCHEMA_VERSION
        ):
            raise ValueError("alignment cache index has unsupported schema")
        entries = index["entries"]
        if not isinstance(entries, dict):
            raise ValueError("alignment cache entries must be an object")
        for digest, entry in entries.items():
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(entry, dict)
                or set(entry) != {"key", "artifact", "sha256"}
            ):
                raise ValueError("alignment cache entry is malformed")
            key_payload = entry["key"]
            if not isinstance(key_payload, dict) or set(key_payload) != {
                "site",
                "sequence",
                "center_frame",
                "support_frame",
            }:
                raise ValueError("alignment cache key payload is malformed")
            try:
                key = AlignmentKey(**key_payload)
            except (TypeError, ValueError) as exc:
                raise ValueError("alignment cache key payload is invalid") from exc
            if _key_digest(key) != digest:
                raise ValueError("alignment cache key digest is mismatched")
            artifact = entry["artifact"]
            checksum = entry["sha256"]
            if (
                not isinstance(artifact, str)
                or not _ARTIFACT_NAME.fullmatch(artifact)
                or Path(artifact).name != artifact
                or not artifact.startswith(f"{digest}-")
                or not isinstance(checksum, str)
                or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or artifact != f"{digest}-{checksum}.npz"
            ):
                raise ValueError("alignment cache artifact path is unsafe")

    def _load_result(self, artifact: str, checksum: str) -> AlignmentResult:
        path = self.root / artifact
        if path.is_symlink() or not path.is_file():
            raise ValueError("alignment cache artifact is not a regular file")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError("alignment cache artifact is missing") from exc
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError("alignment cache artifact checksum is mismatched")
        try:
            with np.load(path, allow_pickle=False) as stored:
                if set(stored.files) != _NPZ_FIELDS:
                    raise ValueError("alignment cache artifact fields are invalid")
                matrix = stored["matrix"]
                correlation = stored["correlation"]
                used_fallback = stored["used_fallback"]
                reason_present = stored["reason_present"]
                reason = stored["reason"]
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError("alignment cache artifact is corrupt") from exc

        if (
            not isinstance(matrix, np.ndarray)
            or matrix.shape != (2, 3)
            or matrix.dtype != np.dtype(np.float32)
            or not np.isfinite(matrix).all()
            or correlation.shape != ()
            or correlation.dtype != np.dtype(np.float64)
            or not np.isfinite(correlation.item())
            or used_fallback.shape != ()
            or used_fallback.dtype != np.dtype(np.bool_)
            or reason_present.shape != ()
            or reason_present.dtype != np.dtype(np.bool_)
            or reason.shape != ()
            or reason.dtype.kind != "U"
        ):
            raise ValueError("alignment cache artifact values are invalid")

        has_reason = bool(reason_present.item())
        reason_text = str(reason.item())
        if has_reason != bool(reason_text):
            raise ValueError("alignment cache artifact reason is inconsistent")
        result = AlignmentResult(
            matrix=matrix.copy(),
            correlation=float(correlation.item()),
            used_fallback=bool(used_fallback.item()),
            reason=reason_text if has_reason else None,
        )
        try:
            _validate_result(result)
        except ValueError as exc:
            raise ValueError("alignment cache artifact result is invalid") from exc
        return result

    def get(self, key: AlignmentKey) -> AlignmentResult | None:
        if not isinstance(key, AlignmentKey):
            raise ValueError("cache key must be an AlignmentKey")
        index = self._read_index()
        entries = index["entries"]
        assert isinstance(entries, dict)
        entry = entries.get(_key_digest(key))
        if entry is None:
            return None
        assert isinstance(entry, dict)
        if entry["key"] != _key_payload(key):
            raise ValueError("alignment cache requested key is mismatched")
        return self._load_result(entry["artifact"], entry["sha256"])

    def put(self, key: AlignmentKey, result: AlignmentResult) -> None:
        if not isinstance(key, AlignmentKey):
            raise ValueError("cache key must be an AlignmentKey")
        _validate_result(result)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError("alignment cache root is not a directory")

        index = self._read_index()
        key_digest = _key_digest(key)
        artifact_temp: Path | None = None
        index_temp: Path | None = None
        try:
            artifact_fd, artifact_name = tempfile.mkstemp(
                prefix=f".{key_digest}-",
                suffix=".npz.tmp",
                dir=self.root,
            )
            artifact_temp = Path(artifact_name)
            with os.fdopen(artifact_fd, "wb") as stream:
                np.savez_compressed(
                    stream,
                    matrix=result.matrix,
                    correlation=np.asarray(result.correlation, dtype=np.float64),
                    used_fallback=np.asarray(
                        result.used_fallback,
                        dtype=np.bool_,
                    ),
                    reason_present=np.asarray(
                        result.reason is not None,
                        dtype=np.bool_,
                    ),
                    reason=np.asarray(result.reason or "", dtype=np.str_),
                )
                stream.flush()
                os.fsync(stream.fileno())
            payload = artifact_temp.read_bytes()
            checksum = hashlib.sha256(payload).hexdigest()
            artifact = f"{key_digest}-{checksum}.npz"
            os.replace(artifact_temp, self.root / artifact)
            artifact_temp = None

            entries = dict(index["entries"])
            entries[key_digest] = {
                "key": _key_payload(key),
                "artifact": artifact,
                "sha256": checksum,
            }
            updated = {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "entries": entries,
            }
            index_fd, index_name = tempfile.mkstemp(
                prefix=".index-",
                suffix=".json.tmp",
                dir=self.root,
            )
            index_temp = Path(index_name)
            with os.fdopen(index_fd, "wb") as stream:
                stream.write(_canonical_json(updated))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(index_temp, self.index_path)
            index_temp = None
        finally:
            if artifact_temp is not None:
                artifact_temp.unlink(missing_ok=True)
            if index_temp is not None:
                index_temp.unlink(missing_ok=True)
