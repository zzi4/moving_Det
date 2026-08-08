from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from types import MappingProxyType
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
_LOCK_NAME = ".alignment-cache.lock"
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


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


@dataclass(frozen=True)
class AlignmentSnapshot:
    fingerprint: str
    _results: Mapping[AlignmentKey, AlignmentResult]

    def get(self, key: AlignmentKey) -> AlignmentResult | None:
        if not isinstance(key, AlignmentKey):
            raise ValueError("snapshot key must be an AlignmentKey")
        return self._results.get(key)


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

    try:
        tile_x = float(tile.x)
        tile_y = float(tile.y)
        global_h = np.eye(3, dtype=np.float64)
        global_h[:2] = matrix.astype(np.float64)
        crop = np.asarray(
            [
                [1.0, 0.0, tile_x],
                [0.0, 1.0, tile_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        with np.errstate(over="raise", invalid="raise"):
            localized = np.linalg.inv(crop) @ global_h @ crop
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "localized affine must have a finite float32 representation"
        ) from exc
    if not np.isfinite(crop).all() or not np.isfinite(localized).all():
        raise ValueError(
            "localized affine must have a finite float32 representation"
        )
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = localized[:2].astype(np.float32)
    except FloatingPointError as exc:
        raise ValueError(
            "localized affine must have a finite float32 representation"
        ) from exc
    if not np.isfinite(result).all():
        raise ValueError(
            "localized affine must have a finite float32 representation"
        )
    return result


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


def _root_thread_lock(root: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(root))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AlignmentCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"

    def _empty_index(self) -> dict[str, object]:
        return {"schema_version": _CACHE_SCHEMA_VERSION, "entries": {}}

    def _read_index(self) -> dict[str, object]:
        if self.index_path.is_symlink():
            raise ValueError("alignment cache index is not a regular file")
        if not self.index_path.exists():
            return self._empty_index()
        if not self.index_path.is_file():
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
            with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
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

    @contextmanager
    def _exclusive_process_lock(self):
        lock_path = self.root / _LOCK_NAME
        if lock_path.is_symlink():
            raise ValueError("alignment cache lock must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            if exc.errno in {
                errno.ELOOP,
                errno.EISDIR,
                errno.ENOENT,
                errno.ENOTDIR,
            }:
                raise ValueError(
                    "alignment cache lock path is unsafe"
                ) from exc
            raise

        locked = False
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise ValueError(
                    "alignment cache lock must be a regular file"
                )
            try:
                path_stat = os.stat(lock_path, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(
                    "alignment cache lock path changed during open"
                ) from exc
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_dev != descriptor_stat.st_dev
                or path_stat.st_ino != descriptor_stat.st_ino
            ):
                raise ValueError(
                    "alignment cache lock path changed during open"
                )

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            try:
                locked_path_stat = os.stat(lock_path, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(
                    "alignment cache lock path changed while locked"
                ) from exc
            if (
                not stat.S_ISREG(locked_path_stat.st_mode)
                or locked_path_stat.st_dev != descriptor_stat.st_dev
                or locked_path_stat.st_ino != descriptor_stat.st_ino
            ):
                raise ValueError(
                    "alignment cache lock path changed while locked"
                )
            yield
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

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

    def _snapshot_from_index(
        self,
        index: dict[str, object],
    ) -> AlignmentSnapshot:
        entries = index["entries"]
        assert isinstance(entries, dict)
        results: dict[AlignmentKey, AlignmentResult] = {}
        for digest in sorted(entries):
            entry = entries[digest]
            assert isinstance(entry, dict)
            key_payload = entry["key"]
            assert isinstance(key_payload, dict)
            key = AlignmentKey(**key_payload)
            loaded = self._load_result(
                entry["artifact"],
                entry["sha256"],
            )
            immutable_matrix = np.frombuffer(
                loaded.matrix.tobytes(order="C"),
                dtype=np.float32,
            ).reshape(2, 3)
            results[key] = AlignmentResult(
                matrix=immutable_matrix,
                correlation=loaded.correlation,
                used_fallback=loaded.used_fallback,
                reason=loaded.reason,
            )
        fingerprint = hashlib.sha256(_canonical_json(index)).hexdigest()
        return AlignmentSnapshot(
            fingerprint=fingerprint,
            _results=MappingProxyType(results),
        )

    def snapshot(self) -> AlignmentSnapshot:
        with _root_thread_lock(self.root):
            if not self.root.exists():
                return self._snapshot_from_index(self._empty_index())
            if not self.root.is_dir():
                raise ValueError("alignment cache root is not a directory")
            with self._exclusive_process_lock():
                return self._snapshot_from_index(self._read_index())

    def _write_artifact(
        self,
        key_digest: str,
        result: AlignmentResult,
    ) -> tuple[str, str]:
        artifact_temp: Path | None = None
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
                    correlation=np.asarray(
                        result.correlation,
                        dtype=np.float64,
                    ),
                    used_fallback=np.asarray(
                        result.used_fallback,
                        dtype=np.bool_,
                    ),
                    reason_present=np.asarray(
                        result.reason is not None,
                        dtype=np.bool_,
                    ),
                    reason=np.asarray(
                        result.reason or "",
                        dtype=np.str_,
                    ),
                )
                stream.flush()
                os.fsync(stream.fileno())
            payload = artifact_temp.read_bytes()
            checksum = hashlib.sha256(payload).hexdigest()
            artifact = f"{key_digest}-{checksum}.npz"
            os.replace(artifact_temp, self.root / artifact)
            artifact_temp = None
            return artifact, checksum
        finally:
            if artifact_temp is not None:
                artifact_temp.unlink(missing_ok=True)

    def _publish_index(self, index: dict[str, object]) -> None:
        index_temp: Path | None = None
        try:
            index_fd, index_name = tempfile.mkstemp(
                prefix=".index-",
                suffix=".json.tmp",
                dir=self.root,
            )
            index_temp = Path(index_name)
            with os.fdopen(index_fd, "wb") as stream:
                stream.write(_canonical_json(index))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(index_temp, self.index_path)
            index_temp = None
            _fsync_directory(self.root)
        finally:
            if index_temp is not None:
                index_temp.unlink(missing_ok=True)

    def put_many(
        self,
        pairs: Iterable[tuple[AlignmentKey, AlignmentResult]],
    ) -> None:
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Iterable):
            raise ValueError("cache batch must be a finite iterable of pairs")
        materialized = tuple(pairs)
        validated: list[tuple[AlignmentKey, AlignmentResult, str]] = []
        key_digests: set[str] = set()
        for item in materialized:
            if isinstance(item, (str, bytes)):
                raise ValueError("cache batch items must be key/result pairs")
            try:
                key, result = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cache batch items must be key/result pairs"
                ) from exc
            if not isinstance(key, AlignmentKey):
                raise ValueError("cache batch key must be an AlignmentKey")
            _validate_result(result)
            key_digest = _key_digest(key)
            if key_digest in key_digests:
                raise ValueError("cache batch contains a duplicate key")
            key_digests.add(key_digest)
            validated.append((key, result, key_digest))
        if not validated:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError("alignment cache root is not a directory")

        with _root_thread_lock(self.root):
            with self._exclusive_process_lock():
                index = self._read_index()
                entries = dict(index["entries"])
                for key, result, key_digest in validated:
                    artifact, checksum = self._write_artifact(
                        key_digest,
                        result,
                    )
                    entries[key_digest] = {
                        "key": _key_payload(key),
                        "artifact": artifact,
                        "sha256": checksum,
                    }
                _fsync_directory(self.root)
                self._publish_index(
                    {
                        "schema_version": _CACHE_SCHEMA_VERSION,
                        "entries": entries,
                    }
                )

    def put(self, key: AlignmentKey, result: AlignmentResult) -> None:
        if not isinstance(key, AlignmentKey):
            raise ValueError("cache key must be an AlignmentKey")
        _validate_result(result)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError("alignment cache root is not a directory")

        with _root_thread_lock(self.root):
            with self._exclusive_process_lock():
                index = self._read_index()
                key_digest = _key_digest(key)
                artifact, checksum = self._write_artifact(key_digest, result)
                _fsync_directory(self.root)
                entries = dict(index["entries"])
                entries[key_digest] = {
                    "key": _key_payload(key),
                    "artifact": artifact,
                    "sha256": checksum,
                }
                self._publish_index(
                    {
                        "schema_version": _CACHE_SCHEMA_VERSION,
                        "entries": entries,
                    }
                )
