from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from moving_det.ml.human_benchmark_artifacts import (
    human_benchmark_fingerprint,
    load_human_benchmark,
)
from moving_det.ml.pretrained_transfer import (
    APPROVED_UNIVERSAL_SHA256 as _APPROVED_UNIVERSAL_SHA256,
    load_frozen_p2_initialization,
)
from moving_det.ml.training import manifest_fingerprint
from moving_det.vrud.alignment import AlignmentCache


APPROVED_HUMAN_SHA256 = (
    "90c00eadb50d38cc3be0ffd8e30399041855f8be81804e83288304160178b851"
)
APPROVED_UNIVERSAL_SHA256 = _APPROVED_UNIVERSAL_SHA256
APPROVED_P2_SHA256 = (
    "d474b9cc8aa113e72de0352bfe4e45aea6b0b7c7a28f67de889214d495428948"
)

_MINIMUM_FORMAL_FREE_BYTES = 100 * 1024**3
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ALIGNMENT_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_sha256",
        "alignment_cache_sha256",
        "seed",
        "job_count",
        "fallback_count",
        "fallback_fraction",
        "fallback_reasons",
        "offsets",
        "center_count",
        "worker_count",
        "opencv_threads_per_worker",
        "center_decode_reuse",
        "cache_write_mode",
    }
)


@dataclass(frozen=True)
class FormalExperimentLayout:
    root: Path
    preflight: Path
    baseline: Path
    baseline_validation: Path
    mg_full: Path
    mg_validation: Path
    mg_motion_off: Path
    mg_frozen: Path
    human_test: Path
    demo: Path
    report: Path

    @classmethod
    def from_root(cls, root: Path) -> "FormalExperimentLayout":
        root = Path(root)
        return cls(
            root=root,
            preflight=root / "preflight",
            baseline=root / "baseline",
            baseline_validation=root / "baseline-validation",
            mg_full=root / "mg-vtod-full",
            mg_validation=root / "mg-validation",
            mg_motion_off=root / "mg-motion-off-validation",
            mg_frozen=root / "mg-frozen",
            human_test=root / "human-test",
            demo=root / "demo",
            report=root / "report",
        )

    def artifact_directories(self) -> tuple[Path, ...]:
        return tuple(value for field, value in vars(self).items() if field != "root")


@dataclass(frozen=True)
class FormalPreflightRequest:
    config: Path
    manifest_dir: Path
    alignment_cache: Path
    benchmark_dir: Path
    p2_init: Path
    output_root: Path
    expected_git_commit: str
    minimum_free_bytes: int


@dataclass(frozen=True)
class FormalPreflightReport:
    schema_version: int
    git_commit: str
    manifest_sha256: str
    alignment_cache_sha256: str
    human_benchmark_sha256: str
    p2_init_sha256: str
    train_record_count: int
    gpu_names: tuple[str, str]
    free_bytes: int
    passed: bool


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _read_strict_json(path: Path) -> object:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"JSON artifact is missing or unsafe: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON artifact is malformed: {source}") from exc


def sha256_file(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"artifact is missing or unsafe: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"artifact cannot be read: {source}") from exc
    return digest.hexdigest()

def count_jsonl_rows(path: Path, *, maximum_rows: int = 13_998) -> int:
    if type(maximum_rows) is not int or maximum_rows < 0:
        raise ValueError("maximum JSONL row count must be a non-negative integer")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"JSONL artifact is missing or unsafe: {source}")
    row_count = 0
    try:
        with source.open("rb") as stream:
            for row_count, _line in enumerate(stream, start=1):
                if row_count > maximum_rows:
                    return row_count
    except OSError as exc:
        raise ValueError(f"JSONL artifact cannot be read: {source}") from exc
    return row_count


def require_alignment_summary(
    alignment_cache: Path,
    *,
    manifest_sha256: str,
    alignment_sha256: str,
) -> None:
    summary = _read_strict_json(Path(alignment_cache) / "summary.json")
    if not isinstance(summary, dict) or set(summary) != _ALIGNMENT_SUMMARY_FIELDS:
        raise ValueError("formal alignment summary schema does not match")
    integer_fields = (
        "seed",
        "job_count",
        "fallback_count",
        "center_count",
        "worker_count",
        "opencv_threads_per_worker",
    )
    if (
        type(summary["schema_version"]) is not int
        or summary["schema_version"] != 1
        or summary["manifest_sha256"] != manifest_sha256
        or summary["alignment_cache_sha256"] != alignment_sha256
        or any(
            type(summary[name]) is not int or summary[name] < 0
            for name in integer_fields
        )
        or summary["opencv_threads_per_worker"] != 1
        or type(summary["center_decode_reuse"]) is not bool
        or not summary["center_decode_reuse"]
        or summary["cache_write_mode"] != "single_bulk_index_publication"
        or not isinstance(summary["offsets"], list)
        or any(type(offset) is not int for offset in summary["offsets"])
        or not isinstance(summary["fallback_reasons"], dict)
        or any(
            not isinstance(reason, str)
            or type(count) is not int
            or count < 0
            for reason, count in summary["fallback_reasons"].items()
        )
        or isinstance(summary["fallback_fraction"], bool)
        or not isinstance(summary["fallback_fraction"], (int, float))
        or not math.isfinite(float(summary["fallback_fraction"]))
        or not 0.0 <= float(summary["fallback_fraction"]) <= 1.0
    ):
        raise ValueError("formal alignment summary contract does not match")


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute_without_symlink_resolution(Path(path))
    candidates = (*reversed(absolute.parents), absolute)
    for candidate in candidates:
        try:
            if candidate.is_symlink():
                raise ValueError(
                    f"formal output path cannot contain a symlink: {candidate}"
                )
        except OSError as exc:
            raise ValueError(
                f"formal output path cannot be inspected: {candidate}"
            ) from exc


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = Path(left).resolve(strict=False)
    right_resolved = Path(right).resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _validate_formal_destination(request: FormalPreflightRequest) -> None:
    root = Path(request.output_root)
    _reject_symlink_components(root)
    if root.exists() or root.is_symlink():
        raise ValueError("formal output root must not already exist")
    for label, input_path in (
        ("config", request.config),
        ("manifest", request.manifest_dir),
        ("alignment cache", request.alignment_cache),
        ("human benchmark", request.benchmark_dir),
        ("P2 initialization", request.p2_init),
    ):
        if _paths_overlap(root, Path(input_path)):
            raise ValueError(f"formal output root overlaps the {label} input")
    layout = FormalExperimentLayout.from_root(root)
    resolved = tuple(path.resolve(strict=False) for path in layout.artifact_directories())
    if len(set(resolved)) != len(resolved) or any(
        path.parent != root.resolve(strict=False) for path in resolved
    ):
        raise ValueError("formal artifact directories must be non-overlapping children")


def _require_config(path: Path) -> None:
    config = Path(path)
    if config.is_symlink() or not config.is_file():
        raise ValueError("formal config must be a regular file")


def _run_probe(command: tuple[str, ...], *, label: str) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"formal preflight could not probe {label}") from exc
    return completed.stdout


def probe_git() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[3]
    commit = _run_probe(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        label="Git commit",
    ).strip()
    status = _run_probe(
        (
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ),
        label="Git status",
    )
    if not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("formal preflight received an invalid Git commit")
    return commit, bool(status.strip())


def probe_gpus() -> Mapping[str, object]:
    names = tuple(
        line.strip()
        for line in _run_probe(
            (
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ),
            label="GPU devices",
        ).splitlines()
        if line.strip()
    )
    process_lines = tuple(
        line.strip()
        for line in _run_probe(
            (
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ),
            label="GPU compute processes",
        ).splitlines()
        if line.strip()
    )
    try:
        compute_pids = tuple(int(line) for line in process_lines)
    except ValueError as exc:
        raise ValueError("formal preflight received malformed GPU process data") from exc
    if any(pid <= 0 for pid in compute_pids):
        raise ValueError("formal preflight received malformed GPU process data")
    return {"devices": names, "compute_pids": compute_pids}


def probe_free_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(Path(path)).free)
    except OSError as exc:
        raise ValueError("formal preflight could not probe output disk") from exc


def preflight_formal_experiment(
    request: FormalPreflightRequest,
    *,
    git_probe: Callable[[], tuple[str, bool]] = probe_git,
    gpu_probe: Callable[[], Mapping[str, object]] = probe_gpus,
    disk_probe: Callable[[Path], int] = probe_free_bytes,
) -> FormalPreflightReport:
    if not isinstance(request, FormalPreflightRequest):
        raise ValueError("formal preflight request has an invalid type")
    _validate_formal_destination(request)
    if (
        not isinstance(request.expected_git_commit, str)
        or not _GIT_COMMIT_PATTERN.fullmatch(request.expected_git_commit)
    ):
        raise ValueError("formal expected Git commit must be a lowercase SHA-1")
    if (
        type(request.minimum_free_bytes) is not int
        or request.minimum_free_bytes < _MINIMUM_FORMAL_FREE_BYTES
    ):
        raise ValueError("formal minimum free bytes must be at least 100 GiB")

    commit, dirty = git_probe()
    if dirty or commit != request.expected_git_commit:
        raise ValueError("formal preflight requires the exact clean Git commit")

    _require_config(request.config)
    benchmark = load_human_benchmark(request.benchmark_dir)
    benchmark_sha = human_benchmark_fingerprint(request.benchmark_dir)
    if (
        len(benchmark.frames),
        benchmark.annotation_count,
        len(benchmark.truths),
        len(benchmark.ignores),
        benchmark_sha,
    ) != (873, 78_335, 53_735, 334, APPROVED_HUMAN_SHA256):
        raise ValueError("formal human benchmark contract does not match")

    p2_state, p2_provenance = load_frozen_p2_initialization(request.p2_init)
    if (
        len(p2_state) != 859
        or p2_provenance.get("loaded_count") != 427
        or p2_provenance.get("source_weights_sha256")
        != APPROVED_UNIVERSAL_SHA256
        or sha256_file(request.p2_init) != APPROVED_P2_SHA256
    ):
        raise ValueError("formal P2 initialization contract does not match")

    manifest_sha = manifest_fingerprint(request.manifest_dir)
    train_count = count_jsonl_rows(request.manifest_dir / "train.jsonl")
    if train_count != 13_998:
        raise ValueError("formal train manifest must contain exactly 13998 rows")
    snapshot = AlignmentCache(request.alignment_cache).snapshot()
    require_alignment_summary(
        request.alignment_cache,
        manifest_sha256=manifest_sha,
        alignment_sha256=snapshot.fingerprint,
    )

    gpu = gpu_probe()
    if not isinstance(gpu, Mapping):
        raise ValueError("formal GPU probe returned an invalid result")
    devices = gpu.get("devices", ())
    compute_processes = gpu.get("compute_pids", ())
    if isinstance(devices, (str, bytes)) or isinstance(
        compute_processes, (str, bytes)
    ):
        raise ValueError("formal GPU probe returned an invalid result")
    try:
        gpu_names = tuple(devices)
        compute_pids = tuple(compute_processes)
    except TypeError as exc:
        raise ValueError("formal GPU probe returned an invalid result") from exc
    if gpu_names != ("NVIDIA RTX A6000", "NVIDIA RTX A6000"):
        raise ValueError("formal preflight requires exactly two RTX A6000 GPUs")
    if compute_pids:
        raise ValueError("formal preflight found GPU busy")

    layout = FormalExperimentLayout.from_root(request.output_root)
    free_bytes = disk_probe(layout.root.parent)
    if type(free_bytes) is not int or free_bytes < request.minimum_free_bytes:
        raise ValueError("formal output disk has insufficient free bytes")
    return FormalPreflightReport(
        schema_version=1,
        git_commit=commit,
        manifest_sha256=manifest_sha,
        alignment_cache_sha256=snapshot.fingerprint,
        human_benchmark_sha256=benchmark_sha,
        p2_init_sha256=APPROVED_P2_SHA256,
        train_record_count=train_count,
        gpu_names=(gpu_names[0], gpu_names[1]),
        free_bytes=free_bytes,
        passed=True,
    )
