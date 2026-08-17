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
from moving_det.temporal_config import load_temporal_config
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
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORMAL_CONFIG_VALUES = {
    "seed": 20260806,
    "fps": 30,
    "tile_size": 1024,
    "tile_overlap": 256,
    "effective_batch_size": 16,
    "pilot_epochs": 80,
    "early_stopping_patience": 15,
    "mg_offsets": (-4, -2, 0, 2, 4),
    "optimizer": "AdamW",
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "warmup_epochs": 3,
    "nms_iou": 0.5,
}
_FORMAL_SPLIT_SEQUENCES = (
    (
        "train",
        (
            ("site19", "DJI_20240919154443_0005_V"),
            ("site19", "DJI_20240919162906_0003_V"),
            ("site22", "DJI_20240719085001_0003_V"),
            ("site22", "DJI_20240719091331_0001_V"),
            ("site22", "DJI_20240719181132_0001_V"),
            ("site22", "DJI_20240719181521_0002_V"),
        ),
    ),
    (
        "validation",
        (
            ("site19", "DJI_20240919150818_0004_V"),
            ("site22", "DJI_20240719085350_0004_V"),
            ("site22", "DJI_20240719171610_0003_V"),
        ),
    ),
    (
        "test",
        (
            ("site19", "DJI_20240919093341_0002_V"),
            ("site22", "DJI_20240719183036_0006_V"),
            ("site22", "DJI_20240719224127_0006_V"),
        ),
    ),
)
_FORMAL_ALIGNMENT_OFFSETS = (-30, -15, -4, -2, 2, 4, 15, 30)
_MANIFEST_CHILDREN = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
)
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
class FormalApprovedInputContract:
    config_relative_path: Path
    manifest_relative_path: Path
    alignment_cache_relative_path: Path
    config_sha256: str
    manifest_sha256: str
    alignment_cache_sha256: str
    p2_init_sha256: str
    split_row_counts: tuple[tuple[str, int], ...]
    split_sequences: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    alignment_offsets: tuple[int, ...]


APPROVED_FORMAL_INPUTS = FormalApprovedInputContract(
    config_relative_path=Path("configs/vrud-temporal-obb.yaml"),
    manifest_relative_path=Path("runs/vrud-pilot/manifest"),
    alignment_cache_relative_path=Path("runs/vrud-pilot/alignment-cache"),
    config_sha256=(
        "0676b10ef913eec20014bd94a9af568ea8e870cdd1a48026ebd5af4345bc0c67"
    ),
    manifest_sha256=(
        "4aee4a44f9dd157f420c6ca1fc2f7b21300bf269a3dd4c1e6e6ad4767bd94044"
    ),
    alignment_cache_sha256=(
        "07e49ef8766d0f1d85c6c368a9cf34bbd57447386f216ca4d73bfb179d91568e"
    ),
    p2_init_sha256=APPROVED_P2_SHA256,
    split_row_counts=(
        ("train", 13_998),
        ("validation", 16_575),
        ("test", 60_900),
    ),
    split_sequences=_FORMAL_SPLIT_SEQUENCES,
    alignment_offsets=_FORMAL_ALIGNMENT_OFFSETS,
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
    config_sha256: str
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
    expected_seed: int,
    expected_offsets: tuple[int, ...],
    manifest_centers: frozenset[tuple[str, str, int]],
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
        or summary["seed"] != expected_seed
        or any(
            type(summary[name]) is not int or summary[name] < 0
            for name in integer_fields
        )
        or summary["opencv_threads_per_worker"] != 1
        or type(summary["center_decode_reuse"]) is not bool
        or not summary["center_decode_reuse"]
        or summary["cache_write_mode"] != "single_bulk_index_publication"
        or summary["offsets"] != list(expected_offsets)
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
    index = _read_strict_json(Path(alignment_cache) / "index.json")
    if (
        not isinstance(index, dict)
        or set(index) != {"schema_version", "entries"}
        or index["schema_version"] != 1
        or not isinstance(index["entries"], dict)
    ):
        raise ValueError("formal alignment coverage index is invalid")
    entries = index["entries"]
    cache_centers: set[tuple[str, str, int]] = set()
    observed_offsets: set[int] = set()
    for entry in entries.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("key"), dict):
            raise ValueError("formal alignment coverage entry is invalid")
        key = entry["key"]
        try:
            center = (key["site"], key["sequence"], key["center_frame"])
            offset = key["support_frame"] - key["center_frame"]
        except (KeyError, TypeError) as exc:
            raise ValueError("formal alignment coverage entry is invalid") from exc
        if center not in manifest_centers or offset not in expected_offsets:
            raise ValueError("formal alignment coverage does not match the manifest")
        cache_centers.add(center)
        observed_offsets.add(offset)
    if (
        not manifest_centers
        or cache_centers != set(manifest_centers)
        or observed_offsets != set(expected_offsets)
        or summary["center_count"] != len(manifest_centers)
        or summary["job_count"] != len(entries)
        or summary["job_count"] <= 0
        or summary["worker_count"] != min(16, len(manifest_centers))
    ):
        raise ValueError("formal alignment coverage contract does not match")


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


def _validated_contract_maps(
    contract: FormalApprovedInputContract,
) -> tuple[
    dict[str, int],
    dict[str, tuple[tuple[str, str], ...]],
]:
    if not isinstance(contract, FormalApprovedInputContract):
        raise ValueError("formal approved input contract has an invalid type")
    counts = dict(contract.split_row_counts)
    sequences = dict(contract.split_sequences)
    if (
        len(counts) != len(contract.split_row_counts)
        or len(sequences) != len(contract.split_sequences)
        or set(counts) != {"train", "validation", "test"}
        or set(sequences) != {"train", "validation", "test"}
        or any(type(count) is not int or count <= 0 for count in counts.values())
        or any(
            not isinstance(value, tuple) or not value
            for value in sequences.values()
        )
        or any(
            not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value)
            for value in (
                contract.config_sha256,
                contract.manifest_sha256,
                contract.alignment_cache_sha256,
                contract.p2_init_sha256,
            )
        )
        or not contract.alignment_offsets
        or any(type(offset) is not int for offset in contract.alignment_offsets)
        or tuple(sorted(set(contract.alignment_offsets)))
        != contract.alignment_offsets
        or 0 in contract.alignment_offsets
    ):
        raise ValueError("formal approved input contract is malformed")
    all_sequences = tuple(
        sequence
        for split_sequences in sequences.values()
        for sequence in split_sequences
    )
    if (
        len(set(all_sequences)) != len(all_sequences)
        or any(
            not isinstance(site, str)
            or not site
            or not isinstance(sequence, str)
            or not sequence
            for site, sequence in all_sequences
        )
    ):
        raise ValueError("formal approved split sequences must be non-overlapping")
    return counts, sequences


def _validated_project_root(project_root: Path | None) -> Path:
    root = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else _absolute_without_symlink_resolution(Path(project_root))
    )
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("formal Git project root must be a directory")
    return root


def _canonical_formal_input(
    supplied: Path,
    *,
    project_root: Path,
    relative_path: Path,
    label: str,
) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"formal canonical {label} path is unsafe")
    expected = _absolute_without_symlink_resolution(project_root / relative)
    actual = _absolute_without_symlink_resolution(Path(supplied))
    _reject_symlink_components(actual)
    if actual != expected:
        raise ValueError(f"formal {label} must use the canonical project path")
    return expected


def _validate_canonical_inputs(
    request: FormalPreflightRequest,
    *,
    project_root: Path,
    contract: FormalApprovedInputContract,
) -> None:
    for supplied, relative, label in (
        (request.config, contract.config_relative_path, "config"),
        (request.manifest_dir, contract.manifest_relative_path, "manifest"),
        (
            request.alignment_cache,
            contract.alignment_cache_relative_path,
            "alignment cache",
        ),
    ):
        _canonical_formal_input(
            supplied,
            project_root=project_root,
            relative_path=relative,
            label=label,
        )


def _validate_formal_config(
    path: Path,
    *,
    approved_sha256: str,
) -> str:
    config_sha256 = sha256_file(path)
    if config_sha256 != approved_sha256:
        raise ValueError("formal config SHA-256 is not approved")
    try:
        config = load_temporal_config(Path(path))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("formal config contract is invalid") from exc
    if any(
        getattr(config, field) != expected
        for field, expected in _FORMAL_CONFIG_VALUES.items()
    ):
        raise ValueError("formal config contract does not match")
    return config_sha256


def _validate_manifest_metadata(manifest_dir: Path) -> None:
    metadata = _read_strict_json(Path(manifest_dir) / "manifest.json")
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"seed", "files"}
        or metadata["seed"] != 20260806
        or not isinstance(metadata["files"], dict)
        or set(metadata["files"]) != set(_MANIFEST_CHILDREN)
    ):
        raise ValueError("formal manifest seed or metadata contract does not match")
    for name in _MANIFEST_CHILDREN:
        declaration = metadata["files"][name]
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"sha256"}
            or declaration["sha256"] != sha256_file(Path(manifest_dir) / name)
        ):
            raise ValueError("formal manifest child fingerprint does not match")


def _validate_manifest_row(
    row: object,
    *,
    split: str,
    allowed_sequences: set[tuple[str, str]],
) -> tuple[
    tuple[str, str, int],
    tuple[object, ...],
    tuple[str, str],
]:
    required = {
        "split",
        "site",
        "sequence",
        "center_frame",
        "tile_xywh",
        "track_keys",
        "source",
    }
    if not isinstance(row, dict) or set(row) != required or row["split"] != split:
        raise ValueError("formal manifest row split contract does not match")
    site = row["site"]
    sequence = row["sequence"]
    center_frame = row["center_frame"]
    tile = row["tile_xywh"]
    if (
        not isinstance(site, str)
        or not isinstance(sequence, str)
        or (site, sequence) not in allowed_sequences
        or type(center_frame) is not int
        or center_frame <= 0
        or not isinstance(tile, list)
        or len(tile) != 4
        or any(type(value) is not int for value in tile)
        or tile[0] < 0
        or tile[1] < 0
        or tile[2] <= 0
        or tile[3] <= 0
        or not isinstance(row["track_keys"], list)
        or not isinstance(row["source"], str)
        or not row["source"]
    ):
        raise ValueError("formal manifest row sequence or geometry is invalid")
    center = (site, sequence, center_frame)
    record = (*center, tuple(tile), row["source"])
    return center, record, (site, sequence)


def _validate_formal_manifest(
    manifest_dir: Path,
    *,
    contract: FormalApprovedInputContract,
    split_counts: Mapping[str, int],
    split_sequences: Mapping[str, tuple[tuple[str, str], ...]],
) -> tuple[str, int, frozenset[tuple[str, str, int]]]:
    manifest_sha256 = manifest_fingerprint(Path(manifest_dir))
    if manifest_sha256 != contract.manifest_sha256:
        raise ValueError("formal manifest fingerprint is not approved")
    _validate_manifest_metadata(manifest_dir)
    center_owners: dict[tuple[str, str, int], str] = {}
    record_owners: dict[tuple[object, ...], str] = {}
    manifest_centers: set[tuple[str, str, int]] = set()
    for split in ("train", "validation", "test"):
        observed_sequences: set[tuple[str, str]] = set()
        row_count = 0
        source = Path(manifest_dir) / f"{split}.jsonl"
        if source.is_symlink() or not source.is_file():
            raise ValueError("formal manifest split is missing or unsafe")
        try:
            with source.open("rb") as stream:
                for row_count, raw_line in enumerate(stream, start=1):
                    if (
                        row_count > split_counts[split]
                        or len(raw_line) > 1024 * 1024
                    ):
                        raise ValueError("formal manifest split row count is invalid")
                    try:
                        row = json.loads(
                            raw_line.decode("utf-8"),
                            object_pairs_hook=_strict_json_object,
                            parse_constant=_reject_json_constant,
                        )
                    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                        raise ValueError("formal manifest row is malformed") from exc
                    center, record, sequence = _validate_manifest_row(
                        row,
                        split=split,
                        allowed_sequences=set(split_sequences[split]),
                    )
                    for identity, owners in (
                        (center, center_owners),
                        (record, record_owners),
                    ):
                        owner = owners.setdefault(identity, split)
                        if owner != split:
                            raise ValueError(
                                "formal manifest records overlap across splits"
                            )
                    manifest_centers.add(center)
                    observed_sequences.add(sequence)
        except OSError as exc:
            raise ValueError("formal manifest split cannot be read") from exc
        if row_count != split_counts[split]:
            raise ValueError("formal manifest split row count is invalid")
        if observed_sequences != set(split_sequences[split]):
            raise ValueError("formal manifest split sequence coverage does not match")
    return manifest_sha256, split_counts["train"], frozenset(manifest_centers)


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


def probe_git(repository: Path | None = None) -> tuple[str, bool]:
    repository = (
        Path(__file__).resolve().parents[3]
        if repository is None
        else Path(repository)
    )
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
    project_root: Path | None = None,
    approved_contract: FormalApprovedInputContract = APPROVED_FORMAL_INPUTS,
    git_probe: Callable[[], tuple[str, bool]] | None = None,
    gpu_probe: Callable[[], Mapping[str, object]] = probe_gpus,
    disk_probe: Callable[[Path], int] = probe_free_bytes,
) -> FormalPreflightReport:
    if not isinstance(request, FormalPreflightRequest):
        raise ValueError("formal preflight request has an invalid type")
    split_counts, split_sequences = _validated_contract_maps(approved_contract)
    resolved_project_root = _validated_project_root(project_root)
    _validate_canonical_inputs(
        request,
        project_root=resolved_project_root,
        contract=approved_contract,
    )
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

    selected_git_probe = (
        (lambda: probe_git(resolved_project_root))
        if git_probe is None
        else git_probe
    )
    commit, dirty = selected_git_probe()
    if dirty or commit != request.expected_git_commit:
        raise ValueError("formal preflight requires the exact clean Git commit")

    config_sha256 = _validate_formal_config(
        request.config,
        approved_sha256=approved_contract.config_sha256,
    )
    manifest_sha, train_count, manifest_centers = _validate_formal_manifest(
        request.manifest_dir,
        contract=approved_contract,
        split_counts=split_counts,
        split_sequences=split_sequences,
    )
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
        or sha256_file(request.p2_init) != approved_contract.p2_init_sha256
    ):
        raise ValueError("formal P2 initialization contract does not match")

    snapshot = AlignmentCache(request.alignment_cache).snapshot()
    if snapshot.fingerprint != approved_contract.alignment_cache_sha256:
        raise ValueError("formal alignment cache fingerprint is not approved")
    require_alignment_summary(
        request.alignment_cache,
        manifest_sha256=manifest_sha,
        alignment_sha256=snapshot.fingerprint,
        expected_seed=20260806,
        expected_offsets=approved_contract.alignment_offsets,
        manifest_centers=manifest_centers,
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
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha,
        alignment_cache_sha256=snapshot.fingerprint,
        human_benchmark_sha256=benchmark_sha,
        p2_init_sha256=approved_contract.p2_init_sha256,
        train_record_count=train_count,
        gpu_names=(gpu_names[0], gpu_names[1]),
        free_bytes=free_bytes,
        passed=True,
    )
