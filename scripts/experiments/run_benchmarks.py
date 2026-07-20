#!/usr/bin/env python3
"""Run strict, repeat-paired GraphAlgorithms overhead benchmarks.

One CSV row represents one executable invocation. Successful rows contain both
the historical rank-maximum timings, dedicated graph-kernel intervals, and the
rank-average timings emitted by the two-GPU binaries. Detection outcomes remain
in raw logs and are intentionally
outside this performance-only experiment layer.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
DEFAULT_METADATA = DATASET_DIR / "metadata.csv"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "experiments"

RESULT_SCHEMA_VERSION = "4"
FIXED_MPI_RANKS = 2
PARSE_ERROR_RETURNCODE = -997
TIMEOUT_RETURNCODE = -998
LAUNCH_ERROR_RETURNCODE = -996

ALGORITHMS = ("bfs", "cc", "kcore", "pagerank")
METHODS = ("basic", "tolerance_queue", "multigpu_basic", "multigpu")
MULTIGPU_METHODS = {"multigpu_basic", "multigpu"}
TOLERANCE_METHODS = {"tolerance_queue", "multigpu"}

EXECUTABLES: Dict[Tuple[str, str], str] = {
    ("bfs", "basic"): "build/bin/bfs/bfs",
    ("bfs", "tolerance_queue"): "build/bin/bfs/bfs_queue",
    ("bfs", "multigpu_basic"): "build/bin/bfs/bfs_multiGPU_basic",
    ("bfs", "multigpu"): "build/bin/bfs/bfs_multiGPU",
    ("cc", "basic"): "build/bin/cc/cc",
    ("cc", "tolerance_queue"): "build/bin/cc/cc_queue",
    ("cc", "multigpu_basic"): "build/bin/cc/cc_multiGPU_basic",
    ("cc", "multigpu"): "build/bin/cc/cc_multiGPU",
    ("kcore", "basic"): "build/bin/kcore/kcore",
    ("kcore", "tolerance_queue"): "build/bin/kcore/kcore_queue",
    ("kcore", "multigpu_basic"): "build/bin/kcore/kcore_multiGPU_basic",
    ("kcore", "multigpu"): "build/bin/kcore/kcore_multiGPU",
    ("pagerank", "basic"): "build/bin/pagerank/pagerank",
    ("pagerank", "tolerance_queue"): "build/bin/pagerank/pagerank_queue",
    ("pagerank", "multigpu_basic"): "build/bin/pagerank/pagerank_multiGPU_basic",
    ("pagerank", "multigpu"): "build/bin/pagerank/pagerank_multiGPU",
}

MAX_TIMING_FIELDS = (
    "gpu_main_ms",
    "graph_kernel_ms",
    "cpu_check_tail_ms",
    "nccl_exchange_ms",
    "mpi_sync_ms",
    "communication_ms",
)
PHASE_TIMING_FIELDS = (
    "main_loop_ms",
    "gpu_compute_ms",
    "cpu_check_drain_ms",
    "postcheck_total_ms",
    "postcheck_mpi_ms",
)
RANK_TIMING_FIELDS = (
    "rank_gpu_main_avg_ms",
    "rank_cpu_check_tail_avg_ms",
    "rank_nccl_exchange_avg_ms",
    "rank_mpi_sync_avg_ms",
    "rank_communication_avg_ms",
    "rank_algorithm_total_avg_ms",
    "rank_algorithm_total_max_ms",
)
RANK_PHASE_TIMING_FIELDS = (
    "rank_main_loop_avg_ms",
    "rank_gpu_compute_avg_ms",
    "rank_graph_kernel_avg_ms",
    "rank_cpu_check_drain_avg_ms",
    "rank_postcheck_total_avg_ms",
    "rank_postcheck_mpi_avg_ms",
)
DERIVED_TIMING_FIELDS = (
    "algorithm_total_ms",
    "rank_imbalance_pct",
    "gpu_time_per_iter_ms",
    "graph_kernel_per_iter_ms",
    "algorithm_total_per_iter_ms",
    "wall_time_per_iter_ms",
    "nominal_gpu_mteps",
    "nominal_algorithm_mteps",
    "nominal_wall_mteps",
)

RESULT_FIELDS = [
    "schema_version", "run_id", "dataset", "algorithm", "method", "repeat",
    "method_order_index", "gpu_count", "mpi_ranks", "rank_count",
    "timing_scope", "timing_aggregation", "returncode", "process_returncode",
    "run_status", "valid", "failure_reason", "started_at_utc",
    "finished_at_utc", "hostname", "slurm_job_id", "slurm_array_job_id",
    "slurm_array_task_id", "cuda_visible_devices", "nodes", "edges",
    "valid_edges", "file_size_bytes", "bfs_src", "k", "alpha", "beta",
    "threshold", "gpu_time_ms", "wall_time_ms", *MAX_TIMING_FIELDS,
    *PHASE_TIMING_FIELDS, *RANK_TIMING_FIELDS, *RANK_PHASE_TIMING_FIELDS,
    *DERIVED_TIMING_FIELDS, "iterations", "benchmark_timing_records",
    "benchmark_iteration_records", "command", "stdout_log", "stderr_log",
]
RESULT_ID_FIELDS = ("dataset", "algorithm", "repeat", "method")

TIMING_PREFIX = "BENCHMARK_TIMING"
ITERATION_PREFIX = "BENCHMARK_ITERATIONS"
KEY_VALUE_TOKEN_RE = re.compile(
    r"([a-z][a-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


class BenchmarkOutputError(ValueError):
    """Raised when a successful process lacks a complete benchmark record."""


@dataclass(frozen=True)
class ProcessResult:
    process_returncode: Optional[int]
    stdout: str
    stderr: str
    wall_time_ms: float
    status: str
    failure_reason: str
    started_at_utc: str
    finished_at_utc: str


@dataclass(frozen=True)
class ParsedBenchmark:
    timing: Dict[str, float]
    iterations: int
    timing_records: int
    iteration_records: int


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def discover_datasets() -> List[str]:
    """Return all MatrixMarket dataset ids in deterministic order."""
    return sorted(path.stem for path in DATASET_DIR.glob("*.mtx"))


def resolve_from_root(path: Path) -> Path:
    """Interpret relative CLI paths relative to the repository root."""
    return path if path.is_absolute() else ROOT / path


def ensure_unique(values: Sequence[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")


def load_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    """Load and validate metadata generated by select_source_nodes.py."""
    path = resolve_from_root(path)
    if not path.exists():
        raise FileNotFoundError(
            f"benchmark metadata not found: {path}\n"
            "Run: uv run python scripts/utils/select_source_nodes.py --top-k 10"
        )

    result: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "dataset", "nodes", "edges", "bfs_source", "file_size_bytes",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            dataset = (row.get("dataset") or "").strip()
            if not dataset:
                raise ValueError(f"{path}:{line_number}: empty dataset id")
            if dataset in result:
                raise ValueError(
                    f"{path}:{line_number}: duplicate dataset id {dataset}"
                )
            result[dataset] = dict(row)
    return result


def metadata_int(metadata: Mapping[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid metadata value for "
            f"{metadata.get('dataset', '<unknown>')}: {key}"
        ) from exc


def optional_metadata_int(
    metadata: Mapping[str, str],
    key: str,
    fallback: int,
) -> int:
    value = (metadata.get(key) or "").strip()
    return int(value) if value else fallback


def validate_inputs(
    datasets: Sequence[str],
    algorithms: Sequence[str],
    methods: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
    mpi_ranks: int,
) -> None:
    """Fail before a long batch if any input or executable is inconsistent."""
    errors: List[str] = []
    ensure_unique(list(datasets), "datasets")
    ensure_unique(list(algorithms), "algorithms")
    ensure_unique(list(methods), "methods")

    if mpi_ranks != FIXED_MPI_RANKS:
        errors.append(
            f"this experiment is fixed to {FIXED_MPI_RANKS} MPI ranks; "
            f"received {mpi_ranks}"
        )

    for dataset in datasets:
        path = DATASET_DIR / f"{dataset}.mtx"
        if not path.is_file():
            errors.append(f"missing dataset: {path}")
            continue
        row = metadata.get(dataset)
        if row is None:
            errors.append(f"missing metadata row: {dataset}")
            continue
        try:
            nodes = metadata_int(row, "nodes")
            edges = metadata_int(row, "edges")
            file_size_bytes = metadata_int(row, "file_size_bytes")
            source = metadata_int(row, "bfs_source")
            if nodes <= 0 or edges <= 0 or file_size_bytes <= 0:
                errors.append(f"non-positive metadata for dataset: {dataset}")
            if "bfs" in algorithms and not 0 <= source < nodes:
                errors.append(
                    f"invalid BFS source for {dataset}: source={source}, "
                    f"nodes={nodes}"
                )
        except ValueError as exc:
            errors.append(str(exc))

    for algorithm in algorithms:
        for method in methods:
            executable = ROOT / EXECUTABLES[(algorithm, method)]
            if not executable.is_file() or not os.access(executable, os.X_OK):
                errors.append(f"missing/non-executable binary: {executable}")

    if MULTIGPU_METHODS.intersection(methods) and shutil.which("mpirun") is None:
        errors.append("mpirun was not found in PATH")

    if errors:
        formatted = "\n".join(f"  - {item}" for item in errors)
        raise RuntimeError(f"benchmark input validation failed:\n{formatted}")


def build_command(
    algorithm: str,
    method: str,
    dataset: str,
    dataset_metadata: Mapping[str, str],
    mpi_ranks: int,
    alpha: float,
    beta: float,
    threshold: float,
    k_value: int,
    extra_mpirun_args: Sequence[str],
) -> List[str]:
    """Build one single-GPU command or one fixed two-rank MPI command."""
    executable = str(ROOT / EXECUTABLES[(algorithm, method)])
    command: List[str] = []
    if method in MULTIGPU_METHODS:
        command.extend(["mpirun", "-np", str(mpi_ranks)])
        command.extend(extra_mpirun_args)
    command.extend([executable, dataset])

    if algorithm == "bfs":
        command.extend(
            ["-s", str(metadata_int(dataset_metadata, "bfs_source"))]
        )
    if algorithm == "kcore":
        command.extend(["-k", str(k_value)])
    if method in TOLERANCE_METHODS:
        command.extend(
            [
                "-a", str(alpha),
                "-b", str(beta),
                "-t", str(threshold),
            ]
        )

    # Disable the full CPU oracle; this experiment measures runtime overhead.
    command.append("-n")
    return command


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_command(command: Sequence[str], timeout: int) -> ProcessResult:
    """Run one command and always return an explicit process-level outcome."""
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    started_at = utc_now()
    start = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            process_returncode=None,
            stdout=_as_text(exc.stdout),
            stderr=_join_failure_stderr(
                _as_text(exc.stderr), f"TIMEOUT after {timeout}s"
            ),
            wall_time_ms=(time.perf_counter() - start) * 1000.0,
            status="timeout",
            failure_reason=f"timeout after {timeout}s",
            started_at_utc=started_at,
            finished_at_utc=utc_now(),
        )
    except OSError as exc:
        return ProcessResult(
            process_returncode=None,
            stdout="",
            stderr=f"LAUNCH ERROR: {exc}\n",
            wall_time_ms=(time.perf_counter() - start) * 1000.0,
            status="launch_error",
            failure_reason=str(exc),
            started_at_utc=started_at,
            finished_at_utc=utc_now(),
        )

    status = "process_error" if process.returncode != 0 else "process_success"
    reason = (
        f"process exited with returncode {process.returncode}"
        if process.returncode != 0
        else ""
    )
    return ProcessResult(
        process_returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
        wall_time_ms=(time.perf_counter() - start) * 1000.0,
        status=status,
        failure_reason=reason,
        started_at_utc=started_at,
        finished_at_utc=utc_now(),
    )


def _join_failure_stderr(stderr: str, message: str) -> str:
    result = stderr
    if result and not result.endswith("\n"):
        result += "\n"
    return result + message + "\n"


def machine_record_lines(stdout: str, prefix: str) -> List[str]:
    """Return payloads for exact machine-record prefixes."""
    payloads: List[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == prefix:
            payloads.append("")
        elif stripped.startswith(prefix + " "):
            payloads.append(stripped[len(prefix) + 1 :])
    return payloads


def parse_numeric_record(payload: str, prefix: str) -> Dict[str, float]:
    """Parse one whitespace-separated key=value numeric record strictly."""
    values: Dict[str, float] = {}
    if not payload:
        raise BenchmarkOutputError(f"{prefix} has an empty payload")
    for token in payload.split():
        match = KEY_VALUE_TOKEN_RE.fullmatch(token)
        if match is None:
            raise BenchmarkOutputError(
                f"{prefix} contains malformed token: {token}"
            )
        key, raw_value = match.groups()
        if key in values:
            raise BenchmarkOutputError(
                f"{prefix} contains duplicate key: {key}"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise BenchmarkOutputError(
                f"{prefix} contains non-finite value: {key}={raw_value}"
            )
        values[key] = value
    return values


def derive_rank_timings(
    timing: Dict[str, float],
    method: str,
) -> Dict[str, float]:
    """Validate disjoint timing phases and derive a rank-mean total."""
    missing_max = [field for field in MAX_TIMING_FIELDS if field not in timing]
    if missing_max:
        raise BenchmarkOutputError(
            "BENCHMARK_TIMING missing fields: " + ", ".join(missing_max)
        )

    if method in MULTIGPU_METHODS:
        required_multi = (
            *PHASE_TIMING_FIELDS,
            *RANK_TIMING_FIELDS,
            *RANK_PHASE_TIMING_FIELDS,
        )
        missing_multi = [
            field for field in required_multi if field not in timing
        ]
        if missing_multi:
            raise BenchmarkOutputError(
                "BENCHMARK_TIMING missing two-rank phase fields: "
                + ", ".join(missing_multi)
            )
    else:
        # A single-GPU record has one CUDA-event main interval and no
        # communication subrange. Retain that exact scope for paired results.
        total = timing["gpu_main_ms"] + timing["cpu_check_tail_ms"]
        timing.update(
            {
                "main_loop_ms": timing["gpu_main_ms"],
                "gpu_compute_ms": timing["gpu_main_ms"],
                "cpu_check_drain_ms": timing["cpu_check_tail_ms"],
                "postcheck_total_ms": 0.0,
                "postcheck_mpi_ms": 0.0,
                "rank_gpu_main_avg_ms": timing["gpu_main_ms"],
                "rank_main_loop_avg_ms": timing["gpu_main_ms"],
                "rank_gpu_compute_avg_ms": timing["gpu_main_ms"],
                "rank_graph_kernel_avg_ms": timing["graph_kernel_ms"],
                "rank_cpu_check_tail_avg_ms": timing[
                    "cpu_check_tail_ms"
                ],
                "rank_cpu_check_drain_avg_ms": timing[
                    "cpu_check_tail_ms"
                ],
                "rank_nccl_exchange_avg_ms": timing["nccl_exchange_ms"],
                "rank_mpi_sync_avg_ms": timing["mpi_sync_ms"],
                "rank_postcheck_total_avg_ms": 0.0,
                "rank_postcheck_mpi_avg_ms": 0.0,
                "rank_communication_avg_ms": timing["communication_ms"],
                "rank_algorithm_total_avg_ms": total,
                "rank_algorithm_total_max_ms": total,
            }
        )

    required = (
        *MAX_TIMING_FIELDS,
        *PHASE_TIMING_FIELDS,
        *RANK_TIMING_FIELDS,
        *RANK_PHASE_TIMING_FIELDS,
    )
    negatives = [field for field in required if timing[field] < 0.0]
    if negatives:
        raise BenchmarkOutputError(
            "BENCHMARK_TIMING contains negative fields: "
            + ", ".join(negatives)
        )

    def assert_close(left: str, right: str, description: str) -> None:
        lhs = timing[left]
        rhs = timing[right]
        tolerance = max(0.002, 1e-5 * max(1.0, abs(lhs), abs(rhs)))
        if abs(lhs - rhs) > tolerance:
            raise BenchmarkOutputError(
                f"{description}: {left}={lhs} differs from {right}={rhs}"
            )

    assert_close(
        "gpu_main_ms", "main_loop_ms",
        "gpu_main_ms compatibility alias is inconsistent",
    )
    assert_close(
        "cpu_check_tail_ms", "cpu_check_drain_ms",
        "cpu_check_tail_ms compatibility alias is inconsistent",
    )
    assert_close(
        "rank_gpu_main_avg_ms", "rank_main_loop_avg_ms",
        "rank GPU-main compatibility alias is inconsistent",
    )
    assert_close(
        "rank_cpu_check_tail_avg_ms", "rank_cpu_check_drain_avg_ms",
        "rank CPU-tail compatibility alias is inconsistent",
    )

    for subset, total, label in (
        ("postcheck_mpi_ms", "postcheck_total_ms", "rank-max post-check"),
        (
            "rank_postcheck_mpi_avg_ms",
            "rank_postcheck_total_avg_ms",
            "rank-mean post-check",
        ),
        ("gpu_compute_ms", "main_loop_ms", "rank-max main loop"),
        (
            "graph_kernel_ms",
            "gpu_compute_ms",
            "rank-max GPU compute",
        ),
        (
            "rank_gpu_compute_avg_ms",
            "rank_main_loop_avg_ms",
            "rank-mean main loop",
        ),
        (
            "rank_graph_kernel_avg_ms",
            "rank_gpu_compute_avg_ms",
            "rank-mean GPU compute",
        ),
    ):
        tolerance = max(
            0.002,
            1e-5 * max(1.0, timing[subset], timing[total]),
        )
        if timing[subset] > timing[total] + tolerance:
            raise BenchmarkOutputError(
                f"{label}: {subset} cannot exceed {total}"
            )

    attributed_main = (
        timing["rank_gpu_compute_avg_ms"]
        + timing["rank_nccl_exchange_avg_ms"]
        + timing["rank_mpi_sync_avg_ms"]
    )
    attributed_tolerance = max(
        0.003,
        1e-5 * max(
            1.0,
            attributed_main,
            timing["rank_main_loop_avg_ms"],
        ),
    )
    if (
        attributed_main
        > timing["rank_main_loop_avg_ms"] + attributed_tolerance
    ):
        raise BenchmarkOutputError(
            "rank-mean CUDA + NCCL + main MPI cannot exceed main loop"
        )

    expected_rank_communication = (
        timing["rank_nccl_exchange_avg_ms"]
        + timing["rank_mpi_sync_avg_ms"]
        + timing["rank_postcheck_mpi_avg_ms"]
    )
    communication_tolerance = max(
        0.003,
        1e-5 * max(
            1.0,
            expected_rank_communication,
            timing["rank_communication_avg_ms"],
        ),
    )
    if (
        abs(
            timing["rank_communication_avg_ms"]
            - expected_rank_communication
        )
        > communication_tolerance
    ):
        raise BenchmarkOutputError(
            "rank_communication_avg_ms must equal NCCL + main MPI "
            "+ post-check MPI"
        )

    expected_rank_total = (
        timing["rank_main_loop_avg_ms"]
        + timing["rank_cpu_check_drain_avg_ms"]
        + timing["rank_postcheck_total_avg_ms"]
    )
    total_tolerance = max(
        0.003,
        1e-5 * max(
            1.0,
            expected_rank_total,
            timing["rank_algorithm_total_avg_ms"],
        ),
    )
    if (
        abs(timing["rank_algorithm_total_avg_ms"] - expected_rank_total)
        > total_tolerance
    ):
        raise BenchmarkOutputError(
            "rank_algorithm_total_avg_ms must equal main loop + "
            "checker drain + post-check total"
        )

    rank_average = timing["rank_algorithm_total_avg_ms"]
    rank_maximum = timing["rank_algorithm_total_max_ms"]
    tolerance = 1e-6 * max(1.0, rank_average, rank_maximum)
    if rank_maximum + tolerance < rank_average:
        raise BenchmarkOutputError(
            "rank_algorithm_total_max_ms is smaller than "
            "rank_algorithm_total_avg_ms"
        )

    timing["algorithm_total_ms"] = rank_average
    timing["rank_imbalance_pct"] = (
        (rank_maximum / rank_average - 1.0) * 100.0
        if rank_average > 0.0
        else 0.0
    )
    return timing


def parse_benchmark_output(stdout: str, method: str) -> ParsedBenchmark:
    """Parse exactly one complete timing record and one iteration record."""
    timing_payloads = machine_record_lines(stdout, TIMING_PREFIX)
    iteration_payloads = machine_record_lines(stdout, ITERATION_PREFIX)
    if len(timing_payloads) != 1:
        raise BenchmarkOutputError(
            f"expected exactly one {TIMING_PREFIX} record, "
            f"found {len(timing_payloads)}"
        )
    if len(iteration_payloads) != 1:
        raise BenchmarkOutputError(
            f"expected exactly one {ITERATION_PREFIX} record, "
            f"found {len(iteration_payloads)}"
        )

    timing = derive_rank_timings(
        parse_numeric_record(timing_payloads[0], TIMING_PREFIX),
        method,
    )
    iteration_values = parse_numeric_record(
        iteration_payloads[0], ITERATION_PREFIX
    )
    if set(iteration_values) != {"iterations"}:
        raise BenchmarkOutputError(
            f"{ITERATION_PREFIX} must contain only iterations=<integer>"
        )
    raw_iterations = iteration_values["iterations"]
    iterations = int(raw_iterations)
    if iterations <= 0 or raw_iterations != iterations:
        raise BenchmarkOutputError(
            f"iterations must be a positive integer, got {raw_iterations}"
        )
    return ParsedBenchmark(
        timing=timing,
        iterations=iterations,
        timing_records=len(timing_payloads),
        iteration_records=len(iteration_payloads),
    )


def per_iteration(
    value_ms: Optional[float],
    iterations: Optional[int],
) -> Optional[float]:
    if value_ms is None or iterations is None or iterations <= 0:
        return None
    return value_ms / iterations


def nominal_mteps(
    edges: int,
    iterations: Optional[int],
    value_ms: Optional[float],
) -> Optional[float]:
    """Return E * iterations / second in millions; not measured TEPS."""
    if (
        edges <= 0
        or iterations is None
        or iterations <= 0
        or value_ms is None
        or value_ms <= 0
    ):
        return None
    return (edges * iterations) / (value_ms * 1000.0)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def display_path(path: Path) -> str:
    """Prefer repository-relative paths, but allow outputs elsewhere."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    datasets: Sequence[str],
    metadata_path: Path,
) -> None:
    config = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "datasets": list(datasets),
        "algorithms": list(args.algorithms),
        "methods": list(args.methods),
        "method_order_policy": (
            "forward on odd repeats; reversed on even repeats"
        ),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "mpi_ranks": args.mpi_ranks,
        "timeout_seconds": args.timeout,
        "k": args.k,
        "alpha": args.alpha,
        "beta": args.beta,
        "threshold": args.threshold,
        "metadata": str(metadata_path),
        "mpirun_args": list(args.mpirun_args),
        "result_fields": RESULT_FIELDS,
    }
    write_json_atomic(path, config)


def base_row(
    *,
    args: argparse.Namespace,
    dataset: str,
    dataset_metadata: Mapping[str, str],
    algorithm: str,
    method: str,
    repeat: int,
    method_order_index: int,
    command: Sequence[str],
    stdout_log: Path,
    stderr_log: Path,
    process: ProcessResult,
) -> Dict[str, object]:
    """Build a complete schema row whose metrics are initially absent."""
    is_multi = method in MULTIGPU_METHODS
    edges = metadata_int(dataset_metadata, "edges")
    row: Dict[str, object] = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": f"{dataset}.{algorithm}.{method}.r{repeat}",
            "dataset": dataset,
            "algorithm": algorithm,
            "method": method,
            "repeat": repeat,
            "method_order_index": method_order_index,
            "gpu_count": FIXED_MPI_RANKS if is_multi else 1,
            "mpi_ranks": FIXED_MPI_RANKS if is_multi else 1,
            "rank_count": FIXED_MPI_RANKS if is_multi else 1,
            "timing_scope": "rank_aggregate" if is_multi else "single_rank",
            "timing_aggregation": (
                "rank_max_and_mean" if is_multi else "identity"
            ),
            "process_returncode": (
                process.process_returncode
                if process.process_returncode is not None
                else ""
            ),
            "run_status": process.status,
            "valid": 0,
            "failure_reason": process.failure_reason,
            "started_at_utc": process.started_at_utc,
            "finished_at_utc": process.finished_at_utc,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_job_id": os.environ.get(
                "SLURM_ARRAY_JOB_ID", ""
            ),
            "slurm_array_task_id": os.environ.get(
                "SLURM_ARRAY_TASK_ID", ""
            ),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", ""
            ),
            "nodes": metadata_int(dataset_metadata, "nodes"),
            "edges": edges,
            "valid_edges": optional_metadata_int(
                dataset_metadata, "valid_edges", edges
            ),
            "file_size_bytes": metadata_int(
                dataset_metadata, "file_size_bytes"
            ),
            "bfs_src": (
                metadata_int(dataset_metadata, "bfs_source")
                if algorithm == "bfs"
                else ""
            ),
            "k": args.k if algorithm == "kcore" else "",
            "alpha": args.alpha,
            "beta": args.beta,
            "threshold": args.threshold,
            "wall_time_ms": process.wall_time_ms,
            "benchmark_timing_records": len(
                machine_record_lines(process.stdout, TIMING_PREFIX)
            ),
            "benchmark_iteration_records": len(
                machine_record_lines(process.stdout, ITERATION_PREFIX)
            ),
            "command": shlex.join(command),
            "stdout_log": display_path(stdout_log),
            "stderr_log": display_path(stderr_log),
        }
    )
    return row


def finalize_row(
    row: Dict[str, object],
    method: str,
    process: ProcessResult,
) -> None:
    """Assign an effective status and parsed metrics without silent zeros."""
    if process.status == "timeout":
        row["returncode"] = TIMEOUT_RETURNCODE
        return
    if process.status == "launch_error":
        row["returncode"] = LAUNCH_ERROR_RETURNCODE
        return
    if process.process_returncode != 0:
        row["returncode"] = process.process_returncode
        return

    try:
        parsed = parse_benchmark_output(process.stdout, method)
    except BenchmarkOutputError as exc:
        row["returncode"] = PARSE_ERROR_RETURNCODE
        row["run_status"] = "parse_error"
        row["failure_reason"] = str(exc)
        return

    timing = parsed.timing
    iterations = parsed.iterations
    gpu_main_ms = timing["gpu_main_ms"]
    algorithm_total_ms = timing["algorithm_total_ms"]
    wall_time_ms = float(row["wall_time_ms"])
    edges = int(row["edges"])

    row.update(timing)
    row.update(
        {
            # Backward-compatible name: this remains rank-max GPU main time.
            "gpu_time_ms": gpu_main_ms,
            "iterations": iterations,
            "gpu_time_per_iter_ms": per_iteration(
                gpu_main_ms, iterations
            ),
            "graph_kernel_per_iter_ms": per_iteration(
                timing["graph_kernel_ms"], iterations
            ),
            "algorithm_total_per_iter_ms": per_iteration(
                algorithm_total_ms, iterations
            ),
            "wall_time_per_iter_ms": per_iteration(
                wall_time_ms, iterations
            ),
            "nominal_gpu_mteps": nominal_mteps(
                edges, iterations, gpu_main_ms
            ),
            "nominal_algorithm_mteps": nominal_mteps(
                edges, iterations, algorithm_total_ms
            ),
            "nominal_wall_mteps": nominal_mteps(
                edges, iterations, wall_time_ms
            ),
            "benchmark_timing_records": parsed.timing_records,
            "benchmark_iteration_records": parsed.iteration_records,
            "returncode": 0,
            "run_status": "success",
            "valid": 1,
            "failure_reason": "",
        }
    )


def run_report_payload(
    *,
    status: str,
    results_csv: Path,
    expected_rows: int,
    written_rows: int,
    successful_rows: int,
    failed_run_ids: Sequence[str],
) -> Dict[str, object]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "expected_rows": expected_rows,
        "written_rows": written_rows,
        "missing_result_rows": max(0, expected_rows - written_rows),
        "successful_rows": successful_rows,
        "failed_rows": len(failed_run_ids),
        "failed_invocations": len(failed_run_ids),
        "failed_run_ids": list(failed_run_ids),
        "results_csv": display_path(results_csv),
        "updated_at": utc_now(),
    }


def run_warmups(
    *,
    args: argparse.Namespace,
    dataset: str,
    dataset_metadata: Mapping[str, str],
    algorithm: str,
    log_dir: Path,
) -> List[str]:
    """Run unmeasured warmups, retain logs, and return failed warmup ids."""
    failed_ids: List[str] = []
    if args.warmup == 0:
        return failed_ids

    warmup_log_dir = log_dir / "warmup"
    warmup_log_dir.mkdir(parents=True, exist_ok=True)
    total = args.warmup * len(args.methods)
    ordinal = 0
    for warmup_round in range(1, args.warmup + 1):
        for method in args.methods:
            ordinal += 1
            command = build_command(
                algorithm=algorithm,
                method=method,
                dataset=dataset,
                dataset_metadata=dataset_metadata,
                mpi_ranks=args.mpi_ranks,
                alpha=args.alpha,
                beta=args.beta,
                threshold=args.threshold,
                k_value=args.k,
                extra_mpirun_args=args.mpirun_args,
            )
            warmup_id = (
                f"{dataset}.{algorithm}.{method}.warmup{warmup_round}"
            )
            print(
                f"[warmup {ordinal}/{total}] {warmup_id}: "
                f"{shlex.join(command)}",
                flush=True,
            )
            process = run_command(command, args.timeout)
            stdout_log = warmup_log_dir / f"{warmup_id}.stdout.log"
            stderr_log = warmup_log_dir / f"{warmup_id}.stderr.log"
            write_text(stdout_log, process.stdout)
            write_text(stderr_log, process.stderr)

            failure_reason = process.failure_reason
            if process.status == "process_success":
                try:
                    parse_benchmark_output(process.stdout, method)
                except BenchmarkOutputError as exc:
                    failure_reason = str(exc)
            if process.status != "process_success" or failure_reason:
                failed_ids.append(warmup_id)
                print(
                    f"  -> warmup failed: {failure_reason}; "
                    f"see {stderr_log} and {stdout_log}",
                    file=sys.stderr,
                )
    return failed_ids


def run_benchmarks(args: argparse.Namespace) -> Tuple[Path, int]:
    datasets = args.datasets or discover_datasets()
    metadata_path = resolve_from_root(args.metadata)
    metadata = load_metadata(metadata_path)
    validate_inputs(
        datasets,
        args.algorithms,
        args.methods,
        metadata,
        args.mpi_ranks,
    )

    out_dir = (
        resolve_from_root(args.out_dir)
        if args.out_dir is not None
        else DEFAULT_OUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(
        out_dir / "run_config.json", args, datasets, metadata_path
    )

    results_csv = out_dir / "results.csv"
    run_report = out_dir / "run_report.json"
    expected_rows = (
        len(datasets)
        * len(args.algorithms)
        * len(args.methods)
        * args.repeat
    )
    written_rows = 0
    successful_rows = 0
    failed_run_ids: List[str] = []
    write_json_atomic(
        run_report,
        run_report_payload(
            status="running",
            results_csv=results_csv,
            expected_rows=expected_rows,
            written_rows=0,
            successful_rows=0,
            failed_run_ids=[],
        ),
    )

    with results_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        for dataset in datasets:
            dataset_metadata = metadata[dataset]
            for algorithm in args.algorithms:
                warmup_failures = run_warmups(
                    args=args,
                    dataset=dataset,
                    dataset_metadata=dataset_metadata,
                    algorithm=algorithm,
                    log_dir=log_dir,
                )
                if warmup_failures:
                    failed_run_ids.extend(warmup_failures)
                    write_json_atomic(
                        run_report,
                        run_report_payload(
                            status="warmup_failed",
                            results_csv=results_csv,
                            expected_rows=expected_rows,
                            written_rows=written_rows,
                            successful_rows=successful_rows,
                            failed_run_ids=failed_run_ids,
                        ),
                    )
                    # Measurements after a failed warmup are not comparable.
                    continue
                for repeat in range(1, args.repeat + 1):
                    # Keep both comparison pairs adjacent and alternate which
                    # member runs first to reduce systematic order bias.
                    repeat_methods = (
                        list(args.methods)
                        if repeat % 2 == 1
                        else list(reversed(args.methods))
                    )
                    for method_order_index, method in enumerate(
                        repeat_methods, start=1
                    ):
                        command = build_command(
                            algorithm=algorithm,
                            method=method,
                            dataset=dataset,
                            dataset_metadata=dataset_metadata,
                            mpi_ranks=args.mpi_ranks,
                            alpha=args.alpha,
                            beta=args.beta,
                            threshold=args.threshold,
                            k_value=args.k,
                            extra_mpirun_args=args.mpirun_args,
                        )
                        run_id = (
                            f"{dataset}.{algorithm}.{method}.r{repeat}"
                        )
                        ordinal = written_rows + 1
                        print(
                            f"[{ordinal}/{expected_rows}] {run_id}: "
                            f"{shlex.join(command)}",
                            flush=True,
                        )

                        process = run_command(command, args.timeout)
                        stdout_log = log_dir / f"{run_id}.stdout.log"
                        stderr_log = log_dir / f"{run_id}.stderr.log"
                        write_text(stdout_log, process.stdout)
                        write_text(stderr_log, process.stderr)

                        row = base_row(
                            args=args,
                            dataset=dataset,
                            dataset_metadata=dataset_metadata,
                            algorithm=algorithm,
                            method=method,
                            repeat=repeat,
                            method_order_index=method_order_index,
                            command=command,
                            stdout_log=stdout_log,
                            stderr_log=stderr_log,
                            process=process,
                        )
                        finalize_row(row, method, process)
                        writer.writerow(row)
                        stream.flush()

                        written_rows += 1
                        if row["returncode"] == 0:
                            successful_rows += 1
                        else:
                            failed_run_ids.append(run_id)
                            print(
                                f"  -> {row['run_status']}: "
                                f"{row['failure_reason']}; "
                                f"see {stderr_log} and {stdout_log}",
                                file=sys.stderr,
                            )
                        write_json_atomic(
                            run_report,
                            run_report_payload(
                                status="running",
                                results_csv=results_csv,
                                expected_rows=expected_rows,
                                written_rows=written_rows,
                                successful_rows=successful_rows,
                                failed_run_ids=failed_run_ids,
                            ),
                        )

    if any(".warmup" in run_id for run_id in failed_run_ids):
        final_status = "warmup_failed"
    elif failed_run_ids:
        final_status = "completed_with_failures"
    else:
        final_status = "complete"
    write_json_atomic(
        run_report,
        run_report_payload(
            status=final_status,
            results_csv=results_csv,
            expected_rows=expected_rows,
            written_rows=written_rows,
            successful_rows=successful_rows,
            failed_run_ids=failed_run_ids,
        ),
    )
    return results_csv, len(failed_run_ids)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict basic/checked single-GPU and fixed two-GPU "
            "overhead benchmarks."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help=(
            "dataset ids without dataset/ or .mtx; "
            "default: all dataset/*.mtx"
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--mpi-ranks",
        type=int,
        default=FIXED_MPI_RANKS,
        help="fixed at 2 for this experiment",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--mpirun-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args after mpirun -np 2; this option must be last",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.mpi_ranks != FIXED_MPI_RANKS:
        parser.error("--mpi-ranks is fixed at 2 for this experiment")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        results_csv, failed_rows = run_benchmarks(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"results: {results_csv}")
    if failed_rows:
        print(
            f"error: {failed_rows} benchmark invocation(s) failed; "
            f"see {results_csv.parent / 'run_report.json'}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
