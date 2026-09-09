#!/usr/bin/env python3
"""Shared primitives for reproducible GraphAlgorithms overhead experiments.

The raw layer keeps executable output separate from normalized metrics so every
derived number can be regenerated and audited without rerunning a GPU job.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import sys
import subprocess
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
SCHEMA_VERSION = "6"

ALGORITHMS: Tuple[str, ...] = ("bfs", "cc", "kcore", "pagerank")
METHODS: Tuple[str, ...] = (
    "basic", "tolerance_queue", "multigpu_basic", "multigpu",
)
SINGLE_GPU_METHODS = frozenset(("basic", "tolerance_queue"))
MULTI_GPU_METHODS = frozenset(("multigpu_basic", "multigpu"))
TOLERANCE_METHODS = frozenset(("tolerance_queue", "multigpu"))

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

TIMING_PREFIX = "BENCHMARK_TIMING"
ITERATION_PREFIX = "BENCHMARK_ITERATIONS"
KEY_VALUE_RE = re.compile(
    r"([a-z][a-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

SINGLE_REQUIRED_TIMINGS: Tuple[str, ...] = (
    "gpu_main_ms", "graph_kernel_ms", "cpu_check_tail_ms",
    "nccl_exchange_ms", "mpi_sync_ms", "communication_ms",
)
MULTI_REQUIRED_TIMINGS: Tuple[str, ...] = (
    "gpu_main_ms", "main_loop_ms", "gpu_compute_ms", "graph_kernel_ms",
    "cpu_check_tail_ms", "cpu_check_drain_ms", "nccl_exchange_ms", "mpi_sync_ms",
    "postcheck_total_ms", "postcheck_mpi_ms", "communication_ms",
    "rank_gpu_main_avg_ms", "rank_main_loop_avg_ms", "rank_gpu_compute_avg_ms",
    "rank_graph_kernel_avg_ms", "rank_cpu_check_tail_avg_ms",
    "rank_cpu_check_drain_avg_ms", "rank_nccl_exchange_avg_ms", "rank_mpi_sync_avg_ms",
    "rank_postcheck_total_avg_ms", "rank_postcheck_mpi_avg_ms",
    "rank_communication_avg_ms", "rank_algorithm_total_avg_ms",
    "rank_algorithm_total_max_ms",
)


class BenchmarkOutputError(ValueError):
    """A successful process did not emit one valid benchmark record."""


def utc_now() -> str:
    """Return an explicit timezone-aware timestamp for an invocation record."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_dump_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON so a collector never reads a partial raw record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_load(path: Path) -> Dict[str, Any]:
    """Load an object JSON document."""
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def text_dump_atomic(path: Path, text: str) -> None:
    """Atomically write UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def csv_dump_atomic(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically write a rectangular CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})
    temporary.replace(path)


def csv_value(value: Any) -> Any:
    """Produce a stable CSV cell, keeping nested evidence JSON encoded."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def relative_to_root(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def relative_to(path: Path, base: Path) -> str:
    """Return a run-relative path when possible."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a content digest without loading a graph or binary into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def file_fingerprint(path: Path, *, include_sha256: bool = True) -> Dict[str, Any]:
    """Capture input/executable provenance."""
    stat = path.stat()
    result: Dict[str, Any] = {
        "path": relative_to_root(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)
    return result


def command_output(command: Sequence[str]) -> Dict[str, Any]:
    """Best-effort command provenance; unavailable tools are recorded, not fatal."""
    try:
        completed = subprocess.run(
            list(command), cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        return {"available": False, "command": list(command), "error": str(exc)}
    return {
        "available": True, "command": list(command),
        "returncode": completed.returncode, "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def repository_provenance() -> Dict[str, Any]:
    """Record revision and important local runtime tools."""
    return {
        "recorded_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "git_head": command_output(("git", "rev-parse", "HEAD")),
        "git_status": command_output(("git", "status", "--short")),
        "python": command_output((sys.executable, "--version")),
        "nvcc": command_output(("nvcc", "--version")),
        "nvidia_smi": command_output((
            "nvidia-smi", "--query-gpu=name,driver_version,uuid",
            "--format=csv,noheader",
        )),
        "mpi": command_output(("mpirun", "--version")),
        "selected_environment": {
            name: os.environ.get(name, "")
            for name in (
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NCCL_DEBUG",
                "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "OMP_NUM_THREADS",
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_NODELIST", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
            )
        },
    }


def read_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    """Load metadata and reject duplicate dataset rows."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"dataset", "nodes", "edges", "file_size_bytes", "bfs_source"}
        missing = needed.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing metadata columns {sorted(missing)}")
        result: Dict[str, Dict[str, str]] = {}
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            if not dataset:
                raise ValueError(f"{path}: metadata row has an empty dataset")
            if dataset in result:
                raise ValueError(f"{path}: duplicate metadata dataset {dataset}")
            result[dataset] = dict(row)
    return result


def metadata_int(row: Mapping[str, str], key: str) -> int:
    """Read one required integer metadata field."""
    value = (row.get(key) or "").strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid metadata {key}={value!r}") from exc


def discover_datasets() -> List[str]:
    """Discover MatrixMarket graph IDs in deterministic order."""
    return sorted(path.stem for path in DATASET_DIR.glob("*.mtx"))


def ensure_subset(values: Sequence[str], allowed: Sequence[str], label: str) -> None:
    """Validate a nonempty unique selection."""
    if not values:
        raise ValueError(f"{label} must not be empty")
    invalid = sorted(set(values).difference(allowed))
    duplicates = sorted({item for item in values if values.count(item) > 1})
    if invalid:
        raise ValueError(f"unknown {label}: {', '.join(invalid)}")
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")


def method_gpu_mode(method: str) -> str:
    """Return the resource-class used by a method."""
    if method in SINGLE_GPU_METHODS:
        return "1gpu"
    if method in MULTI_GPU_METHODS:
        return "multigpu"
    raise ValueError(f"unknown method: {method}")


def executable_path(algorithm: str, method: str) -> Path:
    """Resolve one selected benchmark executable."""
    try:
        return ROOT / EXECUTABLES[(algorithm, method)]
    except KeyError as exc:
        raise ValueError(f"unsupported algorithm/method: {algorithm}/{method}") from exc


def validate_experiment_inputs(
    datasets: Sequence[str], algorithms: Sequence[str], methods: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]], mpi_ranks: int,
    launcher_command: str,
) -> None:
    """Fail before scheduling an experiment whose inputs cannot be compared."""
    ensure_subset(list(algorithms), ALGORITHMS, "algorithms")
    ensure_subset(list(methods), METHODS, "methods")
    if mpi_ranks < 1:
        raise ValueError("mpi_ranks must be positive")
    if MULTI_GPU_METHODS.intersection(methods) and mpi_ranks < 2:
        raise ValueError("multi-GPU methods require mpi_ranks >= 2")
    if not datasets:
        raise ValueError("datasets must not be empty")
    duplicate = sorted({item for item in datasets if datasets.count(item) > 1})
    if duplicate:
        raise ValueError(f"duplicate datasets: {', '.join(duplicate)}")
    errors: List[str] = []
    for dataset in datasets:
        graph = DATASET_DIR / f"{dataset}.mtx"
        row = metadata.get(dataset)
        if not graph.is_file():
            errors.append(f"missing graph: {graph}")
            continue
        if row is None:
            errors.append(f"missing metadata row: {dataset}")
            continue
        try:
            nodes = metadata_int(row, "nodes")
            edges = metadata_int(row, "edges")
            if nodes <= 0 or edges <= 0:
                errors.append(f"non-positive nodes/edges for {dataset}")
            if "bfs" in algorithms:
                source = metadata_int(row, "bfs_source")
                if not 0 <= source < nodes:
                    errors.append(f"invalid BFS source for {dataset}: {source}")
        except ValueError as exc:
            errors.append(f"{dataset}: {exc}")
    for algorithm in algorithms:
        for method in methods:
            binary = executable_path(algorithm, method)
            if not binary.is_file() or not os.access(binary, os.X_OK):
                errors.append(f"missing/non-executable benchmark: {binary}")
    if MULTI_GPU_METHODS.intersection(methods) and shutil.which(launcher_command) is None:
        errors.append(f"MPI launcher not found in PATH: {launcher_command}")
    if errors:
        raise ValueError("invalid experiment inputs:\n  - " + "\n  - ".join(errors))


def build_command(
    config: Mapping[str, Any], dataset: str, dataset_metadata: Mapping[str, str],
    algorithm: str, method: str,
) -> List[str]:
    """Build exact argv for one invocation from frozen manifest configuration."""
    command: List[str] = []
    if method in MULTI_GPU_METHODS:
        launcher = config["launcher"]
        command.extend((str(launcher["command"]), str(launcher["ranks_flag"])))
        command.append(str(config["mpi_ranks"]))
        command.extend(str(item) for item in launcher.get("extra_args", []))
    command.extend((str(executable_path(algorithm, method)), dataset))
    parameters = config["parameters"]
    if algorithm == "bfs":
        command.extend(("-s", str(metadata_int(dataset_metadata, "bfs_source"))))
    if algorithm == "kcore":
        command.extend(("-k", str(parameters["k"])))
    if method in TOLERANCE_METHODS:
        command.extend((
            "-a", str(parameters["alpha"]), "-b", str(parameters["beta"]),
            "-t", str(parameters["threshold"]),
        ))
    # Do not include the expensive algorithm-end CPU oracle in overhead timing.
    command.append("-n")
    return command


def execute_command(command: Sequence[str], timeout_s: int) -> Dict[str, Any]:
    """Run one launcher in its own process group and retain its exact outcome.

    A timeout terminates the whole launcher process group, not just the Python
    child that started it.  This prevents orphan MPI ranks from contaminating
    later measurements on the same allocated GPUs.
    """
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    started_at = utc_now()
    start = time.perf_counter()
    try:
        process = subprocess.Popen(
            list(command), cwd=ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            cleanup = "SIGTERM"
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                cleanup = "already_exited"
            except OSError as exc:
                cleanup = f"SIGTERM_error:{exc}"
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                cleanup += "+SIGKILL"
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    cleanup += f"_error:{exc}"
                stdout, stderr = process.communicate()
            return {
                "status": "timeout", "process_returncode": process.returncode,
                "stdout": _as_text(stdout),
                "stderr": _append_message(_as_text(stderr), f"TIMEOUT after {timeout_s}s"),
                "timeout_cleanup": cleanup,
                "process_wall_ms": (time.perf_counter() - start) * 1000.0,
                "started_at_utc": started_at, "finished_at_utc": utc_now(),
                "failure_reason": f"timeout after {timeout_s}s",
            }
    except OSError as exc:
        return {
            "status": "launch_error", "process_returncode": None,
            "stdout": "", "stderr": f"LAUNCH ERROR: {exc}\n",
            "process_wall_ms": (time.perf_counter() - start) * 1000.0,
            "started_at_utc": started_at, "finished_at_utc": utc_now(),
            "failure_reason": str(exc),
        }
    return {
        "status": "process_success" if process.returncode == 0 else "process_error",
        "process_returncode": process.returncode,
        "stdout": stdout, "stderr": stderr,
        "process_wall_ms": (time.perf_counter() - start) * 1000.0,
        "started_at_utc": started_at, "finished_at_utc": utc_now(),
        "failure_reason": "" if process.returncode == 0 else (
            f"process exited with returncode {process.returncode}"
        ),
    }


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _append_message(text: str, message: str) -> str:
    return text + ("" if not text or text.endswith("\n") else "\n") + message + "\n"


def machine_record_lines(stdout: str, prefix: str) -> List[str]:
    """Extract payloads only from exact machine-readable prefixes."""
    result: List[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == prefix:
            result.append("")
        elif stripped.startswith(prefix + " "):
            result.append(stripped[len(prefix) + 1:])
    return result


def parse_numeric_record(payload: str, prefix: str) -> Dict[str, float]:
    """Strictly parse a whitespace-separated numeric key=value record."""
    if not payload:
        raise BenchmarkOutputError(f"{prefix} has an empty payload")
    result: Dict[str, float] = {}
    for token in payload.split():
        match = KEY_VALUE_RE.fullmatch(token)
        if match is None:
            raise BenchmarkOutputError(f"{prefix} contains malformed token {token!r}")
        key, raw_value = match.groups()
        if key in result:
            raise BenchmarkOutputError(f"{prefix} duplicates key {key}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise BenchmarkOutputError(f"{prefix} has non-finite {key}={raw_value}")
        result[key] = value
    return result


def _require_nonnegative(timing: Mapping[str, float], names: Sequence[str]) -> None:
    missing = [name for name in names if name not in timing]
    negative = [name for name in names if name in timing and timing[name] < 0.0]
    if missing or negative:
        parts: List[str] = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if negative:
            parts.append("negative " + ", ".join(negative))
        raise BenchmarkOutputError("BENCHMARK_TIMING " + "; ".join(parts))


def _close(left: float, right: float, label: str) -> None:
    tolerance = max(0.003, 1e-5 * max(1.0, abs(left), abs(right)))
    if abs(left - right) > tolerance:
        raise BenchmarkOutputError(f"{label}: {left:.6f} != {right:.6f}")


def _subset(subset: float, total: float, label: str) -> None:
    tolerance = max(0.003, 1e-5 * max(1.0, subset, total))
    if subset > total + tolerance:
        raise BenchmarkOutputError(f"{label}: subset {subset:.6f} exceeds {total:.6f}")


def normalise_timing(timing: Mapping[str, float], method: str) -> Dict[str, Any]:
    """Normalize timing while preserving direct/derived and max/mean semantics."""
    if method in SINGLE_GPU_METHODS:
        _require_nonnegative(timing, SINGLE_REQUIRED_TIMINGS)
        main = timing["gpu_main_ms"]
        drain = timing["cpu_check_tail_ms"]
        rank = {
            "main_loop_ms": main, "gpu_compute_ms": main,
            "graph_kernel_ms": timing["graph_kernel_ms"],
            "cpu_check_drain_ms": drain, "nccl_exchange_ms": timing["nccl_exchange_ms"],
            "mpi_sync_ms": timing["mpi_sync_ms"], "postcheck_total_ms": 0.0,
            "postcheck_mpi_ms": 0.0, "communication_ms": timing["communication_ms"],
            "algorithm_total_ms": main + drain,
        }
        return {
            "timing_scope": "single_gpu_cuda_interval_plus_drain",
            "direct_fields": sorted(timing),
            "derived_fields": [
                "main_loop_ms", "gpu_compute_ms", "cpu_check_drain_ms",
                "postcheck_total_ms", "postcheck_mpi_ms", "algorithm_total_ms",
            ],
            "normalization": {
                "main_loop_ms": "emitted gpu_main_ms",
                "gpu_compute_ms": "emitted gpu_main_ms",
                "cpu_check_drain_ms": "emitted cpu_check_tail_ms",
                "algorithm_total_ms": "main_loop_ms + cpu_check_drain_ms",
            },
            "rank_max_ms": rank, "rank_mean_ms": dict(rank),
        }

    if method not in MULTI_GPU_METHODS:
        raise BenchmarkOutputError(f"unknown method {method}")
    _require_nonnegative(timing, MULTI_REQUIRED_TIMINGS)
    _close(timing["main_loop_ms"], timing.get("gpu_main_ms", timing["main_loop_ms"]),
           "gpu_main_ms compatibility alias")
    _close(timing["cpu_check_drain_ms"],
           timing.get("cpu_check_tail_ms", timing["cpu_check_drain_ms"]),
           "cpu_check_tail_ms compatibility alias")
    _close(
        timing["rank_main_loop_avg_ms"], timing["rank_gpu_main_avg_ms"],
        "rank_gpu_main_avg_ms compatibility alias",
    )
    _close(
        timing["rank_cpu_check_drain_avg_ms"], timing["rank_cpu_check_tail_avg_ms"],
        "rank_cpu_check_tail_avg_ms compatibility alias",
    )
    _subset(timing["gpu_compute_ms"], timing["main_loop_ms"], "rank-max GPU compute")
    _subset(timing["graph_kernel_ms"], timing["gpu_compute_ms"], "rank-max graph kernel")
    _subset(timing["postcheck_mpi_ms"], timing["postcheck_total_ms"], "rank-max postcheck MPI")
    _subset(timing["rank_gpu_compute_avg_ms"], timing["rank_main_loop_avg_ms"], "rank-mean GPU compute")
    _subset(timing["rank_graph_kernel_avg_ms"], timing["rank_gpu_compute_avg_ms"], "rank-mean graph kernel")
    _subset(timing["rank_postcheck_mpi_avg_ms"], timing["rank_postcheck_total_avg_ms"], "rank-mean postcheck MPI")
    mean_total = (
        timing["rank_main_loop_avg_ms"] + timing["rank_cpu_check_drain_avg_ms"]
        + timing["rank_postcheck_total_avg_ms"]
    )
    _close(mean_total, timing["rank_algorithm_total_avg_ms"], "rank-mean algorithm total")
    max_vs_mean_tolerance = max(
        0.003,
        1e-5 * max(
            1.0, timing["rank_algorithm_total_max_ms"],
            timing["rank_algorithm_total_avg_ms"],
        ),
    )
    if timing["rank_algorithm_total_max_ms"] + max_vs_mean_tolerance < timing[
        "rank_algorithm_total_avg_ms"
    ]:
        raise BenchmarkOutputError("rank-max algorithm total is smaller than rank mean")
    mean_communication = (
        timing["rank_nccl_exchange_avg_ms"] + timing["rank_mpi_sync_avg_ms"]
        + timing["rank_postcheck_mpi_avg_ms"]
    )
    _close(mean_communication, timing["rank_communication_avg_ms"], "rank-mean communication")
    rank_max = {
        "main_loop_ms": timing["main_loop_ms"], "gpu_compute_ms": timing["gpu_compute_ms"],
        "graph_kernel_ms": timing["graph_kernel_ms"],
        "cpu_check_drain_ms": timing["cpu_check_drain_ms"],
        "nccl_exchange_ms": timing["nccl_exchange_ms"], "mpi_sync_ms": timing["mpi_sync_ms"],
        "postcheck_total_ms": timing["postcheck_total_ms"],
        "postcheck_mpi_ms": timing["postcheck_mpi_ms"],
        "communication_ms": timing["communication_ms"],
        "algorithm_total_ms": timing["rank_algorithm_total_max_ms"],
    }
    rank_mean = {
        "main_loop_ms": timing["rank_main_loop_avg_ms"],
        "gpu_compute_ms": timing["rank_gpu_compute_avg_ms"],
        "graph_kernel_ms": timing["rank_graph_kernel_avg_ms"],
        "cpu_check_drain_ms": timing["rank_cpu_check_drain_avg_ms"],
        "nccl_exchange_ms": timing["rank_nccl_exchange_avg_ms"],
        "mpi_sync_ms": timing["rank_mpi_sync_avg_ms"],
        "postcheck_total_ms": timing["rank_postcheck_total_avg_ms"],
        "postcheck_mpi_ms": timing["rank_postcheck_mpi_avg_ms"],
        "communication_ms": timing["rank_communication_avg_ms"],
        "algorithm_total_ms": timing["rank_algorithm_total_avg_ms"],
    }
    return {
        "timing_scope": "multigpu_rank_aggregate",
        "direct_fields": sorted(timing),
        "derived_fields": [],
        "normalization": {
            "rank_max_ms.algorithm_total_ms": (
                "emitted rank_algorithm_total_max_ms; distributed internal completion latency"
            ),
            "rank_mean_ms.algorithm_total_ms": (
                "emitted rank_algorithm_total_avg_ms; per-rank work mean"
            ),
            "component_rule": (
                "rank-mean components may explain a mean total; rank-max components are not additive"
            ),
        },
        "rank_max_ms": rank_max, "rank_mean_ms": rank_mean,
    }


def parse_benchmark_stdout(stdout: str, method: str) -> Dict[str, Any]:
    """Parse canonical records and retain exact emitted token lines."""
    timing_lines = machine_record_lines(stdout, TIMING_PREFIX)
    iteration_lines = machine_record_lines(stdout, ITERATION_PREFIX)
    if len(timing_lines) != 1:
        raise BenchmarkOutputError(
            f"expected exactly one {TIMING_PREFIX}, found {len(timing_lines)}"
        )
    if len(iteration_lines) != 1:
        raise BenchmarkOutputError(
            f"expected exactly one {ITERATION_PREFIX}, found {len(iteration_lines)}"
        )
    timing = parse_numeric_record(timing_lines[0], TIMING_PREFIX)
    iteration_values = parse_numeric_record(iteration_lines[0], ITERATION_PREFIX)
    if set(iteration_values) != {"iterations"}:
        raise BenchmarkOutputError("BENCHMARK_ITERATIONS must contain only iterations=<integer>")
    iteration_value = iteration_values["iterations"]
    iterations = int(iteration_value)
    if iterations <= 0 or iteration_value != iterations:
        raise BenchmarkOutputError(f"iterations must be a positive integer, got {iteration_value}")
    return {
        "timing_record": TIMING_PREFIX + " " + timing_lines[0],
        "iteration_record": ITERATION_PREFIX + " " + iteration_lines[0],
        "timing": timing, "iterations": iterations,
        "normalized": normalise_timing(timing, method),
    }


def invocation_environment() -> Dict[str, str]:
    """Capture scheduler and GPU assignment context per raw attempt."""
    return {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "nccl_debug": os.environ.get("NCCL_DEBUG", ""),
        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME", ""),
        "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE", ""),
        "ompi_mca_rmaps_base_oversubscribe": os.environ.get(
            "OMPI_MCA_rmaps_base_oversubscribe", "1"
        ),
        "ompi_mca_btl": os.environ.get("OMPI_MCA_btl", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT", ""),
    }


def invocation_id(
    algorithm: str, dataset: str, phase: str, repeat: int, method: str,
) -> str:
    """Return a deterministic identity used by raw record and analysis tables."""
    return f"{algorithm}.{dataset}.{phase}.r{repeat:03d}.{method}"


def display_command(command: Sequence[str]) -> str:
    """Render argv without shell evaluation ambiguity."""
    return shlex.join(list(command))
