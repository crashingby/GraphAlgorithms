#!/usr/bin/env python3
"""Execute one immutable dataset × algorithm experiment work item.

Each command invocation writes a new raw JSON attempt and separate stdout/stderr
logs.  Collectors never need to infer a measurement from a mutable shared CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common import (
    BenchmarkOutputError,
    DATASET_DIR,
    ROOT,
    build_command,
    display_command,
    executable_path,
    execute_command,
    file_fingerprint,
    invocation_environment,
    invocation_id,
    json_dump_atomic,
    json_load,
    machine_record_lines,
    method_gpu_mode,
    read_metadata,
    relative_to,
    text_dump_atomic,
    utc_now,
)


def resolve_path(path: Path) -> Path:
    """Interpret paths from the repository root."""
    return path if path.is_absolute() else ROOT / path


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    """Load a prepared immutable experiment manifest."""
    manifest = json_load(run_dir / "manifest" / "experiment.json")
    if manifest.get("record_type") != "experiment_manifest":
        raise ValueError(f"{run_dir} does not contain a prepared experiment manifest")
    return manifest


def load_work_items(run_dir: Path) -> List[Dict[str, Any]]:
    """Load planned Slurm-safe work items."""
    document = json_load(run_dir / "manifest" / "work_items.json")
    items = document.get("items")
    if not isinstance(items, list):
        raise ValueError("manifest/work_items.json has no item list")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("manifest/work_items.json contains an invalid item")
    return [dict(item) for item in items]


def select_items(
    items: Sequence[Mapping[str, Any]], work_id: Optional[str], all_items: bool,
) -> List[Dict[str, Any]]:
    """Select exactly one worker item, or all items for a serial local run."""
    if all_items:
        if work_id:
            raise ValueError("--all and --work-id cannot be used together")
        return [dict(item) for item in items]
    if not work_id:
        raise ValueError("provide --work-id or --all")
    selected = [dict(item) for item in items if item.get("work_id") == work_id]
    if len(selected) != 1:
        raise ValueError(f"unknown work id: {work_id}")
    return selected


def raw_records_for_identity(
    run_dir: Path, work_id: str, record_id: str,
) -> List[Dict[str, Any]]:
    """Load retained attempts for one planned invocation identity."""
    result: List[Dict[str, Any]] = []
    root = run_dir / "raw" / "attempts" / work_id
    if not root.exists():
        return result
    for path in root.rglob("*.json"):
        try:
            record = json_load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if record.get("record_id") == record_id:
            result.append(record)
    return result


def should_skip(
    prior_records: Sequence[Mapping[str, Any]], resume: bool, retry_failed: bool,
) -> bool:
    """Choose an explicit resume policy without overwriting old evidence."""
    if not resume or not prior_records:
        return False
    has_success = any(record.get("status") == "success" for record in prior_records)
    if has_success:
        return True
    return not retry_failed


def metadata_view(row: Mapping[str, str]) -> Dict[str, Any]:
    """Keep dataset fields needed to interpret raw attempts without CSV lookups."""
    fields = (
        "nodes", "edges", "valid_edges", "file_size_bytes", "bfs_source",
        "bfs_source_outdegree", "bfs_source_indegree", "bfs_source_total_degree",
    )
    return {field: row.get(field, "") for field in fields}


def file_guard(
    path: Path, expected: Mapping[str, Any], *, include_sha256: bool,
) -> Dict[str, Any]:
    """Compare one execution input to the immutable manifest without hashing graphs again."""
    try:
        observed = file_fingerprint(path, include_sha256=include_sha256)
    except OSError as exc:
        return {
            "path": str(path), "observed": None, "matches_manifest": False,
            "differences": {
                "unreadable": {"expected": "readable input", "observed": str(exc)}
            },
        }
    fields = ["path", "size_bytes", "mtime_ns"]
    if include_sha256 and "sha256" in expected:
        fields.append("sha256")
    differences = {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in fields
        if expected.get(field) != observed.get(field)
    }
    return {
        "path": observed["path"], "observed": observed,
        "matches_manifest": not differences, "differences": differences,
    }


def execution_input_guard(
    manifest: Mapping[str, Any], algorithm: str, dataset: str, method: str,
) -> Dict[str, Any]:
    """Record pre/post input evidence and reject a run when frozen inputs drift."""
    try:
        expected_dataset = manifest["dataset_fingerprints"][dataset]
        expected_binary = manifest["executable_fingerprints"][f"{algorithm}/{method}"]
    except (KeyError, TypeError) as exc:
        return {
            "status": "mismatch",
            "files": {},
            "manifest_error": f"missing frozen input fingerprint: {exc}",
        }
    graph = file_guard(
        DATASET_DIR / f"{dataset}.mtx", expected_dataset, include_sha256=False,
    )
    binary = file_guard(
        executable_path(algorithm, method), expected_binary, include_sha256=True,
    )
    return {
        "status": "match" if graph["matches_manifest"] and binary["matches_manifest"] else "mismatch",
        "files": {"dataset": graph, "binary": binary},
    }


def input_guard_reason(guard: Mapping[str, Any]) -> str:
    """Render a concise immutable-input failure reason for an attempt record."""
    if guard.get("manifest_error"):
        return str(guard["manifest_error"])
    broken = [
        f"{name}({', '.join(report.get('differences', {}))})"
        for name, report in (guard.get("files") or {}).items()
        if not report.get("matches_manifest")
    ]
    return "frozen input drift: " + "; ".join(broken or ["unknown input mismatch"])


def input_drift_process(reason: str) -> Dict[str, Any]:
    """Create a retained failed process result when a frozen input no longer matches."""
    timestamp = utc_now()
    return {
        "status": "input_drift", "process_returncode": None,
        "stdout": "", "stderr": f"INPUT DRIFT: {reason}\n",
        "process_wall_ms": 0.0,
        "started_at_utc": timestamp, "finished_at_utc": timestamp,
        "failure_reason": reason,
    }


def run_one(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    item: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, str]],
    phase: str,
    repeat: int,
    method: str,
    method_order_index: int,
    resume: bool,
    retry_failed: bool,
) -> Optional[Dict[str, Any]]:
    """Execute and persist one warmup or measurement command."""
    config = manifest["config"]
    algorithm = str(item["algorithm"])
    dataset = str(item["dataset"])
    record_id = invocation_id(algorithm, dataset, phase, repeat, method)
    prior = raw_records_for_identity(run_dir, str(item["work_id"]), record_id)
    if should_skip(prior, resume, retry_failed):
        has_success = any(record.get("status") == "success" for record in prior)
        state = "successful" if has_success else "failed"
        print(f"skip existing {state} {record_id}", flush=True)
        return {"status": "success" if has_success else "retained_failure"}

    command = build_command(config, dataset, metadata[dataset], algorithm, method)
    attempt_id = str(uuid.uuid4())
    logs_dir = run_dir / "raw" / "logs"
    stdout_log = logs_dir / f"{attempt_id}.stdout.log"
    stderr_log = logs_dir / f"{attempt_id}.stderr.log"

    print(f"run {record_id}: {display_command(command)}", flush=True)
    guard_before = execution_input_guard(manifest, algorithm, dataset, method)
    guard_after: Optional[Dict[str, Any]] = None
    if guard_before["status"] == "match":
        process = execute_command(command, int(config["timeout_s"]))
        guard_after = execution_input_guard(manifest, algorithm, dataset, method)
        if guard_after["status"] != "match":
            reason = input_guard_reason(guard_after)
            process["status"] = "input_drift"
            process["failure_reason"] = reason
            stderr = str(process.get("stderr", ""))
            process["stderr"] = stderr + ("" if not stderr or stderr.endswith("\n") else "\n") + (
                f"INPUT DRIFT AFTER EXECUTION: {reason}\n"
            )
    else:
        process = input_drift_process(input_guard_reason(guard_before))
    input_guard = {"before": guard_before, "after": guard_after}
    text_dump_atomic(stdout_log, str(process.pop("stdout")))
    text_dump_atomic(stderr_log, str(process.pop("stderr")))

    status = process["status"]
    emitted: Dict[str, Any] = {
        "timing_record_count": 0,
        "iteration_record_count": 0,
        "timing_record": None,
        "iteration_record": None,
        "timing_tokens": None,
        "iterations": None,
    }
    normalized: Optional[Dict[str, Any]] = None
    if status == "process_success":
        stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace")
        emitted["timing_record_count"] = len(
            machine_record_lines(stdout_text, "BENCHMARK_TIMING")
        )
        emitted["iteration_record_count"] = len(
            machine_record_lines(stdout_text, "BENCHMARK_ITERATIONS")
        )
        try:
            from common import parse_benchmark_stdout
            parsed = parse_benchmark_stdout(stdout_text, method)
            emitted.update({
                "timing_record": parsed["timing_record"],
                "iteration_record": parsed["iteration_record"],
                "timing_tokens": parsed["timing"],
                "iterations": parsed["iterations"],
            })
            normalized = parsed["normalized"]
            status = "success"
            process["failure_reason"] = ""
        except BenchmarkOutputError as exc:
            status = "parse_error"
            process["failure_reason"] = str(exc)

    role = "tolerance" if method in ("tolerance_queue", "multigpu") else "baseline"
    record: Dict[str, Any] = {
        "schema_version": manifest["schema_version"],
        "record_type": "benchmark_attempt",
        "attempt_id": attempt_id,
        "record_id": record_id,
        "status": status,
        "phase": phase,
        "work_id": item["work_id"],
        "identity": {
            "dataset": dataset,
            "algorithm": algorithm,
            "method": method,
            "repeat": repeat,
            "method_order_index": method_order_index,
            "gpu_mode": method_gpu_mode(method),
            "mpi_ranks": (
                int(config["mpi_ranks"])
                if method_gpu_mode(method) == "multigpu" else 1
            ),
            "comparison_role": role,
            "pair_id": (
                f"{algorithm}|{dataset}|r{repeat:03d}|{method_gpu_mode(method)}"
                if phase == "measure" else None
            ),
        },
        "config": {
            "argv": command,
            "command_display": display_command(command),
            "parameters": config["parameters"],
            "timeout_s": config["timeout_s"],
            "launcher": config["launcher"],
        },
        "dataset_metadata": metadata_view(metadata[dataset]),
        "provenance": {
            "manifest_dataset_fingerprint": manifest["dataset_fingerprints"][dataset],
            "manifest_executable_fingerprint": manifest["executable_fingerprints"][
                f"{algorithm}/{method}"
            ],
            "execution_input_guard": input_guard,
            "execution_environment": invocation_environment(),
        },
        "process": {
            **process,
            "stdout_path": relative_to(stdout_log, run_dir),
            "stderr_path": relative_to(stderr_log, run_dir),
        },
        "emitted": emitted,
        "timing": normalized,
    }
    record_path = (
        run_dir / "raw" / "attempts" / str(item["work_id"]) / phase
        / f"{record_id}.{attempt_id}.json"
    )
    json_dump_atomic(record_path, record)
    print(
        f"  -> {status}; raw record: {relative_to(record_path, run_dir)}",
        flush=True,
    )
    return record


def method_order(methods: Sequence[str], repeat: int) -> List[str]:
    """Alternate pair order on successive measured repeats to reduce order bias."""
    return list(methods) if repeat % 2 else list(reversed(methods))


def run_item(
    run_dir: Path,
    item: Mapping[str, Any],
    *,
    resume: bool,
    retry_failed: bool,
) -> int:
    """Run one item and fail the worker only for failed measured identities.

    Warmups remain fully retained evidence, but they are not part of the plotted
    sample population and therefore produce a warning rather than a failed job.
    """
    manifest = load_manifest(run_dir)
    metadata_snapshot = run_dir / manifest["input_snapshot"]["metadata_snapshot"]
    metadata = read_metadata(metadata_snapshot)
    config = manifest["config"]
    warmup_failures = 0
    measurement_failures = 0

    for warmup_index in range(1, int(config["warmup"]) + 1):
        for position, method in enumerate(config["methods"], start=1):
            record = run_one(
                run_dir=run_dir, manifest=manifest, item=item, metadata=metadata,
                phase="warmup", repeat=warmup_index, method=method,
                method_order_index=position, resume=resume,
                retry_failed=retry_failed,
            )
            if record is not None and record["status"] != "success":
                warmup_failures += 1

    for repeat in range(1, int(config["repeat"]) + 1):
        for position, method in enumerate(method_order(config["methods"], repeat), start=1):
            record = run_one(
                run_dir=run_dir, manifest=manifest, item=item, metadata=metadata,
                phase="measure", repeat=repeat, method=method,
                method_order_index=position, resume=resume,
                retry_failed=retry_failed,
            )
            if record is not None and record["status"] != "success":
                measurement_failures += 1
    if warmup_failures:
        print(
            f"warning: retained {warmup_failures} failed warmup attempts for {item['work_id']}",
            file=sys.stderr,
        )
    return measurement_failures


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build worker CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-id")
    parser.add_argument("--all", action="store_true", help="run every item serially")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip an identity that already has a retained attempt",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="with --resume, retry identities that only have failed attempts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run selected work item(s)."""
    args = parse_args(argv)
    try:
        run_dir = resolve_path(args.run_dir)
        items = select_items(load_work_items(run_dir), args.work_id, args.all)
        failures = sum(
            run_item(
                run_dir, item, resume=args.resume, retry_failed=args.retry_failed,
            )
            for item in items
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if failures:
        print(f"completed with {failures} failed measured identities", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
