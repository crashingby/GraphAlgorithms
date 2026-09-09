#!/usr/bin/env python3
"""Collect immutable raw benchmark attempts into auditable flat observation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from common import (
    ROOT,
    SCHEMA_VERSION,
    csv_dump_atomic,
    invocation_id,
    json_dump_atomic,
    json_load,
    method_gpu_mode,
    relative_to,
)


TIMING_METRICS: Tuple[str, ...] = (
    "algorithm_total_ms", "main_loop_ms", "gpu_compute_ms", "graph_kernel_ms",
    "cpu_check_drain_ms", "nccl_exchange_ms", "mpi_sync_ms",
    "postcheck_total_ms", "postcheck_mpi_ms", "communication_ms",
)
ATTEMPT_STATUSES = frozenset((
    "success", "process_error", "timeout", "launch_error", "parse_error", "input_drift",
))
OBSERVATION_FIELDS: List[str] = [
    "schema_version", "attempt_id", "record_id", "record_path", "selected",
    "selection_note", "status", "phase", "work_id", "dataset", "algorithm",
    "method", "repeat", "method_order_index", "gpu_mode", "mpi_ranks",
    "comparison_role", "pair_id", "process_returncode", "timeout_cleanup", "failure_reason",
    "started_at_utc", "finished_at_utc", "process_wall_ms", "stdout_path",
    "stderr_path", "hostname", "cuda_visible_devices", "cuda_device_order",
    "omp_num_threads", "nccl_debug", "nccl_socket_ifname", "nccl_ib_disable",
    "ompi_mca_rmaps_base_oversubscribe", "ompi_mca_btl", "slurm_job_id",
    "slurm_array_job_id", "slurm_array_task_id", "slurm_nodelist",
    "slurm_partition", "slurm_account", "nodes", "edges", "valid_edges",
    "file_size_bytes", "bfs_source", "iterations", "timing_scope",
    "input_guard_status", "input_guard_before_json", "input_guard_after_json",
    "direct_fields_json", "derived_fields_json", "emitted_timing_record",
    "emitted_iteration_record", "emitted_timing_tokens_json",
]
for aggregate in ("rank_max", "rank_mean"):
    OBSERVATION_FIELDS.extend(
        f"{aggregate}_{metric}" for metric in TIMING_METRICS
    )


def resolve_path(path: Path) -> Path:
    """Interpret CLI paths from the repository root."""
    return path if path.is_absolute() else ROOT / path


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    """Load the immutable run description."""
    manifest = json_load(run_dir / "manifest" / "experiment.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    return manifest


def load_work_items(run_dir: Path) -> List[Dict[str, Any]]:
    """Load planned array tasks."""
    document = json_load(run_dir / "manifest" / "work_items.json")
    items = document.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("invalid manifest/work_items.json")
    return [dict(item) for item in items]


def expected_identities(
    manifest: Mapping[str, Any], items: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Expand the manifest into every expected warmup and measure identity."""
    config = manifest["config"]
    expected: Dict[str, Dict[str, Any]] = {}
    for item in items:
        algorithm = str(item["algorithm"])
        dataset = str(item["dataset"])
        for phase, count in (
            ("warmup", int(config["warmup"])),
            ("measure", int(config["repeat"])),
        ):
            for repeat in range(1, count + 1):
                for method in config["methods"]:
                    record_id = invocation_id(algorithm, dataset, phase, repeat, method)
                    expected[record_id] = {
                        "work_id": item["work_id"], "algorithm": algorithm,
                        "dataset": dataset, "phase": phase, "repeat": repeat,
                        "method": method,
                    }
    return expected


def validate_attempt(
    record: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Optional[str]:
    """Reject semantic-invalid raw JSON before it can become an observation."""
    if record.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version={record.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
    if record.get("record_type") != "benchmark_attempt":
        return "record_type is not benchmark_attempt"
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        return "record_id is missing or invalid"
    contract = expected.get(record_id)
    if contract is None:
        return "record_id is not present in the frozen manifest"
    if record.get("work_id") != contract["work_id"]:
        return "work_id does not match the frozen manifest"
    if record.get("phase") != contract["phase"]:
        return "phase does not match the frozen manifest"
    identity = record.get("identity")
    if not isinstance(identity, dict):
        return "identity is missing or invalid"
    for field in ("algorithm", "dataset", "method", "repeat"):
        if identity.get(field) != contract[field]:
            return f"identity.{field} does not match the frozen manifest"
    method = str(contract["method"])
    gpu_mode = method_gpu_mode(method)
    if identity.get("gpu_mode") != gpu_mode:
        return "identity.gpu_mode does not match method"
    expected_mpi_ranks = (
        int(manifest["config"]["mpi_ranks"])
        if gpu_mode == "multigpu" else 1
    )
    if identity.get("mpi_ranks") != expected_mpi_ranks:
        return "identity.mpi_ranks does not match method"
    status = record.get("status")
    if status not in ATTEMPT_STATUSES:
        return f"unsupported attempt status {status!r}"
    process = record.get("process")
    emitted = record.get("emitted")
    if not isinstance(process, dict) or not isinstance(emitted, dict):
        return "process or emitted evidence is missing or invalid"
    if status == "success":
        if process.get("process_returncode") != 0:
            return "successful attempt has a nonzero process return code"
        iterations = emitted.get("iterations")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            return "successful attempt has no positive integer iteration count"
        timing = record.get("timing")
        if not isinstance(timing, dict):
            return "successful attempt has no normalized timing"
        if not isinstance(timing.get("rank_max_ms"), dict) or not isinstance(
            timing.get("rank_mean_ms"), dict
        ):
            return "successful attempt has no rank timing aggregates"
        provenance = record.get("provenance")
        guard = provenance.get("execution_input_guard") if isinstance(provenance, dict) else None
        if not isinstance(guard, dict):
            return "successful attempt lacks an execution input guard"
        for point in ("before", "after"):
            snapshot = guard.get(point)
            if not isinstance(snapshot, dict) or snapshot.get("status") != "match":
                return f"successful attempt has no matching {point} input guard"
    elif record.get("timing") is not None:
        return "failed attempt must not contain normalized timing"
    return None


def invalid_attempt_entry(
    path: Path, record: Optional[Mapping[str, Any]], reason: str, run_dir: Path,
    expected: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Retain malformed evidence with the best available phase classification."""
    record_id = record.get("record_id") if isinstance(record, dict) else None
    contract = expected.get(record_id) if isinstance(record_id, str) else None
    phase = contract.get("phase") if contract else (
        record.get("phase") if isinstance(record, dict) else ""
    )
    return {
        "record_path": relative_to(path, run_dir), "record_id": record_id or "",
        "phase": phase or "", "reason": reason,
    }


def load_raw_attempts(
    run_dir: Path, expected: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[Dict[str, Any]]]:
    """Load every raw attempt, retaining invalid evidence but never selecting it."""
    attempts_root = run_dir / "raw" / "attempts"
    valid: List[Tuple[Path, Dict[str, Any]]] = []
    invalid: List[Dict[str, Any]] = []
    if not attempts_root.exists():
        return valid, invalid
    for path in sorted(attempts_root.rglob("*.json")):
        record: Optional[Dict[str, Any]] = None
        try:
            record = json_load(path)
            reason = validate_attempt(record, expected, manifest)
            if reason:
                invalid.append(invalid_attempt_entry(path, record, reason, run_dir, expected))
            else:
                valid.append((path, record))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append(invalid_attempt_entry(path, record, str(exc), run_dir, expected))
    return valid, invalid


def attempt_sort_key(item: Tuple[Path, Mapping[str, Any]]) -> Tuple[str, str]:
    """Sort a retry history by finished timestamp and then durable path."""
    path, record = item
    process = record.get("process")
    finish = process.get("finished_at_utc", "") if isinstance(process, dict) else ""
    return str(finish), str(path)


def choose_observations(
    attempts: Sequence[Tuple[Path, Dict[str, Any]]],
) -> Tuple[Dict[str, Tuple[Path, Dict[str, Any]]], Dict[str, str], Dict[str, int]]:
    """Select the latest successful attempt per identity, retaining all attempts."""
    grouped: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
    for item in attempts:
        record_id = item[1].get("record_id")
        if not isinstance(record_id, str) or not record_id:
            continue
        grouped.setdefault(record_id, []).append(item)

    chosen: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    notes: Dict[str, str] = {}
    count: Dict[str, int] = {}
    for record_id, values in grouped.items():
        ordered = sorted(values, key=attempt_sort_key)
        successes = [item for item in ordered if item[1].get("status") == "success"]
        chosen[record_id] = successes[-1] if successes else ordered[-1]
        count[record_id] = len(ordered)
        if len(successes) > 1:
            notes[record_id] = f"{len(successes)} successful attempts; latest success selected"
        elif len(ordered) > 1:
            notes[record_id] = f"{len(ordered)} attempts; latest successful attempt selected when available"
        elif successes:
            notes[record_id] = "only successful attempt"
        else:
            notes[record_id] = "only failed attempt"
    return chosen, notes, count


def value_at(mapping: Any, key: str) -> Any:
    """Safely fetch a JSON object property."""
    return mapping.get(key) if isinstance(mapping, dict) else None


def flatten_attempt(
    path: Path,
    record: Mapping[str, Any],
    run_dir: Path,
    *,
    selected: bool,
    selection_note: str,
) -> Dict[str, Any]:
    """Project one raw JSON record into a wide, explicitly aggregated CSV row."""
    identity = value_at(record, "identity") or {}
    process = value_at(record, "process") or {}
    provenance = value_at(record, "provenance") or {}
    execution = value_at(provenance, "execution_environment") or {}
    input_guard = value_at(provenance, "execution_input_guard") or {}
    input_guard_before = value_at(input_guard, "before") or {}
    input_guard_after = value_at(input_guard, "after")
    metadata = value_at(record, "dataset_metadata") or {}
    emitted = value_at(record, "emitted") or {}
    timing = value_at(record, "timing") or {}
    row: Dict[str, Any] = {
        "schema_version": record.get("schema_version"),
        "attempt_id": record.get("attempt_id"),
        "record_id": record.get("record_id"),
        "record_path": relative_to(path, run_dir),
        "selected": int(selected),
        "selection_note": selection_note,
        "status": record.get("status"),
        "phase": record.get("phase"),
        "work_id": record.get("work_id"),
        "dataset": identity.get("dataset"),
        "algorithm": identity.get("algorithm"),
        "method": identity.get("method"),
        "repeat": identity.get("repeat"),
        "method_order_index": identity.get("method_order_index"),
        "gpu_mode": identity.get("gpu_mode"),
        "mpi_ranks": identity.get("mpi_ranks"),
        "comparison_role": identity.get("comparison_role"),
        "pair_id": identity.get("pair_id"),
        "process_returncode": process.get("process_returncode"),
        "timeout_cleanup": process.get("timeout_cleanup"),
        "failure_reason": process.get("failure_reason"),
        "started_at_utc": process.get("started_at_utc"),
        "finished_at_utc": process.get("finished_at_utc"),
        "process_wall_ms": process.get("process_wall_ms"),
        "stdout_path": process.get("stdout_path"),
        "stderr_path": process.get("stderr_path"),
        "hostname": execution.get("hostname"),
        "cuda_visible_devices": execution.get("cuda_visible_devices"),
        "cuda_device_order": execution.get("cuda_device_order"),
        "omp_num_threads": execution.get("omp_num_threads"),
        "nccl_debug": execution.get("nccl_debug"),
        "nccl_socket_ifname": execution.get("nccl_socket_ifname"),
        "nccl_ib_disable": execution.get("nccl_ib_disable"),
        "ompi_mca_rmaps_base_oversubscribe": execution.get(
            "ompi_mca_rmaps_base_oversubscribe"
        ),
        "ompi_mca_btl": execution.get("ompi_mca_btl"),
        "slurm_job_id": execution.get("slurm_job_id"),
        "slurm_array_job_id": execution.get("slurm_array_job_id"),
        "slurm_array_task_id": execution.get("slurm_array_task_id"),
        "slurm_nodelist": execution.get("slurm_nodelist"),
        "slurm_partition": execution.get("slurm_partition"),
        "slurm_account": execution.get("slurm_account"),
        "nodes": metadata.get("nodes"),
        "edges": metadata.get("edges"),
        "valid_edges": metadata.get("valid_edges"),
        "file_size_bytes": metadata.get("file_size_bytes"),
        "bfs_source": metadata.get("bfs_source"),
        "iterations": emitted.get("iterations"),
        "timing_scope": timing.get("timing_scope"),
        "input_guard_status": input_guard_before.get("status"),
        "input_guard_before_json": input_guard_before,
        "input_guard_after_json": input_guard_after,
        "direct_fields_json": timing.get("direct_fields"),
        "derived_fields_json": timing.get("derived_fields"),
        "emitted_timing_record": emitted.get("timing_record"),
        "emitted_iteration_record": emitted.get("iteration_record"),
        "emitted_timing_tokens_json": emitted.get("timing_tokens"),
    }
    for aggregate in ("rank_max", "rank_mean"):
        values = value_at(timing, f"{aggregate}_ms") or {}
        for metric in TIMING_METRICS:
            row[f"{aggregate}_{metric}"] = values.get(metric)
    return row


def collect(run_dir: Path) -> Dict[str, Any]:
    """Create collected attempts/observations tables and a complete audit report."""
    manifest = load_manifest(run_dir)
    expected = expected_identities(manifest, load_work_items(run_dir))
    attempts, invalid_records = load_raw_attempts(run_dir, expected, manifest)
    selected, notes, attempt_counts = choose_observations(attempts)

    all_rows = [
        flatten_attempt(
            path, record, run_dir,
            selected=(selected.get(str(record.get("record_id"))) == (path, record)),
            selection_note=notes.get(str(record.get("record_id")), "invalid record id"),
        )
        for path, record in attempts
    ]
    all_rows.sort(key=lambda row: (str(row["record_id"]), str(row["attempt_id"])))
    observation_rows = [
        flatten_attempt(path, record, run_dir, selected=True, selection_note=notes[record_id])
        for record_id, (path, record) in sorted(selected.items())
    ]
    observation_rows.sort(key=lambda row: (
        str(row["algorithm"]), str(row["dataset"]), str(row["phase"]),
        int(row["repeat"]) if str(row["repeat"]).isdigit() else -1, str(row["method"]),
    ))

    expected_ids = set(expected)
    observed_ids = set(selected)
    missing = sorted(expected_ids.difference(observed_ids))
    expected_measurement_ids = {
        record_id for record_id, contract in expected.items() if contract["phase"] == "measure"
    }
    expected_warmup_ids = expected_ids.difference(expected_measurement_ids)
    missing_measurements = sorted(expected_measurement_ids.difference(observed_ids))
    missing_warmups = sorted(expected_warmup_ids.difference(observed_ids))
    successful_measurements = [
        row for row in observation_rows
        if row["phase"] == "measure" and row["status"] == "success"
    ]
    failed_selected = [
        row for row in observation_rows if row["status"] != "success"
    ]
    failed_measurements = [
        row for row in failed_selected if row["phase"] == "measure"
    ]
    failed_warmups = [
        row for row in failed_selected if row["phase"] == "warmup"
    ]
    invalid_measurements = [
        row for row in invalid_records if row.get("phase") == "measure"
    ]
    invalid_warmups = [
        row for row in invalid_records if row.get("phase") == "warmup"
    ]
    unclassified_invalid = [
        row for row in invalid_records if row.get("phase") not in {"measure", "warmup"}
    ]
    unexpected = sorted({
        str(row["record_id"]) for row in invalid_records
        if row.get("record_id") and row["record_id"] not in expected_ids
    })
    measurement_complete = not (
        missing_measurements or failed_measurements or invalid_measurements or unclassified_invalid
    )
    warmup_complete = not (missing_warmups or failed_warmups or invalid_warmups)

    output = run_dir / "collected"
    csv_dump_atomic(output / "attempts.csv", OBSERVATION_FIELDS, all_rows)
    csv_dump_atomic(output / "observations.csv", OBSERVATION_FIELDS, observation_rows)
    csv_dump_atomic(output / "successful_measurements.csv", OBSERVATION_FIELDS, successful_measurements)
    csv_dump_atomic(output / "failed_or_invalid_observations.csv", OBSERVATION_FIELDS, failed_selected)
    report = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "collection_report",
        "raw_attempt_count": len(attempts),
        "invalid_raw_record_count": len(invalid_records),
        "invalid_measurement_record_count": len(invalid_measurements),
        "invalid_warmup_record_count": len(invalid_warmups),
        "unclassified_invalid_record_count": len(unclassified_invalid),
        "selected_observation_count": len(observation_rows),
        "successful_measurement_count": len(successful_measurements),
        "failed_selected_measurement_count": len(failed_measurements),
        "failed_selected_warmup_count": len(failed_warmups),
        "failed_or_invalid_selected_count": len(failed_selected),
        "expected_identity_count": len(expected_ids),
        "missing_identity_count": len(missing),
        "missing_measurement_identity_count": len(missing_measurements),
        "missing_warmup_identity_count": len(missing_warmups),
        "unexpected_identity_count": len(unexpected),
        "retry_identity_count": sum(1 for value in attempt_counts.values() if value > 1),
        "missing_identities": [{"record_id": item, **expected[item]} for item in missing],
        "missing_measurement_identities": [
            {"record_id": item, **expected[item]} for item in missing_measurements
        ],
        "missing_warmup_identities": [
            {"record_id": item, **expected[item]} for item in missing_warmups
        ],
        "unexpected_identities": [{"record_id": item} for item in unexpected],
        "invalid_raw_records": invalid_records,
        "measurement_status": "complete" if measurement_complete else "partial",
        "warmup_status": "complete" if warmup_complete else "warning",
        "status": "complete" if measurement_complete else "partial",
    }
    json_dump_atomic(output / "collection_report.json", report)
    print(f"collected attempts: {len(attempts)}")
    print(f"selected observations: {len(observation_rows)}")
    print(f"successful measurements: {len(successful_measurements)}")
    print(f"measurement status: {report['measurement_status']}; warmups: {report['warmup_status']}")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build collection CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--strict", action="store_true",
        help="return nonzero when measured identities are missing, invalid or unsuccessful",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Collect raw data without changing it."""
    args = parse_args(argv)
    try:
        report = collect(resolve_path(args.run_dir))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 3 if args.strict and report["measurement_status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
