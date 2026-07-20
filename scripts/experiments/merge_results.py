#!/usr/bin/env python3
"""Strictly merge dataset-by-algorithm Slurm array benchmark results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from .run_benchmarks import (
        METHODS,
        MULTIGPU_METHODS,
        RESULT_FIELDS,
        RESULT_ID_FIELDS,
        RESULT_SCHEMA_VERSION,
    )
except ImportError:
    from run_benchmarks import (  # type: ignore
        METHODS,
        MULTIGPU_METHODS,
        RESULT_FIELDS,
        RESULT_ID_FIELDS,
        RESULT_SCHEMA_VERSION,
    )

METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
MERGE_VALIDATION_RETURNCODE = -995
Identity = Tuple[str, str, str, str]
WorkItem = Tuple[str, str]


@dataclass(frozen=True)
class MergeOutcome:
    output: Path
    has_issues: bool
    report: Dict[str, object]


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [
            {
                key: ("" if value is None else value)
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]
        return list(reader.fieldnames or []), rows


def write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def identity(row: Mapping[str, str]) -> Identity:
    return tuple((row.get(field) or "").strip() for field in RESULT_ID_FIELDS)  # type: ignore[return-value]


def identity_dict(value: Identity) -> Dict[str, str]:
    return dict(zip(RESULT_ID_FIELDS, value))


def identity_label(value: Identity) -> str:
    dataset, algorithm, repeat, method = value
    return f"{dataset}.{algorithm}.{method}.r{repeat}"


def repeat_sort_value(value: str) -> Tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def row_sort_key(row: Mapping[str, str]) -> Tuple[object, ...]:
    dataset, algorithm, repeat, method = identity(row)
    return (
        dataset,
        algorithm,
        repeat_sort_value(repeat),
        METHOD_ORDER.get(method, 99),
    )


def parse_finite(
    row: Mapping[str, str],
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Optional[str]:
    raw_value = (row.get(field) or "").strip()
    try:
        value = float(raw_value)
    except ValueError:
        return f"{field} is not numeric: {raw_value!r}"
    if not math.isfinite(value):
        return f"{field} is not finite"
    if positive and value <= 0.0:
        return f"{field} must be positive"
    if nonnegative and value < 0.0:
        return f"{field} must be non-negative"
    return None


def append_reason(row: Dict[str, str], reason: str) -> None:
    existing = (row.get("failure_reason") or "").strip()
    row["failure_reason"] = f"{existing}; {reason}" if existing else reason


def mark_merge_failure(row: Dict[str, str], reason: str) -> None:
    """Make a structurally invalid row impossible for plotters to accept."""
    row["returncode"] = str(MERGE_VALIDATION_RETURNCODE)
    row["run_status"] = "merge_validation_error"
    row["valid"] = "0"
    append_reason(row, reason)


def validate_row(row: Dict[str, str]) -> List[str]:
    """Return strict schema/state errors for one row."""
    errors: List[str] = []
    if (row.get("schema_version") or "").strip() != RESULT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {RESULT_SCHEMA_VERSION}"
        )

    row_identity = identity(row)
    if any(not item for item in row_identity):
        errors.append("identity fields must all be non-empty")

    method = (row.get("method") or "").strip()
    try:
        effective_returncode = int((row.get("returncode") or "").strip())
    except ValueError:
        errors.append("returncode is missing or non-integer")
        return errors

    valid = (row.get("valid") or "").strip()
    status = (row.get("run_status") or "").strip()
    process_returncode = (row.get("process_returncode") or "").strip()

    if effective_returncode != 0:
        if valid != "0":
            errors.append("failed row must have valid=0")
        return errors

    if valid != "1" or status != "success" or process_returncode != "0":
        errors.append(
            "returncode=0 requires valid=1, run_status=success, "
            "process_returncode=0"
        )

    required_positive = (
        "nodes",
        "edges",
        "file_size_bytes",
        "wall_time_ms",
        "iterations",
        "gpu_count",
        "mpi_ranks",
        "rank_count",
    )
    required_nonnegative = (
        "gpu_time_ms",
        "gpu_main_ms",
        "main_loop_ms",
        "gpu_compute_ms",
        "graph_kernel_ms",
        "cpu_check_tail_ms",
        "cpu_check_drain_ms",
        "nccl_exchange_ms",
        "mpi_sync_ms",
        "postcheck_total_ms",
        "postcheck_mpi_ms",
        "communication_ms",
        "rank_gpu_main_avg_ms",
        "rank_main_loop_avg_ms",
        "rank_gpu_compute_avg_ms",
        "rank_graph_kernel_avg_ms",
        "rank_cpu_check_tail_avg_ms",
        "rank_cpu_check_drain_avg_ms",
        "rank_nccl_exchange_avg_ms",
        "rank_mpi_sync_avg_ms",
        "rank_postcheck_total_avg_ms",
        "rank_postcheck_mpi_avg_ms",
        "rank_communication_avg_ms",
        "rank_algorithm_total_avg_ms",
        "rank_algorithm_total_max_ms",
        "algorithm_total_ms",
        "rank_imbalance_pct",
        "graph_kernel_per_iter_ms",
    )
    for field in required_positive:
        if error := parse_finite(row, field, positive=True):
            errors.append(error)
    for field in required_nonnegative:
        if error := parse_finite(row, field, nonnegative=True):
            errors.append(error)

    for subset, total in (
        ("graph_kernel_ms", "gpu_compute_ms"),
        ("rank_graph_kernel_avg_ms", "rank_gpu_compute_avg_ms"),
    ):
        try:
            subset_value = float((row.get(subset) or "").strip())
            total_value = float((row.get(total) or "").strip())
        except ValueError:
            continue
        if not math.isfinite(subset_value) or not math.isfinite(total_value):
            continue
        tolerance = max(
            0.002,
            1e-5 * max(1.0, subset_value, total_value),
        )
        if subset_value > total_value + tolerance:
            errors.append(f"{subset} cannot exceed {total}")

    expected_ranks = 2 if method in MULTIGPU_METHODS else 1
    for field in ("gpu_count", "mpi_ranks", "rank_count"):
        raw_value = (row.get(field) or "").strip()
        try:
            actual = int(raw_value)
        except ValueError:
            continue
        if actual != expected_ranks:
            errors.append(
                f"{field}={actual}, expected {expected_ranks} for {method}"
            )
    return errors


def load_work_items(
    run_root: Path,
) -> Tuple[Dict[WorkItem, int], List[str]]:
    """Load algorithm/dataset/repeat expectations from work_items.tsv."""
    path = run_root / "work_items.tsv"
    if not path.exists():
        return {}, []

    errors: List[str] = []
    items: Dict[WorkItem, int] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"algorithm", "dataset", "repeat"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            return {}, [
                f"{path} missing columns: {', '.join(sorted(missing))}"
            ]
        for line_number, row in enumerate(reader, start=2):
            algorithm = (row.get("algorithm") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            raw_repeat = (row.get("repeat") or "").strip()
            try:
                repeat = int(raw_repeat)
            except ValueError:
                errors.append(
                    f"{path}:{line_number}: invalid repeat {raw_repeat!r}"
                )
                continue
            key = (algorithm, dataset)
            if not algorithm or not dataset or repeat < 1:
                errors.append(
                    f"{path}:{line_number}: invalid work item"
                )
            elif key in items:
                errors.append(
                    f"{path}:{line_number}: duplicate work item "
                    f"{algorithm}/{dataset}"
                )
            else:
                items[key] = repeat
    return items, errors


def expected_from_configs(
    input_paths: Sequence[Path],
) -> Tuple[Set[Identity], List[str]]:
    """Build expected identities for local/non-array runs."""
    expected: Set[Identity] = set()
    errors: List[str] = []
    for result_path in input_paths:
        config_path = result_path.parent / "run_config.json"
        if not config_path.exists():
            errors.append(f"missing run config: {config_path}")
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            repeat_count = int(config["repeat"])
            datasets = list(config["datasets"])
            algorithms = list(config["algorithms"])
            methods = list(config["methods"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid run config {config_path}: {exc}")
            continue
        for dataset in datasets:
            for algorithm in algorithms:
                for repeat in range(1, repeat_count + 1):
                    for method in methods:
                        expected.add(
                            (
                                str(dataset),
                                str(algorithm),
                                str(repeat),
                                str(method),
                            )
                        )
    return expected, errors


def expected_from_work_items(
    items: Mapping[WorkItem, int],
) -> Set[Identity]:
    return {
        (dataset, algorithm, str(repeat), method)
        for (algorithm, dataset), repeat_count in items.items()
        for repeat in range(1, repeat_count + 1)
        for method in METHODS
    }


def results_work_item(path: Path, run_root: Path) -> Optional[WorkItem]:
    """Resolve runs/<algorithm>/<dataset>/results.csv."""
    try:
        relative = path.relative_to(run_root / "runs")
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 3 or parts[-1] != "results.csv":
        return None
    return parts[0], parts[1]


def merge_results(run_root: Path, output: Path) -> MergeOutcome:
    runs_root = run_root / "runs"
    input_paths = sorted(runs_root.rglob("results.csv")) if runs_root.exists() else []
    work_items, work_item_errors = load_work_items(run_root)

    accepted_inputs: List[str] = []
    invalid_inputs: List[Dict[str, object]] = []
    rows: List[Dict[str, str]] = []
    work_items_with_files: Set[WorkItem] = set()

    for path in input_paths:
        try:
            input_fields, input_rows = read_csv(path)
        except (OSError, csv.Error) as exc:
            invalid_inputs.append(
                {"path": str(path), "errors": [str(exc)]}
            )
            continue

        header_errors: List[str] = []
        if input_fields != RESULT_FIELDS:
            missing = [
                field for field in RESULT_FIELDS if field not in input_fields
            ]
            extra = [
                field for field in input_fields if field not in RESULT_FIELDS
            ]
            if missing:
                header_errors.append(
                    "missing fields: " + ", ".join(missing)
                )
            if extra:
                header_errors.append(
                    "unexpected fields: " + ", ".join(extra)
                )
            if not missing and not extra:
                header_errors.append("field order differs from schema")
        if header_errors:
            invalid_inputs.append(
                {"path": str(path), "errors": header_errors}
            )
            continue

        file_work_item = results_work_item(path, run_root)
        if work_items and file_work_item is None:
            invalid_inputs.append(
                {
                    "path": str(path),
                    "errors": [
                        "expected path runs/<algorithm>/<dataset>/results.csv"
                    ],
                }
            )
            continue

        accepted_inputs.append(str(path))
        if file_work_item is not None:
            work_items_with_files.add(file_work_item)

        for row in input_rows:
            if file_work_item is not None:
                expected_algorithm, expected_dataset = file_work_item
                if (
                    (row.get("algorithm") or "").strip()
                    != expected_algorithm
                    or (row.get("dataset") or "").strip()
                    != expected_dataset
                ):
                    mark_merge_failure(
                        row,
                        "row identity does not match its "
                        "runs/<algorithm>/<dataset> directory",
                    )
            row_errors = validate_row(row)
            if row_errors:
                mark_merge_failure(row, "; ".join(row_errors))
            rows.append(row)

    by_identity: Dict[Identity, List[int]] = {}
    for row_index, row in enumerate(rows):
        by_identity.setdefault(identity(row), []).append(row_index)

    duplicate_identities: List[Identity] = []
    for row_identity, indices in by_identity.items():
        if len(indices) <= 1:
            continue
        duplicate_identities.append(row_identity)
        for row_index in indices:
            mark_merge_failure(
                rows[row_index],
                "duplicate identity across merged results",
            )

    if work_items:
        expected_identities = expected_from_work_items(work_items)
        config_errors: List[str] = []
    else:
        expected_identities, config_errors = expected_from_configs(
            input_paths
        )

    actual_identities = {
        value for row in rows
        if all(value := identity(row))
    }
    missing_identities = sorted(
        expected_identities.difference(actual_identities)
    )
    unexpected_identities = sorted(
        actual_identities.difference(expected_identities)
        if expected_identities
        else set()
    )
    if unexpected_identities:
        unexpected_set = set(unexpected_identities)
        for row in rows:
            if identity(row) in unexpected_set:
                mark_merge_failure(
                    row, "identity is not declared by experiment manifests"
                )

    rows.sort(key=row_sort_key)
    write_csv_atomic(output, RESULT_FIELDS, rows)

    failures = [
        row
        for row in rows
        if (
            (row.get("returncode") or "").strip() != "0"
            or (row.get("valid") or "").strip() != "1"
            or (row.get("run_status") or "").strip() != "success"
        )
    ]
    failure_fields = [
        "schema_version", "run_id", "dataset", "algorithm", "method",
        "repeat", "returncode", "process_returncode", "run_status", "valid",
        "failure_reason", "hostname", "slurm_job_id", "slurm_array_task_id",
        "command", "stdout_log", "stderr_log",
    ]
    failed_csv = run_root / "failed_runs.csv"
    write_csv_atomic(failed_csv, failure_fields, failures)

    missing_work_items = sorted(
        set(work_items).difference(work_items_with_files)
    )
    all_errors = [*work_item_errors, *config_errors]
    has_issues = bool(
        failures
        or invalid_inputs
        or duplicate_identities
        or missing_identities
        or unexpected_identities
        or missing_work_items
        or all_errors
        or not input_paths
    )
    report: Dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "partial" if has_issues else "complete",
        "input_files_found": len(input_paths),
        "input_files_accepted": len(accepted_inputs),
        "accepted_inputs": accepted_inputs,
        "invalid_inputs": invalid_inputs,
        "result_rows": len(rows),
        "successful_rows": len(rows) - len(failures),
        "failed_rows": len(failures),
        "expected_work_items": len(work_items),
        "work_items_with_results": len(work_items_with_files),
        "missing_work_items": [
            {"algorithm": algorithm, "dataset": dataset}
            for algorithm, dataset in missing_work_items
        ],
        "expected_rows": len(expected_identities),
        "missing_rows": [
            identity_dict(value) for value in missing_identities
        ],
        "unexpected_rows": [
            identity_dict(value) for value in unexpected_identities
        ],
        "duplicate_identities": [
            identity_dict(value) for value in sorted(duplicate_identities)
        ],
        "manifest_errors": all_errors,
        "output": str(output),
        "failed_runs": str(failed_csv),
    }
    write_json_atomic(run_root / "collection_report.json", report)
    return MergeOutcome(
        output=output,
        has_issues=has_issues,
        report=report,
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively merge runs/<algorithm>/<dataset>/results.csv"
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <run_root>/results.csv",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else run_root / "results.csv"
    )
    outcome = merge_results(run_root, output)
    print(f"merged results: {outcome.output}")
    print(f"collection report: {run_root / 'collection_report.json'}")
    if outcome.has_issues:
        print(
            "error: collection completed with failed, missing, duplicate, "
            "or schema-invalid runs",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
