#!/usr/bin/env python3
"""Pair, quality-check and summarize collected fault-tolerance overhead samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from common import ROOT, SCHEMA_VERSION, csv_dump_atomic, json_dump_atomic, json_load


PAIR_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("1gpu", "basic", "tolerance_queue"),
    ("multigpu", "multigpu_basic", "multigpu"),
)
TIMING_METRICS: Tuple[str, ...] = (
    "algorithm_total_ms", "main_loop_ms", "gpu_compute_ms", "graph_kernel_ms",
    "cpu_check_drain_ms", "nccl_exchange_ms", "mpi_sync_ms",
    "postcheck_total_ms", "postcheck_mpi_ms", "communication_ms",
)


def resolve_path(path: Path) -> Path:
    """Interpret paths from the repository root."""
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read one collected table."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def number(row: Mapping[str, Any], field: str) -> Optional[float]:
    """Return one finite numeric CSV field, or None for absent/unusable data."""
    value = row.get(field)
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(row: Mapping[str, Any], field: str) -> Optional[int]:
    """Return one integral positive-ish field."""
    value = number(row, field)
    if value is None or value != int(value):
        return None
    return int(value)


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Use a deterministic linear percentile suitable for small repeat counts."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sample_stats(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    """Summarize finite samples without silently converting missing data to zero."""
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "p25": None, "p75": None, "min": None, "max": None,
        }
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "p25": percentile(clean, 0.25),
        "p75": percentile(clean, 0.75),
        "min": min(clean),
        "max": max(clean),
    }


def add_stats(row: Dict[str, Any], prefix: str, values: Iterable[Optional[float]]) -> None:
    """Append a standard stable sample-statistics column family."""
    for key, value in sample_stats(values).items():
        row[f"{prefix}_{key}"] = value


def valid_success(row: Mapping[str, str]) -> bool:
    """A collected observation is usable only when selected and fully parsed."""
    return (
        row.get("selected") == "1"
        and row.get("phase") == "measure"
        and row.get("status") == "success"
    )


def pair_identity(
    algorithm: str, dataset: str, repeat: int, gpu_mode: str,
) -> Tuple[str, str, int, str]:
    """Use one resource-stable repeat identity."""
    return algorithm, dataset, repeat, gpu_mode


def group_observations(
    rows: Sequence[Mapping[str, str]],
) -> Dict[Tuple[str, str, int, str, str], List[Mapping[str, str]]]:
    """Index selected observations by exact algorithm/dataset/repeat/method."""
    result: Dict[Tuple[str, str, int, str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        repeat = integer(row, "repeat")
        if row.get("phase") != "measure" or repeat is None:
            continue
        key = (
            row.get("algorithm", ""), row.get("dataset", ""), repeat,
            row.get("gpu_mode", ""), row.get("method", ""),
        )
        result.setdefault(key, []).append(row)
    return result


def expected_pair_keys(
    manifest: Mapping[str, Any], run_dir: Path,
) -> List[Tuple[str, str, int, str]]:
    """Expand the plan to all pair identities from the currently supplied run root."""
    work_document = json_load(run_dir / "manifest" / "work_items.json")
    items = work_document.get("items", [])
    config = manifest["config"]
    selected = set(config["methods"])
    keys: List[Tuple[str, str, int, str]] = []
    for item in items:
        for gpu_mode, baseline, tolerance in PAIR_DEFINITIONS:
            if baseline not in selected and tolerance not in selected:
                continue
            for repeat in range(1, int(config["repeat"]) + 1):
                keys.append(pair_identity(
                    str(item["algorithm"]), str(item["dataset"]), repeat, gpu_mode,
                ))
    return sorted(keys)


def pick_row(
    grouped: Mapping[Tuple[str, str, int, str, str], Sequence[Mapping[str, str]]],
    key: Tuple[str, str, int, str, str],
) -> Tuple[Optional[Mapping[str, str]], Optional[str]]:
    """Return one observation or a structural failure reason."""
    rows = grouped.get(key, [])
    if not rows:
        return None, "missing"
    if len(rows) != 1:
        return None, f"duplicate_selected_observations={len(rows)}"
    return rows[0], None


def same_pair_environment(
    baseline: Mapping[str, str], tolerance: Mapping[str, str],
) -> Optional[str]:
    """Ensure a pair used the same effective execution and hardware context."""
    for field in (
        "hostname", "cuda_visible_devices", "mpi_ranks", "cuda_device_order",
        "omp_num_threads", "nccl_debug", "nccl_socket_ifname", "nccl_ib_disable",
        "ompi_mca_rmaps_base_oversubscribe", "ompi_mca_btl",
    ):
        left = (baseline.get(field) or "").strip()
        right = (tolerance.get(field) or "").strip()
        if left != right:
            return f"environment_mismatch:{field}"
    return None


def metric(row: Mapping[str, str], aggregate: str, metric_name: str) -> Optional[float]:
    """Read a clearly named rank aggregation metric from a flattened observation."""
    return number(row, f"{aggregate}_{metric_name}")


def percent_overhead(
    baseline: Optional[float], tolerance: Optional[float],
) -> Optional[float]:
    """Return paired percent overhead only when the baseline is positive."""
    if baseline is None or tolerance is None or baseline <= 0.0:
        return None
    return (tolerance / baseline - 1.0) * 100.0


PAIR_FIELDS: List[str] = [
    "pair_id", "algorithm", "dataset", "repeat", "gpu_mode", "baseline_method",
    "tolerance_method", "baseline_record_id", "tolerance_record_id",
    "baseline_attempt_id", "tolerance_attempt_id", "baseline_status",
    "tolerance_status", "baseline_iterations", "tolerance_iterations",
    "pair_complete_valid", "overhead_eligible", "pair_status",
    "exclusion_reason", "baseline_hostname", "tolerance_hostname",
    "baseline_cuda_visible_devices", "tolerance_cuda_visible_devices",
]
for aggregate in ("rank_max", "rank_mean"):
    for timing_metric in TIMING_METRICS:
        PAIR_FIELDS.extend((
            f"baseline_{aggregate}_{timing_metric}",
            f"tolerance_{aggregate}_{timing_metric}",
        ))
PAIR_FIELDS.extend((
    "baseline_process_wall_ms", "tolerance_process_wall_ms",
    "total_completion_overhead_pct", "rank_mean_graph_kernel_overhead_pct",
    "rank_max_graph_kernel_overhead_pct", "rank_imbalance_basic_pct",
    "rank_imbalance_tolerance_pct",
))


def make_pair_row(
    key: Tuple[str, str, int, str],
    grouped: Mapping[Tuple[str, str, int, str, str], Sequence[Mapping[str, str]]],
) -> Dict[str, Any]:
    """Build one pair row, including every excluded/missing condition explicitly."""
    algorithm, dataset, repeat, gpu_mode = key
    baseline_method, tolerance_method = (
        ("basic", "tolerance_queue") if gpu_mode == "1gpu"
        else ("multigpu_basic", "multigpu")
    )
    baseline, baseline_error = pick_row(
        grouped, (*key, baseline_method),
    )
    tolerance, tolerance_error = pick_row(
        grouped, (*key, tolerance_method),
    )
    row: Dict[str, Any] = {
        "pair_id": f"{algorithm}|{dataset}|r{repeat:03d}|{gpu_mode}",
        "algorithm": algorithm, "dataset": dataset, "repeat": repeat,
        "gpu_mode": gpu_mode, "baseline_method": baseline_method,
        "tolerance_method": tolerance_method,
        "baseline_record_id": baseline.get("record_id") if baseline else None,
        "tolerance_record_id": tolerance.get("record_id") if tolerance else None,
        "baseline_attempt_id": baseline.get("attempt_id") if baseline else None,
        "tolerance_attempt_id": tolerance.get("attempt_id") if tolerance else None,
        "baseline_status": baseline.get("status") if baseline else None,
        "tolerance_status": tolerance.get("status") if tolerance else None,
        "baseline_iterations": integer(baseline, "iterations") if baseline else None,
        "tolerance_iterations": integer(tolerance, "iterations") if tolerance else None,
        "pair_complete_valid": 0, "overhead_eligible": 0,
        "pair_status": "missing_or_invalid", "exclusion_reason": "",
        "baseline_hostname": baseline.get("hostname") if baseline else None,
        "tolerance_hostname": tolerance.get("hostname") if tolerance else None,
        "baseline_cuda_visible_devices": (
            baseline.get("cuda_visible_devices") if baseline else None
        ),
        "tolerance_cuda_visible_devices": (
            tolerance.get("cuda_visible_devices") if tolerance else None
        ),
    }
    for aggregate in ("rank_max", "rank_mean"):
        for timing_metric in TIMING_METRICS:
            row[f"baseline_{aggregate}_{timing_metric}"] = (
                metric(baseline, aggregate, timing_metric) if baseline else None
            )
            row[f"tolerance_{aggregate}_{timing_metric}"] = (
                metric(tolerance, aggregate, timing_metric) if tolerance else None
            )
    row["baseline_process_wall_ms"] = number(baseline, "process_wall_ms") if baseline else None
    row["tolerance_process_wall_ms"] = number(tolerance, "process_wall_ms") if tolerance else None

    failures = [reason for reason in (baseline_error, tolerance_error) if reason]
    if not failures and baseline is not None and tolerance is not None:
        if not valid_success(baseline):
            failures.append(f"baseline_{baseline.get('status', 'invalid')}")
        if not valid_success(tolerance):
            failures.append(f"tolerance_{tolerance.get('status', 'invalid')}")
    if failures:
        row["exclusion_reason"] = ";".join(failures)
    else:
        row["pair_complete_valid"] = 1
        environment_error = same_pair_environment(baseline, tolerance)  # type: ignore[arg-type]
        baseline_iterations = row["baseline_iterations"]
        tolerance_iterations = row["tolerance_iterations"]
        if environment_error:
            row["pair_status"] = "environment_mismatch"
            row["exclusion_reason"] = environment_error
        elif baseline_iterations != tolerance_iterations:
            row["pair_status"] = "iteration_mismatch"
            row["exclusion_reason"] = "iterations differ"
        else:
            row["pair_status"] = "eligible"
            row["overhead_eligible"] = 1

    if row["overhead_eligible"] == 1:
        row["total_completion_overhead_pct"] = percent_overhead(
            row["baseline_rank_max_algorithm_total_ms"],
            row["tolerance_rank_max_algorithm_total_ms"],
        )
        row["rank_mean_graph_kernel_overhead_pct"] = percent_overhead(
            row["baseline_rank_mean_graph_kernel_ms"],
            row["tolerance_rank_mean_graph_kernel_ms"],
        )
        row["rank_max_graph_kernel_overhead_pct"] = percent_overhead(
            row["baseline_rank_max_graph_kernel_ms"],
            row["tolerance_rank_max_graph_kernel_ms"],
        )
    else:
        row["total_completion_overhead_pct"] = None
        row["rank_mean_graph_kernel_overhead_pct"] = None
        row["rank_max_graph_kernel_overhead_pct"] = None
    if gpu_mode == "multigpu":
        basic_mean = row["baseline_rank_mean_algorithm_total_ms"]
        tolerance_mean = row["tolerance_rank_mean_algorithm_total_ms"]
        row["rank_imbalance_basic_pct"] = (
            (row["baseline_rank_max_algorithm_total_ms"] / basic_mean - 1.0) * 100.0
            if basic_mean and basic_mean > 0.0 else None
        )
        row["rank_imbalance_tolerance_pct"] = (
            (row["tolerance_rank_max_algorithm_total_ms"] / tolerance_mean - 1.0) * 100.0
            if tolerance_mean and tolerance_mean > 0.0 else None
        )
    else:
        row["rank_imbalance_basic_pct"] = None
        row["rank_imbalance_tolerance_pct"] = None
    return row


METHOD_SUMMARY_FIELDS: List[str] = [
    "algorithm", "dataset", "method", "gpu_mode", "successful_samples",
]
METHOD_METRICS: Tuple[str, ...] = (
    "completion_internal_ms", "rank_mean_internal_ms", "process_wall_ms",
    "rank_max_graph_kernel_ms", "rank_mean_graph_kernel_ms", "iterations",
    "rank_mean_cpu_check_drain_ms", "rank_mean_postcheck_total_ms",
    "rank_imbalance_pct",
)
for name in METHOD_METRICS:
    METHOD_SUMMARY_FIELDS.extend(
        f"{name}_{suffix}" for suffix in ("n", "mean", "median", "std", "p25", "p75", "min", "max")
    )


def build_method_summary(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Summarize absolute method times without requiring the other pair to exist."""
    grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        if not valid_success(row):
            continue
        key = (
            row.get("algorithm", ""), row.get("dataset", ""),
            row.get("method", ""), row.get("gpu_mode", ""),
        )
        grouped.setdefault(key, []).append(row)
    result: List[Dict[str, Any]] = []
    for (algorithm, dataset, method, gpu_mode), values in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "algorithm": algorithm, "dataset": dataset, "method": method,
            "gpu_mode": gpu_mode, "successful_samples": len(values),
        }
        add_stats(row, "completion_internal_ms", (
            number(item, "rank_max_algorithm_total_ms") for item in values
        ))
        add_stats(row, "rank_mean_internal_ms", (
            number(item, "rank_mean_algorithm_total_ms") for item in values
        ))
        add_stats(row, "process_wall_ms", (number(item, "process_wall_ms") for item in values))
        add_stats(row, "rank_max_graph_kernel_ms", (
            number(item, "rank_max_graph_kernel_ms") for item in values
        ))
        add_stats(row, "rank_mean_graph_kernel_ms", (
            number(item, "rank_mean_graph_kernel_ms") for item in values
        ))
        add_stats(row, "rank_mean_cpu_check_drain_ms", (
            number(item, "rank_mean_cpu_check_drain_ms") for item in values
        ))
        add_stats(row, "rank_mean_postcheck_total_ms", (
            number(item, "rank_mean_postcheck_total_ms") for item in values
        ))
        add_stats(row, "iterations", (number(item, "iterations") for item in values))
        add_stats(row, "rank_imbalance_pct", (
            (
                (number(item, "rank_max_algorithm_total_ms") /
                 number(item, "rank_mean_algorithm_total_ms") - 1.0) * 100.0
            )
            if number(item, "rank_mean_algorithm_total_ms")
            and number(item, "rank_mean_algorithm_total_ms") > 0.0
            else None
            for item in values
        ))
        result.append(row)
    return result


PAIR_SUMMARY_FIELDS: List[str] = [
    "algorithm", "dataset", "gpu_mode", "complete_valid_pairs",
    "eligible_pairs", "excluded_pairs", "missing_or_invalid_pairs",
    "iteration_mismatch_pairs", "environment_mismatch_pairs",
]
PAIR_SUMMARY_METRICS: Tuple[str, ...] = (
    "total_completion_overhead_pct", "rank_mean_graph_kernel_overhead_pct",
    "rank_max_graph_kernel_overhead_pct", "rank_imbalance_basic_pct",
    "rank_imbalance_tolerance_pct",
    "baseline_rank_mean_nccl_exchange_ms", "tolerance_rank_mean_nccl_exchange_ms",
    "baseline_rank_mean_mpi_sync_ms", "tolerance_rank_mean_mpi_sync_ms",
    "baseline_rank_mean_cpu_check_drain_ms", "tolerance_rank_mean_cpu_check_drain_ms",
    "baseline_rank_mean_postcheck_total_ms", "tolerance_rank_mean_postcheck_total_ms",
)
for name in PAIR_SUMMARY_METRICS:
    PAIR_SUMMARY_FIELDS.extend(
        f"{name}_{suffix}" for suffix in ("n", "mean", "median", "std", "p25", "p75", "min", "max")
    )


def build_pair_summary(pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize only strict eligible paired comparisons, plus raw pair counts."""
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in pairs:
        grouped.setdefault(
            (str(row["algorithm"]), str(row["dataset"]), str(row["gpu_mode"])), []
        ).append(row)
    result: List[Dict[str, Any]] = []
    for (algorithm, dataset, gpu_mode), values in sorted(grouped.items()):
        complete = [item for item in values if item["pair_complete_valid"] == 1]
        eligible = [item for item in values if item["overhead_eligible"] == 1]
        row: Dict[str, Any] = {
            "algorithm": algorithm, "dataset": dataset, "gpu_mode": gpu_mode,
            "complete_valid_pairs": len(complete), "eligible_pairs": len(eligible),
            "excluded_pairs": len(values) - len(eligible),
            "missing_or_invalid_pairs": sum(
                item["pair_status"] == "missing_or_invalid" for item in values
            ),
            "iteration_mismatch_pairs": sum(
                item["pair_status"] == "iteration_mismatch" for item in values
            ),
            "environment_mismatch_pairs": sum(
                item["pair_status"] == "environment_mismatch" for item in values
            ),
        }
        for name in PAIR_SUMMARY_METRICS:
            add_stats(row, name, (number(item, name) for item in eligible))
        result.append(row)
    return result


def analyze(run_dir: Path) -> Dict[str, Any]:
    """Write pair-level samples, exclusions and summaries without mutating raw data."""
    manifest = json_load(run_dir / "manifest" / "experiment.json")
    observations = read_csv(run_dir / "collected" / "observations.csv")
    grouped = group_observations(observations)
    pairs = [make_pair_row(key, grouped) for key in expected_pair_keys(manifest, run_dir)]
    pairs.sort(key=lambda row: (
        str(row["algorithm"]), str(row["dataset"]), str(row["gpu_mode"]), int(row["repeat"]),
    ))
    excluded = [row for row in pairs if row["overhead_eligible"] != 1]
    method_summary = build_method_summary(observations)
    pair_summary = build_pair_summary(pairs)

    output = run_dir / "analysis"
    csv_dump_atomic(output / "paired_samples.csv", PAIR_FIELDS, pairs)
    csv_dump_atomic(output / "excluded_pairs.csv", PAIR_FIELDS, excluded)
    csv_dump_atomic(output / "method_summary.csv", METHOD_SUMMARY_FIELDS, method_summary)
    csv_dump_atomic(output / "pair_summary.csv", PAIR_SUMMARY_FIELDS, pair_summary)
    report = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "overhead_analysis_report",
        "pair_count": len(pairs),
        "eligible_pair_count": sum(row["overhead_eligible"] == 1 for row in pairs),
        "excluded_pair_count": len(excluded),
        "status_counts": {
            status: sum(row["pair_status"] == status for row in pairs)
            for status in sorted({str(row["pair_status"]) for row in pairs})
        },
        "method_summary_rows": len(method_summary),
        "pair_summary_rows": len(pair_summary),
        "notes": [
            "Multi-GPU completion time uses rank_max_algorithm_total_ms.",
            "Multi-GPU components and graph-kernel comparison use rank mean and are not additive.",
            "Every baseline/tolerance comparative metric requires both success, same execution context and identical iteration count.",
            "Excluded pairs remain in excluded_pairs.csv with the exact reason; tolerance-only absolute tail data stays in method_summary.csv.",
        ],
    }
    json_dump_atomic(output / "analysis_report.json", report)
    print(f"pair samples: {len(pairs)}")
    print(f"eligible overhead pairs: {report['eligible_pair_count']}")
    print(f"excluded pairs: {len(excluded)}")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build analysis CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--strict", action="store_true",
        help="return nonzero if any pair is excluded",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Analyze a collected run."""
    args = parse_args(argv)
    try:
        report = analyze(resolve_path(args.run_dir))
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 3 if args.strict and report["excluded_pair_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
