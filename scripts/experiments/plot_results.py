#!/usr/bin/env python3
"""Plot paired GraphAlgorithms runtime, graph-kernel, and overhead details.

The four methods are matched inside each dataset + algorithm + repeat tuple.
Statistics are computed from per-repeat values, never from independently
aggregated methods. Failed, missing, or ambiguous rows are not treated as zero.

For multi-GPU runs, this script consumes the real rank-average timing fields
emitted by the executables. It intentionally never estimates a rank average by
dividing a rank maximum by the GPU count.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import textwrap
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

RESULT_SCHEMA_VERSION = "4"
ALGORITHMS = ("bfs", "cc", "kcore", "pagerank")
METHODS = ("basic", "tolerance_queue", "multigpu_basic", "multigpu")
MULTIGPU_METHODS = ("multigpu_basic", "multigpu")

RUNTIME_PREFIXES = (
    "single_basic_internal_total_ms",
    "single_queue_internal_total_ms",
    "two_gpu_basic_rank_internal_total_avg_ms",
    "two_gpu_tolerance_rank_internal_total_avg_ms",
)

KERNEL_PREFIXES = (
    "single_basic_graph_kernel_ms",
    "single_queue_graph_kernel_ms",
    "two_gpu_basic_rank_graph_kernel_avg_ms",
    "two_gpu_tolerance_rank_graph_kernel_avg_ms",
    "single_queue_graph_kernel_overhead_pct",
    "two_gpu_tolerance_rank_graph_kernel_overhead_pct",
)

ITERATION_PREFIXES = (
    "basic_iterations",
    "tolerance_queue_iterations",
    "multigpu_basic_iterations",
    "multigpu_iterations",
)

DETAIL_PREFIXES = (
    "single_queue_gpu_compute_overhead_pct",
    "two_gpu_tolerance_rank_gpu_compute_overhead_pct",
    "single_queue_cpu_drain_ms",
    "two_gpu_tolerance_rank_cpu_drain_avg_ms",
    "two_gpu_tolerance_rank_postcheck_total_avg_ms",
    "two_gpu_tolerance_rank_postcheck_mpi_avg_ms",
    "two_gpu_basic_rank_nccl_exchange_avg_ms",
    "two_gpu_basic_rank_mpi_sync_avg_ms",
    "two_gpu_tolerance_rank_nccl_exchange_avg_ms",
    "two_gpu_tolerance_rank_mpi_sync_avg_ms",
    "two_gpu_basic_rank_main_residual_avg_ms",
    "two_gpu_tolerance_rank_main_residual_avg_ms",
    "two_gpu_basic_rank_total_imbalance_pct",
    "two_gpu_tolerance_rank_total_imbalance_pct",
)

SUMMARY_FIELDS = [
    "dataset",
    "algorithm",
    "nodes",
    "edges",
    "method_matched_repeats",
    "method_repeat_ids",
    "runtime_matched_repeats",
    "runtime_repeat_ids",
    "kernel_matched_repeats",
    "kernel_repeat_ids",
    "iteration_missing_repeats",
    "iteration_missing_repeat_ids",
    "single_pair_iteration_mismatch_repeats",
    "single_pair_iteration_mismatch_repeat_ids",
    "two_gpu_pair_iteration_mismatch_repeats",
    "two_gpu_pair_iteration_mismatch_repeat_ids",
    "four_method_iteration_mismatch_repeats",
    "four_method_iteration_mismatch_repeat_ids",
    "rank_timing_source",
    "scale_score_ms",
]
for _prefix in (
    *ITERATION_PREFIXES, *RUNTIME_PREFIXES, *KERNEL_PREFIXES,
    *DETAIL_PREFIXES,
):
    SUMMARY_FIELDS.extend(
        [f"{_prefix}_samples", f"{_prefix}_mean", f"{_prefix}_std"]
    )

MANIFEST_FIELDS = [
    "algorithm",
    "group",
    "dataset_count",
    "scale_score_min_ms",
    "scale_score_max_ms",
    "scale_ratio",
    "datasets",
    "edges",
    "runtime_figure",
    "detail_figure",
    "kernel_figure",
    "runtime_series",
    "detail_panels",
    "kernel_panels",
    "input_files",
]

Row = Dict[str, str]
Identity = Tuple[str, str, str]
MethodRows = Dict[str, List[Row]]
PairIndex = Dict[Identity, MethodRows]
FourMethodPair = Tuple[str, Dict[str, Row]]


def parse_number(value: object) -> Optional[float]:
    """Convert one cell to a finite float, or return None."""
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def value(row: Mapping[str, str], field: str) -> Optional[float]:
    return parse_number(row.get(field))


def load_success_rows(results_csvs: Sequence[Path]) -> List[Row]:
    """Load successful rows from all input files."""
    rows: List[Row] = []
    for source_index, path in enumerate(results_csvs):
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "schema_version",
                "graph_kernel_ms",
                "rank_graph_kernel_avg_ms",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise RuntimeError(
                    f"{path} is not a kernel-timing result file; "
                    f"missing columns: {', '.join(sorted(missing))}"
                )
            for row_index, row in enumerate(reader, start=2):
                if (row.get("returncode") or "").strip() != "0":
                    continue
                schema = (row.get("schema_version") or "").strip()
                if schema != RESULT_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"{path}:{row_index}: schema_version={schema!r}; "
                        f"expected {RESULT_SCHEMA_VERSION}"
                    )
                for metric in (
                    "graph_kernel_ms",
                    "rank_graph_kernel_avg_ms",
                ):
                    if value(row, metric) is None:
                        raise RuntimeError(
                            f"{path}:{row_index}: missing/non-finite "
                            f"{metric} in successful row"
                        )
                copied = dict(row)
                copied["__source"] = str(path)
                copied["__source_index"] = str(source_index)
                copied["__row_index"] = str(row_index)
                rows.append(copied)
    return rows


def build_pair_index(rows: Iterable[Row]) -> PairIndex:
    """Index rows by the exact repeat identity and then method."""
    result: PairIndex = {}
    for row in rows:
        dataset = (row.get("dataset") or "").strip()
        algorithm = (row.get("algorithm") or "").strip()
        repeat = (row.get("repeat") or "").strip()
        method = (row.get("method") or "").strip()
        if not dataset or not algorithm or not repeat or method not in METHODS:
            continue
        result.setdefault((dataset, algorithm, repeat), {}).setdefault(
            method, []
        ).append(row)
    return result


def repeat_sort_key(repeat: str) -> Tuple[int, object]:
    try:
        return (0, int(repeat))
    except ValueError:
        return (1, repeat)


def reported_gpu_count(row: Mapping[str, str]) -> Optional[int]:
    count = value(row, "gpu_count")
    if count is None:
        count = value(row, "mpi_ranks")
    if count is None or not count.is_integer():
        return None
    return int(count)


def four_method_groups(
    index: PairIndex,
) -> Dict[Tuple[str, str], List[FourMethodPair]]:
    """Return unambiguous four-method pairs for actual two-GPU runs."""
    groups: Dict[Tuple[str, str], List[FourMethodPair]] = {}
    for (dataset, algorithm, repeat), method_rows in index.items():
        duplicate_methods = [
            method for method in METHODS if len(method_rows.get(method, [])) > 1
        ]
        if duplicate_methods:
            print(
                "warning: skipped ambiguous duplicate repeat "
                f"dataset={dataset} algorithm={algorithm} repeat={repeat} "
                f"methods={','.join(duplicate_methods)}",
                file=sys.stderr,
            )
            continue
        if any(len(method_rows.get(method, [])) != 1 for method in METHODS):
            continue

        methods = {method: method_rows[method][0] for method in METHODS}
        rank_counts = [
            reported_gpu_count(methods[method]) for method in MULTIGPU_METHODS
        ]
        if rank_counts != [2, 2]:
            print(
                "warning: skipped non-2-GPU repeat "
                f"dataset={dataset} algorithm={algorithm} repeat={repeat} "
                f"gpu_counts={rank_counts}",
                file=sys.stderr,
            )
            continue

        groups.setdefault((dataset, algorithm), []).append((repeat, methods))

    for pairs in groups.values():
        pairs.sort(key=lambda item: repeat_sort_key(item[0]))
    return groups


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def put_stats(
    output: Dict[str, object],
    prefix: str,
    samples: Sequence[float],
) -> None:
    mean, std = mean_std(samples)
    output[f"{prefix}_samples"] = len(samples)
    output[f"{prefix}_mean"] = mean
    output[f"{prefix}_std"] = std


def percent_overhead(
    baseline: Optional[float],
    checked: Optional[float],
) -> Optional[float]:
    if baseline is None or checked is None or baseline <= 0.0:
        return None
    return (checked / baseline - 1.0) * 100.0


def main_loop_residual(row: Mapping[str, str]) -> Optional[float]:
    """Return rank-mean main time not attributed to CUDA or main MPI."""
    components = (
        value(row, "rank_main_loop_avg_ms"),
        value(row, "rank_gpu_compute_avg_ms"),
        value(row, "rank_nccl_exchange_avg_ms"),
        value(row, "rank_mpi_sync_avg_ms"),
    )
    if any(item is None for item in components):
        return None
    main, compute, nccl, mpi_sync = components
    assert main is not None
    assert compute is not None
    assert nccl is not None
    assert mpi_sync is not None
    # Tiny negatives can result from four-decimal output rounding.
    return max(0.0, main - compute - nccl - mpi_sync)


def rank_imbalance(
    rank_total_avg_ms: Optional[float],
    rank_total_max_ms: Optional[float],
) -> Optional[float]:
    if (
        rank_total_avg_ms is None
        or rank_total_max_ms is None
        or rank_total_avg_ms <= 0.0
    ):
        return None
    return (rank_total_max_ms / rank_total_avg_ms - 1.0) * 100.0


def metadata_from_pairs(
    pairs: Sequence[FourMethodPair],
    field: str,
) -> Optional[float]:
    for _, methods in pairs:
        for method in METHODS:
            result = value(methods[method], field)
            if result is not None:
                return result
    return None


def display_metadata(number: Optional[float]) -> object:
    if number is None:
        return ""
    return int(number) if number.is_integer() else number


def append_if_present(samples: List[float], item: Optional[float]) -> None:
    if item is not None:
        samples.append(item)


def build_summary_rows(index: PairIndex) -> List[Dict[str, object]]:
    """Build one wide summary row for each dataset and algorithm."""
    rows: List[Dict[str, object]] = []
    for (dataset, algorithm), pairs in four_method_groups(index).items():
        output: Dict[str, object] = {
            "dataset": dataset,
            "algorithm": algorithm,
            "nodes": display_metadata(metadata_from_pairs(pairs, "nodes")),
            "edges": display_metadata(metadata_from_pairs(pairs, "edges")),
            "method_matched_repeats": len(pairs),
            "method_repeat_ids": ";".join(repeat for repeat, _ in pairs),
            "rank_timing_source": (
                "rank average fields; no rank-max/GPU-count fallback"
            ),
        }

        iteration_samples = {prefix: [] for prefix in ITERATION_PREFIXES}
        runtime_samples = {prefix: [] for prefix in RUNTIME_PREFIXES}
        kernel_samples = {prefix: [] for prefix in KERNEL_PREFIXES}
        detail_samples = {prefix: [] for prefix in DETAIL_PREFIXES}
        runtime_repeats: List[str] = []
        kernel_repeats: List[str] = []
        iteration_missing_repeats: List[str] = []
        single_pair_mismatch_repeats: List[str] = []
        two_gpu_pair_mismatch_repeats: List[str] = []
        four_method_mismatch_repeats: List[str] = []

        for repeat, methods in pairs:
            basic = methods["basic"]
            queue = methods["tolerance_queue"]
            multi_basic = methods["multigpu_basic"]
            multi_tolerance = methods["multigpu"]

            iterations_by_method = {
                "basic": value(basic, "iterations"),
                "tolerance_queue": value(queue, "iterations"),
                "multigpu_basic": value(multi_basic, "iterations"),
                "multigpu": value(multi_tolerance, "iterations"),
            }
            for prefix, method in zip(ITERATION_PREFIXES, METHODS):
                append_if_present(
                    iteration_samples[prefix],
                    iterations_by_method[method],
                )

            basic_iterations = iterations_by_method["basic"]
            queue_iterations = iterations_by_method["tolerance_queue"]
            multi_basic_iterations = iterations_by_method["multigpu_basic"]
            multi_tolerance_iterations = iterations_by_method["multigpu"]
            single_iterations_match = (
                basic_iterations is not None
                and queue_iterations is not None
                and basic_iterations == queue_iterations
            )
            two_gpu_iterations_match = (
                multi_basic_iterations is not None
                and multi_tolerance_iterations is not None
                and multi_basic_iterations == multi_tolerance_iterations
            )
            if (
                basic_iterations is not None
                and queue_iterations is not None
                and not single_iterations_match
            ):
                single_pair_mismatch_repeats.append(repeat)
            if (
                multi_basic_iterations is not None
                and multi_tolerance_iterations is not None
                and not two_gpu_iterations_match
            ):
                two_gpu_pair_mismatch_repeats.append(repeat)
            complete_iterations = [
                iterations_by_method[method] for method in METHODS
            ]
            if any(item is None for item in complete_iterations):
                iteration_missing_repeats.append(repeat)
            elif len(set(complete_iterations)) > 1:
                four_method_mismatch_repeats.append(repeat)

            single_basic_total = value(basic, "algorithm_total_ms")
            single_queue_total = value(queue, "algorithm_total_ms")
            multi_basic_total_avg = value(
                multi_basic, "rank_algorithm_total_avg_ms"
            )
            multi_tolerance_total_avg = value(
                multi_tolerance, "rank_algorithm_total_avg_ms"
            )
            basic_gpu_compute = value(basic, "gpu_compute_ms")
            queue_gpu_compute = value(queue, "gpu_compute_ms")
            multi_basic_gpu_compute_avg = value(
                multi_basic, "rank_gpu_compute_avg_ms"
            )
            multi_tolerance_gpu_compute_avg = value(
                multi_tolerance, "rank_gpu_compute_avg_ms"
            )
            basic_graph_kernel = value(basic, "graph_kernel_ms")
            queue_graph_kernel = value(queue, "graph_kernel_ms")
            multi_basic_graph_kernel_avg = value(
                multi_basic, "rank_graph_kernel_avg_ms"
            )
            multi_tolerance_graph_kernel_avg = value(
                multi_tolerance, "rank_graph_kernel_avg_ms"
            )

            kernel_values = (
                basic_graph_kernel,
                queue_graph_kernel,
                multi_basic_graph_kernel_avg,
                multi_tolerance_graph_kernel_avg,
            )
            if all(item is not None for item in kernel_values):
                kernel_repeats.append(repeat)
                for prefix, item in zip(
                    KERNEL_PREFIXES[:4], kernel_values
                ):
                    kernel_samples[prefix].append(float(item))

            runtime_values = (
                single_basic_total,
                single_queue_total,
                multi_basic_total_avg,
                multi_tolerance_total_avg,
            )
            if all(item is not None for item in runtime_values):
                runtime_repeats.append(repeat)
                for prefix, item in zip(RUNTIME_PREFIXES, runtime_values):
                    runtime_samples[prefix].append(float(item))

            if single_iterations_match:
                append_if_present(
                    kernel_samples[
                        "single_queue_graph_kernel_overhead_pct"
                    ],
                    percent_overhead(
                        basic_graph_kernel, queue_graph_kernel
                    ),
                )
                append_if_present(
                    detail_samples[
                        "single_queue_gpu_compute_overhead_pct"
                    ],
                    percent_overhead(
                        basic_gpu_compute, queue_gpu_compute
                    ),
                )
            if two_gpu_iterations_match:
                append_if_present(
                    kernel_samples[
                        "two_gpu_tolerance_rank_graph_kernel_overhead_pct"
                    ],
                    percent_overhead(
                        multi_basic_graph_kernel_avg,
                        multi_tolerance_graph_kernel_avg,
                    ),
                )
                append_if_present(
                    detail_samples[
                        "two_gpu_tolerance_rank_gpu_compute_overhead_pct"
                    ],
                    percent_overhead(
                        multi_basic_gpu_compute_avg,
                        multi_tolerance_gpu_compute_avg,
                    ),
                )
            append_if_present(
                detail_samples["single_queue_cpu_drain_ms"],
                value(queue, "cpu_check_drain_ms"),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_tolerance_rank_cpu_drain_avg_ms"
                ],
                value(
                    multi_tolerance,
                    "rank_cpu_check_drain_avg_ms",
                ),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_tolerance_rank_postcheck_total_avg_ms"
                ],
                value(
                    multi_tolerance,
                    "rank_postcheck_total_avg_ms",
                ),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_tolerance_rank_postcheck_mpi_avg_ms"
                ],
                value(
                    multi_tolerance,
                    "rank_postcheck_mpi_avg_ms",
                ),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_basic_rank_nccl_exchange_avg_ms"
                ],
                value(multi_basic, "rank_nccl_exchange_avg_ms"),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_basic_rank_mpi_sync_avg_ms"
                ],
                value(multi_basic, "rank_mpi_sync_avg_ms"),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_tolerance_rank_nccl_exchange_avg_ms"
                ],
                value(multi_tolerance, "rank_nccl_exchange_avg_ms"),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_tolerance_rank_mpi_sync_avg_ms"
                ],
                value(multi_tolerance, "rank_mpi_sync_avg_ms"),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_basic_rank_main_residual_avg_ms"
                ],
                main_loop_residual(multi_basic),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_tolerance_rank_main_residual_avg_ms"
                ],
                main_loop_residual(multi_tolerance),
            )
            append_if_present(
                detail_samples["two_gpu_basic_rank_total_imbalance_pct"],
                rank_imbalance(
                    multi_basic_total_avg,
                    value(multi_basic, "rank_algorithm_total_max_ms"),
                ),
            )
            append_if_present(
                detail_samples[
                    "two_gpu_tolerance_rank_total_imbalance_pct"
                ],
                rank_imbalance(
                    multi_tolerance_total_avg,
                    value(
                        multi_tolerance,
                        "rank_algorithm_total_max_ms",
                    ),
                ),
            )

        output["runtime_matched_repeats"] = len(runtime_repeats)
        output["runtime_repeat_ids"] = ";".join(runtime_repeats)
        output["kernel_matched_repeats"] = len(kernel_repeats)
        output["kernel_repeat_ids"] = ";".join(kernel_repeats)
        output["iteration_missing_repeats"] = len(
            iteration_missing_repeats
        )
        output["iteration_missing_repeat_ids"] = ";".join(
            iteration_missing_repeats
        )
        output["single_pair_iteration_mismatch_repeats"] = len(
            single_pair_mismatch_repeats
        )
        output["single_pair_iteration_mismatch_repeat_ids"] = ";".join(
            single_pair_mismatch_repeats
        )
        output["two_gpu_pair_iteration_mismatch_repeats"] = len(
            two_gpu_pair_mismatch_repeats
        )
        output["two_gpu_pair_iteration_mismatch_repeat_ids"] = ";".join(
            two_gpu_pair_mismatch_repeats
        )
        output["four_method_iteration_mismatch_repeats"] = len(
            four_method_mismatch_repeats
        )
        output["four_method_iteration_mismatch_repeat_ids"] = ";".join(
            four_method_mismatch_repeats
        )
        for prefix, samples in iteration_samples.items():
            put_stats(output, prefix, samples)
        for prefix, samples in runtime_samples.items():
            put_stats(output, prefix, samples)
        for prefix, samples in kernel_samples.items():
            put_stats(output, prefix, samples)
        for prefix, samples in detail_samples.items():
            put_stats(output, prefix, samples)

        runtime_means = [
            parse_number(output.get(f"{prefix}_mean"))
            for prefix in RUNTIME_PREFIXES
        ]
        available_means = [
            item for item in runtime_means if item is not None and item >= 0.0
        ]
        output["scale_score_ms"] = max(available_means) if available_means else None
        rows.append(output)

    rows.sort(key=summary_sort_key)
    return rows


def algorithm_sort_key(algorithm: str) -> Tuple[int, str]:
    try:
        return (ALGORITHMS.index(algorithm), algorithm)
    except ValueError:
        return (len(ALGORITHMS), algorithm)


def summary_sort_key(row: Mapping[str, object]) -> Tuple[object, ...]:
    score = parse_number(row.get("scale_score_ms"))
    return (
        *algorithm_sort_key(str(row.get("algorithm", ""))),
        score if score is not None else math.inf,
        str(row.get("dataset", "")),
    )


def group_by_runtime_scale(
    algorithm_rows: Sequence[Mapping[str, object]],
    datasets_per_figure: int,
    max_scale_ratio: float,
) -> List[List[Mapping[str, object]]]:
    """Group sorted datasets while bounding both count and runtime ratio."""
    candidates = [
        row
        for row in algorithm_rows
        if (
            (score := parse_number(row.get("scale_score_ms"))) is not None
            and score >= 0.0
        )
    ]
    candidates.sort(
        key=lambda row: (
            parse_number(row.get("scale_score_ms")) or 0.0,
            str(row.get("dataset", "")),
        )
    )

    groups: List[List[Mapping[str, object]]] = []
    current: List[Mapping[str, object]] = []
    current_min = 0.0
    for row in candidates:
        score = parse_number(row.get("scale_score_ms")) or 0.0
        count_limit = len(current) >= datasets_per_figure
        if not current:
            scale_limit = False
        elif current_min <= 0.0:
            scale_limit = score > 0.0
        else:
            scale_limit = score / current_min > max_scale_ratio

        if current and (count_limit or scale_limit):
            groups.append(current)
            current = []
        if not current:
            current_min = score
        current.append(row)

    if current:
        groups.append(current)
    return groups


def csv_cell(item: object) -> object:
    if item is None:
        return ""
    if isinstance(item, float):
        return f"{item:.10g}"
    return item


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_cell(row.get(field)) for field in fieldnames}
            )


ITERATION_MISMATCH_NOTE = "\n".join(
    textwrap.wrap(
        "† Iteration counts differ across methods in at least one matched "
        "repeat. Absolute times remain shown; overhead % uses only "
        "equal-iteration baseline/FT pairs.",
        width=84,
        break_long_words=False,
        break_on_hyphens=False,
    )
)


def has_iteration_mismatch(row: Mapping[str, object]) -> bool:
    count = parse_number(
        row.get("four_method_iteration_mismatch_repeats")
    )
    return count is not None and count > 0.0


def dataset_tick_label(row: Mapping[str, object]) -> str:
    dataset = str(row.get("dataset", ""))
    return f"{dataset}†" if has_iteration_mismatch(row) else dataset


def draw_grouped_bars(
    ax: object,
    rows: Sequence[Mapping[str, object]],
    series: Sequence[Tuple[str, str, str]],
    ylabel: str,
    panel_title: str,
    *,
    reference: Optional[float] = 0.0,
    nonnegative_axis: bool = False,
    show_x_labels: bool = False,
) -> List[str]:
    """Draw only available values; missing metrics do not become zero bars."""
    x = list(range(len(rows)))
    width = min(0.32, 0.80 / max(1, len(series)))
    midpoint = (len(series) - 1) / 2.0
    plotted_labels: List[str] = []

    for series_index, (label, prefix, color) in enumerate(series):
        positions: List[float] = []
        means: List[float] = []
        errors: List[float] = []
        for dataset_index, row in enumerate(rows):
            mean = parse_number(row.get(f"{prefix}_mean"))
            if mean is None:
                continue
            std = parse_number(row.get(f"{prefix}_std"))
            positions.append(
                dataset_index + (series_index - midpoint) * width
            )
            means.append(mean)
            errors.append(max(0.0, std or 0.0))

        if not means:
            continue
        ax.bar(
            positions,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=label,
            color=color,
            alpha=0.9,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        plotted_labels.append(label)

    if reference is not None:
        ax.axhline(
            reference,
            color="#555555",
            linewidth=1.0,
            linestyle="--",
        )
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, loc="left", fontsize=10)
    ax.set_xticks(x)
    if show_x_labels:
        ax.set_xticklabels(
            [dataset_tick_label(row) for row in rows],
            rotation=40,
            ha="right",
        )
        ax.set_xlabel("Dataset")
    else:
        ax.set_xticklabels([])
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if nonnegative_axis:
        ax.set_ylim(bottom=0.0)
    if plotted_labels:
        ax.legend(
            fontsize=8,
            ncol=min(2, len(plotted_labels)),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            borderaxespad=0.35,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Metric unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#666666",
        )
    return plotted_labels


def plot_all(
    summary_rows: Sequence[Mapping[str, object]],
    out_dir: Path,
    algorithms: Sequence[str],
    datasets_per_figure: int,
    max_scale_ratio: float,
    input_files: Sequence[Path],
) -> Tuple[List[Path], List[Dict[str, object]]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"matplotlib import failed: {exc}") from exc

    runtime_series = (
        (
            "1 GPU basic: internal total",
            "single_basic_internal_total_ms",
            "#4C78A8",
        ),
        (
            "1 GPU tolerance: main + checker drain",
            "single_queue_internal_total_ms",
            "#F58518",
        ),
        (
            "2 GPU basic: mean-rank internal total",
            "two_gpu_basic_rank_internal_total_avg_ms",
            "#54A24B",
        ),
        (
            "2 GPU tolerance: mean-rank main + drain + post-check",
            "two_gpu_tolerance_rank_internal_total_avg_ms",
            "#E45756",
        ),
    )
    kernel_absolute_series = (
        ("1 GPU basic kernel", "single_basic_graph_kernel_ms", "#4C78A8"),
        ("1 GPU tolerance dual kernel", "single_queue_graph_kernel_ms", "#F58518"),
        (
            "2 GPU basic kernel, rank mean",
            "two_gpu_basic_rank_graph_kernel_avg_ms",
            "#54A24B",
        ),
        (
            "2 GPU tolerance dual kernel, rank mean",
            "two_gpu_tolerance_rank_graph_kernel_avg_ms",
            "#E45756",
        ),
    )
    kernel_overhead_series = (
        (
            "1 GPU tolerance dual vs basic",
            "single_queue_graph_kernel_overhead_pct",
            "#F58518",
        ),
        (
            "2 GPU tolerance dual vs basic, rank mean",
            "two_gpu_tolerance_rank_graph_kernel_overhead_pct",
            "#E45756",
        ),
    )
    detail_panels = (
        (
            "Broader non-NCCL CUDA-path overhead within the same GPU mode",
            "Paired overhead (%)",
            (
                (
                    "1 GPU CUDA main interval",
                    "single_queue_gpu_compute_overhead_pct",
                    "#F58518",
                ),
                (
                    "2 GPU non-NCCL CUDA work, rank mean",
                    "two_gpu_tolerance_rank_gpu_compute_overhead_pct",
                    "#E45756",
                ),
            ),
            False,
        ),
        (
            "Unhidden asynchronous CPU-check drain",
            "Drain after main loop (ms)",
            (
                (
                    "1 GPU tolerance",
                    "single_queue_cpu_drain_ms",
                    "#F58518",
                ),
                (
                    "2 GPU tolerance, rank mean",
                    "two_gpu_tolerance_rank_cpu_drain_avg_ms",
                    "#E45756",
                ),
            ),
            True,
        ),
        (
            "Two-GPU post-convergence detection aggregation",
            "Rank-mean time (ms)",
            (
                (
                    "Post-check total",
                    "two_gpu_tolerance_rank_postcheck_total_avg_ms",
                    "#B279A2",
                ),
                (
                    "MPI inside post-check (subset)",
                    "two_gpu_tolerance_rank_postcheck_mpi_avg_ms",
                    "#9C755F",
                ),
            ),
            True,
        ),
        (
            "Two-GPU communication and synchronization calls",
            "Rank-mean component time (ms)",
            (
                (
                    "Basic NCCL exchange",
                    "two_gpu_basic_rank_nccl_exchange_avg_ms",
                    "#59A14F",
                ),
                (
                    "Basic main MPI sync/wait",
                    "two_gpu_basic_rank_mpi_sync_avg_ms",
                    "#76B7B2",
                ),
                (
                    "Tolerance NCCL exchange",
                    "two_gpu_tolerance_rank_nccl_exchange_avg_ms",
                    "#E15759",
                ),
                (
                    "Tolerance main MPI sync/wait",
                    "two_gpu_tolerance_rank_mpi_sync_avg_ms",
                    "#FF9DA7",
                ),
                (
                    "Tolerance post-check MPI",
                    "two_gpu_tolerance_rank_postcheck_mpi_avg_ms",
                    "#9C755F",
                ),
            ),
            True,
        ),
        (
            "Two-GPU main-loop host/control residual",
            "main - CUDA - NCCL - main MPI (ms)",
            (
                (
                    "2 GPU basic",
                    "two_gpu_basic_rank_main_residual_avg_ms",
                    "#54A24B",
                ),
                (
                    "2 GPU tolerance",
                    "two_gpu_tolerance_rank_main_residual_avg_ms",
                    "#E45756",
                ),
            ),
            True,
        ),
        (
            "Two-GPU rank internal-total imbalance",
            "max / mean - 1 (%)",
            (
                (
                    "2 GPU basic",
                    "two_gpu_basic_rank_total_imbalance_pct",
                    "#54A24B",
                ),
                (
                    "2 GPU tolerance",
                    "two_gpu_tolerance_rank_total_imbalance_pct",
                    "#E45756",
                ),
            ),
            True,
        ),
    )

    figures: List[Path] = []
    manifest_rows: List[Dict[str, object]] = []
    for algorithm in algorithms:
        algorithm_rows = [
            row for row in summary_rows
            if str(row.get("algorithm", "")) == algorithm
        ]
        groups = group_by_runtime_scale(
            algorithm_rows,
            datasets_per_figure=datasets_per_figure,
            max_scale_ratio=max_scale_ratio,
        )
        for group_index, group in enumerate(groups, start=1):
            scores = [
                parse_number(row.get("scale_score_ms"))
                for row in group
            ]
            finite_scores = [score for score in scores if score is not None]
            score_min = min(finite_scores)
            score_max = max(finite_scores)
            scale_ratio = (
                score_max / score_min
                if score_min > 0.0
                else (1.0 if score_max == 0.0 else math.inf)
            )
            figure_width = max(
                10.0,
                min(4.5 + 1.35 * len(group), 22.0),
            )
            group_note = (
                f"group {group_index}/{len(groups)}; "
                f"runtime scale {score_min:.3g}-{score_max:.3g} ms"
            )
            group_has_iteration_mismatch = any(
                has_iteration_mismatch(row) for row in group
            )

            runtime_fig, runtime_ax = plt.subplots(
                figsize=(figure_width, 6.5)
            )
            runtime_labels = draw_grouped_bars(
                runtime_ax,
                group,
                runtime_series,
                "Internal algorithm time (ms)",
                "",
                reference=0.0,
                nonnegative_axis=True,
                show_x_labels=True,
            )
            runtime_ax.set_title(
                f"{algorithm.upper()} — internal algorithm time by dataset\n"
                f"{group_note}; bars show mean ± sample SD"
            )
            if group_has_iteration_mismatch:
                runtime_fig.text(
                    0.01,
                    0.01,
                    ITERATION_MISMATCH_NOTE,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
                runtime_fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
            else:
                runtime_fig.tight_layout()
            runtime_path = out_dir / (
                f"{algorithm}_runtime_group{group_index:02d}.png"
            )
            runtime_fig.savefig(runtime_path, dpi=220)
            plt.close(runtime_fig)
            figures.append(runtime_path)

            detail_fig, detail_axes = plt.subplots(
                nrows=len(detail_panels),
                ncols=1,
                sharex=True,
                figsize=(figure_width, 19.0),
            )
            panel_labels: List[str] = []
            for panel_index, (
                panel_title,
                ylabel,
                panel_series,
                nonnegative,
            ) in enumerate(detail_panels):
                labels = draw_grouped_bars(
                    detail_axes[panel_index],
                    group,
                    panel_series,
                    ylabel,
                    panel_title,
                    reference=0.0,
                    nonnegative_axis=nonnegative,
                    show_x_labels=panel_index == len(detail_panels) - 1,
                )
                panel_labels.extend(labels)
            detail_fig.suptitle(
                f"{algorithm.upper()} — explainable overhead details\n"
                f"{group_note}",
                fontsize=13,
            )
            if group_has_iteration_mismatch:
                detail_fig.text(
                    0.01,
                    0.01,
                    ITERATION_MISMATCH_NOTE,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
                detail_fig.tight_layout(
                    rect=(0.0, 0.06, 1.0, 0.965)
                )
            else:
                detail_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
            detail_path = out_dir / (
                f"{algorithm}_overhead_details_group{group_index:02d}.png"
            )
            detail_fig.savefig(detail_path, dpi=220)
            plt.close(detail_fig)
            figures.append(detail_path)

            kernel_fig, kernel_axes = plt.subplots(
                nrows=2,
                ncols=1,
                sharex=True,
                figsize=(figure_width, 10.5),
            )
            draw_grouped_bars(
                kernel_axes[0],
                group,
                kernel_absolute_series,
                "Accumulated kernel time (ms)",
                "Core graph-algorithm kernel only",
                reference=0.0,
                nonnegative_axis=True,
                show_x_labels=False,
            )
            draw_grouped_bars(
                kernel_axes[1],
                group,
                kernel_overhead_series,
                "Paired overhead (%)",
                "Tolerance redundant kernel vs matching basic kernel",
                reference=0.0,
                nonnegative_axis=False,
                show_x_labels=True,
            )
            kernel_fig.suptitle(
                f"{algorithm.upper()} — core graph-kernel comparison\n"
                f"{group_note}; CUDA-event time, mean ± sample SD",
                fontsize=13,
            )
            if group_has_iteration_mismatch:
                kernel_fig.text(
                    0.01,
                    0.01,
                    ITERATION_MISMATCH_NOTE,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
                kernel_fig.tight_layout(
                    rect=(0.0, 0.10, 1.0, 0.95)
                )
            else:
                kernel_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            kernel_path = out_dir / (
                f"{algorithm}_graph_kernel_group{group_index:02d}.png"
            )
            kernel_fig.savefig(kernel_path, dpi=220)
            plt.close(kernel_fig)
            figures.append(kernel_path)

            manifest_rows.append(
                {
                    "algorithm": algorithm,
                    "group": group_index,
                    "dataset_count": len(group),
                    "scale_score_min_ms": score_min,
                    "scale_score_max_ms": score_max,
                    "scale_ratio": scale_ratio,
                    "datasets": ";".join(
                        str(row.get("dataset", "")) for row in group
                    ),
                    "edges": ";".join(
                        str(row.get("edges", "")) for row in group
                    ),
                    "runtime_figure": runtime_path.name,
                    "detail_figure": detail_path.name,
                    "kernel_figure": kernel_path.name,
                    "runtime_series": ";".join(runtime_labels),
                    "detail_panels": ";".join(
                        panel[0] for panel in detail_panels
                    ),
                    "kernel_panels": (
                        "core graph-kernel absolute time;"
                        "paired tolerance-kernel overhead"
                    ),
                    "input_files": ";".join(
                        str(path) for path in input_files
                    ),
                }
            )

    return figures, manifest_rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired runtime, graph-kernel, and overhead-detail plots from "
            "one or more GraphAlgorithms results.csv files."
        )
    )
    parser.add_argument(
        "results_csv",
        nargs="+",
        type=Path,
        help="One or more outputs/experiments/.../results.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; default: parent of the first results.csv",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
    )
    parser.add_argument(
        "--datasets-per-figure",
        type=int,
        default=8,
        help="Maximum datasets per runtime-scale group",
    )
    parser.add_argument(
        "--max-scale-ratio",
        type=float,
        default=4.0,
        help=(
            "Maximum largest/smallest runtime score in one figure "
            "(default: 4)"
        ),
    )
    args = parser.parse_args(argv)
    if args.datasets_per_figure < 1:
        parser.error("--datasets-per-figure must be at least 1")
    if args.max_scale_ratio < 1.0:
        parser.error("--max-scale-ratio must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir or args.results_csv[0].parent
    try:
        rows = load_success_rows(args.results_csv)
        if not rows:
            raise RuntimeError("no rows with returncode=0")
        pair_index = build_pair_index(rows)
        summary_rows = build_summary_rows(pair_index)
        if not summary_rows:
            raise RuntimeError(
                "no complete four-method, two-GPU repeat pairs"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
        figures, manifest_rows = plot_all(
            summary_rows=summary_rows,
            out_dir=out_dir,
            algorithms=args.algorithms,
            datasets_per_figure=args.datasets_per_figure,
            max_scale_ratio=args.max_scale_ratio,
            input_files=args.results_csv,
        )
        write_csv(
            out_dir / "plot_manifest.csv",
            MANIFEST_FIELDS,
            manifest_rows,
        )
    except (OSError, RuntimeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not figures:
        print(
            "warning: no runtime figures generated; real rank-average timing "
            "fields may be absent",
            file=sys.stderr,
        )
    print(f"summary: {out_dir / 'summary.csv'}")
    print(f"plot manifest: {out_dir / 'plot_manifest.csv'}")
    print(f"generated {len(figures)} figures")
    for figure in figures:
        print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
