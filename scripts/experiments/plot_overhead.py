#!/usr/bin/env python3
"""Draw clear runtime, paired-overhead and non-additive component figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, csv_dump_atomic


METHODS: Tuple[str, ...] = (
    "basic", "tolerance_queue", "multigpu_basic", "multigpu",
)
METHOD_LABELS = {
    "basic": "1 GPU basic",
    "tolerance_queue": "1 GPU tolerance",
}
METHOD_COLORS = {
    "basic": "#4C78A8",
    "tolerance_queue": "#F58518",
    "multigpu_basic": "#54A24B",
    "multigpu": "#E45756",
}
PAIR_COLORS = {"1gpu": "#F58518", "multigpu": "#E45756"}


def multi_gpu_label(mpi_ranks: int) -> str:
    """Return the human-readable resource label for this immutable run."""
    return f"{mpi_ranks} GPU"


def method_label(method: str, mpi_ranks: int) -> str:
    """Label a benchmark method using the manifest-selected GPU count."""
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if method == "multigpu_basic":
        return f"{multi_gpu_label(mpi_ranks)} basic"
    if method == "multigpu":
        return f"{multi_gpu_label(mpi_ranks)} tolerance"
    return method


def pair_label(gpu_mode: str, mpi_ranks: int) -> str:
    """Label one paired comparison using the manifest-selected GPU count."""
    if gpu_mode == "1gpu":
        return "1 GPU tolerance vs basic"
    return f"{multi_gpu_label(mpi_ranks)} tolerance vs basic"


def resolve_path(path: Path) -> Path:
    """Interpret paths relative to the repository root."""
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Load one analysis table."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def number(row: Mapping[str, Any], field: str) -> Optional[float]:
    """Read a finite numerical field."""
    try:
        value = float(str(row.get(field)))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def stats(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    """Compute median and IQR for plotting repeat distributions."""
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    if len(clean) == 1:
        return {"n": 1, "median": clean[0], "p25": clean[0], "p75": clean[0]}
    def percentile(fraction: float) -> float:
        position = (len(clean) - 1) * fraction
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        return clean[lo] + (clean[hi] - clean[lo]) * (position - lo)
    return {
        "n": len(clean), "median": statistics.median(clean),
        "p25": percentile(0.25), "p75": percentile(0.75),
    }


def asymmetric_errors(stat: Mapping[str, Optional[float]]) -> Optional[List[List[float]]]:
    """Return Matplotlib asymmetric IQR errors, or no error when absent."""
    median = stat["median"]
    p25 = stat["p25"]
    p75 = stat["p75"]
    if median is None or p25 is None or p75 is None:
        return None
    return [[max(0.0, median - p25)], [max(0.0, p75 - median)]]


def method_index(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str, str], Mapping[str, str]]:
    """Index one method summary row per algorithm/dataset/method."""
    result: Dict[Tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (row.get("algorithm", ""), row.get("dataset", ""), row.get("method", ""))
        result[key] = row
    return result


def grouped_pairs(
    rows: Sequence[Mapping[str, str]],
) -> Dict[Tuple[str, str, str], List[Mapping[str, str]]]:
    """Group pair-level samples by algorithm, dataset and resource mode."""
    result: Dict[Tuple[str, str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        key = (row.get("algorithm", ""), row.get("dataset", ""), row.get("gpu_mode", ""))
        result.setdefault(key, []).append(row)
    return result


def mismatch_datasets(rows: Sequence[Mapping[str, str]]) -> set[Tuple[str, str]]:
    """Mark any data set with an iteration mismatch in either resource pair."""
    return {
        (row.get("algorithm", ""), row.get("dataset", ""))
        for row in rows
        if row.get("pair_status") == "iteration_mismatch"
    }


def expected_datasets(run_dir: Path) -> Dict[str, List[str]]:
    """Read the selected dataset order from the immutable manifest."""
    import json
    with (run_dir / "manifest" / "experiment.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    config = manifest["config"]
    return {algorithm: list(config["datasets"]) for algorithm in config["algorithms"]}


def configured_mpi_ranks(run_dir: Path) -> int:
    """Read the immutable multi-GPU rank count selected for this run."""
    import json
    with (run_dir / "manifest" / "experiment.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    return int(manifest["config"]["mpi_ranks"])


def runtime_scale(
    methods: Mapping[Tuple[str, str, str], Mapping[str, str]],
    algorithm: str,
    dataset: str,
) -> float:
    """Choose a baseline completion scale for time-compatible figure grouping."""
    for method in ("basic", "multigpu_basic", "tolerance_queue", "multigpu"):
        row = methods.get((algorithm, dataset, method))
        value = number(row, "completion_internal_ms_median") if row else None
        if value is not None and value > 0.0:
            return value
    return float("inf")


def group_datasets(
    datasets: Sequence[str],
    methods: Mapping[Tuple[str, str, str], Mapping[str, str]],
    algorithm: str,
    max_datasets: int,
    max_scale_ratio: float,
) -> List[List[str]]:
    """Greedily keep absolute-runtime groups readable and comparable in scale."""
    ordered = sorted(datasets, key=lambda dataset: runtime_scale(methods, algorithm, dataset))
    groups: List[List[str]] = []
    current: List[str] = []
    current_min: Optional[float] = None
    for dataset in ordered:
        scale = runtime_scale(methods, algorithm, dataset)
        would_exceed_count = len(current) >= max_datasets
        would_exceed_scale = (
            current_min is not None and math.isfinite(scale) and current_min > 0.0
            and scale / current_min > max_scale_ratio
        )
        if current and (would_exceed_count or would_exceed_scale):
            groups.append(current)
            current = []
            current_min = None
        current.append(dataset)
        if math.isfinite(scale) and scale > 0.0:
            current_min = scale if current_min is None else min(current_min, scale)
    if current:
        groups.append(current)
    return groups


def dataset_labels(
    algorithm: str, datasets: Sequence[str], mismatch: set[Tuple[str, str]],
) -> List[str]:
    """Add an explicit dagger only where iteration pairing was imperfect."""
    return [
        dataset + ("†" if (algorithm, dataset) in mismatch else "")
        for dataset in datasets
    ]


def save_figure(
    figure: plt.Figure,
    path_base: Path,
    formats: Sequence[str],
    manifest: List[Dict[str, Any]],
    *,
    figure_type: str,
    algorithm: str,
    group_index: int,
    datasets: Sequence[str],
) -> None:
    """Persist a figure in requested formats and append an audit manifest row."""
    path_base.parent.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for extension in formats:
        path = path_base.with_suffix("." + extension)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        written.append(str(path.name))
    plt.close(figure)
    manifest.append({
        "figure_type": figure_type, "algorithm": algorithm,
        "group_index": group_index, "datasets": ",".join(datasets),
        "files": ",".join(written),
    })


def bar_values(
    dataset_list: Sequence[str],
    source: Mapping[Tuple[str, str, str], Mapping[str, str]],
    algorithm: str,
    method: str,
    metric_prefix: str,
) -> List[Dict[str, Optional[float]]]:
    """Read precomputed median/IQR statistics from method summaries."""
    result: List[Dict[str, Optional[float]]] = []
    for dataset in dataset_list:
        row = source.get((algorithm, dataset, method))
        result.append({
            "n": number(row, f"{metric_prefix}_n") if row else 0,
            "median": number(row, f"{metric_prefix}_median") if row else None,
            "p25": number(row, f"{metric_prefix}_p25") if row else None,
            "p75": number(row, f"{metric_prefix}_p75") if row else None,
        })
    return result


def draw_method_bars(
    ax: plt.Axes,
    datasets: Sequence[str],
    source: Mapping[Tuple[str, str, str], Mapping[str, str]],
    algorithm: str,
    mismatch: set[Tuple[str, str]],
    mpi_ranks: int,
) -> None:
    """Draw the requested four-bar absolute internal completion-time overview."""
    x = np.arange(len(datasets), dtype=float)
    width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for method, offset in zip(METHODS, offsets):
        values = bar_values(datasets, source, algorithm, method, "completion_internal_ms")
        heights = [item["median"] if item["median"] is not None else np.nan for item in values]
        lower = [
            item["median"] - item["p25"]
            if item["median"] is not None and item["p25"] is not None else 0.0
            for item in values
        ]
        upper = [
            item["p75"] - item["median"]
            if item["median"] is not None and item["p75"] is not None else 0.0
            for item in values
        ]
        ax.bar(
            x + offset * width, heights, width=width, color=METHOD_COLORS[method],
            label=method_label(method, mpi_ranks), yerr=np.array([lower, upper]),
            capsize=2.5, error_kw={"elinewidth": 0.8},
        )
    ax.set_xticks(x, dataset_labels(algorithm, datasets, mismatch), rotation=35, ha="right")
    ax.set_ylabel("Internal completion time (ms)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=8, loc="upper left")


def paired_stat(
    pairs: Mapping[Tuple[str, str, str], Sequence[Mapping[str, str]]],
    algorithm: str,
    dataset: str,
    gpu_mode: str,
    metric_name: str,
    *,
    eligible_only: bool,
) -> Dict[str, Optional[float]]:
    """Compute a repeat-level median/IQR from the explicitly retained pair rows."""
    values = pairs.get((algorithm, dataset, gpu_mode), [])
    if eligible_only:
        values = [row for row in values if row.get("overhead_eligible") == "1"]
    else:
        values = [row for row in values if row.get("pair_complete_valid") == "1"]
    return stats(number(row, metric_name) for row in values)


def draw_pair_bars(
    ax: plt.Axes,
    datasets: Sequence[str],
    pairs: Mapping[Tuple[str, str, str], Sequence[Mapping[str, str]]],
    algorithm: str,
    metric_name: str,
    ylabel: str,
    mpi_ranks: int,
) -> None:
    """Draw two strict paired-overhead series and annotate the eligible sample n."""
    x = np.arange(len(datasets), dtype=float)
    width = 0.32
    for gpu_mode, offset in (("1gpu", -0.5), ("multigpu", 0.5)):
        values = [
            paired_stat(pairs, algorithm, dataset, gpu_mode, metric_name, eligible_only=True)
            for dataset in datasets
        ]
        heights = [item["median"] if item["median"] is not None else np.nan for item in values]
        lower = [
            item["median"] - item["p25"]
            if item["median"] is not None and item["p25"] is not None else 0.0
            for item in values
        ]
        upper = [
            item["p75"] - item["median"]
            if item["median"] is not None and item["p75"] is not None else 0.0
            for item in values
        ]
        bars = ax.bar(
            x + offset * width, heights, width=width, color=PAIR_COLORS[gpu_mode],
            label=pair_label(gpu_mode, mpi_ranks), yerr=np.array([lower, upper]),
            capsize=2.5, error_kw={"elinewidth": 0.8},
        )
        for bar, item in zip(bars, values):
            if item["n"] and item["median"] is not None:
                offset_text = 2 if item["median"] >= 0 else -10
                ax.annotate(
                    f"n={int(item['n'])}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, offset_text), textcoords="offset points",
                    ha="center", va="bottom" if offset_text > 0 else "top", fontsize=6.5,
                )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left")


def draw_component_communication(
    ax: plt.Axes,
    datasets: Sequence[str],
    pairs: Mapping[Tuple[str, str, str], Sequence[Mapping[str, str]]],
    algorithm: str,
    mpi_ranks: int,
) -> None:
    """Draw multi-GPU rank-mean communication calls without adding them up."""
    fields = (
        ("baseline_rank_mean_nccl_exchange_ms", "Basic NCCL", "#54A24B"),
        ("tolerance_rank_mean_nccl_exchange_ms", "Tolerance NCCL", "#E45756"),
        ("baseline_rank_mean_mpi_sync_ms", "Basic MPI sync/wait", "#72B7B2"),
        ("tolerance_rank_mean_mpi_sync_ms", "Tolerance MPI sync/wait", "#FF9DA6"),
    )
    x = np.arange(len(datasets), dtype=float)
    width = 0.18
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for (field, label, color), offset in zip(fields, offsets):
        values = [
            paired_stat(pairs, algorithm, dataset, "multigpu", field, eligible_only=True)
            for dataset in datasets
        ]
        ax.bar(
            x + offset * width,
            [item["median"] if item["median"] is not None else np.nan for item in values],
            width=width, color=color, label=label,
        )
    ax.set_ylabel("Per-rank mean call time (ms)")
    ax.set_title(f"{multi_gpu_label(mpi_ranks)} communication calls (non-additive components)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=7, loc="upper left")


def draw_tolerance_tail(
    ax: plt.Axes,
    datasets: Sequence[str],
    methods: Mapping[Tuple[str, str, str], Mapping[str, str]],
    algorithm: str,
    mpi_ranks: int,
) -> None:
    """Draw tolerance-only unhidden checker work as absolute time."""
    series = (
        ("tolerance_queue", "rank_mean_cpu_check_drain_ms", "1 GPU drain", "#F58518"),
        ("multigpu", "rank_mean_cpu_check_drain_ms", f"{multi_gpu_label(mpi_ranks)} drain", "#E45756"),
        ("multigpu", "rank_mean_postcheck_total_ms", f"{multi_gpu_label(mpi_ranks)} post-check", "#B279A2"),
    )
    x = np.arange(len(datasets), dtype=float)
    width = 0.23
    offsets = (-1.0, 0.0, 1.0)
    for (method, metric, label, color), offset in zip(series, offsets):
        values = bar_values(datasets, methods, algorithm, method, metric)
        ax.bar(
            x + offset * width,
            [item["median"] if item["median"] is not None else np.nan for item in values],
            width=width, color=color, label=label,
        )
    ax.set_ylabel("Time (ms)")
    ax.set_title("Tolerance-only CPU tail (not a stacked total)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper left")


def draw_imbalance(
    ax: plt.Axes,
    datasets: Sequence[str],
    pairs: Mapping[Tuple[str, str, str], Sequence[Mapping[str, str]]],
    algorithm: str,
    mpi_ranks: int,
) -> None:
    """Draw rank max/mean imbalance to explain a slow distributed completion."""
    x = np.arange(len(datasets), dtype=float)
    width = 0.32
    for field, label, color, offset in (
        ("rank_imbalance_basic_pct", f"{multi_gpu_label(mpi_ranks)} basic", "#54A24B", -0.5),
        ("rank_imbalance_tolerance_pct", f"{multi_gpu_label(mpi_ranks)} tolerance", "#E45756", 0.5),
    ):
        values = [
            paired_stat(pairs, algorithm, dataset, "multigpu", field, eligible_only=True)
            for dataset in datasets
        ]
        ax.bar(
            x + offset * width,
            [item["median"] if item["median"] is not None else np.nan for item in values],
            width=width, color=color, label=label,
        )
    ax.set_ylabel("max / mean − 1 (%)")
    ax.set_title(f"{multi_gpu_label(mpi_ranks)} internal-total rank imbalance", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper left")


def plot_all(
    run_dir: Path,
    *,
    formats: Sequence[str],
    max_datasets: int,
    max_scale_ratio: float,
) -> List[Dict[str, Any]]:
    """Render all per-algorithm figure groups from analysis-only tables."""
    methods = method_index(read_csv(run_dir / "analysis" / "method_summary.csv"))
    pair_rows = read_csv(run_dir / "analysis" / "paired_samples.csv")
    pairs = grouped_pairs(pair_rows)
    selected_by_algorithm = expected_datasets(run_dir)
    mpi_ranks = configured_mpi_ranks(run_dir)
    mismatch = mismatch_datasets(pair_rows)
    figure_manifest: List[Dict[str, Any]] = []

    plt.rcParams.update({
        "figure.dpi": 120, "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    for algorithm, datasets in selected_by_algorithm.items():
        groups = group_datasets(
            datasets, methods, algorithm, max_datasets, max_scale_ratio,
        )
        for group_index, group in enumerate(groups, start=1):
            suffix = f"{algorithm}_group{group_index:02d}"

            figure, axis = plt.subplots(figsize=(max(8.5, len(group) * 1.3), 5.1))
            draw_method_bars(axis, group, methods, algorithm, mismatch, mpi_ranks)
            axis.set_title(
                f"{algorithm.upper()} — internal completion time\n"
                f"median with IQR; {multi_gpu_label(mpi_ranks)} bars use max(rank internal total)",
            )
            figure.text(
                0.01, 0.01,
                "† At least one paired repeat had a different iteration count; "
                "absolute time remains shown, strict paired overhead excludes that repeat.",
                fontsize=7,
            )
            figure.tight_layout(rect=(0, 0.04, 1, 1))
            save_figure(
                figure, run_dir / "figures" / f"{suffix}_runtime", formats, figure_manifest,
                figure_type="runtime_completion", algorithm=algorithm,
                group_index=group_index, datasets=group,
            )

            figure, axis = plt.subplots(figsize=(max(8.5, len(group) * 1.3), 4.8))
            draw_pair_bars(
                axis, group, pairs, algorithm, "total_completion_overhead_pct",
                "Paired completion overhead (%)", mpi_ranks,
            )
            axis.set_xticks(np.arange(len(group)), dataset_labels(algorithm, group, mismatch),
                              rotation=35, ha="right")
            axis.set_title(
                f"{algorithm.upper()} — strict paired tolerance overhead\n"
                "Only successful same-iteration baseline/tolerance repeats; bars are median with IQR",
            )
            figure.tight_layout()
            save_figure(
                figure, run_dir / "figures" / f"{suffix}_paired_overhead", formats, figure_manifest,
                figure_type="paired_total_overhead", algorithm=algorithm,
                group_index=group_index, datasets=group,
            )

            figure, axes = plt.subplots(
                4, 1, figsize=(max(9.0, len(group) * 1.35), 13.0), sharex=True,
            )
            draw_pair_bars(
                axes[0], group, pairs, algorithm, "rank_mean_graph_kernel_overhead_pct",
                "Paired overhead (%)", mpi_ranks,
            )
            axes[0].set_title(
                "Core graph-kernel overhead (per-rank mean; non-additive)", fontsize=10
            )
            draw_component_communication(axes[1], group, pairs, algorithm, mpi_ranks)
            draw_tolerance_tail(axes[2], group, methods, algorithm, mpi_ranks)
            draw_imbalance(axes[3], group, pairs, algorithm, mpi_ranks)
            axes[3].set_xticks(np.arange(len(group)), dataset_labels(algorithm, group, mismatch),
                               rotation=35, ha="right")
            figure.suptitle(
                f"{algorithm.upper()} — explanatory timing components", y=0.997, fontsize=12
            )
            figure.text(
                0.01, 0.005,
                "Components are separate observations, not stackable costs: graph kernel ⊂ GPU compute; "
                "post-check MPI ⊂ post-check total; rank-max components may originate on different ranks.",
                fontsize=7,
            )
            figure.tight_layout(rect=(0, 0.025, 1, 0.985))
            save_figure(
                figure, run_dir / "figures" / f"{suffix}_components", formats, figure_manifest,
                figure_type="components", algorithm=algorithm,
                group_index=group_index, datasets=group,
            )

    csv_dump_atomic(
        run_dir / "figures" / "figure_manifest.csv",
        ("figure_type", "algorithm", "group_index", "datasets", "files"),
        figure_manifest,
    )
    return figure_manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build plot CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--formats", nargs="+", default=["png"], choices=("png", "pdf"),
        help="output formats; default: png",
    )
    parser.add_argument("--max-datasets", type=int, default=8)
    parser.add_argument("--max-scale-ratio", type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.max_datasets < 1:
        parser.error("--max-datasets must be positive")
    if args.max_scale_ratio <= 1.0:
        parser.error("--max-scale-ratio must exceed one")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render analysis figures."""
    args = parse_args(argv)
    try:
        figures = plot_all(
            resolve_path(args.run_dir), formats=args.formats,
            max_datasets=args.max_datasets, max_scale_ratio=args.max_scale_ratio,
        )
    except (OSError, ValueError, csv.Error, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"figure sets written: {len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
