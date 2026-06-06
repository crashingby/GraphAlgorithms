#!/usr/bin/env python3
"""Plot GraphAlgorithm benchmark results from results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ALGORITHMS = ("bfs", "cc", "kcore", "pagerank")
METHODS = ("basic", "tolerance_queue", "multigpu_basic", "multigpu")

METHOD_LABELS = {
    "basic": "basic",
    "tolerance_queue": "tolerance_queue",
    "multigpu_basic": "multigpu_basic",
    "multigpu": "multigpu",
}

COLORS = {
    "basic": "#4C78A8",
    "tolerance_queue": "#F58518",
    "multigpu_basic": "#54A24B",
    "multigpu": "#E45756",
}


def load_success_rows(results_csv: Path) -> List[Dict[str, str]]:
    with results_csv.open("r", encoding="utf-8") as f:
        return [
            row for row in csv.DictReader(f)
            if row.get("returncode") == "0" and row.get("gpu_time_ms")
        ]


def aggregate(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str], float]:
    buckets: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        key = (row["dataset"], row["algorithm"], row["method"])
        try:
            value = float(row["gpu_time_ms"])
        except (TypeError, ValueError):
            continue
        buckets.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in buckets.items()}


def unique_nonempty(rows: Iterable[Dict[str, str]], key: str) -> List[str]:
    return sorted({row.get(key, "") for row in rows if row.get(key) not in {"", None}})


def format_param(name: str, values: List[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{name}={values[0]}"
    return f"{name}={','.join(values)}"


def algorithm_note(algorithm: str, rows: List[Dict[str, str]]) -> str:
    notes: List[str] = []
    if algorithm == "kcore":
        item = format_param("k", unique_nonempty(rows, "k"))
        if item:
            notes.append(item)

    tolerance_items = [
        format_param("alpha", unique_nonempty(rows, "alpha")),
        format_param("beta", unique_nonempty(rows, "beta")),
        format_param("threshold", unique_nonempty(rows, "threshold")),
    ]
    tolerance_items = [item for item in tolerance_items if item]
    if tolerance_items:
        notes.append("tolerance: " + ", ".join(tolerance_items))

    return " | ".join(notes)


def dataset_label(algorithm: str, dataset: str, rows: List[Dict[str, str]]) -> str:
    if algorithm == "bfs":
        src_values = sorted({
            row.get("bfs_src", "")
            for row in rows
            if row.get("dataset") == dataset and row.get("bfs_src")
        })
        if src_values:
            return f"{dataset}\nsrc={src_values[0]}"
    return dataset


def group_datasets_by_scale(
    datasets: Sequence[str],
    scores: Dict[str, float],
    datasets_per_figure: int,
) -> List[List[str]]:
    ordered = sorted(datasets, key=lambda item: (scores.get(item, 0.0), item))
    return [
        ordered[i:i + datasets_per_figure]
        for i in range(0, len(ordered), datasets_per_figure)
    ]


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = ["algorithm", "group", "score_min_ms", "score_max_ms", "figure", "datasets"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    results_csv: Path,
    out_dir: Optional[Path],
    algorithms: Sequence[str],
    methods: Sequence[str],
    datasets_per_figure: int,
    log_y: bool,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on env
        raise RuntimeError(f"matplotlib import failed: {exc}") from exc

    out_dir = out_dir or results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_success_rows(results_csv)
    if not rows:
        raise RuntimeError("no successful rows with GPU time")

    agg = aggregate(rows)
    figures: List[Path] = []
    manifest_rows: List[Dict[str, object]] = []

    for algorithm in algorithms:
        algorithm_rows = [row for row in rows if row.get("algorithm") == algorithm]
        datasets = sorted({row["dataset"] for row in algorithm_rows})
        if not datasets:
            continue

        score_by_dataset: Dict[str, float] = {}
        for dataset in datasets:
            values = [
                agg.get((dataset, algorithm, method), 0.0)
                for method in methods
            ]
            score_by_dataset[dataset] = max(values) if values else 0.0

        groups = group_datasets_by_scale(datasets, score_by_dataset, datasets_per_figure)
        for group_index, group_datasets in enumerate(groups, start=1):
            width = min(0.18, 0.78 / max(len(methods), 1))
            x = list(range(len(group_datasets)))
            fig_w = max(11, min(1.15 * len(group_datasets) + 4, 22))
            fig, ax = plt.subplots(figsize=(fig_w, 6.6))

            max_value = 0.0
            method_mid = (len(methods) - 1) / 2.0
            for mi, method in enumerate(methods):
                values = [agg.get((dataset, algorithm, method), 0.0) for dataset in group_datasets]
                max_value = max(max_value, max(values) if values else 0.0)
                offsets = [pos + (mi - method_mid) * width for pos in x]
                bars = ax.bar(
                    offsets,
                    values,
                    width=width,
                    label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method),
                )
                for bar, value in zip(bars, values):
                    if value <= 0:
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

            note = algorithm_note(algorithm, algorithm_rows)
            group_scores = [score_by_dataset[dataset] for dataset in group_datasets]
            scale_note = f"group {group_index}/{len(groups)}, max-method range: {min(group_scores):.2f}-{max(group_scores):.2f} ms"
            title = f"{algorithm.upper()} GPU Time by Dataset\n{scale_note}"
            if note:
                title += f" | {note}"
            ax.set_title(title)
            ax.set_ylabel("GPU time (ms)")
            ax.set_xlabel("Dataset")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [dataset_label(algorithm, dataset, algorithm_rows) for dataset in group_datasets],
                rotation=45,
                ha="right",
            )
            if log_y:
                ax.set_yscale("log")
                ax.set_ylabel("GPU time (ms, log scale)")
            elif max_value > 0:
                ax.set_ylim(0, max_value * 1.22)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(ncol=2)
            fig.tight_layout()

            figure = out_dir / f"{algorithm}_gpu_time_group{group_index:02d}.png"
            fig.savefig(figure, dpi=220)
            plt.close(fig)
            figures.append(figure)

            manifest_rows.append({
                "algorithm": algorithm,
                "group": group_index,
                "score_min_ms": f"{min(group_scores):.6f}",
                "score_max_ms": f"{max(group_scores):.6f}",
                "figure": str(figure.relative_to(out_dir)),
                "datasets": ";".join(group_datasets),
            })

    write_manifest(out_dir / "plot_manifest.csv", manifest_rows)
    return figures


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GraphAlgorithm benchmark results from results.csv.")
    parser.add_argument("results_csv", type=Path, help="Path to outputs/experiments/<timestamp>/results.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for figures. Default: results.csv parent")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets-per-figure", type=int, default=8, help="Datasets per grouped figure after sorting by max GPU time")
    parser.add_argument("--log-y", action="store_true", help="Use log scale for y-axis")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    figures = plot_results(
        results_csv=args.results_csv,
        out_dir=args.out_dir,
        algorithms=args.algorithms,
        methods=args.methods,
        datasets_per_figure=max(1, args.datasets_per_figure),
        log_y=args.log_y,
    )
    print(f"generated {len(figures)} figures")
    for figure in figures:
        print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
