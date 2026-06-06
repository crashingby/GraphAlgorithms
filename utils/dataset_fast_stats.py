#!/usr/bin/env python3
"""
Fast structural statistics for dataset/*.mtx.

This intentionally skips triangle-dependent metrics so it can run on very large
graphs such as soc-LiveJournal1. It computes degree-based metrics in streaming
passes over MatrixMarket files.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
DEFAULT_OUTPUT = ROOT / "temp_downloads" / "dataset_stats_summary.md"


def iter_mtx_edges(path: Path):
    header = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if header is None:
                if len(parts) < 3:
                    raise ValueError(f"invalid MatrixMarket size line in {path}: {s}")
                rows, cols, nnz = map(int, parts[:3])
                header = (max(rows, cols), nnz)
                continue
            if len(parts) >= 2:
                yield int(parts[0]), int(parts[1])


def read_header_and_base(path: Path) -> tuple[int, bool]:
    n = 0
    min_id = None
    max_id = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if n == 0:
                rows, cols, _ = map(int, parts[:3])
                n = max(rows, cols)
                continue
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            min_id = u if min_id is None else min(min_id, u, v)
            max_id = u if max_id is None else max(max_id, u, v)

    one_based = bool(min_id is not None and min_id >= 1 and max_id is not None and max_id <= n)
    return n, one_based


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def compute_stats(path: Path) -> dict[str, str]:
    n, one_based = read_header_and_base(path)
    degrees = [0] * n
    edge_count = 0

    for u, v in iter_mtx_edges(path):
        if one_based:
            u -= 1
            v -= 1
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        degrees[u] += 1
        degrees[v] += 1
        edge_count += 1

    dmax = max(degrees) if degrees else 0
    davg = (2.0 * edge_count / n) if n > 0 else float("nan")
    source = max(range(n), key=lambda v: (degrees[v], -v)) if n > 0 else -1

    sum_jk = 0.0
    sum_half_j_plus_k = 0.0
    sum_half_j2_plus_k2 = 0.0
    assort_edges = 0

    for u, v in iter_mtx_edges(path):
        if one_based:
            u -= 1
            v -= 1
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        du = degrees[u]
        dv = degrees[v]
        sum_jk += du * dv
        sum_half_j_plus_k += 0.5 * (du + dv)
        sum_half_j2_plus_k2 += 0.5 * (du * du + dv * dv)
        assort_edges += 1

    if assort_edges == 0:
        assort = float("nan")
    else:
        mean_prod = sum_jk / assort_edges
        mean_sum = sum_half_j_plus_k / assort_edges
        mean_sq_sum = sum_half_j2_plus_k2 / assort_edges
        denom = mean_sq_sum - mean_sum * mean_sum
        assort = float("nan") if denom == 0.0 else (mean_prod - mean_sum * mean_sum) / denom

    return {
        "dataset": path.stem,
        "|V|": fmt(n),
        "|E|": fmt(edge_count),
        "dmax": fmt(dmax),
        "davg": fmt(davg),
        "r": fmt(assort),
        "|T|": "skipped",
        "|T|avg": "skipped",
        "|T|max": "skipped",
        "κavg": "skipped",
        "κ": "skipped",
        "Kmax": "skipped",
        "ωlb": "skipped",
        "source": fmt(source),
        "source_degree": fmt(degrees[source] if source >= 0 else 0),
    }


def md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    headers = [
        "dataset",
        "|V|",
        "|E|",
        "dmax",
        "davg",
        "r",
        "|T|",
        "|T|avg",
        "|T|max",
        "κavg",
        "κ",
        "Kmax",
        "ωlb",
        "source",
        "source_degree",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("# Dataset Statistics Summary\n\n")
        f.write("Triangle-dependent metrics are skipped for large-graph practicality.\n\n")
        f.write("| " + " | ".join(md_cell(h) for h in headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(md_cell(row.get(h, "")) for h in headers) + " |\n")


def dataset_paths(names: list[str]) -> list[Path]:
    if names:
        return [DATASET_DIR / f"{name}.mtx" for name in names]
    return sorted(DATASET_DIR.glob("*.mtx"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast stats for dataset/*.mtx without triangle counting.")
    parser.add_argument("datasets", nargs="*", help="dataset names without .mtx; default: all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    for path in dataset_paths(args.datasets):
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[stats] {path.stem}", flush=True)
        rows.append(compute_stats(path))
    write_summary(rows, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
