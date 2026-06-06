#!/usr/bin/env python3
"""
Select central-ish source vertices for graph datasets.

The CUDA BFS entry points need a source vertex. For large real graphs, a good
practical default is a high-degree vertex in the largest dense area. This script
computes undirected total degree from dataset/<name>.mtx and records the top
candidate vertices.

Outputs:
  dataset/sources/<dataset>_sources.tsv
  dataset/sources/source_summary.md
"""

from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
SOURCE_DIR = DATASET_DIR / "sources"


def iter_mtx_edge_lines(path: Path):
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
                rows, cols, _ = map(int, parts[:3])
                header = (rows, cols)
                continue
            if len(parts) >= 2:
                yield int(parts[0]), int(parts[1])


def read_mtx_degrees(path: Path) -> tuple[int, list[int], int]:
    header = None
    min_id = None
    max_id = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if header is None:
                if len(parts) < 3:
                    raise ValueError(f"invalid MatrixMarket size line in {path}: {s}")
                rows, cols, _ = map(int, parts[:3])
                header = (rows, cols)
                continue
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            min_id = u if min_id is None else min(min_id, u, v)
            max_id = u if max_id is None else max(max_id, u, v)

    if header is None:
        raise ValueError(f"missing MatrixMarket header: {path}")

    n = max(header)
    one_based = bool(min_id is not None and min_id >= 1 and max_id is not None and max_id <= n)
    degrees = [0] * n
    valid_edges = 0

    for u, v in iter_mtx_edge_lines(path):
        if one_based:
            u -= 1
            v -= 1
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        degrees[u] += 1
        degrees[v] += 1
        valid_edges += 1

    return n, degrees, valid_edges

def select_top_sources(path: Path, top_k: int) -> tuple[int, int, list[tuple[int, int]]]:
    n, degrees, edge_count = read_mtx_degrees(path)
    candidates = heapq.nlargest(
        min(top_k, len(degrees)),
        enumerate(degrees),
        key=lambda item: (item[1], -item[0]),
    )
    return n, edge_count, [(vertex, degree) for vertex, degree in candidates]


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def process_dataset(dataset: str, top_k: int, source_dir: Path) -> dict[str, str]:
    path = DATASET_DIR / f"{dataset}.mtx"
    if not path.exists():
        raise FileNotFoundError(path)

    n, edge_count, candidates = select_top_sources(path, top_k)
    source_dir.mkdir(parents=True, exist_ok=True)
    out_path = source_dir / f"{dataset}_sources.tsv"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("rank\tvertex\tdegree\n")
        for rank, (vertex, degree) in enumerate(candidates, start=1):
            f.write(f"{rank}\t{vertex}\t{degree}\n")

    best_vertex, best_degree = candidates[0] if candidates else (-1, 0)
    avg_degree = (2.0 * edge_count / n) if n > 0 else float("nan")
    return {
        "dataset": dataset,
        "nodes": str(n),
        "edges": str(edge_count),
        "avg_degree": fmt_float(avg_degree),
        "source": str(best_vertex),
        "source_degree": str(best_degree),
        "file": str(out_path.relative_to(ROOT)),
    }


def dataset_names_from_args(names: list[str]) -> list[str]:
    if names:
        return names
    return sorted(
        p.stem for p in DATASET_DIR.glob("*.mtx")
        if p.is_file()
    )


def write_summary(rows: list[dict[str, str]], source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    out = source_dir / "source_summary.md"
    headers = ["dataset", "nodes", "edges", "avg_degree", "source", "source_degree", "file"]
    with out.open("w", encoding="utf-8") as f:
        f.write("# Source Vertex Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(h, "") for h in headers) + " |\n")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select high-degree source vertices for dataset/*.mtx.")
    parser.add_argument("datasets", nargs="*", help="dataset names without dataset/ prefix or .mtx suffix")
    parser.add_argument("--top-k", type=int, default=10, help="number of source candidates per dataset")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    args = parser.parse_args()

    rows = []
    for dataset in dataset_names_from_args(args.datasets):
        print(f"[source] {dataset}")
        rows.append(process_dataset(dataset, args.top_k, args.source_dir))
    write_summary(rows, args.source_dir)


if __name__ == "__main__":
    main()
