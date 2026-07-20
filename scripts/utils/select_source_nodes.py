#!/usr/bin/env python3
"""Select nontrivial directed-BFS source vertices for graph datasets.

The CUDA BFS entry points traverse outgoing edges. A high total-degree vertex
can still be a pure sink, so source ranking must use outdegree first. This
script records top outgoing-degree candidates and their incoming/total degrees.
It does not claim that outdegree alone maximizes full reachable coverage.

Outputs:
  dataset/sources/<dataset>_sources.tsv
  dataset/sources/source_summary.md
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
SOURCE_DIR = DATASET_DIR / "sources"
DEFAULT_METADATA = DATASET_DIR / "metadata.csv"


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


def read_mtx_degrees(
    path: Path,
) -> tuple[int, list[int], list[int], int]:
    """Return node count, directed out/in degrees, and valid edge count."""
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
                    raise ValueError(
                        f"invalid MatrixMarket size line in {path}: {s}"
                    )
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
    one_based = bool(
        min_id is not None
        and min_id >= 1
        and max_id is not None
        and max_id <= n
    )
    outdegrees = [0] * n
    indegrees = [0] * n
    valid_edges = 0

    for u, v in iter_mtx_edge_lines(path):
        if one_based:
            u -= 1
            v -= 1
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        outdegrees[u] += 1
        indegrees[v] += 1
        valid_edges += 1

    return n, outdegrees, indegrees, valid_edges


def read_mtx_header(path: Path) -> tuple[int, int]:
    """Return the declared vertex and edge counts from a MatrixMarket file."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("%"):
                continue
            parts = text.split()
            if len(parts) < 3:
                raise ValueError(f"invalid MatrixMarket size line in {path}: {text}")
            rows, cols, nnz = map(int, parts[:3])
            return max(rows, cols), nnz
    raise ValueError(f"missing MatrixMarket size line: {path}")


def select_top_sources(
    path: Path,
    top_k: int,
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """Rank candidates by outdegree, then total degree, then vertex id."""
    n, outdegrees, indegrees, edge_count = read_mtx_degrees(path)
    candidates = heapq.nlargest(
        min(top_k, n),
        range(n),
        key=lambda vertex: (
            outdegrees[vertex],
            outdegrees[vertex] + indegrees[vertex],
            -vertex,
        ),
    )
    ranked = [
        (
            vertex,
            outdegrees[vertex],
            indegrees[vertex],
            outdegrees[vertex] + indegrees[vertex],
        )
        for vertex in candidates
    ]
    return n, edge_count, ranked


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def process_dataset(dataset: str, top_k: int, source_dir: Path) -> dict[str, str]:
    path = DATASET_DIR / f"{dataset}.mtx"
    if not path.exists():
        raise FileNotFoundError(path)

    declared_nodes, declared_edges = read_mtx_header(path)
    n, valid_edge_count, candidates = select_top_sources(path, top_k)
    if n != declared_nodes:
        raise ValueError(
            f"header/degrees node-count mismatch for {dataset}: "
            f"header={declared_nodes}, degrees={n}"
        )
    source_dir.mkdir(parents=True, exist_ok=True)
    out_path = source_dir / f"{dataset}_sources.tsv"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "rank\tvertex\toutdegree\tindegree\ttotal_degree\n"
        )
        for rank, candidate in enumerate(candidates, start=1):
            vertex, outdegree, indegree, total_degree = candidate
            f.write(
                f"{rank}\t{vertex}\t{outdegree}\t{indegree}\t"
                f"{total_degree}\n"
            )

    best = candidates[0] if candidates else (-1, 0, 0, 0)
    best_vertex, best_outdegree, best_indegree, best_total_degree = best
    avg_degree = (
        2.0 * valid_edge_count / n if n > 0 else float("nan")
    )
    avg_outdegree = (
        valid_edge_count / n if n > 0 else float("nan")
    )
    try:
        source_file = str(out_path.relative_to(ROOT))
    except ValueError:
        source_file = str(out_path)
    return {
        "dataset": dataset,
        "nodes": str(n),
        # Match the CUDA loader: its edge count is the MatrixMarket header nnz.
        "edges": str(declared_edges),
        "valid_edges": str(valid_edge_count),
        "avg_degree": fmt_float(avg_degree),
        "avg_outdegree": fmt_float(avg_outdegree),
        "bfs_source": str(best_vertex),
        # Backward-compatible degree now means the traversal-relevant degree.
        "bfs_source_degree": str(best_outdegree),
        "bfs_source_outdegree": str(best_outdegree),
        "bfs_source_indegree": str(best_indegree),
        "bfs_source_total_degree": str(best_total_degree),
        "file_size_bytes": str(path.stat().st_size),
        "file": source_file,
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
    headers = [
        "dataset", "nodes", "edges", "valid_edges", "avg_degree",
        "avg_outdegree", "bfs_source", "bfs_source_outdegree",
        "bfs_source_indegree", "bfs_source_total_degree",
        "file_size_bytes", "file",
    ]
    with out.open("w", encoding="utf-8") as f:
        f.write("# Source Vertex Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(h, "") for h in headers) + " |\n")
    print(f"wrote {out}")


def write_metadata(rows: list[dict[str, str]], output: Path) -> None:
    """Write the machine-readable metadata consumed by benchmark scripts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset", "nodes", "edges", "valid_edges", "file_size_bytes",
        "bfs_source", "bfs_source_degree", "bfs_source_outdegree",
        "bfs_source_indegree", "bfs_source_total_degree", "avg_degree",
        "avg_outdegree", "source_file",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset": row["dataset"],
                "nodes": row["nodes"],
                "edges": row["edges"],
                "valid_edges": row["valid_edges"],
                "file_size_bytes": row["file_size_bytes"],
                "bfs_source": row["bfs_source"],
                "bfs_source_degree": row["bfs_source_degree"],
                "bfs_source_outdegree": row["bfs_source_outdegree"],
                "bfs_source_indegree": row["bfs_source_indegree"],
                "bfs_source_total_degree": row[
                    "bfs_source_total_degree"
                ],
                "avg_degree": row["avg_degree"],
                "avg_outdegree": row["avg_outdegree"],
                "source_file": row["file"],
            })
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select high-outdegree directed BFS sources for dataset/*.mtx."
        )
    )
    parser.add_argument("datasets", nargs="*", help="dataset names without dataset/ prefix or .mtx suffix")
    parser.add_argument("--top-k", type=int, default=10, help="number of source candidates per dataset")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA,
        help="machine-readable dataset metadata CSV",
    )
    args = parser.parse_args()

    rows = []
    for dataset in dataset_names_from_args(args.datasets):
        print(f"[source] {dataset}")
        rows.append(process_dataset(dataset, args.top_k, args.source_dir))
    write_summary(rows, args.source_dir)
    write_metadata(rows, args.metadata_output)


if __name__ == "__main__":
    main()
