#!/usr/bin/env python3
import argparse
import heapq
import math
from pathlib import Path


def read_mtx_graph(dataset_name):
    path = Path("dataset") / f"{dataset_name}.mtx"
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    header = None
    raw_edges = []
    min_id = None
    max_id = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if header is None:
                if len(parts) < 3:
                    raise ValueError(f"invalid MatrixMarket size line: {line}")
                rows, cols, nnz = map(int, parts[:3])
                header = (rows, cols, nnz)
                continue

            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            raw_edges.append((u, v))
            min_id = u if min_id is None else min(min_id, u, v)
            max_id = u if max_id is None else max(max_id, u, v)

    if header is None:
        raise ValueError(f"missing MatrixMarket size line: {path}")

    n = max(header[0], header[1])
    one_based = bool(raw_edges and min_id is not None and min_id >= 1 and max_id <= n)

    adj = [set() for _ in range(n)]
    for u, v in raw_edges:
        if one_based:
            u -= 1
            v -= 1
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        adj[u].add(v)
        adj[v].add(u)

    return path, n, adj


def edge_count(adj):
    return sum(len(nei) for nei in adj) // 2


def degree_assortativity(adj):
    # Newman degree assortativity for undirected graphs, computed over edges.
    m = edge_count(adj)
    if m == 0:
        return float("nan")

    deg = [len(nei) for nei in adj]
    sum_jk = 0.0
    sum_half_j_plus_k = 0.0
    sum_half_j2_plus_k2 = 0.0

    for u, nei in enumerate(adj):
        du = deg[u]
        for v in nei:
            if u < v:
                dv = deg[v]
                sum_jk += du * dv
                sum_half_j_plus_k += 0.5 * (du + dv)
                sum_half_j2_plus_k2 += 0.5 * (du * du + dv * dv)

    mean_prod = sum_jk / m
    mean_sum = sum_half_j_plus_k / m
    mean_sq_sum = sum_half_j2_plus_k2 / m
    denom = mean_sq_sum - mean_sum * mean_sum
    if denom == 0.0:
        return float("nan")
    return (mean_prod - mean_sum * mean_sum) / denom


def triangle_stats(adj):
    n = len(adj)
    deg = [len(nei) for nei in adj]

    forward = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            if deg[u] < deg[v] or (deg[u] == deg[v] and u < v):
                forward[u].add(v)

    triangles = 0
    local_triangles = [0] * n
    max_edge_triangles = 0

    for u in range(n):
        fu = forward[u]
        for v in fu:
            common = fu.intersection(forward[v])
            c = len(common)
            triangles += c
            for w in common:
                local_triangles[u] += 1
                local_triangles[v] += 1
                local_triangles[w] += 1

    # Edge triangle counts need common-neighbor counts for every undirected edge.
    for u, nei in enumerate(adj):
        for v in nei:
            if u < v:
                c = len(adj[u].intersection(adj[v]))
                if c > max_edge_triangles:
                    max_edge_triangles = c

    total_local_coeff = 0.0
    connected_triples = 0
    for v, d in enumerate(deg):
        if d >= 2:
            connected_triples += d * (d - 1) // 2
            total_local_coeff += (2.0 * local_triangles[v]) / (d * (d - 1))

    avg_local_clustering = total_local_coeff / n if n > 0 else float("nan")
    global_clustering = (
        (3.0 * triangles) / connected_triples
        if connected_triples > 0
        else float("nan")
    )

    m = edge_count(adj)
    avg_triangles_per_edge = (3.0 * triangles / m) if m > 0 else float("nan")
    return (
        triangles,
        avg_triangles_per_edge,
        max_edge_triangles,
        avg_local_clustering,
        global_clustering,
    )


def max_core_number(adj):
    n = len(adj)
    deg = [len(nei) for nei in adj]
    heap = [(deg[v], v) for v in range(n)]
    heapq.heapify(heap)
    removed = [False] * n
    max_core = 0

    while heap:
        d, v = heapq.heappop(heap)
        if removed[v] or d != deg[v]:
            continue
        removed[v] = True
        if d > max_core:
            max_core = d
        for u in adj[v]:
            if not removed[u]:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))

    return max_core


def greedy_clique_lower_bound(adj):
    deg = [len(nei) for nei in adj]
    orders = [
        sorted(range(len(adj)), key=lambda v: deg[v], reverse=True),
        sorted(range(len(adj)), key=lambda v: (deg[v], v), reverse=True),
    ]

    best = 0
    for order in orders:
        clique = []
        candidates = set(order)
        for v in order:
            if v not in candidates:
                continue
            if all(v in adj[u] for u in clique):
                clique.append(v)
                candidates.intersection_update(adj[v])
                if len(clique) > best:
                    best = len(clique)

    return best


def fmt(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Print structural statistics for dataset/<name>.mtx."
    )
    parser.add_argument("dataset", help="dataset name without dataset/ prefix or .mtx suffix")
    args = parser.parse_args()

    path, n, adj = read_mtx_graph(args.dataset)
    m = edge_count(adj)
    degrees = [len(nei) for nei in adj]
    dmax = max(degrees) if degrees else 0
    davg = (2.0 * m / n) if n > 0 else float("nan")

    assort = degree_assortativity(adj)
    (
        triangles,
        avg_triangles_per_edge,
        max_edge_triangles,
        avg_local_clustering,
        global_clustering,
    ) = triangle_stats(adj)
    kmax = max_core_number(adj)
    clique_lb = greedy_clique_lower_bound(adj)

    rows = [
        ("dataset", "Input file", path),
        ("|V|", "Number of nodes", n),
        ("|E|", "Number of edges", m),
        ("dmax", "Maximum degree", dmax),
        ("davg", "Average degree", davg),
        ("r", "Assort. Coeff.", assort),
        ("|T|", "Number of triangles (3-clique)", triangles),
        ("|T|avg", "Average triangles formed by a edge", avg_triangles_per_edge),
        ("|T|max", "Maximum number of triangles formed by a edge", max_edge_triangles),
        ("κavg", "Average local clustering coefficient", avg_local_clustering),
        ("κ", "Global clustering coefficient", global_clustering),
        ("Kmax", "Maximum k-core number", kmax),
        ("ωlb", "Lower bound on the size of the maximum clique", clique_lb),
    ]

    metric_width = max(len(metric) for metric, _, _ in rows)
    desc_width = max(len(desc) for _, desc, _ in rows)
    print(f"{'Metric':<{metric_width}}  {'Description':<{desc_width}}  Value")
    print(f"{'-' * metric_width}  {'-' * desc_width}  {'-' * 12}")
    for metric, desc, value in rows:
        print(f"{metric:<{metric_width}}  {desc:<{desc_width}}  {fmt(value)}")


if __name__ == "__main__":
    main()
