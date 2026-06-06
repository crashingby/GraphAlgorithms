#!/usr/bin/env python3
"""Run GraphAlgorithm benchmarks and write results.csv.

Examples:
    conda run -n test310 python scripts/experiments/run_benchmarks.py --datasets flickr 364 --repeat 3
    python scripts/experiments/run_benchmarks.py --algorithms bfs cc --methods basic tolerance_queue --datasets flickr
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
SOURCE_DIR = DATASET_DIR / "sources"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "experiments"

GPU_TIME_RE = re.compile(r"GPU time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
ITER_RE = re.compile(r"(?:finished in|迭代)\s*([0-9]+)\s*(?:iterations|次)")

ALGORITHMS = ("bfs", "cc", "kcore", "pagerank")
METHODS = ("basic", "tolerance_queue", "multigpu_basic", "multigpu")

EXECUTABLES: Dict[Tuple[str, str], str] = {
    ("bfs", "basic"): "build/bin/bfs/bfs",
    ("bfs", "tolerance_queue"): "build/bin/bfs/bfs_queue",
    ("bfs", "multigpu_basic"): "build/bin/bfs/bfs_multiGPU_basic",
    ("bfs", "multigpu"): "build/bin/bfs/bfs_multiGPU",

    ("cc", "basic"): "build/bin/cc/cc",
    ("cc", "tolerance_queue"): "build/bin/cc/cc_queue",
    ("cc", "multigpu_basic"): "build/bin/cc/cc_multiGPU_basic",
    ("cc", "multigpu"): "build/bin/cc/cc_multiGPU",

    ("kcore", "basic"): "build/bin/kcore/kcore",
    ("kcore", "tolerance_queue"): "build/bin/kcore/kcore_queue",
    ("kcore", "multigpu_basic"): "build/bin/kcore/kcore_multiGPU_basic",
    ("kcore", "multigpu"): "build/bin/kcore/kcore_multiGPU",

    ("pagerank", "basic"): "build/bin/pagerank/pagerank",
    ("pagerank", "tolerance_queue"): "build/bin/pagerank/pagerank_queue",
    ("pagerank", "multigpu_basic"): "build/bin/pagerank/pagerank_multiGPU_basic",
    ("pagerank", "multigpu"): "build/bin/pagerank/pagerank_multiGPU",
}


def discover_datasets() -> List[str]:
    return sorted(p.stem for p in DATASET_DIR.glob("*.mtx"))


def read_bfs_source(dataset: str) -> int:
    path = SOURCE_DIR / f"{dataset}_sources.tsv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vertex = row.get("vertex")
            if vertex is None:
                continue
            try:
                return int(vertex)
            except ValueError:
                continue
    return 0


def build_command(
    algorithm: str,
    method: str,
    dataset: str,
    mpi_ranks: int,
    alpha: float,
    beta: float,
    threshold: float,
    k_value: int,
    extra_mpirun_args: Sequence[str],
) -> List[str]:
    exe = str(ROOT / EXECUTABLES[(algorithm, method)])
    cmd: List[str] = []
    if method in {"multigpu_basic", "multigpu"}:
        cmd.extend(["mpirun", "-np", str(mpi_ranks)])
        cmd.extend(extra_mpirun_args)
    cmd.extend([exe, dataset])

    if algorithm == "bfs":
        cmd.extend(["-s", str(read_bfs_source(dataset))])
    if algorithm == "kcore":
        cmd.extend(["-k", str(k_value)])
    if method in {"tolerance_queue", "multigpu"}:
        cmd.extend(["-a", str(alpha), "-b", str(beta), "-t", str(threshold)])

    cmd.append("-n")
    return cmd


def run_command(cmd: Sequence[str], timeout: int) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_gpu_time(stdout: str) -> Optional[float]:
    match = GPU_TIME_RE.search(stdout)
    return float(match.group(1)) if match else None


def parse_iterations(stdout: str) -> Optional[int]:
    matches = ITER_RE.findall(stdout)
    if not matches:
        return None
    return int(matches[-1])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_benchmarks(args: argparse.Namespace) -> Path:
    datasets = args.datasets or discover_datasets()
    algorithms = args.algorithms
    methods = args.methods
    out_dir = args.out_dir or DEFAULT_OUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    total = len(datasets) * len(algorithms) * len(methods) * args.repeat
    idx = 0

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset", "algorithm", "method", "repeat", "returncode",
            "gpu_time_ms", "iterations", "bfs_src", "k", "alpha", "beta",
            "threshold", "command", "stdout_log", "stderr_log",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset in datasets:
            if not (DATASET_DIR / f"{dataset}.mtx").exists():
                print(f"[skip] dataset/{dataset}.mtx not found", file=sys.stderr)
                continue
            for algorithm in algorithms:
                for method in methods:
                    for rep in range(1, args.repeat + 1):
                        idx += 1
                        cmd = build_command(
                            algorithm=algorithm,
                            method=method,
                            dataset=dataset,
                            mpi_ranks=args.mpi_ranks,
                            alpha=args.alpha,
                            beta=args.beta,
                            threshold=args.threshold,
                            k_value=args.k,
                            extra_mpirun_args=args.mpirun_args,
                        )
                        label = f"{dataset}.{algorithm}.{method}.r{rep}"
                        print(f"[{idx}/{total}] {label}: {shlex.join(cmd)}", flush=True)

                        stdout = ""
                        stderr = ""
                        returncode = -999
                        try:
                            returncode, stdout, stderr = run_command(cmd, args.timeout)
                        except subprocess.TimeoutExpired as exc:
                            stdout = exc.stdout or ""
                            stderr = (exc.stderr or "") + f"\nTIMEOUT after {args.timeout}s\n"
                            returncode = -998

                        stdout_log = log_dir / f"{label}.stdout.log"
                        stderr_log = log_dir / f"{label}.stderr.log"
                        write_text(stdout_log, stdout)
                        write_text(stderr_log, stderr)

                        row = {
                            "dataset": dataset,
                            "algorithm": algorithm,
                            "method": method,
                            "repeat": rep,
                            "returncode": returncode,
                            "gpu_time_ms": parse_gpu_time(stdout),
                            "iterations": parse_iterations(stdout),
                            "bfs_src": read_bfs_source(dataset) if algorithm == "bfs" else "",
                            "k": args.k if algorithm == "kcore" else "",
                            "alpha": args.alpha if method in {"tolerance_queue", "multigpu"} else "",
                            "beta": args.beta if method in {"tolerance_queue", "multigpu"} else "",
                            "threshold": args.threshold if method in {"tolerance_queue", "multigpu"} else "",
                            "command": shlex.join(cmd),
                            "stdout_log": str(stdout_log.relative_to(ROOT)),
                            "stderr_log": str(stderr_log.relative_to(ROOT)),
                        }
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()

                        if returncode != 0:
                            print(f"  -> failed returncode={returncode}; see {stderr_log}", file=sys.stderr)
                        elif row["gpu_time_ms"] is None:
                            print(f"  -> warning: GPU time not found; see {stdout_log}", file=sys.stderr)

    return results_csv



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4 algorithms x 4 methods benchmarks and write results.csv.")
    parser.add_argument("--datasets", nargs="*", help="Dataset ids without dataset/ prefix or .mtx suffix. Default: all dataset/*.mtx")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mpirun-args", nargs=argparse.REMAINDER, default=[], help="Extra args inserted after mpirun -np N. Put this option last.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results_csv = run_benchmarks(args)
    print(f"results: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
