#!/usr/bin/env python3
"""Create an immutable experiment manifest before running any GPU benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common import (
    ALGORITHMS,
    DATASET_DIR,
    METHODS,
    ROOT,
    SCHEMA_VERSION,
    executable_path,
    file_fingerprint,
    json_dump_atomic,
    read_metadata,
    relative_to_root,
    repository_provenance,
    validate_experiment_inputs,
)


DEFAULT_CONFIG = ROOT / "scripts" / "experiments" / "configs" / "overhead.json"


def resolve_path(path: Path) -> Path:
    """Interpret CLI paths relative to the repository root."""
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    """Load a JSON experiment configuration."""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_list(config: Mapping[str, Any], name: str) -> List[str]:
    """Read a nonempty JSON string list."""
    value = config.get(name)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"config.{name} must be a nonempty string list")
    return list(value)


def require_int(config: Mapping[str, Any], name: str, minimum: int) -> int:
    """Read an integer configuration setting with a lower bound."""
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"config.{name} must be an integer >= {minimum}")
    return value


def normalize_config(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Apply safe CLI selection overrides and validate the frozen config shape."""
    result = dict(config)
    result["datasets"] = list(args.datasets) if args.datasets else require_list(config, "datasets")
    result["algorithms"] = list(args.algorithms) if args.algorithms else require_list(config, "algorithms")
    result["methods"] = list(args.methods) if args.methods else require_list(config, "methods")
    result["repeat"] = args.repeat if args.repeat is not None else require_int(config, "repeat", 1)
    result["warmup"] = args.warmup if args.warmup is not None else require_int(config, "warmup", 0)
    result["timeout_s"] = args.timeout if args.timeout is not None else require_int(config, "timeout_s", 1)
    result["mpi_ranks"] = args.mpi_ranks if args.mpi_ranks is not None else require_int(config, "mpi_ranks", 2)

    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("config.parameters must be an object")
    for name in ("k", "alpha", "beta", "threshold"):
        if name not in parameters or isinstance(parameters[name], bool) or not isinstance(parameters[name], (int, float)):
            raise ValueError(f"config.parameters.{name} must be numeric")
    result["parameters"] = dict(parameters)

    launcher = config.get("launcher")
    if not isinstance(launcher, dict):
        raise ValueError("config.launcher must be an object")
    command = args.mpi_launcher or launcher.get("command")
    ranks_flag = launcher.get("ranks_flag")
    extra_args = list(args.launcher_args) if args.launcher_args else launcher.get("extra_args", [])
    if not isinstance(command, str) or not command:
        raise ValueError("config.launcher.command must be a nonempty string")
    if not isinstance(ranks_flag, str) or not ranks_flag:
        raise ValueError("config.launcher.ranks_flag must be a nonempty string")
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise ValueError("config.launcher.extra_args must be a string list")
    result["launcher"] = {
        "command": command,
        "ranks_flag": ranks_flag,
        "extra_args": extra_args,
    }

    names = config.get("name", "fault_tolerance_overhead")
    if not isinstance(names, str) or not names:
        raise ValueError("config.name must be a nonempty string")
    result["name"] = names
    result["schema_version"] = SCHEMA_VERSION
    result["config_source"] = str(resolve_path(args.config))
    return result


def validate_pair_selection(methods: Sequence[str]) -> None:
    """Require complete method pairs so a selected mode remains comparable."""
    selected = set(methods)
    pairs = (
        ("basic", "tolerance_queue"),
        ("multigpu_basic", "multigpu"),
    )
    errors = [
        f"methods must include both {baseline} and {tolerance}, or neither"
        for baseline, tolerance in pairs
        if (baseline in selected) != (tolerance in selected)
    ]
    if errors:
        raise ValueError("; ".join(errors))


def snapshot_metadata(
    run_dir: Path,
    metadata_path: Path,
    metadata: Mapping[str, Mapping[str, str]],
    datasets: Sequence[str],
    algorithms: Sequence[str],
) -> Dict[str, Any]:
    """Snapshot metadata and only the BFS source evidence a run can consume."""
    manifest_dir = run_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    copied_metadata = manifest_dir / "dataset_metadata.csv"
    shutil.copy2(metadata_path, copied_metadata)

    sources: List[Dict[str, Any]] = []
    source_dir = manifest_dir / "sources"
    if "bfs" in algorithms:
        for dataset in datasets:
            row = metadata[dataset]
            source_value = (
                row.get("source_file") or f"dataset/sources/{dataset}_sources.tsv"
            ).strip()
            source_path = Path(source_value)
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            source_path = source_path.resolve()
            if not source_path.is_file():
                raise ValueError(
                    f"missing source-candidate file for {dataset}: {source_path}; "
                    "run scripts/utils/select_source_nodes.py first"
                )
            destination = source_dir / f"{dataset}_sources.tsv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            sources.append({
                "dataset": dataset,
                "original": relative_to_root(source_path),
                "snapshot": str(destination.relative_to(run_dir)),
                "fingerprint": file_fingerprint(source_path),
            })
    return {
        "metadata_original": relative_to_root(metadata_path),
        "metadata_snapshot": str(copied_metadata.relative_to(run_dir)),
        "metadata_fingerprint": file_fingerprint(metadata_path),
        "source_snapshots": sources,
    }


def make_work_items(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create one Slurm-safe dataset × algorithm work item per planned task."""
    items: List[Dict[str, Any]] = []
    index = 0
    for algorithm in config["algorithms"]:
        for dataset in config["datasets"]:
            index += 1
            items.append({
                "index": index - 1,
                "work_id": f"{index:03d}_{algorithm}_{dataset}",
                "algorithm": algorithm,
                "dataset": dataset,
                "measurement_invocations": config["repeat"] * len(config["methods"]),
                "warmup_invocations": config["warmup"] * len(config["methods"]),
            })
    return items


def write_work_item_tsv(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    """Write the compact array lookup file used by the Slurm worker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("index", "work_id", "algorithm", "dataset"),
            delimiter="\t",
        )
        writer.writeheader()
        for item in items:
            writer.writerow({field: item[field] for field in writer.fieldnames})


def prepare(args: argparse.Namespace) -> Path:
    """Freeze config, inputs, binary hashes and Slurm work items in a new run root."""
    config_path = resolve_path(args.config)
    run_dir = resolve_path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(
            f"run directory already exists and is nonempty: {run_dir}; choose a new run tag"
        )

    config = normalize_config(load_config(config_path), args)
    validate_pair_selection(config["methods"])
    metadata_path = resolve_path(args.metadata)
    metadata = read_metadata(metadata_path)
    validate_experiment_inputs(
        config["datasets"], config["algorithms"], config["methods"], metadata,
        config["mpi_ranks"], config["launcher"]["command"],
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    input_snapshot = snapshot_metadata(
        run_dir, metadata_path, metadata, config["datasets"], config["algorithms"],
    )

    dataset_fingerprints = {
        dataset: file_fingerprint(
            DATASET_DIR / f"{dataset}.mtx", include_sha256=not args.fast_fingerprints
        )
        for dataset in config["datasets"]
    }
    executable_fingerprints = {
        f"{algorithm}/{method}": file_fingerprint(executable_path(algorithm, method))
        for algorithm in config["algorithms"]
        for method in config["methods"]
    }
    work_items = make_work_items(config)
    provenance = repository_provenance()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "experiment_manifest",
        "prepared_at_utc": provenance["recorded_at_utc"],
        "run_dir": str(run_dir),
        "config": config,
        "input_snapshot": input_snapshot,
        "dataset_fingerprints": dataset_fingerprints,
        "executable_fingerprints": executable_fingerprints,
        "provenance": provenance,
        "work_item_count": len(work_items),
        "planned_measurement_invocations": sum(
            int(item["measurement_invocations"]) for item in work_items
        ),
        "planned_warmup_invocations": sum(
            int(item["warmup_invocations"]) for item in work_items
        ),
    }
    json_dump_atomic(run_dir / "manifest" / "experiment.json", manifest)
    json_dump_atomic(run_dir / "manifest" / "work_items.json", {"items": work_items})
    write_work_item_tsv(run_dir / "manifest" / "work_items.tsv", work_items)
    print(f"prepared run: {run_dir}")
    print(f"work items: {len(work_items)}")
    print(f"planned measurements: {manifest['planned_measurement_invocations']}")
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build the CLI for manifest creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument(
        "--mpi-ranks", type=int,
        help="MPI ranks for multi-GPU methods (must be at least 2 when selected)",
    )
    parser.add_argument("--mpi-launcher")
    parser.add_argument(
        "--launcher-args", nargs=argparse.REMAINDER,
        help="extra launcher arguments; this option must be last",
    )
    parser.add_argument(
        "--fast-fingerprints", action="store_true",
        help="record graph size/mtime only; default additionally SHA256 hashes every selected graph",
    )
    args = parser.parse_args(argv)
    for option in ("repeat", "warmup", "timeout", "mpi_ranks"):
        value = getattr(args, option)
        if value is not None and value < (0 if option == "warmup" else 1):
            parser.error(f"--{option.replace('_', '-')} has an invalid value")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run manifest preparation."""
    try:
        prepare(parse_args(argv))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
