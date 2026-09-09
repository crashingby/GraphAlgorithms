#!/usr/bin/env bash
# Run the complete raw -> collect -> analyze -> plot pipeline on the current node.

set -o pipefail

script_dir="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
python_bin="$PYTHON_BIN"
config="$CONFIG"
run_tag="$RUN_TAG"
run_root="$RUN_ROOT"
[[ -z "$python_bin" ]] && python_bin="$project_root/.venv/bin/python"
[[ -z "$config" ]] && config="$project_root/scripts/experiments/configs/overhead.json"
[[ -z "$run_tag" ]] && run_tag="overhead_$(date +%Y%m%d_%H%M%S)"
[[ -z "$run_root" ]] && run_root="$project_root/outputs/experiments/$run_tag"

if [[ ! -x "$python_bin" ]]; then
    echo "Missing executable Python: $python_bin" >&2
    echo "Create the repository-local uv environment first: uv venv .venv" >&2
    exit 2
fi
if [[ ! -f "$config" ]]; then
    echo "Missing experiment config: $config" >&2
    exit 2
fi

prepare_args=(
    scripts/experiments/prepare_run.py
    --config "$config"
    --run-dir "$run_root"
)
[[ -n "$REPEAT" ]] && prepare_args+=(--repeat "$REPEAT")
[[ -n "$WARMUP" ]] && prepare_args+=(--warmup "$WARMUP")
[[ -n "$TIMEOUT" ]] && prepare_args+=(--timeout "$TIMEOUT")
[[ -n "$MPI_RANKS" ]] && prepare_args+=(--mpi-ranks "$MPI_RANKS")
if [[ -n "$DATASETS" ]]; then
    read -r -a chosen_datasets <<< "$DATASETS"
    prepare_args+=(--datasets "${chosen_datasets[@]}")
fi
if [[ -n "$ALGORITHMS" ]]; then
    read -r -a chosen_algorithms <<< "$ALGORITHMS"
    prepare_args+=(--algorithms "${chosen_algorithms[@]}")
fi
[[ "$FAST_FINGERPRINTS" == "1" ]] && prepare_args+=(--fast-fingerprints)

cd "$project_root"
if [[ -f "$run_root/manifest/experiment.json" ]]; then
    echo "Reusing prepared run directory with --resume: $run_root"
else
    "$python_bin" "${prepare_args[@]}" || exit $?
fi
if [[ "$PREPARE_ONLY" == "1" ]]; then
    echo "Prepared only: $run_root"
    exit 0
fi

run_args=(scripts/experiments/run_item.py --run-dir "$run_root" --all --resume)
[[ "$RETRY_FAILED" == "1" ]] && run_args+=(--retry-failed)
set +e
"$python_bin" "${run_args[@]}"
run_status=$?
"$python_bin" scripts/experiments/collect_results.py --run-dir "$run_root" --strict
collect_status=$?
"$python_bin" scripts/experiments/analyze_overhead.py --run-dir "$run_root"
analyze_status=$?
plot_args=(scripts/experiments/plot_overhead.py --run-dir "$run_root")
if [[ -n "$PLOT_FORMATS" ]]; then
    read -r -a formats <<< "$PLOT_FORMATS"
    plot_args+=(--formats "${formats[@]}")
fi
"$python_bin" "${plot_args[@]}"
plot_status=$?
set -e

echo "run root: $run_root"
echo "worker status: $run_status; collect: $collect_status; analyze: $analyze_status; plot: $plot_status"
if [[ "$run_status" -ne 0 || "$collect_status" -ne 0 || "$analyze_status" -ne 0 || "$plot_status" -ne 0 ]]; then
    exit 3
fi
