#!/usr/bin/env bash
# Prepare one immutable experiment run and submit dataset × algorithm array work.

set -eo pipefail

script_dir="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
project_root="$(cd "$script_dir/../../.." && pwd)"
python_bin="${PYTHON_BIN:-}"
config="${CONFIG:-}"
run_tag="${RUN_TAG:-}"
run_root="${RUN_ROOT:-}"
array_concurrency="${ARRAY_CONCURRENCY:-}"
resume="${RESUME:-0}"
retry_failed="${RETRY_FAILED:-0}"
[[ -z "$python_bin" ]] && python_bin="$project_root/.venv/bin/python"
[[ -z "$config" ]] && config="$project_root/scripts/experiments/configs/overhead.json"
[[ -z "$run_tag" ]] && run_tag="overhead_$(date +%Y%m%d_%H%M%S)"
[[ -z "$run_root" ]] && run_root="$project_root/outputs/experiments/$run_tag"
[[ -z "$array_concurrency" ]] && array_concurrency=1

if [[ ! -x "$python_bin" ]]; then
    echo "Missing executable Python: $python_bin" >&2
    echo "Use the uv-created local environment or set PYTHON_BIN." >&2
    exit 2
fi
if [[ ! -f "$config" ]]; then
    echo "Missing experiment config: $config" >&2
    exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found in PATH." >&2
    exit 2
fi
if [[ ! "$array_concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo "ARRAY_CONCURRENCY must be a positive integer." >&2
    exit 2
fi
if [[ ! "$resume" =~ ^[01]$ || ! "$retry_failed" =~ ^[01]$ ]]; then
    echo "RESUME and RETRY_FAILED must be 0 or 1." >&2
    exit 2
fi

prepare_args=(
    scripts/experiments/prepare_run.py
    --config "$config"
    --run-dir "$run_root"
)
[[ -n "${REPEAT:-}" ]] && prepare_args+=(--repeat "$REPEAT")
[[ -n "${WARMUP:-}" ]] && prepare_args+=(--warmup "$WARMUP")
[[ -n "${TIMEOUT:-}" ]] && prepare_args+=(--timeout "$TIMEOUT")
[[ -n "${MPI_RANKS:-}" ]] && prepare_args+=(--mpi-ranks "$MPI_RANKS")
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a chosen_datasets <<< "$DATASETS"
    prepare_args+=(--datasets "${chosen_datasets[@]}")
fi
if [[ -n "${ALGORITHMS:-}" ]]; then
    read -r -a chosen_algorithms <<< "$ALGORITHMS"
    prepare_args+=(--algorithms "${chosen_algorithms[@]}")
fi
[[ "${FAST_FINGERPRINTS:-}" == "1" ]] && prepare_args+=(--fast-fingerprints)

cd "$project_root"
if [[ -e "$run_root" ]]; then
    if [[ "$resume" != "1" ]]; then
        echo "RUN_ROOT already exists: $run_root" >&2
        echo "Use a new RUN_TAG/RUN_ROOT, or set RESUME=1 to reuse its immutable manifest." >&2
        exit 2
    fi
    if [[ ! -f "$run_root/manifest/experiment.json" ]]; then
        echo "RESUME=1 requires a prepared manifest: $run_root/manifest/experiment.json" >&2
        exit 2
    fi
    echo "Reusing immutable manifest with --resume: $run_root"
else
    if [[ "$resume" == "1" ]]; then
        echo "RESUME=1 was requested but RUN_ROOT does not exist: $run_root" >&2
        exit 2
    fi
    if ! "$python_bin" "${prepare_args[@]}"; then
        echo "Manifest preparation failed; no Slurm jobs were submitted." >&2
        exit 2
    fi
fi

manifest_values="$("$python_bin" -c 'import json, sys; manifest = json.load(open(sys.argv[1], encoding="utf-8")); config = manifest["config"]; print(manifest["work_item_count"], config["repeat"], config["warmup"], len(config["methods"]), config["timeout_s"], config["mpi_ranks"])' "$run_root/manifest/experiment.json")"
read -r work_count repeat warmup method_count timeout_s mpi_ranks <<< "$manifest_values"
if ! [[ "$work_count" =~ ^[1-9][0-9]*$ && "$repeat" =~ ^[1-9][0-9]*$ && "$warmup" =~ ^[0-9]+$ && "$method_count" =~ ^[1-9][0-9]*$ && "$timeout_s" =~ ^[1-9][0-9]*$ && "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "Prepared manifest has invalid work/time settings: $manifest_values" >&2
    exit 2
fi

format_slurm_time() {
    local total_seconds="$1"
    local days=$((total_seconds / 86400))
    local remainder=$((total_seconds % 86400))
    local hours=$((remainder / 3600))
    remainder=$((remainder % 3600))
    local minutes=$((remainder / 60))
    local seconds=$((remainder % 60))
    if (( days > 0 )); then
        printf '%d-%02d:%02d:%02d' "$days" "$hours" "$minutes" "$seconds"
    else
        printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
    fi
}

invocations_per_item=$(((repeat + warmup) * method_count))
worst_case_seconds=$((invocations_per_item * timeout_s + 600))
computed_worker_time="$(format_slurm_time "$worst_case_seconds")"
worker_time="${WORKER_TIME:-$computed_worker_time}"
collector_time="${COLLECTOR_TIME:-01:00:00}"
last_index=$((work_count - 1))
mkdir -p "$run_root/slurm"

sbatch_options=()
[[ -n "${PARTITION:-}" ]] && sbatch_options+=(--partition "$PARTITION")
[[ -n "${ACCOUNT:-}" ]] && sbatch_options+=(--account "$ACCOUNT")
[[ -n "${QOS:-}" ]] && sbatch_options+=(--qos "$QOS")

for export_value in "$project_root" "$run_root" "$python_bin" "$retry_failed"; do
    if [[ "$export_value" == *","* || "$export_value" == *$'
'* ]]; then
        echo "Slurm export values cannot contain commas or newlines: $export_value" >&2
        exit 2
    fi
done
exports="ALL,PROJECT_ROOT=$project_root,RUN_ROOT=$run_root,PYTHON_BIN=$python_bin,RETRY_FAILED=$retry_failed"

if ! worker_job="$(sbatch --parsable "${sbatch_options[@]}" \
    --time="$worker_time" \
    --ntasks="$mpi_ranks" \
    --gpus-per-node="$mpi_ranks" \
    --array="0-$last_index%$array_concurrency" \
    --output="$run_root/slurm/%A_%a.out" \
    --error="$run_root/slurm/%A_%a.err" \
    --export="$exports" \
    "$script_dir/run_item.sbatch")"; then
    echo "Worker array submission failed; collector was not submitted." >&2
    exit 3
fi
if [[ -z "$worker_job" ]]; then
    echo "Worker array submission returned an empty job ID; collector was not submitted." >&2
    exit 3
fi
if ! collector_job="$(sbatch --parsable "${sbatch_options[@]}" \
    --time="$collector_time" \
    --dependency="afterany:$worker_job" \
    --output="$run_root/slurm/collect-%j.out" \
    --error="$run_root/slurm/collect-%j.err" \
    --export="$exports" \
    "$script_dir/collect.sbatch")"; then
    echo "Collector submission failed after worker array $worker_job was submitted." >&2
    exit 3
fi
if [[ -z "$collector_job" ]]; then
    echo "Collector submission returned an empty job ID." >&2
    exit 3
fi

echo "run root:      $run_root"
echo "work items:    $work_count (dataset × algorithm)"
echo "array job:     $worker_job (0-$last_index%$array_concurrency)"
echo "collector job: $collector_job"
echo "worker time:   $worker_time (worst-case command budget: $computed_worker_time)"
echo "MPI ranks:     $mpi_ranks (worker allocation requests the same GPU count on one node)"
echo "retry failed:  $retry_failed"
echo "Note: PARTITION, ACCOUNT and QOS are optional environment variables; no account is forced."
