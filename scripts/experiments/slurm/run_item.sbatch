#!/usr/bin/env bash
#SBATCH --job-name=graph-overhead
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-01:00:00
# submit_overhead.sh overrides task/GPU count and worker time from the manifest.

set -eo pipefail

if [[ -z "$PROJECT_ROOT" || -z "$RUN_ROOT" || -z "$PYTHON_BIN" || -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "PROJECT_ROOT, RUN_ROOT, PYTHON_BIN and SLURM_ARRAY_TASK_ID are required." >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi

work_items="$RUN_ROOT/manifest/work_items.tsv"
line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$work_items")"
if [[ -z "$line" ]]; then
    echo "No work item for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
IFS=$'\t' read -r index work_id algorithm dataset <<< "$line"

cd "$PROJECT_ROOT"
[[ -z "$OMP_NUM_THREADS" ]] && export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ -z "$NCCL_DEBUG" ]] && export NCCL_DEBUG=WARN

echo "work item: $work_id ($algorithm, $dataset)"
run_args=(
    scripts/experiments/run_item.py
    --run-dir "$RUN_ROOT"
    --work-id "$work_id"
    --resume
)
[[ "${RETRY_FAILED:-0}" == "1" ]] && run_args+=(--retry-failed)
"$PYTHON_BIN" "${run_args[@]}"
