#!/usr/bin/env bash
#SBATCH --job-name=graph-overhead
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must be exported by submit_overhead.sh}"
: "${RUN_ROOT:?RUN_ROOT must be exported by submit_overhead.sh}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"

REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-1}"
TIMEOUT="${TIMEOUT:-3600}"
MPI_RANKS="${MPI_RANKS:-2}"
PYTHON_MODE="${PYTHON_MODE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
UV_BIN="${UV_BIN:-}"

if [[ "${MPI_RANKS}" -ne 2 ]]; then
    echo "run_overhead.sbatch is fixed to two MPI ranks." >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
case "${PYTHON_MODE}" in
    auto|venv)
        if [[ -z "${PYTHON_BIN}" ]]; then
            PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
        fi
        if [[ -x "${PYTHON_BIN}" ]]; then
            PYTHON_MODE="binary"
            python_cmd=("${PYTHON_BIN}")
        elif command -v uv >/dev/null 2>&1; then
            PYTHON_MODE="uv"
            UV_BIN="$(command -v uv)"
            python_cmd=("${UV_BIN}" run python)
        else
            echo "No executable .venv Python or uv found." >&2
            exit 2
        fi
        ;;
    binary)
        if [[ ! -x "${PYTHON_BIN}" ]]; then
            echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
            exit 2
        fi
        python_cmd=("${PYTHON_BIN}")
        ;;
    uv)
        if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
            echo "uv executable is not available: ${UV_BIN:-<empty>}" >&2
            exit 2
        fi
        python_cmd=("${UV_BIN}" run python)
        ;;
    *)
        echo "Unsupported PYTHON_MODE: ${PYTHON_MODE}" >&2
        exit 2
        ;;
esac

work_items="${RUN_ROOT}/work_items.tsv"
if [[ ! -f "${work_items}" ]]; then
    echo "Missing work item manifest: ${work_items}" >&2
    exit 2
fi
work_line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${work_items}")"
if [[ -z "${work_line}" ]]; then
    echo "No work item for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi
IFS=$'\t' read -r algorithm dataset manifest_repeat <<< "${work_line}"
case "${algorithm}" in
    bfs|cc|kcore|pagerank) ;;
    *)
        echo "Invalid algorithm in work_items.tsv: ${algorithm}" >&2
        exit 2
        ;;
esac
if [[ -z "${dataset}" || ! "${manifest_repeat}" =~ ^[0-9]+$ || \
      "${manifest_repeat}" -lt 1 ]]; then
    echo "Invalid work item: ${work_line}" >&2
    exit 2
fi
if [[ "${REPEAT}" -ne "${manifest_repeat}" ]]; then
    echo "REPEAT=${REPEAT} disagrees with manifest repeat=${manifest_repeat}." >&2
    exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/dataset/${dataset}.mtx" ]]; then
    echo "Missing dataset/${dataset}.mtx" >&2
    exit 2
fi

run_dir="${RUN_ROOT}/runs/${algorithm}/${dataset}"
mkdir -p "${run_dir}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "work item: algorithm=${algorithm} dataset=${dataset} repeat=${REPEAT} warmup=${WARMUP}"
echo "python mode: ${PYTHON_MODE}"
"${python_cmd[@]}" scripts/experiments/run_benchmarks.py \
    --datasets "${dataset}" \
    --algorithms "${algorithm}" \
    --methods basic tolerance_queue multigpu_basic multigpu \
    --repeat "${REPEAT}" \
    --warmup "${WARMUP}" \
    --mpi-ranks 2 \
    --timeout "${TIMEOUT}" \
    --metadata dataset/metadata.csv \
    --out-dir "${run_dir}" \
    --mpirun-args --bind-to none
