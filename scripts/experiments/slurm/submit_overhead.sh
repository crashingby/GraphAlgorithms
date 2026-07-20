#!/usr/bin/env bash
# Submit dataset x algorithm work items as fixed two-GPU Slurm array tasks.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../../.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${project_root}/outputs/experiments/${RUN_TAG}}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-1}"
TIMEOUT="${TIMEOUT:-3600}"
MPI_RANKS="${MPI_RANKS:-2}"
PYTHON_MODE="${PYTHON_MODE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
UV_BIN="${UV_BIN:-}"
ALLOW_EXISTING_RUN_ROOT="${ALLOW_EXISTING_RUN_ROOT:-0}"

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${name} must be a non-negative integer, got: ${value}" >&2
        exit 2
    fi
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    require_nonnegative_integer "${name}" "${value}"
    if [[ "${value}" -eq 0 ]]; then
        echo "${name} must be greater than zero." >&2
        exit 2
    fi
}

require_positive_integer ARRAY_CONCURRENCY "${ARRAY_CONCURRENCY}"
require_positive_integer REPEAT "${REPEAT}"
require_nonnegative_integer WARMUP "${WARMUP}"
require_positive_integer TIMEOUT "${TIMEOUT}"
if [[ "${MPI_RANKS}" -ne 2 ]]; then
    echo "This workflow is fixed to exactly two MPI ranks and two GPUs." >&2
    exit 2
fi
if [[ "${ALLOW_EXISTING_RUN_ROOT}" != "0" && \
      "${ALLOW_EXISTING_RUN_ROOT}" != "1" ]]; then
    echo "ALLOW_EXISTING_RUN_ROOT must be 0 or 1." >&2
    exit 2
fi
if [[ -s "${RUN_ROOT}/work_items.tsv" && \
      "${ALLOW_EXISTING_RUN_ROOT}" != "1" ]]; then
    echo "Refusing to reuse populated run root: ${RUN_ROOT}" >&2
    echo "Choose a new RUN_TAG, or explicitly set ALLOW_EXISTING_RUN_ROOT=1." >&2
    exit 2
fi

case "${PYTHON_MODE}" in
    auto)
        if [[ -n "${PYTHON_BIN}" ]]; then
            PYTHON_MODE="binary"
        elif [[ -x "${project_root}/.venv/bin/python" ]]; then
            PYTHON_MODE="binary"
            PYTHON_BIN="${project_root}/.venv/bin/python"
        elif command -v uv >/dev/null 2>&1; then
            PYTHON_MODE="uv"
            UV_BIN="$(command -v uv)"
        else
            echo "No usable .venv/bin/python or uv executable found." >&2
            exit 2
        fi
        ;;
    venv)
        PYTHON_MODE="binary"
        PYTHON_BIN="${PYTHON_BIN:-${project_root}/.venv/bin/python}"
        ;;
    binary)
        PYTHON_BIN="${PYTHON_BIN:-${project_root}/.venv/bin/python}"
        ;;
    uv)
        UV_BIN="${UV_BIN:-$(command -v uv || true)}"
        ;;
    *)
        echo "PYTHON_MODE must be auto, venv, binary, or uv." >&2
        exit 2
        ;;
esac

if [[ "${PYTHON_MODE}" == "binary" ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
        echo "Run 'uv venv', or set PYTHON_MODE=uv/PYTHON_BIN." >&2
        exit 2
    fi
    python_check=("${PYTHON_BIN}")
else
    if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
        echo "uv executable is not available: ${UV_BIN:-<empty>}" >&2
        exit 2
    fi
    python_check=("${UV_BIN}" run python)
fi

cd "${project_root}"
"${python_check[@]}" -c 'import sys; assert sys.version_info >= (3, 9)'
"${python_check[@]}" -c 'import matplotlib'
if ! command -v mpirun >/dev/null 2>&1; then
    echo "mpirun was not found in PATH." >&2
    exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found in PATH." >&2
    exit 2
fi

expected_binaries=(
    "build/bin/bfs/bfs"
    "build/bin/bfs/bfs_queue"
    "build/bin/bfs/bfs_multiGPU_basic"
    "build/bin/bfs/bfs_multiGPU"
    "build/bin/cc/cc"
    "build/bin/cc/cc_queue"
    "build/bin/cc/cc_multiGPU_basic"
    "build/bin/cc/cc_multiGPU"
    "build/bin/kcore/kcore"
    "build/bin/kcore/kcore_queue"
    "build/bin/kcore/kcore_multiGPU_basic"
    "build/bin/kcore/kcore_multiGPU"
    "build/bin/pagerank/pagerank"
    "build/bin/pagerank/pagerank_queue"
    "build/bin/pagerank/pagerank_multiGPU_basic"
    "build/bin/pagerank/pagerank_multiGPU"
)
missing_binaries=0
for executable in "${expected_binaries[@]}"; do
    if [[ ! -x "${project_root}/${executable}" ]]; then
        echo "Missing/non-executable benchmark binary: ${executable}" >&2
        missing_binaries="$((missing_binaries + 1))"
    fi
done
if [[ "${missing_binaries}" -ne 0 ]]; then
    echo "Build all 16 benchmark targets before submitting." >&2
    exit 2
fi

if [[ ! -f "${project_root}/dataset/metadata.csv" ]]; then
    echo "Missing dataset/metadata.csv." >&2
    echo "Run: uv run python scripts/utils/select_source_nodes.py --top-k 10" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}/runs" "${RUN_ROOT}/slurm"
find "${project_root}/dataset" -maxdepth 1 -type f -name '*.mtx' -printf '%f\n' \
    | sed 's/\.mtx$//' | sort > "${RUN_ROOT}/datasets.txt"
dataset_count="$(wc -l < "${RUN_ROOT}/datasets.txt")"
if [[ "${dataset_count}" -eq 0 ]]; then
    echo "No dataset/*.mtx files found." >&2
    exit 2
fi

metadata_count="$(tail -n +2 "${project_root}/dataset/metadata.csv" | wc -l)"
source_count="$(
    find "${project_root}/dataset/sources" -maxdepth 1 \
        -type f -name '*_sources.tsv' | wc -l
)"
if [[ "${metadata_count}" -ne "${dataset_count}" || \
      "${source_count}" -ne "${dataset_count}" ]]; then
    echo "Dataset preparation is incomplete: mtx=${dataset_count}, metadata=${metadata_count}, sources=${source_count}." >&2
    echo "Run: uv run python scripts/utils/select_source_nodes.py --top-k 10" >&2
    exit 2
fi

work_items="${RUN_ROOT}/work_items.tsv"
printf 'algorithm\tdataset\trepeat\n' > "${work_items}"
while IFS= read -r dataset; do
    for algorithm in bfs cc kcore pagerank; do
        printf '%s\t%s\t%s\n' \
            "${algorithm}" "${dataset}" "${REPEAT}" >> "${work_items}"
    done
done < "${RUN_ROOT}/datasets.txt"

work_count="$(( $(wc -l < "${work_items}") - 1 ))"
if [[ "${work_count}" -ne "$((dataset_count * 4))" ]]; then
    echo "Internal error while generating work_items.tsv." >&2
    exit 2
fi
last_index="$((work_count - 1))"

sbatch_extra=()
if [[ -n "${PARTITION:-}" ]]; then
    sbatch_extra+=(--partition "${PARTITION}")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
    sbatch_extra+=(--account "${ACCOUNT}")
fi
if [[ -n "${QOS:-}" ]]; then
    sbatch_extra+=(--qos "${QOS}")
fi

for export_value in \
    "${project_root}" "${RUN_ROOT}" "${PYTHON_BIN}" "${UV_BIN}"; do
    if [[ "${export_value}" == *","* || "${export_value}" == *$'\n'* ]]; then
        echo "Slurm export values must not contain commas/newlines: ${export_value}" >&2
        exit 2
    fi
done

export_values="ALL,PROJECT_ROOT=${project_root},RUN_ROOT=${RUN_ROOT},REPEAT=${REPEAT},WARMUP=${WARMUP},TIMEOUT=${TIMEOUT},MPI_RANKS=2,PYTHON_MODE=${PYTHON_MODE},PYTHON_BIN=${PYTHON_BIN},UV_BIN=${UV_BIN}"
worker_job="$(sbatch --parsable \
    "${sbatch_extra[@]}" \
    --array="0-${last_index}%${ARRAY_CONCURRENCY}" \
    --output="${RUN_ROOT}/slurm/%A_%a.out" \
    --error="${RUN_ROOT}/slurm/%A_%a.err" \
    --export="${export_values}" \
    "${script_dir}/run_overhead.sbatch")"

collector_job="$(sbatch --parsable \
    "${sbatch_extra[@]}" \
    --dependency="afterany:${worker_job}" \
    --output="${RUN_ROOT}/slurm/collect-%j.out" \
    --error="${RUN_ROOT}/slurm/collect-%j.err" \
    --export="${export_values}" \
    "${script_dir}/collect_overhead.sbatch")"

echo "run root:       ${RUN_ROOT}"
echo "work items:     ${work_count} (${dataset_count} datasets x 4 algorithms)"
echo "array job:      ${worker_job} (0-${last_index}%${ARRAY_CONCURRENCY})"
echo "collector job:  ${collector_job}"
echo "python mode:    ${PYTHON_MODE}"
