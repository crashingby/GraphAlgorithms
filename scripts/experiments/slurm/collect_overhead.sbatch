#!/usr/bin/env bash
#SBATCH --job-name=graph-overhead-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must be exported by submit_overhead.sh}"
: "${RUN_ROOT:?RUN_ROOT must be exported by submit_overhead.sh}"

PYTHON_MODE="${PYTHON_MODE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
UV_BIN="${UV_BIN:-}"

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

mkdir -p "${RUN_ROOT}/figures"

set +e
"${python_cmd[@]}" scripts/experiments/merge_results.py "${RUN_ROOT}"
merge_status=$?
set -e

plot_status=0
if [[ -s "${RUN_ROOT}/results.csv" ]]; then
    set +e
    "${python_cmd[@]}" scripts/experiments/plot_results.py \
        "${RUN_ROOT}/results.csv" \
        --out-dir "${RUN_ROOT}/figures"
    plot_status=$?
    set -e
else
    echo "Merged results.csv is missing; plot step cannot run." >&2
    plot_status=2
fi

echo "merge status: ${merge_status}"
echo "plot status:  ${plot_status}"
if [[ "${merge_status}" -ne 0 || "${plot_status}" -ne 0 ]]; then
    echo "Collection produced partial/failed outputs; inspect collection_report.json and Slurm logs." >&2
    exit 3
fi
