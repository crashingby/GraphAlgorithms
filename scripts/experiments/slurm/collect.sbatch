#!/usr/bin/env bash
#SBATCH --job-name=graph-overhead-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -u -o pipefail

: "$PROJECT_ROOT"
: "$RUN_ROOT"
: "$PYTHON_BIN"

cd "$PROJECT_ROOT"
set +e
"$PYTHON_BIN" scripts/experiments/collect_results.py --run-dir "$RUN_ROOT" --strict
collect_status=$?
"$PYTHON_BIN" scripts/experiments/analyze_overhead.py --run-dir "$RUN_ROOT"
analyze_status=$?
"$PYTHON_BIN" scripts/experiments/plot_overhead.py --run-dir "$RUN_ROOT"
plot_status=$?
set -e

echo "run root: $RUN_ROOT"
echo "collect=$collect_status analyze=$analyze_status plot=$plot_status"
if [[ "$collect_status" -ne 0 || "$analyze_status" -ne 0 || "$plot_status" -ne 0 ]]; then
    exit 3
fi
