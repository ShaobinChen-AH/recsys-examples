#!/bin/bash

# Embedding Access Pattern Trace Collection
# Single run at 131MB budget to instrument actual access patterns

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSTU_DIR="${REPO_ROOT}"
CHECKPOINT_DIR="/workspace/data/iter500"
GIN_CONFIG="${HSTU_DIR}/inference/configs/kuairand_1k_inference_ranking.gin"
RESULTS_DIR="${HSTU_DIR}/results/embedding_trace"

mkdir -p "${RESULTS_DIR}"

export PYTHONPATH="${PYTHONPATH}:$(cd ${REPO_ROOT}/../ && pwd)"

echo "=== Embedding Access Pattern Trace Collection ==="
echo "Objective: Measure ACTUAL embedding access patterns to derive value density formula"
echo ""
echo "Running ONE trace at 131MB budget (baseline system)"
echo "This will collect:"
echo "  - Unique items accessed per batch"
echo "  - Access frequency distribution"
echo "  - Working set size over time"
echo "  - Temporal locality patterns"
echo ""

# Run baseline to get clean trace without HotState interference
export HOTSTATE_ENABLED=0
export HBM_BUDGET_GIB=0.128
cd "${HSTU_DIR}"

echo "Starting trace collection..."
python3 "./inference/inference_gr_ranking.py" \
    --gin_config_file "${GIN_CONFIG}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --mode simulate \
    --max_bs 1 \
    2>&1 | tee "${RESULTS_DIR}/embedding_access_trace.log"

echo ""
echo "=== Trace collection complete ==="
echo "Results in: ${RESULTS_DIR}/embedding_access_trace.log"
echo ""
echo "Look for section: 'EMBEDDING ACCESS PATTERN ANALYSIS (Data-Driven)'"
echo "This contains measured constants to replace hardcoded guesses in value density formula"
