#!/bin/bash

# Fair Comparison: Original System vs HotState
# Three budgets: 0.4 GiB, 1.0 GiB, 2.0 GiB
# 5 runs per config

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSTU_DIR="${REPO_ROOT}"
CHECKPOINT_DIR="/workspace/data/iter500"
GIN_CONFIG="${HSTU_DIR}/inference/configs/kuairand_1k_inference_ranking.gin"
RESULTS_DIR="${HSTU_DIR}/results/fair_comparison"

BUDGETS=(0.4 1.0 2.0)
RUNS=5
MAX_BS=1

mkdir -p "${RESULTS_DIR}"

export PYTHONPATH="${PYTHONPATH}:$(cd ${REPO_ROOT}/../ && pwd)"

for budget in "${BUDGETS[@]}"; do
    echo "=== Testing budget: ${budget} GiB ==="

    for run in $(seq 1 ${RUNS}); do
        echo "  Run ${run}/${RUNS}"

        # Original system with fixed split
        export HOTSTATE_ENABLED=0
        export HBM_BUDGET_GIB=${budget}
        cd "${HSTU_DIR}"
        python3 "./inference/inference_latency_benchmark.py" \
            --gin_config_file "${GIN_CONFIG}" \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --max_bs ${MAX_BS} \
            --output_json "${RESULTS_DIR}/original_${budget}gib_run${run}.json"

        # Cleanup GPU memory
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

        # HotState system with dynamic split
        export HOTSTATE_ENABLED=1
        cd "${HSTU_DIR}"
        python3 "./inference/inference_latency_benchmark.py" \
            --gin_config_file "${GIN_CONFIG}" \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --max_bs ${MAX_BS} \
            --output_json "${RESULTS_DIR}/hotstate_${budget}gib_run${run}.json"

        # Cleanup GPU memory
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    done
done

echo "=== All experiments complete ==="
cd "${HSTU_DIR}"
python3 "./analysis/aggregate_results.py" \
    --results_dir "${RESULTS_DIR}" \
    --output_csv "${RESULTS_DIR}/comparison_summary.csv"
