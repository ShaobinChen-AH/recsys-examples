#!/bin/bash

# Tight Budget Comparison: Original System vs HotState
# Designed to force KV constraint binding
# Three tight budgets: 64 MB, 128 MB, 256 MB
# 3 runs per config

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSTU_DIR="${REPO_ROOT}"
CHECKPOINT_DIR="/workspace/data/iter500"
GIN_CONFIG="${HSTU_DIR}/inference/configs/kuairand_1k_inference_ranking.gin"
RESULTS_DIR="${HSTU_DIR}/results/tight_budget_comparison"

# Tight budgets to force constraint binding
# KuaiRand-1K KV working set ≈ 32.8 MB
# 64 MB = 100% utilization (maximum pressure)
# 128 MB = 50% utilization (strong pressure)
# 256 MB = 25% utilization (moderate pressure)
BUDGETS=(0.064 0.128 0.256)
RUNS=3
MAX_BS=1

mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/logs"

export PYTHONPATH="${PYTHONPATH}:$(cd ${REPO_ROOT}/../ && pwd)"

echo "=== Tight Budget Experiment: Forcing KV Constraint Binding ==="
echo "Target: Exercise in-pool KV residency control under a constrained KV budget"
echo ""

for budget in "${BUDGETS[@]}"; do
    budget_mb=$(python3 -c "print(int($budget * 1024))")
    echo "=== Testing budget: ${budget} GiB (${budget_mb} MB) ==="
    echo "    Expected: KV working set ≈ 32.8 MB"
    utilization=$(python3 -c "print(f'{32.8 * 100 / ($budget * 1024):.1f}')")
    echo "    Utilization: ${utilization}%"
    echo ""

    for run in $(seq 1 ${RUNS}); do
        echo "  Run ${run}/${RUNS}"

        # Original system with fixed split
        export HOTSTATE_ENABLED=0
        export HBM_BUDGET_GIB=${budget}
        cd "${HSTU_DIR}"
        echo "    [Original] Starting with HBM_BUDGET_GIB=${budget}, HOTSTATE_ENABLED=0"
        python3 "./inference/inference_gr_ranking.py" \
            --gin_config_file "${GIN_CONFIG}" \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --mode simulate \
            --max_bs ${MAX_BS} \
            2>&1 | tee "${RESULTS_DIR}/logs/original_${budget_mb}mb_run${run}.log"

        # Cleanup GPU memory
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 2

        # HotState system with dynamic split
        export HOTSTATE_ENABLED=1
        cd "${HSTU_DIR}"
        echo "    [HotState] Starting with HBM_BUDGET_GIB=${budget}, HOTSTATE_ENABLED=1"
        python3 "./inference/inference_gr_ranking.py" \
            --gin_config_file "${GIN_CONFIG}" \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --mode simulate \
            --max_bs ${MAX_BS} \
            2>&1 | tee "${RESULTS_DIR}/logs/hotstate_${budget_mb}mb_run${run}.log"

        # Cleanup GPU memory
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 2
    done
done

echo "=== All tight budget experiments complete ==="
echo "Results in: ${RESULTS_DIR}"
echo "Logs in: ${RESULTS_DIR}/logs"
