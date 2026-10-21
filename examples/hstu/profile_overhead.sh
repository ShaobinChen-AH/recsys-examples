#!/bin/bash

# Profiling script for HotState control plane overhead
# Usage: bash profile_overhead.sh <checkpoint_dir> <gin_config>

CHECKPOINT_DIR=${1:-/workspace/data/iter500}
GIN_CONFIG=${2:-./inference/configs/kuairand_1k_inference_ranking.gin}
OUTPUT_DIR=./profiles/overhead_analysis
BATCH_COUNT=100

mkdir -p $OUTPUT_DIR

echo "=== HotState Control Plane Overhead Profiling ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Config: $GIN_CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

# Test 1: Baseline (HotState disabled)
echo "[1/3] Running baseline (HotState disabled)..."
export HOTSTATE_ENABLED=0
python3 ./inference/inference_gr_ranking.py \
  --gin_config_file $GIN_CONFIG \
  --checkpoint_dir $CHECKPOINT_DIR \
  --mode simulate \
  --max_bs 1 \
  2>&1 | tee $OUTPUT_DIR/baseline_no_hotstate.log

echo ""
echo "[1/3] Baseline complete. Extracting metrics..."
grep "^\[PROFILE" $OUTPUT_DIR/baseline_no_hotstate.log > $OUTPUT_DIR/baseline_timings.txt
grep "^Total time" $OUTPUT_DIR/baseline_no_hotstate.log >> $OUTPUT_DIR/baseline_timings.txt

# Test 2: HotState enabled, verbose
echo "[2/3] Running with HotState enabled (verbose profiling)..."
export HOTSTATE_ENABLED=1
python3 ./inference/inference_gr_ranking.py \
  --gin_config_file $GIN_CONFIG \
  --checkpoint_dir $CHECKPOINT_DIR \
  --mode simulate \
  --max_bs 1 \
  2>&1 | tee $OUTPUT_DIR/hotstate_enabled.log

echo ""
echo "[2/3] HotState enabled complete. Extracting metrics..."
grep "^\[PROFILE" $OUTPUT_DIR/hotstate_enabled.log > $OUTPUT_DIR/hotstate_timings.txt
grep "^Total time" $OUTPUT_DIR/hotstate_enabled.log >> $OUTPUT_DIR/hotstate_timings.txt

# Test 3: HotState enabled with residency changes frozen to isolate overhead
echo "[3/3] Running with HotState frozen (overhead only, no benefit)..."
export HOTSTATE_ENABLED=1
export HOTSTATE_FREEZE_ALLOCATION=1
python3 ./inference/inference_gr_ranking.py \
  --gin_config_file $GIN_CONFIG \
  --checkpoint_dir $CHECKPOINT_DIR \
  --mode simulate \
  --max_bs 1 \
  2>&1 | tee $OUTPUT_DIR/hotstate_frozen.log

echo ""
echo "[3/3] HotState frozen complete. Extracting metrics..."
grep "^\[PROFILE" $OUTPUT_DIR/hotstate_frozen.log > $OUTPUT_DIR/hotstate_frozen_timings.txt
grep "^Total time" $OUTPUT_DIR/hotstate_frozen.log >> $OUTPUT_DIR/hotstate_frozen_timings.txt

# Generate summary
echo ""
echo "=== PROFILING RESULTS ==="
echo "Baseline (no HotState):"
tail -2 $OUTPUT_DIR/baseline_timings.txt
echo ""
echo "HotState Enabled:"
tail -2 $OUTPUT_DIR/hotstate_timings.txt
echo ""
echo "HotState Frozen (overhead only):"
tail -2 $OUTPUT_DIR/hotstate_frozen_timings.txt
echo ""
echo "Full timings saved to:"
echo "  $OUTPUT_DIR/baseline_timings.txt"
echo "  $OUTPUT_DIR/hotstate_timings.txt"
echo "  $OUTPUT_DIR/hotstate_frozen_timings.txt"
echo ""
echo "Detailed logs:"
echo "  $OUTPUT_DIR/baseline_no_hotstate.log"
echo "  $OUTPUT_DIR/hotstate_enabled.log"
echo "  $OUTPUT_DIR/hotstate_frozen.log"

