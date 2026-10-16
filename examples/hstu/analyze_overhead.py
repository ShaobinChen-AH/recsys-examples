#!/usr/bin/env python3
"""
Analyze HotState control plane overhead from profiling logs.
"""
import re
import sys
from pathlib import Path

def parse_timings(log_file):
    """Extract [PROFILE ...] and [PROFILE BATCH ...] lines."""
    timings = []
    with open(log_file, 'r') as f:
        for line in f:
            if '[PROFILE BATCH' in line:
                # Extract: before_batch=X.XXms forward=Y.YYms net=Z.ZZms
                match = re.search(r'before_batch=([\d.]+)ms forward=([\d.]+)ms net=([\d.]+)ms', line)
                if match:
                    timings.append({
                        'before_batch': float(match.group(1)),
                        'forward': float(match.group(2)),
                        'net': float(match.group(3)),
                    })
            elif '[PROFILE epoch' in line:
                # Extract: extract=Xms record_keys=Xms run_epoch=Xms scheduler=Xms total=Xms
                match = re.search(r'extract=([\d.]+)ms record_keys=([\d.]+)ms run_epoch=([\d.]+)ms scheduler=([\d.]+)ms total=([\d.]+)ms', line)
                if match:
                    if not timings or 'components' not in timings[-1]:
                        if timings:
                            timings[-1]['components'] = {}
                        else:
                            timings.append({})
                            timings[-1]['components'] = {}
                    timings[-1]['components'] = {
                        'extract': float(match.group(1)),
                        'record_keys': float(match.group(2)),
                        'run_epoch': float(match.group(3)),
                        'scheduler': float(match.group(4)),
                        'total': float(match.group(5)),
                    }
    return timings

def parse_total_time(log_file):
    """Extract 'Total time(s):' line."""
    with open(log_file, 'r') as f:
        for line in f:
            if 'Total time(s):' in line:
                match = re.search(r'Total time\(s\):\s+([\d.]+)', line)
                if match:
                    return float(match.group(1))
    return None

def analyze(baseline_log, hotstate_log, frozen_log=None):
    """Compare logs and compute overhead metrics."""
    print("=== HotState Control Plane Overhead Analysis ===\n")

    # Parse all logs
    baseline_timings = parse_timings(baseline_log)
    hotstate_timings = parse_timings(hotstate_log)
    frozen_timings = parse_timings(frozen_log) if frozen_log and Path(frozen_log).exists() else []

    baseline_total = parse_total_time(baseline_log)
    hotstate_total = parse_total_time(hotstate_log)
    frozen_total = parse_total_time(frozen_log) if frozen_log and Path(frozen_log).exists() else None

    # Parse batch counts
    baseline_batches = 0
    hotstate_batches = 0
    with open(baseline_log, 'r') as f:
        for line in f:
            if 'Total #batch:' in line:
                match = re.search(r'Total #batch:\s+(\d+)', line)
                if match:
                    baseline_batches = int(match.group(1))
    with open(hotstate_log, 'r') as f:
        for line in f:
            if 'Total #batch:' in line:
                match = re.search(r'Total #batch:\s+(\d+)', line)
                if match:
                    hotstate_batches = int(match.group(1))

    print(f"Baseline (no HotState):")
    print(f"  Total time: {baseline_total:.2f}s")
    print(f"  Batches: {baseline_batches}")
    if baseline_batches > 0:
        print(f"  Avg batch time: {(baseline_total / baseline_batches * 1000):.2f}ms")
    print()

    print(f"HotState Enabled:")
    print(f"  Total time: {hotstate_total:.2f}s")
    print(f"  Batches: {hotstate_batches}")
    if hotstate_timings:
        avg_before_batch = sum(t.get('before_batch', 0) for t in hotstate_timings) / len(hotstate_timings)
        avg_forward = sum(t.get('forward', 0) for t in hotstate_timings) / len(hotstate_timings)
        avg_net = sum(t.get('net', 0) for t in hotstate_timings) / len(hotstate_timings)
        print(f"  Avg before_batch: {avg_before_batch:.2f}ms")
        print(f"  Avg forward: {avg_forward:.2f}ms")
        print(f"  Avg net (forward - before_batch): {avg_net:.2f}ms")
    print()

    if frozen_timings:
        print(f"HotState Frozen (overhead only):")
        frozen_batches = 0
        with open(frozen_log, 'r') as f:
            for line in f:
                if 'Total #batch:' in line:
                    match = re.search(r'Total #batch:\s+(\d+)', line)
                    if match:
                        frozen_batches = int(match.group(1))
        print(f"  Total time: {frozen_total:.2f}s")
        print(f"  Batches: {frozen_batches}")
        if frozen_timings:
            avg_before_batch_frozen = sum(t.get('before_batch', 0) for t in frozen_timings) / len(frozen_timings)
            print(f"  Avg before_batch: {avg_before_batch_frozen:.2f}ms")
        print()

    # Comparison
    if baseline_total and hotstate_total and baseline_batches > 0:
        total_overhead_ms = (hotstate_total - baseline_total) * 1000
        overhead_pct = (total_overhead_ms / (baseline_total * 1000)) * 100 if baseline_total > 0 else 0
        avg_overhead_per_batch = total_overhead_ms / baseline_batches if baseline_batches > 0 else 0

        print(f"Overhead Impact:")
        print(f"  Absolute: {total_overhead_ms:.1f}ms total")
        print(f"  Relative: {overhead_pct:+.2f}%")
        print(f"  Per batch: {avg_overhead_per_batch:.2f}ms avg")
        print(f"  Recommendation:")
        if avg_overhead_per_batch > 5.0:
            print(f"    ⚠ OVERHEAD > 5ms: control plane is expensive")
        elif avg_overhead_per_batch > 2.0:
            print(f"    ⚠ OVERHEAD > 2ms: non-negligible")
        else:
            print(f"    ✓ OVERHEAD < 2ms: acceptable if benefit exists")
        print()

    # Component breakdown
    if hotstate_timings and hotstate_timings[0].get('components'):
        print(f"Control Plane Component Breakdown (avg over {len(hotstate_timings)} batches):")
        all_components = {}
        for t in hotstate_timings:
            if t.get('components'):
                for key, val in t['components'].items():
                    if key not in all_components:
                        all_components[key] = []
                    all_components[key].append(val)

        for key in ['extract', 'record_keys', 'run_epoch', 'scheduler', 'total']:
            if key in all_components:
                avg_val = sum(all_components[key]) / len(all_components[key])
                print(f"  {key:15s}: {avg_val:6.2f}ms")
        print()

    return {
        'baseline_total_s': baseline_total,
        'baseline_batches': baseline_batches,
        'hotstate_total_s': hotstate_total,
        'hotstate_batches': hotstate_batches,
        'overhead_pct': overhead_pct if baseline_total and hotstate_total else 0,
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python analyze_overhead.py <baseline_log> <hotstate_log> [frozen_log]")
        sys.exit(1)

    baseline = sys.argv[1]
    hotstate = sys.argv[2]
    frozen = sys.argv[3] if len(sys.argv) > 3 else None

    analyze(baseline, hotstate, frozen)
