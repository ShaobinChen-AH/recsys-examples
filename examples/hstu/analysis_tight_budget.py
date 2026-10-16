#!/usr/bin/env python3
"""
Analysis script for tight budget experiments.
Extracts latency, budget allocation, and constraint pressure metrics.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def extract_metrics_from_log(log_path: str) -> Dict:
    """Extract budget allocation and latency metrics from log file."""
    metrics = {
        "total_hbm_budget_bytes": 0,
        "hotstate_enabled": False,
        "embedding_hbm_bytes": 0,
        "kv_hbm_bytes": 0,
        "kv_page_bytes": 0,
        "blocks_in_primary_pool": 0,
        "total_batches": 0,
        "total_time_sec": 0,
        "mean_latency_ms": 0,
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Extract budget allocation metrics
        budget_pattern = r'\[BUDGET_ALLOCATION\] (\w+)=(\d+)'
        for match in re.finditer(budget_pattern, content):
            key, value = match.groups()
            if key in metrics:
                metrics[key] = int(value)

        # Extract hotstate_enabled flag
        hotstate_pattern = r'\[BUDGET_ALLOCATION\] hotstate_enabled=(True|False)'
        match = re.search(hotstate_pattern, content)
        if match:
            metrics["hotstate_enabled"] = match.group(1) == "True"

        # Extract performance metrics
        total_pattern = r'Total #batch:\s*(\d+)'
        match = re.search(total_pattern, content)
        if match:
            metrics["total_batches"] = int(match.group(1))

        time_pattern = r'Total time\(s\):\s*([\d\.]+)'
        match = re.search(time_pattern, content)
        if match:
            metrics["total_time_sec"] = float(match.group(1))

        # Calculate mean latency if we have batch count
        if metrics["total_batches"] > 0 and metrics["total_time_sec"] > 0:
            metrics["mean_latency_ms"] = (metrics["total_time_sec"] * 1000) / metrics["total_batches"]

    except Exception as e:
        print(f"[ERROR] Failed to parse {log_path}: {e}", file=sys.stderr)

    return metrics


def analyze_tight_budget_results(results_dir: str) -> None:
    """Analyze tight budget comparison results."""

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[ERROR] Results directory not found: {results_dir}", file=sys.stderr)
        return

    logs_dir = results_path / "logs"
    if not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}", file=sys.stderr)
        return

    # Group logs by budget and mode
    budgets = {}
    for log_file in sorted(logs_dir.glob("*.log")):
        # Parse filename: original_64mb_run1.log or hotstate_128mb_run2.log
        match = re.match(r"(original|hotstate)_(\d+)mb_run(\d+)\.log", log_file.name)
        if not match:
            continue

        mode, budget_mb, run = match.groups()
        budget_key = f"{budget_mb}MB"

        if budget_key not in budgets:
            budgets[budget_key] = {"original": [], "hotstate": []}

        metrics = extract_metrics_from_log(str(log_file))
        budgets[budget_key][mode].append(metrics)

    # Print analysis
    print("\n" + "="*100)
    print("TIGHT BUDGET CONSTRAINT BINDING ANALYSIS")
    print("="*100)
    print(f"Working Set (KuaiRand-1K): ~32.8 MB")
    print(f"KV Page Size: ~256 KB (8 layers × 2 × 32 tokens × 4 heads × 128 dims × 2 bytes)")
    print("="*100 + "\n")

    for budget_key in sorted(budgets.keys(), key=lambda x: int(x.split("M")[0])):
        original_runs = budgets[budget_key]["original"]
        hotstate_runs = budgets[budget_key]["hotstate"]

        if not original_runs or not hotstate_runs:
            print(f"[SKIP] {budget_key}: Missing data")
            continue

        budget_mb = int(budget_key.split("M")[0])
        utilization = (32.8 * 100) / budget_mb

        print(f"\nBudGET: {budget_key} ({utilization:.1f}% KV working set utilization)")
        print("-" * 100)

        # Budget allocation analysis
        orig_first = original_runs[0]
        hotstate_first = hotstate_runs[0]

        print(f"  Budget Allocation (Run 1):")
        print(f"    Original:  embedding={orig_first['embedding_hbm_bytes']:>12,} bytes, kv={orig_first['kv_hbm_bytes']:>12,} bytes, blocks={orig_first['blocks_in_primary_pool']:>4}")
        print(f"    HotState:  embedding={hotstate_first['embedding_hbm_bytes']:>12,} bytes, kv={hotstate_first['kv_hbm_bytes']:>12,} bytes, blocks={hotstate_first['blocks_in_primary_pool']:>4}")

        # Performance metrics
        orig_latencies = [m["mean_latency_ms"] for m in original_runs if m["mean_latency_ms"] > 0]
        hotstate_latencies = [m["mean_latency_ms"] for m in hotstate_runs if m["mean_latency_ms"] > 0]

        if orig_latencies and hotstate_latencies:
            orig_mean = sum(orig_latencies) / len(orig_latencies)
            hotstate_mean = sum(hotstate_latencies) / len(hotstate_latencies)
            improvement = ((orig_mean - hotstate_mean) / orig_mean) * 100

            print(f"\n  Latency (mean of {len(orig_latencies)} runs):")
            print(f"    Original:  {orig_mean:.3f} ms")
            print(f"    HotState:  {hotstate_mean:.3f} ms")
            print(f"    Improvement: {improvement:+.2f}%", end="")

            if improvement > 0.5:
                print(" ✓ MEANINGFUL")
            elif improvement > -0.5:
                print(" ≈ NOISE")
            else:
                print(" ✗ REGRESSION")
        else:
            print(f"\n  [SKIP] Latency metrics not available")

        print()

    print("\n" + "="*100)
    print("KEY FINDINGS:")
    print("  - If constraint binding is working: HotState should show improvement at tight budgets")
    print("  - If constraint is still not binding: All budgets will show similar results (no improvement)")
    print("  - Target utilization: >50% to force meaningful constraint pressure")
    print("="*100 + "\n")


if __name__ == "__main__":
    results_dir = "results/tight_budget_comparison"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    analyze_tight_budget_results(results_dir)
