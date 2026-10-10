import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary = defaultdict(lambda: {"original": [], "hotstate": []})

    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        parts = json_file.stem.split("_")
        system = parts[0]  # "original" or "hotstate"
        budget = "_".join(parts[1:-1])  # e.g., "0.4gib"

        summary[budget][system].append(data["mean_latency_ms"])

    # Write CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w") as f:
        f.write("Budget,Original_Mean_ms,HotState_Mean_ms,Reduction_%,P95_Original,P95_HotState,P99_Original,P99_HotState\n")
        for budget in sorted(summary.keys()):
            orig_latencies = summary[budget]["original"]
            hotstate_latencies = summary[budget]["hotstate"]

            if not orig_latencies or not hotstate_latencies:
                continue

            orig_mean = statistics.mean(orig_latencies)
            hotstate_mean = statistics.mean(hotstate_latencies)
            reduction = ((orig_mean - hotstate_mean) / orig_mean) * 100 if orig_mean > 0 else 0

            # Compute percentiles across all runs
            orig_all = sorted(orig_latencies)
            hotstate_all = sorted(hotstate_latencies)

            def percentile(values, q):
                if not values:
                    return 0.0
                if len(values) == 1:
                    return float(values[0])
                sorted_values = sorted(values)
                index = max(0, min(len(sorted_values) - 1, int(q * len(sorted_values))))
                return float(sorted_values[index])

            p95_orig = percentile(orig_all, 0.95)
            p95_hotstate = percentile(hotstate_all, 0.95)
            p99_orig = percentile(orig_all, 0.99)
            p99_hotstate = percentile(hotstate_all, 0.99)

            f.write(f"{budget},{orig_mean:.3f},{hotstate_mean:.3f},{reduction:.2f},{p95_orig:.3f},{p95_hotstate:.3f},{p99_orig:.3f},{p99_hotstate:.3f}\n")

    print(f"Results aggregated to {args.output_csv}")

if __name__ == "__main__":
    main()
