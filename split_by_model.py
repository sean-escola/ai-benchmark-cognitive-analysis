#!/usr/bin/env python3
"""
Split results from a --model all run into per-model summaries.

Usage:
    python split_by_model.py <run_directory>

This reads the output_run_*_transformed.csv files and tier_variability_summary.csv
from an "all" run and creates filtered versions for each model (gemini, claude, gpt).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

BENCHMARK_INFO_CSV = "benchmark_info.csv"


def load_model_benchmarks():
    """
    Load which benchmarks belong to which model(s).

    Returns:
        dict: {model_name: set of benchmark names}
    """
    model_col_map = {
        "gemini": "Gemini 3 Pro",
        "claude": "Claude Opus 4.5",
        "gpt": "GPT 5.2"
    }

    model_benchmarks = {model: set() for model in model_col_map}

    with open(BENCHMARK_INFO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for model, col in model_col_map.items():
                if row.get(col, "").strip().upper() == "TRUE":
                    model_benchmarks[model].add(row["Name"])

    return model_benchmarks


def compute_variability_summary(results, benchmarks):
    """
    Compute tier variability summary for a subset of benchmarks.

    Args:
        results: List of (run_id, rows) tuples
        benchmarks: Set of benchmark names to include

    Returns:
        List of summary dicts
    """
    benchmark_tier_counts = defaultdict(lambda: {'L1': 0, 'L2': 0, 'L3': 0})

    for run_id, rows in results:
        for row in rows:
            benchmark = row.get('Benchmark', '')
            if benchmark not in benchmarks:
                continue
            max_tier = row.get('Max AI Tier', '').strip()
            if benchmark and max_tier in ['L1', 'L2', 'L3']:
                benchmark_tier_counts[benchmark][max_tier] += 1

    total_runs = len(results)
    summary_data = []

    # Maintain order from benchmark_info.csv
    with open(BENCHMARK_INFO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark = row["Name"]
            if benchmark not in benchmarks:
                continue

            counts = benchmark_tier_counts[benchmark]
            l1_count = counts['L1']
            l2_count = counts['L2']
            l3_count = counts['L3']

            max_count = max(l1_count, l2_count, l3_count)
            if max_count == 0:
                mode_tier = "N/A"
            elif l1_count == max_count:
                mode_tier = "L1"
            elif l2_count == max_count:
                mode_tier = "L2"
            else:
                mode_tier = "L3"

            distinct_tiers = sum(1 for count in [l1_count, l2_count, l3_count] if count > 0)

            summary_data.append({
                'Benchmark': benchmark,
                'L1_count': l1_count,
                'L2_count': l2_count,
                'L3_count': l3_count,
                'Total_runs': total_runs,
                'Mode_tier': mode_tier,
                'Distinct_tiers': distinct_tiers,
            })

    return summary_data


def main():
    parser = argparse.ArgumentParser(description='Split --model all results by model')
    parser.add_argument('run_dir', help='Path to the run_all_* directory')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    # Load model -> benchmark mappings
    model_benchmarks = load_model_benchmarks()
    print(f"Loaded benchmark sets:")
    for model, benchmarks in model_benchmarks.items():
        print(f"  {model}: {len(benchmarks)} benchmarks")

    # Load all transformed CSV files
    results = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith('output_run_') and fname.endswith('_transformed.csv'):
            run_id = int(fname.split('_')[2])
            fpath = run_dir / fname
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            results.append((run_id, rows))

    print(f"Loaded {len(results)} runs from {run_dir}")

    if not results:
        print("Error: No transformed CSV files found")
        sys.exit(1)

    # Generate per-model summaries
    for model, benchmarks in model_benchmarks.items():
        summary = compute_variability_summary(results, benchmarks)

        # Save to CSV
        output_path = run_dir / f"tier_variability_summary_{model}.csv"
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['Benchmark', 'L1_count', 'L2_count', 'L3_count', 'Total_runs', 'Mode_tier', 'Distinct_tiers']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)

        print(f"Saved {model} summary to {output_path}")

        # Print L3 benchmarks
        l3_benchmarks = [row for row in summary if row['Mode_tier'] == 'L3']
        if l3_benchmarks:
            print(f"  L3 benchmarks: {', '.join(row['Benchmark'] for row in l3_benchmarks)}")

    # Also print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON: Modal tiers by model")
    print(f"{'='*80}")
    print(f"{'Benchmark':<30} {'Gemini':<10} {'Claude':<10} {'GPT':<10}")
    print("-" * 60)

    # Get all benchmarks
    all_benchmarks = set()
    for benchmarks in model_benchmarks.values():
        all_benchmarks.update(benchmarks)

    # Compute summaries
    summaries = {}
    for model, benchmarks in model_benchmarks.items():
        summary = compute_variability_summary(results, benchmarks)
        summaries[model] = {row['Benchmark']: row for row in summary}

    # Print table
    with open(BENCHMARK_INFO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark = row["Name"]
            if benchmark not in all_benchmarks:
                continue

            cols = [benchmark[:29]]
            for model in ['gemini', 'claude', 'gpt']:
                if benchmark in model_benchmarks[model]:
                    summary_row = summaries[model].get(benchmark, {})
                    mode = summary_row.get('Mode_tier', '?')
                    l3_count = summary_row.get('L3_count', 0)
                    total = summary_row.get('Total_runs', 0)
                    if total > 0:
                        l3_pct = l3_count * 100 // total
                        cols.append(f"{mode} ({l3_pct}%)")
                    else:
                        cols.append(mode)
                else:
                    cols.append("-")

            print(f"{cols[0]:<30} {cols[1]:<10} {cols[2]:<10} {cols[3]:<10}")


if __name__ == '__main__':
    main()
