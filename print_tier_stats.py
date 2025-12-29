#!/usr/bin/env python3
"""
Print tier statistics from a benchmark analysis run.

Usage:
    python print_tier_stats.py <run_directory>
    python print_tier_stats.py <run_directory> --split

Without --split: prints stats for all benchmarks
With --split: prints stats for all benchmarks plus per-model subsets
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCHMARK_INFO_CSV = "benchmark_info.csv"


def load_model_benchmarks():
    """Load which benchmarks belong to which model(s)."""
    model_col_map = {
        "gemini": "Gemini 3 Pro",
        "claude": "Claude Opus 4.5",
        "gpt": "GPT 5.2"
    }

    model_benchmarks = {model: set() for model in model_col_map}
    all_benchmarks = set()

    with open(BENCHMARK_INFO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for model, col in model_col_map.items():
                if row.get(col, "").strip().upper() == "TRUE":
                    model_benchmarks[model].add(row["Name"])
                    all_benchmarks.add(row["Name"])

    return model_benchmarks, all_benchmarks


def load_results(run_dir):
    """Load all transformed CSV results from a directory."""
    results = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith('output_run_') and fname.endswith('_transformed.csv'):
            run_id = int(fname.split('_')[2])
            fpath = run_dir / fname
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            results.append((run_id, rows))
    return results


def extract_l3_functions(results, benchmark_filter=None):
    """
    Extract L3 functions from results.

    Args:
        results: List of (run_id, rows)
        benchmark_filter: Optional set of benchmarks to include

    Returns:
        dict: {benchmark: {run_id: [functions]}}
    """
    l3_by_benchmark = defaultdict(lambda: defaultdict(list))

    for run_id, rows in results:
        for row in rows:
            benchmark = row.get('Benchmark', '')
            if benchmark_filter and benchmark not in benchmark_filter:
                continue

            max_tier = row.get('Max AI Tier', '').strip()
            if max_tier == 'L3':
                cf_text = row.get('Cognitive Functions', '')
                l3_line = [line for line in cf_text.split('\n') if line.startswith('L3:')]
                if l3_line:
                    l3_funcs = l3_line[0].replace('L3:', '').strip()
                    # Remove (minor) functions
                    funcs = [f.strip() for f in l3_funcs.split(',') if f.strip() and '(minor)' not in f]
                    l3_by_benchmark[benchmark][run_id] = funcs

    return l3_by_benchmark


def compute_tier_stats(results, benchmark_filter=None):
    """
    Compute tier statistics for a set of results.

    Args:
        results: List of (run_id, rows)
        benchmark_filter: Optional set of benchmarks to include

    Returns:
        tuple: (tier_counts_per_run, l3_assignments, benchmark_tier_counts)
    """
    tier_counts_per_run = []
    l3_assignments = extract_l3_functions(results, benchmark_filter)
    benchmark_tier_counts = defaultdict(lambda: {'L1': 0, 'L2': 0, 'L3': 0})

    for run_id, rows in results:
        tier_counts = {'L1': 0, 'L2': 0, 'L3': 0}

        for row in rows:
            benchmark = row.get('Benchmark', '')
            if benchmark_filter and benchmark not in benchmark_filter:
                continue

            max_tier = row.get('Max AI Tier', '').strip()
            if max_tier in tier_counts:
                tier_counts[max_tier] += 1
                benchmark_tier_counts[benchmark][max_tier] += 1

        tier_counts_per_run.append(tier_counts)

    return tier_counts_per_run, l3_assignments, benchmark_tier_counts


def print_stats(name, tier_counts_per_run, l3_assignments, benchmark_tier_counts, n_runs, verbose=True):
    """Print statistics for a set of results."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    # Tier assignment statistics
    print(f"\nAI Tier Assignment Statistics ({n_runs} runs):")
    print(f"{'Tier':<10} {'Mean':<10} {'Std Err':<10} {'%':<10}")
    print("-" * 40)

    total_benchmarks = sum(tier_counts_per_run[0].values()) if tier_counts_per_run else 0

    for tier in ['L1', 'L2', 'L3']:
        counts = [tc[tier] for tc in tier_counts_per_run]
        mean = np.mean(counts)
        stderr = np.std(counts, ddof=1) / np.sqrt(len(counts)) if len(counts) > 1 else 0
        pct = mean / total_benchmarks * 100 if total_benchmarks > 0 else 0
        print(f"{tier:<10} {mean:<10.2f} {stderr:<10.2f} {pct:<10.1f}%")

    # Modal L3 benchmarks
    modal_l3 = []
    for benchmark, counts in benchmark_tier_counts.items():
        total = sum(counts.values())
        if total > 0:
            l3_pct = counts['L3'] / total * 100
            mode = max(counts, key=counts.get)
            if mode == 'L3':
                # Tally cognitive functions for this benchmark
                func_counts = defaultdict(int)
                if benchmark in l3_assignments:
                    for run_id, funcs in l3_assignments[benchmark].items():
                        for func in funcs:
                            func_counts[func] += 1
                modal_l3.append((benchmark, l3_pct, counts['L3'], total, func_counts))

    if modal_l3 and verbose:
        print(f"\nModal L3 Benchmarks:")
        for benchmark, pct, l3_count, total, func_counts in sorted(modal_l3, key=lambda x: -x[1]):
            func_str = ", ".join([f"{func} x{count}" for func, count in sorted(func_counts.items(), key=lambda x: -x[1])])
            print(f"  {benchmark}: {l3_count}/{total} ({pct:.0f}%) — {func_str}")

    # L3 function breakdown
    if l3_assignments and verbose:
        print(f"\nL3 Function Frequency (total):")
        func_counts = defaultdict(int)
        for benchmark, run_funcs in l3_assignments.items():
            for run_id, funcs in run_funcs.items():
                for func in funcs:
                    func_counts[func] += 1

        for func, count in sorted(func_counts.items(), key=lambda x: -x[1]):
            print(f"  {func}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Print tier statistics from benchmark analysis run')
    parser.add_argument('run_dir', help='Path to the run directory')
    parser.add_argument('--split', action='store_true', help='Also show per-model statistics')
    parser.add_argument('--quiet', '-q', action='store_true', help='Only show summary stats, not L3 details')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    # Load benchmark info
    model_benchmarks, all_benchmarks = load_model_benchmarks()
    print(f"Expected benchmarks: {len(all_benchmarks)}")
    print(f"  Gemini: {len(model_benchmarks['gemini'])}")
    print(f"  Claude: {len(model_benchmarks['claude'])}")
    print(f"  GPT: {len(model_benchmarks['gpt'])}")

    # Load results
    results = load_results(run_dir)
    if not results:
        print(f"Error: No transformed CSV files found in {run_dir}")
        sys.exit(1)

    n_runs = len(results)
    print(f"\nLoaded {n_runs} runs from {run_dir}")

    # Check that all benchmarks are present
    found_benchmarks = set()
    for run_id, rows in results:
        for row in rows:
            found_benchmarks.add(row.get('Benchmark', ''))

    missing = all_benchmarks - found_benchmarks
    extra = found_benchmarks - all_benchmarks

    if missing:
        print(f"\nWarning: Missing benchmarks: {missing}")
    if extra:
        print(f"\nWarning: Extra benchmarks: {extra}")

    if missing:
        print("\nError: Results don't contain all expected benchmarks.")
        print("This script requires --model all results.")
        sys.exit(1)

    # Compute and print overall stats
    tier_counts, l3_assignments, benchmark_counts = compute_tier_stats(results)
    print_stats("ALL BENCHMARKS", tier_counts, l3_assignments, benchmark_counts, n_runs, verbose=not args.quiet)

    # Per-model stats if --split
    if args.split:
        for model in ['gemini', 'claude', 'gpt']:
            benchmarks = model_benchmarks[model]
            tier_counts, l3_assignments, benchmark_counts = compute_tier_stats(results, benchmarks)
            print_stats(f"{model.upper()} BENCHMARKS ({len(benchmarks)} benchmarks)",
                       tier_counts, l3_assignments, benchmark_counts, n_runs, verbose=not args.quiet)


if __name__ == '__main__':
    main()
