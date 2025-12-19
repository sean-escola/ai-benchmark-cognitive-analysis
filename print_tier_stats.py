#!/usr/bin/env python3
"""Print AI tier assignment statistics from a run directory."""

import csv
import sys
from pathlib import Path
from collections import Counter
import numpy as np

def get_tier_counts_from_run(csv_file):
    """Count L1, L2, L3 assignments in a single run's transformed CSV."""
    tiers = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tiers.append(row['Max AI Tier'])

    counts = Counter(tiers)
    return {
        'L1': counts.get('L1', 0),
        'L2': counts.get('L2', 0),
        'L3': counts.get('L3', 0)
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_tier_stats.py <run_directory>")
        print("Example: python print_tier_stats.py run_gemini_no-minors_20251219_132748")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Error: Directory '{run_dir}' not found")
        sys.exit(1)

    # Find all transformed CSV files
    csv_files = sorted(run_dir.glob('output_run_*_transformed.csv'))

    if not csv_files:
        print(f"Error: No output_run_*_transformed.csv files found in {run_dir}")
        sys.exit(1)

    # Collect tier counts from each run
    all_counts = []
    for csv_file in csv_files:
        counts = get_tier_counts_from_run(csv_file)
        all_counts.append(counts)

    # Calculate statistics
    l1_counts = [c['L1'] for c in all_counts]
    l2_counts = [c['L2'] for c in all_counts]
    l3_counts = [c['L3'] for c in all_counts]

    l1_mean = np.mean(l1_counts)
    l2_mean = np.mean(l2_counts)
    l3_mean = np.mean(l3_counts)

    l1_stderr = np.std(l1_counts, ddof=1) / np.sqrt(len(l1_counts)) if len(l1_counts) > 1 else 0
    l2_stderr = np.std(l2_counts, ddof=1) / np.sqrt(len(l2_counts)) if len(l2_counts) > 1 else 0
    l3_stderr = np.std(l3_counts, ddof=1) / np.sqrt(len(l3_counts)) if len(l3_counts) > 1 else 0

    # Print formatted output
    print()
    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print()
    print("AI Tier Assignment Statistics:")
    print("Tier       Mean       Std Err")
    print("-" * 30)
    print(f"L1         {l1_mean:<10.2f} {l1_stderr:.2f}")
    print(f"L2         {l2_mean:<10.2f} {l2_stderr:.2f}")
    print(f"L3         {l3_mean:<10.2f} {l3_stderr:.2f}")
    print()

if __name__ == '__main__':
    main()
