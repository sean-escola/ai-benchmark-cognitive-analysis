#!/usr/bin/env python3
"""
Script to call GPT-5.2 to analyze benchmarks and categorize cognitive functions by AI tier.

Usage:
    python benchmark_analysis.py --model gemini --runs 5

This script:
1. Calls GPT-5.2 with high reasoning effort
2. Attaches PDFs for context
3. Uses JSON schema for structured outputs
4. Validates the generated data
5. Transforms data to group cognitive functions by AI tier
6. Outputs CSV files
7. Runs multiple times in parallel
8. Computes summary statistics
"""

import argparse
import asyncio
import csv
from datetime import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI

# Configuration
PDF_LIU = "Liu et al., Ch 1.pdf"
PDF_GEMINI = "Gemini 3 Pro - eval info.pdf"
PDF_CLAUDE = "Claude Opus 4.5 - eval info.pdf"
PDF_GPT = "GPT 5.2 - eval info.pdf"
BENCHMARK_INFO_CSV = "benchmark_info.csv"
COGNITIVE_FUNCTIONS_CSV = "cognitive_functions.csv"
FILE_IDS_CACHE = "uploaded_file_ids.json"
MAX_RETRIES = 3



def load_benchmarks(model: str) -> List[Dict[str, str]]:
    """
    Load benchmarks from CSV based on model's evaluation set.

    Args:
        model: One of "gemini", "claude", "gpt", or "all"

    Returns:
        List of dicts with keys: name, website, paper
    """
    model_col_map = {
        "gemini": "Gemini 3 Pro",
        "claude": "Claude Opus 4.5",
        "gpt": "GPT 5.2"
    }

    if model not in model_col_map and model != "all":
        raise ValueError(f"Invalid model value: {model}. Must be one of: gemini, claude, gpt, all")

    benchmarks = []

    with open(BENCHMARK_INFO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if model == "all":
                # Include benchmark if ANY model uses it
                use_benchmark = any(
                    row.get(col, "").strip().upper() == "TRUE"
                    for col in model_col_map.values()
                )
            else:
                # Include benchmark if specific model uses it
                model_col = model_col_map[model]
                use_benchmark = row.get(model_col, "").strip().upper() == "TRUE"

            if use_benchmark:
                benchmarks.append({
                    "name": row["Name"],
                    "website": row["Website"],
                    "paper": row["Paper"]
                })

    return benchmarks


def load_cognitive_functions() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load cognitive function names, descriptions, and tiers from CSV.

    Returns:
        Tuple of (name -> description dict, name -> tier dict)
    """
    descriptions = {}
    tiers = {}
    with open(COGNITIVE_FUNCTIONS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Capacity']
            descriptions[name] = row['Description']
            tiers[name] = row['AI_Tier']
    return descriptions, tiers


class BenchmarkAnalyzer:
    """Main class for running benchmark analysis with GPT-5.2."""

    def __init__(self, output_dir: str, benchmarks: List[Dict[str, str]], exclude_minors: bool = False, debug: bool = False, cognitive_functions: Dict[str, str] = None, func_to_tier: Dict[str, str] = None, include_cf_descriptions: bool = True):
        self._client = None  # Lazy initialization
        self.func_to_tier = func_to_tier or {}  # name -> tier (L1/L2/L3)
        self.cognitive_functions = cognitive_functions or {}  # name -> description
        self.expected_cognitive_functions = sorted(self.cognitive_functions.keys())
        self.include_cf_descriptions = include_cf_descriptions
        self.benchmarks = benchmarks
        self.exclude_minors = exclude_minors
        self.debug = debug
        self.file_ids = {}
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._upload_lock = asyncio.Lock()  # Prevent concurrent uploads

        # Get unique tiers
        unique_tiers = sorted(set(self.func_to_tier.values()))
        print(f"Loaded {len(self.benchmarks)} benchmarks")
        print(f"Loaded {len(self.expected_cognitive_functions)} cognitive functions")
        print(f"Loaded AI tiers: {unique_tiers}")

    @property
    def client(self):
        """Lazy initialization of OpenAI client - only created when needed."""
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI()
        return self._client

    def build_json_schema(self) -> Dict[str, Any]:
        """Build JSON schema for structured outputs."""
        # Define cognitive_functions schema based on exclude_minors flag
        if self.exclude_minors:
            # With --exclude-minors: array of objects with {name, is_minor}
            cf_schema = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "is_minor"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": self.expected_cognitive_functions
                        },
                        "is_minor": {
                            "type": "boolean"
                        }
                    }
                }
            }
        else:
            # Without --exclude-minors: array of strings
            cf_schema = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "enum": self.expected_cognitive_functions
                }
            }

        num_benchmarks = len(self.benchmarks)
        benchmark_names = [b["name"] for b in self.benchmarks]

        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rows": {
                    "type": "array",
                    "minItems": num_benchmarks,
                    "maxItems": num_benchmarks,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "benchmark_name",
                            "description",
                            "cognitive_functions",
                        ],
                        "properties": {
                            "benchmark_name": {
                                "type": "string",
                                "enum": benchmark_names
                            },
                            "description": {
                                "type": "string",
                                "minLength": 10
                            },
                            "cognitive_functions": cf_schema,
                        },
                    },
                }
            },
            "required": ["rows"],
        }

    def build_prompt(self) -> str:
        """Build the prompt for the model."""
        num_evals = len(self.benchmarks)

        # Build cognitive functions section (table with descriptions or simple list)
        if self.include_cf_descriptions:
            cf_table_lines = ["| Cognitive Function | Description |", "|---|---|"]
            for name in self.expected_cognitive_functions:
                desc = self.cognitive_functions.get(name, "")
                desc_escaped = desc.replace("|", "\\|")
                cf_table_lines.append(f"| {name} | {desc_escaped} |")
            cf_section = "\n".join(cf_table_lines)
        else:
            cf_section = "\n".join([f"- {name}" for name in self.expected_cognitive_functions])

        # Build benchmark table
        benchmark_table_lines = []
        for b in self.benchmarks:
            name = b["name"]
            website = b["website"]
            paper = b["paper"] if b["paper"] else "N/A"
            benchmark_table_lines.append(f"  {name} | {website} | {paper}")

        benchmark_table = "Benchmark Name | Website Link | Paper Link\n" + "\n".join(benchmark_table_lines)

        # Build constraint 3 and example based on exclude_minors flag
        cf_source = "table" if self.include_cf_descriptions else "list"
        if self.exclude_minors:
            constraint_3 = f"""3) cognitive_functions must be an array of objects, each with:
   - "name": chosen ONLY from the cognitive functions {cf_source} below
   - "is_minor": false if this is a core cognitive function probed by the benchmark; true if this function is only minimally probed and clearly not the emphasis of the benchmark"""
            cf_example = """[
        {{"name": "A cognitive function", "is_minor": false}},
        {{"name": "Another cognitive function", "is_minor": true}},
        {{"name": "Yet another cognitive function", "is_minor": false}}
      ]"""
        else:
            constraint_3 = f"""3) cognitive_functions must be an array of cognitive function names chosen ONLY from the cognitive functions {cf_source} below"""
            cf_example = """[
        "A cognitive function",
        "Another cognitive function",
        "Yet another cognitive function"
      ]"""

        # Build task description based on whether we include descriptions
        if self.include_cf_descriptions:
            cf_guidance = "Use the descriptions in the cognitive function table below as well as what is typically understood about each cognitive function from a cognitive science perspective when assigning cognitive functions to the benchmark."
        else:
            cf_guidance = ""

        return f"""Return JSON ONLY.

Task: Make a JSON table about a set of common benchmarks used in evaluation of the latest generation of AI models including Gemini 3 Pro, Claude Opus 4.5, and GPT 5.2. See attached documents for information about the evaluations of these models. Fill in a {num_evals}-row table (fixed roster/order) with the following information about each benchmark:
- 2-3 sentence description of each benchmark. In the list of benchmarks below, we give you a website link and (where available) a paper link to begin your research into the details of each benchmark. However, don't use this information alone; also search online for more information about each benchmark.
- a list of cognitive functions that are probed by the benchmark as inferred from your research about the benchmark and from a cognitive science point of view. When assigning a cognitive function to the benchmark, consider whether an AI model requires or strongly benefits from that cognitive function in order to perform well at the benchmark. {cf_guidance}

Hard constraints:
1) Exactly {num_evals} rows, exactly in this order, using exactly these benchmark names.
2) benchmark_name MUST match the first column of the roster EXACTLY.
{constraint_3}
4) Each benchmark must have at least one cognitive function.

Use the attached PDFs (Chapter 1 of Liu et al.) for context.

Fixed roster (exact names in this order with website and paper links):
{benchmark_table}

Cognitive Functions (choose ONLY from this {cf_source}):
{cf_section}

JSON schema (no extra keys):
{{
  "rows": [
    {{
      "benchmark_name": "Benchmark Name",
      "description": "2–3 sentence description.",
      "cognitive_functions": {cf_example}
    }}
    // ... {num_evals} rows total
  ]
}}
"""

    def upload_pdfs(self):
        """Upload PDFs to OpenAI and store file IDs, with caching."""
        if self.file_ids:
            return

        # Define PDF files
        pdf_files = {
            'pdf_liu': PDF_LIU,
            'pdf_gemini': PDF_GEMINI,
            'pdf_claude': PDF_CLAUDE,
            'pdf_gpt': PDF_GPT
        }

        # Try to load cached file IDs
        cache_valid = False
        if os.path.exists(FILE_IDS_CACHE):
            try:
                with open(FILE_IDS_CACHE, 'r') as f:
                    cache_data = json.load(f)

                # Check if all PDFs have same modification time as when cached
                cache_valid = True
                for key, filepath in pdf_files.items():
                    cached_mtime = cache_data.get('mtimes', {}).get(filepath)
                    current_mtime = os.path.getmtime(filepath)
                    if cached_mtime != current_mtime:
                        cache_valid = False
                        break

                if cache_valid:
                    # Validate that cached file IDs still exist on OpenAI servers
                    cached_ids = cache_data.get('file_ids', {})
                    print(f"Validating cached file IDs from {FILE_IDS_CACHE}...")
                    all_valid = True
                    for key, file_id in cached_ids.items():
                        try:
                            self.client.files.retrieve(file_id)
                            print(f"  ✓ {pdf_files[key]} -> {file_id}")
                        except Exception as e:
                            print(f"  ✗ {pdf_files[key]} -> {file_id} (invalid: {e})")
                            all_valid = False
                            break

                    if all_valid:
                        self.file_ids = cached_ids
                        print("All cached file IDs are valid")
                        return
                    else:
                        print("Cached file IDs expired, will re-upload")
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass

        # Upload PDFs if cache invalid or doesn't exist
        print("Uploading PDFs...")
        mtimes = {}

        for key, filepath in pdf_files.items():
            with open(filepath, 'rb') as f:
                upload = self.client.files.create(file=f, purpose="user_data")
                self.file_ids[key] = upload.id
                mtimes[filepath] = os.path.getmtime(filepath)
                print(f"Uploaded {filepath} -> {upload.id}")

        # Save to cache
        cache_data = {
            'file_ids': self.file_ids,
            'mtimes': mtimes
        }
        with open(FILE_IDS_CACHE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved file IDs to {FILE_IDS_CACHE}")

    async def call_gpt52(self) -> Dict[str, Any]:
        """Call GPT-5.2 with high reasoning effort and PDF attachments."""
        # Lazily upload PDFs only when first needed (thread-safe)
        async with self._upload_lock:
            if not self.file_ids:
                self.upload_pdfs()

        print("Calling GPT-5.2 with high reasoning effort...")

        prompt = self.build_prompt()
        json_schema = self.build_json_schema()

        # Use Responses API with reasoning effort and structured outputs
        # Run blocking OpenAI call in executor to not block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.responses.create(
                model="gpt-5.2",
                reasoning={"effort": "high"},
                max_output_tokens=20000,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "benchmark_table",
                        "strict": True,
                        "schema": json_schema,
                    }
                },
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": self.file_ids['pdf_liu']},
                            {"type": "input_file", "file_id": self.file_ids['pdf_gemini']},
                            {"type": "input_file", "file_id": self.file_ids['pdf_claude']},
                            {"type": "input_file", "file_id": self.file_ids['pdf_gpt']},
                            {"type": "input_text", "text": prompt},
                        ],
                    },
                ],
            )
        )

        content = response.output_text or ""
        print(f"Received response ({len(content)} chars)")

        payload = json.loads(content)
        return payload

    def validate_payload(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate benchmark order (only thing schema can't enforce).
        All other validation is guaranteed by the JSON schema.

        Returns:
            (is_valid, error_message)
        """
        rows = payload.get("rows", [])

        # Validate benchmarks are in the correct order
        for i, row in enumerate(rows):
            expected_name = self.benchmarks[i]["name"]
            actual_name = row.get("benchmark_name", "")
            if actual_name != expected_name:
                return False, f"Row {i}: Expected '{expected_name}', got '{actual_name}'"

        return True, ""

    def payload_to_csv_rows(self, payload: Dict[str, Any]) -> List[Dict]:
        """Convert JSON payload to CSV rows with tier groupings and Max AI Tier."""
        csv_rows = []

        # Create lookup dict for benchmark info
        benchmark_lookup = {b["name"]: b for b in self.benchmarks}

        for row in payload["rows"]:
            benchmark_name = row.get("benchmark_name", "")
            benchmark_info = benchmark_lookup.get(benchmark_name, {})
            cfs = row.get("cognitive_functions", [])

            # Group cognitive functions by tier
            tier_groups = {'L1': [], 'L2': [], 'L3': []}

            if self.exclude_minors:
                # With --exclude-minors flag: list of {name, is_minor}
                for cf in cfs:
                    name = cf.get("name", "")
                    is_minor = cf.get("is_minor", False)
                    display_name = f"{name} (minor)" if is_minor else name
                    tier = self.func_to_tier.get(name)
                    if tier:
                        tier_groups[tier].append(display_name)
            else:
                # Without flag: list of strings
                for name in cfs:
                    tier = self.func_to_tier.get(name)
                    if tier:
                        tier_groups[tier].append(name)

            # Format as multi-line string
            tier_lines = [f"{tier}: {', '.join(funcs)}" for tier, funcs in tier_groups.items()]
            cf_by_tier = '\n'.join(tier_lines)

            # Determine max tier (excluding minor functions)
            max_tier = None
            for tier in ['L3', 'L2', 'L1']:  # Check from highest to lowest
                non_minor = [cf for cf in tier_groups[tier] if '(minor)' not in cf]
                if non_minor:
                    max_tier = tier
                    break

            csv_rows.append({
                "Benchmark": benchmark_name,
                "Website": benchmark_info.get("website", ""),
                "Paper": benchmark_info.get("paper", ""),
                "Description": row.get("description", ""),
                "Cognitive Functions": cf_by_tier,
                "Max AI Tier": max_tier
            })

        return csv_rows

    async def run_single_analysis(self, run_id: int) -> Tuple[int, bool, str, List[Dict]]:
        """
        Run a single analysis attempt with retries.
        If output files already exist, load and return them instead.

        Returns:
            (run_id, success, error_message, transformed_rows)
        """
        # Check if output files already exist for this run
        transformed_csv_path = self.output_dir / f"output_run_{run_id}_transformed.csv"
        if transformed_csv_path.exists():
            print(f"\n{'='*60}")
            print(f"Run {run_id}: Loading existing results")
            print(f"{'='*60}")
            try:
                with open(transformed_csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    transformed_rows = list(reader)
                print(f"Loaded {len(transformed_rows)} rows from {transformed_csv_path}")
                return run_id, True, "", transformed_rows
            except Exception as e:
                print(f"Error loading existing file: {e}")
                print("Will re-run this analysis...")

        debug_paths = []  # Track debug files for cleanup

        for attempt in range(MAX_RETRIES):
            try:
                print(f"\n{'='*60}")
                print(f"Run {run_id}, Attempt {attempt + 1}/{MAX_RETRIES}")
                print(f"{'='*60}")

                # Call GPT-5.2
                payload = await self.call_gpt52()

                # Save raw JSON payload for debugging
                debug_json_path = self.output_dir / f"output_run_{run_id}_attempt_{attempt + 1}_payload.json"
                with open(debug_json_path, 'w') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"Saved debug payload to {debug_json_path}")
                debug_paths.append(debug_json_path)

                # Validate payload
                valid, error = self.validate_payload(payload)
                if not valid:
                    print(f"Validation failed: {error}")
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                        continue
                    return run_id, False, f"Payload validation failed: {error}", []

                print("Payload validated successfully")

                # Convert payload to CSV rows with tier groupings
                transformed_rows = self.payload_to_csv_rows(payload)

                # Save CSV
                transformed_csv_path = self.output_dir / f"output_run_{run_id}_transformed.csv"
                with open(transformed_csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=transformed_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(transformed_rows)
                print(f"Saved transformed CSV to {transformed_csv_path}")

                # Clean up debug files on success
                for debug_path in debug_paths:
                    if debug_path.exists():
                        debug_path.unlink()
                        print(f"Cleaned up debug file: {debug_path}")

                print(f"Run {run_id} completed successfully!")
                return run_id, True, "", transformed_rows

            except Exception as e:
                print(f"Error in run {run_id}, attempt {attempt + 1}: {str(e)}")

                # Print traceback only in debug mode
                if self.debug:
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()

                if attempt < MAX_RETRIES - 1:
                    print("Retrying...")
                    continue
                return run_id, False, f"Exception: {str(e)}", []

    async def run_parallel_analysis(self, num_runs: int):
        """Run multiple analyses in parallel."""
        print(f"\nStarting {num_runs} parallel runs...")

        # Save prompt to output directory for future verifiability
        if self.output_dir:
            prompt_path = self.output_dir / "prompt.txt"
            with open(prompt_path, 'w') as f:
                f.write(self.build_prompt())
            print(f"Saved prompt to {prompt_path}")

        # Run all analyses in parallel (PDFs will be uploaded lazily if needed)
        tasks = [self.run_single_analysis(i + 1) for i in range(num_runs)]
        results = await asyncio.gather(*tasks)

        # Filter successful runs
        successful_results = [(run_id, rows) for run_id, success, _, rows in results if success]
        failed_results = [(run_id, error) for run_id, success, error, _ in results if not success]

        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(successful_results)}/{num_runs} runs successful")
        print(f"{'='*60}")

        if failed_results:
            print("\nFailed runs:")
            for run_id, error in failed_results:
                print(f"  Run {run_id}: {error}")

        if not successful_results:
            print("\nNo successful runs. Cannot compute statistics.")
            return

        # Compute statistics
        self.compute_statistics(successful_results)

        # Generate variability summary CSV
        self.generate_variability_summary(successful_results)

    def compute_statistics(self, results: List[Tuple[int, List[Dict]]]):
        """Compute summary statistics across runs."""
        lines = []  # Collect output for both screen and file

        lines.append(f"\n{'='*60}")
        lines.append("STATISTICS")
        lines.append(f"{'='*60}")

        # Count assignments per tier per run
        tier_counts_per_run = []
        l3_assignments = defaultdict(lambda: defaultdict(list))  # benchmark -> run_id -> [functions]

        for run_id, rows in results:
            tier_counts = {'L1': 0, 'L2': 0, 'L3': 0}

            for row in rows:
                max_tier = row.get('Max AI Tier', '').strip()
                if max_tier in tier_counts:
                    tier_counts[max_tier] += 1

                # Track L3 assignments
                if max_tier == 'L3':
                    benchmark = row.get('Benchmark', '')
                    cf_text = row.get('Cognitive Functions', '')

                    # Extract L3 functions
                    l3_line = [line for line in cf_text.split('\n') if line.startswith('L3:')]
                    if l3_line:
                        l3_funcs = l3_line[0].replace('L3:', '').strip()
                        # Remove (minor) functions for this analysis
                        funcs = [f.strip() for f in l3_funcs.split(',') if f.strip() and '(minor)' not in f]
                        l3_assignments[benchmark][run_id] = funcs

            tier_counts_per_run.append(tier_counts)

        # Compute mean and std err for each tier
        lines.append("\nAI Tier Assignment Statistics:")
        lines.append(f"{'Tier':<10} {'Mean':<10} {'Std Err':<10}")
        lines.append("-" * 30)

        for tier in ['L1', 'L2', 'L3']:
            counts = [tc[tier] for tc in tier_counts_per_run]
            mean = np.mean(counts)
            stderr = np.std(counts, ddof=1) / np.sqrt(len(counts)) if len(counts) > 1 else 0
            lines.append(f"{tier:<10} {mean:<10.2f} {stderr:<10.2f}")

        # Report L3 assignments
        if l3_assignments:
            lines.append(f"\n{'='*60}")
            lines.append("L3 ASSIGNMENTS DETAIL")
            lines.append(f"{'='*60}")

            for benchmark in sorted(l3_assignments.keys()):
                lines.append(f"\n{benchmark}:")
                for run_id in sorted(l3_assignments[benchmark].keys()):
                    funcs = l3_assignments[benchmark][run_id]
                    if funcs:
                        lines.append(f"  Run {run_id}: {', '.join(funcs)}")
                    else:
                        lines.append(f"  Run {run_id}: (only minor L3 functions)")

        # Print to screen
        for line in lines:
            print(line)

        # Write to file
        if self.output_dir:
            stats_path = self.output_dir / "statistics.txt"
            with open(stats_path, 'w') as f:
                f.write('\n'.join(lines))
            print(f"\nSaved statistics to {stats_path}")

    def generate_variability_summary(self, results: List[Tuple[int, List[Dict]]]):
        """Generate CSV summary of tier assignment variability across runs."""
        print(f"\n{'='*60}")
        print("GENERATING VARIABILITY SUMMARY CSV")
        print(f"{'='*60}")

        # Collect tier assignments per benchmark
        benchmark_tier_counts = defaultdict(lambda: {'L1': 0, 'L2': 0, 'L3': 0})

        for run_id, rows in results:
            for row in rows:
                benchmark = row.get('Benchmark', '')
                max_tier = row.get('Max AI Tier', '').strip()
                if benchmark and max_tier in ['L1', 'L2', 'L3']:
                    benchmark_tier_counts[benchmark][max_tier] += 1

        # Calculate additional statistics
        total_runs = len(results)
        summary_data = []
        benchmark_names = [b["name"] for b in self.benchmarks]

        for benchmark in benchmark_names:
            counts = benchmark_tier_counts[benchmark]
            l1_count = counts['L1']
            l2_count = counts['L2']
            l3_count = counts['L3']

            # Find mode (most common tier)
            max_count = max(l1_count, l2_count, l3_count)
            if max_count == 0:
                mode_tier = "N/A"
            elif l1_count == max_count:
                mode_tier = "L1"
            elif l2_count == max_count:
                mode_tier = "L2"
            else:
                mode_tier = "L3"

            # Calculate number of distinct tiers assigned
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

        # Save to CSV
        variability_csv_path = self.output_dir / "tier_variability_summary.csv"
        with open(variability_csv_path, 'w', newline='') as f:
            fieldnames = ['Benchmark', 'L1_count', 'L2_count', 'L3_count', 'Total_runs', 'Mode_tier', 'Distinct_tiers']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)

        print(f"Tier variability summary saved to {variability_csv_path}")

        # Print summary of benchmarks with variability
        print("\nBenchmarks with tier variability (assigned to multiple tiers):")
        has_variability = False
        for row in summary_data:
            if row['Distinct_tiers'] > 1:
                has_variability = True
                print(f"  {row['Benchmark']}: L1={row['L1_count']}, L2={row['L2_count']}, L3={row['L3_count']} (mode={row['Mode_tier']})")

        if not has_variability:
            print("  (none - all benchmarks consistently assigned to same tier)")


async def main():
    parser = argparse.ArgumentParser(description='Analyze benchmarks with GPT-5.2')
    parser.add_argument('--model', type=str, required=True, choices=['gemini', 'claude', 'gpt', 'all'],
                        help='Which model\'s benchmark set to analyze: gemini, claude, gpt, or all (union of all three)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of parallel runs (default: 1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Existing output directory to resume/extend (default: create new timestamped directory)')
    parser.add_argument('--exclude-minors', action='store_true',
                        help='Include is_minor flag for cognitive functions (default: False)')
    parser.add_argument('--print-prompt', action='store_true',
                        help='Print the prompt and exit without running analysis')
    parser.add_argument('--debug', action='store_true',
                        help='Print detailed error tracebacks (default: False)')
    parser.add_argument('--no-cf-descriptions', action='store_true',
                        help='Use simple list of cognitive function names instead of table with descriptions')
    args = parser.parse_args()

    # Check CSV files exist first (before loading)
    if not os.path.exists(BENCHMARK_INFO_CSV):
        print(f"Error: Required file not found: {BENCHMARK_INFO_CSV}")
        sys.exit(1)

    if not os.path.exists(COGNITIVE_FUNCTIONS_CSV):
        print(f"Error: Required file not found: {COGNITIVE_FUNCTIONS_CSV}")
        sys.exit(1)

    # Load benchmarks based on model flag
    benchmarks = load_benchmarks(args.model)
    print(f"Loaded {len(benchmarks)} benchmarks for {args.model}")

    # Load cognitive function descriptions and tier mappings
    cognitive_functions, func_to_tier = load_cognitive_functions()
    print(f"Loaded {len(cognitive_functions)} cognitive function descriptions")

    include_cf_descriptions = not args.no_cf_descriptions

    # If --print-prompt, just print the prompt and exit
    if args.print_prompt:
        # Create a temporary analyzer just to build the prompt (no output directory needed)
        temp_analyzer = BenchmarkAnalyzer(None, benchmarks, args.exclude_minors, args.debug, cognitive_functions, func_to_tier, include_cf_descriptions)
        prompt = temp_analyzer.build_prompt()
        print("\n" + "="*80)
        print("PROMPT")
        print("="*80)
        print(prompt)
        print("="*80)
        return

    if args.runs < 1:
        print("Error: Number of runs must be at least 1")
        sys.exit(1)

    # Check PDF files exist (only needed for actual API calls, not --print-prompt)
    pdf_files = [PDF_LIU, PDF_GEMINI, PDF_CLAUDE, PDF_GPT]
    for file_path in pdf_files:
        if not os.path.exists(file_path):
            print(f"Error: Required PDF not found: {file_path}")
            sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        print(f"Using existing output directory: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        minors_suffix = "no-minors" if args.exclude_minors else "with-minors"
        output_dir = f"run_{args.model}_{minors_suffix}_{timestamp}"
        print(f"Creating new output directory: {output_dir}")

    # Run analysis
    analyzer = BenchmarkAnalyzer(output_dir, benchmarks, exclude_minors=args.exclude_minors, debug=args.debug, cognitive_functions=cognitive_functions, func_to_tier=func_to_tier, include_cf_descriptions=include_cf_descriptions)
    await analyzer.run_parallel_analysis(args.runs)


if __name__ == "__main__":
    asyncio.run(main())
