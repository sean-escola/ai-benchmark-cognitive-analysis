# AI Benchmark Cognitive Function Analysis

This repository uses GPT-5.2 with high reasoning effort to analyze AI benchmarks and categorize them by cognitive functions and AI tiers, based on the cognitive neuroscience framework from [Liu et al., 2025](https://arxiv.org/abs/2504.01990).

## Overview

The script analyzes common benchmarks used in evaluation of the latest generation of AI models (Gemini 3 Pro, Claude Opus 4.5, and GPT 5.2) and assigns cognitive functions to each benchmark. It then groups these cognitive functions by AI tier (i.e., how capable current models are):
- **L1**: Highly capable -- E.g., basic perceptual and language processing
- **L2**: Partially capable -- E.g., executive functions, planning, reasoning, memory
- **L3**: Poorly capable -- E.g., higher-order cognition (cognitive flexibility, theory of mind, self-reflection, etc.)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Print the prompt to see what will be sent to GPT-5.2
python benchmark_analysis.py --model all --print-prompt

# Run analysis (50 parallel runs for statistical robustness)
python benchmark_analysis.py --model all --runs 50 --exclude-minors
```

## Usage

### Basic Usage

```bash
python benchmark_analysis.py --model {gemini|claude|gpt|all} [OPTIONS]
```

### Required Arguments

- `--model {gemini,claude,gpt,all}`: Select which model's benchmark set to analyze
  - `gemini`: Benchmarks used in Gemini 3 Pro's launch documents (20 benchmarks)
  - `claude`: Benchmarks used in Claude Opus 4.5's launch documents (20 benchmarks)
  - `gpt`: Benchmarks used in GPT 5.2's launch documents (21 benchmarks)
  - `all`: Union of all three sets (37 benchmarks) - recommended for best calibration

### Optional Arguments

- `--runs N`: Number of parallel runs (default: 1)
  - Use 25+ runs for statistical analysis
- `--exclude-minors`: Include is_minor flag for cognitive functions
  - Without flag (default): All cognitive functions treated equally
  - With flag: Distinguishes core vs. minimally-probed functions, excludes minor functions from AI tier calculation
- `--output-dir PATH`: Resume/extend existing run directory
- `--print-prompt`: Print the prompt and exit (useful for debugging)
- `--no-cf-descriptions`: Use simple list of cognitive function names instead of table with descriptions (default: include descriptions)

### Examples

```bash
# All benchmarks with minor distinction (recommended)
python benchmark_analysis.py --model all --runs 50 --exclude-minors

# Print prompt without running
python benchmark_analysis.py --model all --print-prompt

# Model-specific benchmarks (may have different calibration due to batch effects)
python benchmark_analysis.py --model gemini --runs 25

# Resume/extend existing run
python benchmark_analysis.py --model all --runs 10 --output-dir run_all
```

### Splitting Results by Model

After running with `--model all`, use `split_by_model.py` to generate per-model summaries:

```bash
python split_by_model.py run_all
```

This creates `tier_variability_summary_{gemini,claude,gpt}.csv` files filtered to each model's benchmark set.

### Printing Statistics

Use `print_tier_stats.py` to view tier statistics from a run:

```bash
# Overall statistics
python print_tier_stats.py run_all

# Overall + per-model breakdown
python print_tier_stats.py run_all --split

# Summary only (no L3 function details)
python print_tier_stats.py run_all --split --quiet
```

This requires `--model all` results (all 37 benchmarks).

## Input Files

The repository includes all necessary input files:

- **`benchmark_info.csv`**: Master list of 37 benchmarks with website/paper links
  - Contains the union of all benchmarks from Gemini 3 Pro, Claude Opus 4.5, and GPT 5.2 launch documents
  - Each model's column (TRUE/FALSE) indicates which benchmarks that model's evaluation used
- **`cognitive_functions.csv`**: 34 cognitive functions with AI tiers and descriptions
  - Columns: Capacity, AI_Tier, Description
  - Descriptions are included in the prompt by default (use `--no-cf-descriptions` to exclude)
- **`Liu et al., Ch 1.pdf`**: First chapter of [Liu et al., 2025](https://arxiv.org/abs/2504.01990) - Cognitive neuroscience framework reference
- **`Gemini 3 Pro - eval info.pdf`**: Gemini 3 Pro evaluation document
- **`Claude Opus 4.5 - eval info.pdf`**: Claude Opus 4.5 evaluation document
- **`GPT 5.2 - eval info.pdf`**: GPT 5.2 evaluation document

All PDFs are uploaded to OpenAI on first run and file IDs are cached in `uploaded_file_ids.json` for reuse.

## Output Structure

### Directory Naming

When you run the script, output directories are created with timestamps: `run_{model}_{minors-mode}_{timestamp}`

Example: `run_gemini_with-minors_20251218_172548/`

The pre-generated result directories in this repository have had timestamps removed for cleanliness.

### Output Files

- `prompt.txt`: The exact prompt sent to GPT-5.2 (saved once at start)

For each successful run N:
- `output_run_N_transformed.csv`: Analysis grouped by AI tier with max tier
  - Columns: Benchmark, Website, Paper, Description, Cognitive Functions, Max AI Tier
  - Cognitive Functions column shows: "L1: ...\nL2: ...\nL3: ..."
  - Max AI Tier column shows highest tier (L1/L2/L3)

After all runs:
- `tier_variability_summary.csv`: Statistics on AI tier assignment variability

### Example Output

```
================================================================================
STATISTICS
================================================================================

AI Tier Assignment Statistics:
Tier       Mean       Std Err
------------------------------
L1         0.00       0.00
L2         11.50      0.50
L3         6.50       0.50

================================================================================
L3 ASSIGNMENTS DETAIL
================================================================================

ARC-AGI:
  Run 1: Cognitive Flexibility
  Run 2: Cognitive Flexibility

FACTS Benchmark Suite:
  Run 1: Inhibitory Control, Self-reflection
  Run 2: Inhibitory Control, Self-reflection
```

## Pre-Generated Results

The repository includes pre-generated results in `run_all/`, created with:

```bash
python benchmark_analysis.py --model all --runs 50 --exclude-minors --output-dir run_all
```

This directory contains:
- `output_run_N_transformed.csv`: Results from each of 50 runs
- `tier_variability_summary.csv`: Aggregated statistics for all 37 benchmarks
- `tier_variability_summary_{gemini,claude,gpt}.csv`: Per-model filtered summaries
- `prompt.txt`: The exact prompt sent to GPT-5.2
- `statistics.txt`: Summary statistics and L3 assignment details

Using `--model all` (37 benchmarks together) provides better calibration than model-specific runs due to batch effects in how GPT-5.2 assigns cognitive functions.

These results can be used directly without regenerating (which requires OpenAI API access and costs).

## How It Works

1. **Benchmark Selection**: Loads benchmarks from `benchmark_info.csv` based on `--model` flag
2. **Cognitive Functions**: Loads function names, descriptions, and tiers from `cognitive_functions.csv`
3. **PDF Upload**: Uploads PDFs to OpenAI (cached after first upload)
4. **GPT-5.2 Call**: Sends prompt with benchmarks, cognitive function descriptions, and PDFs; uses `reasoning={effort: "high"}`
5. **Structured Output**: Uses JSON schema to enforce valid responses
6. **Validation**: Ensures benchmark order and cognitive function validity
7. **Transformation**: Groups cognitive functions by AI tier (L1, L2, L3)
8. **AI Tier Assignment**: Determines max AI tier for each benchmark
9. **Statistics**: Aggregates results across multiple runs

## Cognitive Functions

The script uses 34 cognitive functions organized by AI tier, loaded from `cognitive_functions.csv` (adapted from [Liu et al., 2025](https://arxiv.org/abs/2504.01990)). Each function includes a neuroscience-based description that helps GPT-5.2 make accurate assignments:

**L1 (Basic Processing)**: Visual Perception, Language Comprehension, Language Production, Face Recognition, Auditory Processing, Reflexive Responses

**L2 (Executive Functions)**: Planning, Logical Reasoning, Decision-making, Working Memory, Reward Mechanisms, Multisensory Integration, Spatial Representation & Mapping, Attention, Sensorimotor Coordination, Scene Understanding & Visual Reasoning, Visual Attention & Eye Movements, Episodic Memory, Semantic Understanding & Context Recognition, Adaptive Error Correction, Motor Skill Learning, Motor Coordination

**L3 (Higher-Order)**: Cognitive Flexibility, Inhibitory Control, Social Reasoning & Theory of Mind, Empathy, Emotional Processing, Self-reflection, Tactile Perception, Lifelong Learning, Cognitive Timing & Predictive Modeling, Autonomic Regulation, Arousal & Attention States, Motivational Drives

## Technical Details

- **Async Execution**: Multiple runs execute in parallel for efficiency
- **Retry Logic**: Up to 3 retries per run if validation fails
- **File Caching**: PDF file IDs are cached to avoid re-uploading
- **JSON Schema**: Enforces exact benchmark names and cognitive function values

## API Costs

Each run uploads 4 PDFs (~20MB total) and uses GPT-5.2 with high reasoning effort. Costs can be significant for multiple runs. Start with `--runs 1` to test.

## License

MIT
