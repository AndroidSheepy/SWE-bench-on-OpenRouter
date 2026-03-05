# SWE-bench Evaluation via OpenRouter API

This guide documents the complete workflow for evaluating LLMs on SWE-bench using third-party model providers via the [OpenRouter](https://openrouter.ai) API. It covers environment setup, inference, Docker-based evaluation, and result interpretation.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Code Modifications for OpenRouter Support](#code-modifications-for-openrouter-support)
5. [Phase 1: Inference — Generating Patches](#phase-1-inference--generating-patches)
6. [Phase 2: Evaluation — Running Docker Tests](#phase-2-evaluation--running-docker-tests)
7. [Interpreting Results](#interpreting-results)
8. [Quick Reference](#quick-reference)

---

## Overview

SWE-bench evaluates whether a model can solve real GitHub issues by generating a code patch. The pipeline has two independent phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 — Inference                                            │
│  SWE-bench_oracle dataset  ──►  run_api.py  ──►  predictions.jsonl │
│  (issue + oracle context)        (LLM via API)   (model_patch)  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2 — Evaluation                                           │
│  predictions.jsonl  ──►  run_evaluation  ──►  report.json       │
│  (model_patch)           (Docker + pytest)   (resolved rate)    │
└─────────────────────────────────────────────────────────────────┘
```

**Key datasets:**
| Dataset | Used for | Contains |
|---|---|---|
| `princeton-nlp/SWE-bench_oracle` | Inference | Pre-built prompts with oracle file context (`text` field) |
| `princeton-nlp/SWE-bench` | Evaluation only | Test patches, environment specs, gold fixes |

> **Why two datasets?** `SWE-bench_oracle` already has the full prompt (issue + relevant source files) pre-assembled, which saves you from building RAG pipelines. It uses the *gold* patch files as context — this is an upper-bound setting. The evaluation harness always uses the original `princeton-nlp/SWE-bench` to run the actual tests.

---

## Prerequisites

- **macOS / Linux** (this guide was tested on macOS Apple Silicon)
- **Docker Desktop** installed and running
  - Minimum **6 GB memory** allocated to Docker (8 GB recommended)
  - Ensure you are **logged in** to Docker Hub (`docker login`) to avoid anonymous pull rate limits
- **Conda** (Miniconda or Anaconda)
- An **OpenRouter API key** — get one at [openrouter.ai/keys](https://openrouter.ai/keys)

---

## Environment Setup

### 1. Create a Conda environment

```bash
conda create -n swebench python=3.11 -y
conda activate swebench
```

### 2. Install SWE-bench with dataset dependencies

```bash
cd /path/to/SWE-bench
pip install -e ".[datasets]"
```

This installs all required packages including:
- `swebench` (4.1.0+)
- `openai` — used as the OpenRouter client (OpenRouter is OpenAI-compatible)
- `anthropic` — for Claude models
- `tiktoken` — for token counting
- `datasets` — for loading HuggingFace datasets

### 3. Set your OpenRouter API key

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxx"
```

Add this to your `~/.zshrc` or `~/.bashrc` to make it persistent.

---

## Code Modifications for OpenRouter Support

The original `run_api.py` only supports OpenAI and Anthropic models. The following changes were made to add OpenRouter support. **These modifications are already applied** to `swebench/inference/run_api.py` in this repository.

### Summary of changes

#### 1. Add model metadata (`MODEL_LIMITS`, `MODEL_COST_PER_INPUT`, `MODEL_COST_PER_OUTPUT`)

```python
MODEL_LIMITS = {
    ...,
    "z-ai/glm-5": 202_752,
}
MODEL_COST_PER_INPUT = {
    ...,
    "z-ai/glm-5": 0.00000080,   # $0.80 / 1M input tokens
}
MODEL_COST_PER_OUTPUT = {
    ...,
    "z-ai/glm-5": 0.00000256,   # $2.56 / 1M output tokens
}
```

To add a **new model**, append its context window size and cost to all three dicts. Check the model's pricing page on OpenRouter for accurate values. **You can easily add configs of other models in run_api.py.**

#### 2. OpenRouter base URL and token estimator

```python
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_ENCODING = tiktoken.get_encoding("cl100k_base")

def rough_tokenize(string: str) -> int:
    """Estimate token count using cl100k_base (GPT-4 encoding)."""
    return len(_OPENROUTER_ENCODING.encode(string))
```

`cl100k_base` is a reasonable approximation for most modern LLMs when the model's own tokenizer is unavailable.

#### 3. `call_openrouter()` — single API call

```python
@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_openrouter(client, model_name_or_path, inputs, temperature, top_p, **model_args):
    system_message, user_message = inputs.split("\n", 1)
    response = client.chat.completions.create(
        model=model_name_or_path,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
        **model_args,
    )
    cost = calc_cost(model_name_or_path,
                     response.usage.prompt_tokens,
                     response.usage.completion_tokens)
    return response, cost
```

#### 4. `openrouter_inference()` — full dataset loop

Reads `OPENROUTER_API_KEY` from environment, filters instances that exceed the context window, iterates through the dataset, and writes results to a JSONL file. Supports **resuming** — already-processed instance IDs are skipped.

#### 5. CLI arguments added

| Argument | Type | Description |
|---|---|---|
| `--openrouter` | flag | Enable OpenRouter backend |
| `--num_instances` | int | Process only the first N instances (sorted by prompt length, shortest first) |

The `--model_name_or_path` argument no longer has a restricted `choices` list, allowing any OpenRouter model identifier.

---

## Phase 1: Inference — Generating Patches

### Command

```bash
cd /path/to/SWE-bench

python -m swebench.inference.run_api \
    --dataset_name princeton-nlp/SWE-bench_oracle \
    --split test \
    --model_name_or_path z-ai/glm-5 \
    --output_dir ./predictions \
    --openrouter \
    --num_instances 5 \
    --max_cost 10.0
```

### Key arguments

| Argument | Example | Description |
|---|---|---|
| `--dataset_name` | `princeton-nlp/SWE-bench_oracle` | Use the oracle dataset (has `text` field) |
| `--split` | `test` | Dataset split (`test` has 2294 instances) |
| `--model_name_or_path` | `z-ai/glm-5` | OpenRouter model identifier |
| `--output_dir` | `./predictions` | Directory for output JSONL |
| `--openrouter` | *(flag)* | Use OpenRouter backend |
| `--num_instances` | `5` | Limit to first N instances (for testing) |
| `--max_cost` | `10.0` | Stop after spending $10 USD |

### Output format

The script writes one JSON object per line to:
```
predictions/<model_slug>__<dataset_slug>__<split>.jsonl
```

Each line contains:
```json
{
  "instance_id": "django__django-15127",
  "model_name_or_path": "z-ai/glm-5",
  "text": "<full prompt>",
  "full_output": "<model raw response>",
  "model_patch": "diff --git a/...\n..."
}
```

The `model_patch` field is extracted from the raw output by parsing the first unified diff block found in the response.

### Resuming interrupted runs

If the script is stopped, simply re-run the same command. It detects existing `instance_id` values in the output file and skips them automatically.

---

## Phase 2: Evaluation — Running Docker Tests

### Docker requirements

SWE-bench builds a separate Docker image per repository environment and runs the test suite inside it. Before running:

1. **Ensure Docker Desktop is running** with at least 6 GB memory:
   - Docker Desktop → Settings (⚙️) → Resources → Memory → set to 6–8 GB → Apply & Restart

2. **Log in to Docker Hub** to avoid anonymous rate limits:
   ```bash
   docker login
   ```

3. **Check your architecture.** On Apple Silicon (arm64), you must pass `--namespace ''` so Docker builds images locally rather than pulling pre-built amd64 images:
   ```bash
   uname -m   # should print "arm64"
   ```

### Command

```bash
cd /path/to/SWE-bench

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench \
    --predictions_path ./predictions/glm-5__SWE-bench_oracle__test.jsonl \
    --max_workers 4 \
    --run_id my_experiment
```

### Key arguments

| Argument | Example | Description |
|---|---|---|
| `--dataset_name` | `princeton-nlp/SWE-bench` | **Always use the original dataset** (not oracle) |
| `--predictions_path` | `./predictions/...jsonl` | Output file from Phase 1 |
| `--max_workers` | `4` | Parallel Docker containers (lower if memory-constrained) |
| `--run_id` | `my_experiment` | Identifier used in log/result filenames |
| `--namespace ''` | *(empty string)* | Required on arm64 — disables pre-built image pulls |

> **Important:** `--dataset_name` must be `princeton-nlp/SWE-bench` (or `princeton-nlp/SWE-bench_Lite`), **not** the oracle dataset. The oracle dataset is only used for inference prompts; the evaluation harness needs the original dataset to access test patches and environment specs.

### What happens during evaluation

1. **Base image build** (`sweb.base.*`) — pulls Python/Ubuntu base, installs conda (~5 min, cached after first run)
2. **Environment image build** (`sweb.env.*`) — creates a conda env with the specific Python version and repo dependencies per task group (~10–20 min per unique env, cached)
3. **Instance evaluation** — applies the model patch, runs `pytest` on the target tests, records pass/fail

Docker images are cached — subsequent runs skip the build phase entirely.

### Outputs

| Path | Description |
|---|---|
| `<model>.<run_id>.json` | Summary report (resolve rate, instance list) |
| `logs/build_images/` | Docker build logs |
| `logs/run_evaluation/<run_id>/<model>/` | Per-instance logs |

---

## Interpreting Results

### Summary report (`<model>.<run_id>.json`)

```json
{
  "total_instances": 2294,
  "submitted_instances": 5,
  "completed_instances": 5,
  "resolved_instances": 1,
  "unresolved_instances": 4,
  "resolved_ids": ["django__django-15127"],
  "unresolved_ids": ["django__django-12933", "..."]
}
```

**Resolve rate** = `resolved_instances / submitted_instances`

### Per-instance report (`logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json`)

```json
{
  "django__django-15127": {
    "patch_is_None": false,
    "patch_exists": true,
    "patch_successfully_applied": true,
    "resolved": false,
    "tests_status": {
      "FAIL_TO_PASS": {
        "success": [],
        "failure": ["test_override_settings_level_tags (...)"]
      },
      "PASS_TO_PASS": {
        "success": ["test_eq (...)"],
        "failure": []
      },
      "FAIL_TO_FAIL": { "success": [], "failure": [] },
      "PASS_TO_FAIL": { "success": [], "failure": [] }
    }
  }
}
```

### Test status categories explained

| Category | Before patch | After patch | Meaning |
|---|:---:|:---:|---|
| **FAIL_TO_PASS** | ✗ failing | ✓ passing | The bug-specific tests — **must all pass** for `resolved: true` |
| **PASS_TO_PASS** | ✓ passing | ✓ passing | Regression guard — existing tests must not break |
| **FAIL_TO_FAIL** | ✗ failing | ✗ failing | Other known issues — allowed to remain failing |
| **PASS_TO_FAIL** | ✓ passing | ✗ failing | **Regression introduced** — patch broke something that was working |

**`resolved: true` requires:**
1. All `FAIL_TO_PASS.failure` lists are empty (bug-fixing tests all pass)
2. All `PASS_TO_PASS.failure` lists are empty (no regressions introduced)

### Detailed test output (`test_output.txt`)

For debugging why a patch failed, read the raw pytest output:

```bash
# See the failure traceback
grep -A 20 "FAIL\|AssertionError\|Error" \
    logs/run_evaluation/<run_id>/<model>/<instance_id>/test_output.txt

# See what patch was actually applied
cat logs/run_evaluation/<run_id>/<model>/<instance_id>/patch.diff
```

---

## Quick Reference

```bash
# Step 1: Activate environment
conda activate swebench
export OPENROUTER_API_KEY="sk-or-v1-..."

# Step 2: Run inference (test with 5 instances, $5 budget)
python -m swebench.inference.run_api \
    --dataset_name princeton-nlp/SWE-bench_oracle \
    --split test \
    --model_name_or_path z-ai/glm-5 \
    --output_dir ./predictions \
    --openrouter \
    --num_instances 5 \
    --max_cost 5.0

# Step 3: Log in to Docker Hub (first time only)
docker login

# Step 4: Run evaluation (arm64 Mac — requires --namespace '')
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench \
    --predictions_path ./predictions/z-ai__glm-5__SWE-bench_oracle__test.jsonl \
    --max_workers 4 \
    --run_id my_run \
    --namespace ''

# Step 5: View results
cat z-ai__glm-5.my_run.json | python3 -c "
import json, sys; d = json.load(sys.stdin)
print(f'Resolved: {d[\"resolved_instances\"]} / {d[\"submitted_instances\"]}')
print(f'Resolved IDs: {d[\"resolved_ids\"]}')
"
```
