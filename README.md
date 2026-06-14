# AfriMedQA Evaluation Framework

This repository evaluates vision-language models (VLMs) and machine translation systems on the **AfriMedQA** dataset — a multilingual African medical QA benchmark covering 14 languages.

There are two independent evaluation pipelines:

| Pipeline | Entry point | What it evaluates |
|---|---|---|
| **VLM Eval** | `run.py` | Model accuracy on MCQ / SAQ questions (with images) |
| **MT Eval** | `mt_eval/run.py` | Translation quality of English SAQ pairs into African languages |

---

## Setup

```bash
conda create -n afrimedqa_vlmeval python=3.10 -y
conda activate afrimedqa_vlmeval
pip install -r requirements.txt
```

For HTML → TSV conversion only:
```bash
pip install beautifulsoup4 lxml
```

---

## Dataset Preparation

### HTML → TSV (if starting from raw HTML)

```bash
python html_to_tsv.py \
  --html All_Pics_Questions/All_Pics_Questions.html \
  --out AfrimedQA.tsv
```

The script extracts questions, answer options, correct answers, and embeds images as base64 in the TSV.

### Expected TSV layout

Each TSV must be placed in the directory pointed to by the `LMUData` environment variable. The filename is the dataset name used in configs (e.g. `AMHARIC_FULL.tsv`).

The 14 language splits used across all full-split configs are:

`AMHARIC_FULL` · `ARABIC_FULL` · `ENGLISH_FULL` · `FRENCH_FULL` · `HAUSA_FULL` · `IGBO_FULL` · `ISIZULU_FULL` · `PORTUGUESE_FULL` · `SESOTHO_FULL` · `SWAHILI_FULL` · `TWI_FULL` · `WOLLOF_FULL` · `XHOSA_FULL` · `YORUBA_FULL`

---

## Pipeline 1 — VLM Evaluation (`run.py`)

### Dataset classes

| Class | Base | Filters | Prompt style | Scoring |
|---|---|---|---|---|
| `AfrimedQA` | `ImageMCQDataset` | MCQ only | CoT + `Answer: <letter>` | Regex extraction → exact match / LLM judge |
| `AfrimedShortQA` | `ImageShortQADataset` | SAQ only | CoT + `FINAL ANSWER:` | DeepEval G-Eval (5 clinical axes) |
| `AfrimedTextQA` | `TextMCQDataset` | MCQ only | CoT + `FINAL ANSWER: <letter>` | Regex extraction → exact match |

### Config structure

All runs are driven by a JSON config with `model` and `data` blocks. Place configs in `configs/`.

**Multimodal MCQ — full language splits** (`configs/medgemma_27b_mcq_full_splits.json`):
```json
{
    "model": {
        "MedGemma-27B": {
            "class": "Gemma3",
            "model_path": "google/medgemma-27b-it",
            "use_vllm": false,
            "device_map": "auto"
        }
    },
    "data": {
        "AMHARIC_FULL":    { "class": "AfrimedQA", "dataset": "AMHARIC_FULL" },
        "ENGLISH_FULL":    { "class": "AfrimedQA", "dataset": "ENGLISH_FULL" },
        "SWAHILI_FULL":    { "class": "AfrimedQA", "dataset": "SWAHILI_FULL" }
    }
}
```

**Text-only MCQ baseline** (`configs/gemma_text_baseline.json`):
```json
{
    "model": {
        "gemma-3-4b-text": {
            "class": "Gemma3",
            "model_path": "google/gemma-3-4b-it"
        }
    },
    "data": {
        "ENGLISH_FULL": { "class": "AfrimedTextQA", "dataset": "ENGLISH_FULL" }
    }
}
```

**SAQ with LLM judge** (`configs/afrimedsaq.json`):
```json
{
    "model": {
        "medgemma-27b-it": {
            "class": "Gemma3",
            "model_path": "google/medgemma-27b-it",
            "use_vllm": true,
            "tensor_parallel_size": 2
        }
    },
    "data": {
        "ENGLISH_TEST__Sheet1": {
            "class": "AfrimedShortQA",
            "dataset": "ENGLISH_TEST__Sheet1"
        }
    }
}
```

### Supported models (pre-built configs)

| Config | Model | Notes |
|---|---|---|
| `medgemma_27b_mcq_full_splits.json` | `google/medgemma-27b-it` | HuggingFace, `device_map=auto` |
| `gemma3_4b_mcq_full_splits.json` | `google/gemma-3-4b-it` | |
| `gemma3_12b_mcq_full_splits.json` | `google/gemma-3-12b-it` | |
| `gemma3_27b_mcq_full_splits.json` | `google/gemma-3-27b-it` | |
| `gemma4_e4b_mcq_full_splits.json` | `google/gemma-4-E4B-it` | |
| `deepseek_vl2_mcq_full_splits.json` | `deepseek-ai/deepseek-vl2` | |
| `deepseek_vl2_small_mcq_full_splits.json` | `deepseek-ai/deepseek-vl2-small` | |
| `qwen3_vl_32b_mcq_full_splits.json` | `Qwen/Qwen3-VL-32B-Instruct` | |
| `gemini_flash_mcq_full_splits.json` | `gemini-3.5-flash` | API, requires `GEMINI_API_KEY` |
| `gemini_flash_lite_mcq_full_splits.json` | `gemini-3.5-flash-lite` | API |
| `gemini_pro_mcq_full_splits.json` | `gemini-3.5-pro` | API |

### Running evaluations

Set `LMUData` to the directory containing your `.tsv` dataset files.

**Multimodal MCQ (all 14 language splits):**
```bash
LMUData=full_splits python run.py --config configs/medgemma_27b_mcq_full_splits.json \
    --work-dir outputs/vlmeval
```

**Text-only MCQ baseline:**
```bash
LMUData=full_splits python run.py --config configs/gemma_text_baseline.json \
    --work-dir outputs/vlmeval
```

**SAQ with LLM judge:**
```bash
LMUData=test_files python run.py --config configs/afrimedsaq.json \
    --judge gpt-4o --work-dir outputs/vlmeval
```

**Inference only (skip scoring):**
```bash
LMUData=full_splits python run.py --config configs/medgemma_27b_mcq_full_splits.json \
    --mode infer --work-dir outputs/vlmeval
```

**Resume a partial run:**
```bash
LMUData=full_splits python run.py --config configs/medgemma_27b_mcq_full_splits.json \
    --reuse --work-dir outputs/vlmeval
```

### VLM Eval outputs

Results are written to `outputs/vlmeval/{model_name}/{eval_id}/{dataset_name}/`:

| File | Content |
|---|---|
| `{model}_{dataset}.xlsx` | Raw per-question predictions |
| `{model}_{dataset}_acc_all.csv` | Accuracy summary |
| `{model}_{dataset}_full_data.csv` | Per-question predictions, extracted letters, hit/miss |
| `{model}_{dataset}_eval_report.md` | SAQ: full DeepEval G-Eval report |
| `{model}_{dataset}_metrics.csv` | SAQ: averaged clinical axis scores |

### SAQ evaluation metrics (DeepEval G-Eval)

SAQ predictions are judged by an LLM on 5 axes adapted from the Med-PaLM 2 clinical rubric, each scored 1/3/5:

| Axis | What is checked |
|---|---|
| `Accuracy_and_Appropriateness` | Medical correctness vs. reference answer |
| `Completeness` | Coverage of clinically important content |
| `Harm_Severity` | Danger of the advice if a patient acts on it |
| `Harm_Probability` | Likelihood a patient would experience harm |
| `Bias_Detection` | Demographic stereotyping or discriminatory language |

### Slurm scripts (VLM Eval)

| Script | GPUs | Use case |
|---|---|---|
| `slurm/run_vlmeval_gpu.sh` | 1× A40 | Small models |
| `slurm/run_vlmeval_gpu2.sh` | 2× A40 | 27B models (e.g. Gemma3-27B, MedGemma-27B) |
| `slurm/run_vlmeval_gpu2_deepseek_vl2.sh` | 2× A40 | DeepSeek-VL2 |
| `slurm/run_vlmeval_gpu2_qwen3_vl.sh` | 2× A40 | Qwen3-VL-32B |
| `slurm/run_vlmeval_gpu_gemma4.sh` | varies | Gemma4 |
| `slurm/run_vlmeval_api.sh` | CPU | API models (Gemini, GPT) |

```bash
# Example: submit a 27B GPU job
sbatch slurm/run_vlmeval_gpu2.sh configs/medgemma_27b_mcq_full_splits.json
```

All scripts set `LMUData=full_splits` and write outputs to `outputs/vlmeval`.

---

## Pipeline 2 — Machine Translation Evaluation (`mt_eval/`)

Translates AfriMedQA SAQ (English question+answer pairs) into African languages, then scores with **ChrF** and **SSA-COMET**.

### Running an MT evaluation

```bash
python -m mt_eval.run mt_eval/configs/en_twi_vllm.json
```

### Config structure

```json
{
    "data": {
        "source_file": "full_splits/ENGLISH_FULL.tsv",
        "target_file": "full_splits/AMHARIC_FULL.tsv",
        "source_lang": "English",
        "target_lang": "Amharic"
    },
    "model": { ... },
    "output_dir": "outputs/mt_eval"
}
```

`source_file` and `target_file` must be TSV files with aligned question/answer rows.

### Model backends

| `"type"` | Backend | Example |
|---|---|---|
| `"vllm"` | vLLM server-side generation | `Qwen/Qwen2.5-7B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| `"gemini"` | Google GenAI SDK | `gemini-3.5-flash` |
| `"hf"` | HuggingFace chat pipeline | `google/gemma-3-27b-it` |
| `"seq2seq"` | HuggingFace seq2seq (NLLB) | `facebook/nllb-200-3.3B` |
| `"chat"` | OpenAI-compatible chat API | any OpenAI-format endpoint |

**vLLM config** (with repetition penalty for reasoning models):
```json
{
    "model": {
        "type": "vllm",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "max_tokens": 512,
        "temperature": 0.0,
        "repetition_penalty": 1.15,
        "vllm_kwargs": {
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
            "trust_remote_code": true,
            "enforce_eager": true,
            "max_model_len": 4096
        }
    }
}
```

**Gemini config** (reads key from env var `GEMINI_API_KEY`):
```json
{
    "model": {
        "type": "gemini",
        "model_name": "gemini-3.5-flash",
        "api_key": "GEMINI_API_KEY",
        "max_tokens": 512,
        "temperature": 0.0
    }
}
```

**HuggingFace chat config**:
```json
{
    "model": {
        "type": "hf",
        "model_name": "google/gemma-3-27b-it",
        "max_tokens": 512,
        "temperature": 0.0,
        "hf_kwargs": { "device_map": "auto", "torch_dtype": "bfloat16" }
    }
}
```

**NLLB seq2seq config**:
```json
{
    "model": {
        "type": "seq2seq",
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "eng_Latn",
        "tgt_lang": "twi_Latn",
        "max_tokens": 512
    }
}
```

### Full-splits configs

Pre-built configs for all 14 language pairs × multiple models live in `mt_eval/configs/full_splits/`. Naming pattern: `en_{language}_{model}.json`.

Models with full-splits configs: `gemma3_4b`, `gemma3_12b`, `gemma3_27b`, `gemma4_e4b`, `deepseek_r1_32b`, `deepseek_v4_flash`, `qwen3_6_27b`, `qwen3_vl_32b`, `medgemma_27b`, `gemini_flash`, `gemini_flash_lite`, `gemini` (pro).

### MT Eval outputs

Outputs are written to `outputs/mt_eval/{model_tag}/{source_lang}_{target_lang}/` (e.g. `outputs/mt_eval/google_gemma-3-27b-it/english_twi/`):

| File | Content |
|---|---|
| `{src}_{tgt}_{model}_results.csv` | Per-sample translations + sentence-level ChrF and SSA-COMET |
| `{src}_{tgt}_{model}_summary.csv` | Corpus-level summary: mean & corpus ChrF (Q/A/combined), SSA-COMET system scores |

### MT Eval metrics

- **ChrF**: character n-gram F-score (sentence-level and corpus-level)
- **SSA-COMET**: reference-based neural MT metric fine-tuned for Sub-Saharan African languages; requires GPU

### System prompt

The pipeline uses a structured medical translation prompt that:
- Instructs the model to read Q+A together before translating
- Enforces domain-appropriate African healthcare terminology
- Strips `<think>...</think>` blocks from reasoning models (e.g. Qwen3, DeepSeek-R1) before parsing output

### Slurm scripts (MT Eval)

| Script | GPUs | Use case |
|---|---|---|
| `slurm/run_mt_eval_gpu.sh` | 1× A40 | Single config, small model |
| `slurm/run_mt_eval_batch_gpu.sh` | 1× A40 | Directory or glob of configs, sequential |
| `slurm/run_mt_eval_batch_gpu2.sh` | 2× A40 | 27B models (Gemma3-27B, MedGemma-27B) |
| `slurm/run_mt_eval_batch_gpu2_deepseek_r1_32b.sh` | 2× A40 | DeepSeek-R1-32B |
| `slurm/run_mt_eval_batch_gpu2_qwen3_6_27b.sh` | 2× A40 | Qwen3-6/27B |
| `slurm/run_mt_eval_batch_gpu2_qwen3_vl.sh` | 2× A40 | Qwen3-VL |
| `slurm/run_mt_eval_batch_gpu4_deepseek_v4_flash.sh` | 4× A40 | DeepSeek-V4-Flash |
| `slurm/run_mt_eval_batch_gpu_gemma4.sh` | varies | Gemma4 |
| `slurm/run_mt_eval_batch_api.sh` | CPU | Gemini / OpenAI API models |
| `slurm/run_mt_eval_api.sh` | CPU | Single API config |
| `slurm/run_mt_eval_array_gpu.sh` | 1× A40 | SLURM array (one job per config) |

```bash
# Run all Gemma3-27B full-split MT configs sequentially on 2 GPUs
sbatch slurm/run_mt_eval_batch_gpu2.sh "mt_eval/configs/full_splits/en_*_gemma3_27b.json"

# Or pass a whole directory
sbatch slurm/run_mt_eval_batch_gpu2.sh mt_eval/configs/full_splits/
```

The batch scripts iterate configs sequentially in a single job, report per-config pass/fail, and print a summary at the end.

---

## Project structure

```
afrimedqa-vlmeval/
├── run.py                        # VLM eval entry point
├── html_to_tsv.py                # Dataset conversion utility
├── text_only.py                  # Text-only inference helper
├── requirements.txt
├── configs/                      # VLM eval JSON configs
│   ├── medgemma_27b_mcq_full_splits.json
│   ├── qwen3_vl_32b_mcq_full_splits.json
│   ├── gemini_flash_mcq_full_splits.json
│   └── ...
├── mt_eval/                      # MT eval pipeline
│   ├── run.py                    # MT eval entry point
│   ├── configs/                  # MT eval JSON configs
│   │   ├── en_twi_vllm.json
│   │   ├── en_twi_gemini.json
│   │   └── full_splits/          # Per-language × per-model configs
│   ├── datasets/                 # Parallel corpus loader
│   ├── metrics/                  # ChrF + SSA-COMET scorers
│   └── models/                   # vllm / gemini / hf / seq2seq / chat backends
├── vlmeval/
│   ├── dataset/
│   │   ├── afrimedqa.py          # AfrimedQA — multimodal MCQ
│   │   ├── afrimedqa_shortqa.py  # AfrimedShortQA — multimodal SAQ
│   │   └── afrimedqa_text.py     # AfrimedTextQA — text-only MCQ
│   ├── vlm/                      # VLM model classes (Gemma3, Qwen3VL, DeepSeekVL2, …)
│   ├── api/                      # API model classes (Gemini, GPT4V, …)
│   └── inference_mt.py           # Multi-turn inference dispatcher
└── slurm/                        # SLURM submission scripts
    ├── run_vlmeval_gpu2.sh
    ├── run_mt_eval_batch_gpu2.sh
    └── ...
```

---

## Environment variables

| Variable | Used by | Description |
|---|---|---|
| `LMUData` | VLM eval | Path to directory containing dataset `.tsv` files |
| `GEMINI_API_KEY` | VLM eval + MT eval | Google Gemini API key |
| `OPENAI_API_KEY` | VLM eval (judge) | OpenAI key for LLM judge / SAQ G-Eval |
| `HF_HOME` | Both | HuggingFace model cache directory |
| `VLLM_WORKER_MULTIPROC_METHOD` | Both | Set to `spawn` for vLLM multi-GPU |

Set API keys and paths in a `.env` file at the project root — the Slurm scripts source it automatically via `set -a; source .env; set +a`.
