# Can LLMs Infer Basic Psychological Need Satisfaction?

**Caroline Müller & Sebastian Oscarson**  
Stockholm University — PSMT42 Master's Thesis  
*As of 18 May 2026*

---

## Study Overview

This repository contains the data pipeline and analysis code for a thesis investigating whether large language models (LLMs) can infer **basic psychological need satisfaction** (autonomy, competence, relatedness — Self-Determination Theory) from workplace appreciation texts.

The study uses AI-generated surrogate data: synthetic personas write appreciation messages to coworkers, and a separate LLM scorer rates the texts for need satisfaction. Surrogate scores are compared against questionnaire-derived need satisfaction scores for the same personas.

**Final sample:** N = 400 personas × 3 text lengths (50/150/300 words) × 3 text counts (1/3/5 texts) × 3 scoring repeats = 10,800 scoring observations.

---

## Repository Structure

```
10_data/          Canonical pipeline outputs (JSONL + CSV + RDS)
20_scripts/       Python pipeline scripts
30_prompts/       LLM prompt templates and Pydantic output schemas
40_analyses/      R/Quarto analysis files (RQ2, RQ3, H1c/d)
config.yaml       Stage configuration (models, prompts, parameters)
pricing.yaml      Token pricing table (USD/1M tokens)
requirements.txt  Python dependencies
```

---

## Pipeline Stages

| Stage | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 00 | `00_sample_personas.py` | SCOPE dataset (HuggingFace) | `00_personas.jsonl` | Sample synthetic personas meeting inclusion criteria |
| 10 | `10_prep_personas.py` | `00_personas.jsonl` | `00_personas_prod_800.jsonl` | Split into production set |
| 30 | `batch_stage.py --stage 30_coworkerfilter` | `00_personas_prod_800.jsonl` | `30_coworkerfilter_*.jsonl` | Filter personas to those with coworkers (OpenAI Batch API) |
| 35 | `35_extract_personas.py` | Stage 30 output | `35_personas_prod_*.jsonl` | Extract included personas |
| 40 | `batch_stage.py --stage 40_textgeneration` | Stage 35 output | `40_textgeneration_*.jsonl` | Generate appreciation texts (3 lengths × 5 coworkers per persona) |
| 45 | `45_filter_complete_texts.py` | Stage 40 output | `45_filter_complete_texts_*.jsonl` | Filter to personas with all 15 texts complete |
| 50 | `batch_stage.py --stage 50_complete_bpns` | Stage 35 output | `50_complete_bpns_*.jsonl` | Complete BPNS questionnaire items for each persona |
| 60 | `claude_batch_stage.py --stage 60_scoring_claude` | Stage 45 output | `60_scoring_claude_*.jsonl` | LLM-score texts for need satisfaction (Anthropic Batch API) |
| 70 | `70_merge_outputs.py` | Stages 50 + 60 | `70_merge_outputs_*.csv` | Merge BPNS scores + LLM scores into wide-format analysis file |

**Model assignments (cross-family separation):**
- Text generation (stage 40) + BPNS completion (stage 50): GPT-5 mini (OpenAI)
- LLM scoring (stage 60): Claude Haiku 4.5 (Anthropic)

---

## Reproducing the Analyses

The final merged analysis file is at:
```
10_data/70_merge_outputs_20260412-1426.csv
```

Open the Quarto files in RStudio:
- `40_analyses/10_Analyses_RQ2.qmd` — RQ2: convergent validity of LLM surrogate scores
- `40_analyses/20_Analyses_RQ3.qmd` — RQ3: sensitivity to text length and count conditions
- `40_analyses/15_Analyses_H1cd.qmd` — H1c/d: ICC and within-person reliability

The RQ3 and H1c/d analyses also read pre-computed objects from `10_data/80_analyses_exports/`.

---

## Reproducing the Pipeline

### Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Running stages

```bash
# Dry run (no API calls, cost estimate only)
python 20_scripts/batch_stage.py --stage 30_coworkerfilter -n 5 --dry-run

# Full pipeline orchestrator
python 20_scripts/run_pipeline.py --input 10_data/00_personas_prod_800.jsonl -n 800
```

See `config.yaml` for stage definitions, model versions, and prompt references.

---

## Data

All pipeline inputs and outputs are included in `10_data/`. Raw source data (SCOPE personas, HuggingFace dataset) is not included — see `20_scripts/00_sample_personas.py` and the [SCOPE dataset](https://huggingface.co/datasets) for reproduction.

Human rater data is not included in this repository (privacy).

---

## Citation

If you use this code or data, please cite:

> Müller, C., & Oscarson, S. (2026). *Can LLMs infer basic psychological need satisfaction? A surrogate data approach using workplace appreciation texts.* Master's thesis, Stockholm University.
