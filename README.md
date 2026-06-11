# Using AI to Detect Motivation: Inferring Basic Psychological Need Satisfaction with a Large Language Model from Employee Appreciation Texts with Humans and AI Surrogates

**Caroline Müller & Sebastian Oscarson**  
Stockholm University — PSMT42 Master's Thesis  
*As of 18 May 2026*

---

## Study Overview

This repository contains the data pipeline and analysis code for a thesis investigating whether large language models (LLMs) can infer **basic psychological need satisfaction** (autonomy, competence, relatedness — Self-Determination Theory) from workplace appreciation texts.

The study uses two parallel samples:

- **AI surrogates** *(Caroline Müller)*: Synthetic personas (GPT-5 mini) write appreciation messages to coworkers. A separate LLM scorer (Claude Haiku 4.5) rates the texts for need satisfaction. Surrogate scores are compared against questionnaire-derived need satisfaction scores from the same personas.
- **Human participants** *(Sebastian Oscarson)*: Human raters provide a parallel reference sample for comparison.

**AI surrogate sample:** N = 400 personas × 3 text lengths (50/150/300 words) × 3 text counts (1/3/5 texts) × 3 scoring repeats = 10,800 scoring observations.

---

## Repository Structure

```
10_data/          Canonical pipeline outputs (JSONL + CSV + RDS)  [AI surrogates]
20_scripts/       Python pipeline scripts                         [AI surrogates]
30_prompts/       LLM prompt templates and Pydantic schemas       [AI surrogates]
40_analyses/      R/Quarto analysis files — see breakdown below   [AI surrogates]
50_humans/        Human rater data and analysis scripts           [Human participants]
config.yaml       Stage configuration (models, prompts, params)   [AI surrogates]
pricing.yaml      Token pricing table (USD/1M tokens)             [AI surrogates]
requirements.txt  Python dependencies
```

### 40_analyses/ — AI Surrogate Analyses

**Current files:**

| File | Description |
|------|-------------|
| `00_common.R` | Shared setup: libraries, constants, APA styling functions |
| `10_RQ2_data.qmd` | RQ2 data preparation: loading, cleaning, exclusions |
| `20_RQ2_analyses.qmd` | RQ2: convergent validity of LLM surrogate scores vs questionnaire |
| `30_RQ3_data.qmd` | RQ3 data preparation |
| `40_RQ3_analyses.qmd` | RQ3: sensitivity to text length and count conditions |
| `50_additional_analyses.qmd` | Additional and robustness analyses |

**Archived files** (`version1/` subfolder — superseded, kept for reference):

| File | Description |
|------|-------------|
| `version1/10_Analyses_RQ2.qmd` | Earlier monolithic RQ2 analysis file |
| `version1/15_Analyses_H1cd.qmd` | Earlier H1c/d ICC and within-person reliability file |
| `version1/20_Analyses_RQ3.qmd` | Earlier RQ3 analysis file |

### 50_humans/ — Human Participant Analyses

Contains all materials related to the human rater sample collected and analysed by Sebastian Oscarson:

| File | Description |
|------|-------------|
| `Data_Complete_Run2.csv` | Full human rater dataset (Run 2) |
| `Data_Complete_Run2_flagged.csv` | Same dataset with quality flags applied |
| `Data_Complete_Run2_wc.csv` | Dataset with word count covariates |
| `_Data Human Sample RQ1` | Raw data file for the RQ1 human sample |
| `Descriptives_and_Reliability_v2.R` | Descriptive statistics and reliability analyses |
| `Exclusions_questionnaire.R` | Exclusion criteria and filtering for questionnaire responses |
| `H1a_b_c_+ExplAim_Hypothesis_Tests_v2.R` | Hypothesis tests for H1a, H1b, H1c and exploratory aims |
| `H1b_partial_correlations_Run2.R` | Partial correlation analyses for H1b |
| `H1b_wordcount_covariate_Run2.R` | H1b re-analyses with word count as covariate |
| `Power Simulation (two-tailed).r` | Power simulation script |

---

## Pipeline Stages

> All pipeline stages concern the **AI surrogate** sample.

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

### AI Surrogate Analyses

The final merged analysis file is at:
```
10_data/70_merge_outputs_20260412-1426.csv
```

Open the Quarto files in RStudio (render in order):

```
40_analyses/10_RQ2_data.qmd            ← run first (data prep for RQ2)
40_analyses/20_RQ2_analyses.qmd        ← RQ2: LLM surrogate convergent validity
40_analyses/30_RQ3_data.qmd            ← run first (data prep for RQ3)
40_analyses/40_RQ3_analyses.qmd        ← RQ3: sensitivity to text length/count
40_analyses/50_additional_analyses.qmd ← additional and robustness analyses
```

Pre-computed bootstrap objects (cached for speed) are in `10_data/80_analyses_exports/`.

### Human Participant Analyses

See `50_humans/` for R scripts. These are independent of the AI surrogate pipeline.

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

All AI surrogate pipeline inputs and outputs are included in `10_data/`. Raw source data (SCOPE personas, HuggingFace dataset) is not included — see `20_scripts/00_sample_personas.py` and the [SCOPE dataset](https://huggingface.co/datasets) for reproduction.

Human rater data and analysis scripts are in `50_humans/`.

---

## Citation

If you use this code or data, please cite:

> Müller, C., & Oscarson, S. (2026). *Using AI to detect motivation: Inferring basic psychological need satisfaction with a large language model from employee appreciation texts with humans and AI surrogates.* Master's thesis, Stockholm University.