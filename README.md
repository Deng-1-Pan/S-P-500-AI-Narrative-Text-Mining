# S&P 500 AI Narrative Text Mining

Analysis of AI narratives in S&P 500 earnings call transcripts using dictionary methods, topic modelling, and econometric mechanism analysis (2020–2025).

---

## 📋 Engineering & Reproducibility Statement

This codebase is written to professional software engineering standards:

- **Modular architecture**: all logic is organised into focused, single-responsibility modules under `src/`; `run_pipeline.py` acts purely as an orchestrator.
- **Single entry point**: the entire analysis, from raw data to all figures and tables, is driven by one command (`python run_pipeline.py`).
- **Deterministic seeds**: `seed=42` is propagated to every random process (sampling, cross-validation, topic modelling, Lasso).
- **Fail-fast validation**: the pipeline validates the input dataset schema and referential integrity before any computation begins.
- **Automatic logging**: every run writes a timestamped log file to `outputs/logs/` and a `outputs/pipeline_manifest.json` recording all parameters, input SHA-256 hashes, and the Git HEAD commit.
- **Large-file management**: all binary data files (`.csv`, `.parquet`, `.pkl`) are tracked via **Git LFS** (configured in `.gitattributes`), so cloning the repository gives you the full dataset without any manual download step.
- **Environment secrets**: the `.env` file (containing the HuggingFace token) is listed in `.gitignore` and is therefore **never committed to the repository**. A `.env.example` template is provided instead.

> **Note on output volume.** Running the full pipeline generates a large number of figures, feature tables, and reports under `outputs/`. This is by design: the project explores the dataset extensively across 16 pipeline stages. Because this volume of binary artefacts is not appropriate for a Git repository, `outputs/` is listed in `.gitignore`. All outputs are fully reproducible by running the pipeline as documented below.

---

## 📁 Project Structure

```
.
├── src/
│   ├── preprocessing/      # Stage 0–2: data download, transcript parsing, sentence splitting
│   ├── baselines/          # Stage 3:  dictionary keyword detection
│   ├── metrics/            # Stage 5–6: AI intensity & initiation score computation
│   ├── analysis/           # Stage 7–14: EDA, quadrants, regression,
│   │                       #             benchmark, Lasso, rankings, wordclouds,
│   │                       #             research report, metadata analysis
│   ├── research/           # Stage 15: WRDS × AI Narrative panel linkage
│   └── utils/              # Shared utilities
├── scripts/                # Standalone analytical tools (see § Scripts below)
├── data/
│   ├── final_dataset.parquet   # Full earnings-call dataset (tracked via Git LFS)
│   ├── final_dataset.csv       # CSV copy (tracked via Git LFS)
│   ├── wrds.csv                # WRDS Compustat fundamentals (tracked via Git LFS)
│   └── human_annotation/       # Double-annotated ground-truth CSVs
├── outputs/                # Generated at runtime – NOT committed to the repo
│   ├── features/           # Intermediate parquet feature tables
│   ├── figures/            # All plots and charts
│   ├── logs/               # Pipeline run logs
│   ├── report/             # Auto-generated research-grade Markdown report
│   └── pipeline_manifest.json  # Auto-generated run record
├── tests/                  # Unit tests
├── run_pipeline.py         # ← Main entry point
├── requirements.txt        # Python dependencies
├── .gitattributes          # Git LFS rules for large binary files
└── .env.example            # Template for the HuggingFace token
```

---

## 🚀 Reproduction Guide

### Step 1 — Clone the repository

```bash
git clone <repo-url>
cd <repo-directory>
```

Because large data files are managed by Git LFS, `git clone` will automatically download `data/final_dataset.parquet`, `data/final_dataset.csv`, and `data/wrds.csv`. Ensure that Git LFS is installed on your machine (`git lfs install`) before cloning.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure the HuggingFace token *(only needed for Stage 0)*

Stage 0 re-downloads the raw transcript dataset from HuggingFace. **This step is not required** if you use the pre-downloaded `data/final_dataset.parquet` that is already in the repository (recommended).

If you do want to re-run the download, create a `.env` file from the template:

```bash
cp .env.example .env
# Then edit .env and set HF_TOKEN=<your_token>
```

### Step 4 — Run the full pipeline

```bash
# Recommended: skip Stage 0 (use the pre-downloaded dataset already in data/)
python run_pipeline.py --start-stage 1

# Alternative: full run including data download (requires HF_TOKEN in .env)
python run_pipeline.py --run-download
```

All figures, tables, and the research report will be written to `outputs/`.

### Step 5 — Reproduce human-annotation validation figures

The manual validation analysis (confusion matrices, inter-annotator agreement) is a **separate script** that is not part of the main pipeline. Run it independently after the pipeline has completed at least through Stage 3:

```bash
python scripts/manual_validation.py
```

Output figures are saved to `outputs/figures/validation/`.

---

## ⚙️ Pipeline Arguments Reference

```
python run_pipeline.py [OPTIONS]

Core options:
  --start-stage INT         Start from a specific stage (0–15). Default: 0.
                            Stage 1 is the recommended starting point when
                            using the pre-downloaded dataset.
  --run-download            Include Stage 0 (fetch from HuggingFace + merge WRDS).
  --dev                     Development mode: processes a small sample (~100 docs
                            per stage) for rapid iteration.
  --dev-sample INT          Sample size for dev mode. Default: 100.
  --seed INT                Global random seed. Default: 42.

Skip flags (for partial reruns):
  --skip-lasso              Skip Stage 11 (Lasso / volcano plots).
  --skip-benchmark          Skip Stage 10 (outperformance classification).
  --skip-eda-foundation     Skip Stage 7 (funnel + zero-inflation visuals).
  --skip-research-report    Skip Stage 13 (dual-path regressions + report).
  --skip-metadata           Skip Stage 14 (AI narrative metadata analysis).
  --skip-stage15            Skip Stage 15 (WRDS × AI metadata linkage).
  --run-stage15-only        Run only Stage 15 (sets --start-stage 15).

Advanced options:
  --input PATH              Path to earnings dataset. Default: data/final_dataset.parquet.
  --wrds PATH               Path to WRDS data. Default: data/wrds.csv.
  --lasso-max-features INT  TF-IDF vocabulary size for Lasso. Default: 5000.
  --lasso-ngram-max INT     Max n-gram length for Lasso. Default: 2.
  --lasso-cv INT            Cross-validation folds for LassoCV. Default: 5.
  --benchmark-cv-folds INT  CV folds for benchmark comparison. Default: 5.
  --research-target STR     Economic target variable. Default: y_next_mktcap_growth.
```

---

## 🛠️ Scripts Reference

The `scripts/` directory currently contains the following standalone tools:

- `manual_validation.py`
- `inspect_extremes.py`
- `inspect_doc_extremes.py`
- `export_annotation_samples.py`

All scripts are intended to be run from the repository root. Scripts that read pipeline artefacts support the current stage-folder layout under `outputs/features/stageXX/` and will also fall back to legacy flat paths if present.

---

### `manual_validation.py` — Human Annotation Validation

Compares the pipeline's algorithmic labels against the four double-annotated ground-truth datasets in `data/human_annotation/`. Produces precision / recall / F1 metrics, inter-annotator Cohen's Kappa agreement tables, and confusion matrix visualisations for each sub-task.

**Prerequisite:** pipeline run through at least Stage 3 (so that keyword features exist), plus the completed annotation CSVs in `data/human_annotation/`.

```bash
python scripts/manual_validation.py
```

**Outputs** (written to `outputs/figures/validation/`):
| File | Description |
|---|---|
| `ai_keyword_confusion_matrix.png` | 2×2 CM: AI Keyword Detector vs adjudicated labels |
| `role_confusion_matrix.png` | 3×3 CM: Speaker Role Classifier vs adjudicated labels |
| `boundary_confusion_matrix.png` | 2×2 CM: QA Boundary Parser annotator vote vs adjudicated |
| `initiation_confusion_matrix.png` | 4×4 CM: Initiation Type classifier vs adjudicated labels |
| `role_performance_bars.png` | Per-role Accuracy & F1 bar chart |

A timestamped log is saved to `outputs/logs/inspect/`.

---

### `inspect_extremes.py` — Company & Document AI-Intensity Extremes

Prints ranked tables of the most AI-intensive companies and earnings call documents, broken down by speech vs. Q&A sections and by quadrant (Aligned / Self-Promoting / Passive / Avoider). Useful for sanity-checking outliers and generating talking points about specific firms.

**Prerequisite:** pipeline run through at least Stage 5.

```bash
python scripts/inspect_extremes.py
```

**Output:** printed to terminal and saved to `outputs/logs/inspect/extremes_<timestamp>.txt`.

---

### `inspect_doc_extremes.py` — Sentence-Level AI Context for Extreme Documents

For each identified extreme document (highest combined AI intensity, most self-promoting, most passive, etc.), prints the top AI-keyword sentences together with ±2 sentences of surrounding context. Useful for qualitative review of what specific companies actually said.

**Prerequisite:** pipeline run through at least Stage 5.

```bash
python scripts/inspect_doc_extremes.py
```

**Output:** printed to terminal and saved to `outputs/logs/inspect/doc_extremes_<timestamp>.txt`.

---

### `export_annotation_samples.py` — Generate Human-Annotation Templates

Samples random sentences, Q&A turns, document boundaries, and initiation exchanges from the pipeline outputs, then writes blank CSV templates ready for human annotators to fill in. This is the script used to produce the files in `data/human_annotation/` (which have already been completed and committed).

**Prerequisite:** pipeline run through at least Stage 5.

```bash
python scripts/export_annotation_samples.py \
    --features-dir outputs/features \
    --output-dir outputs/annotation_samples \
    --ai-pos-n 60 --ai-neg-n 60 \
    --role-n 80 --boundary-n 30 --initiation-n 50
```

| Argument | Default | Description |
|---|---|---|
| `--features-dir` | `outputs/features` | Directory containing pipeline feature parquets |
| `--output-dir` | `outputs/annotation_samples` | Where to write the CSV templates |
| `--seed` | `42` | Random seed for deterministic sampling |
| `--ai-pos-n` | `60` | Number of AI-positive sentence samples |
| `--ai-neg-n` | `60` | Number of AI-negative sentence samples |
| `--role-n` | `80` | Number of Q&A turn role samples |
| `--boundary-n` | `30` | Number of document boundary samples |
| `--initiation-n` | `50` | Number of initiation exchange samples |

**Outputs** (in `--output-dir`):
- `ai_sentence_audit.csv` — sentence-level AI keyword validation template
- `role_audit_qa_turns.csv` — Q&A turn role validation template
- `qa_boundary_audit_docs.csv` — document boundary / pairing spot-check template
- `initiation_audit_exchanges.csv` — Q&A exchange initiation label template
- `*.jsonl` — full earnings-call context sidecars for the annotation webapp

---

## 📊 Data Sources

| Dataset | Location in repo | Access method |
|---|---|---|
| S&P 500 Earnings Transcripts (2020–2025) | `data/final_dataset.parquet` / `.csv` | Cloned via Git LFS (no extra step needed) |
| WRDS Compustat Fundamentals Quarterly | `data/wrds.csv` | Cloned via Git LFS (no extra step needed) |
| Human Annotation Ground Truth | `data/human_annotation/*.csv` | Committed to repo directly |

### WRDS Variable Reference

The `data/wrds.csv` file was downloaded from **Compustat North America — Fundamentals Quarterly** with the following variables selected:

```
costat, curcdq, datafmt, indfmt, consol, tic, datadate, gvkey, conm,
ggroup, gind, gsector, gsubind, naics, sic, spcindcd, datafqtr, fqtr,
fyearq, cshoq, epspxq, xrdq, capxy, mkvaltq, prccq
```

The file is included in the repository for direct use by TAs without any additional download.

---

## 📄 License

Academic research project — Text Mining for Economics and Finance, ICBS FinTech Programme.
