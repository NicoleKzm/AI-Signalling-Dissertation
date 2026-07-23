# Talk vs. Performance - Does the operational specificity of AI disclosure predict the financial performance of listed European B2C e-commerce firms over the period 2021 to 2025?

Code and data for the dissertation's empirical analysis: LLM classification of
AI-related disclosure in annual reports, panel construction, and two-way
fixed effects regression against financial outcomes.

## Source of truth

**`LOCKED_NUMBERS.md` is the single source of truth for every reported
statistic** — coefficients, CIs, kappa, TOST, Hausman, robustness checks. If a
number in the write up doesn't match `LOCKED_NUMBERS.md`, the write up is
wrong, not the other way around.

## Setup

Developed and run on Python 3.13.

```bash
pip install -r requirements.txt
```

`classify.py` calls the Claude API and needs a local `.env` (git-ignored) containing:

```
ANTHROPIC_API_KEY=sk-ant-...
```

No other script needs an API key or network access.

## A note on reproducibility

`classify.py` runs at `temperature=0`, but LLM APIs are **not guaranteed
bit-for-bit deterministic** across calls — re-running it can shift individual
passage classifications. Because of this, `all_classifications.csv` and
`signalling_scores.csv` are committed as the fixed, authoritative outputs of
that step. **Every downstream script reads these committed CSVs, not
`classify.py`'s live output** — reproducing the regression tables, TOST,
kappa, and Chapter 4/Appendix C results does **not** require an API key or
re-running classification. `LOCKED_NUMBERS.md` explicitly warns against
re-running `classify.py` for this reason.

## Repository layout

```
src/         classify.py, merge.py, regression_clean.py, run_lag_models.py,
             financial_data.py - the core pipeline-generation scripts
validation/  every QA/robustness/inference script (kappa, extraction
             precision/recall, determinism, leave-one-out, TOST,
             small-sample inference, Chapter 4/Appendix C robustness checks)
data/        every input CSV - raw pipeline data plus hand-coded/sample
             validation inputs (second_coder_sample.csv, kappa_sample.csv, etc.)
results/     regression_results.csv and every robustness/validation output CSV
figures/     make_figures*.py, plot_results.py, and figure outputs (PNG/PDF)
deprecated/  abandoned or superseded code (FinBERT branch, the old
             contemporaneous-only regression.py, the broken lagged_models.py
             duplicate) - not part of the reported analysis
```

All scripts assume they are run **from the repository root** (e.g.
`python3 src/regression_clean.py`, not `cd src && python3 regression_clean.py`)
- every path inside a script is root-relative (`data/...`, `results/...`),
not relative to the script's own folder.

## Pipeline, in order

1. **Extraction + classification** — `src/classify.py` reads annual report
   PDFs (path set by `ANNUAL_REPORTS_DIR` in the script), extracts AI relevant
   passages via a two-tier keyword screen, and classifies each with Claude
   (Symbolic/Transitional/Substantive) → `data/signalling_scores.csv`
   (firm-year scores) + `data/all_classifications.csv` (passage-level, 512 rows).

2. **Financial data** — `src/financial_data.py` pulls stock price/revenue data
   via yfinance → `data/financial_data.csv`. This was manually completed
   (revenue backfill, EUR standardisation) into `data/final_dataset.csv`,
   which is what the next step actually uses — `financial_data.csv` is not a
   direct input to anything downstream.

3. **Panel assembly** — `src/merge.py` joins `data/signalling_scores.csv` with
   `data/final_dataset.csv` on firm + year → `data/panel_dataset.csv`, the
   panel every analysis script below reads.

4. **Primary regression** — `src/regression_clean.py` (imports
   `validation/retest_table_4_8.py` for the TOST bound/function) reads
   `data/panel_dataset.csv` → `results/regression_results.csv`.
   `deprecated/regression.py` is an earlier, superseded version (sector
   dummies, no lag-1 spec) kept for audit trail only — not part of the
   reported results.

5. **Chapter 4 / Appendix C analyses** (all in `validation/`, all reading
   `data/panel_dataset.csv`):
   - `h1_lag_primary.py` → `results/h1_lag_primary_results.csv` (H1 lag-1 primary spec)
   - `small_sample_inference.py` → `results/small_sample_inference.csv`
     (CR2/Bell-McCaffrey, wild cluster bootstrap; picks up H1's lag-1 spec if
     `results/h1_lag_primary_results.csv` is present)
   - `randomisation_inference.py` → `results/randomisation_inference_results.csv`
   - `tost_mde_v2.py` (reads `results/regression_results.csv` too) →
     `results/tost_mde_results.csv`
   - `chapter4_gaps.py` (reads `results/small_sample_inference.csv` too) →
     `results/chapter4_gaps.csv` — descriptives, correlation matrix, H1 CI,
     Table 4.9 robustness. **Asserts its recomputed H1/H2/H3 coefficients
     match `LOCKED_NUMBERS.md` exactly and exits with an error if they
     don't** — the closest thing this repo has to a single reproduction check.
   - `fye_robustness.py`, `h2_exclusion_smallsample.py`,
     `aggregation_sensitivity.py`, `fragmentation_robustness.py`,
     `two_part_specification.py`, `leave_one_out_primary.py` — further
     robustness checks, each independently writing its own `results/*.csv`.
   - `figures/make_regression_figures.py` (reads `results/regression_results.csv`
     only, no recomputation) → `figures/fig_4_4_primary_estimates.png` — the
     four-point forest plot of the primary estimates.

6. **Validation** (all in `validation/`) — `kappa_retest_v3.py` (calibration
   round, `data/kappa_sample.csv`), `fresh_kappa_sample_v2.py` (draws the
   held-out sample; `human_label` column is filled in by hand, not by
   script), `second_coder_kappa.py` (reads `data/second_coder_sample.csv`,
   also hand-labeled). `validity_check.py` is **deprecated** — see its own
   header block; superseded by `randomisation_inference.py` and
   `small_sample_inference.py`.

`extraction_recall.py`, `extraction_precision*.py`, `determinism_check.py`,
`check_tier_consistency.py`, `verify_zero_score_pct.py`, and
`verify_locked_numbers.py` (all in `validation/`) are targeted QA checks
against `data/all_classifications.csv`/`data/panel_dataset.csv`, not part of
the main chain.

## Reproducing the Chapter 4 / Appendix C results

Run from the repository root:

```bash
pip install -r requirements.txt
python3 src/regression_clean.py
python3 validation/h1_lag_primary.py
python3 validation/small_sample_inference.py
python3 validation/randomisation_inference.py
python3 validation/tost_mde_v2.py
python3 validation/chapter4_gaps.py
python3 figures/make_regression_figures.py
```

No API key needed — all of the above run entirely from the committed
`data/panel_dataset.csv`, `data/all_classifications.csv`, and each other's
outputs in `results/`. `chapter4_gaps.py` will stop with an error if its
recomputed coefficients don't match `LOCKED_NUMBERS.md`. A clean run
confirms this reproduces the locked numbers exactly.
`figures/make_regression_figures.py` reads only `results/regression_results.csv`
(no recomputation) and writes Figure 4.4 to `figures/`, created automatically
if missing.

## `deprecated/`

Holds `finbert.py` and `finbert_news.py` (an abandoned FinBERT
sentiment-analysis approach on news headlines, dropped in favour of the LLM
passage classification above, plus their `all_headlines.csv`/
`sentiment_scores.csv` inputs), `regression.py` and its `robustness_results.csv`
output (the superseded pre-lag-1 regression script), `lagged_models.py` (a
broken duplicate of `src/run_lag_models.py` referencing a non-existent
`data/processed/` path), and `panel_dataset.py` (a dead debug scratch file).
Not part of the reported analysis.
