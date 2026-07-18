# Talk vs. Performance — AI Signalling Dissertation

Code and data for the dissertation's empirical analysis: LLM classification of
AI-related disclosure in annual reports, panel construction, and two-way
fixed-effects regression against financial outcomes.

## Source of truth

**`LOCKED_NUMBERS.md` is the single source of truth for every reported
statistic** — coefficients, CIs, kappa, TOST, Hausman, robustness checks. If a
number in the write-up doesn't match `LOCKED_NUMBERS.md`, the write-up is
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

`classify.py` runs at `temperature=0` but LLM APIs are **not guaranteed
bit-for-bit deterministic** across calls — re-running it can shift individual
passage classifications. Because of this, `all_classifications.csv` and
`signalling_scores.csv` are committed as the fixed, authoritative outputs of
that step. **Every downstream script reads these committed CSVs, not
`classify.py`'s live output** — reproducing the regression tables, TOST,
kappa, and Chapter 4/Appendix C results does **not** require an API key or
re-running classification. `LOCKED_NUMBERS.md` explicitly warns against
re-running `classify.py` for this reason.

## Pipeline, in order

1. **Extraction + classification** — `classify.py` reads annual report PDFs
   (path set by `ANNUAL_REPORTS_DIR` in the script), extracts AI-relevant
   passages via a two-tier keyword screen, classifies each with Claude
   (Symbolic/Transitional/Substantive) → `signalling_scores.csv` (firm-year
   scores) + `all_classifications.csv` (passage-level, 512 rows).

2. **Financial data** — `financial_data.py` pulls stock price/revenue data via
   yfinance → `financial_data.csv`. This was manually completed (revenue
   backfill, EUR standardisation) into `final_dataset.csv`, which is what the
   next step actually uses — `financial_data.csv` is not a direct input to
   anything downstream.

3. **Panel assembly** — `merge.py` joins `signalling_scores.csv` with
   `final_dataset.csv` on firm + year → `panel_dataset.csv`, the panel every
   analysis script below reads.

4. **Primary regression** — `regression_clean.py` (imports `retest_table_4_8.py`
   for the TOST bound/function) reads `panel_dataset.csv` → `regression_results.csv`.
   `regression.py` is an earlier, superseded version (sector dummies, no
   lag-1 spec) kept for audit trail only — not part of the reported results.

5. **Chapter 4 / Appendix C analyses**, all reading `panel_dataset.csv`:
   - `h1_lag_primary.py` → `h1_lag_primary_results.csv` (H1 lag-1 primary spec)
   - `small_sample_inference.py` → `small_sample_inference.csv` (CR2/Bell-McCaffrey,
     wild cluster bootstrap; picks up H1's lag-1 spec if `h1_lag_primary_results.csv`
     is present)
   - `randomisation_inference.py` → `randomisation_inference_results.csv`
   - `tost_mde_v2.py` (reads `regression_results.csv` too) → `tost_mde_results.csv`
   - `chapter4_gaps.py` (reads `small_sample_inference.csv` too) →
     `chapter4_gaps.csv` — descriptives, correlation matrix, H1 CI, Table 4.9
     robustness. **Asserts its recomputed H1/H2/H3 coefficients match
     `LOCKED_NUMBERS.md` exactly and exits with an error if they don't** — the
     closest thing this repo has to a single reproduction check.
   - `fye_robustness.py`, `h2_exclusion_smallsample.py`,
     `aggregation_sensitivity.py`, `fragmentation_robustness.py`,
     `two_part_specification.py`, `leave_one_out_primary.py` — further
     robustness checks, each independently writing its own `*.csv`.

6. **Validation** — `kappa_retest_v3.py` (calibration round, `kappa_sample.csv`),
   `fresh_kappa_sample_v2.py` (draws the held-out sample; `human_label` column
   is filled in by hand, not by script), `second_coder_kappa.py` (reads
   `second_coder_sample.csv`, also hand-labeled). `validity_check.py` is
   **deprecated** — see its own header block; superseded by
   `randomisation_inference.py` and `small_sample_inference.py`.

`extraction_recall.py`, `extraction_precision*.py`, `determinism_check.py`,
`check_tier_consistency.py`, `verify_zero_score_pct.py`, and
`verify_locked_numbers.py` are targeted QA checks against
`all_classifications.csv`/`panel_dataset.csv`, not part of the main chain.

## Reproducing the Chapter 4 / Appendix C results

```bash
pip install -r requirements.txt
python3 regression_clean.py
python3 h1_lag_primary.py
python3 small_sample_inference.py
python3 randomisation_inference.py
python3 tost_mde_v2.py
python3 chapter4_gaps.py
```

No API key needed — all of the above run entirely from the committed
`panel_dataset.csv`, `all_classifications.csv`, and each other's outputs.
`chapter4_gaps.py` will stop with an error if its recomputed coefficients
don't match `LOCKED_NUMBERS.md`; a clean run confirms this reproduces the
locked numbers exactly.

## `deprecated/`

Holds `finbert.py` and `finbert_news.py` — an abandoned FinBERT
sentiment-analysis approach on news headlines, dropped in favour of the LLM
passage classification above. Not part of the reported analysis.
