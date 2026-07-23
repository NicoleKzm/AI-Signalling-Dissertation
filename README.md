# Talk vs. Performance: Does the operational specificity of AI disclosure predict the financial performance of listed European B2C e-commerce firms over the period 2021 to 2025?

This repo is the reproduction code and data for the dissertation's empirical
analysis: LLM classification of AI-related disclosure in European e-commerce
annual reports (2021-2025), panel construction, and two-way fixed effects
regression against financial outcomes (stock price movement, revenue growth,
gross margin).

**`LOCKED_NUMBERS.md` is the single source of truth for every reported
statistic** — coefficients, standard errors, CIs, kappa, TOST, Hausman,
robustness checks. If a number in the dissertation doesn't match
`LOCKED_NUMBERS.md`, the dissertation is wrong, not the other way around.

## Repository structure

| Folder | Contents |
|---|---|
| `src/` | The core pipeline: extraction/classification, financial data, panel assembly, primary regression. |
| `validation/` | Every QA and robustness script: kappa reliability, extraction precision/recall, determinism, leave-one-out, TOST/equivalence, small-sample inference, Chapter 4/Appendix C robustness checks. |
| `data/` | Input CSVs — raw pipeline data plus hand-coded validation samples (`second_coder_sample.csv`, `kappa_sample.csv`, etc.). |
| `results/` | `regression_results.csv` and every robustness/validation output CSV. |
| `figures/` | Figure-generating scripts and their PNG/PDF outputs. |
| `deprecated/` | Abandoned or superseded code (an earlier FinBERT sentiment-analysis branch, the pre-lag-1 `regression.py`, a broken duplicate script) — not part of the reported analysis. |

## Reproduction

```bash
git clone https://github.com/NicoleKzm/AI-Signalling-Dissertation.git
cd AI-Signalling-Dissertation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 src/regression_clean.py         # primary H1/H2/H3 regressions
python3 figures/make_regression_figures.py
```

`src/regression_clean.py` reproduces every primary and robustness coefficient
in one run and hard-fails if its output doesn't match `LOCKED_NUMBERS.md`.
Everything in `validation/` (kappa, TOST, small-sample inference, Chapter 4
gap-fill checks, etc.) is supporting evidence, not required to reproduce the
headline numbers — run any of it individually if you want to verify a
specific robustness claim.

**The pipeline chain — four scripts, each feeding the next:**

1. `src/classify.py` → extracts + classifies AI disclosure passages → `data/signalling_scores.csv`, `data/all_classifications.csv`
2. `src/financial_data.py` → pulls stock price/revenue data → `data/financial_data.csv` (manually completed into `data/final_dataset.csv`)
3. `src/merge.py` → joins the two on firm + year → `data/panel_dataset.csv`
4. `src/regression_clean.py` → runs the primary and robustness regressions → `results/regression_results.csv`

Everything else in `validation/` and `figures/` reads from these outputs; none
of it re-runs the pipeline itself.

## A note on reproducibility

`src/classify.py` calls the Claude API at `temperature=0`, but LLM APIs are
**not guaranteed bit-for-bit deterministic** across calls — re-running it can
shift individual passage classifications. Because of this,
`data/all_classifications.csv` and `data/signalling_scores.csv` are committed
as the fixed, authoritative outputs of that step. **Every downstream script
reads these committed CSVs, not `classify.py`'s live output** — reproducing
the regression tables, TOST, kappa, and Chapter 4/Appendix C results does
**not** require an API key or re-running classification.

If you do need to re-run classification, `classify.py` reads annual report
PDFs (path set by `ANNUAL_REPORTS_DIR` in the script) and needs a local
`.env` (git-ignored) containing `ANTHROPIC_API_KEY=sk-ant-...`. No other
script needs an API key or network access.
