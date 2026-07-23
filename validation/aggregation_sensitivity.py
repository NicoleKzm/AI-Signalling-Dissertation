"""
aggregation_sensitivity.py

Supervisor critique (items 15-16): a mean signal score can FALL when a firm
adds a symbolic passage even alongside genuine substantive disclosure --
the aggregation rule itself is a modelling choice, not a neutral summary.
Tests sensitivity to five alternative firm-year aggregations, plus a
zero-passage-exclusion specification. No such results exist in
regression_results.csv -- this is a from-scratch build.

Mirrors regression_clean.py's primary specification exactly: PanelOLS,
entity+time FE, firm-clustered SE, Revenue column, H1 contemporaneous
(full N=69), H2/H3 lag-1 IV + lag-1 log_revenue control, Zalando 2025
excluded from H2/H3 only.

Does not modify classify.py, regression_clean.py, regression_results.csv,
signalling_scores.csv, or all_classifications.csv. Reads all_classifications.csv,
signalling_scores.csv, panel_dataset.csv. Writes only to
aggregation_sensitivity.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats

ALPHA = 0.05
H3_BOUND = 3.462
SCORE_MAP = {'Symbolic': 0, 'Transitional': 1, 'Substantive': 2}

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Build all 69 firm-years' base identity from signalling_scores.csv
# (this already has every firm-year, including the 21 with zero passages,
# which never appear as rows in all_classifications.csv at all)
# ══════════════════════════════════════════════════════════════════
base_fy = pd.read_csv('data/signalling_scores.csv')[['firm', 'year', 'total_passages', 'mean_signal_score']].copy()
base_fy = base_fy.rename(columns={'mean_signal_score': 'mean_signal_score_baseline'})

classifications = pd.read_csv('data/all_classifications.csv')
classifications['passage_score'] = classifications['classification'].map(SCORE_MAP)

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Construct the five alternative measures
# ══════════════════════════════════════════════════════════════════
print("=" * 90)
print("STEP 2: Constructing five alternative firm-year measures")
print("=" * 90)

agg = classifications.groupby(['firm', 'year']).agg(
    mean_signal_score=('passage_score', 'mean'),
    max_signal_score=('passage_score', 'max'),
    passage_count=('passage_score', 'count'),
).reset_index()
# Round to 3dp to match classify.py's get_firm_year_score() EXACTLY (it
# stores round(mean_signal, 3)) -- without this, mean_signal_score would
# still equal signalling_scores.csv after display rounding, but the
# regression itself would run on higher-precision values than the
# production pipeline ever actually used, producing a tiny but real
# drift from the canonical regression_results.csv primary H3 figure.
agg['mean_signal_score'] = agg['mean_signal_score'].round(3)


# modal_signal_score: modal passage category, converted to score.
# TIE-BREAKING RULE (explicit, mirrors classify.py's own modal_class()
# exactly, for consistency with the production pipeline's own convention):
# if there is a single most-frequent category, use it; if tied for most
# frequent, break the tie using the firm-year's mean score threshold
# (mean >= 2.0 -> Substantive, >= 1.0 -> Transitional, else Symbolic).
def modal_with_tiebreak(group):
    counts = group['classification'].value_counts()
    top_n = counts.max()
    top_classes = counts[counts == top_n].index.tolist()
    if len(top_classes) == 1:
        return SCORE_MAP[top_classes[0]]
    mean_score = group['passage_score'].mean()
    if mean_score >= 2.0:
        return SCORE_MAP['Substantive']
    elif mean_score >= 1.0:
        return SCORE_MAP['Transitional']
    else:
        return SCORE_MAP['Symbolic']


modal = classifications.groupby(['firm', 'year']).apply(modal_with_tiebreak, include_groups=False)
modal.name = 'modal_signal_score'
agg = agg.merge(modal.reset_index(), on=['firm', 'year'], how='left')

# Merge onto the full 69-row base; firm-years with zero passages get 0 for
# all measures except log_passage_count (log(1+0)=0, also 0).
panel_measures = base_fy.merge(agg, on=['firm', 'year'], how='left')
for col in ['mean_signal_score', 'max_signal_score', 'modal_signal_score', 'passage_count']:
    panel_measures[col] = panel_measures[col].fillna(0)
panel_measures['log_passage_count'] = np.log1p(panel_measures['passage_count'])

# ── Verification: mean_signal_score must match signalling_scores.csv exactly ──
mismatch = (panel_measures['mean_signal_score'].round(3) != panel_measures['mean_signal_score_baseline'].round(3))
print(f"mean_signal_score reproduction check: {mismatch.sum()} mismatches out of {len(panel_measures)}")
if mismatch.any():
    print("MISMATCHES:")
    print(panel_measures.loc[mismatch, ['firm', 'year', 'mean_signal_score', 'mean_signal_score_baseline']])
else:
    print("CONFIRMED: reproduced mean_signal_score matches signalling_scores.csv exactly for all 69 firm-years.")

MEASURES = ['mean_signal_score', 'max_signal_score', 'modal_signal_score', 'passage_count', 'log_passage_count']

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Descriptive statistics + pairwise correlations
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 3: Descriptive statistics (N=69)")
print("=" * 90)
desc = panel_measures[MEASURES].describe().T
print(desc.to_string())

print("\nPairwise correlations:")
corr = panel_measures[MEASURES].corr()
print(corr.round(4).to_string())

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Merge into panel_dataset.csv structure, build lags
# ══════════════════════════════════════════════════════════════════
panel = pd.read_csv('data/panel_dataset.csv')
panel = panel.drop(columns=['mean_signal_score', 'total_passages'], errors='ignore').merge(
    panel_measures[['firm', 'year'] + MEASURES + ['total_passages']], on=['firm', 'year'], how='left')
panel = panel.sort_values(['firm', 'year']).reset_index(drop=True)

panel['log_revenue'] = np.log(panel['Revenue'].replace(0, np.nan))
year_gap1 = panel.groupby('firm')['year'].diff()
panel['log_revenue_lag1'] = panel.groupby('firm')['log_revenue'].shift(1)
panel.loc[year_gap1 != 1, 'log_revenue_lag1'] = np.nan

for m in MEASURES:
    lag_col = f'{m}_lag1'
    panel[lag_col] = panel.groupby('firm')[m].shift(1)
    panel.loc[year_gap1 != 1, lag_col] = np.nan

panel = panel.set_index(['firm', 'year'])
zalando_2025_mask = ~((panel.index.get_level_values('firm') == 'Zalando') &
                       (panel.index.get_level_values('year') == 2025))
panel_excl_zalando = panel[zalando_2025_mask]

HYP_DVS = {'H1': 'Stock_Price_Movement_%', 'H2': 'Revenue_Growth_%', 'H3': 'Gross_Margin_%'}


def fit(data, dv, iv, control):
    cols = [dv, iv, control]
    model_df = data[cols].dropna()
    n = len(model_df)
    if n < 10:
        return None, n
    exog = sm.add_constant(model_df[[iv, control]])
    result = PanelOLS(model_df[dv], exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    return result, n


rows_out = []
h3_fitted = {}  # measure_label -> (result, iv_name) for the TOST step


def record(hyp, measure_label, iv, result, n):
    if result is None:
        print(f"  {hyp} [{measure_label}]: INSUFFICIENT N ({n})")
        rows_out.append({'Hypothesis': hyp, 'Measure': measure_label, 'IV': iv,
                          'Coefficient': np.nan, 'Std_Error': np.nan, 'T_stat': np.nan,
                          'P_value': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan, 'N': n})
        return
    beta = result.params[iv]
    se = result.std_errors[iv]
    t = result.tstats[iv]
    p = result.pvalues[iv]
    ci = result.conf_int().loc[iv]
    print(f"  {hyp} [{measure_label:22s}]: beta={beta:10.4f}  SE={se:9.4f}  t={t:8.4f}  "
          f"p={p:.4f}  95% CI=[{ci['lower']:10.4f}, {ci['upper']:10.4f}]  N={n}")
    rows_out.append({'Hypothesis': hyp, 'Measure': measure_label, 'IV': iv,
                      'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
                      'T_stat': round(t, 4), 'P_value': round(p, 4),
                      'CI_lower': round(ci['lower'], 4), 'CI_upper': round(ci['upper'], 4), 'N': n})
    if hyp == 'H3':
        h3_fitted[measure_label] = (result, iv, n)


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Re-estimate H1/H2/H3 for each of the five measures
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 5: Primary regressions, each alternative measure as IV")
print("=" * 90)

for hyp, dv in HYP_DVS.items():
    print(f"\n-- {hyp}: DV={dv} --")
    for m in MEASURES:
        if hyp == 'H1':
            result, n = fit(panel, dv, m, 'log_revenue')
            record(hyp, m, m, result, n)
        else:
            iv_lag = f'{m}_lag1'
            result, n = fit(panel_excl_zalando, dv, iv_lag, 'log_revenue_lag1')
            record(hyp, m, iv_lag, result, n)

# ══════════════════════════════════════════════════════════════════
# STEP 6 — Zero-passage-exclusion specification, mean_signal_score only
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 6: Excluding zero-passage firm-years (current-year total_passages > 0), "
      "mean_signal_score")
print("=" * 90)

panel_nonzero = panel[panel['total_passages'] > 0]
panel_nonzero_excl_zalando = panel_excl_zalando[panel_excl_zalando['total_passages'] > 0]
n_excluded = int((panel['total_passages'] == 0).sum())
print(f"Firm-years excluded (zero passages): {n_excluded} [expected 21]")

for hyp, dv in HYP_DVS.items():
    if hyp == 'H1':
        result, n = fit(panel_nonzero, dv, 'mean_signal_score', 'log_revenue')
        record(hyp, 'mean_signal_score_ExclZero', 'mean_signal_score', result, n)
    else:
        result, n = fit(panel_nonzero_excl_zalando, dv, 'mean_signal_score_lag1', 'log_revenue_lag1')
        record(hyp, 'mean_signal_score_ExclZero', 'mean_signal_score_lag1', result, n)

# ══════════════════════════════════════════════════════════════════
# STEP 7 — H3 TOST for each alternative measure, df = df_resid (read
# directly from the fitted model object)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 7: H3 TOST equivalence (bound=+/-3.462), df=df_resid from fitted model")
print("=" * 90)

tost_rows = []
for measure_label, (result, iv, n) in h3_fitted.items():
    beta = result.params[iv]
    se = result.std_errors[iv]
    dof = result.df_resid
    t_lower = (beta - (-H3_BOUND)) / se
    t_upper = (H3_BOUND - beta) / se
    p_lower = 1 - stats.t.cdf(t_lower, dof)
    p_upper = 1 - stats.t.cdf(t_upper, dof)
    equivalent = (p_lower < ALPHA) and (p_upper < ALPHA)
    print(f"  {measure_label:28s}: beta={beta:8.4f}  SE={se:7.4f}  N={n:3d}  df_resid={dof:5.1f}  "
          f"p_lower={p_lower:.4f}  p_upper={p_upper:.4f}  equivalent={equivalent}")
    tost_rows.append({'Measure': measure_label, 'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
                       'N': n, 'df_resid': dof, 'bound': H3_BOUND,
                       'p_lower': round(p_lower, 4), 'p_upper': round(p_upper, 4),
                       'equivalent': equivalent})

tost_df = pd.DataFrame(tost_rows)

# ══════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(rows_out)
results_df.to_csv('results/aggregation_sensitivity.csv', index=False)
print(f"\nresults/aggregation_sensitivity.csv saved -- {len(results_df)} rows")

tost_df.to_csv('results/aggregation_sensitivity_h3_tost.csv', index=False)
print(f"results/aggregation_sensitivity_h3_tost.csv saved -- {len(tost_df)} rows")

print("\n" + "=" * 90)
print("SUMMARY TABLE -- coefficient x measure x hypothesis")
print("=" * 90)
pivot = results_df.pivot_table(index='Hypothesis', columns='Measure', values='Coefficient', aggfunc='first')
col_order = MEASURES + ['mean_signal_score_ExclZero']
pivot = pivot[[c for c in col_order if c in pivot.columns]]
print(pivot.to_string())

print("\n" + "=" * 90)
print("SUMMARY TABLE -- p-value x measure x hypothesis")
print("=" * 90)
pivot_p = results_df.pivot_table(index='Hypothesis', columns='Measure', values='P_value', aggfunc='first')
pivot_p = pivot_p[[c for c in col_order if c in pivot_p.columns]]
print(pivot_p.to_string())

print("\nDone.")
