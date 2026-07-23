"""
two_part_specification.py

Builds the two-part (extensive/intensive margin) specification requested by
the supervisor's critique (items 13-14): coding zero-passage firm-years as
0 makes silence numerically identical to vague disclosure, which cheap-talk
theory does not require. This decomposes mean_signal_score into:
  - discloses_ai: whether the firm-year disclosed anything at all
  - conditional_signal: how substantiated that disclosure was, GIVEN disclosure

No such model exists in regression_results.csv -- this is a new, from-scratch
build, mirroring regression_clean.py's primary specification exactly
(PanelOLS, entity+time FE, firm-clustered SE, Revenue column, Zalando 2025
excluded from H2/H3, lag-1 control for H2/H3).

CORRECTED per explicit follow-up instruction: for H2/H3 the treatment
variables are now LAGGED too (discloses_ai_lag1, conditional_signal_lag1,
conditional_signal_filled_lag1), matching the primary specification's use
of signal_lag1 -- not just the control. H1 remains fully contemporaneous.
conditional_signal_filled is filled with 0 BEFORE lagging (not after), so
"prior year disclosed nothing" correctly lags to 0, while a genuine panel
gap (e.g. About You's missing year) still correctly lags to NaN via the
same year-gap guard used everywhere else in this codebase.

Does not modify classify.py, regression_clean.py, regression_results.csv,
or signalling_scores.csv. Reads panel_dataset.csv and signalling_scores.csv.
Writes only to two_part_results.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# ══════════════════════════════════════════════════════════════════
# Construct discloses_ai / conditional_signal from signalling_scores.csv
# ══════════════════════════════════════════════════════════════════
scores = pd.read_csv('data/signalling_scores.csv')
scores['discloses_ai'] = (scores['total_passages'] > 0).astype(int)
scores['conditional_signal'] = scores['mean_signal_score'].where(scores['total_passages'] > 0, np.nan)

n_disclose = int((scores['discloses_ai'] == 1).sum())
n_nondisclose = int((scores['discloses_ai'] == 0).sum())
print("=" * 90)
print("CONSTRUCTED VARIABLES (from signalling_scores.csv)")
print("=" * 90)
print(f"discloses_ai = 1: {n_disclose} firm-years  [expected 48]")
print(f"discloses_ai = 0: {n_nondisclose} firm-years  [expected 21]")
print(f"conditional_signal non-missing: {scores['conditional_signal'].notna().sum()} "
      f"(mean={scores.loc[scores['discloses_ai']==1,'conditional_signal'].mean():.4f})")

# ══════════════════════════════════════════════════════════════════
# Merge into panel_dataset.csv's structure (financial covariates, DVs) --
# mirrors regression_clean.py's data construction exactly.
# ══════════════════════════════════════════════════════════════════
panel = pd.read_csv('data/panel_dataset.csv')
panel = panel.drop(columns=['mean_signal_score'], errors='ignore').merge(
    scores[['firm', 'year', 'discloses_ai', 'conditional_signal']], on=['firm', 'year'], how='left')
panel = panel.sort_values(['firm', 'year']).reset_index(drop=True)

panel['log_revenue'] = np.log(panel['Revenue'].replace(0, np.nan))
year_gap1 = panel.groupby('firm')['year'].diff()
panel['log_revenue_lag1'] = panel.groupby('firm')['log_revenue'].shift(1)
panel.loc[year_gap1 != 1, 'log_revenue_lag1'] = np.nan

# Model 3's filled intensive-margin variable: 0 where discloses_ai == 0,
# equal to conditional_signal where discloses_ai == 1 (identical to
# .fillna(0) given conditional_signal is NaN exactly when discloses_ai==0).
panel['conditional_signal_filled'] = panel['conditional_signal'].fillna(0.0)

# ── Lagged treatment variables for H2/H3 (matches primary spec's signal_lag1) ──
# conditional_signal_filled is lagged AFTER filling, so "prior year had zero
# disclosure" correctly lags to 0, while a genuine panel gap still lags to
# NaN via the same year-gap guard.
panel['discloses_ai_lag1'] = panel.groupby('firm')['discloses_ai'].shift(1)
panel.loc[year_gap1 != 1, 'discloses_ai_lag1'] = np.nan
panel['conditional_signal_lag1'] = panel.groupby('firm')['conditional_signal'].shift(1)
panel.loc[year_gap1 != 1, 'conditional_signal_lag1'] = np.nan
panel['conditional_signal_filled_lag1'] = panel.groupby('firm')['conditional_signal_filled'].shift(1)
panel.loc[year_gap1 != 1, 'conditional_signal_filled_lag1'] = np.nan

panel = panel.set_index(['firm', 'year'])
zalando_2025_mask = ~((panel.index.get_level_values('firm') == 'Zalando') &
                       (panel.index.get_level_values('year') == 2025))
panel_excl_zalando = panel[zalando_2025_mask]

HYPS = {
    'H1': {'dv': 'Stock_Price_Movement_%', 'data': panel, 'control': 'log_revenue', 'lag': False,
           'extensive_iv': 'discloses_ai', 'intensive_iv': 'conditional_signal',
           'filled_iv': 'conditional_signal_filled'},
    'H2': {'dv': 'Revenue_Growth_%', 'data': panel_excl_zalando, 'control': 'log_revenue_lag1', 'lag': True,
           'extensive_iv': 'discloses_ai_lag1', 'intensive_iv': 'conditional_signal_lag1',
           'filled_iv': 'conditional_signal_filled_lag1'},
    'H3': {'dv': 'Gross_Margin_%', 'data': panel_excl_zalando, 'control': 'log_revenue_lag1', 'lag': True,
           'extensive_iv': 'discloses_ai_lag1', 'intensive_iv': 'conditional_signal_lag1',
           'filled_iv': 'conditional_signal_filled_lag1'},
}


def fit(data, dv, ivs, control):
    cols = [dv] + ivs + [control]
    model_df = data[cols].dropna()
    n = len(model_df)
    if n < 10:
        return None, n
    exog = sm.add_constant(model_df[ivs + [control]])
    result = PanelOLS(model_df[dv], exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    return result, n


rows_out = []


def record(hyp, model_label, result, n, iv_of_interest, extra_iv=None):
    if result is None:
        print(f"  {hyp} {model_label}: INSUFFICIENT N ({n})")
        rows_out.append({'Hypothesis': hyp, 'Model': model_label, 'IV': iv_of_interest,
                          'Coefficient': np.nan, 'Std_Error': np.nan, 'T_stat': np.nan,
                          'P_value': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan,
                          'N': n, 'R2_within': np.nan})
        return
    beta = result.params[iv_of_interest]
    se = result.std_errors[iv_of_interest]
    t = result.tstats[iv_of_interest]
    p = result.pvalues[iv_of_interest]
    ci = result.conf_int().loc[iv_of_interest]
    r2 = result.rsquared
    print(f"  {hyp} {model_label} [{iv_of_interest}]: beta={beta:.4f}, SE={se:.4f}, t={t:.4f}, "
          f"p={p:.4f}, 95% CI=[{ci['lower']:.4f}, {ci['upper']:.4f}], N={n}, R2_within={r2:.4f}")
    rows_out.append({'Hypothesis': hyp, 'Model': model_label, 'IV': iv_of_interest,
                      'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
                      'T_stat': round(t, 4), 'P_value': round(p, 4),
                      'CI_lower': round(ci['lower'], 4), 'CI_upper': round(ci['upper'], 4),
                      'N': n, 'R2_within': round(r2, 4)})
    if extra_iv:
        beta2 = result.params[extra_iv]
        se2 = result.std_errors[extra_iv]
        t2 = result.tstats[extra_iv]
        p2 = result.pvalues[extra_iv]
        ci2 = result.conf_int().loc[extra_iv]
        print(f"  {hyp} {model_label} [{extra_iv}]: beta={beta2:.4f}, SE={se2:.4f}, t={t2:.4f}, "
              f"p={p2:.4f}, 95% CI=[{ci2['lower']:.4f}, {ci2['upper']:.4f}], N={n}, R2_within={r2:.4f}")
        rows_out.append({'Hypothesis': hyp, 'Model': model_label, 'IV': extra_iv,
                          'Coefficient': round(beta2, 4), 'Std_Error': round(se2, 4),
                          'T_stat': round(t2, 4), 'P_value': round(p2, 4),
                          'CI_lower': round(ci2['lower'], 4), 'CI_upper': round(ci2['upper'], 4),
                          'N': n, 'R2_within': round(r2, 4)})


n_summary = {}

for hyp, spec in HYPS.items():
    ext_iv, int_iv, filled_iv = spec['extensive_iv'], spec['intensive_iv'], spec['filled_iv']
    print("\n" + "=" * 90)
    print(f"{hyp}: DV={spec['dv']}, treatment lag={'yes (lag-1)' if spec['lag'] else 'no (contemporaneous)'}, "
          f"control={spec['control']}" + (" (Zalando 2025 excluded)" if spec['lag'] else ""))
    print("=" * 90)

    # MODEL 1 -- extensive margin, full sample
    r1, n1 = fit(spec['data'], spec['dv'], [ext_iv], spec['control'])
    record(hyp, 'Model1_extensive', r1, n1, ext_iv)

    # MODEL 2 -- intensive margin, disclosers only (disclosure status as of
    # the SAME period as the treatment variable -- current-year for H1,
    # lag-1 for H2/H3, so the subsample matches where int_iv is defined)
    disclosers_only = spec['data'][spec['data'][ext_iv] == 1]
    r2, n2 = fit(disclosers_only, spec['dv'], [int_iv], spec['control'])
    record(hyp, 'Model2_intensive', r2, n2, int_iv)

    # MODEL 3 -- both, full sample, filled intensive-margin variable
    r3, n3 = fit(spec['data'], spec['dv'], [ext_iv, filled_iv], spec['control'])
    record(hyp, 'Model3_both', r3, n3, ext_iv, extra_iv=filled_iv)

    n_summary[hyp] = {'model1_n': n1, 'model2_n': n2, 'model3_n': n3}

# ══════════════════════════════════════════════════════════════════
# Output + N explanation
# ══════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(rows_out)
results_df.to_csv('results/two_part_results.csv', index=False)
print(f"\nresults/two_part_results.csv saved -- {len(results_df)} rows")

print("\n" + "=" * 90)
print("WHY N DIFFERS ACROSS MODELS")
print("=" * 90)
for hyp, spec in HYPS.items():
    n1, n2, n3 = n_summary[hyp]['model1_n'], n_summary[hyp]['model2_n'], n_summary[hyp]['model3_n']
    ext_iv, int_iv = spec['extensive_iv'], spec['intensive_iv']
    print(f"\n{hyp}:")
    if not spec['lag']:
        print(f"  Model 1 (extensive, full sample): N={n1} -- full panel, contemporaneous "
              f"{ext_iv}+log_revenue, no lag attrition")
        print(f"  Model 2 (intensive, disclosers only): N={n2} -- Model 1's sample FURTHER "
              f"restricted to {ext_iv}==1 rows only ({int_iv} is undefined, i.e. dropped by "
              f"dropna(), wherever there was no disclosure)")
    else:
        print(f"  Model 1 (extensive, full sample): N={n1} -- lag-1 {ext_iv}+log_revenue_lag1 "
              f"drops each firm's first year (no prior-year value to lag), minus Zalando 2025")
        print(f"  Model 2 (intensive, disclosers only): N={n2} -- Model 1's sample FURTHER "
              f"restricted to {ext_iv}==1 (i.e. the PRIOR year disclosed), since {int_iv} is "
              f"undefined whenever last year had no disclosure OR there's no valid prior year "
              f"to lag from at all -- this is a stricter filter than the unlagged Model 2, so N "
              f"is unchanged from before only if disclosure status happens to be stable firm-to-firm")
    print(f"  Model 3 (both, full sample): N={n3} -- same sample as Model 1; the filled "
          f"intensive-margin variable is 0 (not missing) for non-disclosers so no additional "
          f"rows are dropped for missingness on that variable, so N3 == N1")

print("\nDone.")
