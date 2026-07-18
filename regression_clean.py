import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Reuse the ACTUAL TOST implementation rather than reimplementing it.
# Verified before writing this: validity_check.py contains NO TOST code
# at all (only a signal-score histogram, a year-trend regression, and a
# randomization-inference permutation test). The real equivalence-test
# logic (tost_equivalence, EQUIVALENCE_BOUNDS) lives in
# retest_table_4_8.py -- imported directly below, not copied.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retest_table_4_8 import tost_equivalence, EQUIVALENCE_BOUNDS

# ── Load Data ──────────────────────────────────────────────────────
df = pd.read_csv('panel_dataset.csv')
df = df.sort_values(['firm', 'year']).reset_index(drop=True)
print(f"Loaded panel_dataset.csv: {len(df)} rows, "
      f"{df['firm'].nunique()} firms")

df['log_revenue'] = np.log(df['Revenue'].replace(0, np.nan))

# ── Lag construction (H2/H3 only) ───────────────────────────────────
# signal_lag1 mirrors run_lag_models.py exactly: the row for firm-year t
# carries the PRIOR year's (t-1) signal score. year_gap1 != 1 -> NaN
# guards against pairing non-adjacent years across any gap in a firm's
# panel (no firm actually has an interior gap in this dataset -- About
# You's panel just ends early at 2024 -- but the guard is kept for
# correctness/consistency with run_lag_models.py).
df['signal_lag1'] = df.groupby('firm')['mean_signal_score'].shift(1)
year_gap1 = df.groupby('firm')['year'].diff()
df.loc[year_gap1 != 1, 'signal_lag1'] = np.nan

# TEMPORAL ALIGNMENT -- verified against run_lag_models.py's existing row
# structure, not assumed: that script regresses the DV at year t on
# signal_lag1 = score at year t-1, i.e. "outcome at t+1 on signal at t"
# read from the signal's own year label -- the relationship this
# refactor specifies. log_revenue_lag1 below uses the IDENTICAL
# groupby('firm').shift(1) + year-gap guard as signal_lag1, so it lands
# on the same year t-1: firm size is measured in the same prior year as
# the lagged signal, one year before the DV's own year t. That is the
# "firm size measured in the year prior to the outcome window"
# alignment called for -- log_revenue_lag1 and signal_lag1 share the
# same t-1 reference point relative to the DV at t.
df['log_revenue_lag1'] = df.groupby('firm')['log_revenue'].shift(1)
df.loc[year_gap1 != 1, 'log_revenue_lag1'] = np.nan

df = df.set_index(['firm', 'year'])

# ── Sample construction: symmetric acquisition-event exclusion ─────
# Zalando's FY2025 accounts consolidate the About You acquisition,
# mechanically contaminating revenue growth and gross margin for that
# firm-year; this mirrors the existing treatment of About You itself,
# whose panel already ends in 2024 for the same class of event. Applies
# ONLY to the H2/H3 (accounting-based DV) primary sample -- H1 keeps the
# full N=69 sample, untouched.
zalando_2025_mask = ~((df.index.get_level_values('firm') == 'Zalando') &
                       (df.index.get_level_values('year') == 2025))

print("\n" + "=" * 70)
print("SAMPLE ATTRITION")
print("=" * 70)
print(f"Full panel: {len(df)} rows")
print(f"After excluding Zalando 2025 (H2/H3 primary sample only): "
      f"{int(zalando_2025_mask.sum())} rows")

results_rows = []

# ══════════════════════════════════════════════════════════════════
# H1 -- STOCK PRICE (unchanged, full sample, contemporaneous)
# This model is not modified from its original specification in any way.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("H1: Stock Price Movement (contemporaneous, unchanged, full sample)")
print("=" * 70)

h1_full = df[['Stock_Price_Movement_%', 'mean_signal_score', 'log_revenue']]
h1_model_df = h1_full.dropna()
print(f"N before per-model dropna: {len(h1_full)}, after: {len(h1_model_df)} "
      f"(dropped {len(h1_full) - len(h1_model_df)})")

h1_exog = sm.add_constant(h1_model_df[['mean_signal_score', 'log_revenue']])
h1_result = PanelOLS(
    h1_model_df['Stock_Price_Movement_%'], h1_exog,
    entity_effects=True, time_effects=True, drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

h1_beta = h1_result.params['mean_signal_score']
h1_se = h1_result.std_errors['mean_signal_score']
h1_n = int(h1_result.nobs)
h1_ci = h1_result.conf_int().loc['mean_signal_score']

print(f"H1: beta={h1_beta:.4f}, SE={h1_se:.4f}, N={h1_n}, "
      f"p={h1_result.pvalues['mean_signal_score']:.4f}")

results_rows.append({
    'Model': 'H1_Stock_Price', 'Type': 'Primary', 'DV': 'Stock_Price_Movement_%',
    'IV': 'mean_signal_score', 'Control': 'log_revenue',
    'Coefficient': round(h1_beta, 4), 'Std_Error': round(h1_se, 4),
    'T_stat': round(h1_result.tstats['mean_signal_score'], 4),
    'P_value': round(h1_result.pvalues['mean_signal_score'], 4),
    'CI_lower': round(h1_ci['lower'], 4), 'CI_upper': round(h1_ci['upper'], 4),
    'N_obs': h1_n, 'R2_within': round(h1_result.rsquared, 4),
})

# ── VERIFICATION CHECK -- must not proceed past this point if it fails ──
EXPECTED_H1_BETA, EXPECTED_H1_SE, EXPECTED_H1_N = -22.624, 35.090, 69
TOL = 0.001
if (abs(h1_beta - EXPECTED_H1_BETA) > TOL or abs(h1_se - EXPECTED_H1_SE) > TOL
        or h1_n != EXPECTED_H1_N):
    print("\n" + "!" * 70)
    print("H1 VERIFICATION FAILED -- STOPPING. NOT proceeding to H2/H3.")
    print(f"  Expected: beta={EXPECTED_H1_BETA}, SE={EXPECTED_H1_SE}, N={EXPECTED_H1_N}")
    print(f"  Got:      beta={h1_beta:.4f}, SE={h1_se:.4f}, N={h1_n}")
    print("!" * 70)
    sys.exit(1)
print(f"H1 verification PASSED — matches original baseline "
      f"(beta={EXPECTED_H1_BETA}, SE={EXPECTED_H1_SE}, N={EXPECTED_H1_N}) exactly.\n")

# ══════════════════════════════════════════════════════════════════
# H2 / H3 -- PRIMARY: lag-1 signal, lag-1 log revenue control,
# excl. Zalando 2025
# ══════════════════════════════════════════════════════════════════
PRIMARY_LAG_DVS = {
    'H2_Revenue_Growth': 'Revenue_Growth_%',
    'H3_Gross_Margin': 'Gross_Margin_%',
}
df_excl_zalando = df[zalando_2025_mask]


def fit_lag_model(data, dv, label):
    full = data[[dv, 'signal_lag1', 'log_revenue_lag1']]
    model_df = full.dropna()
    print(f"{label}: N before per-model dropna={len(full)}, after={len(model_df)} "
          f"(dropped {len(full) - len(model_df)})")
    exog = sm.add_constant(model_df[['signal_lag1', 'log_revenue_lag1']])
    result = PanelOLS(model_df[dv], exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    return result, model_df


print("\n" + "=" * 70)
print("H2 / H3 PRIMARY: lag-1 signal + lag-1 log revenue, excl. Zalando 2025")
print("=" * 70)

primary_results = {}
for label, dv in PRIMARY_LAG_DVS.items():
    result, model_df = fit_lag_model(df_excl_zalando, dv, f"{label} (primary)")
    n = len(model_df)
    beta = result.params['signal_lag1']
    se = result.std_errors['signal_lag1']
    ci = result.conf_int().loc['signal_lag1']
    print(f"  {label}: beta={beta:.4f}, SE={se:.4f}, N={n}, "
          f"p={result.pvalues['signal_lag1']:.4f}")
    primary_results[label] = {'result': result, 'dv': dv, 'n': n,
                               'beta': beta, 'se': se}
    results_rows.append({
        'Model': label, 'Type': 'Primary', 'DV': dv, 'IV': 'signal_lag1',
        'Control': 'log_revenue_lag1',
        'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
        'T_stat': round(result.tstats['signal_lag1'], 4),
        'P_value': round(result.pvalues['signal_lag1'], 4),
        'CI_lower': round(ci['lower'], 4), 'CI_upper': round(ci['upper'], 4),
        'N_obs': n, 'R2_within': round(result.rsquared, 4),
    })

# ══════════════════════════════════════════════════════════════════
# H1 -- LAG-1 (ADDED): same primary spec as H2/H3 above (lag-1 signal,
# lag-1 log revenue control, excl. Zalando 2025), applied to H1's DV.
# The dissertation reports this spec but it did not previously exist in
# this script -- H1 above is contemporaneous only. Kept as an ADDITION;
# the existing contemporaneous H1 block above is untouched, still
# labelled 'Primary', still gated by its own verification check.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("H1 -- LAG-1 (added): lag-1 signal + lag-1 log revenue, excl. Zalando 2025, "
      "same primary spec as H2/H3")
print("=" * 70)
h1_lag_result, h1_lag_model_df = fit_lag_model(
    df_excl_zalando, 'Stock_Price_Movement_%', "H1 (lag-1)")
h1_lag_n = len(h1_lag_model_df)
h1_lag_beta = h1_lag_result.params['signal_lag1']
h1_lag_se = h1_lag_result.std_errors['signal_lag1']
h1_lag_ci = h1_lag_result.conf_int().loc['signal_lag1']
print(f"  H1 (lag-1): beta={h1_lag_beta:.4f}, SE={h1_lag_se:.4f}, N={h1_lag_n}, "
      f"p={h1_lag_result.pvalues['signal_lag1']:.4f}")
results_rows.append({
    'Model': 'H1_Stock_Price', 'Type': 'Primary: lag-1 (matches H2/H3 primary spec)',
    'DV': 'Stock_Price_Movement_%', 'IV': 'signal_lag1', 'Control': 'log_revenue_lag1',
    'Coefficient': round(h1_lag_beta, 4), 'Std_Error': round(h1_lag_se, 4),
    'T_stat': round(h1_lag_result.tstats['signal_lag1'], 4),
    'P_value': round(h1_lag_result.pvalues['signal_lag1'], 4),
    'CI_lower': round(h1_lag_ci['lower'], 4), 'CI_upper': round(h1_lag_ci['upper'], 4),
    'N_obs': h1_lag_n, 'R2_within': round(h1_lag_result.rsquared, 4),
})

# ══════════════════════════════════════════════════════════════════
# ROBUSTNESS / SENSITIVITY (kept strictly separate from primary output
# above via the 'Type' column; same CSV, not a separate file)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ROBUSTNESS (a): H2/H3 with Zalando 2025 INCLUDED")
print("=" * 70)
for label, dv in PRIMARY_LAG_DVS.items():
    result, model_df = fit_lag_model(df, dv, f"{label} (Zalando 2025 included)")
    n = len(model_df)
    beta = result.params['signal_lag1']
    se = result.std_errors['signal_lag1']
    ci = result.conf_int().loc['signal_lag1']
    print(f"  {label}: beta={beta:.4f}, SE={se:.4f}, N={n}, "
          f"p={result.pvalues['signal_lag1']:.4f}")
    results_rows.append({
        'Model': label, 'Type': 'Robustness: Zalando 2025 included', 'DV': dv,
        'IV': 'signal_lag1', 'Control': 'log_revenue_lag1',
        'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
        'T_stat': round(result.tstats['signal_lag1'], 4),
        'P_value': round(result.pvalues['signal_lag1'], 4),
        'CI_lower': round(ci['lower'], 4), 'CI_upper': round(ci['upper'], 4),
        'N_obs': n, 'R2_within': round(result.rsquared, 4),
    })

print("\n" + "=" * 70)
print("ROBUSTNESS (b): DocMorris/Boohoo exclusions on top of the new "
      "primary baseline (lag-1 control, excl. Zalando 2025)")
print("=" * 70)
docmorris_checks = {
    "Excl. DocMorris 2023": lambda d: d[~((d.index.get_level_values('firm') == 'DocMorris') &
                                           (d.index.get_level_values('year') == 2023))],
    # ADDED: excludes DocMorris for ALL five years, not just the 2023
    # anomaly -- distinct from the single-year check above. The
    # dissertation's robustness section reports this full-firm exclusion.
    "Excl. DocMorris entirely": lambda d: d[d.index.get_level_values('firm') != 'DocMorris'],
    "Excl. Boohoo": lambda d: d[d.index.get_level_values('firm') != 'Boohoo'],
    "Excl. DocMorris 2023 + Boohoo": lambda d: d[
        ~((d.index.get_level_values('firm') == 'DocMorris') &
          (d.index.get_level_values('year') == 2023)) &
        (d.index.get_level_values('firm') != 'Boohoo')],
}
for check_name, fn in docmorris_checks.items():
    sub = fn(df_excl_zalando)
    print(f"\n-- {check_name} (on new baseline, N={len(sub)}) --")
    for label, dv in PRIMARY_LAG_DVS.items():
        full = sub[[dv, 'signal_lag1', 'log_revenue_lag1']]
        model_df = full.dropna()
        if len(model_df) < 15:
            print(f"  {label}: insufficient observations ({len(model_df)})")
            continue
        exog = sm.add_constant(model_df[['signal_lag1', 'log_revenue_lag1']])
        result = PanelOLS(model_df[dv], exog, entity_effects=True, time_effects=True,
                           drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
        n = len(model_df)
        beta = result.params['signal_lag1']
        se = result.std_errors['signal_lag1']
        ci = result.conf_int().loc['signal_lag1']
        print(f"  {label}: beta={beta:.4f}, SE={se:.4f}, N={n}, "
              f"p={result.pvalues['signal_lag1']:.4f}")
        results_rows.append({
            'Model': label, 'Type': f'Robustness: {check_name}', 'DV': dv,
            'IV': 'signal_lag1', 'Control': 'log_revenue_lag1',
            'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
            'T_stat': round(result.tstats['signal_lag1'], 4),
            'P_value': round(result.pvalues['signal_lag1'], 4),
            'CI_lower': round(ci['lower'], 4), 'CI_upper': round(ci['upper'], 4),
            'N_obs': n, 'R2_within': round(result.rsquared, 4),
        })

print("\n" + "=" * 70)
print("ROBUSTNESS: H1 DocMorris/Boohoo exclusions -- same H1 primary spec "
      "(contemporaneous, mean_signal_score, log_revenue control), full "
      "sample basis -- no Zalando 2025 exclusion, since that exclusion "
      "applies only to H2/H3")
print("=" * 70)
h1_docmorris_checks = {
    "Excl. DocMorris 2023": lambda d: d[~((d.index.get_level_values('firm') == 'DocMorris') &
                                           (d.index.get_level_values('year') == 2023))],
    "Excl. Boohoo": lambda d: d[d.index.get_level_values('firm') != 'Boohoo'],
    "Excl. DocMorris 2023 + Boohoo": lambda d: d[
        ~((d.index.get_level_values('firm') == 'DocMorris') &
          (d.index.get_level_values('year') == 2023)) &
        (d.index.get_level_values('firm') != 'Boohoo')],
}
for check_name, fn in h1_docmorris_checks.items():
    sub = fn(df)
    full = sub[['Stock_Price_Movement_%', 'mean_signal_score', 'log_revenue']]
    model_df = full.dropna()
    exog = sm.add_constant(model_df[['mean_signal_score', 'log_revenue']])
    result = PanelOLS(
        model_df['Stock_Price_Movement_%'], exog,
        entity_effects=True, time_effects=True, drop_absorbed=True
    ).fit(cov_type='clustered', cluster_entity=True)
    n = len(model_df)
    beta = result.params['mean_signal_score']
    se = result.std_errors['mean_signal_score']
    ci = result.conf_int().loc['mean_signal_score']
    print(f"  H1 ({check_name}): beta={beta:.4f}, SE={se:.4f}, N={n}, "
          f"p={result.pvalues['mean_signal_score']:.4f}")
    results_rows.append({
        'Model': 'H1_Stock_Price', 'Type': f'Robustness: {check_name}',
        'DV': 'Stock_Price_Movement_%', 'IV': 'mean_signal_score',
        'Control': 'log_revenue',
        'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
        'T_stat': round(result.tstats['mean_signal_score'], 4),
        'P_value': round(result.pvalues['mean_signal_score'], 4),
        'CI_lower': round(ci['lower'], 4), 'CI_upper': round(ci['upper'], 4),
        'N_obs': n, 'R2_within': round(result.rsquared, 4),
    })

print("\n" + "=" * 70)
print("ROBUSTNESS (c): H1 with log_revenue_lag1 as a symmetry check on "
      "the control choice")
print("=" * 70)
h1_sym_full = df[['Stock_Price_Movement_%', 'mean_signal_score', 'log_revenue_lag1']]
h1_sym_df = h1_sym_full.dropna()
print(f"N before dropna: {len(h1_sym_full)}, after: {len(h1_sym_df)}")
h1_sym_exog = sm.add_constant(h1_sym_df[['mean_signal_score', 'log_revenue_lag1']])
h1_sym_result = PanelOLS(
    h1_sym_df['Stock_Price_Movement_%'], h1_sym_exog,
    entity_effects=True, time_effects=True, drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)
h1_sym_beta = h1_sym_result.params['mean_signal_score']
h1_sym_se = h1_sym_result.std_errors['mean_signal_score']
h1_sym_ci = h1_sym_result.conf_int().loc['mean_signal_score']
print(f"  H1 (log_revenue_lag1 control): beta={h1_sym_beta:.4f}, "
      f"SE={h1_sym_se:.4f}, N={len(h1_sym_df)}")
results_rows.append({
    'Model': 'H1_Stock_Price', 'Type': 'Robustness: log_revenue_lag1 symmetry check',
    'DV': 'Stock_Price_Movement_%', 'IV': 'mean_signal_score',
    'Control': 'log_revenue_lag1',
    'Coefficient': round(h1_sym_beta, 4), 'Std_Error': round(h1_sym_se, 4),
    'T_stat': round(h1_sym_result.tstats['mean_signal_score'], 4),
    'P_value': round(h1_sym_result.pvalues['mean_signal_score'], 4),
    'CI_lower': round(h1_sym_ci['lower'], 4), 'CI_upper': round(h1_sym_ci['upper'], 4),
    'N_obs': len(h1_sym_df), 'R2_within': round(h1_sym_result.rsquared, 4),
})

# ══════════════════════════════════════════════════════════════════
# Write canonical output -- one file, overwritten, not a variant
# ══════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results_rows)
results_df.to_csv('regression_results.csv', index=False)
print("\n" + "=" * 70)
print("regression_results.csv saved (overwritten) -- "
      f"{len(results_df)} rows (primary + all robustness runs)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# HAUSMAN TESTS -- H2/H3 under the new primary specification
# Values only. Not interpreted here as validating/invalidating fixed
# effects -- the FE justification in the dissertation rests on
# conceptual grounds (Section 3.7), not on this test.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("HAUSMAN TESTS (H2/H3, new primary spec: lag-1 signal + lag-1 "
      "log revenue, excl. Zalando 2025)")
print("=" * 70)


def hausman(fe_res, re_res, common):
    b_diff = fe_res.params[common] - re_res.params[common]
    cov_diff = fe_res.cov.loc[common, common] - re_res.cov.loc[common, common]
    stat = float(b_diff.values @ np.linalg.inv(cov_diff.values) @ b_diff.values)
    dof = len(common)
    pval = 1 - scipy_stats.chi2.cdf(stat, dof)
    return stat, dof, pval


hausman_results = {}
for label, dv in PRIMARY_LAG_DVS.items():
    m = df_excl_zalando[[dv, 'signal_lag1', 'log_revenue_lag1']].dropna().copy()
    years = m.index.get_level_values('year')
    year_dum = pd.get_dummies(years, prefix='yr', drop_first=True).set_index(m.index).astype(float)
    x_fe = sm.add_constant(m[['signal_lag1', 'log_revenue_lag1']])
    x_re = sm.add_constant(pd.concat([m[['signal_lag1', 'log_revenue_lag1']], year_dum], axis=1))
    fe = PanelOLS(m[dv], x_fe, entity_effects=True, time_effects=True).fit(cov_type='unadjusted')
    re = RandomEffects(m[dv], x_re).fit(cov_type='unadjusted')
    stat, dof, pval = hausman(fe, re, ['signal_lag1', 'log_revenue_lag1'])
    hausman_results[label] = (stat, dof, pval)
    print(f"  {label}: chi2({dof}) = {stat:.2f}, p = {pval:.4f}")

# ══════════════════════════════════════════════════════════════════
# H3 TOST EQUIVALENCE -- reusing retest_table_4_8.py directly
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("H3 TOST EQUIVALENCE (new primary spec)")
print("=" * 70)

h3 = primary_results['H3_Gross_Margin']
h3_bound = EQUIVALENCE_BOUNDS["H3 (lag-1, primary)"]

# Calling tost_equivalence(beta, se, n, bound, alpha=0.05) imported from
# retest_table_4_8.py, with the NEW H3 lag-1 (excl. Zalando 2025)
# beta/SE/N computed above. The bound itself is reused UNCHANGED from
# retest_table_4_8.py's EQUIVALENCE_BOUNDS -- 0.2 x the population SD of
# Gross_Margin_% across the full N=69 panel (17.31), not recomputed on
# this subsample, so the result stays comparable to the original TOST
# run. It resolves numerically to 0.2 x 17.31 = 3.462 percentage points
# of gross margin.
print(f"Calling tost_equivalence(beta={h3['beta']:.4f}, se={h3['se']:.4f}, "
      f"n={h3['n']}, bound={h3_bound:.4f})")
print(f"Bound resolves to {h3_bound:.4f} pp of gross margin "
      f"(= 0.2 x 17.31, unchanged from the original run's full-panel SD).")

p_lo, p_hi, equivalent = tost_equivalence(h3['beta'], h3['se'], h3['n'], h3_bound)
print(f"H3 TOST under new spec: p_lower={p_lo:.4f}, p_upper={p_hi:.4f}, "
      f"equivalent={equivalent}")

print("\nDone.")
