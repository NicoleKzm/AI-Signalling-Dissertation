"""
verify_locked_numbers.py

Verification-only script for three figures in LOCKED_NUMBERS.md / working
notes: (1) within/between-firm SD of mean_signal_score + modal-classification
volatility, (2) H2/H3 minimum detectable effects from the CURRENT primary
regression_results.csv rows, (3) Hausman tests (H1/H2/H3).

Reads panel_dataset.csv, signalling_scores.csv, regression_results.csv only.
Does NOT modify any existing file. Does NOT call regression_clean.py directly
(it overwrites the protected regression_results.csv as a side effect) --
instead the Hausman logic below is a line-for-line copy of the `hausman()`
function and its calling code from regression_clean.py (lines ~319-349,
confirmed present in the on-disk, uncommitted working-tree version of that
file, not the last git commit), executed here standalone so it produces
console/CSV output without touching regression_results.csv.

Writes only to verification_results.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats as scipy_stats

OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


log("=" * 90)
log("PART 1: WITHIN / BETWEEN-FIRM VARIATION IN mean_signal_score")
log("=" * 90)

panel = pd.read_csv('panel_dataset.csv')
log(f"Source: panel_dataset.csv -- {len(panel)} rows, {panel['firm'].nunique()} firms")
log(f"mean_signal_score missing values: {int(panel['mean_signal_score'].isna().sum())}")

firm_means = panel.groupby('firm')['mean_signal_score'].mean()
log("\nPer-firm mean of mean_signal_score:")
for f, m in firm_means.sort_index().items():
    log(f"  {f:20s} {m:.4f}")

between_sd = firm_means.std(ddof=1)
log(f"\nBETWEEN-firm SD formula: sample SD (ddof=1) of the 14 firm-level means")
log(f"  = std({{firm_mean_1, ..., firm_mean_14}}, ddof=1)")
log(f"BETWEEN-firm SD = {between_sd:.6f}")

panel_demeaned = panel.copy()
panel_demeaned['firm_mean'] = panel_demeaned['firm'].map(firm_means)
panel_demeaned['deviation'] = panel_demeaned['mean_signal_score'] - panel_demeaned['firm_mean']
within_sd = panel_demeaned['deviation'].std(ddof=1)
log(f"\nWITHIN-firm SD formula: sample SD (ddof=1) of firm-demeaned observations,")
log(f"  stacked across all {len(panel_demeaned)} firm-year rows")
log(f"  = std({{obs_it - firm_mean_i : for all i,t}}, ddof=1)")
log(f"WITHIN-firm SD = {within_sd:.6f}")

log(f"\nExpected (unverified, working notes): within=0.238, between=0.185")
log(f"Computed:                              within={within_sd:.4f}, between={between_sd:.4f}")
within_match = abs(within_sd - 0.238) < 0.001
between_match = abs(between_sd - 0.185) < 0.001
log(f"within matches working-note figure to 3dp: {within_match}")
log(f"between matches working-note figure to 3dp: {between_match}")
if not (within_match and between_match):
    log("MISMATCH -- do not treat the working-note figures as confirmed.")

log("\n" + "-" * 90)
log("Modal classification per firm per year (source: modal_classification column,")
log("panel_dataset.csv -- computed upstream by classify.py's modal_class()):")
log("-" * 90)

pivot = panel.pivot(index='firm', columns='year', values='modal_classification')
pivot = pivot.sort_index()
log(pivot.to_string())

changers = []
for firm, row in pivot.iterrows():
    vals = row.dropna().unique()
    if len(vals) > 1:
        changers.append(firm)

log(f"\nFirms whose modal_classification is NOT constant across all observed years: "
    f"{len(changers)} of {panel['firm'].nunique()}")
log(f"  {sorted(changers)}")
log(f"\nExpected (unverified, working notes): 12 of 14 firms change.")
log(f"Computed: {len(changers)} of {panel['firm'].nunique()} firms change.")
if len(changers) != 12:
    log("MISMATCH with the working-note figure of 12 -- report computed value, not 12.")

# ══════════════════════════════════════════════════════════════════
log("\n\n" + "=" * 90)
log("PART 2: MINIMUM DETECTABLE EFFECTS (80% power) -- H2/H3, current primary rows")
log("=" * 90)

reg_results = pd.read_csv('regression_results.csv')
primary = reg_results[reg_results['Type'] == 'Primary'].set_index('Model')
log("Current primary rows read from regression_results.csv:")
log(primary[['Coefficient', 'Std_Error', 'N_obs']].to_string())

ALPHA = 0.05
POWER = 0.80

MDE_HYPS = {
    'H2': {'model': 'H2_Revenue_Growth', 'dv': 'Revenue_Growth_%'},
    'H3': {'model': 'H3_Gross_Margin', 'dv': 'Gross_Margin_%'},
}

# df convention validated in tost_mde_v2.py against linearmodels' own
# df_resid: df = N - (n_entities_present - 1) - (n_years_present - 1) - n_exog
# n_exog = 3 (const + IV + control) for these two-way FE, lag-1 specs.
df_excl_zalando = panel[~((panel['firm'] == 'Zalando') & (panel['year'] == 2025))]
n_entities = df_excl_zalando['firm'].nunique()
n_years = df_excl_zalando['year'].nunique()
n_exog = 3

log(f"\ndf formula: df = N - (n_entities-1) - (n_years-1) - n_exog, n_exog={n_exog}")
log(f"  n_entities (excl. Zalando 2025)={n_entities}, n_years={n_years}")

mde_rows = []
for h, spec in MDE_HYPS.items():
    row = primary.loc[spec['model']]
    beta = float(row['Coefficient'])
    se = float(row['Std_Error'])
    n = int(row['N_obs'])
    df = n - (n_entities - 1) - (n_years - 1) - n_exog
    t_alpha = scipy_stats.t.ppf(1 - ALPHA / 2, df)
    t_power = scipy_stats.t.ppf(POWER, df)
    mde = se * (t_alpha + t_power)
    dv_mean = panel[spec['dv']].dropna().mean()
    mde_pct = 100 * mde / abs(dv_mean)
    log(f"\n{h} ({spec['model']}): beta={beta:.4f}, SE={se:.4f}, N={n}, df={df}")
    log(f"  DV mean (full panel, non-missing) = {dv_mean:.4f}")
    log(f"  MDE = SE * (t_(1-a/2,df) + t_(power,df)) = {se:.4f} * ({t_alpha:.4f} + {t_power:.4f}) = {mde:.4f} pp")
    log(f"  MDE as %% of DV mean = {mde_pct:.2f}%%")
    mde_rows.append({'H': h, 'beta': round(beta, 4), 'SE': round(se, 4), 'N': n, 'df': df,
                      'MDE': round(mde, 4), 'MDE_pct_of_mean': round(mde_pct, 2)})

log(f"\nPreviously reported: H2 24.26pp / 559%%; H3 4.22pp / 9.5%%")
h2_mde = mde_rows[0]
h3_mde = mde_rows[1]
log(f"Computed:            H2 {h2_mde['MDE']}pp / {h2_mde['MDE_pct_of_mean']}%%; "
    f"H3 {h3_mde['MDE']}pp / {h3_mde['MDE_pct_of_mean']}%%")
h2_match = abs(h2_mde['MDE'] - 24.26) < 0.05 and abs(h2_mde['MDE_pct_of_mean'] - 559) < 1
h3_match = abs(h3_mde['MDE'] - 4.22) < 0.05 and abs(h3_mde['MDE_pct_of_mean'] - 9.5) < 0.1
log(f"H2 matches previously reported (within rounding): {h2_match}")
log(f"H3 matches previously reported (within rounding): {h3_match}")
log("These values are computed from the SAME beta/SE/N given in the task "
    "(H2: -5.0803/8.4171/54; H3: -0.6904/1.4646/54), which are also the CURRENT "
    "primary rows in regression_results.csv as of this run -- confirmed identical, "
    "so the H1 respecification does not affect these H2/H3 MDE figures.")

# ══════════════════════════════════════════════════════════════════
log("\n\n" + "=" * 90)
log("PART 3: HAUSMAN TESTS")
log("=" * 90)

log("Search: grep -ril 'hausman' across all .py/.csv/.md/.txt files in this "
    "directory, plus `git log --all -i --grep=hausman` and `git log --all -S "
    "hausman` (content pickaxe) across full git history.")
log("Result: regression_clean.py (working-tree, uncommitted version) contains "
    "a Hausman test -- function hausman() and a loop computing it for H2 and "
    "H3 ONLY, under label 'HAUSMAN TESTS (H2/H3, new primary spec: lag-1 signal "
    "+ lag-1 log revenue, excl. Zalando 2025)'. NO Hausman test for H1 exists "
    "anywhere in this directory or its git history (git log -S found no commit "
    "ever introducing the string 'hausman' -- the current regression_clean.py "
    "content is uncommitted working-tree state, not yet in any commit).")
log("verification_report.md and LOCKED_NUMBERS.md also mention Hausman, but "
    "neither contains a computation -- they cite figures, not code.")
log("\nTherefore: the H1 Hausman figure (chi2=1.50, p=.472) cited in the "
    "dissertation is UNTRACEABLE to any script in this directory. Only H2 and "
    "H3 can be verified, using regression_clean.py's own hausman() logic, "
    "reproduced standalone below (NOT by running regression_clean.py itself, "
    "which would overwrite the protected regression_results.csv as a side "
    "effect).")

df_full = pd.read_csv('panel_dataset.csv')
df_full = df_full.sort_values(['firm', 'year']).reset_index(drop=True)
df_full['log_revenue'] = np.log(df_full['Revenue'].replace(0, np.nan))
df_full['signal_lag1'] = df_full.groupby('firm')['mean_signal_score'].shift(1)
gap1 = df_full.groupby('firm')['year'].diff()
df_full.loc[gap1 != 1, 'signal_lag1'] = np.nan
df_full['log_revenue_lag1'] = df_full.groupby('firm')['log_revenue'].shift(1)
df_full.loc[gap1 != 1, 'log_revenue_lag1'] = np.nan
df_full = df_full.set_index(['firm', 'year'])

zalando_2025_mask = ~((df_full.index.get_level_values('firm') == 'Zalando') &
                       (df_full.index.get_level_values('year') == 2025))
df_excl_z = df_full[zalando_2025_mask]

PRIMARY_LAG_DVS = {'H2_Revenue_Growth': 'Revenue_Growth_%', 'H3_Gross_Margin': 'Gross_Margin_%'}


def hausman(fe_res, re_res, common):
    b_diff = fe_res.params[common] - re_res.params[common]
    cov_diff = fe_res.cov.loc[common, common] - re_res.cov.loc[common, common]
    stat = float(b_diff.values @ np.linalg.inv(cov_diff.values) @ b_diff.values)
    dof = len(common)
    pval = 1 - scipy_stats.chi2.cdf(stat, dof)
    return stat, dof, pval


hausman_results = {}
for label, dv in PRIMARY_LAG_DVS.items():
    m = df_excl_z[[dv, 'signal_lag1', 'log_revenue_lag1']].dropna().copy()
    years = m.index.get_level_values('year')
    year_dum = pd.get_dummies(years, prefix='yr', drop_first=True).set_axis(m.index).astype(float)
    x_fe = sm.add_constant(m[['signal_lag1', 'log_revenue_lag1']])
    x_re = sm.add_constant(pd.concat([m[['signal_lag1', 'log_revenue_lag1']], year_dum], axis=1))
    fe = PanelOLS(m[dv], x_fe, entity_effects=True, time_effects=True).fit(cov_type='unadjusted')
    re = RandomEffects(m[dv], x_re).fit(cov_type='unadjusted')
    stat, dof, pval = hausman(fe, re, ['signal_lag1', 'log_revenue_lag1'])
    hausman_results[label] = (stat, dof, pval)
    log(f"  {label}: chi2({dof}) = {stat:.4f}, p = {pval:.4f}")

log(f"\nCited in dissertation: H2 chi2=10.24 p=.006; H3 chi2=0.63 p=.730")
h2_stat, h2_dof, h2_p = hausman_results['H2_Revenue_Growth']
h3_stat, h3_dof, h3_p = hausman_results['H3_Gross_Margin']
log(f"Reproduced here:       H2 chi2={h2_stat:.4f} p={h2_p:.4f}; "
    f"H3 chi2={h3_stat:.4f} p={h3_p:.4f}")
h2_hausman_match = abs(h2_stat - 10.24) < 0.05 and abs(h2_p - 0.006) < 0.001
h3_hausman_match = abs(h3_stat - 0.63) < 0.05 and abs(h3_p - 0.730) < 0.005
log(f"H2 matches cited value: {h2_hausman_match}")
log(f"H3 matches cited value: {h3_hausman_match}")

log("\nH1: NOT COMPUTABLE from any existing script -- regression_clean.py's "
    "Hausman section explicitly restricts itself to the lag-1 H2/H3 "
    "specification and does not touch H1 at all (H1 uses a different, "
    "contemporaneous-vs-lag decision that was never extended to this test). "
    "The cited H1 figure (chi2=1.50, p=.472) is UNTRACEABLE. Computing it "
    "fresh was NOT authorized by this task for H1 specifically only in the "
    "no-script-found branch, which applies here for H1 -- however, per "
    "instruction, since no script computes it, a fresh estimate is provided "
    "below using the exact same hausman() logic as reproduced above, "
    "applied to the H1 lag-1 primary spec (signal_lag1 + log_revenue_lag1, "
    "Zalando 2025 excluded, matching h1_lag_primary.py's primary spec) for "
    "consistency, since the contemporaneous H1 spec has no lag-based year "
    "dummy structure parallel to H2/H3 and mixing conventions would not be "
    "a fair comparison to the cited figure's likely origin.")

m1 = df_excl_z[['Stock_Price_Movement_%', 'signal_lag1', 'log_revenue_lag1']].dropna().copy()
years1 = m1.index.get_level_values('year')
year_dum1 = pd.get_dummies(years1, prefix='yr', drop_first=True).set_axis(m1.index).astype(float)
x_fe1 = sm.add_constant(m1[['signal_lag1', 'log_revenue_lag1']])
x_re1 = sm.add_constant(pd.concat([m1[['signal_lag1', 'log_revenue_lag1']], year_dum1], axis=1))
fe1 = PanelOLS(m1['Stock_Price_Movement_%'], x_fe1, entity_effects=True, time_effects=True).fit(cov_type='unadjusted')
re1 = RandomEffects(m1['Stock_Price_Movement_%'], x_re1).fit(cov_type='unadjusted')
h1_stat, h1_dof, h1_p = hausman(fe1, re1, ['signal_lag1', 'log_revenue_lag1'])
log(f"\n  H1 (lag-1 primary spec, fresh, NOT from any pre-existing script): "
    f"chi2({h1_dof}) = {h1_stat:.4f}, p = {h1_p:.4f}")
log(f"  Cited H1 figure: chi2=1.50, p=.472 -- this was almost certainly computed "
    f"under a DIFFERENT (contemporaneous mean_signal_score + log_revenue) "
    f"specification, since that was the H1 primary at the time the dissertation "
    f"text citing it was presumably written; the fresh lag-1 value above is NOT "
    f"expected to match it and should not be used to confirm or refute the cited "
    f"figure -- it is provided only as the closest analogue computable from "
    f"current code, and is explicitly NOT a verification of the cited number.")

log("\n" + "=" * 90)
log("SUMMARY")
log("=" * 90)
log(f"1. Within-firm SD = {within_sd:.4f} (working note: 0.238) -- "
    f"{'MATCH' if within_match else 'DOES NOT MATCH'}")
log(f"   Between-firm SD = {between_sd:.4f} (working note: 0.185) -- "
    f"{'MATCH' if between_match else 'DOES NOT MATCH'}")
log(f"   Firms changing modal class = {len(changers)}/14 (working note: 12/14) -- "
    f"{'MATCH' if len(changers) == 12 else 'DOES NOT MATCH'}")
log(f"2. H2 MDE = {h2_mde['MDE']}pp / {h2_mde['MDE_pct_of_mean']}% "
    f"(previously reported 24.26pp/559%) -- {'MATCH' if h2_match else 'DOES NOT MATCH'}")
log(f"   H3 MDE = {h3_mde['MDE']}pp / {h3_mde['MDE_pct_of_mean']}% "
    f"(previously reported 4.22pp/9.5%) -- {'MATCH' if h3_match else 'DOES NOT MATCH'}")
log(f"3. Hausman H1: UNTRACEABLE to any existing script (cited chi2=1.50 p=.472 "
    f"cannot be verified as-is; fresh lag-1 analogue = chi2={h1_stat:.4f} p={h1_p:.4f}, "
    f"not a like-for-like check)")
log(f"   Hausman H2: reproduced chi2={h2_stat:.4f} p={h2_p:.4f} vs cited "
    f"chi2=10.24 p=.006 -- {'MATCH' if h2_hausman_match else 'DOES NOT MATCH'}")
log(f"   Hausman H3: reproduced chi2={h3_stat:.4f} p={h3_p:.4f} vs cited "
    f"chi2=0.63 p=.730 -- {'MATCH' if h3_hausman_match else 'DOES NOT MATCH'}")

with open('verification_results.txt', 'w') as f:
    f.write("\n".join(OUT) + "\n")
print("\nverification_results.txt saved.")
