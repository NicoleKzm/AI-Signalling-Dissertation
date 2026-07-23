"""
h1_lag_primary.py

Supervisor directive: the contemporaneous H1 specification cannot be the
primary test -- a report published after fiscal year-end cannot explain
returns that occurred before the report was public. Contemporaneous H1
is relabelled DESCRIPTIVE/SUPPLEMENTARY; lag-1 becomes PRIMARY, matching
H2/H3's existing temporal logic and symmetric Zalando-2025 treatment.

Mirrors regression_clean.py's exact estimator throughout: PanelOLS,
entity+time FE, firm-clustered SE, Revenue column (not Revenue_EUR).

Does not modify regression_clean.py, regression_results.csv, classify.py,
signalling_scores.csv, or panel_dataset.csv. Reads panel_dataset.csv only.
Writes only to h1_lag_primary_results.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats

ALPHA = 0.05
POWER = 0.80
H1_BOUND = 10.512  # 0.2 x SD(Stock_Price_Movement_%) = 0.2 x 52.560, unchanged

# ══════════════════════════════════════════════════════════════════
# Data construction -- mirrors regression_clean.py exactly
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv('data/panel_dataset.csv')
df = df.sort_values(['firm', 'year']).reset_index(drop=True)
print(f"Loaded data/panel_dataset.csv: {len(df)} rows, {df['firm'].nunique()} firms")

df['log_revenue'] = np.log(df['Revenue'].replace(0, np.nan))
df['signal_lag1'] = df.groupby('firm')['mean_signal_score'].shift(1)
year_gap1 = df.groupby('firm')['year'].diff()
df.loc[year_gap1 != 1, 'signal_lag1'] = np.nan
df['log_revenue_lag1'] = df.groupby('firm')['log_revenue'].shift(1)
df.loc[year_gap1 != 1, 'log_revenue_lag1'] = np.nan

df = df.set_index(['firm', 'year'])
zalando_2025_mask = ~((df.index.get_level_values('firm') == 'Zalando') &
                       (df.index.get_level_values('year') == 2025))
df_excl_zalando = df[zalando_2025_mask]

DV = 'Stock_Price_Movement_%'


def fit(data, dv, iv, control):
    model_df = data[[dv, iv, control]].dropna()
    n = len(model_df)
    exog = sm.add_constant(model_df[[iv, control]])
    result = PanelOLS(model_df[dv], exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    return result, model_df, n


rows_out = []


def record(label, result, iv, n, extra=None):
    beta = result.params[iv]
    se = result.std_errors[iv]
    t = result.tstats[iv]
    p = result.pvalues[iv]
    ci = result.conf_int().loc[iv]
    r2 = result.rsquared
    dof = result.df_resid
    print(f"  {label}: beta={beta:.4f}, SE={se:.4f}, t={t:.4f}, p={p:.4f}, "
          f"95% CI=[{ci['lower']:.4f}, {ci['upper']:.4f}], N={n}, R2_within={r2:.4f}, df_resid={dof}")
    row = {'Model': label, 'IV': iv, 'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
           'T_stat': round(t, 4), 'P_value': round(p, 4), 'CI_lower': round(ci['lower'], 4),
           'CI_upper': round(ci['upper'], 4), 'N': n, 'R2_within': round(r2, 4), 'df_resid': dof}
    if extra:
        row.update(extra)
    rows_out.append(row)
    return {'beta': beta, 'se': se, 't': t, 'p': p, 'ci': ci, 'n': n, 'r2': r2, 'df_resid': dof}


# ══════════════════════════════════════════════════════════════════
# 1. NEW PRIMARY: H1 lag-1, log_revenue_lag1 control, Zalando 2025 excluded
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("1. NEW PRIMARY: H1 lag-1 (signal_lag1 + log_revenue_lag1, excl. Zalando 2025)")
print("=" * 90)
result_new_primary, model_df_new_primary, n_new_primary = fit(
    df_excl_zalando, DV, 'signal_lag1', 'log_revenue_lag1')
new_primary = record('H1_NewPrimary_Lag1_LagControl', result_new_primary, 'signal_lag1', n_new_primary)

# ══════════════════════════════════════════════════════════════════
# 2. Same IV (signal_lag1), contemporaneous log_revenue control instead
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("2. H1 lag-1 signal, CONTEMPORANEOUS log_revenue control (does control timing matter?)")
print("=" * 90)
result_contctrl, model_df_contctrl, n_contctrl = fit(
    df_excl_zalando, DV, 'signal_lag1', 'log_revenue')
contctrl = record('H1_Lag1Signal_ContempControl', result_contctrl, 'signal_lag1', n_contctrl)

# ══════════════════════════════════════════════════════════════════
# 3. SUPPLEMENTARY: original contemporaneous H1 (full N=69, unchanged)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("3. SUPPLEMENTARY (formerly primary): H1 contemporaneous, full N=69")
print("=" * 90)
result_supp, model_df_supp, n_supp = fit(df, DV, 'mean_signal_score', 'log_revenue')
supplementary = record('H1_Supplementary_Contemporaneous', result_supp, 'mean_signal_score', n_supp)

# ══════════════════════════════════════════════════════════════════
# 4. MDE at 80% power under the NEW primary, df=df_resid
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("4. Minimum detectable effect (80% power), NEW primary (lag-1) specification")
print("=" * 90)
se_new = new_primary['se']
dof_new = new_primary['df_resid']
t_alpha = stats.t.ppf(1 - ALPHA / 2, dof_new)
t_power = stats.t.ppf(POWER, dof_new)
mde = se_new * (t_alpha + t_power)
dv_mean_full = df[DV].dropna().mean()
mde_pct_of_mean = 100 * mde / abs(dv_mean_full)
print(f"SE={se_new:.4f}, df_resid={dof_new}, t_alpha={t_alpha:.4f}, t_power={t_power:.4f}")
print(f"MDE = {mde:.4f} percentage points")
print(f"Full-panel mean of {DV}: {dv_mean_full:.4f}")
print(f"MDE as % of DV mean: {mde_pct_of_mean:.2f}%")

# ══════════════════════════════════════════════════════════════════
# 5. H1 TOST under the NEW primary, bound=+/-10.512, df=df_resid
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("5. H1 TOST equivalence, NEW primary (lag-1) specification, bound=+/-10.512")
print("=" * 90)
beta_new = new_primary['beta']
t_lower = (beta_new - (-H1_BOUND)) / se_new
t_upper = (H1_BOUND - beta_new) / se_new
p_lower = 1 - stats.t.cdf(t_lower, dof_new)
p_upper = 1 - stats.t.cdf(t_upper, dof_new)
equivalent = (p_lower < ALPHA) and (p_upper < ALPHA)

crit_90 = stats.t.ppf(1 - ALPHA, dof_new)
ci90_lower = beta_new - crit_90 * se_new
ci90_upper = beta_new + crit_90 * se_new

print(f"beta={beta_new:.4f}, SE={se_new:.4f}, df_resid={dof_new}, bound=+/-{H1_BOUND}")
print(f"t_lower={t_lower:.4f}, t_upper={t_upper:.4f}")
print(f"p_lower={p_lower:.4f}, p_upper={p_upper:.4f}")
print(f"90% CI = [{ci90_lower:.4f}, {ci90_upper:.4f}]")
print(f"Equivalence holds: {equivalent}")

rows_out.append({'Model': 'H1_NewPrimary_TOST', 'IV': 'signal_lag1',
                  'Coefficient': round(beta_new, 4), 'Std_Error': round(se_new, 4),
                  'N': n_new_primary, 'df_resid': dof_new, 'bound': H1_BOUND,
                  't_lower': round(t_lower, 4), 't_upper': round(t_upper, 4),
                  'p_lower': round(p_lower, 4), 'p_upper': round(p_upper, 4),
                  'CI90_lower': round(ci90_lower, 4), 'CI90_upper': round(ci90_upper, 4),
                  'Equivalent': equivalent})

# ══════════════════════════════════════════════════════════════════
# 6. Leave-one-firm-out on the NEW H1 primary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("6. Leave-one-firm-out, NEW H1 primary (lag-1, excl. Zalando 2025)")
print("=" * 90)

all_firms = sorted(df_excl_zalando.index.get_level_values('firm').unique())
loo_results = []
for firm in all_firms:
    sub = df_excl_zalando[df_excl_zalando.index.get_level_values('firm') != firm]
    result_loo, model_df_loo, n_loo = fit(sub, DV, 'signal_lag1', 'log_revenue_lag1')
    beta_loo = result_loo.params['signal_lag1']
    se_loo = result_loo.std_errors['signal_lag1']
    t_loo = result_loo.tstats['signal_lag1']
    p_loo = result_loo.pvalues['signal_lag1']
    ci_loo = result_loo.conf_int().loc['signal_lag1']
    print(f"  drop {firm:18s}: beta={beta_loo:9.4f}  SE={se_loo:8.4f}  t={t_loo:7.4f}  "
          f"p={p_loo:.4f}  N={n_loo:3d}  95% CI=[{ci_loo['lower']:9.4f}, {ci_loo['upper']:9.4f}]")
    loo_results.append({'Firm_Dropped': firm, 'Coefficient': round(beta_loo, 4),
                         'Std_Error': round(se_loo, 4), 'T_stat': round(t_loo, 4),
                         'P_value': round(p_loo, 4), 'CI_lower': round(ci_loo['lower'], 4),
                         'CI_upper': round(ci_loo['upper'], 4), 'N': n_loo})
    rows_out.append({'Model': f'H1_NewPrimary_LOO_excl_{firm}', 'IV': 'signal_lag1',
                      'Coefficient': round(beta_loo, 4), 'Std_Error': round(se_loo, 4),
                      'T_stat': round(t_loo, 4), 'P_value': round(p_loo, 4),
                      'CI_lower': round(ci_loo['lower'], 4), 'CI_upper': round(ci_loo['upper'], 4),
                      'N': n_loo})

loo_df = pd.DataFrame(loo_results)
min_row = loo_df.loc[loo_df['Coefficient'].idxmin()]
max_row = loo_df.loc[loo_df['Coefficient'].idxmax()]
ref_sign = beta_new > 0
sign_changes = loo_df[(loo_df['Coefficient'] > 0) != ref_sign]
sig_rows = loo_df[loo_df['P_value'] < 0.05]

print(f"\nMin coefficient: {min_row['Coefficient']:.4f} (dropping {min_row['Firm_Dropped']})")
print(f"Max coefficient: {max_row['Coefficient']:.4f} (dropping {max_row['Firm_Dropped']})")
print(f"Sign changes vs. full-sample sign ({'+' if ref_sign else '-'}): "
      f"{'YES -- ' + ', '.join(sign_changes['Firm_Dropped']) if len(sign_changes) else 'NO'}")
print(f"p < .05 in any iteration: "
      f"{'YES -- ' + ', '.join(sig_rows['Firm_Dropped']) if len(sig_rows) else 'NO'}")

# ══════════════════════════════════════════════════════════════════
# Output + side-by-side summary
# ══════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(rows_out)
results_df.to_csv('results/h1_lag_primary_results.csv', index=False)
print(f"\nresults/h1_lag_primary_results.csv saved -- {len(results_df)} rows")

print("\n" + "=" * 90)
print("SUMMARY: OLD PRIMARY (contemporaneous) vs. NEW PRIMARY (lag-1) side by side")
print("=" * 90)
print(f"{'':30s} {'OLD primary (now supplementary)':>34s} {'NEW primary (lag-1)':>24s}")
print(f"{'IV':30s} {'mean_signal_score':>34s} {'signal_lag1':>24s}")
print(f"{'Control':30s} {'log_revenue':>34s} {'log_revenue_lag1':>24s}")
print(f"{'Zalando 2025':30s} {'included (full N)':>34s} {'excluded':>24s}")
print(f"{'Beta':30s} {supplementary['beta']:34.4f} {new_primary['beta']:24.4f}")
print(f"{'SE':30s} {supplementary['se']:34.4f} {new_primary['se']:24.4f}")
print(f"{'t':30s} {supplementary['t']:34.4f} {new_primary['t']:24.4f}")
print(f"{'p':30s} {supplementary['p']:34.4f} {new_primary['p']:24.4f}")
print(f"{'N':30s} {supplementary['n']:34d} {new_primary['n']:24d}")
print(f"{'R2 (within)':30s} {supplementary['r2']:34.4f} {new_primary['r2']:24.4f}")
print(f"{'df_resid':30s} {supplementary['df_resid']:34d} {new_primary['df_resid']:24d}")
print(f"\nMDE (new primary, 80% power): {mde:.4f} pp ({mde_pct_of_mean:.2f}% of DV mean)")
print(f"TOST (new primary): p_lower={p_lower:.4f}, p_upper={p_upper:.4f}, equivalent={equivalent}")
print(f"LOO (new primary): range [{min_row['Coefficient']:.4f}, {max_row['Coefficient']:.4f}], "
      f"sign changes: {'YES' if len(sign_changes) else 'NO'}, "
      f"any p<.05: {'YES' if len(sig_rows) else 'NO'}")

print("\nDone.")
