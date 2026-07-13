"""
leave_one_out_primary.py

Leave-one-firm-out robustness check on the ACTUAL primary specifications
from regression_clean.py -- not the divergent contemporaneous/Revenue_EUR/
Zalando-included/OLS-LSDV specification make_figures.py currently runs.

Mirrors regression_clean.py exactly:
  - linearmodels PanelOLS, entity + time fixed effects
  - clustered SEs by firm (cov_type='clustered', cluster_entity=True)
  - H1: IV=mean_signal_score, control=log_revenue=ln(Revenue), full N=69
  - H2/H3: IV=signal_lag1, control=log_revenue_lag1, excl. Zalando 2025, N=54
  - Revenue column (NOT Revenue_EUR)

Does NOT modify classify.py, regression_clean.py, regression_results.csv,
make_figures.py, panel_dataset.csv, or retest_table_4_8.py -- reads
panel_dataset.csv only. EQUIVALENCE_BOUNDS is defined locally below (no
dependency on retest_table_4_8.py, whose stale hardcoded beta/SE/N values
and import-time print side effect caused a phantom TOST discrepancy --
see the diagnosis that led to this fix). Writes only to
leave_one_out_primary.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats as scipy_stats

EQUIVALENCE_BOUNDS = {
    "H1": 10.512,   # 0.2 x SD(Stock_Price_Movement_%) = 0.2 x 52.560
    "H2": 4.038,    # 0.2 x SD(Revenue_Growth_%)      = 0.2 x 20.189
    "H3": 3.462,    # 0.2 x SD(Gross_Margin_%)        = 0.2 x 17.308
}
# Pre-specified prior to estimation. Verified against dynamic computation
# from panel_dataset.csv in tost_mde_v2.py. Do NOT import from
# retest_table_4_8.py -- that file holds stale pre-refactor beta/SE/N values.

ALPHA = 0.05
N_EXOG = 3  # const + IV + control, same convention as tost_mde_v2.py

# ══════════════════════════════════════════════════════════════════
# Replicate regression_clean.py's data construction EXACTLY
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv('panel_dataset.csv')
df = df.sort_values(['firm', 'year']).reset_index(drop=True)
print(f"Loaded panel_dataset.csv: {len(df)} rows, {df['firm'].nunique()} firms")

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

ALL_FIRMS = sorted(df.index.get_level_values('firm').unique())
print(f"Firms ({len(ALL_FIRMS)}): {ALL_FIRMS}")


def fit(data, dv, iv, control):
    full = data[[dv, iv, control]]
    model_df = full.dropna()
    exog = sm.add_constant(model_df[[iv, control]])
    result = PanelOLS(model_df[dv], exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    return result, model_df


def compute_df(model_df):
    n_entities = model_df.index.get_level_values('firm').nunique()
    n_years = model_df.index.get_level_values('year').nunique()
    n = len(model_df)
    return n - (n_entities - 1) - (n_years - 1) - N_EXOG


HYPS = {
    'H1_Stock_Price': {'data': df, 'dv': 'Stock_Price_Movement_%',
                        'iv': 'mean_signal_score', 'control': 'log_revenue'},
    'H2_Revenue_Growth': {'data': df_excl_zalando, 'dv': 'Revenue_Growth_%',
                           'iv': 'signal_lag1', 'control': 'log_revenue_lag1'},
    'H3_Gross_Margin': {'data': df_excl_zalando, 'dv': 'Gross_Margin_%',
                         'iv': 'signal_lag1', 'control': 'log_revenue_lag1'},
}

# ══════════════════════════════════════════════════════════════════
# Full-sample primary models (reference) + sanity check vs known baseline
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("FULL-SAMPLE PRIMARY MODELS (reference)")
print("=" * 90)
primary_ref = {}
for h, spec in HYPS.items():
    result, model_df = fit(spec['data'], spec['dv'], spec['iv'], spec['control'])
    beta = result.params[spec['iv']]
    se = result.std_errors[spec['iv']]
    n = len(model_df)
    primary_ref[h] = {'beta': beta, 'se': se, 'n': n}
    print(f"{h}: beta={beta:.4f}, SE={se:.4f}, N={n}, p={result.pvalues[spec['iv']]:.4f}")

h1_ref = primary_ref['H1_Stock_Price']
sanity_ok = (abs(h1_ref['beta'] - (-22.624)) < 0.001 and abs(h1_ref['se'] - 35.090) < 0.001
             and h1_ref['n'] == 69)
print(f"\nSanity check vs. known H1 baseline (-22.624, 35.090, N=69): "
      f"{'PASSED' if sanity_ok else 'FAILED -- investigate before trusting the LOO results below'}")

h3_bound = EQUIVALENCE_BOUNDS["H3"]
print(f"H3 equivalence bound (local, pre-specified): +/-{h3_bound:.3f}")

# ══════════════════════════════════════════════════════════════════
# Leave-one-firm-out, 14 iterations per hypothesis
# ══════════════════════════════════════════════════════════════════
results_rows = []
loo_by_hyp = {h: [] for h in HYPS}

for h, spec in HYPS.items():
    print("\n" + "=" * 90)
    print(f"LEAVE-ONE-OUT: {h}  (IV={spec['iv']}, control={spec['control']})")
    print("=" * 90)
    for firm_dropped in ALL_FIRMS:
        sub = spec['data'][spec['data'].index.get_level_values('firm') != firm_dropped]
        result, model_df = fit(sub, spec['dv'], spec['iv'], spec['control'])
        beta = result.params[spec['iv']]
        se = result.std_errors[spec['iv']]
        t = result.tstats[spec['iv']]
        p = result.pvalues[spec['iv']]
        n = len(model_df)
        ci = result.conf_int().loc[spec['iv']]
        ci_lo, ci_hi = ci['lower'], ci['upper']

        row = {'Hypothesis': h, 'Firm_Dropped': firm_dropped,
               'Coefficient': round(beta, 4), 'Std_Error': round(se, 4),
               'T_stat': round(t, 4), 'P_value': round(p, 4), 'N': n,
               'CI_lower': round(ci_lo, 4), 'CI_upper': round(ci_hi, 4),
               'TOST_p_lower': np.nan, 'TOST_p_upper': np.nan, 'TOST_Equivalent': np.nan}

        if h == 'H3_Gross_Margin':
            dof = compute_df(model_df)
            t_lower = (beta - (-h3_bound)) / se
            t_upper = (h3_bound - beta) / se
            p_lo = 1 - scipy_stats.t.cdf(t_lower, dof)
            p_hi = 1 - scipy_stats.t.cdf(t_upper, dof)
            equiv = (p_lo < ALPHA) and (p_hi < ALPHA)
            row['TOST_p_lower'] = round(p_lo, 4)
            row['TOST_p_upper'] = round(p_hi, 4)
            row['TOST_Equivalent'] = equiv
            print(f"  drop {firm_dropped:18s}: beta={beta:8.4f}  SE={se:7.4f}  "
                  f"t={t:7.4f}  p={p:.4f}  N={n:3d}  95% CI=[{ci_lo:8.4f}, {ci_hi:8.4f}]  "
                  f"df={dof}  TOST p_lo={p_lo:.4f} p_hi={p_hi:.4f} equiv={equiv}")
        else:
            print(f"  drop {firm_dropped:18s}: beta={beta:8.4f}  SE={se:7.4f}  "
                  f"t={t:7.4f}  p={p:.4f}  N={n:3d}  95% CI=[{ci_lo:8.4f}, {ci_hi:8.4f}]")

        results_rows.append(row)
        loo_by_hyp[h].append(row)

# ══════════════════════════════════════════════════════════════════
# Write CSV
# ══════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results_rows, columns=[
    'Hypothesis', 'Firm_Dropped', 'Coefficient', 'Std_Error', 'T_stat', 'P_value', 'N',
    'CI_lower', 'CI_upper', 'TOST_p_lower', 'TOST_p_upper', 'TOST_Equivalent'
])
results_df.to_csv('leave_one_out_primary.csv', index=False)
print(f"\nleave_one_out_primary.csv saved -- {len(results_df)} rows")

# ══════════════════════════════════════════════════════════════════
# Per-hypothesis summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

for h in HYPS:
    rows = loo_by_hyp[h]
    coefs = [r['Coefficient'] for r in rows]
    ref = primary_ref[h]
    min_row = min(rows, key=lambda r: r['Coefficient'])
    max_row = max(rows, key=lambda r: r['Coefficient'])

    sign_changes = [r['Firm_Dropped'] for r in rows if (r['Coefficient'] > 0) != (ref['beta'] > 0)]
    sig_rows = [r['Firm_Dropped'] for r in rows if r['P_value'] < 0.05]

    print(f"\n{h}:")
    print(f"  Full-sample primary: beta={ref['beta']:.4f}, SE={ref['se']:.4f}, N={ref['n']}")
    print(f"  Min coefficient across 14 LOO iterations: {min_row['Coefficient']:.4f} "
          f"(dropping {min_row['Firm_Dropped']})")
    print(f"  Max coefficient across 14 LOO iterations: {max_row['Coefficient']:.4f} "
          f"(dropping {max_row['Firm_Dropped']})")
    print(f"  Sign changes vs. full-sample sign ({'+' if ref['beta']>0 else '-'}): "
          f"{'YES -- ' + ', '.join(sign_changes) if sign_changes else 'NO'}")
    print(f"  p < .05 in any iteration: "
          f"{'YES -- ' + ', '.join(sig_rows) if sig_rows else 'NO'}")

    if h == 'H3_Gross_Margin':
        breaks = [r['Firm_Dropped'] for r in rows if not r['TOST_Equivalent']]
        print(f"  H3 TOST equivalence holds in all 14 leave-one-out iterations: "
              f"{'YES' if not breaks else 'NO -- broken when dropping: ' + ', '.join(breaks)}")

print("\nDone.")
