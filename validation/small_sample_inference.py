"""
Computes CR2/Bell-McCaffrey SEs and a firm-level wild cluster bootstrap for H1/H2/H3
(and the H3 TOST); reads data/panel_dataset.csv, writes
results/small_sample_inference.csv. Implemented from the published formulas directly,
since 14 clusters makes conventional CR1 SEs unreliable and neither linearmodels nor
pyfixest supports CR2/Bell-McCaffrey.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats

RNG_SEED = 42
N_BOOT = 9999
ALPHA = 0.05
H3_BOUND = 3.462

H1_LAG_FILE = 'results/h1_lag_primary_results.csv'
USE_H1_LAG = os.path.exists(H1_LAG_FILE)
print(f"h1_lag_primary_results.csv present: {USE_H1_LAG} "
      f"-> H1 primary = {'lag-1 (signal_lag1)' if USE_H1_LAG else 'contemporaneous (mean_signal_score) -- FLAGGED, script was expected to find the lag-1 file'}")

# ══════════════════════════════════════════════════════════════════
# Data construction -- mirrors regression_clean.py exactly
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv('data/panel_dataset.csv')
df = df.sort_values(['firm', 'year']).reset_index(drop=True)
df['log_revenue'] = np.log(df['Revenue'].replace(0, np.nan))
df['signal_lag1'] = df.groupby('firm')['mean_signal_score'].shift(1)
year_gap1 = df.groupby('firm')['year'].diff()
df.loc[year_gap1 != 1, 'signal_lag1'] = np.nan
df['log_revenue_lag1'] = df.groupby('firm')['log_revenue'].shift(1)
df.loc[year_gap1 != 1, 'log_revenue_lag1'] = np.nan

df_idx = df.set_index(['firm', 'year'])
zalando_mask = ~((df_idx.index.get_level_values('firm') == 'Zalando') &
                  (df_idx.index.get_level_values('year') == 2025))
df_excl_zalando = df_idx[zalando_mask]

HYPS = {
    'H1': {'dv': 'Stock_Price_Movement_%',
           'iv': 'signal_lag1' if USE_H1_LAG else 'mean_signal_score',
           'control': 'log_revenue_lag1' if USE_H1_LAG else 'log_revenue',
           'data': df_excl_zalando if USE_H1_LAG else df_idx},
    'H2': {'dv': 'Revenue_Growth_%', 'iv': 'signal_lag1', 'control': 'log_revenue_lag1',
           'data': df_excl_zalando},
    'H3': {'dv': 'Gross_Margin_%', 'iv': 'signal_lag1', 'control': 'log_revenue_lag1',
           'data': df_excl_zalando},
}


# ══════════════════════════════════════════════════════════════════
# LSDV (dummy-variable) design matrix -- equivalent to PanelOLS's within
# estimator for point estimates; needed in explicit-regressor form for
# CR2/BM-df (which require the model's TRUE hat matrix, including FE).
# ══════════════════════════════════════════════════════════════════
def build_lsdv(data, dv, iv, control):
    model_df = data[[dv, iv, control]].dropna().copy()
    firms = model_df.index.get_level_values('firm')
    years = model_df.index.get_level_values('year')
    firm_dum = pd.get_dummies(firms, prefix='firm', drop_first=True).astype(float).set_axis(model_df.index)
    year_dum = pd.get_dummies(years, prefix='year', drop_first=True).astype(float).set_axis(model_df.index)
    X_df = pd.concat([
        pd.Series(1.0, index=model_df.index, name='const'),
        model_df[iv], model_df[control], firm_dum, year_dum
    ], axis=1)
    X = X_df.values
    y = model_df[dv].values
    cluster_ids = firms.values
    return X, y, cluster_ids, list(X_df.columns), model_df


def ols_fit(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return beta, resid, XtX_inv


def cr1_se(X, resid, cluster_ids, XtX_inv, coef_idx):
    n, k = X.shape
    clusters = np.unique(cluster_ids)
    G = len(clusters)
    meat = np.zeros((k, k))
    for g in clusters:
        idx = cluster_ids == g
        score_g = X[idx].T @ resid[idx]
        meat += np.outer(score_g, score_g)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction * (XtX_inv @ meat @ XtX_inv)
    return np.sqrt(V[coef_idx, coef_idx]), G


def cr2_se_and_df(X, resid, cluster_ids, XtX_inv, coef_idx):
    n, k = X.shape
    clusters = np.unique(cluster_ids)
    c = np.zeros(k)
    c[coef_idx] = 1.0
    meat = np.zeros((k, k))
    v_g_list = []
    for g in clusters:
        idx = cluster_ids == g
        Xg = X[idx]
        eg = resid[idx]
        ng = Xg.shape[0]
        Hgg = Xg @ XtX_inv @ Xg.T
        M = np.eye(ng) - Hgg
        eigvals, eigvecs = np.linalg.eigh((M + M.T) / 2)  # force symmetry against fp noise
        tol = 1e-8 * max(eigvals.max(), 1.0)
        inv_sqrt = np.where(eigvals > tol, eigvals ** -0.5, 0.0)
        Ag = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T
        e_tilde = Ag @ eg
        meat += Xg.T @ np.outer(e_tilde, e_tilde) @ Xg
        Gg = c @ XtX_inv @ Xg.T @ Ag
        v_g_list.append(Gg @ Gg.T)
    V_CR2 = XtX_inv @ meat @ XtX_inv
    se_cr2 = np.sqrt(V_CR2[coef_idx, coef_idx])
    v_g_arr = np.array(v_g_list)
    df_bm = (v_g_arr.sum() ** 2) / (v_g_arr ** 2).sum()
    return se_cr2, df_bm


def wild_cluster_bootstrap(X, y, cluster_ids, coef_idx, beta0, B, rng):
    """Restricted (null-imposed) wild cluster bootstrap, Rademacher weights.
    Returns array of bootstrap t-stats (CR1-studentized) and the observed t-stat."""
    n, k = X.shape
    clusters = np.unique(cluster_ids)
    G = len(clusters)

    # Restricted fit: impose beta[coef_idx] = beta0 by regressing (y - X[:,coef_idx]*beta0)
    # on the remaining columns.
    other_idx = [j for j in range(k) if j != coef_idx]
    X_restricted = X[:, other_idx]
    y_adj = y - X[:, coef_idx] * beta0
    XtX_r_inv = np.linalg.inv(X_restricted.T @ X_restricted)
    gamma = XtX_r_inv @ X_restricted.T @ y_adj
    fitted_restricted = X_restricted @ gamma
    resid_restricted = y_adj - fitted_restricted  # restricted residuals, at beta0

    XtX_inv = np.linalg.inv(X.T @ X)

    cluster_row_idx = {g: np.where(cluster_ids == g)[0] for g in clusters}
    weight_draws = rng.choice([-1.0, 1.0], size=(B, G))

    boot_t = np.empty(B)
    for b in range(B):
        y_star = fitted_restricted + X[:, coef_idx] * beta0
        for gi, g in enumerate(clusters):
            rows = cluster_row_idx[g]
            y_star[rows] = (fitted_restricted[rows] + X[rows, coef_idx] * beta0
                             + weight_draws[b, gi] * resid_restricted[rows])
        beta_star = XtX_inv @ X.T @ y_star
        resid_star = y_star - X @ beta_star
        se_star, _ = cr1_se(X, resid_star, cluster_ids, XtX_inv, coef_idx)
        boot_t[b] = (beta_star[coef_idx] - beta0) / se_star

    return boot_t


def bootstrap_pvalue_at_null(X, y, cluster_ids, coef_idx, beta0, B, rng, observed_t=None):
    boot_t = wild_cluster_bootstrap(X, y, cluster_ids, coef_idx, beta0, B, rng)
    if observed_t is None:
        beta_full, resid_full, XtX_inv_full = ols_fit(X, y)
        se_full, _ = cr1_se(X, resid_full, cluster_ids, XtX_inv_full, coef_idx)
        observed_t = (beta_full[coef_idx] - beta0) / se_full
    pval = np.mean(np.abs(boot_t) >= np.abs(observed_t))
    return pval, boot_t


def wild_bootstrap_ci(X, y, cluster_ids, coef_idx, beta_hat, se_hat, B, rng, alpha=0.05, tol=1e-3, max_iter=40):
    """95% CI via test inversion: find beta0 values where the restricted
    wild-bootstrap p-value crosses alpha, via bisection. Uses a smaller B
    for the search (faster), consistent with standard practice."""

    def p_at(beta0):
        t_obs = (beta_hat - beta0) / se_hat
        boot_t = wild_cluster_bootstrap(X, y, cluster_ids, coef_idx, beta0, B, rng)
        return np.mean(np.abs(boot_t) >= np.abs(t_obs))

    # Bracket using +/- a wide multiple of the CR1 SE, then bisect inward.
    lo_bracket, hi_bracket = beta_hat - 8 * se_hat, beta_hat + 8 * se_hat

    def bisect_boundary(direction):
        # direction=-1 for lower bound (search left of beta_hat),
        # direction=+1 for upper bound (search right of beta_hat)
        inner = beta_hat
        outer = lo_bracket if direction < 0 else hi_bracket
        for _ in range(max_iter):
            mid = (inner + outer) / 2
            p = p_at(mid)
            if p > alpha:
                inner = mid
            else:
                outer = mid
            if abs(outer - inner) < tol * abs(se_hat):
                break
        return (inner + outer) / 2

    ci_lower = bisect_boundary(-1)
    ci_upper = bisect_boundary(+1)
    return ci_lower, ci_upper


# ══════════════════════════════════════════════════════════════════
# Run for each hypothesis
# ══════════════════════════════════════════════════════════════════
rows_out = []
h3_bootstrap_t = None
h3_result_summary = {}

for hyp, spec in HYPS.items():
    print("\n" + "=" * 90)
    print(f"{hyp}: DV={spec['dv']}, IV={spec['iv']}, control={spec['control']}")
    print("=" * 90)

    X, y, cluster_ids, colnames, model_df = build_lsdv(spec['data'], spec['dv'], spec['iv'], spec['control'])
    coef_idx = colnames.index(spec['iv'])
    n, k = X.shape
    G = len(np.unique(cluster_ids))
    print(f"N={n}, K={k} (incl. FE dummies + const), G={G} clusters")

    beta_lsdv, resid_lsdv, XtX_inv = ols_fit(X, y)

    # ── Validation: LSDV point estimate must match PanelOLS exactly ──
    exog = sm.add_constant(model_df[[spec['iv'], spec['control']]])
    panelols_result = PanelOLS(model_df[spec['dv']], exog, entity_effects=True, time_effects=True,
                                drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    beta_panelols = panelols_result.params[spec['iv']]
    se_panelols_cr1 = panelols_result.std_errors[spec['iv']]
    beta_lsdv_iv = beta_lsdv[coef_idx]
    match = abs(beta_lsdv_iv - beta_panelols) < 1e-6
    print(f"VALIDATION: LSDV beta={beta_lsdv_iv:.6f} vs PanelOLS beta={beta_panelols:.6f} "
          f"-> {'MATCH' if match else 'MISMATCH -- STOP, investigate'}")
    if not match:
        raise SystemExit(f"{hyp}: LSDV/PanelOLS point-estimate mismatch, aborting.")

    # ── CR1 (own implementation, for internal comparison) ──
    se_cr1_own, G_check = cr1_se(X, resid_lsdv, cluster_ids, XtX_inv, coef_idx)
    print(f"\nCR1 (own LSDV impl.): SE={se_cr1_own:.4f}  "
          f"(PanelOLS reported CR1: SE={se_panelols_cr1:.4f}, "
          f"{'close match' if abs(se_cr1_own - se_panelols_cr1) < 0.5 else 'differs -- likely small-sample correction convention'})")

    # ── CR0 (no small-sample correction) for reference ──
    n_, k_ = X.shape
    meat0 = np.zeros((k_, k_))
    for g in np.unique(cluster_ids):
        idx = cluster_ids == g
        s = X[idx].T @ resid_lsdv[idx]
        meat0 += np.outer(s, s)
    V0 = XtX_inv @ meat0 @ XtX_inv
    se_cr0 = np.sqrt(V0[coef_idx, coef_idx])

    # ── 1. CR2 + Bell-McCaffrey df ──
    se_cr2, df_bm = cr2_se_and_df(X, resid_lsdv, cluster_ids, XtX_inv, coef_idx)
    t_cr2 = beta_lsdv_iv / se_cr2
    p_cr2 = 2 * (1 - stats.t.cdf(abs(t_cr2), df_bm))
    crit_cr2 = stats.t.ppf(1 - ALPHA / 2, df_bm)
    ci_cr2 = (beta_lsdv_iv - crit_cr2 * se_cr2, beta_lsdv_iv + crit_cr2 * se_cr2)

    print(f"\n1. CR2 + Bell-McCaffrey df:")
    print(f"   beta={beta_lsdv_iv:.4f}, CR0 SE={se_cr0:.4f}, CR1 SE={se_cr1_own:.4f}, "
          f"CR2 SE={se_cr2:.4f}")
    print(f"   BM adjusted df={df_bm:.2f} (vs. conventional G-1={G-1})")
    print(f"   t={t_cr2:.4f}, p={p_cr2:.4f}, 95% CI=[{ci_cr2[0]:.4f}, {ci_cr2[1]:.4f}]")

    rows_out.append({'Hypothesis': hyp, 'Metric': 'CR2_BellMcCaffrey', 'Coefficient': round(beta_lsdv_iv, 4),
                      'SE': round(se_cr2, 4), 'df': round(df_bm, 2), 'T_stat': round(t_cr2, 4),
                      'P_value': round(p_cr2, 4), 'CI_lower': round(ci_cr2[0], 4),
                      'CI_upper': round(ci_cr2[1], 4), 'N': n, 'G': G})
    rows_out.append({'Hypothesis': hyp, 'Metric': 'CR1_conventional', 'Coefficient': round(beta_panelols, 4),
                      'SE': round(se_panelols_cr1, 4), 'df': G - 1,
                      'T_stat': round(beta_panelols / se_panelols_cr1, 4),
                      'P_value': round(panelols_result.pvalues[spec['iv']], 4), 'N': n, 'G': G})
    rows_out.append({'Hypothesis': hyp, 'Metric': 'CR0_no_correction', 'Coefficient': round(beta_lsdv_iv, 4),
                      'SE': round(se_cr0, 4), 'N': n, 'G': G})

    # ── 2. Wild cluster bootstrap (restricted, Rademacher, N_BOOT reps) ──
    rng = np.random.default_rng(RNG_SEED)
    p_boot, boot_t_dist = bootstrap_pvalue_at_null(X, y, cluster_ids, coef_idx, beta0=0.0,
                                                     B=N_BOOT, rng=rng, observed_t=None)
    print(f"\n2. Wild cluster bootstrap (Rademacher, {N_BOOT} reps, restricted at beta=0):")
    print(f"   Bootstrap p-value: {p_boot:.4f}")

    rng_ci = np.random.default_rng(RNG_SEED + 1)
    ci_boot_lower, ci_boot_upper = wild_bootstrap_ci(X, y, cluster_ids, coef_idx, beta_lsdv_iv,
                                                       se_cr1_own, B=999, rng=rng_ci, alpha=ALPHA)
    print(f"   Bootstrap 95% CI (test inversion, B=999/grid-point): "
          f"[{ci_boot_lower:.4f}, {ci_boot_upper:.4f}]")

    rows_out.append({'Hypothesis': hyp, 'Metric': 'WildClusterBootstrap', 'Coefficient': round(beta_lsdv_iv, 4),
                      'Bootstrap_p': round(p_boot, 4), 'CI_lower': round(ci_boot_lower, 4),
                      'CI_upper': round(ci_boot_upper, 4), 'N_reps': N_BOOT, 'N': n, 'G': G})

    if hyp == 'H3':
        h3_result_summary = {'beta': beta_lsdv_iv, 'se_cr2': se_cr2, 'df_bm': df_bm,
                              'X': X, 'y': y, 'cluster_ids': cluster_ids, 'coef_idx': coef_idx,
                              'n': n, 'G': G}

# ══════════════════════════════════════════════════════════════════
# 3. H3 TOST under (a) CR2+BM df and (b) wild cluster bootstrap
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("3. H3 TOST (bound=+/-3.462) under small-sample-corrected inference")
print("=" * 90)

beta_h3 = h3_result_summary['beta']
se_h3_cr2 = h3_result_summary['se_cr2']
df_h3_bm = h3_result_summary['df_bm']

t_lower_cr2 = (beta_h3 - (-H3_BOUND)) / se_h3_cr2
t_upper_cr2 = (H3_BOUND - beta_h3) / se_h3_cr2
p_lower_cr2 = 1 - stats.t.cdf(t_lower_cr2, df_h3_bm)
p_upper_cr2 = 1 - stats.t.cdf(t_upper_cr2, df_h3_bm)
equiv_cr2 = (p_lower_cr2 < ALPHA) and (p_upper_cr2 < ALPHA)

print(f"\n(a) CR2 + Bell-McCaffrey df={df_h3_bm:.2f}:")
print(f"    beta={beta_h3:.4f}, SE_CR2={se_h3_cr2:.4f}")
print(f"    t_lower={t_lower_cr2:.4f}, t_upper={t_upper_cr2:.4f}")
print(f"    p_lower={p_lower_cr2:.4f}, p_upper={p_upper_cr2:.4f}")
print(f"    Equivalent: {equiv_cr2}")

# (b) Wild cluster bootstrap TOST: two one-sided restricted bootstraps,
# one testing beta0 = -bound (H0: beta <= -bound, want to reject in favor
# of beta > -bound), one testing beta0 = +bound (H0: beta >= bound).
rng_tost = np.random.default_rng(RNG_SEED + 2)
Xh3, yh3, cidh3, coefh3 = (h3_result_summary['X'], h3_result_summary['y'],
                            h3_result_summary['cluster_ids'], h3_result_summary['coef_idx'])

boot_t_lower_bound = wild_cluster_bootstrap(Xh3, yh3, cidh3, coefh3, beta0=-H3_BOUND, B=N_BOOT, rng=rng_tost)
t_obs_lower = t_lower_cr2 * se_h3_cr2 / se_h3_cr2  # recompute cleanly below instead
beta_full_h3, resid_full_h3, XtXinv_h3 = ols_fit(Xh3, yh3)
se_cr1_h3, _ = cr1_se(Xh3, resid_full_h3, cidh3, XtXinv_h3, coefh3)
t_obs_lower_cr1 = (beta_h3 - (-H3_BOUND)) / se_cr1_h3
p_lower_boot = np.mean(boot_t_lower_bound >= t_obs_lower_cr1)  # one-sided: is beta credibly > -bound

rng_tost2 = np.random.default_rng(RNG_SEED + 3)
boot_t_upper_bound = wild_cluster_bootstrap(Xh3, yh3, cidh3, coefh3, beta0=H3_BOUND, B=N_BOOT, rng=rng_tost2)
t_obs_upper_cr1 = (beta_h3 - H3_BOUND) / se_cr1_h3
p_upper_boot = np.mean(boot_t_upper_bound <= t_obs_upper_cr1)  # one-sided: is beta credibly < bound

equiv_boot = (p_lower_boot < ALPHA) and (p_upper_boot < ALPHA)

print(f"\n(b) Wild cluster bootstrap ({N_BOOT} reps per one-sided test):")
print(f"    p_lower (H0: beta <= -3.462): {p_lower_boot:.4f}")
print(f"    p_upper (H0: beta >= +3.462): {p_upper_boot:.4f}")
print(f"    Equivalent: {equiv_boot}")

print(f"\nConventional result (for comparison): p_lower=0.0334, p_upper=0.0038, df=35, equivalent=True")

rows_out.append({'Hypothesis': 'H3', 'Metric': 'TOST_CR2_BellMcCaffrey', 'Coefficient': round(beta_h3, 4),
                  'df': round(df_h3_bm, 2), 'p_lower': round(p_lower_cr2, 4),
                  'p_upper': round(p_upper_cr2, 4), 'Equivalent': equiv_cr2})
rows_out.append({'Hypothesis': 'H3', 'Metric': 'TOST_WildClusterBootstrap', 'Coefficient': round(beta_h3, 4),
                  'p_lower': round(p_lower_boot, 4), 'p_upper': round(p_upper_boot, 4),
                  'Equivalent': equiv_boot, 'N_reps': N_BOOT})
rows_out.append({'Hypothesis': 'H3', 'Metric': 'TOST_Conventional_reference', 'df': 35,
                  'p_lower': 0.0334, 'p_upper': 0.0038, 'Equivalent': True})

# ══════════════════════════════════════════════════════════════════
# 4. Plain statement
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("4. Does H3 equivalence survive small-sample-corrected inference?")
print("=" * 90)
if equiv_cr2 and equiv_boot:
    print("YES -- H3 equivalence holds under BOTH CR2+Bell-McCaffrey and the wild cluster bootstrap.")
else:
    print("NO -- H3 equivalence does NOT survive under at least one small-sample-corrected procedure:")
    if not equiv_cr2:
        print(f"  FAILS under CR2+Bell-McCaffrey: p_lower={p_lower_cr2:.4f}, p_upper={p_upper_cr2:.4f} "
              f"(needs both < 0.05; {'p_lower' if p_lower_cr2 >= ALPHA else 'p_upper'} fails)")
    if not equiv_boot:
        print(f"  FAILS under wild cluster bootstrap: p_lower={p_lower_boot:.4f}, p_upper={p_upper_boot:.4f} "
              f"(needs both < 0.05; {'p_lower' if p_lower_boot >= ALPHA else 'p_upper'} fails)")

# ══════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(rows_out)
results_df.to_csv('results/small_sample_inference.csv', index=False)
print(f"\nresults/small_sample_inference.csv saved -- {len(results_df)} rows")
print("\nDone.")
