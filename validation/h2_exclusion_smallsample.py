"""
Re-estimates H2's lag-1 spec excluding DocMorris and Boohoo entirely (N=46, 12
clusters) -- the one significant coefficient in the analysis -- under CR2/Bell-McCaffrey
and the wild cluster bootstrap; reads data/panel_dataset.csv, writes
results/h2_exclusion_smallsample.csv. 12 clusters makes conventional CR1 SEs unreliable,
so this duplicates the CR2/bootstrap functions from small_sample_inference.py verbatim
rather than importing it.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

RNG_SEED = 42
N_BOOT = 9999
ALPHA = 0.05

# ══════════════════════════════════════════════════════════════════
# Data construction -- mirrors regression_clean.py / small_sample_inference.py
# ══════════════════════════════════════════════════════════════════
raw = pd.read_csv("data/panel_dataset.csv").sort_values(["firm", "year"])
raw["signal_lag1"] = raw.groupby("firm")["mean_signal_score"].shift(1)
gap1 = raw.groupby("firm")["year"].diff()
raw.loc[gap1 != 1, "signal_lag1"] = np.nan

df = raw.set_index(["firm", "year"])
df["log_revenue"] = np.log(df["Revenue"].replace(0, np.nan))
df["log_revenue_lag1"] = df.groupby("firm")["log_revenue"].shift(1)
df["log_revenue_lag1"] = df["log_revenue_lag1"].where(gap1.values == 1, np.nan)

df_primary = df[~((df.index.get_level_values("firm") == "Zalando") &
                   (df.index.get_level_values("year") == 2025))]

DV, IV, CONTROL = "Revenue_Growth_%", "signal_lag1", "log_revenue_lag1"

sample = df_primary[(df_primary.index.get_level_values("firm") != "DocMorris") &
                     (df_primary.index.get_level_values("firm") != "Boohoo")]

# ── Verify against chapter4_gaps.csv before proceeding ─────────────
exog_check = sm.add_constant(sample[[IV, CONTROL]].dropna())
md_check = sample[[DV, IV, CONTROL]].dropna()
check_res = PanelOLS(md_check[DV], sm.add_constant(md_check[[IV, CONTROL]]),
                      entity_effects=True, time_effects=True,
                      drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
EXPECTED_BETA, EXPECTED_P, EXPECTED_N = -15.3376, 0.0030, 46
beta_check = check_res.params[IV]
p_check = check_res.pvalues[IV]
n_check = int(check_res.nobs)
if (abs(beta_check - EXPECTED_BETA) > 0.001 or abs(p_check - EXPECTED_P) > 0.001
        or n_check != EXPECTED_N):
    print(f"FATAL: recomputed spec (beta={beta_check:.4f}, p={p_check:.4f}, N={n_check}) "
          f"does not match chapter4_gaps.csv (beta={EXPECTED_BETA}, p={EXPECTED_P}, "
          f"N={EXPECTED_N}). Refusing to proceed.")
    sys.exit(1)
print(f"Verification passed: beta={beta_check:.4f}, p={p_check:.4f}, N={n_check} "
      f"matches chapter4_gaps.csv exactly.\n")


# ══════════════════════════════════════════════════════════════════
# Verbatim from small_sample_inference.py
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
        eigvals, eigvecs = np.linalg.eigh((M + M.T) / 2)
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
    n, k = X.shape
    clusters = np.unique(cluster_ids)
    G = len(clusters)
    other_idx = [j for j in range(k) if j != coef_idx]
    X_restricted = X[:, other_idx]
    y_adj = y - X[:, coef_idx] * beta0
    XtX_r_inv = np.linalg.inv(X_restricted.T @ X_restricted)
    gamma = XtX_r_inv @ X_restricted.T @ y_adj
    fitted_restricted = X_restricted @ gamma
    resid_restricted = y_adj - fitted_restricted

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


# ══════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════
print("=" * 90)
print(f"H2 lag-1 primary, excl. DocMorris + Boohoo entirely (N=46, 12 clusters)")
print("=" * 90)

X, y, cluster_ids, colnames, model_df = build_lsdv(sample, DV, IV, CONTROL)
coef_idx = colnames.index(IV)
n, k = X.shape
G = len(np.unique(cluster_ids))
print(f"N={n}, K={k} (incl. FE dummies + const), G={G} clusters")

beta_lsdv, resid_lsdv, XtX_inv = ols_fit(X, y)
beta_iv = beta_lsdv[coef_idx]

# validate LSDV vs PanelOLS
match = abs(beta_iv - beta_check) < 1e-6
print(f"VALIDATION: LSDV beta={beta_iv:.6f} vs PanelOLS beta={beta_check:.6f} -> "
      f"{'MATCH' if match else 'MISMATCH -- STOP'}")
if not match:
    sys.exit(1)

se_cr1, G_check = cr1_se(X, resid_lsdv, cluster_ids, XtX_inv, coef_idx)
t_cr1 = beta_iv / se_cr1
p_cr1_conventional = 2 * (1 - stats.t.cdf(abs(t_cr1), G - 1))
print(f"\nCR1 (conventional, df=G-1={G-1}): SE={se_cr1:.4f}, t={t_cr1:.4f}, "
      f"p={p_cr1_conventional:.4f}")

se_cr2, df_bm = cr2_se_and_df(X, resid_lsdv, cluster_ids, XtX_inv, coef_idx)
t_cr2 = beta_iv / se_cr2
p_cr2 = 2 * (1 - stats.t.cdf(abs(t_cr2), df_bm))
print(f"CR2 + Bell-McCaffrey df: SE={se_cr2:.4f}, BM df={df_bm:.2f}, t={t_cr2:.4f}, "
      f"p={p_cr2:.4f}")

rng = np.random.default_rng(RNG_SEED)
p_boot, boot_t = bootstrap_pvalue_at_null(X, y, cluster_ids, coef_idx, 0.0, N_BOOT, rng,
                                           observed_t=t_cr1)
print(f"Wild cluster bootstrap (B={N_BOOT}, Rademacher, restricted/null-imposed): "
      f"p={p_boot:.4f}")

out = pd.DataFrame([{
    "Hypothesis": "H2", "Exclusion": "DocMorris_and_Boohoo_entirely",
    "beta": round(beta_iv, 4), "N": n, "G_clusters": G,
    "CR1_SE": round(se_cr1, 4), "CR2_SE": round(se_cr2, 4), "BM_df": round(df_bm, 2),
    "p_conventional_CR1": round(p_cr1_conventional, 4),
    "p_CR2_BellMcCaffrey": round(p_cr2, 4),
    "p_wild_bootstrap": round(p_boot, 4),
    "N_BOOT": N_BOOT, "seed": RNG_SEED,
}])
print("\n" + out.to_string(index=False))
out.to_csv("results/h2_exclusion_smallsample.csv", index=False)
print("\nresults/h2_exclusion_smallsample.csv saved")
