"""
Runs permutation inference (1000 shuffles, seed 42) on H1/H2/H3's lag-1 primary
specifications; reads data/panel_dataset.csv, writes
results/randomisation_inference_results.csv. Shuffles the signal score to build a null
distribution of coefficients, a small-cluster-robust alternative to clustered SEs
relevant with only 14 firms.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

# ── Build lag-1 signal + lag-1 log revenue, current primary spec ───
raw = pd.read_csv("data/panel_dataset.csv").sort_values(["firm", "year"])
raw["signal_lag1"] = raw.groupby("firm")["mean_signal_score"].shift(1)
gap1 = raw.groupby("firm")["year"].diff()
raw.loc[gap1 != 1, "signal_lag1"] = np.nan

df = raw.set_index(["firm", "year"])
df["log_revenue"] = np.log(df["Revenue"].replace(0, np.nan))
df["log_revenue_lag1"] = df.groupby("firm")["log_revenue"].shift(1)
df["log_revenue_lag1"] = df["log_revenue_lag1"].where(gap1.values == 1, np.nan)

# Symmetric acquisition rule: exclude Zalando 2025 (current primary sample basis, N=54)
df = df[~((df.index.get_level_values("firm") == "Zalando") &
          (df.index.get_level_values("year") == 2025))]

IV = "signal_lag1"
CONTROLS = ["log_revenue_lag1"]
DVS = {
    "H1": "Stock_Price_Movement_%",
    "H2": "Revenue_Growth_%",
    "H3": "Gross_Margin_%",
}

N_PERM = 1000
SEED = 42
rng = np.random.default_rng(SEED)

EXPECTED = {
    "H1": -11.7093,
    "H2": -5.0803,
    "H3": -0.6904,
}


def fit_full(frame, dv, signal_col):
    """Full fit WITH clustered SEs -- used once per model for the real estimate."""
    md = frame[[dv, signal_col] + CONTROLS].dropna()
    exog = sm.add_constant(md[[signal_col] + CONTROLS])
    res = PanelOLS(md[dv], exog, entity_effects=True, time_effects=True,
                    drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    return res.params[signal_col], res.pvalues[signal_col], int(res.nobs)


def fit_coef_only(dv_series, exog_df):
    """Coefficient ONLY, default covariance -- fast, used inside the shuffle loop."""
    res = PanelOLS(dv_series, exog_df, entity_effects=True, time_effects=True,
                    drop_absorbed=True).fit()
    return res.params.iloc[1]  # signal score is the first regressor after const


print("Randomisation inference on CURRENT primary specs (lag-1 signal, lag-1 "
      "log_revenue, Zalando 2025 excluded, N=54)")
print(f"N_PERM={N_PERM}, seed={SEED}\n")

base = df.reset_index()
rows = []
for hyp, dv in DVS.items():
    real_coef, clust_p, n = fit_full(df, dv, IV)

    expected = EXPECTED[hyp]
    if abs(real_coef - expected) > 0.001:
        print(f"FATAL: {hyp} real coefficient {real_coef:.4f} does not match "
              f"LOCKED_NUMBERS.md ({expected}). Refusing to proceed.")
        sys.exit(1)

    md = base[["firm", "year", dv, IV] + CONTROLS].dropna().reset_index(drop=True)
    signal_vals = md[IV].values
    perm_coefs = np.empty(N_PERM)
    for k in range(N_PERM):
        md["perm_signal"] = rng.permutation(signal_vals)
        sh = md.set_index(["firm", "year"])
        exog = sm.add_constant(sh[["perm_signal"] + CONTROLS])
        try:
            perm_coefs[k] = fit_coef_only(sh[dv], exog)
        except Exception:
            perm_coefs[k] = np.nan

    perm_coefs = perm_coefs[~np.isnan(perm_coefs)]
    ri_p = np.mean(np.abs(perm_coefs) >= np.abs(real_coef))

    rows.append({
        "H": hyp,
        "real_coef": round(real_coef, 4),
        "N": n,
        "clustered_p": round(clust_p, 4),
        "randomisation_p": round(ri_p, 4),
        "n_perm": N_PERM,
        "seed": SEED,
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("results/randomisation_inference_results.csv", index=False)
print("\nresults/randomisation_inference_results.csv saved")
