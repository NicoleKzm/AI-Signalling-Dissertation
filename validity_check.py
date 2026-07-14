# ============================================================
# DEPRECATED — DO NOT CITE. DO NOT RUN.
# This script is hardcoded to the OLD contemporaneous specification
# (unlagged IV, unlagged control, Zalando 2025 included).
# It does NOT reflect the current primary specification.
# Superseded by: randomisation_inference.py (RI) and
# small_sample_inference.py (CR2, wild cluster bootstrap).
# Retained for audit trail only.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lets the script save figures without a display
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
 
# ------------------------------------------------------------------
# 0. Load exactly as regression.py does, so the samples match
# ------------------------------------------------------------------
df = pd.read_csv("panel_dataset.csv")
df = df.dropna(subset=["Revenue_Growth_%"])          # same first-year drop -> N=55
df["log_revenue"] = np.log(df["Revenue"].replace(0, np.nan))
df = df.set_index(["firm", "year"])
 
IV = "mean_signal_score"
CONTROLS = ["log_revenue"]
DVS = {
    "H1_Stock_Price":   "Stock_Price_Movement_%",
    "H2_Revenue_Growth":"Revenue_Growth_%",
    "H3_Gross_Margin":  "Gross_Margin_%",
}
 
print(f"Estimation sample: {len(df)} firm-year observations\n")
 
# ==================================================================
# 1. SIGNAL SCORE DISTRIBUTION  (turns "restricted variance" into a picture)
# ==================================================================
scores = df[IV].dropna()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(scores, bins=20, edgecolor="black", linewidth=0.6)
ax.axvline(scores.mean(), color="crimson", linestyle="--", linewidth=1.5,
           label=f"Mean = {scores.mean():.3f}")
ax.set_xlabel("Firm-year mean signal score (0 = Symbolic, 2 = Substantive)")
ax.set_ylabel("Number of firm-years")
ax.set_title("Distribution of AI signalling scores, 2021\u20132025")
ax.legend()
fig.tight_layout()
fig.savefig("signal_score_distribution.png", dpi=150)
plt.close(fig)
print("[1] Saved signal_score_distribution.png")
print(f"    mean={scores.mean():.3f}  sd={scores.std():.3f}  "
      f"share at exactly 0: {(scores==0).mean()*100:.0f}%\n")
 
# ==================================================================
# 2. POSITIVE CONTROL: does signalling rise over time as theory predicts?
# ==================================================================
# Mean signal score by calendar year
by_year = df.reset_index().groupby("year")[IV].mean()
print("[2] Mean signal score by year (positive control):")
for y, v in by_year.items():
    print(f"    {y}: {v:.3f}")
 
# Simple test: regress signal score on a linear year trend (pooled OLS).
# A positive, sizeable slope = signalling grows over the window, as expected
# after the late-2022 ChatGPT discourse surge.
tmp = df.reset_index()[["year", IV]].dropna()
X = sm.add_constant(tmp["year"] - tmp["year"].min())  # year recentred to 0..4
trend = sm.OLS(tmp[IV], X).fit()
slope = trend.params.iloc[1]
slope_p = trend.pvalues.iloc[1]
print(f"\n    Linear trend: +{slope:.3f} signal-score points per year "
      f"(p = {slope_p:.3f})")
print("    Interpretation: a positive, significant slope supports construct")
print("    validity -- the instrument tracks the real rise in AI discourse,")
print("    so the financial nulls are not an artefact of a dead variable.\n")
 
# Plot the trend
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(by_year.index, by_year.values, marker="o", linewidth=1.8)
ax.set_xlabel("Year")
ax.set_ylabel("Mean signal score")
ax.set_title("AI signalling intensity over time (positive control)")
ax.set_xticks(list(by_year.index))
fig.tight_layout()
fig.savefig("signal_trend_by_year.png", dpi=150)
plt.close(fig)
print("    Saved signal_trend_by_year.png\n")
 
# ==================================================================
# 3. RANDOMIZATION INFERENCE  (correct inference with only 14 clusters)
# ==================================================================
# Logic: if the signal score truly has no relationship with the outcome,
# then SHUFFLING the signal score across observations should produce a
# coefficient just as large as the one you actually estimated. We shuffle
# many times, build the null distribution of coefficients, and ask how
# often a random shuffle beats your real estimate. That fraction is the
# randomization-inference p-value -- and it makes no assumption about the
# number of clusters.
 
N_PERM = 1000          # 1000 shuffles gives a p stable to ~0.01; raise to 5000 for the final run
rng = np.random.default_rng(42)
 
def fit_full(frame, dv, signal_col):
    """Full fit WITH clustered SEs -- used once per model for the real estimate."""
    md = frame[[dv, signal_col] + CONTROLS].dropna()
    exog = sm.add_constant(md[[signal_col] + CONTROLS])
    res = PanelOLS(md[dv], exog,
                   entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(cov_type="clustered",
                                           cluster_entity=True)
    return res.params[signal_col], res.pvalues[signal_col]
 
def fit_coef_only(dv_series, exog_df):
    """Coefficient ONLY, default covariance -- fast, used inside the shuffle loop.
    Randomization inference needs the coefficient distribution, not the SEs,
    so we skip the expensive clustered covariance here."""
    res = PanelOLS(dv_series, exog_df,
                   entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit()      # default cov = fast
    return res.params.iloc[1]  # signal score is the first regressor after const
 
print("[3] Randomization inference (N = %d shuffles):" % N_PERM)
print("    %-20s %10s %12s %14s" % ("Hypothesis", "RealCoef", "Clustered-p", "Randomiz.-p"))
 
base = df.reset_index()
for hyp, dv in DVS.items():
    # real estimate (with proper clustered SE, once)
    real_coef, clust_p = fit_full(df, dv, IV)
 
    # pre-build the clean modelling frame once, then just swap the shuffled column
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
    print("    %-20s %10.4f %12.4f %14.4f" % (hyp, real_coef, clust_p, ri_p))
 
print("""
    How to read this:
    - If the randomization p is also > 0.10, your null SURVIVES the correct
      small-sample test -- a much stronger statement than the clustered p alone.
    - If clustered and randomization p disagree sharply, report BOTH and trust
      the randomization p, because 14 clusters violates the clustered-SE
      assumptions (you already note this in Section 3.9).
""")
 
print("Done. Three outputs generated. Read them, then write up what you found.")