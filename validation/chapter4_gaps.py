import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

results_rows = []

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

IV = "signal_lag1"
CONTROLS = ["log_revenue_lag1"]
DVS = {"H1": "Stock_Price_Movement_%", "H2": "Revenue_Growth_%", "H3": "Gross_Margin_%"}


def fit(frame, dv):
    md = frame[[dv, IV] + CONTROLS].dropna()
    exog = sm.add_constant(md[[IV] + CONTROLS])
    res = PanelOLS(md[dv], exog, entity_effects=True, time_effects=True,
                    drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    return res, md


# ══════════════════════════════════════════════════════════════════
# CONSTRAINT: assert full-sample coefficients match LOCKED_NUMBERS.md
# ══════════════════════════════════════════════════════════════════
EXPECTED = {"H1": -11.7093, "H2": -5.0803, "H3": -0.6904}
fitted = {}
for hyp, dv in DVS.items():
    res, md = fit(df_primary, dv)
    fitted[hyp] = (res, md)
    beta = res.params[IV]
    if abs(beta - EXPECTED[hyp]) > 0.001:
        print(f"FATAL: {hyp} coefficient {beta:.4f} does not match LOCKED_NUMBERS.md "
              f"({EXPECTED[hyp]}). Refusing to proceed.")
        sys.exit(1)
print("Verification passed: H1/H2/H3 full-sample coefficients match LOCKED_NUMBERS.md "
      f"exactly ({EXPECTED}).\n")

# ══════════════════════════════════════════════════════════════════
# TASK 1 — DESCRIPTIVES (rebuild of Table 4.3)
# ══════════════════════════════════════════════════════════════════
print("=" * 90)
print("TASK 1: DESCRIPTIVES")
print("=" * 90)

DESC_VARS = ["mean_signal_score", "Stock_Price_Movement_%", "Revenue_Growth_%", "Gross_Margin_%"]

# View A: full 69-firm-year panel (contemporaneous variables, no lag/exclusion applied)
full_panel = pd.read_csv("data/panel_dataset.csv")
print("\n-- View A: full panel (no lag, no exclusion) --")
for v in DESC_VARS:
    s = full_panel[v].dropna()
    print(f"  {v}: N={len(s)}, mean={s.mean():.4f}, SD={s.std():.4f}, "
          f"min={s.min():.4f}, max={s.max():.4f}")
    results_rows.append({"Section": "1_Descriptives", "Item": f"{v}_full_panel",
                          "Value1": "N", "Value2": len(s), "Value3": round(s.mean(), 4),
                          "Value4": round(s.std(), 4), "Value5": round(s.min(), 4),
                          "Value6": round(s.max(), 4), "Note": "full 69-obs panel, contemporaneous"})

# View B: current regression sample (N=54 -- the rows that actually enter the
# current lag-1 primary spec: Zalando 2025 excluded + lag-1 requires a prior-year
# observation, which drops each firm's first panel year). Same N=54 row set as
# the H1/H2/H3 estimation sample; contemporaneous (non-lagged) values reported.
current_sample_idx = fitted["H1"][1].index  # the 54-row estimation sample from the H1 fit
current_rows = full_panel.set_index(["firm", "year"]).loc[current_sample_idx].reset_index()
print(f"\n-- View B: current regression sample (N={len(current_rows)}, same 54 rows "
      "that enter the H1/H2/H3 lag-1 estimation) --")
for v in DESC_VARS:
    s = current_rows[v].dropna()
    print(f"  {v}: N={len(s)}, mean={s.mean():.4f}, SD={s.std():.4f}, "
          f"min={s.min():.4f}, max={s.max():.4f}")
    results_rows.append({"Section": "1_Descriptives", "Item": f"{v}_current_sample",
                          "Value1": "N", "Value2": len(s), "Value3": round(s.mean(), 4),
                          "Value4": round(s.std(), 4), "Value5": round(s.min(), 4),
                          "Value6": round(s.max(), 4),
                          "Note": "N=54 current lag-1 primary estimation sample, contemporaneous values"})

print("\nWhich N applies to which variable, and why:")
print("  Full panel: Stock_Price_Movement_% and Gross_Margin_% are N=69 (populated for")
print("  every firm-year). Revenue_Growth_% is N=55 in the full panel (needs a prior-year")
print("  revenue figure, so each firm's first year is naturally NaN). mean_signal_score")
print("  is N=69 (defined as 0 for firm-years with no AI disclosure, never missing).")
print("  Current regression sample: ALL FOUR variables are N=54 uniformly, because the")
print("  current primary spec's own sample restriction (lag-1 signal requires a prior-year")
print("  observation, PLUS Zalando 2025 excluded) already drops every row that would")
print("  otherwise be missing on Revenue_Growth_% -- the lag-1 requirement and the")
print("  first-year revenue-growth gap coincide for every firm.")

# ══════════════════════════════════════════════════════════════════
# TASK 2 — CORRELATION MATRIX (rebuild of Table 4.4)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TASK 2: CORRELATION MATRIX (Pearson, pairwise deletion, current N=54 sample)")
print("=" * 90)

CORR_VARS = ["mean_signal_score", "log_revenue", "Stock_Price_Movement_%",
             "Revenue_Growth_%", "Gross_Margin_%"]
corr_df = current_rows.copy()
corr_df["log_revenue"] = np.log(corr_df["Revenue"].replace(0, np.nan))
corr_mat = corr_df[CORR_VARS].corr(method="pearson")
print(corr_mat.round(4).to_string())

print("\nN underlying each pairwise cell:")
n_mat = pd.DataFrame(index=CORR_VARS, columns=CORR_VARS, dtype=int)
for a in CORR_VARS:
    for b in CORR_VARS:
        n_mat.loc[a, b] = corr_df[[a, b]].dropna().shape[0]
print(n_mat.to_string())

for a in CORR_VARS:
    for b in CORR_VARS:
        results_rows.append({"Section": "2_Correlation", "Item": f"{a}__{b}",
                              "Value1": "r", "Value2": round(corr_mat.loc[a, b], 4),
                              "Value3": "N", "Value4": int(n_mat.loc[a, b]),
                              "Value5": "", "Value6": "", "Note": ""})

# ══════════════════════════════════════════════════════════════════
# TASK 3 — H1 LAG-1 95% CI
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TASK 3: H1 LAG-1 95% CI")
print("=" * 90)

h1_res, h1_md = fitted["H1"]
beta_h1 = h1_res.params[IV]
se_h1 = h1_res.std_errors[IV]
df_resid = 35  # matches LOCKED_NUMBERS.md's cited df_resid for this spec
t_crit = stats.t.ppf(0.975, df_resid)
ci_lower = beta_h1 - t_crit * se_h1
ci_upper = beta_h1 + t_crit * se_h1
print(f"Conventional (clustered SE, df_resid={df_resid}):")
print(f"  beta={beta_h1:.4f}, SE={se_h1:.4f}, t_crit={t_crit:.4f}, "
      f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
results_rows.append({"Section": "3_H1_CI", "Item": "conventional",
                      "Value1": "beta", "Value2": round(beta_h1, 4), "Value3": "SE",
                      "Value4": round(se_h1, 4), "Value5": round(ci_lower, 4),
                      "Value6": round(ci_upper, 4), "Note": f"df_resid={df_resid}"})

# CR2/Bell-McCaffrey: READ from small_sample_inference.csv, not recomputed here.
# That script already validates its LSDV point estimate against PanelOLS before
# trusting its CR2/BM-df output (see its own internal check), and already loops
# over H1 (via h1_lag_primary_results.csv) as well as H2/H3 -- reusing it avoids
# a second, potentially-diverging implementation of the same CR2 math.
ssi = pd.read_csv("results/small_sample_inference.csv")
h1_cr2 = ssi[(ssi["Hypothesis"] == "H1") & (ssi["Metric"] == "CR2_BellMcCaffrey")]
if len(h1_cr2) != 1:
    print("FATAL: could not find exactly one H1 CR2_BellMcCaffrey row in "
          "small_sample_inference.csv.")
    sys.exit(1)
h1_cr2 = h1_cr2.iloc[0]
print(f"\nCR2/Bell-McCaffrey (read from small_sample_inference.csv, df={h1_cr2['df']}):")
print(f"  beta={h1_cr2['Coefficient']:.4f}, SE={h1_cr2['SE']:.4f}, "
      f"95% CI=[{h1_cr2['CI_lower']:.4f}, {h1_cr2['CI_upper']:.4f}]")
results_rows.append({"Section": "3_H1_CI", "Item": "CR2_BellMcCaffrey",
                      "Value1": "beta", "Value2": h1_cr2["Coefficient"], "Value3": "SE",
                      "Value4": h1_cr2["SE"], "Value5": h1_cr2["CI_lower"],
                      "Value6": h1_cr2["CI_upper"],
                      "Note": f"df={h1_cr2['df']}, read from small_sample_inference.csv, not recomputed"})

# ══════════════════════════════════════════════════════════════════
# TASK 4 — TABLE 4.9 REBUILD: SAMPLE COMPOSITION ROBUSTNESS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TASK 4: TABLE 4.9 REBUILD -- SAMPLE COMPOSITION ROBUSTNESS (lag-1 primary spec)")
print("=" * 90)

EXCLUSIONS = {
    "excl_DocMorris_2023_only": lambda d: d[~((d.index.get_level_values("firm") == "DocMorris") &
                                                (d.index.get_level_values("year") == 2023))],
    "excl_DocMorris_entirely": lambda d: d[d.index.get_level_values("firm") != "DocMorris"],
    "excl_Boohoo_entirely": lambda d: d[d.index.get_level_values("firm") != "Boohoo"],
    "excl_DocMorris_and_Boohoo_entirely": lambda d: d[
        (d.index.get_level_values("firm") != "DocMorris") &
        (d.index.get_level_values("firm") != "Boohoo")],
}

h1_sign_flip_seen = False
for excl_name, fn in EXCLUSIONS.items():
    sub = fn(df_primary)
    print(f"\n-- {excl_name} --")
    for hyp, dv in DVS.items():
        res, md = fit(sub, dv)
        beta = res.params[IV]
        se = res.std_errors[IV]
        p = res.pvalues[IV]
        n = int(res.nobs)
        print(f"  {hyp}: beta={beta:.4f}, SE={se:.4f}, p={p:.4f}, N={n}")
        results_rows.append({"Section": "4_Table49", "Item": f"{excl_name}__{hyp}",
                              "Value1": "beta", "Value2": round(beta, 4), "Value3": "SE",
                              "Value4": round(se, 4), "Value5": round(p, 4), "Value6": n,
                              "Note": ""})
        if hyp == "H1" and "DocMorris_entirely" in excl_name:
            flipped = (beta > 0) != (EXPECTED["H1"] > 0)
            h1_sign_flip_seen = flipped
            print(f"    H1 sign vs. full sample ({EXPECTED['H1']}): "
                  f"{'FLIPPED' if flipped else 'same sign'}")

print(f"\nLOCKED_NUMBERS.md check: 'DocMorris flips H1 sign under leave-one-out' -- "
      f"{'REPRODUCES here' if h1_sign_flip_seen else 'DOES NOT reproduce here'} "
      f"under full-firm DocMorris exclusion on the current lag-1 primary spec.")
results_rows.append({"Section": "4_Table49", "Item": "H1_DocMorris_sign_flip_check",
                      "Value1": "reproduces", "Value2": h1_sign_flip_seen, "Value3": "",
                      "Value4": "", "Value5": "", "Value6": "",
                      "Note": "cross-check against LOCKED_NUMBERS.md leave-one-out section"})

# ══════════════════════════════════════════════════════════════════
# WRITE OUTPUT
# ══════════════════════════════════════════════════════════════════
out = pd.DataFrame(results_rows)
out.to_csv("results/chapter4_gaps.csv", index=False)
print(f"\nresults/chapter4_gaps.csv saved -- {len(out)} rows")
