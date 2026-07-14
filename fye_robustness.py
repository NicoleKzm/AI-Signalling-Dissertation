"""
fye_robustness.py

Robustness check: Stock_Price_Movement_% (H1's DV) is computed on calendar-year
windows (1 Jan - 31 Dec, financial_data.py), but five firms have confirmed
non-December fiscal year ends -- AO World (March), ASOS (Aug/Sep), About You
(February), Moonpig (April), Mytheresa (June) -- 23 firm-year observations
(see fiscal_year_end.csv). This creates a signal/return window misalignment for
H1 only: the AI-signal score is read off the fiscal-year annual report, but the
stock return window is calendar-year, so for these five firms the two are
measuring different 12-month windows. H2 and H3 draw both the signal and the
outcome from the same annual report, so both are already on the firm's fiscal
clock -- no misalignment for them. This script re-estimates H1 (and, for
completeness, H2/H3) on the current lag-1 primary specification with these five
firms excluded entirely, to see whether H1's null result is an artefact of this
misalignment.

Does not modify financial_data.py, panel_dataset.csv, or any other script.
Reads panel_dataset.csv (read-only). Writes fye_robustness.csv.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

NON_DECEMBER_FYE_FIRMS = ["AO World", "ASOS", "About You", "Moonpig", "Mytheresa"]

# ── Build lag-1 signal + lag-1 log revenue, current primary spec ───
raw = pd.read_csv("panel_dataset.csv").sort_values(["firm", "year"])
raw["signal_lag1"] = raw.groupby("firm")["mean_signal_score"].shift(1)
gap1 = raw.groupby("firm")["year"].diff()
raw.loc[gap1 != 1, "signal_lag1"] = np.nan

df = raw.set_index(["firm", "year"])
df["log_revenue"] = np.log(df["Revenue"].replace(0, np.nan))
df["log_revenue_lag1"] = df.groupby("firm")["log_revenue"].shift(1)
df["log_revenue_lag1"] = df["log_revenue_lag1"].where(gap1.values == 1, np.nan)

# Symmetric acquisition rule: exclude Zalando 2025 (current primary sample basis)
df_primary = df[~((df.index.get_level_values("firm") == "Zalando") &
                   (df.index.get_level_values("year") == 2025))]

IV = "signal_lag1"
CONTROLS = ["log_revenue_lag1"]
DVS = {
    "H1": "Stock_Price_Movement_%",
    "H2": "Revenue_Growth_%",
    "H3": "Gross_Margin_%",
}


def fit(frame, dv):
    md = frame[[dv, IV] + CONTROLS].dropna()
    exog = sm.add_constant(md[[IV] + CONTROLS])
    res = PanelOLS(md[dv], exog, entity_effects=True, time_effects=True,
                    drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    n_firms = md.index.get_level_values("firm").nunique()
    return res.params[IV], res.std_errors[IV], res.pvalues[IV], int(res.nobs), n_firms


# ── Verification: full-sample H1 lag-1 must match LOCKED_NUMBERS.md exactly ──
beta, se, p, n, n_firms = fit(df_primary, DVS["H1"])
EXPECTED_H1 = -11.7093
if abs(beta - EXPECTED_H1) > 0.001:
    print(f"FATAL: full-sample H1 lag-1 coefficient {beta:.4f} does not match "
          f"LOCKED_NUMBERS.md ({EXPECTED_H1}). Refusing to proceed.")
    sys.exit(1)
print(f"Verification passed: full-sample H1 lag-1 beta={beta:.4f} matches "
      f"LOCKED_NUMBERS.md ({EXPECTED_H1}).\n")

# ── Robustness sample: exclude the 5 non-December-FYE firms ────────
df_fye = df_primary[~df_primary.index.get_level_values("firm").isin(NON_DECEMBER_FYE_FIRMS)]

print(f"Excluding: {', '.join(NON_DECEMBER_FYE_FIRMS)}")
print(f"Firms retained: {df_fye.index.get_level_values('firm').nunique()} of "
      f"{df_primary.index.get_level_values('firm').nunique()}\n")

rows = []
print(f"{'Hyp':4s} {'Sample':10s} {'beta':>10s} {'SE':>10s} {'p':>8s} {'N':>5s} {'n_firms':>8s}")
for hyp, dv in DVS.items():
    b0, se0, p0, n0, f0 = fit(df_primary, dv)
    rows.append({"H": hyp, "sample": "full", "beta": round(b0, 4), "SE": round(se0, 4),
                 "p": round(p0, 4), "N": n0, "n_firms": f0})
    print(f"{hyp:4s} {'full':10s} {b0:10.4f} {se0:10.4f} {p0:8.4f} {n0:5d} {f0:8d}")

    b1, se1, p1, n1, f1 = fit(df_fye, dv)
    rows.append({"H": hyp, "sample": "excl_non_dec_fye", "beta": round(b1, 4), "SE": round(se1, 4),
                 "p": round(p1, 4), "N": n1, "n_firms": f1})
    print(f"{hyp:4s} {'excl_fye':10s} {b1:10.4f} {se1:10.4f} {p1:8.4f} {n1:5d} {f1:8d}")

out = pd.DataFrame(rows)
out.to_csv("fye_robustness.csv", index=False)
print("\nfye_robustness.csv saved")
