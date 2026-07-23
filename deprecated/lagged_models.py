from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RESULTS = ROOT / "data" / "results"

df = pd.read_csv(DATA_PROCESSED / "panel_dataset.csv").dropna(subset=["Revenue_Growth_%"])
df = df.sort_values(["firm", "year"])

# Build lags within firm on the full panel, guarding against year gaps
df["signal_lag1"] = df.groupby("firm")["mean_signal_score"].shift(1)
df["signal_lag2"] = df.groupby("firm")["mean_signal_score"].shift(2)
df["year_gap1"] = df.groupby("firm")["year"].diff()
df.loc[df["year_gap1"] != 1, "signal_lag1"] = np.nan
df["year_gap2"] = df.groupby("firm")["year"].diff(2)
df.loc[df["year_gap2"] != 2, "signal_lag2"] = np.nan

df = df.set_index(["firm", "year"])
df["log_revenue"] = np.log(df["Revenue"].replace(0, np.nan))

MODELS = [
    # (label, DV, IV) — lagged primary tests plus contemporaneous benchmarks
    ("H1 contemporaneous (primary)", "Stock_Price_Movement_%", "mean_signal_score"),
    ("H1 lag-1 (benchmark)", "Stock_Price_Movement_%", "signal_lag1"),
    ("H2 contemporaneous (benchmark)", "Revenue_Growth_%", "mean_signal_score"),
    ("H2 lag-1 (primary)", "Revenue_Growth_%", "signal_lag1"),
    ("H3 contemporaneous (benchmark)", "Gross_Margin_%", "mean_signal_score"),
    ("H3 lag-1 (primary)", "Gross_Margin_%", "signal_lag1"),
    ("H3 lag-2 (exploratory)", "Gross_Margin_%", "signal_lag2"),
]

rows = []
print("Lagged panel specifications (two-way FE, firm-clustered SEs)")
print("=" * 64)
for label, dv, iv in MODELS:
    m = df[[dv, iv, "log_revenue"]].dropna()
    if m.index.get_level_values("year").nunique() < 3 or len(m) < 25:
        print(f"{label}: insufficient observations (N={len(m)}), skipped")
        continue
    res = PanelOLS(
        m[dv],
        sm.add_constant(m[[iv, "log_revenue"]]),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True)
    b = res.params[iv]
    se = res.std_errors[iv]
    p = res.pvalues[iv]
    ci = res.conf_int().loc[iv]
    rows.append(
        {
            "Model": label,
            "DV": dv,
            "IV": iv,
            "beta": round(b, 3),
            "SE": round(se, 3),
            "p_value": round(p, 4),
            "CI_lo": round(ci["lower"], 2),
            "CI_hi": round(ci["upper"], 2),
            "N": int(res.nobs),
            "R2_within": round(res.rsquared, 4),
        }
    )
    print(
        f"{label}\n  beta={b:.3f}, SE={se:.3f}, p={p:.4f}, "
        f"CI=[{ci['lower']:.2f}, {ci['upper']:.2f}], N={int(res.nobs)}"
    )

out = pd.DataFrame(rows)
out.to_csv(DATA_RESULTS / "lagged_models_results.csv", index=False)
print(f"\nSaved {DATA_RESULTS / 'lagged_models_results.csv'}")

