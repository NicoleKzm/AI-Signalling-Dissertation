import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf

base = Path(__file__).resolve().parent
out = base / "output"
out.mkdir(exist_ok=True)
print("Saving files to:", out.resolve())

df = pd.read_csv(base / "panel_dataset.csv")
df.columns = [c.strip() for c in df.columns]

rename_map = {
    "Stock_Price_Movement_%": "stock_return",
    "Revenue_Growth_%": "revenue_growth",
    "Gross_Margin_%": "gross_margin",
    "Revenue_EUR": "revenue_eur",
    "Revenue": "revenue",
    "Ticker": "ticker"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

for c in ["mean_signal_score", "stock_return", "revenue_growth", "gross_margin", "revenue_eur", "revenue"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "log_revenue" not in df.columns:
    if "revenue_eur" in df.columns:
        df["log_revenue"] = np.log(df["revenue_eur"])
    elif "revenue" in df.columns:
        df["log_revenue"] = np.log(df["revenue"])

firm_stats = df.groupby("firm").agg(
    n_obs=("mean_signal_score", "count"),
    mean_signal=("mean_signal_score", "mean"),
    sd_signal=("mean_signal_score", "std"),
    min_signal=("mean_signal_score", "min"),
    max_signal=("mean_signal_score", "max"),
    zero_years=("mean_signal_score", lambda x: (x == 0).sum())
).reset_index().sort_values("mean_signal", ascending=False)
firm_stats["sd_signal"] = firm_stats["sd_signal"].fillna(0)
firm_stats.to_csv(out / "firm_level_signal_stats.csv", index=False)

plt.figure(figsize=(10, 6))
plt.barh(firm_stats["firm"], firm_stats["mean_signal"], color="steelblue")
plt.xlabel("Mean signal score (2021-2025)")
plt.ylabel("")
plt.title("Firm-level mean AI signalling score")
plt.tight_layout()
plt.savefig(out / "figure_firm_mean_signal.png", dpi=300)
plt.savefig(out / "figure_firm_mean_signal.pdf")
plt.close()

class_counts = pd.DataFrame({
    "classification": ["No_AI_Disclosure", "Symbolic", "Transitional", "Substantive"],
    "firm_year_count": [
        int((df["modal_classification"] == "No_AI_Disclosure").sum()),
        int((df["modal_classification"] == "Symbolic").sum()),
        int((df["modal_classification"] == "Transitional").sum()),
        int((df["modal_classification"] == "Substantive").sum())
    ]
})
class_counts["share"] = class_counts["firm_year_count"] / class_counts["firm_year_count"].sum()
class_counts.to_csv(out / "firm_year_classification_counts.csv", index=False)

year_counts = df.groupby(["year", "modal_classification"]).size().reset_index(name="n")
pivot = year_counts.pivot(index="year", columns="modal_classification", values="n").fillna(0)
pivot.to_csv(out / "yearly_classification_counts.csv")
cols = [c for c in ["No_AI_Disclosure", "Symbolic", "Transitional", "Substantive"] if c in pivot.columns]
if cols:
    ax = pivot[cols].plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20c")
    ax.set_xlabel("Reporting year")
    ax.set_ylabel("Firm-years")
    ax.set_title("Yearly distribution of AI disclosure classifications")
    plt.tight_layout()
    plt.savefig(out / "yearly_classification_stacked.png", dpi=300)
    plt.savefig(out / "yearly_classification_stacked.pdf")
    plt.close()

outcomes = {
    "stock_return": "stock_return",
    "revenue_growth": "revenue_growth",
    "gross_margin": "gross_margin"
}

main_rows = []
for h, y in outcomes.items():
    if y not in df.columns:
        continue
    d = df[["firm", "year", "mean_signal_score", "log_revenue", y]].dropna().copy()
    d = pd.get_dummies(d, columns=["firm", "year"], drop_first=True)
    d.columns = [c.replace(" ", "_").replace("-", "_") for c in d.columns]
    rhs = "mean_signal_score + log_revenue"
    dummy_cols = [c for c in d.columns if c.startswith("firm_") or c.startswith("year_")]
    if dummy_cols:
        rhs += " + " + " + ".join(dummy_cols)
    mod = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="cluster", cov_kwds={"groups": df.loc[d.index, "firm"]})
    ci = mod.conf_int().loc["mean_signal_score"]
    main_rows.append({
        "hypothesis": h,
        "outcome": y,
        "coef": mod.params.get("mean_signal_score", np.nan),
        "se": mod.bse.get("mean_signal_score", np.nan),
        "pvalue": mod.pvalues.get("mean_signal_score", np.nan),
        "n": int(mod.nobs),
        "ci_low": ci.iloc[0],
        "ci_high": ci.iloc[1],
    })

main_reg = pd.DataFrame(main_rows)
main_reg.to_csv(out / "main_regression_signal.csv", index=False)

loo_rows = []
for left_out in df["firm"].dropna().unique():
    sub = df[df["firm"] != left_out].copy()
    for h, y in outcomes.items():
        if y not in sub.columns:
            continue
        d = sub[["firm", "year", "mean_signal_score", "log_revenue", y]].dropna().copy()
        d = pd.get_dummies(d, columns=["firm", "year"], drop_first=True)
        d.columns = [c.replace(" ", "_").replace("-", "_") for c in d.columns]
        rhs = "mean_signal_score + log_revenue"
        dummy_cols = [c for c in d.columns if c.startswith("firm_") or c.startswith("year_")]
        if dummy_cols:
            rhs += " + " + " + ".join(dummy_cols)
        mod = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="cluster", cov_kwds={"groups": sub.loc[d.index, "firm"]})
        loo_rows.append({
            "left_out": left_out,
            "hypothesis": h,
            "outcome": y,
            "coef": mod.params.get("mean_signal_score", np.nan),
            "se": mod.bse.get("mean_signal_score", np.nan),
            "pvalue": mod.pvalues.get("mean_signal_score", np.nan),
            "n": int(mod.nobs)
        })

loo_df = pd.DataFrame(loo_rows)
loo_df.to_csv(out / "leave_one_out_results.csv", index=False)

summary_rows = []
for h in main_reg["hypothesis"].unique():
    vals = loo_df.loc[loo_df["hypothesis"] == h, "coef"].dropna()
    maincoef = main_reg.loc[main_reg["hypothesis"] == h, "coef"].iloc[0]
    summary_rows.append({
        "hypothesis": h,
        "main_coef": maincoef,
        "min_coef": vals.min(),
        "max_coef": vals.max(),
        "coef_range": vals.max() - vals.min(),
        "sign_change": bool((vals > 0).any() and (vals < 0).any())
    })

loo_summary = pd.DataFrame(summary_rows)
loo_summary.to_csv(out / "loo_summary.csv", index=False)

if not loo_df[loo_df["hypothesis"] == "stock_return"].empty:
    s = loo_df[loo_df["hypothesis"] == "stock_return"].sort_values("coef")
    plt.figure(figsize=(10, 5))
    plt.plot(s["left_out"], s["coef"], marker="o")
    maincoef = main_reg.loc[main_reg["hypothesis"] == "stock_return", "coef"].iloc[0]
    plt.axhline(maincoef, color="red", linestyle="--", label="Main coef")
    plt.xticks(rotation=90)
    plt.ylabel("Signalling coefficient")
    plt.title("Leave-one-firm-out coefficients: stock_return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loo_stock_return_coeffs.png", dpi=300)
    plt.savefig(out / "loo_stock_return_coeffs.pdf")
    plt.close()

print("DONE")
print(sorted([p.name for p in out.iterdir()]))
print(main_reg.to_string(index=False))
print(loo_summary.to_string(index=False))