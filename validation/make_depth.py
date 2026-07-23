import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

base = Path(__file__).resolve().parent
out = base / "output"
out.mkdir(exist_ok=True)
RESULTS = base.parent / "results"

df = pd.read_csv(base.parent / "data" / "panel_dataset.csv")
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

for c in ["mean_signal_score", "revenue_eur", "revenue"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "log_revenue" not in df.columns:
    if "revenue_eur" in df.columns:
        df["log_revenue"] = np.log(df["revenue_eur"])
    elif "revenue" in df.columns:
        df["log_revenue"] = np.log(df["revenue"])

map_score = {"No_AI_Disclosure": 0, "Symbolic": 1, "Transitional": 2, "Substantive": 3}
df["disclosure_depth"] = df["modal_classification"].map(map_score)
df["is_substantive"] = (df["modal_classification"] == "Substantive").astype(int)
df["is_transitional_or_better"] = df["modal_classification"].isin(["Transitional", "Substantive"]).astype(int)
df["is_symbolic_or_better"] = df["modal_classification"].isin(["Symbolic", "Transitional", "Substantive"]).astype(int)

df.to_csv(RESULTS / "disclosure_depth_dataset.csv", index=False)

models = [
    "disclosure_depth",
    "is_substantive",
    "is_transitional_or_better",
    "is_symbolic_or_better"
]

rows = []
for y in models:
    d = df[[y, "year", "log_revenue"]].dropna().copy()
    mod = smf.ols(f"{y} ~ C(year) + log_revenue", data=d).fit(cov_type="HC1")
    rows.append({
        "dependent": y,
        "coef_log_revenue": mod.params.get("log_revenue", np.nan),
        "se_log_revenue": mod.bse.get("log_revenue", np.nan),
        "pvalue_log_revenue": mod.pvalues.get("log_revenue", np.nan),
        "n": int(mod.nobs),
        "r2": mod.rsquared
    })

res = pd.DataFrame(rows)
res.to_csv(RESULTS / "depth_models_summary.csv", index=False)

print("DONE")
print(sorted([p.name for p in RESULTS.iterdir() if p.name.startswith("depth") or p.name == "disclosure_depth_dataset.csv"]))
