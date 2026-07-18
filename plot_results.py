import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent
out = base / "output"
out.mkdir(exist_ok=True)

firm_stats = pd.read_csv(out / "firm_level_signal_stats.csv")
class_counts = pd.read_csv(out / "firm_year_classification_counts.csv")
year_counts = pd.read_csv(out / "yearly_classification_counts.csv", index_col=0)
main_reg = pd.read_csv(out / "main_regression_signal.csv")
loo_summary = pd.read_csv(out / "loo_summary.csv")

# 1) Firm-year class distribution
plt.figure(figsize=(7, 4.5))
labels = class_counts["classification"].str.replace("_", " ")
colors = ["#9e9e9e", "#5b8db8", "#f0a35e", "#5bb85c"][:len(labels)]
plt.bar(labels, class_counts["firm_year_count"], color=colors)
plt.ylabel("Firm-years")
plt.title("Firm-year distribution of disclosure classes")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(out / "fig_class_counts.png", dpi=300)
plt.savefig(out / "fig_class_counts.pdf")
plt.close()

# 2) Yearly stacked classification
pivot = year_counts.copy()
pivot.index = pivot.index.astype(int)
cols = [c for c in ["No_AI_Disclosure", "Symbolic", "Transitional", "Substantive"] if c in pivot.columns]
colors = ["#9e9e9e", "#5b8db8", "#f0a35e", "#5bb85c"][:len(cols)]
ax = pivot[cols].plot(kind="bar", stacked=True, figsize=(8, 5), color=colors)
ax.set_xlabel("Reporting year")
ax.set_ylabel("Firm-years")
ax.set_title("Yearly classification mix")
plt.tight_layout()
plt.savefig(out / "fig_yearly_classification.png", dpi=300)
plt.savefig(out / "fig_yearly_classification.pdf")
plt.close()

# 3) Main regression coefficients with 95% CI
order = ["stock_return", "revenue_growth", "gross_margin"]
main_reg["hypothesis"] = pd.Categorical(main_reg["hypothesis"], categories=order, ordered=True)
main_reg = main_reg.sort_values("hypothesis")
plt.figure(figsize=(7, 4.5))
plt.errorbar(
    main_reg["hypothesis"],
    main_reg["coef"],
    yerr=[main_reg["coef"] - main_reg["ci_low"], main_reg["ci_high"] - main_reg["coef"]],
    fmt="o",
    capsize=5,
    color="#2c3e50"
)
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("Signal coefficient")
plt.title("Main FE estimates with 95% CI")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(out / "fig_main_regression.png", dpi=300)
plt.savefig(out / "fig_main_regression.pdf")
plt.close()

# 4) Leave-one-out coefficient ranges
plt.figure(figsize=(7, 4.5))
xs = np.arange(len(loo_summary))
plt.errorbar(
    xs,
    loo_summary["main_coef"],
    yerr=[loo_summary["main_coef"] - loo_summary["min_coef"], loo_summary["max_coef"] - loo_summary["main_coef"]],
    fmt="o",
    capsize=5,
    color="#8e44ad"
)
plt.axhline(0, color="black", linewidth=1)
plt.xticks(xs, loo_summary["hypothesis"], rotation=15)
plt.ylabel("Signal coefficient")
plt.title("Leave-one-out coefficient range")
plt.tight_layout()
plt.savefig(out / "fig_loo_range.png", dpi=300)
plt.savefig(out / "fig_loo_range.pdf")
plt.close()

# 5) Ranked firm mean signal
plt.figure(figsize=(10, 6))
plt.barh(firm_stats["firm"], firm_stats["mean_signal"], color="#5b8db8")
plt.xlabel("Mean signal score (2021-2025)")
plt.ylabel("")
plt.title("Firm-level mean AI signalling score")
plt.tight_layout()
plt.savefig(out / "fig_firm_ranked.png", dpi=300)
plt.savefig(out / "fig_firm_ranked.pdf")
plt.close()

print(sorted([p.name for p in out.iterdir() if p.name.startswith("fig_")]))