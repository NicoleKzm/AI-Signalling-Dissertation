"""
make_figures_4_4_5.py

Regenerates Figures 4.4 and 4.5 from the CANONICAL, already-locked estimates
(regression_results.csv primary rows + LOCKED_NUMBERS.md's H1 lag-1 CI, which
is not itself in regression_results.csv). No computation performed here --
every number plotted is read from the files of record and printed alongside
the figure for cross-checking. Two complementary standard presentations of
the same four estimates: Fig 4.4 = horizontal coefficient/forest plot,
Fig 4.5 = vertical bar chart with 95% CI error bars. If a different content
split between the two figures was intended, say so and this can be redrawn.

House style: Times New Roman, grey-blue palette (#4A6274 primary,
#A3B1BA secondary), matching the earlier viva-dashboard figures in this
project.

Does not modify classify.py, regression_results.csv, LOCKED_NUMBERS.md, or
any other file. Writes only to output/fig_4_4.png/.pdf and
output/fig_4_5.png/.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

PRIMARY = "#4A6274"
SECONDARY = "#A3B1BA"

# Locked estimates -- hardcoded from the files of record, not recomputed:
#   H1 lag-1: LOCKED_NUMBERS.md line 5 (beta, SE) + line 86 (conventional 95% CI)
#   H1 contemp, H2 lag-1, H3 lag-1: regression_results.csv, Primary rows
ESTIMATES = [
    {"label": "H1 (lag-1, primary)",       "beta": -11.7093, "ci_lo": -70.6743, "ci_hi": 47.2557},
    {"label": "H1 (contemporaneous)",      "beta": -22.6239, "ci_lo": -93.1406, "ci_hi": 47.8927},
    {"label": "H2 (lag-1, primary)",       "beta": -5.0803,  "ci_lo": -22.1680, "ci_hi": 12.0074},
    {"label": "H3 (lag-1, primary)",       "beta": -0.6904,  "ci_lo": -3.6636,  "ci_hi": 2.2828},
]

print("Locked estimates being plotted (source: regression_results.csv Primary rows, "
      "LOCKED_NUMBERS.md line 5/86 for H1 lag-1 CI):")
for e in ESTIMATES:
    print(f"  {e['label']}: beta={e['beta']}, 95% CI=[{e['ci_lo']}, {e['ci_hi']}]")

# Try Times New Roman; fall back with a warning if unavailable (does not affect
# plotted values, only font rendering).
available_fonts = {f.name for f in fm.fontManager.ttflist}
if "Times New Roman" in available_fonts:
    plt.rcParams["font.family"] = "Times New Roman"
else:
    print("\nWARNING: 'Times New Roman' not found in matplotlib's font list on this "
          "machine -- falling back to matplotlib's default serif font. Coefficients "
          "and CIs plotted are unaffected; only the typeface differs. Install the "
          "font and rerun this script to get exact Times New Roman rendering.")
    plt.rcParams["font.family"] = "serif"

out = Path("output")
out.mkdir(exist_ok=True)

# ── Figure 4.4: horizontal coefficient / forest plot ────────────────
fig, ax = plt.subplots(figsize=(7.5, 4))
y_pos = range(len(ESTIMATES))
labels = [e["label"] for e in ESTIMATES]
betas = [e["beta"] for e in ESTIMATES]
err_lo = [e["beta"] - e["ci_lo"] for e in ESTIMATES]
err_hi = [e["ci_hi"] - e["beta"] for e in ESTIMATES]

ax.errorbar(betas, list(y_pos), xerr=[err_lo, err_hi], fmt="o", color=PRIMARY,
            ecolor=SECONDARY, elinewidth=2.5, capsize=5, markersize=8, markeredgecolor=PRIMARY)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Coefficient on signal score (95% CI)")
ax.set_title("Figure 4.4 — Primary regression coefficients with 95% CIs")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(out / "fig_4_4.png", dpi=300)
fig.savefig(out / "fig_4_4.pdf")
plt.close(fig)
print("\nSaved output/fig_4_4.png / .pdf")

# ── Figure 4.5: vertical bar chart with error bars ──────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x_pos = range(len(ESTIMATES))
colors = [PRIMARY, SECONDARY, PRIMARY, SECONDARY]
ax.bar(list(x_pos), betas, yerr=[err_lo, err_hi], color=colors, capsize=6,
       error_kw={"ecolor": "#2F3E48", "elinewidth": 1.5})
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(list(x_pos))
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Coefficient on signal score")
ax.set_title("Figure 4.5 — Primary regression coefficients (bar, 95% CI)")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(out / "fig_4_5.png", dpi=300)
fig.savefig(out / "fig_4_5.pdf")
plt.close(fig)
print("Saved output/fig_4_5.png / .pdf")
