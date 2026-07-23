"""
make_regression_figures.py

Generates Figure 4.4 -- a single forest plot of the four primary/reported
coefficient estimates (H1 lag-1, H1 contemporaneous, H2 lag-1, H3 lag-1),
each with its 95% CI against a zero reference line.

Reads ONLY regression_results.csv. Does not recompute, re-estimate, or
hardcode any beta/SE/CI -- every plotted value is read directly from that
CSV's Coefficient/CI_lower/CI_upper columns. Does not touch panel_dataset.csv,
regression_clean.py, classify.py, or the existing make_figures.py (separate,
unrelated firm-stats/leave-one-out script writing to output/).

Writes only to figures/ (created via os.makedirs if missing, so this works
on a fresh clone).

NOTE: this script previously also generated Figure 4.5 (H3 equivalence),
reading tost_mde_results.csv. That code is removed here, not regenerated --
its bound label read "pre-specified equivalence bound", which contradicts
the TOST-provenance correction already recorded in LOCKED_NUMBERS.md (the
bound is NOT pre-specified). Re-add Figure 4.5 as its own task once that
label is fixed, rather than re-emit the stale wording today.
"""
import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style: navy/slate, serif, no chartjunk ──────────────────────────
NAVY = "#1A3A63"
SLATE = "#5B6B7C"
INK = "#2B2B2B"
ZERO_LINE_GRAY = "#8A8A85"

available_fonts = {f.name for f in fm.fontManager.ttflist}
serif_face = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
matplotlib.rcParams.update({
    "font.family": serif_face,
    "font.size": 11,
    "axes.edgecolor": SLATE,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

# ══════════════════════════════════════════════════════════════════
# Read regression_results.csv (read-only) -- no computation performed
# ══════════════════════════════════════════════════════════════════
reg = pd.read_csv("regression_results.csv")

# (Model, Type) identifies each row uniquely in this CSV -- Model alone is
# not unique, since H1_Stock_Price appears under several Types.
POINTS = [
    ("H1_Stock_Price", "Primary: lag-1 (matches H2/H3 primary spec)", "H1 (lag-1)"),
    ("H1_Stock_Price", "Primary", "H1 (contemporaneous)"),
    ("H2_Revenue_Growth", "Primary", "H2 (lag-1)"),
    ("H3_Gross_Margin", "Primary", "H3 (lag-1)"),
]

print("=" * 80)
print("VALUES PLOTTED -- FIGURE 4.4 (read directly from regression_results.csv, "
      "not recomputed)")
print("=" * 80)
plot_data = []
for model, type_, label in POINTS:
    row = reg[(reg["Model"] == model) & (reg["Type"] == type_)]
    assert len(row) == 1, f"{model} / {type_}: expected exactly 1 row, got {len(row)}"
    row = row.iloc[0]
    beta = float(row["Coefficient"])
    ci_lo = float(row["CI_lower"])
    ci_hi = float(row["CI_upper"])
    n = int(row["N_obs"])
    p = float(row["P_value"])
    plot_data.append({"label": label, "beta": beta, "ci_lo": ci_lo, "ci_hi": ci_hi,
                       "n": n, "p": p})
    print(f"{label}: beta={beta:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}], N={n}, p={p:.4f}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4.4 -- single forest plot, all four points on one axis
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 4.2))

y_positions = list(range(len(plot_data)))[::-1]  # top-to-bottom in list order

ax.axvline(0, color=ZERO_LINE_GRAY, linestyle="--", linewidth=1.1, zorder=1)

for y, d in zip(y_positions, plot_data):
    ax.hlines(y=y, xmin=d["ci_lo"], xmax=d["ci_hi"], color=NAVY, linewidth=2.2, zorder=2)
    cap_h = 0.14
    ax.vlines([d["ci_lo"], d["ci_hi"]], y - cap_h, y + cap_h, color=NAVY,
              linewidth=1.5, zorder=2)
    ax.plot(d["beta"], y, marker="o", markersize=10, markerfacecolor=NAVY,
             markeredgecolor="white", markeredgewidth=1.3, zorder=3)
    ax.text(d["ci_hi"] + (max(abs(d["ci_lo"]), abs(d["ci_hi"])) * 0.04), y,
            f"β = {d['beta']:.3f}, N = {d['n']}", va="center", ha="left",
            fontsize=9, color=INK)

ax.set_yticks(y_positions)
ax.set_yticklabels([d["label"] for d in plot_data], fontsize=10.5)
ax.set_ylim(min(y_positions) - 0.7, max(y_positions) + 0.7)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Coefficient on signal score (95% CI)", fontsize=10, color=SLATE)
ax.set_title("Figure 4.4 — Primary regression estimates", fontsize=11.5, color=INK)

fig.tight_layout()
out_path = os.path.join(FIG_DIR, "fig_4_4_primary_estimates.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_path}")
print(f"Font used: {serif_face}")
