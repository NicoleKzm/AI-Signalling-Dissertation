"""
make_regression_figures.py

Generates Figure 4.4 (primary coefficient estimates, H1/H2/H3) and Figure 4.5
(H3 equivalence plot) from the CANONICAL regression output.

Reads ONLY regression_results.csv and tost_mde_results.csv -- does not touch
panel_dataset.csv, regression_clean.py, or classify.py. Does not touch the
existing make_figures.py (separate, unrelated firm-stats/leave-one-out script)
or its output/ directory. Writes PNGs only to figures/.
"""
import re
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

base = Path(__file__).resolve().parent
fig_dir = base / "figures"
fig_dir.mkdir(exist_ok=True)

# ── Style: navy/slate, serif, no chartjunk ──────────────────────────
NAVY = "#1A3A63"
SLATE = "#5B6B7C"
SLATE_LIGHT = "#5B6B7C"  # used at low alpha for shading/gridlines
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
# Read inputs (read-only)
# ══════════════════════════════════════════════════════════════════
reg = pd.read_csv(base / "regression_results.csv")
primary = reg[reg["Type"] == "Primary"].set_index("Model")
tost = pd.read_csv(base / "tost_mde_results.csv").set_index("H")

HYPS = [
    ("H1_Stock_Price", "H1", "Stock Return\n(contemporaneous)"),
    ("H2_Revenue_Growth", "H2", "Revenue Growth\n(lag-1)"),
    ("H3_Gross_Margin", "H3", "Gross Margin\n(lag-1)"),
]

print("=" * 80)
print("VALUES PLOTTED -- FIGURE 4.4 (primary estimates, 95% CI from regression_results.csv)")
print("=" * 80)
panel_data = []
for model_key, h_label, panel_title in HYPS:
    row = primary.loc[model_key]
    beta = float(row["Coefficient"])
    ci_lo = float(row["CI_lower"])
    ci_hi = float(row["CI_upper"])
    n = int(row["N_obs"])
    panel_data.append({"h": h_label, "title": panel_title, "beta": beta,
                        "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n})
    print(f"{h_label}: beta={beta:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}], N={n}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4.4 -- three stacked panels, each with its own x-scale
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5))

for ax, d in zip(axes, panel_data):
    ax.axvline(0, color=ZERO_LINE_GRAY, linestyle="--", linewidth=1.0, zorder=1)

    ax.hlines(y=0, xmin=d["ci_lo"], xmax=d["ci_hi"], color=NAVY, linewidth=2.0, zorder=2)
    # whisker caps
    cap_h = 0.12
    ax.vlines([d["ci_lo"], d["ci_hi"]], -cap_h, cap_h, color=NAVY, linewidth=1.4, zorder=2)
    ax.plot(d["beta"], 0, marker="o", markersize=9, markerfacecolor=NAVY,
             markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    span = d["ci_hi"] - d["ci_lo"]
    pad = span * 0.35 if span > 0 else 5
    ax.set_xlim(d["ci_lo"] - pad, d["ci_hi"] + pad)
    ax.set_ylim(-1, 1)
    ax.set_yticks([0])
    ax.set_yticklabels([d["title"]], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Coefficient (percentage points)", fontsize=9.5, color=SLATE)

    label = f"β = {d['beta']:.2f}   N = {d['n']}"
    y_frac = 0.82
    ax.text(0.02, y_frac, label, transform=ax.transAxes, fontsize=9.5,
            color=INK, va="top", ha="left")

plt.tight_layout(h_pad=2.2)
out_44 = fig_dir / "fig_4_4_primary_estimates.png"
plt.savefig(out_44, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_44}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4.5 -- H3 equivalence plot
# ══════════════════════════════════════════════════════════════════
h3 = tost.loc["H3"]
h3_beta = float(h3["beta"])
h3_bound = float(h3["bound"])
h3_p_lower = float(h3["p_lower"])
h3_p_upper = float(h3["p_upper"])

ci_match = re.match(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", str(h3["90%_CI"]))
h3_ci_lo, h3_ci_hi = float(ci_match.group(1)), float(ci_match.group(2))

print("\n" + "=" * 80)
print("VALUES PLOTTED -- FIGURE 4.5 (H3 equivalence, from tost_mde_results.csv)")
print("=" * 80)
print(f"H3 beta = {h3_beta:.4f}")
print(f"H3 90% CI = [{h3_ci_lo:.4f}, {h3_ci_hi:.4f}]")
print(f"Equivalence bound = +/-{h3_bound:.3f}")
print(f"TOST: p_lower = {h3_p_lower:.4f}, p_upper = {h3_p_upper:.4f}")
print(f"CI entirely within bounds: {bool(h3['CI_within_bounds'])}")

fig2, ax2 = plt.subplots(figsize=(7.5, 3.3))

ax2.axvspan(-h3_bound, h3_bound, color=SLATE_LIGHT, alpha=0.12, zorder=0)

ax2.axvline(-h3_bound, color=SLATE, linestyle="--", linewidth=1.4, zorder=1)
ax2.axvline(h3_bound, color=SLATE, linestyle="--", linewidth=1.4, zorder=1)

ax2.axvline(0, color=ZERO_LINE_GRAY, linestyle=":", linewidth=1.2, zorder=1)

ax2.hlines(y=0, xmin=h3_ci_lo, xmax=h3_ci_hi, color=NAVY, linewidth=2.4, zorder=3)
cap_h = 0.10
ax2.vlines([h3_ci_lo, h3_ci_hi], -cap_h, cap_h, color=NAVY, linewidth=1.6, zorder=3)
ax2.plot(h3_beta, 0, marker="o", markersize=11, markerfacecolor=NAVY,
          markeredgecolor="white", markeredgewidth=1.4, zorder=4)

axis_pad = h3_bound * 0.45
ax2.set_xlim(-h3_bound - axis_pad, h3_bound + axis_pad)
ax2.set_ylim(-1, 1.55)
ax2.set_yticks([])
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlabel("H3 coefficient: Gross Margin, lag-1 (percentage points)", fontsize=9.5, color=SLATE)

# Bound labels sit OUTSIDE their line with a small data-unit gap (never
# touching the stroke), pinned to the top of the axes, well clear of the
# annotation block below.
label_gap = h3_bound * 0.06
ax2.text(-h3_bound - label_gap, 1.42, "pre-specified equivalence\nbound (±0.2 SD)",
          ha="right", va="top", fontsize=8.3, color=SLATE)
ax2.text(h3_bound + label_gap, 1.42, "pre-specified equivalence\nbound (±0.2 SD)",
          ha="left", va="top", fontsize=8.3, color=SLATE)

# Annotation block: anchored in DATA coordinates just right of the left
# bound line (not axes-fraction), so it can never cross that line
# regardless of final axis limits; vertically clear of the bound labels
# above and the whisker/marker below.
ann_x = -h3_bound + label_gap
ax2.text(ann_x, 0.95, f"β = {h3_beta:.4f}   90% CI = [{h3_ci_lo:.2f}, {h3_ci_hi:.2f}]",
          fontsize=9.5, color=INK, va="top", ha="left")
ax2.text(ann_x, 0.72, f"TOST: p_lower = {h3_p_lower:.3f}, p_upper = {h3_p_upper:.3f}",
          fontsize=9.5, color=INK, va="top", ha="left")

plt.tight_layout()
out_45 = fig_dir / "fig_4_5_h3_equivalence.png"
plt.savefig(out_45, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"\nSaved {out_45}")

print(f"\nFont used: {serif_face}")
