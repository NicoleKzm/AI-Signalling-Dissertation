# ══════════════════════════════════════════════════════════════════
# DEPRECATED -- superseded by tost_mde_v2.py.
#
# The MODELS dict below hardcodes pre-refactor beta/SE/N values (notably
# H2 and H3 "primary": n=55, from before the lag-1 signal + Zalando-2025
# exclusion refactor) that do NOT match the current primary specifications
# in regression_results.csv (H2/H3 primary: n=54). Do not read numbers
# from this file's output as current results.
#
# This file also executes a print block at module import time (below),
# which leaks these stale numbers into the terminal of any script that
# merely imports a name from this module -- this caused a phantom
# TOST discrepancy (tost_mde_v2.py's correct p_lower=0.0334 vs. this
# file's stale p_lower=0.0488) that took a day to diagnose. See the
# diagnosis in the project history / commit log for the full trace.
#
# Kept in place, unmodified otherwise, as part of the audit trail --
# do NOT delete. Do NOT import from this file in new code; use
# tost_mde_v2.py (dynamic bounds + df from panel_dataset.csv /
# regression_results.csv) instead.
# ══════════════════════════════════════════════════════════════════
import numpy as np
from scipy import stats

MODELS = {
    "H1 (contemporaneous, primary)": {"beta": -22.624, "se": 35.090, "n": 69},
    "H2 (lag-1, primary)":           {"beta": -1.298,  "se": 8.137,  "n": 55},
    "H3 (lag-1, primary)":           {"beta": -0.832,  "se": 1.559,  "n": 55},
}

EQUIVALENCE_BOUNDS = {
    "H1 (contemporaneous, primary)": 0.2 * 52.56,
    "H2 (lag-1, primary)":           0.2 * 20.19,
    "H3 (lag-1, primary)":           0.2 * 17.31,
}

POWER = 0.80
ALPHA = 0.05

def minimum_detectable_effect(se, n, power=POWER, alpha=ALPHA):
    df = n - 3
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    t_power = stats.t.ppf(power, df)
    return se * (t_alpha + t_power)

def tost_equivalence(beta, se, n, bound, alpha=ALPHA):
    df = n - 3
    t_lower = (beta - (-bound)) / se
    t_upper = (bound - beta) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = 1 - stats.t.cdf(t_upper, df)
    equivalent = (p_lower < alpha) and (p_upper < alpha)
    return p_lower, p_upper, equivalent

print(f"{'Model':35s} {'MDE':>10s} {'Bound':>10s} {'p_lo':>8s} {'p_hi':>8s} {'Equivalent?':>12s}")
print("-" * 90)
for label, m in MODELS.items():
    mde = minimum_detectable_effect(m["se"], m["n"])
    bound = EQUIVALENCE_BOUNDS[label]
    p_lo, p_hi, equiv = tost_equivalence(m["beta"], m["se"], m["n"], bound)
    print(f"{label:35s} {mde:10.2f} {bound:10.2f} {p_lo:8.4f} {p_hi:8.4f} {str(equiv):>12s}")
