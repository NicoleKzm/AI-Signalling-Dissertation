# DEPRECATED -- superseded by tost_mde_v2.py. Hardcodes stale pre-refactor beta/SE/N
# values (H2/H3 n=55, not the current n=54). Also prints those stale numbers to stdout
# at import time, so importing this module leaks them into any script's output -- kept
# only for audit trail; do not import from it or cite its numbers.
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
