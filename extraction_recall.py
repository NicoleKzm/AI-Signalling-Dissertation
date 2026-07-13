"""
extraction_recall.py

Computes the false-negative rate of classify.py's extraction screen from
the hand-coded extraction_recall_sample.csv (60 non-extracted chunks,
stratified: 20 with Tier 2 present/Tier 1 absent + 40 baseline), then
extrapolates to a stratum-weighted, corpus-wide recall estimate with
Wilson 95% CIs.

YES rows (human judged the chunk genuinely AI-relevant despite never being
extracted) are false negatives of the extraction rule.

INFERENCE FLAGGED: the originating task spec cut off after "separately
for" on the second stratum. Read as stratum (b) = the 40 baseline chunks
(contains_tier2 & ~contains_tier1 is False), the natural complement to
stratum (a). Confirm before trusting stratum (b)'s numbers if that's not
what was intended.

Does not modify extraction_recall_sample.csv, classify.py, or any other
file -- read-only.
"""
import sys

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

SAMPLE_FILE = 'extraction_recall_sample.csv'
VALID_LABELS = {'YES', 'NO'}

# Corpus-wide population sizes (from extraction_recall_sample.py's full-
# corpus scan): total extracted (Tier 1 present, survived dedup), and the
# population each stratum was drawn from.
N_EXTRACTED = 514
N_POP_A = 1527    # Tier 2 present, Tier 1 absent
N_POP_B = 220996  # neither tier present

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Read and normalise human_label_ai_relevant
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv(SAMPLE_FILE)
print(f"Loaded {SAMPLE_FILE}: {len(df)} rows")

raw = df['human_label_ai_relevant']
normalized = raw.astype(str).str.strip().str.upper()
# pandas reads a truly empty cell as float NaN -> str(NaN) == 'nan'
normalized = normalized.where(raw.notna() & (raw.astype(str).str.strip() != ''), other=pd.NA)

unparseable_mask = normalized.notna() & ~normalized.isin(VALID_LABELS)
blank_mask = normalized.isna()

if unparseable_mask.any():
    bad = df[unparseable_mask]
    print("\n" + "!" * 90)
    print(f"STOPPING: {len(bad)} row(s) have a human_label_ai_relevant value that is "
          f"neither YES nor NO after stripping whitespace and uppercasing:")
    for _, row in bad.iterrows():
        print(f"  {row['chunk_id']}: raw value = {row['human_label_ai_relevant']!r}")
    print("Not proceeding. Fix these values and re-run.")
    print("!" * 90)
    sys.exit(1)

df['label_clean'] = normalized

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Report counts
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 2: Label counts")
print("=" * 90)
total = len(df)
n_yes = int((df['label_clean'] == 'YES').sum())
n_no = int((df['label_clean'] == 'NO').sum())
n_blank = int(blank_mask.sum())

print(f"Total rows: {total}")
print(f"YES: {n_yes}")
print(f"NO: {n_no}")
print(f"Blank (unlabeled): {n_blank}")

if n_blank > 0:
    print("\nBlank rows:")
    print(df.loc[blank_mask, ['chunk_id', 'firm', 'year', 'page']].to_string(index=False))
    print("\nNOTE: blank rows are EXCLUDED from the false-negative rate calculations below "
          "(denominator = labeled rows only), not counted as NO.")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — False-negative rate: overall + by stratum
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 3: False-negative rate (YES = false negative of the extraction screen)")
print("=" * 90)

labeled = df[~blank_mask].copy()

# Stratum (a): Tier 2 present, Tier 1 absent (the oversampled category)
stratum_a = labeled[labeled['contains_tier2'] & ~labeled['contains_tier1']]
# Stratum (b): everything else in the labeled sample (the 40-chunk baseline
# draw) -- INFERRED, see module docstring.
stratum_b = labeled[~(labeled['contains_tier2'] & ~labeled['contains_tier1'])]


def fn_rate(subset, label):
    n = len(subset)
    if n == 0:
        print(f"{label}: N=0, rate undefined")
        return
    n_fn = int((subset['label_clean'] == 'YES').sum())
    rate = n_fn / n * 100
    print(f"{label}: {n_fn}/{n} = {rate:.2f}% false-negative rate")


print("\nOverall (all labeled rows):")
fn_rate(labeled, "  Overall")

print("\nStratum (a): Tier 2 present, Tier 1 absent (n=20 drawn):")
fn_rate(stratum_a, "  Stratum (a)")

print("\nStratum (b): baseline chunks (n=40 drawn) [INFERRED stratum definition -- confirm]:")
fn_rate(stratum_b, "  Stratum (b)")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Stratum-weighted corpus-wide recall extrapolation, with
# Wilson 95% CIs on each stratum's rate, propagated to recall.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 4: Stratum-weighted corpus-wide recall extrapolation")
print("=" * 90)

n_a, fn_a = len(stratum_a), int((stratum_a['label_clean'] == 'YES').sum())
n_b, fn_b = len(stratum_b), int((stratum_b['label_clean'] == 'YES').sum())
p_a = fn_a / n_a
p_b = fn_b / n_b

missed_a = N_POP_A * p_a
missed_b = N_POP_B * p_b
total_missed = missed_a + missed_b
recall_point = N_EXTRACTED / (N_EXTRACTED + total_missed)

print(f"Stratum (a): {fn_a}/{n_a} = {p_a*100:.2f}%, population={N_POP_A} "
      f"-> est. missed = {missed_a:.2f}")
print(f"Stratum (b): {fn_b}/{n_b} = {p_b*100:.2f}%, population={N_POP_B} "
      f"-> est. missed = {missed_b:.2f}")
print(f"Total estimated missed: {total_missed:.2f}")
print(f"Recall = {N_EXTRACTED} / ({N_EXTRACTED} + {total_missed:.2f}) = "
      f"{recall_point:.4f} = {recall_point*100:.2f}%")

ci_a_lo, ci_a_hi = proportion_confint(fn_a, n_a, alpha=0.05, method='wilson')
ci_b_lo, ci_b_hi = proportion_confint(fn_b, n_b, alpha=0.05, method='wilson')
print(f"\nStratum (a) Wilson 95% CI: [{ci_a_lo*100:.2f}%, {ci_a_hi*100:.2f}%]")
print(f"Stratum (b) Wilson 95% CI: [{ci_b_lo*100:.2f}%, {ci_b_hi*100:.2f}%]")


def recall_from_rates(pa, pb):
    missed = N_POP_A * pa + N_POP_B * pb
    return N_EXTRACTED / (N_EXTRACTED + missed), missed


recall_low, missed_low = recall_from_rates(ci_a_hi, ci_b_hi)
recall_high, missed_high = recall_from_rates(ci_a_lo, ci_b_lo)
print(f"\nRecall propagated range (stratum CI extremes):")
print(f"  Low-recall bound (both strata at CI upper, missed={missed_low:.1f}): "
      f"{recall_low*100:.2f}%")
print(f"  Point estimate: {recall_point*100:.2f}%")
print(f"  High-recall bound (both strata at CI lower, missed={missed_high:.1f}): "
      f"{recall_high*100:.2f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 4b — Zero-count case: if stratum (b) has 0 observed FN, what's the
# rule-of-three / Wilson upper bound on its true rate, and what recall
# does that floor imply?
# ══════════════════════════════════════════════════════════════════
if fn_b == 0:
    print("\n" + "-" * 90)
    print("Stratum (b) has ZERO observed false negatives (n=40).")
    rule_of_three = 3 / n_b
    _, wilson_upper_zero = proportion_confint(0, n_b, alpha=0.05, method='wilson')
    print(f"  Rule-of-three approximate 95% upper bound: 3/{n_b} = {rule_of_three*100:.2f}%")
    print(f"  Exact Wilson 95% upper bound: {wilson_upper_zero*100:.2f}%")
    missed_b_bound = N_POP_B * wilson_upper_zero
    missed_total_bound = missed_a + missed_b_bound
    recall_floor = N_EXTRACTED / (N_EXTRACTED + missed_total_bound)
    print(f"  Implied est. missed in stratum (b) at that bound: "
          f"{N_POP_B} x {wilson_upper_zero*100:.2f}% = {missed_b_bound:.1f}")
    print(f"  Implied recall FLOOR (stratum (a) at its point estimate, "
          f"stratum (b) at its Wilson upper bound): "
          f"{N_EXTRACTED}/({N_EXTRACTED}+{missed_a:.1f}+{missed_b_bound:.1f}) = "
          f"{recall_floor*100:.2f}%")
    print("  (i.e. even in the most pessimistic case consistent with observing zero")
    print("   false negatives in 40 draws, recall cannot be estimated below this floor")
    print("   from this sample alone.)")

print("\nDone.")
