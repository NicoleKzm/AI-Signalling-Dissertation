"""
Computes inter-coder agreement (unweighted/linear/quadratic Cohen's kappa, raw
agreement, confusion matrix) between the LLM classification and the second human coder
on the 29 valid rows of data/second_coder_sample.csv (1 row excluded as unclassifiable);
read-only, writes nothing.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

TIER_ORDER = {'Symbolic': 0, 'Transitional': 1, 'Substantive': 2}
LABELS = ['Symbolic', 'Transitional', 'Substantive']

df = pd.read_csv('data/second_coder_sample.csv')
print(f"Loaded second_coder_sample.csv: {len(df)} rows")

excluded = df[df['excluded_flag'] == True]
valid = df[df['excluded_flag'] == False].copy()
print(f"Excluded rows: {len(excluded)} -- {excluded['passage_id'].tolist()}")
print(f"Valid rows for kappa: {len(valid)}")

if valid['second_coder_label'].isna().any() or (valid['second_coder_label'] == '').any():
    missing = valid[valid['second_coder_label'].isna() | (valid['second_coder_label'] == '')]
    print(f"\nSTOPPING: {len(missing)} valid row(s) have no second_coder_label:")
    print(missing[['passage_id', 'firm', 'year']].to_string(index=False))
    raise SystemExit(1)

orig = valid['original_classification']
second = valid['second_coder_label']

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Kappa, raw agreement, confusion matrix
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 4: Agreement statistics")
print("=" * 90)

n = len(valid)
kappa_unweighted = cohen_kappa_score(orig, second, labels=LABELS)
kappa_linear = cohen_kappa_score(orig, second, labels=LABELS, weights='linear')
kappa_quadratic = cohen_kappa_score(orig, second, labels=LABELS, weights='quadratic')
raw_agreement = accuracy_score(orig, second)
n_agree = int((orig.values == second.values).sum())

print(f"N = {n}")
print(f"Unweighted Cohen's kappa: {kappa_unweighted:.4f}")
print(f"Linear-weighted Cohen's kappa: {kappa_linear:.4f}")
print(f"Quadratic-weighted Cohen's kappa: {kappa_quadratic:.4f}")
print(f"Raw agreement: {n_agree}/{n} = {raw_agreement*100:.2f}%")

print("\n3x3 confusion matrix (rows=original_classification, cols=second_coder_label):")
cm = confusion_matrix(orig, second, labels=LABELS)
header = f"{'orig \\ 2nd':15s}" + "".join(f"{l:14s}" for l in LABELS)
print(header)
for i, l in enumerate(LABELS):
    row = f"{l:15s}" + "".join(f"{cm[i,j]:14d}" for j in range(len(LABELS)))
    print(row)

# ══════════════════════════════════════════════════════════════════
# STEP 5 — Every disagreement individually
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 5: Individual disagreements")
print("=" * 90)

valid['orig_rank'] = valid['original_classification'].map(TIER_ORDER)
valid['second_rank'] = valid['second_coder_label'].map(TIER_ORDER)
disagreements = valid[valid['original_classification'] != valid['second_coder_label']].copy()

direction_counts = {'higher': 0, 'lower': 0}
adjacency_counts = {'adjacent': 0, 'non-adjacent': 0}

if len(disagreements) == 0:
    print("(no disagreements)")
else:
    for _, row in disagreements.iterrows():
        diff = row['second_rank'] - row['orig_rank']
        direction = 'higher' if diff > 0 else 'lower'
        adjacency = 'adjacent' if abs(diff) == 1 else 'non-adjacent'
        direction_counts[direction] += 1
        adjacency_counts[adjacency] += 1
        print(f"passage_id={row['passage_id']}  firm={row['firm']:15s} year={row['year']}  "
              f"original={row['original_classification']:13s} second_coder={row['second_coder_label']:13s}  "
              f"direction=second-coder-{direction}  {adjacency}")

# ══════════════════════════════════════════════════════════════════
# STEP 6 — Disagreement count and direction breakdown
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 6: Disagreement summary")
print("=" * 90)
print(f"Total disagreements: {len(disagreements)} / {n}")
print(f"Direction -- second coder assigned HIGHER tier than original: {direction_counts['higher']}")
print(f"Direction -- second coder assigned LOWER tier than original:  {direction_counts['lower']}")
print(f"Adjacent (one tier apart): {adjacency_counts['adjacent']}")
print(f"Non-adjacent (two tiers apart): {adjacency_counts['non-adjacent']}")

# ══════════════════════════════════════════════════════════════════
# STEP 7 — Reported vs. computed, side by side. NO adjustment.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 7: Reported vs. computed -- side by side")
print("=" * 90)
print(f"{'Metric':35s} {'Reported':>15s} {'Computed':>15s} {'Match?':>10s}")
print("-" * 78)

def cmp_row(label, reported, computed, round_dp, is_pct=False):
    # Reported figures were themselves rounded to round_dp decimal places
    # (e.g. kappa reported to 3dp, % reported to 1dp) -- match if the
    # computed value rounds to the same reported figure, not by an
    # arbitrary absolute tolerance.
    computed_rounded = round(computed, round_dp)
    match = "YES" if computed_rounded == reported else "NO"
    rep_str = f"{reported:.1f}%" if is_pct else f"{reported:.{round_dp}f}"
    comp_str = f"{computed:.1f}%" if is_pct else f"{computed:.4f}"
    print(f"{label:35s} {rep_str:>15s} {comp_str:>15s} {match:>10s}")
    return match == "YES"

r1 = cmp_row("Unweighted kappa", 0.791, kappa_unweighted, round_dp=3)
r2 = cmp_row("Linear-weighted kappa", 0.833, kappa_linear, round_dp=3)
r3 = cmp_row("Raw agreement", 86.2, raw_agreement*100, round_dp=1, is_pct=True)
r4 = cmp_row("Disagreement count", 4, len(disagreements), round_dp=0)

any_mismatch = not (r1 and r2 and r3 and r4)

print()
if any_mismatch:
    print("!" * 90)
    print("DISCREPANCY: the computed figures do NOT match the previously reported ones.")
    print("The computed figures above are the correct ones. Reported figures should be corrected")
    print("to match, not the other way around.")
    print("!" * 90)
else:
    print("All computed figures match the previously reported ones exactly.")

print("\nDone.")
