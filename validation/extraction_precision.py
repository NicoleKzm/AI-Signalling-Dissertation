"""
Computes precision of the extraction screen (classify.py's `is_ai_relevant()`) from the
hand-coded data/extraction_precision_sample.csv and data/extraction_precision_key.csv,
printing results to stdout (no file write).
"""
import sys

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

SAMPLE_FILE = 'data/extraction_precision_sample.csv'
KEY_FILE = 'data/extraction_precision_key.csv'
VALID_LABELS = {'YES', 'NO'}

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Read and normalise human_label_ai_relevant
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv(SAMPLE_FILE)
print(f"Loaded {SAMPLE_FILE}: {len(df)} rows")

key = pd.read_csv(KEY_FILE)
print(f"Loaded {KEY_FILE}: {len(key)} rows")

df = df.merge(key, on='passage_id', how='left', validate='one_to_one')
if df['assigned_classification'].isna().any():
    missing = df[df['assigned_classification'].isna()]
    print("\n" + "!" * 90)
    print(f"STOPPING: {len(missing)} passage_id(s) in {SAMPLE_FILE} have no match in "
          f"{KEY_FILE}:")
    for _, row in missing.iterrows():
        print(f"  {row['passage_id']}")
    print("!" * 90)
    sys.exit(1)

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
        print(f"  {row['passage_id']}: raw value = {row['human_label_ai_relevant']!r}")
    print("Not proceeding. Fix these values and re-run.")
    print("!" * 90)
    sys.exit(1)

if blank_mask.any():
    bad = df[blank_mask]
    print("\n" + "!" * 90)
    print(f"STOPPING: {len(bad)} row(s) have a BLANK human_label_ai_relevant -- "
          f"hand-coding is incomplete:")
    for _, row in bad.iterrows():
        print(f"  {row['passage_id']} ({row['firm']} {row['year']})")
    print("Not proceeding. Complete the hand-coding and re-run.")
    print("!" * 90)
    sys.exit(1)

df['label_clean'] = normalized

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Counts
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 2: Label counts")
print("=" * 90)
n_coded = len(df)
n_yes = int((df['label_clean'] == 'YES').sum())
n_no = int((df['label_clean'] == 'NO').sum())
print(f"n coded: {n_coded}")
print(f"YES: {n_yes}")
print(f"NO: {n_no}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Precision + Wilson 95% CI
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 3: Precision of the extraction screen")
print("=" * 90)
precision = n_yes / n_coded
print(f"PRECISION = YES / total sampled = {n_yes}/{n_coded} = {precision*100:.2f}%")

ci_lower, ci_upper = proportion_confint(n_yes, n_coded, alpha=0.05, method='wilson')
print(f"Wilson 95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — False positives (NO rows), printed in full
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 4: False positives of the extraction screen (human_label = NO)")
print("=" * 90)
false_positives = df[df['label_clean'] == 'NO']
if len(false_positives) == 0:
    print("None -- every sampled passage was judged genuinely AI-relevant.")
else:
    for _, row in false_positives.iterrows():
        print(f"\npassage_id: {row['passage_id']}")
        print(f"  firm: {row['firm']}, year: {row['year']}, page: {row['page']}")
        print(f"  assigned_classification: {row['assigned_classification']}")
        print(f"  passage_text ({len(str(row['passage_text']))} chars):")
        print(f"    {row['passage_text']!r}")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — Cross-tab: false positives vs assigned_classification
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 5: False positives x assigned_classification")
print("=" * 90)
print("\nFull sample, assigned_classification distribution:")
print(df['assigned_classification'].value_counts().to_string())

print("\nFalse positives (NO) by assigned_classification:")
if len(false_positives) == 0:
    print("N/A -- no false positives.")
else:
    fp_by_class = false_positives['assigned_classification'].value_counts()
    print(fp_by_class.to_string())
    print("\nFalse-positive rate WITHIN each assigned_classification (n_FP / n_sampled_in_that_class):")
    for cls in df['assigned_classification'].unique():
        n_in_class = int((df['assigned_classification'] == cls).sum())
        n_fp_in_class = int((false_positives['assigned_classification'] == cls).sum())
        rate = 100 * n_fp_in_class / n_in_class if n_in_class else float('nan')
        print(f"  {cls}: {n_fp_in_class}/{n_in_class} = {rate:.2f}%")
    print("\nConcentration check: are false positives disproportionately Symbolic "
          "relative to the full sample's Symbolic share?")
    sample_symbolic_share = 100 * (df['assigned_classification'] == 'Symbolic').sum() / len(df)
    fp_symbolic_share = 100 * (false_positives['assigned_classification'] == 'Symbolic').sum() / len(false_positives)
    print(f"  Symbolic share of full sample: {sample_symbolic_share:.2f}%")
    print(f"  Symbolic share of false positives: {fp_symbolic_share:.2f}%")

print("\nDone.")
