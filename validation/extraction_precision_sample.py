"""
extraction_precision_sample.py

Builds a blind-coding sample to estimate PRECISION of the extraction screen
(classify.py's is_ai_relevant()): of the 512 passages that were extracted
and sent to the classifier, how many are genuinely AI-relevant? This is the
counterpart to extraction_recall.py, which estimated the false-negative
rate (recall) on NON-extracted chunks. This script only builds the sample
for hand-coding -- it does not compute precision itself (that is
extraction_precision.py, written separately, not run yet).

Sampling: n=40, stratified ACROSS FIRMS so no single firm dominates (Allegro
alone is 176/512 = 34.4% of all extracted passages -- a plain random sample
of 40 would be expected to draw ~14 Allegro rows). Allocation method:
40 // 13 firms = 3 per firm baseline (39), remainder of 1 assigned to the
first firm in Python's default (ASCII) sort order -- which is "AO World",
not "About You": uppercase 'O' sorts before lowercase 'b'. Confirmed
against the actual run: AO World received 4, every other firm 3.
Alphabetical order is used (not size- or count-based) so the allocation
itself doesn't systematically favour or penalise any firm. Within each
firm, the specific rows drawn use pandas .sample(random_state=42) for
reproducibility.

Output columns are deliberately restricted to keep the human coder blind to
what the extraction screen and classifier concluded: passage_id, firm,
year, page, passage_text (full stored text, as-is -- already <=300 chars
in all_classifications.csv), and a blank human_label_ai_relevant column
for hand-coding. tier / contains_tier1 / contains_tier2 (the columns
identifying which keyword tier triggered extraction), the model's
assigned_classification, and justification (the model's own reasoning)
are ALL excluded from this file on purpose -- coding must be blind.

assigned_classification is written separately to
extraction_precision_key.csv, keyed on passage_id, so it can be joined
back in AFTER hand-coding (by extraction_precision.py) without ever being
visible during coding.

Reads all_classifications.csv (read-only). Writes
extraction_precision_sample.csv (blind, for hand-coding) and
extraction_precision_key.csv (the answer key, not for the coder). Does
not modify classify.py, all_classifications.csv, or any results file.
"""
import pandas as pd

df = pd.read_csv('data/all_classifications.csv')
print(f"Loaded all_classifications.csv: {len(df)} rows, {df['firm'].nunique()} firms")
print("\nFirm counts in full extracted set:")
print(df['firm'].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Stratified-by-firm allocation, n=40
# ══════════════════════════════════════════════════════════════════
firms = sorted(df['firm'].unique())
n_firms = len(firms)
N_TARGET = 40
base = N_TARGET // n_firms
remainder = N_TARGET - base * n_firms

allocation = {f: base for f in firms}
for f in firms[:remainder]:  # alphabetical order, deterministic
    allocation[f] += 1

print(f"\n{n_firms} firms present. Base allocation = {N_TARGET}//{n_firms} = {base} per firm, "
      f"remainder = {remainder} (assigned to first {remainder} firm(s) alphabetically).")
print("Per-firm allocation:")
for f in firms:
    available = int((df['firm'] == f).sum())
    print(f"  {f:20s} allocate={allocation[f]}  available={available}")
    if allocation[f] > available:
        raise SystemExit(f"FATAL: allocation for {f} ({allocation[f]}) exceeds available rows "
                          f"({available}) -- cannot draw sample as specified.")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Draw sample
# ══════════════════════════════════════════════════════════════════
sample_parts = []
for f in firms:
    pool = df[df['firm'] == f]
    drawn = pool.sample(n=allocation[f], random_state=42)
    sample_parts.append(drawn)

sample = pd.concat(sample_parts).reset_index(drop=True)
print(f"\nTotal drawn: {len(sample)}")

print("\nFirm distribution of the sample:")
print(sample['firm'].value_counts().sort_index().to_string())

print("\nYear distribution of the sample:")
print(sample['year'].value_counts().sort_index().to_string())

print("\nassigned_classification distribution of the sample (for reference only -- "
      "not shown to the human coder):")
print(sample['classification'].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Build outputs: blind coding CSV + separate answer key
# ══════════════════════════════════════════════════════════════════
out = pd.DataFrame({
    'passage_id': sample['passage_id'],
    'firm': sample['firm'],
    'year': sample['year'],
    'page': sample['page_number'],
    'passage_text': sample['passage_text'],
    'human_label_ai_relevant': '',
})

key = pd.DataFrame({
    'passage_id': sample['passage_id'],
    'assigned_classification': sample['classification'],
})

out.to_csv('data/extraction_precision_sample.csv', index=False)
key.to_csv('data/extraction_precision_key.csv', index=False)

print(f"\nextraction_precision_sample.csv saved -- {len(out)} rows, columns: {list(out.columns)}")
print(f"extraction_precision_key.csv saved -- {len(key)} rows, columns: {list(key.columns)} "
      f"(answer key, NOT for the coder)")
print("\nExcluded from the blind CSV on purpose: assigned_classification (see key file), "
      "tier, contains_tier1, contains_tier2, justification, q1/q2/q3 scores, "
      "classification_score, source_document, llm_model, validation_sample, human_label, "
      "agreement.")
print("\nDone. Hand-code human_label_ai_relevant with YES or NO for each row in "
      "extraction_precision_sample.csv. Do not open extraction_precision_key.csv until "
      "coding is complete.")
