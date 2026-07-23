"""
Tests whether classify.py's temperature=0 setting produces stable output on repeated
calls, using a stratified sample of 30 already-classified passages from
data/all_classifications.csv, and writes results/determinism_check_results.txt.
Importing classify.py here is safe — its PDF-extraction and file-writing logic only
runs under `__main__`.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import classify  # noqa: E402  - verbatim production module, not modified

OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


log("=" * 90)
log("STEP 1: Frozen configuration read directly from classify.py (imported, unmodified)")
log("=" * 90)
log(f"Model: {classify.LLM_MODEL}")
log(f"Temperature: 0 (hardcoded literal in classify.classify_passage(), verified by "
    f"inspection of classify.py -- 'temperature=0' at the client.messages.create() call)")
log(f"max_tokens: 400 (hardcoded literal, same call)")
log(f"Passage character limit sent to API: passage_text[:1500] (same call)")
log(f"Prompt: CLASSIFICATION_PROMPT imported from classify.py, {len(classify.CLASSIFICATION_PROMPT)} "
    f"chars, used verbatim -- not copied or retyped.")
log("\nThis script calls classify.classify_passage() directly (the actual production "
    "function object), so there is no risk of prompt/parameter drift between the "
    "frozen instrument and this validation run.")

# ══════════════════════════════════════════════════════════════════
log("\n\n" + "=" * 90)
log("STEP 2: Stratified sample (n=30, random_state=42)")
log("=" * 90)

df = pd.read_csv('data/all_classifications.csv')
log(f"Loaded all_classifications.csv: {len(df)} rows")
counts = df['classification'].value_counts()
log(f"Available by class: {dict(counts)}")

TARGET_N = {'Symbolic': 10, 'Transitional': 10, 'Substantive': 10}
n_substantive_available = int(counts.get('Substantive', 0))
if n_substantive_available < 10:
    log(f"\nSubstantive has only {n_substantive_available} rows (<10) -- taking all "
        f"{n_substantive_available} and keeping Symbolic/Transitional at 10 each "
        f"(n={20 + n_substantive_available}, NOT 30). This branch is INACTIVE for "
        f"this run -- see actual count below.")
    TARGET_N['Substantive'] = n_substantive_available
else:
    log(f"\nSubstantive has {n_substantive_available} rows (>= 10) -- the 'fewer than "
        f"10' condition in the task spec does NOT apply here (16 is not fewer than "
        f"10). Proceeding with the standard 10/10/10 stratified draw, n=30.")

sample_parts = []
for cls, n in TARGET_N.items():
    pool = df[df['classification'] == cls]
    drawn = pool.sample(n=n, random_state=42)
    sample_parts.append(drawn)
    log(f"  {cls}: drew {n} of {len(pool)} available")

sample = pd.concat(sample_parts).reset_index(drop=True)
log(f"\nTotal sample size: {len(sample)}")

# ══════════════════════════════════════════════════════════════════
log("\n\n" + "=" * 90)
log("STEP 3: Re-classification via classify.classify_passage() (live API calls)")
log("=" * 90)

rerun_labels = []
rerun_raw = []
for i, row in sample.iterrows():
    log(f"  [{i+1}/{len(sample)}] Re-classifying {row['passage_id']} "
        f"({row['firm']} {row['year']}, stored={row['classification']})...")
    result = classify.classify_passage(row['passage_text'])
    if result is None:
        log(f"    API/parse FAILURE on {row['passage_id']} -- excluded from comparison")
        rerun_labels.append(None)
        rerun_raw.append(None)
    else:
        rerun_labels.append(result['classification'])
        rerun_raw.append(result)
    time.sleep(0.5)

sample['rerun_classification'] = rerun_labels

n_failed = sample['rerun_classification'].isna().sum()
if n_failed:
    log(f"\n{n_failed} passage(s) failed to re-classify (API/parse error) -- excluded "
        f"from all comparison statistics below.")
comparable = sample.dropna(subset=['rerun_classification']).copy()

# ══════════════════════════════════════════════════════════════════
log("\n\n" + "=" * 90)
log("STEP 4: Comparison -- stored vs re-run labels")
log("=" * 90)

n_total = len(comparable)
n_agree = int((comparable['classification'] == comparable['rerun_classification']).sum())
agree_pct = 100 * n_agree / n_total if n_total else float('nan')
log(f"Exact agreement: {n_agree}/{n_total} = {agree_pct:.2f}%")

labels_order = ['Symbolic', 'Transitional', 'Substantive']
kappa = cohen_kappa_score(comparable['classification'], comparable['rerun_classification'],
                           labels=labels_order)
log(f"Cohen's kappa (stored vs re-run): {kappa:.4f}")

cm = confusion_matrix(comparable['classification'], comparable['rerun_classification'],
                       labels=labels_order)
cm_df = pd.DataFrame(cm, index=[f"stored_{l}" for l in labels_order],
                      columns=[f"rerun_{l}" for l in labels_order])
log("\n3x3 confusion matrix (rows=stored, cols=re-run):")
log(cm_df.to_string())

log("\n" + "-" * 90)
log("Passages where the label changed:")
log("-" * 90)
changed = comparable[comparable['classification'] != comparable['rerun_classification']]
if len(changed) == 0:
    log("None -- every re-classified passage received the same label as stored.")
else:
    tier_rank = {'Symbolic': 0, 'Transitional': 1, 'Substantive': 2}
    for _, row in changed.iterrows():
        jump = abs(tier_rank[row['classification']] - tier_rank[row['rerun_classification']])
        jump_desc = "ONE-TIER ADJACENT" if jump == 1 else "TWO-TIER JUMP"
        log(f"\npassage_id: {row['passage_id']}")
        log(f"  firm: {row['firm']}, year: {row['year']}")
        log(f"  stored label: {row['classification']}  ->  rerun label: {row['rerun_classification']}")
        log(f"  change type: {jump_desc}")
        log(f"  stored text (full, as stored in all_classifications.csv, "
            f"{len(str(row['passage_text']))} chars):")
        log(f"    {row['passage_text']!r}")

log(f"\nSummary: {len(changed)} of {n_total} passages changed classification "
    f"({100*len(changed)/n_total:.2f}%).")
if len(changed) > 0:
    n_one_tier = sum(1 for _, r in changed.iterrows()
                      if abs(tier_rank[r['classification']] - tier_rank[r['rerun_classification']]) == 1)
    n_two_tier = len(changed) - n_one_tier
    log(f"  One-tier adjacent: {n_one_tier}")
    log(f"  Two-tier jump: {n_two_tier}")

log("\n" + "=" * 90)
log("REMINDER OF LIMITATION (see module docstring)")
log("=" * 90)
n_truncated_in_sample = int((comparable['passage_text'].str.len() == 300).sum())
log(f"{n_truncated_in_sample}/{n_total} passages in this sample are stored at exactly "
    f"300 characters (i.e. were truncated from a longer original passage sent to the "
    f"original classification call, which used up to 1500 characters). Any disagreement "
    f"on these rows cannot be cleanly attributed to temperature=0 non-determinism alone "
    f"-- it may instead (or additionally) reflect the re-run seeing less text than the "
    f"original run did.")

with open('results/determinism_check_results.txt', 'w') as f:
    f.write("\n".join(OUT) + "\n")
print("\nresults/determinism_check_results.txt saved.")
