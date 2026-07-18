"""
extraction_recall_sample.py

Supervisor critique: the reported kappa validates only passages that PASSED
the keyword screen -- it says nothing about what the extraction rule MISSED.
Builds a random, stratified sample of NON-extracted text chunks for manual
coding, to let a human estimate false negatives / recall of the extraction
rule. Does NOT compute kappa or recall itself -- that requires the human
labels this script leaves blank.

PREMISE CORRECTION (reported, not silently applied): the task asked to
oversample chunks "with a Tier 1 anchor but rejected for lacking a Tier 2
trigger." Per classify.py's actual is_ai_relevant(), Tier 1 alone is
SUFFICIENT for extraction -- Tier 2 only changes the tier TAG
('tier1' vs 'tier_1_and_2'), never inclusion/exclusion. A Tier-1-present
chunk is never rejected for lacking Tier 2; the only way a Tier-1 chunk
is excluded is the unrelated 60%-overlap same-page deduplication rule.
The actual highest-false-negative-risk category is the OPPOSITE:
Tier 2 present, Tier 1 ABSENT -- these chunks contain real
application-level AI language but never even reach the relevance check,
since is_ai_relevant() requires Tier 1 regardless of Tier 2. This matches
confirmed false negatives found earlier (e.g. "predictive models",
"demand forecasting" passages with no co-occurring Tier 1 term). That is
the category oversampled below.

Does not modify classify.py and does not re-run classification -- reuses
classify.py's TIER_1_KEYWORDS, TIER_2_KEYWORDS, is_ai_relevant(),
deduplicate_passages(), parse_firm_year(), ANNUAL_REPORTS_DIR by import
only. Writes only to extraction_recall_sample.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify  # noqa: E402 -- reuse only; __main__ guard prevents re-running the pipeline

RNG_SEED = 42
TARGET_TOTAL = 60
TARGET_OVERSAMPLE = 20  # Tier-2-present/Tier-1-absent quota (see premise correction above)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Report the extraction methodology exactly as coded
# ══════════════════════════════════════════════════════════════════
print("=" * 100)
print("STEP 1: EXTRACTION METHODOLOGY (as implemented in classify.py)")
print("=" * 100)

print(f"\nTier 1 anchor list ({len(classify.TIER_1_KEYWORDS)} terms) -- classify.py lines 23-31:")
for kw in classify.TIER_1_KEYWORDS:
    print(f"  - {kw}")

print(f"\nTier 2 trigger list ({len(classify.TIER_2_KEYWORDS)} terms) -- classify.py lines 33-46:")
for kw in classify.TIER_2_KEYWORDS:
    print(f"  - {kw}")

print("""
Relevance rule (classify.py is_ai_relevant(), lines 157-172):
  has_tier1 = any Tier-1 term present (word-boundary regex match)
  has_tier2 = any Tier-2 term present (word-boundary regex match)
  if has_tier1 and has_tier2: relevant=True, tag='tier_1_and_2'
  elif has_tier1:             relevant=True, tag='tier1'
  else:                       relevant=False  (chunk is discarded, regardless of Tier 2)
  ==> Tier 1 is REQUIRED for extraction. Tier 2 alone is NEVER sufficient.
      Tier 2 only ever affects the tag on an ALREADY-Tier-1 chunk.

Chunking window (classify.py extract_ai_passages(), lines 196-225):
  Each page's text is whitespace-normalised, split into sentences on
  (?<=\\.)\\s+, then slid through 3-consecutive-sentence windows
  (sentences[i:i+3] for every i), each window >= 60 characters.
  Each window is independently tested by is_ai_relevant(); overlapping
  windows over the same sentences are all tested (no window is skipped
  because a neighbouring window already matched).

Deduplication rule (classify.py deduplicate_passages(), lines 176-192):
  Applied ONLY to chunks that already passed is_ai_relevant() (i.e. only
  within a report's Tier-1-relevant candidate pool -- never to
  Tier-1-absent chunks, which never enter this pool at all).
  Two candidates on the SAME PDF page are duplicates if the character-by-
  character match of their first 250 characters (lowercased) exceeds 60%
  of the longer signature's length; only the first-seen survives.
""")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Enumerate the FULL chunk universe across the corpus, tracking
# extraction status per chunk exactly as classify.py would determine it.
# ══════════════════════════════════════════════════════════════════
print("=" * 100)
print("STEP 2: Enumerating full chunk universe across the corpus")
print("=" * 100)

pdf_files = sorted(Path(classify.ANNUAL_REPORTS_DIR).glob("**/*.pdf"))
print(f"PDF files found: {len(pdf_files)}")

all_chunk_records = []
extracted_count_check = 0
reports_processed = 0

for pdf_path in pdf_files:
    firm, year = classify.parse_firm_year(pdf_path)
    if not firm or not year or year < 2021 or year > 2025:
        continue
    if firm.strip().lower() == "about you" and year == 2025:
        continue
    reports_processed += 1

    raw_relevant = []  # only Tier-1-relevant chunks enter this (mirrors extract_ai_passages exactly)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                text = ' '.join(text.split())
                sentences = re.split(r'(?<=\.)\s+', text)
                for i in range(len(sentences)):
                    chunk = " ".join(sentences[i:i + 3]).strip()
                    if len(chunk) < 60:
                        continue
                    relevant, tier = classify.is_ai_relevant(chunk)
                    text_lower = chunk.lower()
                    has_tier1 = any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
                                     for kw in classify.TIER_1_KEYWORDS)
                    has_tier2 = any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
                                     for kw in classify.TIER_2_KEYWORDS)
                    rec = {'firm': firm, 'year': year, 'page': page_num, 'chunk_text': chunk,
                           'contains_tier1': has_tier1, 'contains_tier2': has_tier2,
                           'raw_idx': len(all_chunk_records)}
                    all_chunk_records.append(rec)
                    if relevant:
                        raw_relevant.append(dict(rec, text=chunk, page_number=page_num))
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
        continue

    kept = classify.deduplicate_passages(raw_relevant)
    kept_idx = {p['raw_idx'] for p in kept}
    for p in raw_relevant:
        all_chunk_records[p['raw_idx']]['extracted'] = (p['raw_idx'] in kept_idx)
    extracted_count_check += len(kept)

# Chunks that never entered raw_relevant (Tier-1-absent) are never extracted
for rec in all_chunk_records:
    if 'extracted' not in rec:
        rec['extracted'] = False

df_all = pd.DataFrame(all_chunk_records)
total_chunks = len(df_all)
total_extracted = int(df_all['extracted'].sum())

print(f"\nReports processed: {reports_processed}")
print(f"Total chunks (>=60 chars, 3-sentence windows) across corpus: {total_chunks}")
print(f"Total chunks extracted (relevant AND survived dedup): {total_extracted}")
print(f"Expected extracted count (all_classifications.csv row count): 512")
print(f"{'MATCH' if total_extracted == 512 else 'MISMATCH -- investigate'}")
print(f"Proportion of corpus extracted: {total_extracted / total_chunks * 100:.4f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 3/4 — Build the non-extracted sample, oversampling Tier-2-present/
# Tier-1-absent chunks (the actual highest-false-negative-risk category)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STEP 3/4: Building the 60-chunk non-extracted sample (seed=42)")
print("=" * 100)

non_extracted = df_all[~df_all['extracted']].copy()
print(f"Non-extracted chunk pool: {len(non_extracted)}")

oversample_pool = non_extracted[non_extracted['contains_tier2'] & ~non_extracted['contains_tier1']].copy()
baseline_pool = non_extracted[~(non_extracted['contains_tier2'] & ~non_extracted['contains_tier1'])].copy()
dedup_removed_pool = non_extracted[non_extracted['contains_tier1']].copy()  # tier1 present but deduped away

print(f"Tier 2 present / Tier 1 absent (oversample target category): {len(oversample_pool)}")
print(f"Tier 1 present but removed by same-page dedup: {len(dedup_removed_pool)}")
print(f"Neither tier present (baseline irrelevant text): "
      f"{len(non_extracted) - len(oversample_pool) - len(dedup_removed_pool)}")

rng = np.random.default_rng(RNG_SEED)


def stratified_sample(pool, n, rng):
    """Sample n rows from pool, spreading as evenly as possible across firms
    so no single firm dominates, using the given rng for reproducibility."""
    if len(pool) <= n:
        return pool.copy()
    firms = pool['firm'].unique()
    per_firm_quota = max(1, n // len(firms))
    parts = []
    remaining = n
    firm_list = list(rng.permutation(firms))
    for idx, firm in enumerate(firm_list):
        firm_pool = pool[pool['firm'] == firm]
        firms_left = len(firm_list) - idx
        take = min(len(firm_pool), max(1, remaining // firms_left))
        if take > 0 and len(firm_pool) > 0:
            sampled_idx = rng.choice(firm_pool.index, size=min(take, len(firm_pool)), replace=False)
            parts.append(pool.loc[sampled_idx])
            remaining -= len(sampled_idx)
    result = pd.concat(parts) if parts else pool.iloc[0:0]
    if len(result) < n:
        leftover = pool.drop(result.index)
        if len(leftover) > 0:
            extra_n = min(n - len(result), len(leftover))
            extra_idx = rng.choice(leftover.index, size=extra_n, replace=False)
            result = pd.concat([result, pool.loc[extra_idx]])
    return result


n_oversample = min(TARGET_OVERSAMPLE, len(oversample_pool))
oversample_drawn = stratified_sample(oversample_pool, n_oversample, rng)

n_baseline = TARGET_TOTAL - len(oversample_drawn)
baseline_drawn = stratified_sample(baseline_pool, n_baseline, rng)

sample = pd.concat([oversample_drawn, baseline_drawn]).reset_index(drop=True)

print(f"\nDrawn: {len(oversample_drawn)} from oversample category (Tier 2 present/Tier 1 absent)")
print(f"Drawn: {len(baseline_drawn)} from baseline category")
print(f"Total drawn: {len(sample)}")

print("\nFirm distribution of the 60-chunk sample:")
print(sample['firm'].value_counts().to_string())
print("\nYear distribution of the 60-chunk sample:")
print(sample['year'].value_counts().sort_index().to_string())


def rejection_reason(row):
    if row['contains_tier2'] and not row['contains_tier1']:
        return 'Tier 2 present, Tier 1 absent (extraction requires Tier 1; Tier 2 alone insufficient)'
    elif row['contains_tier1']:
        return 'Tier 1 present but removed by same-page 60% deduplication'
    else:
        return 'No Tier 1 or Tier 2 keyword match'


sample['rejection_reason'] = sample.apply(rejection_reason, axis=1)
sample['chunk_id'] = ['REC' + str(i + 1).zfill(3) for i in range(len(sample))]
sample['human_label_ai_relevant'] = ''

out_cols = ['chunk_id', 'firm', 'year', 'page', 'chunk_text', 'contains_tier1', 'contains_tier2',
            'rejection_reason', 'human_label_ai_relevant']
sample[out_cols].to_csv('extraction_recall_sample.csv', index=False)
print(f"\nextraction_recall_sample.csv saved -- {len(sample)} rows")

print("\n" + "=" * 100)
print("STEP 5: Corpus-wide extraction summary")
print("=" * 100)
print(f"Total chunks in corpus: {total_chunks}")
print(f"Total chunks extracted: {total_extracted}")
print(f"Proportion of corpus extracted: {total_extracted / total_chunks * 100:.4f}%")
print(f"Proportion of corpus NOT extracted: {(total_chunks - total_extracted) / total_chunks * 100:.4f}%")

print("\nDone.")
