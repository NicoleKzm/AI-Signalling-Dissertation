"""
Rebuilds the fragmentation-exclusion robustness check behind Chapter 4's zero-score-share
statistic (63.8% -> 65.2%), reading data/all_classifications.csv, data/signalling_scores.csv,
and data/panel_dataset.csv and writing results/fragmentation_robustness.csv.
"""
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats

print("=" * 90)
print("LIMITATION NOTICE")
print("=" * 90)
print("classify.py sends up to 1500 characters of each extracted passage to the")
print("classification model (classify.py line 237: passage_text[:1500]), but stores")
print("only the first 300 characters in all_classifications.csv (line 378:")
print("passage['text'][:300]). The fragmentation flag below is computed on the")
print("STORED (300-char) text only. Apparent fragmentation may therefore be partly")
print("a storage-truncation artefact rather than something the model itself saw --")
print("a passage that reads as fragmented at 300 characters may have been coherent")
print("in the fuller 1500-character context the model actually classified. This")
print("robustness check should be read with that caveat: it tests sensitivity to")
print("passages that LOOK fragmented in the stored record, not to passages confirmed")
print("fragmented in the model's actual input.")

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Fragmentation flag (same heuristic as the prior audit)
# ══════════════════════════════════════════════════════════════════
CONTINUATION_WORDS = {
    'and', 'but', 'or', 'so', 'because', 'while', 'which', 'that', 'also',
    'however', 'then', 'of', 'in', 'to', 'for', 'with', 'since', 'although',
    'though', 'this', 'these', 'those', 'it', 'its', 'as', 'than', 'if',
    'when', 'where', 'who', 'whom', 'whose', 'yet', 'nor', 'from', 'on',
    'at', 'by'
}


def first_word(text):
    m = re.match(r"[^\w]*(\w+)", text)
    return m.group(1) if m else ""


def starts_lowercase(text):
    stripped = text.strip()
    if not stripped:
        return False
    c = stripped[0]
    return c.isalpha() and c.islower()


def starts_with_continuation_word(text):
    return first_word(text).lower() in CONTINUATION_WORDS


def has_midtext_restart(text):
    return bool(re.search(r'\.\s+[a-z]', text))


def flag_fragment(text):
    if pd.isna(text):
        return False
    return starts_lowercase(text) or starts_with_continuation_word(text) or has_midtext_restart(text)


class_df = pd.read_csv('data/all_classifications.csv')
class_df['flagged_fragmented'] = class_df['passage_text'].apply(flag_fragment)

n_total = len(class_df)
n_flagged = int(class_df['flagged_fragmented'].sum())
print("\n" + "=" * 90)
print("STEP 1: Fragmentation flag")
print("=" * 90)
print(f"Total passages: {n_total}")
print(f"Flagged as fragmented: {n_flagged} ({n_flagged/n_total*100:.1f}%)  [expected ~247 of 512]")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Recompute firm-year signal scores with flagged passages
# excluded, using the EXACT aggregation rule from classify.py's
# get_firm_year_score(): mean of per-passage scores
# (Symbolic=0, Transitional=1, Substantive=2); firm-years with zero
# remaining qualifying passages score 0.
# ══════════════════════════════════════════════════════════════════
score_map = {'Symbolic': 0, 'Transitional': 1, 'Substantive': 2}
class_df['passage_score'] = class_df['classification'].map(score_map)

kept = class_df[~class_df['flagged_fragmented']]

baseline = pd.read_csv('data/signalling_scores.csv')  # authoritative list of all 69 firm-years
all_firm_years = baseline[['firm', 'year']].copy()

agg = (kept.groupby(['firm', 'year'])['passage_score']
       .agg(mean_signal_score_fragexcl='mean', total_passages_fragexcl='count')
       .reset_index())

merged = all_firm_years.merge(agg, on=['firm', 'year'], how='left')
merged['mean_signal_score_fragexcl'] = merged['mean_signal_score_fragexcl'].fillna(0.0)
merged['total_passages_fragexcl'] = merged['total_passages_fragexcl'].fillna(0).astype(int)

compare = baseline[['firm', 'year', 'total_passages', 'mean_signal_score']].merge(
    merged, on=['firm', 'year'])
compare['zero_before'] = compare['mean_signal_score'] == 0
compare['zero_after'] = compare['mean_signal_score_fragexcl'] == 0

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Compare distributions
# ══════════════════════════════════════════════════════════════════
n_fy = len(compare)
zero_before_n = int(compare['zero_before'].sum())
zero_after_n = int(compare['zero_after'].sum())
zero_before_pct = zero_before_n / n_fy * 100
zero_after_pct = zero_after_n / n_fy * 100

mean_before = compare['mean_signal_score'].mean()
mean_after = compare['mean_signal_score_fragexcl'].mean()
median_after = compare['mean_signal_score_fragexcl'].median()
sd_after = compare['mean_signal_score_fragexcl'].std()

shift_pos_to_zero = compare[(~compare['zero_before']) & (compare['zero_after'])]
shift_zero_to_pos = compare[(compare['zero_before']) & (~compare['zero_after'])]

print("\n" + "=" * 90)
print("STEP 3: Fragmentation-excluded firm-year signal scores vs. baseline")
print("=" * 90)
print(f"N firm-years: {n_fy}")
print(f"\nZero-score share:")
print(f"  Baseline (verified):        {zero_before_n}/{n_fy} = {zero_before_pct:.1f}%  "
      f"(cf. Chapter 4.4.3's implied 63.8%)")
print(f"  Fragmentation-excluded:     {zero_after_n}/{n_fy} = {zero_after_pct:.1f}%")
print(f"  Change: {zero_after_pct - zero_before_pct:+.1f} pp  "
      f"(cf. Chapter 4.4.2's claimed 63.8% -> 65.2%, i.e. +1.4pp)")

print(f"\nMean signal score:")
print(f"  Baseline (verified):        {mean_before:.4f}  (cf. Chapter 4's cited 0.181)")
print(f"  Fragmentation-excluded:     {mean_after:.4f}")
print(f"  Median (fragmentation-excluded): {median_after:.4f}")
print(f"  SD (fragmentation-excluded):     {sd_after:.4f}")

print(f"\nFirm-years shifting POSITIVE -> ZERO ({len(shift_pos_to_zero)}):")
if len(shift_pos_to_zero):
    print(shift_pos_to_zero[['firm', 'year', 'mean_signal_score', 'total_passages',
                              'mean_signal_score_fragexcl', 'total_passages_fragexcl']].to_string(index=False))
else:
    print("  (none)")

print(f"\nFirm-years shifting ZERO -> POSITIVE ({len(shift_zero_to_pos)}) [should be 0 -- "
      f"excluding passages can only remove evidence, never add it]:")
if len(shift_zero_to_pos):
    print(shift_zero_to_pos[['firm', 'year', 'mean_signal_score', 'total_passages',
                              'mean_signal_score_fragexcl', 'total_passages_fragexcl']].to_string(index=False))
    print("  *** LOGICAL ANOMALY: this should be impossible under this aggregation rule. Investigate. ***")
else:
    print("  (none, as expected)")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Re-run H1/H2/H3 on the fragmentation-excluded signal score,
# mirroring regression_clean.py's exact primary specification.
# ══════════════════════════════════════════════════════════════════
panel = pd.read_csv('data/panel_dataset.csv')
panel = panel.drop(columns=['mean_signal_score'], errors='ignore').merge(
    compare[['firm', 'year', 'mean_signal_score_fragexcl']], on=['firm', 'year'], how='left')
panel = panel.sort_values(['firm', 'year']).reset_index(drop=True)

panel['log_revenue'] = np.log(panel['Revenue'].replace(0, np.nan))
year_gap1 = panel.groupby('firm')['year'].diff()
panel['signal_lag1_fragexcl'] = panel.groupby('firm')['mean_signal_score_fragexcl'].shift(1)
panel.loc[year_gap1 != 1, 'signal_lag1_fragexcl'] = np.nan
panel['log_revenue_lag1'] = panel.groupby('firm')['log_revenue'].shift(1)
panel.loc[year_gap1 != 1, 'log_revenue_lag1'] = np.nan

panel = panel.set_index(['firm', 'year'])
zalando_2025_mask = ~((panel.index.get_level_values('firm') == 'Zalando') &
                       (panel.index.get_level_values('year') == 2025))
panel_excl_zalando = panel[zalando_2025_mask]

print("\n" + "=" * 90)
print("STEP 4: Primary regressions on fragmentation-excluded signal score")
print("=" * 90)


def fit(data, dv, iv, control):
    full = data[[dv, iv, control]]
    model_df = full.dropna()
    exog = sm.add_constant(model_df[[iv, control]])
    result = PanelOLS(model_df[dv], exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    return result, model_df


h1_result, h1_df = fit(panel, 'Stock_Price_Movement_%', 'mean_signal_score_fragexcl', 'log_revenue')
h1_beta = h1_result.params['mean_signal_score_fragexcl']
h1_se = h1_result.std_errors['mean_signal_score_fragexcl']
h1_p = h1_result.pvalues['mean_signal_score_fragexcl']
h1_n = len(h1_df)
print(f"H1: beta={h1_beta:.4f}, SE={h1_se:.4f}, p={h1_p:.4f}, N={h1_n}")

h2_result, h2_df = fit(panel_excl_zalando, 'Revenue_Growth_%', 'signal_lag1_fragexcl', 'log_revenue_lag1')
h2_beta = h2_result.params['signal_lag1_fragexcl']
h2_se = h2_result.std_errors['signal_lag1_fragexcl']
h2_p = h2_result.pvalues['signal_lag1_fragexcl']
h2_n = len(h2_df)
print(f"H2: beta={h2_beta:.4f}, SE={h2_se:.4f}, p={h2_p:.4f}, N={h2_n}")

h3_result, h3_df = fit(panel_excl_zalando, 'Gross_Margin_%', 'signal_lag1_fragexcl', 'log_revenue_lag1')
h3_beta = h3_result.params['signal_lag1_fragexcl']
h3_se = h3_result.std_errors['signal_lag1_fragexcl']
h3_p = h3_result.pvalues['signal_lag1_fragexcl']
h3_n = len(h3_df)
h3_df_resid = h3_result.df_resid
print(f"H3: beta={h3_beta:.4f}, SE={h3_se:.4f}, p={h3_p:.4f}, N={h3_n}, df_resid={h3_df_resid}")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — H3 TOST, df = df_resid read directly from the fitted model
# ══════════════════════════════════════════════════════════════════
BOUND = 3.462
t_lower = (h3_beta - (-BOUND)) / h3_se
t_upper = (BOUND - h3_beta) / h3_se
p_lower = 1 - stats.t.cdf(t_lower, h3_df_resid)
p_upper = 1 - stats.t.cdf(t_upper, h3_df_resid)
equivalent = (p_lower < 0.05) and (p_upper < 0.05)

print("\n" + "=" * 90)
print("STEP 5: H3 TOST on fragmentation-excluded data")
print("=" * 90)
print(f"beta={h3_beta:.4f}, SE={h3_se:.4f}, N={h3_n}, df=df_resid={h3_df_resid}, bound=+/-{BOUND}")
print(f"t_lower={t_lower:.4f}, t_upper={t_upper:.4f}")
print(f"p_lower={p_lower:.4f}, p_upper={p_upper:.4f}")
print(f"Equivalence holds: {equivalent}")

# ══════════════════════════════════════════════════════════════════
# STEP 6 — Settle the 64.8% vs 63.8% Chapter 4 discrepancy
# ══════════════════════════════════════════════════════════════════
full_baseline = pd.read_csv('data/signalling_scores.csv')
zc = (full_baseline['mean_signal_score'] == 0).sum()
tot = len(full_baseline)
pct = zc / tot * 100

print("\n" + "=" * 90)
print("STEP 6: Baseline zero-score share -- settling the 64.8% vs 63.8% discrepancy")
print("=" * 90)
print(f"signalling_scores.csv: {zc} firm-years with mean_signal_score == 0, out of {tot} total")
print(f"Arithmetic: {zc} / {tot} = {zc/tot:.6f} = {pct:.4f}% -> rounds to {pct:.1f}%")
print(f"\nChapter 4.2.1 claims: 64.8%")
print(f"Chapter 4.4.3 implies baseline: 63.8%")
print(f"Computed value: {pct:.1f}%")
if abs(pct - 63.8) < 0.05:
    print("-> CORRECT figure is 63.8% (matches 4.4.3's baseline). "
          "4.2.1's '64.8%' is wrong and should be corrected to 63.8%.")
elif abs(pct - 64.8) < 0.05:
    print("-> CORRECT figure is 64.8% (matches 4.2.1). "
          "4.4.3's '63.8%' baseline is wrong and should be corrected to 64.8%.")
else:
    print(f"-> NEITHER cited figure matches. True value is {pct:.1f}%. Both need correcting.")

# ══════════════════════════════════════════════════════════════════
# Write output CSV
# ══════════════════════════════════════════════════════════════════
summary_rows = [
    {'Metric': 'passages_total', 'Value': n_total},
    {'Metric': 'passages_flagged_fragmented', 'Value': n_flagged},
    {'Metric': 'passages_flagged_pct', 'Value': round(n_flagged/n_total*100, 2)},
    {'Metric': 'baseline_zero_score_n', 'Value': zero_before_n},
    {'Metric': 'baseline_zero_score_pct', 'Value': round(zero_before_pct, 2)},
    {'Metric': 'baseline_mean_signal_score', 'Value': round(mean_before, 4)},
    {'Metric': 'fragexcl_zero_score_n', 'Value': zero_after_n},
    {'Metric': 'fragexcl_zero_score_pct', 'Value': round(zero_after_pct, 2)},
    {'Metric': 'fragexcl_mean_signal_score', 'Value': round(mean_after, 4)},
    {'Metric': 'fragexcl_median_signal_score', 'Value': round(median_after, 4)},
    {'Metric': 'fragexcl_sd_signal_score', 'Value': round(sd_after, 4)},
    {'Metric': 'n_shift_positive_to_zero', 'Value': len(shift_pos_to_zero)},
    {'Metric': 'n_shift_zero_to_positive', 'Value': len(shift_zero_to_pos)},
    {'Metric': 'H1_beta', 'Value': round(h1_beta, 4)},
    {'Metric': 'H1_SE', 'Value': round(h1_se, 4)},
    {'Metric': 'H1_p', 'Value': round(h1_p, 4)},
    {'Metric': 'H1_N', 'Value': h1_n},
    {'Metric': 'H2_beta', 'Value': round(h2_beta, 4)},
    {'Metric': 'H2_SE', 'Value': round(h2_se, 4)},
    {'Metric': 'H2_p', 'Value': round(h2_p, 4)},
    {'Metric': 'H2_N', 'Value': h2_n},
    {'Metric': 'H3_beta', 'Value': round(h3_beta, 4)},
    {'Metric': 'H3_SE', 'Value': round(h3_se, 4)},
    {'Metric': 'H3_p', 'Value': round(h3_p, 4)},
    {'Metric': 'H3_N', 'Value': h3_n},
    {'Metric': 'H3_df_resid', 'Value': h3_df_resid},
    {'Metric': 'H3_TOST_bound', 'Value': BOUND},
    {'Metric': 'H3_TOST_t_lower', 'Value': round(t_lower, 4)},
    {'Metric': 'H3_TOST_t_upper', 'Value': round(t_upper, 4)},
    {'Metric': 'H3_TOST_p_lower', 'Value': round(p_lower, 4)},
    {'Metric': 'H3_TOST_p_upper', 'Value': round(p_upper, 4)},
    {'Metric': 'H3_TOST_equivalent', 'Value': equivalent},
    {'Metric': 'baseline_full_zero_score_n', 'Value': int(zc)},
    {'Metric': 'baseline_full_zero_score_total', 'Value': int(tot)},
    {'Metric': 'baseline_full_zero_score_pct', 'Value': round(pct, 2)},
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/fragmentation_robustness.csv', index=False)
print(f"\nresults/fragmentation_robustness.csv saved -- {len(summary_df)} rows")
print("\nDone.")
