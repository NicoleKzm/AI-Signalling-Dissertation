# Verification Report — Dissertation Data & Regression Outputs

Read-only verification. No source files were modified. All computations reproduced
fresh from `panel_dataset.csv` and `signalling_scores.csv` in
`/Users/user/Desktop/VSCode Dissertation/`, using the same model spec as
`regression_clean.py` (PanelOLS, firm + year fixed effects, clustered SEs,
`log_revenue` control).

**General note before the item-by-item results:** `regression_results.csv` in this
exact folder is a **stale, pre-bugfix** file (H1 N=55, no CI columns) — a corrected
version (H1 N=69) exists in a sibling `VSCode Dissertation-clean/` folder from a
prior session. This report does not read that CSV at all — every number below is
recomputed directly from `panel_dataset.csv` — but if any appendix table was built
by hand-copying from this specific file, it will carry the stale N=55 for H1.

---

## 1. DocMorris/Boohoo exclusion sample sizes — **DISCREPANCY FOUND**

**Root cause identified.** Your two tables aren't reporting the same exclusion at
all, and one of them additionally carries a code bug that has since been fixed.

**Base sample (no exclusion):** contemporaneous H1=69, H2=55, H3=69. Lag-1: all
three =55.

| Exclusion | Contemporaneous N (H1/H2/H3) | Lag-1 N (H1/H2/H3) |
|---|---|---|
| Excl. DocMorris **2023 only** (correct, current code) | 68 / 54 / 68 | 54 / 54 / 54 |
| Excl. DocMorris **entirely** (all 5 years) | 64 / 51 / 64 | 51 / 51 / 51 |
| Excl. Boohoo entirely | 64 / 51 / 64 | 51 / 51 / 51 |
| Excl. DocMorris 2023 only + Boohoo entirely | 63 / 50 / 63 | 50 / 50 / 50 |
| Excl. DocMorris entirely + Boohoo entirely | 59 / 47 / 59 | 47 / 47 / 47 |

**Your "N=64 (H1/H3) / N=51 (H2 lag-1)" table is correct** — it matches "Excl.
DocMorris entirely" computed on the current, corrected code exactly.

**Your "N=54 across all three hypotheses" table is wrong.** I reproduced the old,
pre-bugfix version of `regression_clean.py` (which applied
`dropna(subset=['Revenue_Growth_%'])` globally, before splitting into H1/H2/H3,
instead of scoping it per-hypothesis). Under that stale logic, "Excl. DocMorris 2023
only" gives exactly **54, 54, 54** — reproducing your reported figures precisely.
Under the corrected code, that same exclusion (2023 only, not the whole firm) should
read **68, 54, 68** — not uniformly 54. H2's 54 is coincidentally correct either way
(Revenue_Growth_% already restricts H2 to a same-sized base with or without the
bug); H1 and H3's 54 are artifacts of the bug and are both wrong.

**Recommendation:** decide which exclusion you intend to report — "2023 only" (the
anomalous-year check already built into `regression_clean.py`) or "DocMorris
entirely" (a stricter, whole-firm check) — and regenerate that specific appendix
table from the current, corrected pipeline. Do not present both tables as the same
check; they use different exclusion criteria.

---

## 2. Appendix C row label — **DISCREPANCY FOUND (confirms your suspicion)**

The specification with N=41, β≈0.1452, SE≈3.1987 has:
- **DV = `Gross_Margin_%`** → this is H3's dependent variable, not H2's
- **IV = `signal_lag2`** → a two-year lag, not lag-1

Freshly reproduced: N=41, β=0.1452, SE=3.1987 — exact match to what's in your
appendix numerically, only the label is wrong.

**Correct label: "H3 lag-2 (exploratory)"**, not "H2 lag-1 (exploratory)" as
currently captioned. Fix the label; the numbers themselves are fine.

---

## 3. Zero-scoring firm-year breakdown — **PASS** (numbers computed and confirmed)

Of N=69 firm-years:
- **44 firm-years (63.8%)** have `mean_signal_score == 0.00`
- Of those 44: **21 (30.4% of N=69)** have zero classified AI passages at all
  (`total_passages == 0`, i.e. genuine "no disclosure")
- Of those 44: **23 (33.3% of N=69)** have one or more classified passages, all of
  which are Symbolic ("loud but empty")
- 21 + 23 = 44, no unexplained rows.

---

## 4. Modal classification reconciliation — **PASS**, with the detail requested

**36 firm-years** have `modal_classification == 'Symbolic'` — matches your reported
figure exactly (52.2% of 69 ≈ 36).

- **23 of the 36** have `mean_signal_score == 0` exactly (all-Symbolic) — matches
  your "23 zero-scoring" figure.
- **13 of the 36** have `mean_signal_score > 0` — modal-Symbolic but with a minority
  of Transitional (and in two cases, one Substantive) passages pulling the mean
  above zero. Confirmed. Full list, sorted by mean score:

| Firm | Year | Total passages | Symbolic | Transitional | Substantive | Mean score |
|---|---|---|---|---|---|---|
| Allegro | 2022 | 26 | 24 | 2 | 0 | 0.077 |
| HelloFresh | 2025 | 24 | 21 | 3 | 0 | 0.125 |
| Redcare Pharmacy | 2024 | 19 | 16 | 3 | 0 | 0.158 |
| Redcare Pharmacy | 2025 | 18 | 15 | 3 | 0 | 0.167 |
| Allegro | 2021 | 17 | 14 | 3 | 0 | 0.176 |
| Allegro | 2025 | 58 | 47 | 10 | 1 | 0.207 |
| Zalando | 2025 | 18 | 15 | 2 | 1 | 0.222 |
| AO World | 2022 | 6 | 4 | 2 | 0 | 0.333 |
| Boozt | 2023 | 3 | 2 | 1 | 0 | 0.333 |
| Moonpig | 2021 | 3 | 2 | 1 | 0 | 0.333 |
| Moonpig | 2022 | 6 | 4 | 2 | 0 | 0.333 |
| DocMorris | 2025 | 14 | 8 | 5 | 1 | 0.500 |
| Allegro | 2023 | 75 | 49 | 14 | 12 | 0.507 |

23 + 13 = 36. Reconciles exactly.

---

## 5. Full coefficient spot-check — **PASS**

| Spec | Reported β / SE | Freshly computed β / SE | N | Verdict |
|---|---|---|---|---|
| H1 contemporaneous | -22.624 / 35.090 | -22.6239 / 35.0903 | 69 | MATCH |
| H2 lag-1 | -1.298 / 8.137 | -1.2984 / 8.1372 | 55 | MATCH |
| H3 lag-1 | -0.832 / 1.559 | -0.8322 / 1.5588 | 55 | MATCH |

No discrepancy at any decimal place reported in your write-up. These three match
the corrected pipeline (post-dropna-fix), not the stale `regression_results.csv`
sitting in this folder — see the general note above.

---

## 6. Word count — **COULD NOT VERIFY**

No Word or markdown export of the dissertation (Chapters 1–6, or otherwise) exists
inside `/Users/user/Desktop/VSCode Dissertation/` or any subfolder — the directory
contains only Python scripts, CSVs, and figure images (`.png`/`.pdf`). I did not
guess or substitute another file.

For reference only, **outside** the directory you specified, there is a
`Dissertation Draft Reference.md` directly on the Desktop (one level up). I did not
word-count it since it wasn't in the folder you named and I can't confirm it's
scoped to Chapters 1–6 only (excluding bibliography/appendices) without you
confirming that's the right file and the right scope.

---

## Summary

| # | Check | Verdict |
|---|---|---|
| 1 | DocMorris/Boohoo exclusion N's | **DISCREPANCY FOUND** — two different exclusion definitions conflated; one also carries a fixed bug |
| 2 | Appendix C row label | **DISCREPANCY FOUND** — mislabeled H2 lag-1, should be H3 lag-2 |
| 3 | Zero-scoring breakdown | PASS |
| 4 | Modal classification reconciliation | PASS |
| 5 | Coefficient spot-check | PASS |
| 6 | Word count | COULD NOT VERIFY — no dissertation document in the specified directory |

---
---

# Follow-up — Table 4.9 regeneration & corrected word count

## 7. Table 4.9 regeneration — **REGENERATED, with a definition conflict flagged**

Before presenting the table: the follow-up request specified "H1 (contemporaneous),
H2 (lag-1), H3 (lag-1)" with a cross-check that the DocMorris-entirely row must
match Appendix B's N=64/51/64. **Those two instructions are mutually
inconsistent** — I computed both to show why.

| Definition | DocMorris-entirely N (H1/H2/H3) | Matches Appendix B (64/51/64)? |
|---|---|---|
| **A: all three hypotheses at the contemporaneous/benchmark spec** (mean_signal_score, no lag) | 64 / 51 / 64 | **YES** |
| **B: each hypothesis at its own primary spec** (H1 contemporaneous, H2 lag-1, H3 lag-1 — mixing specs, matching Table 4.6's headline convention) | 64 / 51 / 51 | NO |

Definition A is also what `regression_clean.py`'s existing "ROBUSTNESS CHECKS"
section actually computes in code — it runs all three hypotheses against
`mean_signal_score` (contemporaneous) uniformly; it has no lag-1 branch at all.
So "H3 (lag-1)" in the follow-up instructions appears to be a mis-statement,
likely carried over from Table 4.6's primary-spec convention, which does not apply
to the robustness table. **I used Definition A below**, since it's the one that
(a) matches the already-confirmed-correct Appendix B exactly and (b) matches the
actual robustness-check code in this repository. If Table 4.9 was intentionally
meant to test lag-1 specs for H2/H3, that would be a different, new check — not a
regeneration of the existing one — and Appendix B's own N=64/51/64 would need to
change too, which contradicts you calling it already confirmed correct.

**Table 4.9 (regenerated, corrected pipeline, contemporaneous/benchmark spec, full-firm exclusions):**

| Exclusion | H1 p-value | H1 N | H2 p-value | H2 N | H3 p-value | H3 N |
|---|---|---|---|---|---|---|
| Excl. DocMorris entirely | 0.6314 | 64 | 0.4583 | 51 | 0.5058 | 64 |
| Excl. Boohoo entirely | 0.5270 | 64 | 0.9469 | 51 | 0.7120 | 64 |
| Excl. DocMorris + Boohoo entirely | 0.6161 | 59 | 0.4594 | 47 | 0.5144 | 59 |

Cross-check: DocMorris-entirely row = **64 / 51 / 64** — exact match to Appendix B.

Coefficients for reference (not requested in the table format, but useful context):
DocMorris-excl row: H1 β=8.5421, H2 β=-9.6942, H3 β=-1.3179. All three hypotheses
remain not significant (p > 0.45) under every exclusion — the substantive
conclusion (null result, robust to DocMorris/Boohoo) does not change. Only the
stale N's needed fixing.

**Action needed:** replace Table 4.9 in §4.4.1 with the table above. This makes it
consistent with Appendix B and with Chapter 3's text promising a DocMorris-entirely
specification.

---

## 8. Word count — **WRONG FILE, do not use this number**

`Dissertation Draft Reference.md` is **not your dissertation text**. Its own first
line is:

> `# MODEL DISSERTATION DRAFT (REFERENCE ONLY)`

and it opens with an explicit disclaimer: *"This is a structural and stylistic
reference, not text to submit... Bracketed [INSERT] markers show where your own
numbers, citations, and judgment must go."* It is a supervisor-style scaffold —
part outline, part fully-written model prose — with per-chapter **word-count
targets**, not your actual current draft.

Section-by-section word count of this reference file, for transparency (**these
are word counts of the model/scaffold, not of your dissertation**):

| Section | Word count (this file) | Stated target (per the file's own §ing) | Status in file |
|---|---|---|---|
| Abstract | 235 | ~230 | fully written model |
| Chapter 1: Introduction | 564 | 1,100 | partial (outline + partial prose) |
| Chapter 2: Literature Review | 580 | 3,500 | mostly outline, not full prose |
| Chapter 3: Methodology | 832 | 2,800 | mostly outline, not full prose |
| Chapter 4: Findings | 305 | 2,100 | mostly outline, not full prose |
| Chapter 5: Discussion | 1,660 | ~1,900 | fully written model |
| Chapter 6: Conclusion | 527 | ~550 | fully written model |
| **Total (this file)** | **4,913** | **11,950** (target total, per the file's own word budget note) | — |

**I am not reporting 4,913 (or any number from this file) as "your dissertation's
word count"** — that would be reporting the length of a reference template, not
your submission. Chapters 1–4 in this file are mostly structural guidance and
bracketed placeholders, well short of their own stated targets, while Chapters 5–6
are marked "fully written model" and close to target — consistent with this being
a partially-worked scaffold, not draft prose.

**To get a real word count, point me at the actual current chapter files/document**
(wherever your live Chapters 1–6 draft lives — it isn't in
`/Users/user/Desktop/VSCode Dissertation/` or in this reference file). I did not
substitute the model file's own word budget targets as a stand-in for that number.

---

## Updated summary

| # | Check | Verdict |
|---|---|---|
| 1 | DocMorris/Boohoo exclusion N's | DISCREPANCY FOUND (see §1) |
| 2 | Appendix C row label | DISCREPANCY FOUND (see §2) |
| 3 | Zero-scoring breakdown | PASS |
| 4 | Modal classification reconciliation | PASS |
| 5 | Coefficient spot-check | PASS |
| 6 (orig.) | Word count, VSCode Dissertation folder | COULD NOT VERIFY — no document found |
| 7 | Table 4.9 regeneration | **DONE** — regenerated table above, definition conflict in the request flagged and resolved using the code- and Appendix-B-consistent definition |
| 8 | Word count, Dissertation Draft Reference.md | **COULD NOT VERIFY** — file is an explicitly-labeled model/reference scaffold, not the actual dissertation; real draft location still needed |

---
---

# Second follow-up — Tables 4.1–4.8 and Appendix C

## 9. Table 4.1/4.2 raw recount — **PASS, every sub-check**

All five reported figures reproduce exactly, both from `signalling_scores.csv`
(summing its count columns) and cross-checked independently against
`all_classifications.csv` (passage-level ground truth) directly:

| Metric | Reported | Recomputed | Verdict |
|---|---|---|---|
| Passage-level Symbolic/Transitional/Substantive | 396/100/16 of 512 | 396/100/16 of 512 | MATCH |
| Firm-year modal No_disclosure/Symbolic/Transitional/Substantive | 21/36/12/0 of 69 | 21/36/12/0 of 69 | MATCH |
| Passages naming an AI function (q1_score==1) | 134/512 (26.2%) | 134/512 (26.2%) | MATCH |
| Passages with measurable outcome (q2_score==1) | 20/512 (3.9%) | 20/512 (3.9%) | MATCH |
| Both (q1 AND q2) | 16 (16/134=11.9%) | 16 (16/134=11.9%) | MATCH |

Note: `signalling_scores.csv` doesn't carry per-passage q1/q2 flags, so the last
three rows required `all_classifications.csv`; the first two rows were verified
both ways and agree exactly.

---

## 10. Table 4.3/4.4 descriptive stats and correlations — **PASS**

**Descriptive statistics** (recomputed from `panel_dataset.csv`):

| Variable | Mean | SD | Min | Max | N |
|---|---|---|---|---|---|
| Signal Score | 0.1807 | 0.2983 | 0.00 | 1.125 | 69 |
| Stock Price Movement % | -13.9283 | 52.5603 | -89.33 | 182.22 | 69 |
| Revenue Growth % | 4.3411 | 20.1885 | -47.52 | 64.39 | 55 |
| Gross Margin % | 44.5348 | 17.3079 | 15.06 | 93.68 | 69 |

These N's match the reported 69/69/55/69 exactly. As a side confirmation: the SDs
here (52.56, 20.19, 17.31) are exactly the constants hardcoded in
`retest_table_4_8.py`'s `EQUIVALENCE_BOUNDS` — good internal consistency, and it
means Check 13 below is testing the same numbers this table reports.

**Pairwise Pearson correlation matrix** (signal score, log revenue, 3 outcomes):

| | Signal | log_revenue | Stock Price | Revenue Growth | Gross Margin |
|---|---|---|---|---|---|
| Signal | 1.0000 | 0.1956 | 0.0367 | -0.0163 | -0.1625 |
| log_revenue | 0.1956 | 1.0000 | -0.0763 | 0.2214 | 0.2516 |
| Stock Price | 0.0367 | -0.0763 | 1.0000 | 0.1336 | -0.0539 |
| Revenue Growth | -0.0163 | 0.2214 | 0.1336 | 1.0000 | 0.1170 |
| Gross Margin | -0.1625 | 0.2516 | -0.0539 | 0.1170 | 1.0000 |

Compare these against your Table 4.3/4.4 directly — no reported values were given
to me to diff against for this item, so I can't render a MATCH/MISMATCH verdict,
only supply the authoritative recomputed numbers.

---

## 11. Table 4.5 Hausman tests — **COULD NOT REPRODUCE — flagged for real scrutiny**

This is the most serious finding in this pass. I tried **five** distinct, each
individually standard, implementations of the Hausman specification test (FE vs
RE), comparing the {signal-score, log-revenue} coefficient vector under classical
(non-robust) covariance, as the Hausman formula requires:

| Approach | H1 chi2 (df) | H2 chi2 (df) | H3 chi2 (df) |
|---|---|---|---|
| A: entity FE/RE + year dummies, df=2 | 0.99 (2) | 3.35 (2) | 0.70 (2) |
| B: entity FE/RE only, no year dummies, df=2 | 0.11 (2) | 3.75 (2) | 1.42 (2) |
| C: two-way FE vs entity RE+year dummies, df=2 | 0.99 (2) | 3.35 (2) | 0.70 (2) |
| D: entity FE/RE, IV-only, df=1 | 0.11 (1) | 0.29 (1) | 0.23 (1) |
| E: entity FE/RE + year dummies, IV-only, df=1 | 0.94 (1) | 0.28 (1) | 0.01 (1) |
| **Reported in dissertation** | **5.46** | **4.91** | **9.66** |

None of the five approaches come close in magnitude — the reported statistics are
roughly 5–10× larger than anything I can produce from this data under any
reasonable specification. More importantly, **the ranking is inverted**: in every
one of my five approaches, H3 has the *smallest* Hausman statistic of the three;
the dissertation reports H3 as by far the *largest* (9.66, p=.008, the only one
significant at 5%). That's not a tolerance/rounding issue or an implementation
nuance (like clustered vs. classical SEs, or one-way vs two-way effects) — it's a
qualitatively different result.

**I cannot confirm these numbers from `panel_dataset.csv` under any specification I
tried.** Possible explanations, in rough order of likelihood: (a) Table 4.5 was
computed against a different/older snapshot of the data than the current
`panel_dataset.csv`; (b) it used a specific canned routine (e.g. Stata's
`hausman`/`xtoverid` with non-default options) that differs from all five variants
above in a way I haven't hit on; (c) the numbers are placeholder/illustrative and
were never actually computed from this dataset. I'd treat this table as unverified
until you can either point me to the exact method used or confirm which data
snapshot it came from — this is worth resolving before submission, more so than
items 1/2/7 above, since I have no working theory that reproduces it.

---

## 12. Table 4.6 benchmark specs — **PASS**

| Spec | Reported β / SE / N | Recomputed β / SE / N | Verdict |
|---|---|---|---|
| H1 lag-1 (benchmark) | -10.265 / 26.098 / 55 | -10.2646 / 26.0976 / 55 | MATCH |
| H2 contemporaneous (benchmark) | -0.052 / 12.831 / 55 | -0.0519 / 12.8313 / 55 | MATCH |
| H3 contemporaneous (benchmark) | -0.535 / 1.756 / 69 | -0.5350 / 1.7557 / 69 | MATCH |

Combined with the primary specs already confirmed in Check 5, all six Table 4.6
rows now reproduce exactly.

---

## 13. Table 4.8 power and equivalence — **PASS**

| Hypothesis | Reported MDE | Recomputed MDE | Verdict |
|---|---|---|---|
| H1 (contemporaneous, primary) | 99.8pp | 99.79pp | MATCH |
| H2 (lag-1, primary) | 23.2pp | 23.23pp | MATCH |
| H3 (lag-1, primary) | 4.5pp | 4.45pp | MATCH |

**H3 TOST equivalence:** gross margin SD = 17.3079 → bound = 0.2×SD = 3.4616
(reported ≈3.46, MATCH). p_lower = 0.0488, p_upper = 0.0040 (both < 0.05) →
**equivalence confirmed** — matches "equivalent to zero" exactly, and matches the
.049/.004 figures already confirmed independently earlier in this project.

---

## 14. Appendix C remaining rows — **values supplied, no verdict possible**

No reported target values were given for this item, so I can't produce
MATCH/MISMATCH — only the authoritative freshly-computed figures for you to diff
directly against Tables C.1–C.6. (Signal-score β/SE/p for all six specs were
already confirmed in Checks 5 and 12.)

| Spec | LogRev β | LogRev SE | LogRev p | R²(within) | R²(between) | R²(overall) | F-stat | F p-value | N |
|---|---|---|---|---|---|---|---|---|---|
| H1 contemporaneous (primary) | -10.4367 | 15.4173 | 0.5016 | -0.0145 | -0.6361 | -0.0506 | 0.4848 | 0.6187 | 69 |
| H1 lag-1 (benchmark) | -14.9019 | 27.5156 | 0.5914 | -0.0021 | -0.5415 | -0.0451 | 0.1102 | 0.8959 | 55 |
| H2 contemporaneous (benchmark) | 28.0638 | 10.4208 | 0.0107 | 0.1163 | -3.8884 | -1.7288 | 2.3471 | 0.1101 | 55 |
| H2 lag-1 (primary) | 28.0236 | 10.7138 | 0.0129 | 0.1171 | -3.8443 | -1.7080 | 2.3551 | 0.1093 | 55 |
| H3 contemporaneous (benchmark) | -3.2377 | 2.4557 | 0.1935 | 0.0497 | -0.1490 | -0.1421 | 1.3955 | 0.2574 | 69 |
| H3 lag-1 (primary) | -1.0928 | 3.9322 | 0.7827 | -0.0047 | -0.0405 | -0.0393 | 0.1423 | 0.8678 | 55 |

If Appendix C also has a row for H3 lag-2 (exploratory, N=41), that was already
computed in the earlier "Appendix A" extraction this project produced; ask if you
want it re-supplied here too.

---

## Final summary (all 14 checks)

| # | Check | Verdict |
|---|---|---|
| 1 | DocMorris/Boohoo exclusion N's | DISCREPANCY FOUND — resolved in §1/§7 |
| 2 | Appendix C row label | DISCREPANCY FOUND — mislabeled, corrected in §2 |
| 3 | Zero-scoring breakdown | PASS |
| 4 | Modal classification reconciliation | PASS |
| 5 | Coefficient spot-check (primary specs) | PASS |
| 6 | Word count, VSCode Dissertation folder | COULD NOT VERIFY — no document found |
| 7 | Table 4.9 regeneration | DONE — see §7 |
| 8 | Word count, Dissertation Draft Reference.md | COULD NOT VERIFY — wrong file (model/reference scaffold) |
| 9 | Table 4.1/4.2 raw recount | PASS |
| 10 | Table 4.3/4.4 descriptive stats & correlations | PASS (no reported values supplied to diff against) |
| 11 | Table 4.5 Hausman tests | **COULD NOT REPRODUCE — needs real scrutiny before submission** |
| 12 | Table 4.6 benchmark specs | PASS |
| 13 | Table 4.8 power and equivalence | PASS |
| 14 | Appendix C remaining fields | Values supplied, no target given to verify against |
