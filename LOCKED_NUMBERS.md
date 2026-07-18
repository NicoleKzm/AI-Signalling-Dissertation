# LOCKED NUMBERS — Dissertation
Last verified: 14 July 2026. Every figure below traces to a script in this directory.

## PRIMARY REGRESSIONS (regression_clean.py → regression_results.csv)
H1 (NEW primary, lag-1):   β = -11.7093, SE = 29.0452, p = .689, N = 54, df_resid = 35
H1 (supplementary, contemp): β = -22.6239, SE = 35.0903, p = .522, N = 69
H2 (lag-1):                β = -5.0803,  SE = 8.4171,  p = .550, N = 54
H3 (lag-1):                β = -0.6904,  SE = 1.4646,  p = .640, N = 54, df_resid = 35
All: PanelOLS, entity + time FE, firm-clustered SE, lagged log revenue control.
Zalando 2025 excluded from H2/H3 (symmetric acquisition rule).
UPDATE: H1 lag-1 (β=-11.7093, SE=29.0452, N=54, p=.6893) and the DocMorris-
entirely H2/H3 exclusion are now BOTH computed directly inside
regression_clean.py itself (previously only in h1_lag_primary.py and
chapter4_gaps.py respectively) -- added as pure ADDITIONS, no existing code
in the script touched. Both reproduce their prior values exactly. See the
Table 4.9 note below for a draft-citation correction found while adding the
DocMorris-entirely check.

## TOST — H3 (tost_mde_v2.py)
Bound: ±3.462 pp = 0.2 × SD(Gross_Margin_%) = 0.2 × 17.308.
**NOT pre-specified. The bound is a conventional small-effect benchmark
(0.2 x SD of the outcome), applied after estimation. The equivalence analysis
is EXPLORATORY, not confirmatory. Established by git forensic audit, 14 July
2026: no pre-registration or standalone statement of the bound exists in
either repo's history; the bound and the coefficients first co-occur in
retest_table_4_8.py with the coefficients hardcoded as inputs.**
Conventional (df=35):      p_lower = .0334, p_upper = .0038  → EQUIVALENT
CR2 / Bell-McCaffrey (df=4.76): p_lower = .0463, p_upper = .0136 → EQUIVALENT (marginal)
Wild cluster bootstrap (9,999): p_lower = .0646, p_upper = .0022 → NOT EQUIVALENT

## FINAL H3 CLAIM
The predicted POSITIVE effect is rejected under EVERY procedure: conventional (p=.004),
CR2 (p=.014), wild bootstrap (p=.002), all 14 leave-one-out iterations, all 5 aggregation rules.
The two-sided equivalence claim is NOT sustained — it fails under the wild bootstrap on the
lower bound. State in ONE direction only: a positive association of the same conventional
magnitude can be excluded; a negative association of comparable magnitude cannot. Both
claims are reported within an EXPLORATORY equivalence framework (see Bound note above).

## MINIMUM DETECTABLE EFFECTS (80% power)
H1 (lag-1 primary): 83.71 pp = 601% of DV mean
H2: 24.26 pp = 559% of DV mean
H3: 4.22 pp = 9.5% of DV mean

## RANDOMISATION INFERENCE (supplementary) (randomisation_inference.py)
Lag-1 signal, lag-1 log revenue control, Zalando 2025 excluded, N=54. N_PERM=1000, seed=42.
H1: coef -11.7093, clustered_p .6893, randomisation_p .7010
H2: coef -5.0803,  clustered_p .5500, randomisation_p .5790
H3: coef -0.6904,  clustered_p .6403, randomisation_p .7090
RI is SUPPLEMENTARY. Primary small-sample inference is CR2 and the wild cluster
bootstrap (small_sample_inference.py). RI assumes exchangeability, which is not
guaranteed in an observational panel.

## FISCAL YEAR END ROBUSTNESS (fye_robustness.py)
Stock_Price_Movement_% (H1's DV) is computed on calendar-year windows
(1 Jan-31 Dec, financial_data.py). 5 firms have confirmed non-December FYE:
AO World (March), ASOS (Aug/Sep), About You (February), Moonpig (April),
Mytheresa (June) -- 23 firm-year observations. This misaligns signal and
outcome windows for H1 only -- H2/H3 draw both from the same annual report,
so both are already on the firm's fiscal clock.
Current lag-1 primary spec, re-estimated excluding these 5 firms (9 of 14
firms retained, N=35 vs full N=54):
H1: full beta=-11.7093, SE=29.0452, p=.689, N=54, 14 firms
    excl. non-Dec FYE: beta=-17.0316, SE=30.8510, p=.587, N=35, 9 firms
H2 (unaffected, reported for completeness):
    full beta=-5.0803, p=.550, N=54  |  excl.: beta=2.9275, p=.796, N=35
H3 (unaffected, reported for completeness):
    full beta=-0.6904, p=.640, N=54  |  excl.: beta=0.8401, p=.551, N=35
H1's null is NOT an artefact of the calendar/fiscal misalignment: the
coefficient sign is unchanged (negative), magnitude increases rather than
shrinks, and the result remains far from significant (p=.587). H2/H3 sign-flip
under this exclusion, consistent with a small subsample (N=35, 9 firms) rather
than any real effect -- underscores these estimates are noisy at this scale,
not evidence of anything for H2/H3 specifically.

## CHAPTER 4 GAPS (chapter4_gaps.py)
Four numeric gaps closed for Ch4, current lag-1 primary spec (N=54 uniform).

**Descriptives (Table 4.3 rebuild):** full 69-obs panel vs. current N=54 sample.
  Full panel: signal N=69 mean=.1807 SD=.2983; Stock Price N=69 mean=-13.93 SD=52.56;
  Revenue Growth N=55 mean=4.34 SD=20.19 (naturally missing each firm's 1st year);
  Gross Margin N=69 mean=44.53 SD=17.31.
  Current N=54 sample: signal mean=.1989 SD=.3064; Stock Price mean=-8.71 SD=57.01;
  Revenue Growth mean=4.11 SD=20.31; Gross Margin mean=44.50 SD=17.00. All 4 vars
  are N=54 uniformly in the current sample -- the lag-1 requirement (drops each
  firm's 1st year) coincides exactly with Revenue_Growth_%'s own missingness.

**Correlation matrix (Table 4.4 rebuild):** Pearson, pairwise, current N=54 sample,
  every cell N=54 (no additional missingness beyond the sample restriction itself).
  signal-logrev=.197, signal-stockprice=-.027, signal-revgrowth=-.017,
  signal-grossmargin=-.225, logrev-grossmargin=.267. Full matrix in chapter4_gaps.csv.

**H1 lag-1 95% CI** (was missing from the PRIMARY REGRESSIONS entry above):
  Conventional (df_resid=35): [-70.6743, 47.2557]
  CR2/Bell-McCaffrey (df=4.76, read from small_sample_inference.csv, not
  recomputed -- that script already validates this figure): [-82.9267, 59.5081]

**Table 4.9 rebuild** (lag-1 primary spec, all 3 hypotheses, 4 exclusions):
  Excl. DocMorris 2023 only (N=53): H1 beta=8.2801 p=.718 | H2 beta=-4.6263 p=.606 | H3 beta=-0.5725 p=.717
  Excl. DocMorris entirely (N=50):  H1 beta=12.8619 p=.688 | H2 beta=-13.0608 p=.119 | H3 beta=-1.1504 p=.556
  Excl. Boohoo entirely (N=50):     H1 beta=-13.4510 p=.659 | H2 beta=-7.8058 p=.280 | H3 beta=-0.9386 p=.495
  Excl. both entirely (N=46):       H1 beta=12.3997 p=.715 | H2 beta=-15.3376 p=.003 | H3 beta=-1.1542 p=.541
  DocMorris sign-flip on H1 (full-firm exclusion): CONFIRMED REPRODUCES here
  (-11.7093 -> +12.8619), matching the leave-one-out section above. Note: H2
  becomes significant (p=.003) excluding both DocMorris and Boohoo (N=46) --
  flagged for awareness, not otherwise interpreted here. Small-sample follow-up
  below.
**DRAFT CORRECTION (found while adding this exclusion to regression_clean.py):**
if the dissertation text cites H2 beta=-15.338, N=46, p=.003 under a
"DocMorris entirely" label, THAT IS WRONG -- -15.338/N=46/.003 is the
DocMorris-AND-Boohoo-BOTH-entirely-excluded result ("Excl. both entirely"
row above), not DocMorris alone. DocMorris excluded ALONE gives H2
beta=-13.0608, SE=8.1618, N=50, p=.119 ("Excl. DocMorris entirely" row
above) -- not significant. Re-verified fresh from regression_clean.py's
newly-added check: beta=-13.0608, SE=8.1618, N=50, p=0.1194, exact match.
Fix the citation/label in the draft, not the number.

## TABLE 4.8 — COMPLETE ROBUSTNESS TABLE (regression_clean.py, all 6 rows x 3 hyps)
H1 basis: contemporaneous, mean_signal_score, log_revenue, full sample, NO
Zalando 2025 exclusion. H2/H3 basis: lag-1 signal, lag-1 log_revenue,
Zalando 2025 excluded. All previously missing cells (H1 "DocMorris entirely"
alone; H1/H2/H3 "DocMorris entirely + Boohoo entirely") added to
regression_clean.py this session, same additive pattern as before -- nothing
existing changed.

1. Full sample (no exclusion):
   H1 (contemp): beta=-22.6239, SE=35.0903, N=69, p=.5221
   H1 (lag-1):   beta=-11.7093, SE=29.0452, N=54, p=.6893
   H2: beta=-5.0803, SE=8.4171, N=54, p=.5500
   H3: beta=-0.6904, SE=1.4646, N=54, p=.6403
2. DocMorris 2023 only:
   H1: beta=-6.7035, SE=23.5510, N=68, p=.7771
   H2: beta=-4.6263, SE=8.8842, N=53, p=.6059
   H3: beta=-0.5725, SE=1.5633, N=53, p=.7165
3. DocMorris entirely (all 5 years):
   H1: beta=8.5421, SE=17.6856, N=64, p=.6314
   H2: beta=-13.0608, SE=8.1618, N=50, p=.1194
   H3: beta=-1.1504, SE=1.9333, N=50, p=.5560
4. Boohoo entirely:
   H1: beta=-22.7685, SE=35.7147, N=64, p=.5270
   H2: beta=-7.8058, SE=7.1033, N=50, p=.2800
   H3: beta=-0.9386, SE=1.3607, N=50, p=.4953
5. DocMorris 2023 + Boohoo entirely:
   H1: beta=-6.3530, SE=23.7882, N=63, p=.7907
   H2: beta=-7.6041, SE=7.3572, N=49, p=.3093
   H3: beta=-0.8812, SE=1.4560, N=49, p=.5494
6. DocMorris entirely + Boohoo entirely:
   H1: beta=9.1138, SE=18.0387, N=59, p=.6161
   H2: beta=-15.3376, SE=4.7414, N=46, p=.0030  <-- ONLY significant cell (p<.05)
   H3: beta=-1.1542, SE=1.8667, N=46, p=.5412
CONFIRMED: exactly one cell across all 18 (6 rows x 3 hyps) is significant at
5% -- row 6, H2. Every other cell has p > .10.

## H2 EXCLUSION SMALL-SAMPLE CHECK (h2_exclusion_smallsample.py)
chapter4_gaps.csv's H2, lag-1 primary, excl. DocMorris+Boohoo entirely (N=46,
G=12 clusters) is the only significant coefficient anywhere in the analysis
(naive PanelOLS-clustered p=.0030). Re-estimated under small-sample-robust
inference, same procedures as small_sample_inference.py (CR2/Bell-McCaffrey,
wild cluster bootstrap, B=9,999, Rademacher, seed=42):
beta=-15.3376, N=46, G=12 clusters
  CR1 (proper df=G-1=11):        SE=4.8982, p=.0096
  CR2 + Bell-McCaffrey (df=3.87): SE=4.2492, p=.0239
  Wild cluster bootstrap:                    p=.0268
FINDING SURVIVES all three small-sample-robust corrections (all p < .05), but
the p-value is considerably more conservative than the naive .0030 once cluster
count (G=12) is properly accounted for -- inflates .0030 -> .0096 -> .0239 ->
.0268 across increasingly conservative methods. Report as significant but
borderline under a 12-cluster sample, not as a strong result.

## LEAVE-ONE-OUT (leave_one_out_primary.py)
H3 equivalence holds 9/14. Breaks: ASOS, Allegro, Boozt, DocMorris, Mytheresa — all on lower bound.
UPPER bound holds 14/14.
Sign flips: Moonpig (H3), DocMorris (H1, under both timings).

## AGGREGATION SENSITIVITY (aggregation_sensitivity.py)
H3 equivalence holds under mean, max, modal, passage_count, log_passage_count.
BREAKS when zero-passage firm-years excluded (p_lower = .256, N=39).
KEY: corr(mean_signal_score, passage_count) = 0.16 — volume nearly uncorrelated with substantiation.

## TWO-PART SPECIFICATION (two_part_specification.py)
Null on BOTH margins, all three hypotheses. discloses_ai: 48/69 = 1, 21/69 = 0.
Intensive margin N = 34 (H2/H3) — very small.

## VALIDATION
Held-out κ = 0.822 (unweighted), 0.837 (linear-weighted), 92.2% raw, n = 51.
  CORRECTED from 0.785: the prior value used sklearn's default alphabetical
  label ordering ("Substantive, Symbolic, Transitional"), which corrupts
  linear/quadratic distance weighting. Recomputed with explicit ordinal
  ordering (Symbolic < Transitional < Substantive) gives 0.837. Unweighted
  kappa (0.822) is unaffected by this bug. Verified against a manual
  weighted-kappa formula and consistent with the second-coder figures below
  (0.833 linear, 0.881 quadratic), computed the same corrected way.
  4 disagreements, all one-tier adjacent, model always assigns HIGHER tier.
Second coder κ = 0.791 (unweighted), 0.833 (linear-weighted), 86.2% raw, n = 29.
  4 disagreements, ALL Substantive→Transitional, ALL Allegro 2023. Second coder stricter.
Calibration: κ = 0.218 → 0.856 (in-sample, cannot establish out-of-sample performance).

## EXTRACTION RECALL (extraction_recall.py)
Stratum (a) Tier-2-only: 7/20 = 35% false-negative rate.
Stratum (b) baseline:    0/40 = 0% (after REC046 recoded to NO, documented).
Corpus-wide recall: point est. 49%, 95% CI [2.5%, 65%] — TOO IMPRECISE. DO NOT REPORT.
FINDING: all 7 misses use hyphenated compounds (AI-powered, AI-driven, AI-based)
with no Tier 1 anchor. Systematic, characterisable blind spot.

## FRAGMENTATION (fragmentation heuristic, all_classifications.csv)
Flagged: 78.1% Symbolic / 19.0% Transitional / 2.8% Substantive
Unflagged: 76.6% / 19.9% / 3.4%
→ Classification INVARIANT to fragmentation. Model saw 1500 chars; CSV stores 300.

## CORRECTIONS TO THE DOCUMENT
- Extraction rule: Tier 1 ALONE admits. Tier 2 only TAGS. Ch3 §3.5 currently says otherwise. WRONG.
- 514 extracted, 512 classified. 2 lost to silent API/parse failure. Disclose.
- Zero-score share = 63.8% (44/69). §4.2.1's "64.8%" is WRONG.
- Two-year lag (N=41) DOES NOT EXIST. Delete every reference.
- 65.2% fragmentation figure: NEVER COMPUTED. Deleted.
- Within-firm SD = 0.2376 EXCEEDS between-firm SD = 0.1851. 12 of 14 firms change modal
  classification (all except ASOS and Boohoo). VERIFIED from signalling_scores.csv.
  Limitation 7 ('limited within-firm variation') is FALSE as written — invert it.
- **TOST bound provenance**: Bound is NOT pre-specified. Label equivalence analysis
  EXPLORATORY in Ch3 §3.8, Ch4 §4.3.4, Ch5 §5.3. Strike "affirmatively established" /
  "genuine equivalence to zero" / "affirmatively supported" from the Abstract, §4.5 and §6.1.

## HAUSMAN TESTS
H2: χ² = 10.24, p = .006 — VERIFIED, reproduced from regression_clean.py
H3: χ² = 0.63,  p = .730 — VERIFIED, reproduced from regression_clean.py
H1: χ² = 1.22,  p = .545 — computed today on the CURRENT lag-1 primary specification.
    No prior H1 Hausman script exists in this directory or its git history.
    The figure χ² = 1.50, p = .472 previously cited in the dissertation CANNOT BE REPRODUCED
    and its provenance is unknown. It must NOT be cited. Use 1.22 / .545, which traces to
    a script in this directory and matches the specification actually estimated.

## DO NOT
- Re-run classify.py. It is frozen. Re-running invalidates every κ and every locked statistic.
- Work in any folder other than /Users/user/Desktop/VSCode Dissertation/
- **Do not describe the TOST bound as pre-specified or confirmatory anywhere.**
- **Do not cite or run validity_check.py. Deprecated — hardcoded to the old contemporaneous
  specification. Superseded by randomisation_inference.py and small_sample_inference.py.**

## ⚠️ UNVERIFIED — DO NOT CITE UNTIL CONFIRMED
These figures appear in the dissertation or in working notes but have NOT been
traced to a script in this directory. Verify or remove before submission.

- χ² = 1.50, p = .472 (H1 Hausman): UNREPRODUCIBLE, provenance unknown. Do not cite.
  Superseded by χ² = 1.22, p = .545 on the current lag-1 specification.
