# LOCKED NUMBERS — Dissertation
Last verified: 14 July 2026. Every figure below traces to a script in this directory.

## PRIMARY REGRESSIONS (regression_clean.py → regression_results.csv)
H1 (NEW primary, lag-1):   β = -11.7093, SE = 29.0452, p = .689, N = 54, df_resid = 35
H1 (supplementary, contemp): β = -22.6239, SE = 35.0903, p = .522, N = 69
H2 (lag-1):                β = -5.0803,  SE = 8.4171,  p = .550, N = 54
H3 (lag-1):                β = -0.6904,  SE = 1.4646,  p = .640, N = 54, df_resid = 35
All: PanelOLS, entity + time FE, firm-clustered SE, lagged log revenue control.
Zalando 2025 excluded from H2/H3 (symmetric acquisition rule).

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
Held-out κ = 0.822 (unweighted), 0.785 (linear-weighted), 92.2% raw, n = 51.
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
