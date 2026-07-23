LOCKED NUMBERS
Authoritative record of every statistic reported in the dissertation. Each entry names the script that produces it. If a number in the write-up disagrees with this file, the write-up is wrong.

Last verified: 14 July 2026.

1. Primary regressions
regression_clean.py → results/regression_results.csv

PanelOLS, entity + time fixed effects, firm-clustered SE, lagged log revenue control. Zalando 2025 excluded from the lag-1 samples (symmetric acquisition rule, §3.3.1).

Hypothesis	Spec	β	SE	p	N
H1	lag-1 (primary)	-11.7093	29.0452	.6893	54
H1	contemporaneous (supplementary)	-22.6239	35.0903	.5221	69
H2	lag-1 (primary)	-5.0803	8.4171	.5500	54
H3	lag-1 (primary)	-0.6904	1.4646	.6403	54
Residual df = 35 for the lag-1 primary specifications.

H1 lag-1 95% CI: conventional (df=35) [-70.6743, 47.2557]; CR2/Bell-McCaffrey (df=4.76) [-82.9267, 59.5081].

2. Small-sample inference
small_sample_inference.py

Bell-McCaffrey effective df = 4.76 across all three specifications (nominal 35). Wild cluster bootstrap: Rademacher weights, 9,999 replications, seed 42. No coefficient approaches significance under CR2 or the bootstrap.

Randomisation inference (supplementary)
randomisation_inference.py — N_PERM = 1000, seed 42.

Hypothesis	coef	clustered p	randomisation p
H1	-11.7093	.6893	.7010
H2	-5.0803	.5500	.5790
H3	-0.6904	.6403	.7090
RI is supplementary only. It assumes exchangeability, which an observational panel with a serially correlated regressor does not guarantee. Where RI and the bootstrap diverge, the bootstrap is preferred.

3. Power and equivalence
Minimum detectable effects (80% power)
tost_mde_v2.py

Hypothesis	MDE	As % of DV mean
H1 (lag-1)	83.71 pp	601%
H2	24.26 pp	559%
H3	4.22 pp	9.5%
TOST equivalence — H3
Bound: ±3.462 pp = 0.2 × SD(Gross_Margin_%) = 0.2 × 17.308.

Procedure	p_lower	p_upper	Equivalent?
Conventional (df=35)	.0334	.0038	Yes
CR2 / Bell-McCaffrey (df=4.76)	.0463	.0136	Yes (marginal)
Wild cluster bootstrap (9,999)	.0646	.0022	No
Bound provenance — carry this caveat wherever the bound is cited. The bound is NOT pre-specified. It is a conventional small-effect benchmark (0.2 × SD of the outcome) applied after estimation. Established by git forensic audit, 14 July 2026: no pre-registration or standalone statement of the bound exists in either repo's history; the bound and the coefficients first co-occur in retest_table_4_8.py with the coefficients hardcoded as inputs. The equivalence analysis is exploratory, not confirmatory. Never describe it as pre-specified.

Final H3 claim, state in one direction only. The predicted positive effect is rejected under every procedure (conventional p=.004, CR2 p=.014, bootstrap p=.002), across all 14 leave-one-out iterations and all 5 aggregation rules. The two-sided equivalence claim is NOT sustained — it fails on the lower bound under the bootstrap. A positive association of the stated magnitude can be excluded; a negative association of comparable magnitude cannot.

4. Hausman tests
regression_clean.py

Hypothesis	χ²	p	Preferred
H1	1.22	.545	RE favoured — FE retained on conceptual grounds
H2	10.24	.006	FE favoured statistically
H3	0.63	.730	RE favoured — FE retained on conceptual grounds
H1 is computed on the current lag-1 primary specification. The previously circulated figure χ² = 1.50, p = .472 cannot be reproduced, has unknown provenance, and must not be cited.

5. Descriptives
chapter4_gaps.py

Full 69-obs panel: signal N=69 mean=.1807 SD=.2983 · Stock price N=69 mean=-13.93 SD=52.56 · Revenue growth N=55 mean=4.34 SD=20.19 (each firm's first year is naturally missing) · Gross margin N=69 mean=44.53 SD=17.31.

Estimation sample (N=54, uniform across all four variables): signal mean=.1989 SD=.3064 · Stock price mean=-8.71 SD=57.01 · Revenue growth mean=4.11 SD=20.31 · Gross margin mean=44.50 SD=17.00.

The lag-1 requirement drops each firm's first year, which coincides exactly with Revenue_Growth_%'s own missingness — hence uniform N=54.

Correlations (Pearson, N=54): signal–logrev .197 · signal–stockprice -.027 · signal–revgrowth -.017 · signal–grossmargin -.225 · logrev–grossmargin .267. Full matrix in results/chapter4_gaps.csv.

Distribution: zero-scoring share = 63.8% (44/69). 514 passages extracted, 512 classified; 2 lost to a silent API/parse failure.

Identifying variation: within-firm SD = 0.2376 exceeds between-firm SD = 0.1851. 12 of 14 firms change modal classification (all except ASOS and Boohoo).

6. Robustness — sample composition
regression_clean.py

Basis note. H1 rows below run on the contemporaneous full-sample spec; H2/H3 rows run on the lag-1 spec with Zalando 2025 excluded. H1 lag-1 exclusion figures are in §7.

Exclusion	H1 β (SE) [p], N	H2 β (SE) [p], N	H3 β (SE) [p], N
Full sample	-22.6239 (35.0903) [.5221], 69	-5.0803 (8.4171) [.5500], 54	-0.6904 (1.4646) [.6403], 54
DocMorris 2023 only	-6.7035 (23.5510) [.7771], 68	-4.6263 (8.8842) [.6059], 53	-0.5725 (1.5633) [.7165], 53
DocMorris entirely	8.5421 (17.6856) [.6314], 64	-13.0608 (8.1618) [.1194], 50	-1.1504 (1.9333) [.5560], 50
Boohoo entirely	-22.7685 (35.7147) [.5270], 64	-7.8058 (7.1033) [.2800], 50	-0.9386 (1.3607) [.4953], 50
DocMorris 2023 + Boohoo entirely	-6.3530 (23.7882) [.7907], 63	-7.6041 (7.3572) [.3093], 49	-0.8812 (1.4560) [.5494], 49
DocMorris entirely + Boohoo entirely	9.1138 (18.0387) [.6161], 59	-15.3376 (4.7414) [.0030], 46	-1.1542 (1.8667) [.5412], 46
Exactly one of the 18 cells is significant at 5%: row 6, H2. Every other cell has p > .10.

Label warning. The significant result (-15.3376, N=46, p=.003) belongs to DocMorris-AND-Boohoo-both-entirely-excluded. DocMorris excluded alone gives β=-13.0608, SE=8.1618, N=50, p=.1194 — not significant. Do not attach the significant figure to a "DocMorris entirely" label.

Small-sample check on the one significant cell
h2_exclusion_smallsample.py — N=46, G=12 clusters.

CR1 (df=G-1=11): SE=4.8982, p=.0096 · CR2/Bell-McCaffrey (df=3.87): SE=4.2492, p=.0239 · wild cluster bootstrap: p=.0268.

Survives all three corrections, but the p-value inflates from .0030 → .0096 → .0239 → .0268 as cluster count is properly accounted for. Report as significant but borderline, not strong.

7. Robustness — H1 lag-1, leave-one-out, fiscal year
regression_clean.py, leave_one_out_primary.py, fye_robustness.py

H1 lag-1 exclusions: DocMorris 2023 only β=8.2801, SE=22.7357, N=53, p=.7180 · DocMorris entirely β=12.8619, SE=31.7558, N=50, p=.6882 · Boohoo entirely β=-13.4510, SE=30.1772, N=50, p=.6588 · DocMorris 2023 + Boohoo entirely β=6.8489, SE=23.6229, N=49, p=.7738 · both entirely β=12.3997, SE=33.6602, N=46, p=.7153. None significant.

Leave-one-out (H1 lag-1): full-sample β=-11.7093, N=54. Range -25.2573 (drop Zalando) to +12.8619 (drop DocMorris). DocMorris is the only firm that flips H1's sign. No iteration reaches p<.05.

Leave-one-out (H3): full-sample β=-0.6904. Dropping Moonpig gives β=+0.4282, SE=0.9919, N=50, p=.6688. Moonpig is the only firm that flips H3's sign.

H3 equivalence under LOO: holds 9/14. Breaks on ASOS, Allegro, Boozt, DocMorris, Mytheresa — all on the lower bound. The upper bound holds 14/14.

Fiscal-year-end check. Stock_Price_Movement_% uses calendar-year windows. Five firms have non-December FYE (AO World, ASOS, About You, Moonpig, Mytheresa), 23 firm-year observations. This affects H1 only, since H2/H3 draw both signal and outcome from the same report. Excluding those five firms: H1 β=-17.0316, SE=30.8510, N=35, 9 firms retained, p=.587. The coefficient moves further from zero, so H1's null is not an artefact of the misalignment. H2 (β=2.9275, p=.796) and H3 (β=0.8401, p=.551) sign-flip under this exclusion, consistent with a 35-observation subsample rather than any real effect.

8. Robustness — aggregation and two-part specification
aggregation_sensitivity.py, two_part_specification.py

H3 equivalence holds under mean, max, modal, passage_count and log_passage_count aggregation. It breaks when zero-passage firm-years are excluded (p_lower = .256, N=39).

corr(mean_signal_score, passage_count) = 0.16 — volume is nearly uncorrelated with substantiation.

Two-part specification returns null on both margins for all three hypotheses. discloses_ai: 48/69 = 1, 21/69 = 0. Intensive-margin N = 34 (H2/H3), very small.

9. Validation
kappa_retest_v3.py, second_coder_kappa.py, extraction_recall.py

Held-out agreement (n=51): κ = 0.822 unweighted, 0.837 linear-weighted, 92.2% raw. Four disagreements, all one-tier adjacent, model always assigns the higher tier.

Correction on record: an earlier value of 0.785 used sklearn's default alphabetical label ordering, which corrupts distance weighting. Recomputed with explicit ordinal ordering (Symbolic < Transitional < Substantive) gives 0.837. The unweighted κ is unaffected.

Second coder (n=29): κ = 0.791 unweighted, 0.833 linear-weighted, 86.2% raw. Four disagreements, all Substantive→Transitional, all Allegro 2023. Second coder is stricter.

Calibration: κ = 0.218 → 0.856 across three rounds. In-sample only; establishes nothing about out-of-sample performance.

Extraction recall: Tier-2-only stratum 7/20 = 35% false-negative rate; baseline stratum 0/40. All seven misses use hyphenated compounds (AI-powered, AI-driven, AI-based) with no Tier 1 anchor — a systematic, characterisable blind spot. Corpus-wide recall point estimate 49%, 95% CI [2.5%, 65%] — too imprecise, do not report.

Fragmentation: flagged 78.1% / 19.0% / 2.8% against unflagged 76.6% / 19.9% / 3.4%. Classification is invariant to fragmentation. The model saw 1,500 characters; the CSV stores 300, so the flag over-identifies fragmentation and this is a lower bound on robustness.

Extraction rule: Tier 1 alone admits a passage. Tier 2 only tags.

Rules
Do not re-run classify.py. It is frozen. Re-running invalidates every κ and every locked statistic downstream.
Do not describe the TOST bound as pre-specified or confirmatory anywhere in the dissertation.
Do not cite the H1 Hausman figure χ² = 1.50, p = .472. Unreproducible, provenance unknown. Use 1.22 / .545.
Do not cite a two-year lag specification (N=41). It does not exist.
Do not cite or run validity_check.py. Deprecated, hardcoded to the old contemporaneous specification. Superseded by randomisation_inference.py and small_sample_inference.py.


