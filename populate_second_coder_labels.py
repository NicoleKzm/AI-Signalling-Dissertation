"""
populate_second_coder_labels.py

Parses the second coder's raw working notes into a clean second_coder_label
field (Symbolic / Transitional / Substantive only), verifies row-by-row that
second_coder_sample.csv's shuffle order actually corresponds to the passage
text the coder was shown, then populates second_coder_label / coder_notes /
excluded_flag / exclusion_reason in second_coder_sample.csv.

STOPS without writing anything if any row fails to parse to one of the three
permitted tiers (other than the pre-specified row 20 exclusion), or if any
row's text does not match between all_classifications.csv and what the coder
was shown.

Does not modify classify.py, regression_clean.py, all_classifications.csv,
or regression_results.csv.
"""
import sys

import pandas as pd

VALID_TIERS = {'symbolic': 'Symbolic', 'transitional': 'Transitional', 'substantive': 'Substantive'}

# ══════════════════════════════════════════════════════════════════
# Raw data as pasted: (shuffle_row_1_30, passage_text_shown_to_coder, raw_label)
# ══════════════════════════════════════════════════════════════════
RAW = [
(1, "As a crucial first step towards achieving efficient and accurate compliance across multiple European markets, our solution leverages machine learning to automate the classification of our vast product assortment — approximately 18 million articles. This automated approach is essential for overcoming", "Substantive"),
(2, "It offers opportunities for platform evolves constantly, ensuring that every ferences where they share Allegro`s digitalization women to retrain, start businesses, and expand their online interaction is intuitive and swift, thus meeting Machine Learning Research is Allegro’s R&D lab journey and AI a", "Transitional "),
(3, "action – from AI adoption training to enhanced onboarding. By asking questions, listening, and acting on what we hear, we continue building trust and transparency across the organisation. We embrace the chaos.", "Symbolic"),
(4, "Generative AI, to enhance the accuracy and detail Combined with the proven impact of a credible GRI Standard/ of product descriptions in offers on the platform. promise of fast delivery on conversion and GMV, other source Disclosure Location Good descriptions available in the catalogue will this res", "Transitional "),
(5, "DocMorris | Annual Report 2025 | At a Glance DocMorris at a glance 2025 facts and figures + 11.1 % revenue DocMorris AI Assistant Growth across all business units already used by one out of three app users – just three months after launch 12.2 million + 33.2 % Rx revenue active customers, therof wit", "Substantive"),
(6, "Its goal is to make the A llegro platform accessible to non-Polish speakers globally and ships with technological education institutions like learning to process text queries and use results contribute to the machine translation community. the Central House of Technology reflect Allegro’s ranking me", "Transitional"),
(7, "As we scale, our global data repository grows turning the buying process into a data enhanced science. While we have been able to build our capabilities in house, we will evaluate partnerships, alliances and acquisition opportunities that enable new go-to-market strategies to further our reach and c", "Symbolic"),
(8, "Big numbers At MLR Computer Vision, the primary objective is to elevate the user experience by leveraging machine learning image processing algorithms. The team specifically concentrated on image representation learning for Visual Search and the development of robust image classification • 2,243 wor", "Transitional"),
(9, "The fast delivery. The prediction mechanism allowed for GRI 1 used GRI 1: Foundation 2021 use of deep learning and generative AI techniques to an increase of +3pp in the number of orders with the build a semantic search engine is being developed. In promise of next-day delivery compared to previous ", "Substantive"),
(10, "We introduced the beta version of an AI-powered fashion assistant in 2023 in selected markets. The technology allows customers to ask questions using their own fashion terms and words, helping them navigate Zalando’s large assortment in a more intuitive way. We are excited to see how we can leverage", "Transitional"),
(11, "Upgrading our MarTech1 capability remained a priority in FY22 following on from large-scale migrations in FY21. As a result of these upgrades, we expect to see improved efficiency, driven by an ever-increasing adoption of machine learning and automation. Our bespoke data-driven attribution model lau", "Transitional"),
(12, "It is also interested in topics such as reranking and features interaction architectures and personalization. Big numbers At MLR Computer Vision, the primary objective is to elevate the user experience by leveraging machine learning image processing algorithms. The team specifically concentrated on ", "Transitional "),
(13, "tablished a complex verification process in vendor that it may be prudent to focus on other priorities in creation which scope depends on yearly turnover. In the short term immediately after an acquisition, such • Content created by artificial intelligence does • The use of various unverified AI sys", "Symbolic"),
(14, "Whilst continuing work is needed to Risks assigned Risk Averse risk appetites should be develop potential responses based on managed more conservatively, have stronger and possible scenarios, key focus areas in our more formal controls and increased focus for discussions included: assurance activiti", "Symbolic"),
(15, "damages against the Group and could damage the of their payment processing functions could lead to Group's reputation. Although the Group insures If the transaction is not finalized before the end user claims that purchases or payments were not Inventory risk may adversely affect the Group’s op- its", "Transitional"),
(16, "Towards the end of drove shares up to EUR 4.42 on January 11, FY 2023/2024, equity markets started to rally 2024. The start of FY 2023/2024 was characterized due to persistent disinflationary trends and by a challenging market environment for the anticipation of multiple rate cuts in 2024 In the fol", "symbolic"),
(17, "GROUP CONSOLIDATED MANAGEMENT REPORT for the year ended 31 December 2022 2.3. DIGITAL SERVICES ACT AND DIGITAL MARKETS ACT Regulatory Matters In 2022 the European Commission adopted two The Digital Market Act, introduces obligations and regulatory proposals relevant for digital services and limitati", "Symbolic"),
(18, "enhancing product descriptions and size recommendations. These in distribution, fulfilment, payment solutions, as well as marketing improvements are made possible through the utilization of big and media. Continuous focus on the Nordics data and machine learning.", "Transitional "),
(19, "Zalando continued to play a pioneering role in many areas in 2023, taking a leadership role to solve challenges that are relevant to the fashion and lifestyle industry. For example, we introduced tools that improve our customers’ size recommendations based on their unique body measurements, helping ", "Transitional "),
(20, "DTC Order Multi- Buyer & F C u o n m ct m ion e a rc li e ty Ma S n y a s g t e e m m ent M A an u a d g ie e n m ce en t Cu M rre u n lt c i- i es fu O lfi p l t m io e n n s t Cr S o h ss ip B p o in r g der T O el r e d p e h ri o n n g e S A K n U al - y le ti v c e s l Average UK standard click", "Not sure"),
(21, "This allowed THG Ingenuity total control of all data points, increasing the ability to squeeze efficiencies in real time via machine learning and AI. In 2023 we launched our Headless Commerce solution which gives clients even greater flexibility of deployment of relevant 5* Trustpilot reviews 2023 c", "Transitional "),
(22, "The program aims to promote knowledge and skills sharing, provide opportunities for senior employees to develop leadership skills, create spaces for role models from traditionally unrepresented groups, and enable productive relationships through a proactive learning culture. The program was conducte", "Transitional"),
(23, "To make the experience of shopping on our webstores more convenient, we Continuous focus on the Nordics constantly work on improving the personalisation options, such as search, With a clear ambition to significantly organically outgrow the Nordic online sorting, filtering options, as well as the pr", "Transitional "),
(24, "The team specifically concentrated on image representation learning for Visual Search and the development of robust image classification • 2,243 workers in Technology team in an environment where digital tools streamline Computer Vision models. Presently, its research is focused on the integration o", "Substantive "),
(25, "All Directors are required to complete our annual compliance training modules covering a range of subjects including anti-bribery and anti-corruption, anti-money laundering, data protection and anti-modern slavery. Additional training is available on request, where appropriate, so that Directors can", "Symbolic"),
(26, "We work with multiple producer responsibility organisations across our markets to ensure responsible collection, recycling and disposal of products and packaging. We are currently developing a scalable EPR solution, which is not yet applicable to ABOUT YOU. As a crucial first step towards achieving ", "Symbolic"),
(27, "The prediction mechanism allowed for GRI 1 used GRI 1: Foundation 2021 use of deep learning and generative AI techniques to an increase of +3pp in the number of orders with the build a semantic search engine is being developed. In promise of next-day delivery compared to previous commitment to digit", "Substantive"),
(28, "Additionally, difficulties with implementing new technology systems, delays in our timeline for planned improvements, significant system failures, or our inability to successfully modify our information systems to respond to changes in our business needs may cause disruptions in our business operati", "Symbolic"),
(29, "Key factors to give our customers the best service are convenient, we constantly work on improving the personalisation convenience, fast delivery, and easy return options. We aim for a options, such as search, sorting, filtering options, as well as the delivery time of 1-2 business days, offered to ", "Transitional"),
(30, "In The Group recognizes significant value in expanding order to deliver this part of the strategic framework Allegro's proposition beyond core product retail by Group is working on a number of initiatives such as: offering a wider range of services on the platform. Beyond unlocking new customer segm", "Symbolic"),
]

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Parse raw labels into clean tiers
# ══════════════════════════════════════════════════════════════════
print("=" * 100)
print("STEP 1: Parsing raw labels")
print("=" * 100)

parsed = {}
unparseable = []
for row_num, text_shown, raw in RAW:
    if row_num == 20:
        parsed[row_num] = {'label': '', 'notes': '', 'excluded': True,
                            'reason': 'Unclassifiable: fragmented PDF extraction (column interleaving)'}
        print(f"Row {row_num:2d}: raw='{raw}' -> EXCLUDED (pre-specified: unclassifiable)")
        continue

    stripped = raw.strip()
    lower = stripped.lower()
    matched_tier = None
    for key, canon in VALID_TIERS.items():
        if lower == key:
            matched_tier = canon
            notes = ''
            break
    else:
        # Not an exact match -- check for "Tier - commentary" pattern
        for key, canon in VALID_TIERS.items():
            if lower.startswith(key):
                matched_tier = canon
                notes = stripped[len(key):].strip(' -–—:').strip()
                break

    if matched_tier is None:
        unparseable.append((row_num, raw))
        print(f"Row {row_num:2d}: raw='{raw}' -> COULD NOT PARSE")
    else:
        parsed[row_num] = {'label': matched_tier, 'notes': notes if matched_tier else '',
                            'excluded': False, 'reason': ''}
        note_str = f", notes='{notes}'" if notes else ""
        print(f"Row {row_num:2d}: raw='{raw}' -> {matched_tier}{note_str}")

if unparseable:
    print("\n" + "!" * 100)
    print(f"STOPPING: {len(unparseable)} row(s) could not be parsed to a valid tier:")
    for row_num, raw in unparseable:
        print(f"  Row {row_num}: '{raw}'")
    print("Not proceeding. Fix these labels and re-run.")
    print("!" * 100)
    sys.exit(1)

print(f"\nAll {len(RAW)} rows parsed successfully "
      f"({sum(1 for v in parsed.values() if v['excluded'])} excluded, "
      f"{sum(1 for v in parsed.values() if not v['excluded'])} valid tier labels).")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — MANDATORY verification: row position 1-30 vs second_coder_sample.csv
# vs all_classifications.csv, comparing first 60 chars of passage text
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("STEP 2: Mandatory row-position verification")
print("=" * 100)

sample_df = pd.read_csv('second_coder_sample.csv')
main_df = pd.read_csv('all_classifications.csv')

if len(sample_df) != 30:
    print(f"STOPPING: second_coder_sample.csv has {len(sample_df)} rows, expected 30.")
    sys.exit(1)

sample_df = sample_df.reset_index(drop=True)
sample_df.insert(0, 'row_position', range(1, len(sample_df) + 1))

main_lookup = main_df.set_index('passage_id')['passage_text'].to_dict()

mismatches = []
print(f"{'Row':>4s}  {'passage_id':10s}  {'firm':18s}  {'year':5s}  "
      f"{'all_classifications.csv (60c)':45s}  {'coder was shown (60c)':45s}  match")
print("-" * 145)

for row_num, text_shown, raw in RAW:
    srow = sample_df[sample_df['row_position'] == row_num].iloc[0]
    pid = srow['passage_id']
    firm = srow['firm']
    year = srow['year']

    actual_text = main_lookup.get(pid, None)
    actual_60 = (actual_text[:60] if actual_text is not None else "MISSING passage_id")
    shown_60 = text_shown[:60]

    is_match = (actual_text is not None and actual_text[:60] == shown_60)
    if not is_match:
        mismatches.append(row_num)

    flag = "OK" if is_match else "*** MISMATCH ***"
    print(f"{row_num:4d}  {pid:10s}  {str(firm):18s}  {str(year):5s}  "
          f"{actual_60:45s}  {shown_60:45s}  {flag}")

if mismatches:
    print("\n" + "!" * 100)
    print(f"STOPPING: {len(mismatches)} row(s) failed verification: {mismatches}")
    print("Not proceeding to populate the CSV or compute kappa.")
    print("!" * 100)
    sys.exit(1)

print(f"\nAll 30 rows verified: passage text shown to the second coder matches "
      f"all_classifications.csv exactly for the claimed passage_id in every row.")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Populate second_coder_sample.csv
# ══════════════════════════════════════════════════════════════════
sample_df['second_coder_label'] = sample_df['row_position'].map(lambda r: parsed[r]['label'])
sample_df['coder_notes'] = sample_df['row_position'].map(lambda r: parsed[r]['notes'])
sample_df['excluded_flag'] = sample_df['row_position'].map(lambda r: parsed[r]['excluded'])
sample_df['exclusion_reason'] = sample_df['row_position'].map(lambda r: parsed[r]['reason'])

out_cols = ['passage_id', 'firm', 'year', 'original_classification',
            'second_coder_label', 'coder_notes', 'excluded_flag', 'exclusion_reason']
sample_df[out_cols].to_csv('second_coder_sample.csv', index=False)
print(f"\nsecond_coder_sample.csv updated -- {len(sample_df)} rows "
      f"({sample_df['excluded_flag'].sum()} excluded).")
