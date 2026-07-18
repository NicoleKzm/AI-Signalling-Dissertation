import pandas as pd
from difflib import SequenceMatcher

df = pd.read_csv("all_classifications.csv")

# ── Paste in the FULL passage_text of every passage already used in
#    your calibration / retest rounds (kappa_sample.csv, kappa_retest_v3, etc.)
#    Only the first ~150 characters are needed for reliable matching.
CALIBRATION_TEXT_SNIPPETS = [
    "In addition to our Code of Ethics, we have implemented further Group-wide compliance policies specifying our standards and expectations regarding ethical trading",
    "Zalando\u2019s strategy will continue to position the company as an integral player in both the consumer and business markets",
    "The platform provides and we recognise that having an effective, brands, customers and partners on responsible",
    "It is also interested in topics such as reranking and features interaction architectures and personalization",
    "members of the supervisory board committees. Additionally, the supervisory board reviewed the status and anticipated next steps regarding the voluntary public takeover offer",
    "requirements. tablished a complex verification process in vendor that it may be prudent to focus on other priorities",
    "cease and desist the infringement, forced public apology etc. \u2022 Mall Group inability to succeed in transforming",
    "We work with multiple producer responsibility organisations across our markets to ensure responsible collection",
    "Additional training is available on request, where appropriate, so that Directors can update their skills and knowledge as applicable. During FY24",
    "A significant improvement in 2024 has been the installation proprietary approach ensures full control over performance",
    "We also continuously invest in and implement, significant modifications and upgrades to our information technology systems",
    "New technologies, including those based on artificial intelligence, can provide more immediate information technology",
    "A key structural factor influencing the macroeconomic environment is the increasing scale of investment related to artificial intelligence",
    "Rules regulating platforms\u2019 Average time it takes to receive a refund \u2013 SMART! services users 12h 12h 12h",
    "We will continue our application modernisation agenda, replacing other key systems such as our Contact Centre platform",
    "All Directors are required to complete our annual compliance training modules covering a range of subjects including anti-bribery",
    "ALLEGRO.EU S.A. GROUP CONSOLIDATED MANAGEMENT REPORT for the years ended 31 December 2021 WORKS ON THE NEW EU LEGAL",
    "One aspect of this is better prioritisation, ensuring we are allocating resource to projects that will generate a return",
    "General Ensuring accessibility on our e-commerce platform Disclosures 2-2 Entities included in the organization\u2019s",
    "We expanding our operations in the CEE region, and local markets but throughout Europe. by applying advanced technology",
    "GROUP CONSOLIDATED MANAGEMENT REPORT for the year ended 31 December 2025 2.4 Material impacts, risks and opportunities",
    "Output created by artificial intelligence may contain content that is inappropriate, unethical or violating the law",
    "In 2025 we spent MEUR 199.1, or 2.9% in percentage of revenue",
    "If these risks materialize, they could significantly hinder the successful execution of the Group\u2019s strategy",
    "Disruptions in the machine learning tools, any failure to avoid or limit or other factors cause the Group",
    "Both brands will play an important role in developing critical intra-company shipments is instrumental for a low environmental impact",
    "Although the Group insures If the transaction is not finalized before the end user claims that purchases",
    "Dedicated product teams ensure we maintain high agility, the process. enabling quick enhancements to our digital front-end",
    "Tesla, Inc. in Palo Alto, California, as Vice President Global Supply Chain",
    "training on artificial intelligence. No additional training needs were identified during the Board\u2019s annual evaluation",
    "enhancing product descriptions and size recommendations. These in distribution, fulfilment, payment solutions",
    "ALLEGRO.EU S.A. GROUP CONSOLIDATED MANAGEMENT REPORT for the year ended 31 December 2023 WORKS ON THE NEW EU LEGAL CONSUMER CREDIT",
    "Priorities for FY25 This year will see us begin the second phase of our Enterprise Resource Planning transformation",
    "types of goods that merchants offer on its mar- the foregoing were to occur, the Group's business",
    "in generative AI, we are accelerating our AI strategy to enhance efficiency, reduce costs and introduce innovative features",
    "Building on a successful partnership, Zalando and OpenAI will continue collaborating to develop even more generative AI solutions",
    "It concerns two areas: Allegro also identified a significant risk related to marketplace (3P) and e-commerce sales",
    "In addition, machine learning and other forms of ketplace. The popularity of certain categories of results",
    "ALLEGRO.EU S.A. GROUP CONSOLIDATED MANAGEMENT REPORT for the year ended 31 December 2023 Allegro also has an image search mechanism",
    "This time, THG Ingenuity built delivery timeframe 2023 the software capability to drive all automated instruction",
    "term, the Group aims to grow Allegro loans written years. The Ceneo management team has responded",
    "The team specifically concentrated on image representation learning for Visual Search and the development of robust image classification",
    "2 ASOS.com additionally has a branch registered in the Netherlands. 3 ASOS Projects Limited has a 2.9% interest in Action Artificial Intelligence",
    "External stakeholders were given the chance to extend the shortlist in the next Planet Earth Extensive desk research",
    "GenAI Literacy In 2025, we launched a Global AI Strategy initiative that focuses on upskilling our salaried employees",
    "While the Group has changes in user demand and user spending pat- Group cannot guarantee that all material events",
    "2.2 Report on economic position 2.2.1 Macroeconomic and sector-specific environment In 2025 the global economy proved more resilient",
    "Examples of emerging risks that we continue to monitor include: \u2022 The pace of technological change with regards to Artificial Intelligence",
    "We are excited to see how we can leverage generative AI in new ways to make our customer experience even more engaging",
    "Any disruption in the properly authorised or were transmitted in error, of these operations by optimising the 1P selection",
    "These The data comes from the internal register of court cases and concerns reports in 2025",
]

PROMPT_EXAMPLE_SNIPPETS = [
    "will play a crucial role in our journey ahead",
    "training on artificial intelligence during the year",
    "are transforming the marketplace industry",
    "product descriptions and size recommendations",
    "implemented a fraud detection system based on machine learning",
    "global ai strategy initiative",
    "accelerating our ai strategy with custom in-house solutions",
    "computer vision models",
    "utilised generative ai to scale product background images",
]

def fuzzy_overlap(text, snippet, threshold=0.55):
    text_snip = str(text)[:150].lower()
    snip = snippet[:150].lower()
    ratio = SequenceMatcher(None, text_snip, snip).ratio()
    return ratio >= threshold

def is_contaminated(text):
    for s in CALIBRATION_TEXT_SNIPPETS + PROMPT_EXAMPLE_SNIPPETS:
        if fuzzy_overlap(text, s):
            return True
    return False

df["contaminated"] = df["passage_text"].apply(is_contaminated)

# Also drop near-duplicate passages WITHIN the remaining pool itself
# (same 60% char-overlap logic used in your own dedup step)
def dedupe_pool(rows, threshold=0.6):
    kept = []
    kept_sigs = []
    for _, row in rows.iterrows():
        sig = str(row["passage_text"])[:250].lower()
        is_dup = False
        for ksig in kept_sigs:
            shared = sum(1 for a, b in zip(sig, ksig) if a == b)
            overlap = shared / max(len(sig), len(ksig), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(row)
            kept_sigs.append(sig)
    return pd.DataFrame(kept)

pool = df[~df["contaminated"]].copy()
pool = dedupe_pool(pool)

print(f"Total passages: {len(df)}")
print(f"Excluded (contaminated - calibration/prompt overlap): {df['contaminated'].sum()}")
print(f"Clean pool size after internal dedup: {len(pool)}")

SAMPLE_SIZE = 52
fresh_sample = pool.sample(n=min(SAMPLE_SIZE, len(pool)), random_state=7)

cols_to_keep = ["passage_id","firm","year","page_number","passage_text","classification"]
fresh_sample_out = fresh_sample[cols_to_keep].copy()
fresh_sample_out = fresh_sample_out.rename(columns={"classification": "llm_label"})
fresh_sample_out["human_label"] = ""

fresh_sample_out.to_csv("fresh_kappa_sample_v2.csv", index=False)
print(f"\nFresh sample saved to fresh_kappa_sample_v2.csv with {len(fresh_sample_out)} passages")

