import anthropic
import os
import json
import pandas as pd
import pdfplumber
import re
import uuid
import time
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

ANNUAL_REPORTS_DIR = Path("/Users/user/Desktop/Annual Reports ")
SCORES_FILE = "data/signalling_scores.csv"
CLASSIFICATIONS_FILE = "data/all_classifications.csv"
LLM_MODEL = "claude-sonnet-4-6"

# ── Two-Tier Keyword System ────────────────────────────────────────
TIER_1_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning",
    "large language model", "natural language processing", "computer vision",
    "neural network", "generative ai", "llm", "gpt", "nlp",
    "foundation model", "responsible ai", "ethical ai", "ai governance",
    "ai strategy", "ai roadmap", "ai investment", "ai adoption",
    "ai integration", "ai implementation", "ai capabilities",
    "ai assistant", "conversational ai",
]

TIER_2_KEYWORDS = [
    "ai-powered", "ai-driven", "ai-enabled", "ai-based",
    "ai solutions", "ai platform", "ai tools",
    "recommendation engine", "recommendation algorithm", "recommendation system",
    "predictive analytics", "predictive model", "demand forecasting",
    "dynamic pricing", "price optimisation", "price optimization",
    "personalisation", "personalization", "personalised", "personalized",
    "chatbot", "virtual assistant", "image recognition", "visual search",
    "voice search", "fraud detection", "fraud prevention",
    "supply chain optimisation", "supply chain optimization",
    "warehouse automation", "customer segmentation", "churn prediction",
    "search optimisation", "search optimization", "inventory management",
    "demand prediction", "transformer", "copilot",
]

# ── Classification Prompt (v4 calibrated - kappa 0.856) ────────────
CLASSIFICATION_PROMPT = """You are an academic research assistant classifying corporate AI language from annual reports. Your classifications will be used in a peer-reviewed dissertation on AI signalling in European e-commerce firms. Accuracy is essential.

There are THREE classification categories:
- Symbolic: vague, aspirational, governance-only, or non-operational AI language
- Transitional: specific AI application named and currently deployed, but no measurable outcomes
- Substantive: specific AI application named, currently deployed, AND measurable outcomes provided

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
STEP 1 \u2014 TEXT QUALITY CHECK
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
If the passage is completely incoherent with no discernible meaning, classify as Symbolic with justification "FRAGMENTED TEXT \u2014 no extractable content."

However: if the passage is partially fragmented but still contains clearly identifiable AI content \u2014 for example a named AI application, specific tool, or deployment description \u2014 proceed to Step 3 and classify based on what IS readable. Do not reject a passage as fragmented if meaningful AI content is still visible.

Only apply FRAGMENTED TEXT if the passage is so broken that you cannot identify any coherent AI-related claim.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
STEP 2 \u2014 AUTOMATIC SYMBOLIC CATEGORIES
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Classify as Symbolic immediately WITHOUT proceeding to Step 3 if the passage ONLY contains:

a) Board or employee training on AI as a governance topic
b) Regulatory or legal framework references \u2014 EU AI Act, AI governance regulations, AI compliance policies
c) Risk disclosures about AI as an industry threat \u2014 "AI may disrupt our industry"
d) Macroeconomic commentary about AI in the broader economy or in other companies
e) Biography references \u2014 a board member who worked with AI at a previous employer
f) Subsidiary listings \u2014 a company name contains "AI" but no operational description
g) Pure aspirational statements with NO named application \u2014 "AI will play a crucial role in our future"
h) AI mentioned only in passing in a list alongside unrelated topics

IMPORTANT: Do NOT apply automatic Symbolic if the passage describes a specific named AI tool or system that is currently deployed, even if governance or strategy language also appears nearby.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
STEP 3 \u2014 THREE DIAGNOSTIC QUESTIONS
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
QUESTION 1: Does the passage name a SPECIFIC business function or application?
YES requires a concrete named AI tool, system, or application:
- "fraud detection system based on machine learning"
- "image search mechanism using computer vision"
- "recommendation engine for personalised product suggestions"
- "demand forecasting model"
- "generative AI for product background image creation"
- "AI-powered size recommendation feature"
- "automated sorting of garments using machine learning"

NO if AI is mentioned only generically:
- "we use AI to improve customer experience"
- "AI capabilities", "AI solutions", "AI strategy" without naming what it does

QUESTION 2: Does the passage reference a MEASURABLE outcome, metric, or timeline?
YES requires specific numbers, percentages, monetary figures, or timeframes DIRECTLY linked to the AI application.
NO if outcomes described qualitatively without numbers, or numbers relate to non-AI activities.

QUESTION 3: Is the AI use CURRENT, MIXED, or ASPIRATIONAL?
CURRENT = past or present tense, deployed or actively running
MIXED = combination of current deployment AND future plans
ASPIRATIONAL = future tense only, no deployment evidence
q3_score = 1 if CURRENT or MIXED, 0 if ASPIRATIONAL

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SCORING
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
classification_score = q1_score + q2_score + q3_score (0 to 3)
- Score 3 = Substantive
- Score 2 = Transitional
- Score 0 or 1 = Symbolic

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
CALIBRATION EXAMPLES
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SYMBOLIC: "Artificial Intelligence will play a crucial role in our journey ahead." \u2014 No named function, aspirational. Score 0.
SYMBOLIC: "Directors received training on artificial intelligence during the year." \u2014 Board training, automatic Symbolic.
SYMBOLIC: "machine learning and AI are transforming the marketplace industry." \u2014 Industry commentary, no named application.

TRANSITIONAL: "enhancing product descriptions and size recommendations are made possible through machine learning." \u2014 Named functions, current, no metric. Score 2.
TRANSITIONAL: "the Group has implemented a fraud detection system based on machine learning." \u2014 Named function, current, no metric. Score 2.
TRANSITIONAL: "we launched a Global AI Strategy initiative focusing on upskilling employees on AI fundamentals." \u2014 Named initiative, current, no metric. Score 2.
TRANSITIONAL: "in generative AI, we are accelerating our AI strategy with custom in-house solutions." \u2014 Named approach, current, no quantified metric. Score 2.

SUBSTANTIVE: "Visual Search and Computer Vision models. 2,243 workers in Technology team." \u2014 Named functions, metric, current. Score 3.
SUBSTANTIVE: "Zalando utilised generative AI to scale product background images, reducing costs and increasing customer engagement." \u2014 Named function, named outcomes, current. Score 3.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
CRITICAL REMINDERS
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
- When in doubt between Symbolic and Transitional: is there a NAMED specific AI application? If no, Symbolic.
- When in doubt between Transitional and Substantive: is there a QUANTIFIED metric? If no, Transitional.
- Completely incoherent text = Symbolic.
- Do not over-promote. Default to Symbolic unless criteria are clearly met.

Passage to classify:
{passage}

Respond in JSON format only, no preamble:
{{
  "q1_named_function": "YES" or "NO",
  "q1_score": 1 or 0,
  "q2_measurable_outcome": "YES" or "NO",
  "q2_score": 1 or 0,
  "q3_tense": "CURRENT" or "MIXED" or "ASPIRATIONAL",
  "q3_score": 1 or 0,
  "classification_score": 0 to 3,
  "classification": "Symbolic" or "Transitional" or "Substantive",
  "justification": "one sentence explanation referencing which specific criteria was applied"
}}"""


# ── Tier Logic ─────────────────────────────────────────────────────
def is_ai_relevant(passage_text):
    text_lower = passage_text.lower()
    has_tier1 = any(
        re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        for kw in TIER_1_KEYWORDS
    )
    has_tier2 = any(
        re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        for kw in TIER_2_KEYWORDS
    )
    if has_tier1 and has_tier2:
        return True, "tier_1_and_2"
    elif has_tier1:
        return True, "tier1"
    else:
        return False, None


# ── Deduplication ──────────────────────────────────────────────────
def deduplicate_passages(passages):
    kept = []
    for candidate in passages:
        candidate_sig = candidate['text'][:250].lower()
        is_duplicate = False
        for existing in kept:
            if existing['page_number'] != candidate['page_number']:
                continue
            existing_sig = existing['text'][:250].lower()
            shared = sum(1 for a, b in zip(candidate_sig, existing_sig) if a == b)
            overlap_ratio = shared / max(len(candidate_sig), len(existing_sig), 1)
            if overlap_ratio > 0.60:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


# ── PDF Extraction ─────────────────────────────────────────────────
def extract_ai_passages(pdf_path):
    raw_passages = []
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
                    relevant, tier = is_ai_relevant(chunk)
                    if relevant:
                        raw_passages.append({
                            'text': chunk,
                            'tier': tier,
                            'page_number': page_num
                        })
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
        return []

    deduped = deduplicate_passages(raw_passages)
    removed = len(raw_passages) - len(deduped)
    if removed > 0:
        print(f"  Deduplication removed {removed} overlapping passages")
    return deduped


# ── API Classification ─────────────────────────────────────────────
def classify_passage(passage_text):
    try:
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=400,
            temperature=0,
            messages=[{
                "role": "user",
                "content": CLASSIFICATION_PROMPT.format(passage=passage_text[:1500])
            }]
        )
        response_text = message.content[0].text.strip()
        response_text = re.sub(
            r'^```json|^```|```$', '', response_text, flags=re.MULTILINE
        ).strip()
        result = json.loads(response_text)
        result['q1_score'] = int(result.get('q1_score', 0))
        result['q2_score'] = int(result.get('q2_score', 0))
        result['q3_score'] = int(result.get('q3_score', 0))
        result['classification_score'] = int(result.get('classification_score', 0))
        return result
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


# ── Tie-Safe Modal Classification ──────────────────────────────────
def modal_class(classifications):
    if not classifications:
        return "No_AI_Disclosure"
    counts = Counter(c['classification'] for c in classifications)
    top_n = max(counts.values())
    top_classes = [k for k, v in counts.items() if v == top_n]
    if len(top_classes) > 1:
        mean_score = sum(
            c['classification_score'] for c in classifications
        ) / len(classifications)
        if mean_score >= 2.0:
            return "Substantive"
        elif mean_score >= 1.0:
            return "Transitional"
        else:
            return "Symbolic"
    return top_classes[0]


# ── Firm-Year Aggregation ──────────────────────────────────────────
def get_firm_year_score(classifications, firm, year, pdf_path):
    total = len(classifications)
    substantive = sum(1 for c in classifications if c['classification'] == 'Substantive')
    transitional = sum(1 for c in classifications if c['classification'] == 'Transitional')
    symbolic = sum(1 for c in classifications if c['classification'] == 'Symbolic')

    prop_substantive = round(substantive / total * 100, 2) if total > 0 else 0
    prop_transitional = round(transitional / total * 100, 2) if total > 0 else 0
    prop_symbolic = round(symbolic / total * 100, 2) if total > 0 else 0

    score_map = {'Symbolic': 0, 'Transitional': 1, 'Substantive': 2}
    mean_signal = round(
        sum(score_map[c['classification']] for c in classifications) / total, 3
    ) if total > 0 else 0

    return {
        'firm': firm,
        'year': year,
        'source_document': Path(pdf_path).name,
        'llm_model': LLM_MODEL,
        'total_passages': total,
        'substantive_count': substantive,
        'transitional_count': transitional,
        'symbolic_count': symbolic,
        'prop_substantive_%': prop_substantive,
        'prop_transitional_%': prop_transitional,
        'prop_symbolic_%': prop_symbolic,
        'mean_signal_score': mean_signal,
        'modal_classification': modal_class(classifications),
    }


# ── Parse Firm and Year ────────────────────────────────────────────
FIRM_NAME_MAP = {
    "about you": "About You",
    "asos": "ASOS",
    "ao world": "AO World",
    "allegro": "Allegro",
    "boohoo": "Boohoo",
    "boozt": "Boozt",
    "doc morris": "DocMorris",
    "docmorris": "DocMorris",
    "hello fresh": "HelloFresh",
    "hellofresh": "HelloFresh",
    "moonpig": "Moonpig",
    "mytheresa": "Mytheresa",
    "redcare pharmacy": "Redcare Pharmacy",
    "thg": "THG",
    "westwing": "Westwing",
    "zalando": "Zalando",
}

def parse_firm_year(pdf_path):
    pdf_path = Path(pdf_path)
    parent_folder = pdf_path.parent.name.strip()
    stem = pdf_path.stem

    year_match = re.search(r'(202[0-6])', stem)
    if not year_match:
        return None, None
    year = int(year_match.group(1))

    root_name = ANNUAL_REPORTS_DIR.name.strip()
    if parent_folder != root_name and parent_folder != '.':
        raw_firm = parent_folder.strip()
    else:
        parts = stem.split('_')
        raw_firm = ' '.join(parts[:-1]) if len(parts) >= 2 else stem.replace(str(year), '').strip('_- ')

    firm = FIRM_NAME_MAP.get(raw_firm.lower(), raw_firm.strip())
    return firm, year


# ── Process One Report ─────────────────────────────────────────────
def process_report(pdf_path, firm, year):
    print(f"\nProcessing {firm} {year}...")

    passages = extract_ai_passages(pdf_path)
    tier1_count = sum(1 for p in passages if p['tier'] == 'tier1')
    tier2_count = sum(1 for p in passages if p['tier'] == 'tier_1_and_2')
    print(f"  Found {len(passages)} passages after dedup (Tier1: {tier1_count}, Tier1+2: {tier2_count})")

    if not passages:
        print(f"  No AI passages found — recording as No_AI_Disclosure")
        return get_firm_year_score([], firm, year, pdf_path), []

    classifications = []
    for i, passage in enumerate(passages):
        print(f"  Classifying passage {i+1}/{len(passages)}...")
        result = classify_passage(passage['text'])
        if result:
            result['passage_id'] = str(uuid.uuid4())[:8]
            result['firm'] = firm
            result['year'] = year
            result['page_number'] = passage['page_number']
            result['tier'] = passage['tier']
            result['source_document'] = Path(pdf_path).name
            result['source_type'] = 'annual_report'
            result['llm_model'] = LLM_MODEL
            result['passage_text'] = passage['text'][:300]
            result['validation_sample'] = 'no'
            result['human_label'] = ''
            result['agreement'] = ''
            classifications.append(result)
        time.sleep(0.5)

    score = get_firm_year_score(classifications, firm, year, pdf_path)
    return score, classifications


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_files = list(Path(ANNUAL_REPORTS_DIR).glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {ANNUAL_REPORTS_DIR}")

    if not pdf_files:
        print("No PDF files found. Check your ANNUAL_REPORTS_DIR path.")
        exit()

    results = []
    all_classifications = []

    for pdf_path in sorted(pdf_files):
        firm, year = parse_firm_year(pdf_path)

        if not firm or not year:
            print(f"Skipping {pdf_path.name} - cannot parse firm/year")
            continue

        if year < 2021 or year > 2025:
            print(f"Skipping {pdf_path.name} - year outside 2021-2025")
            continue

        if firm.strip().lower() == "about you" and year == 2025:
            print(f"Skipping About You 2025 - partial panel")
            continue

        try:
            score, classifications = process_report(pdf_path, firm, year)
            results.append(score)
            all_classifications.extend(classifications)

            pd.DataFrame(results).to_csv(SCORES_FILE, index=False)
            if all_classifications:
                pd.DataFrame(all_classifications).to_csv(CLASSIFICATIONS_FILE, index=False)

        except Exception as e:
            print(f"  FATAL ERROR on {firm} {year}: {e}")
            print(f"  Progress saved. {len(results)} reports completed.")
            continue

    if results:
        df_scores = pd.DataFrame(results)
        df_scores = df_scores.sort_values(['firm', 'year']).reset_index(drop=True)
        df_scores.to_csv(SCORES_FILE, index=False)
        print(f"\nDone — {SCORES_FILE} saved with {len(results)} firm-year scores")
        print(df_scores[[
            'firm', 'year', 'total_passages',
            'mean_signal_score', 'modal_classification'
        ]].to_string())

    if all_classifications:
        df_class = pd.DataFrame(all_classifications)
        df_class.to_csv(CLASSIFICATIONS_FILE, index=False)
        print(f"{CLASSIFICATIONS_FILE} saved with {len(all_classifications)} passage classifications")
    else:
        print("No passages classified.")