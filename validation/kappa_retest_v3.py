import anthropic
import os
import json
import pandas as pd
import re
import time
from sklearn.metrics import cohen_kappa_score

client = anthropic.Anthropic(api_key="REDACTED")
LLM_MODEL = "claude-sonnet-4-6"

CLASSIFICATION_PROMPT = """You are an academic research assistant classifying corporate AI language from annual reports. Your classifications will be used in a peer-reviewed dissertation on AI signalling in European e-commerce firms. Accuracy is essential.

There are THREE classification categories:
- Symbolic: vague, aspirational, governance-only, or non-operational AI language
- Transitional: specific AI application named and currently deployed, but no measurable outcomes
- Substantive: specific AI application named, currently deployed, AND measurable outcomes provided

═══════════════════════════════════════════════
STEP 1 — TEXT QUALITY CHECK
═══════════════════════════════════════════════
If the passage is completely incoherent with no discernible meaning, classify as Symbolic with justification "FRAGMENTED TEXT — no extractable content."

However: if the passage is partially fragmented but still contains clearly identifiable AI content — for example a named AI application, specific tool, or deployment description — proceed to Step 3 and classify based on what IS readable. Do not reject a passage as fragmented if meaningful AI content is still visible.

Only apply FRAGMENTED TEXT if the passage is so broken that you cannot identify any coherent AI-related claim.

═══════════════════════════════════════════════
STEP 2 — AUTOMATIC SYMBOLIC CATEGORIES
═══════════════════════════════════════════════
Classify as Symbolic immediately WITHOUT proceeding to Step 3 if the passage ONLY contains:

a) Board or employee training on AI as a governance topic — e.g. "Directors received training on artificial intelligence during the year"
b) Regulatory/legal framework references — e.g. EU AI Act, AI governance regulations, AI compliance policies
c) Risk disclosures about AI as an industry threat — e.g. "AI may disrupt our industry", "risks associated with AI adoption"
d) Macroeconomic commentary about AI in the broader economy or in other companies
e) Biography references — a board member who worked with AI at a previous employer
f) Subsidiary listings — a company name contains "AI" or "Artificial Intelligence" but no operational description
g) Pure aspirational statements with NO named application — e.g. "AI will play a crucial role in our future", "we aim to leverage AI for growth"
h) AI mentioned only in passing in a list alongside unrelated topics — e.g. "our priorities include AI, sustainability, and logistics"

IMPORTANT: Do NOT apply automatic Symbolic if the passage describes a specific named AI tool or system that is currently deployed, even if governance or strategy language also appears nearby. A passage that mentions a fraud detection system, recommendation engine, visual search tool, or similar specific application should proceed to Step 3.

═══════════════════════════════════════════════
STEP 3 — THREE DIAGNOSTIC QUESTIONS
═══════════════════════════════════════════════
Only reach this step if the passage passed Steps 1 and 2.

QUESTION 1: Does the passage name a SPECIFIC business function or application?
YES requires a concrete named AI tool, system, or application. Examples of YES:
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
- "machine learning and other technologies" without specifying the application

QUESTION 2: Does the passage reference a MEASURABLE outcome, metric, or timeline?
YES requires specific numbers, percentages, monetary figures, or timeframes DIRECTLY linked to the AI application:
- "reduced costs by 12%"
- "83.7 million EUR invested in AI development"
- "18-month project completing in 2025"
- "deployed across 10 European markets"
- "2,243 workers in Technology team" (when directly describing the AI team)

NO if:
- Outcomes described qualitatively without numbers ("improved efficiency", "enhanced experience")
- Numbers present but relate to non-AI activities (e.g. loan targets, revenue figures unrelated to AI)

QUESTION 3: Is the AI use CURRENT, MIXED, or ASPIRATIONAL?
CURRENT = past or present tense, deployed or actively running ("we deployed", "we launched", "the system processes", "we use", "we have implemented")
MIXED = combination of current deployment AND future plans in same passage
ASPIRATIONAL = future tense only, no deployment evidence ("we will", "we aim to", "we plan", "we are exploring")
q3_score = 1 if CURRENT or MIXED, 0 if ASPIRATIONAL

═══════════════════════════════════════════════
SCORING
═══════════════════════════════════════════════
classification_score = q1_score + q2_score + q3_score (0 to 3)
- Score 3 = Substantive
- Score 2 = Transitional
- Score 0 or 1 = Symbolic

═══════════════════════════════════════════════
CALIBRATION EXAMPLES
═══════════════════════════════════════════════
SYMBOLIC example: "Artificial Intelligence will play a crucial role in our journey ahead." — No named function, no metric, aspirational. Score 0.

SYMBOLIC example: "During FY24, the Board requested training on artificial intelligence, which was delivered." — Board training only, automatic Symbolic.

SYMBOLIC example: "machine learning and other forms of artificial intelligence are transforming aspects of the marketplace." — Industry commentary, no named application, automatic Symbolic.

TRANSITIONAL example: "enhancing product descriptions and size recommendations are made possible through the utilization of big data and machine learning." — Named functions (product descriptions, size recommendations), current tense, but no measurable metric. Score 2 = Transitional.

TRANSITIONAL example: "While the Group has implemented a fraud detection system based on machine learning." — Named function (fraud detection), current deployment confirmed, no metric. Score 2 = Transitional.

TRANSITIONAL example: "we launched a Global AI Strategy initiative that focuses on upskilling employees on AI fundamentals." — Named initiative, current deployment (launched), no measurable outcome. Score 2 = Transitional.

TRANSITIONAL example: "in generative AI, we are accelerating our AI strategy to enhance efficiency, reduce costs and introduce innovative features. This involves both custom in-house solutions and adoption of leading external technologies." — Named approach (generative AI, custom in-house solutions), current tense, stated goals but no quantified metrics. Score 2 = Transitional.

SUBSTANTIVE example: "The team specifically concentrated on image representation learning for Visual Search and development of robust Computer Vision models. 2,243 workers in Technology team." — Named functions (Visual Search, Computer Vision), measurable metric (2,243 workers), current. Score 3 = Substantive.

SUBSTANTIVE example: "Zalando utilised generative AI to scale the creation of product background images, thus enhancing content, reducing costs and increasing customer engagement." — Named function (generative AI for product images), named outcomes (reducing costs, increasing engagement), current. Score 3 = Substantive.

═══════════════════════════════════════════════
CRITICAL REMINDERS
═══════════════════════════════════════════════
- When in doubt between Symbolic and Transitional, ask: is there a NAMED specific AI application? If no, it is Symbolic.
- When in doubt between Transitional and Substantive, ask: is there a QUANTIFIED metric? If no, it is Transitional.
- Fragmented or incoherent text = always Symbolic.
- Do not over-promote. The default should be Symbolic unless the passage clearly passes the criteria.

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

HUMAN_LABELS = {
    'da158643': 'Symbolic', '569d0bec': 'Symbolic', 'c453af1d': 'Symbolic',
    'cfc9b248': 'Transitional', '6e401e5d': 'Symbolic', '907946ff': 'Symbolic',
    '9af11ea0': 'Symbolic', '87de711c': 'Symbolic', '00b389ab': 'Symbolic',
    'ee522d88': 'Symbolic', 'e55f892e': 'Symbolic', '6d8df2b6': 'Symbolic',
    'a0f066ce': 'Symbolic', 'bab12905': 'Symbolic', '5a80b553': 'Symbolic',
    '892b64ae': 'Symbolic', '9b4dddf4': 'Symbolic', '0fe31ad4': 'Symbolic',
    '73aec12e': 'Symbolic', '189b82e5': 'Symbolic', 'c668c1c1': 'Symbolic',
    'ffe9fd7f': 'Symbolic', 'a2aef007': 'Transitional', '315c8550': 'Symbolic',
    'e5492c06': 'Symbolic', 'a883e833': 'Symbolic', '2db7183b': 'Symbolic',
    'da5086e2': 'Symbolic', '513b8a27': 'Symbolic', 'c8e14327': 'Symbolic',
    '17e8d287': 'Transitional', '8f7bae1c': 'Symbolic', '7a37e5bd': 'Symbolic',
    '8da81300': 'Symbolic', '25b2c81f': 'Symbolic', '1b3276ab': 'Transitional',
    '8e2828f5': 'Symbolic', '34d8830d': 'Symbolic', 'c9560ab9': 'Symbolic',
    '38a2717e': 'Transitional', '104f00a9': 'Transitional', '868e929b': 'Symbolic',
    '88498c21': 'Substantive', '6033a4ad': 'Symbolic', '56e26e75': 'Symbolic',
    'f31af4f6': 'Transitional', '7432a7ae': 'Transitional', '910ae3c6': 'Symbolic',
    'f7501d1d': 'Symbolic', '4938eec4': 'Symbolic', '2b66d1f9': 'Symbolic',
    '72a97cbc': 'Symbolic',
}

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
        response_text = re.sub(r'^```json|^```|```$', '', response_text, flags=re.MULTILINE).strip()
        result = json.loads(response_text)
        result['classification_score'] = int(result.get('classification_score', 0))
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None

if __name__ == "__main__":
    df = pd.read_csv('data/kappa_sample.csv')
    print(f"Loaded {len(df)} passages for retest\n")

    new_labels = []
    human_labels = []
    results = []

    for _, row in df.iterrows():
        pid = row['passage_id']
        text = row['passage_text']
        human = HUMAN_LABELS.get(pid)

        if not human:
            print(f"  Skipping {pid} - no human label")
            continue

        print(f"Classifying {pid} ({row['firm']} {row['year']})...")
        result = classify_passage(text)

        if result:
            new_label = result['classification']
            new_labels.append(new_label)
            human_labels.append(human)
            match = "✅" if new_label == human else "❌"
            print(f"  {match} New: {new_label} | Human: {human} | {result.get('justification','')[:80]}")
            results.append({
                'passage_id': pid,
                'firm': row['firm'],
                'year': row['year'],
                'human_label': human,
                'old_llm_label': row['classification'],
                'new_llm_label': new_label,
                'justification': result.get('justification', ''),
                'agreement': new_label == human,
                'changed_from_original': row['classification'] != new_label
            })

        time.sleep(0.5)

    kappa = cohen_kappa_score(new_labels, human_labels)
    agree = sum(n == h for n, h in zip(new_labels, human_labels))

    print(f"\n{'='*50}")
    print(f"KAPPA RETEST v3 RESULTS")
    print(f"{'='*50}")
    print(f"Agreements: {agree}/{len(new_labels)} ({round(agree/len(new_labels)*100,1)}%)")
    print(f"Cohen Kappa: {round(kappa, 3)}")

    df_results = pd.DataFrame(results)
    disagreements = df_results[df_results['agreement'] == False]
    print(f"\nRemaining disagreements: {len(disagreements)}")
    if len(disagreements) > 0:
        print(disagreements[['passage_id','firm','old_llm_label',
                             'new_llm_label','human_label']].to_string(index=False))

    df_results.to_csv('results/kappa_retest_v3_results.csv', index=False)
    print("\nresults/kappa_retest_v3_results.csv saved")