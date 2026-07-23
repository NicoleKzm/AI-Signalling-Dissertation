import anthropic
import os
import json
import pandas as pd
import re
import time
from sklearn.metrics import cohen_kappa_score

client = anthropic.Anthropic(api_key="REDACTED")
LLM_MODEL = "claude-sonnet-4-6"

# ── Updated prompt ─────────────────────────────────────────────────
CLASSIFICATION_PROMPT = """You are an academic research assistant classifying corporate AI language from annual reports. Your classifications will be used in a peer-reviewed dissertation. Accuracy and consistency are essential.

STEP 1 — TEXT QUALITY CHECK (do this first, before anything else)
Read the passage carefully. If the passage is fragmented, incoherent, or clearly a PDF extraction artefact where sentences are cut off, mixed up, or unreadable, you MUST classify it as Symbolic with classification_score 0 and write "FRAGMENTED TEXT" in the justification. Do not attempt to infer meaning from broken text.

Examples of fragmented text to classify as Symbolic immediately:
- Sentences that end mid-word or mid-phrase
- Text that mixes multiple unrelated sentences together
- Passages where the grammar is broken and meaning is unclear
- Table of contents entries or header text without substance

STEP 2 — AI RELEVANCE CHECK (do this second)
Ask yourself: does this passage describe something THIS COMPANY is actually doing or planning to do with AI? If the AI mention falls into ANY of the following categories, classify immediately as Symbolic with classification_score 0:

AUTOMATIC SYMBOLIC categories — classify immediately, do not proceed to diagnostic questions:
a) Board or employee training on AI as a topic (e.g. "Directors received training on artificial intelligence")
b) Regulatory or legal framework references (e.g. EU AI Act, AI governance regulations, compliance with AI laws)
c) Risk disclosures about AI as a general industry trend or threat (e.g. "AI may disrupt our industry")
d) Macroeconomic commentary about AI investment globally or in other companies
e) Biography of a board member who worked with AI at a previous employer
f) Corporate subsidiary listings where a company name contains the word "AI" or "Artificial Intelligence"
g) Generic strategic statements that AI "will play a role" or "is important to our future" with no specific application named
h) AI mentioned only in passing alongside many other unrelated topics
i) AI risk management or ethics policies without any operational deployment described
j) Aspirational statements about what AI "could" or "might" achieve with no evidence of current deployment

STEP 3 — DIAGNOSTIC QUESTIONS (only reach this step if the passage passed Steps 1 and 2)
For passages that describe genuine operational AI activity by this company, answer these three questions:

QUESTION 1: Does the passage name a SPECIFIC business function or application?
- YES requires: a concrete named AI tool, system, or application (e.g. "fraud detection system", "visual search using computer vision", "recommendation engine", "demand forecasting model", "generative AI for product image creation")
- NO if: AI is mentioned generically ("we use AI", "AI capabilities", "AI solutions", "AI strategy") without naming what it specifically does
- NO if: the function is described so vaguely it could apply to any company

QUESTION 2: Does the passage reference a MEASURABLE outcome, metric, or timeline?
- YES requires: specific numbers, percentages, monetary figures, timeframes, or quantified results directly linked to the AI application
- NO if: outcomes are described qualitatively without numbers
- NO if: metrics present but relate to something other than the AI application itself

QUESTION 3: Is the AI use described as CURRENT, MIXED, or ASPIRATIONAL?
- CURRENT: past or present tense describing deployed or actively running AI
- MIXED: combination of current deployment AND future development plans
- ASPIRATIONAL: future tense only, no evidence of deployment
- q3_score = 1 if CURRENT or MIXED, 0 if ASPIRATIONAL

SCORING:
- q1_score + q2_score + q3_score = classification_score (0 to 3)
- 3 = Substantive
- 2 = Transitional
- 0 or 1 = Symbolic

CRITICAL REMINDERS:
- When in doubt, classify lower not higher
- AI governance, ethics, or regulation = ALWAYS Symbolic
- Fragmented or incoherent text = ALWAYS Symbolic
- Board training on AI = ALWAYS Symbolic
- "AI strategy" or "AI roadmap" alone = Symbolic
- Only answer YES to Q1 if you could describe exactly what the AI system does in one specific sentence

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
  "justification": "one sentence explanation referencing which specific rule was applied"
}}"""

# ── Your human labels from the Kappa session ───────────────────────
HUMAN_LABELS = {
    'da158643': 'Symbolic',
    '569d0bec': 'Symbolic',
    'c453af1d': 'Symbolic',
    'cfc9b248': 'Transitional',
    '6e401e5d': 'Symbolic',
    '907946ff': 'Symbolic',
    '9af11ea0': 'Symbolic',
    '87de711c': 'Symbolic',
    '00b389ab': 'Symbolic',
    'ee522d88': 'Symbolic',
    'e55f892e': 'Symbolic',
    '6d8df2b6': 'Symbolic',
    'a0f066ce': 'Symbolic',
    'bab12905': 'Symbolic',
    '5a80b553': 'Symbolic',
    '892b64ae': 'Symbolic',
    '9b4dddf4': 'Symbolic',
    '0fe31ad4': 'Symbolic',
    '73aec12e': 'Symbolic',
    '189b82e5': 'Symbolic',
    'c668c1c1': 'Symbolic',
    'ffe9fd7f': 'Symbolic',
    'a2aef007': 'Transitional',
    '315c8550': 'Symbolic',
    'e5492c06': 'Symbolic',
    'a883e833': 'Symbolic',
    '2db7183b': 'Symbolic',
    'da5086e2': 'Symbolic',
    '513b8a27': 'Symbolic',
    'c8e14327': 'Symbolic',
    '17e8d287': 'Transitional',
    '8f7bae1c': 'Symbolic',
    '7a37e5bd': 'Symbolic',
    '8da81300': 'Symbolic',
    '25b2c81f': 'Symbolic',
    '1b3276ab': 'Transitional',
    '8e2828f5': 'Symbolic',
    '34d8830d': 'Symbolic',
    'c9560ab9': 'Symbolic',
    '38a2717e': 'Transitional',
    '104f00a9': 'Transitional',
    '868e929b': 'Symbolic',
    '88498c21': 'Substantive',
    '6033a4ad': 'Symbolic',
    '56e26e75': 'Symbolic',
    'f31af4f6': 'Transitional',
    '7432a7ae': 'Transitional',
    '910ae3c6': 'Symbolic',
    'f7501d1d': 'Symbolic',
    '4938eec4': 'Symbolic',
    '2b66d1f9': 'Symbolic',
    '72a97cbc': 'Symbolic',
}

def classify_passage(passage_text):
    try:
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
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
    # Load kappa sample
    df = pd.read_csv('data/kappa_sample.csv')
    print(f"Loaded {len(df)} passages for retest")

    new_labels = []
    human_labels = []
    results = []

    for _, row in df.iterrows():
        pid = row['passage_id']
        text = row['passage_text']
        human = HUMAN_LABELS.get(pid)

        if not human:
            print(f"  Skipping {pid} - no human label found")
            continue

        print(f"Classifying {pid} ({row['firm']} {row['year']})...")
        result = classify_passage(text)

        if result:
            new_label = result['classification']
            new_labels.append(new_label)
            human_labels.append(human)
            results.append({
                'passage_id': pid,
                'firm': row['firm'],
                'year': row['year'],
                'human_label': human,
                'old_llm_label': row['classification'],
                'new_llm_label': new_label,
                'justification': result.get('justification', ''),
                'changed': row['classification'] != new_label
            })
            print(f"  Old: {row['classification']} | New: {new_label} | Human: {human}")

        time.sleep(0.5)

    # Calculate Kappa
    kappa = cohen_kappa_score(new_labels, human_labels)
    agree = sum(n == h for n, h in zip(new_labels, human_labels))

    print(f"\n{'='*50}")
    print(f"KAPPA RETEST RESULTS")
    print(f"{'='*50}")
    print(f"Agreements: {agree}/{len(new_labels)} ({round(agree/len(new_labels)*100,1)}%)")
    print(f"Cohen Kappa: {round(kappa, 3)}")

    # Show changes
    df_results = pd.DataFrame(results)
    changed = df_results[df_results['changed'] == True]
    print(f"\nPassages where classification changed: {len(changed)}")
    print(changed[['passage_id', 'firm', 'old_llm_label',
                    'new_llm_label', 'human_label']].to_string(index=False))

    df_results.to_csv('results/kappa_retest_results.csv', index=False)
    print("\nresults/kappa_retest_results.csv saved")