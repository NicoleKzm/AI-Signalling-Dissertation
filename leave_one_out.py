"""
Leave-one-out reclassification for exemplar-contaminated passages.

IMPORTANT: before running this, make sure you have revoked the hardcoded API
key from classify.py and set a new one as an environment variable:
    export ANTHROPIC_API_KEY="your-new-key"
This script reads the key from the environment (client = anthropic.Anthropic()
with no api_key argument) rather than hardcoding it.

What this does
---------------
9 unique exemplar strings live inside CLASSIFICATION_PROMPT. One of them
("Directors received training on artificial intelligence during the year")
appears identically in 3 separate passages in your corpus (Moonpig), so it
maps to 3 passage_ids even though it's 1 exemplar line. The other 8 exemplars
each map to exactly 1 passage_id. Total: 9 exemplars -> 11 contaminated rows.

For each unique exemplar, this script builds a leave-one-out prompt (the base
prompt with that exemplar line removed) and reclassifies every passage_id
that shares that exemplar's text, then compares old vs. new label.
"""

import json
import re
import pandas as pd
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
LLM_MODEL = "claude-sonnet-4-6"

# --- imports CLASSIFICATION_PROMPT unchanged from classify.py ---
from classify import CLASSIFICATION_PROMPT  # noqa: E402

# Map: unique exemplar text -> list of passage_ids in the real corpus that
# share that exact text and are therefore contaminated by it.
EXEMPLAR_TO_PASSAGE_IDS = {
    "Artificial Intelligence will play a crucial role in our journey ahead": ["da5086e2"],
    "Directors received training on artificial intelligence during the year": [
        "00b389ab", "c8e14327", "8f7bae1c"
    ],
    "machine learning and AI are transforming the marketplace industry": ["c9560ab9"],
    "enhancing product descriptions and size recommendations are made possible through machine learning": ["17e8d287"],
    "the Group has implemented a fraud detection system based on machine learning": ["7432a7ae"],
    "we launched a Global AI Strategy initiative focusing on upskilling employees on AI fundamentals": ["f31af4f6"],
    "in generative AI, we are accelerating our AI strategy with custom in-house solutions": ["1b3276ab"],
    "Visual Search and Computer Vision models. 2,243 workers in Technology team": ["88498c21"],
    "Zalando utilised generative AI to scale product background images, reducing costs and increasing customer engagement": ["765b94eb"],
}


def build_leave_one_out_prompt(exemplar_snippet: str) -> str:
    """
    Remove the calibration line containing this exemplar snippet from the
    prompt. Matches on the distinctive quoted snippet rather than the full
    line, since exact whitespace/quote characters can be finicky to reproduce.
    """
    lines = CLASSIFICATION_PROMPT.split("\n")
    filtered = [line for line in lines if exemplar_snippet[:40] not in line]
    if len(filtered) == len(lines):
        raise ValueError(f"Exemplar snippet not found in prompt: {exemplar_snippet[:60]!r}")
    return "\n".join(filtered)


def classify_with_prompt(prompt_template: str, passage_text: str):
    try:
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=400,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt_template.format(passage=passage_text[:1500])
            }]
        )
        response_text = message.content[0].text.strip()
        response_text = re.sub(r'^```json|^```|```$', '', response_text, flags=re.MULTILINE).strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    df = pd.read_csv("all_classifications.csv")
    results = []

    for exemplar_text, passage_ids in EXEMPLAR_TO_PASSAGE_IDS.items():
        loo_prompt = build_leave_one_out_prompt(exemplar_text)

        for pid in passage_ids:
            row = df[df["passage_id"] == pid]
            if row.empty:
                print(f"WARNING: {pid} not found in all_classifications.csv — skipping")
                continue

            original_label = row.iloc[0]["classification"]
            passage_text = row.iloc[0]["passage_text"]

            new_result = classify_with_prompt(loo_prompt, passage_text)
            new_label = new_result["classification"] if new_result else "ERROR"

            results.append({
                "passage_id": pid,
                "exemplar_shared": exemplar_text[:50] + "...",
                "original_label": original_label,
                "leave_one_out_label": new_label,
                "changed": original_label != new_label,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("leave_one_out_results.csv", index=False)
    print(results_df.to_string(index=False))

    n_changed = results_df["changed"].sum()
    print(f"\n{n_changed} of {len(results_df)} passages changed classification.")

    if n_changed == 0:
        print("\nSuggested Chapter 3 sentence:")
        print(
            '"Eleven passages shared text with nine few-shot exemplars embedded '
            'in the classification prompt (one exemplar sentence recurred '
            'verbatim across three passages as standard governance boilerplate). '
            'All eleven were reclassified using a leave-one-out variant of the '
            'prompt with the relevant exemplar removed; all retained their '
            'original classification, confirming the contamination did not '
            'materially affect the results."'
        )
    else:
        print("\nRecompute signalling_scores.csv for affected firm-years:")
        print(results_df[results_df["changed"]].to_string(index=False))


if __name__ == "__main__":
    main()