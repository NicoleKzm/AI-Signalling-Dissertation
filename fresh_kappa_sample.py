import pandas as pd

# ── Load full passage dataset ──────────────────────────────────────
df = pd.read_csv("all_classifications.csv")

# ── IDs already used in original calibration/retest rounds ─────────
CALIBRATION_IDS = [
    "da158643","569d0bec","c453af1d","cfc9b248","6e401e5d","907946ff",
    "9af11ea0","87de711c","00b389ab","ee522d88","e55f892e","6d8df2b6",
    "a0f066ce","bab12905","5a80b553","892b64ae","9b4dddf4","0fe31ad4",
    "73aec12e","189b82e5","c668c1c1","ffe9fd7f","a2aef007","315c8550",
    "e5492c06","a883e833","2db7183b","da5086e2","513b8a27","c8e14327",
    "17e8d287","8f7bae1c","7a37e5bd","8da81300","25b2c81f","1b3276ab",
    "8e2828f5","34d8830d","c9560ab9","38a2717e","104f00a9","868e929b",
    "88498c21","6033a4ad","56e26e75","f31af4f6","7432a7ae","910ae3c6",
    "f7501d1d","4938eec4","2b66d1f9","72a97cbc"
]

# ── Snippets copied verbatim into the prompt's few-shot examples ───
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

def contains_example_text(text, snippets):
    text_l = str(text).lower()
    return any(s.lower() in text_l for s in snippets)

cal_mask = df["passage_id"].isin(CALIBRATION_IDS)
example_mask = df["passage_text"].apply(lambda t: contains_example_text(t, PROMPT_EXAMPLE_SNIPPETS))

pool = df[~cal_mask & ~example_mask].copy()

print(f"Total passages: {len(df)}")
print(f"Excluded (calibration IDs): {cal_mask.sum()}")
print(f"Excluded (prompt example overlap): {example_mask.sum()}")
print(f"Overlap between the two exclusion sets: {(cal_mask & example_mask).sum()}")
print(f"Clean pool size: {len(pool)}")

SAMPLE_SIZE = 52
fresh_sample = pool.sample(n=min(SAMPLE_SIZE, len(pool)), random_state=42)

cols_to_keep = ["passage_id","firm","year","page_number","passage_text","classification"]
fresh_sample_out = fresh_sample[cols_to_keep].copy()
fresh_sample_out = fresh_sample_out.rename(columns={"classification": "llm_label"})
fresh_sample_out["human_label"] = ""

fresh_sample_out.to_csv("fresh_kappa_sample.csv", index=False)
print(f"\nFresh sample saved to fresh_kappa_sample.csv with {len(fresh_sample_out)} passages")

