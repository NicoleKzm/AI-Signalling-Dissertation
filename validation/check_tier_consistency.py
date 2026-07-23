import pandas as pd

df = pd.read_csv("data/all_classifications.csv")

# Check 1: does classification_score always equal the sum of the three sub-scores?
df["computed_score"] = df["q1_score"] + df["q2_score"] + df["q3_score"]
score_mismatches = df[df["classification_score"] != df["computed_score"]]

print(f"Total rows: {len(df)}")
print(f"Score arithmetic mismatches (classification_score != q1+q2+q3): {len(score_mismatches)}")
if len(score_mismatches) > 0:
    print(score_mismatches[["passage_id", "firm", "year", "q1_score", "q2_score",
                             "q3_score", "classification_score"]].to_string(index=False))

# Check 2: does classification match the expected tier, allowing for the
# documented q1=0 override (score 2 without named function -> Symbolic)?
def expected_tier(row):
    score = row["classification_score"]
    if score == 3:
        return "Substantive"
    elif score == 2:
        return "Symbolic" if row["q1_score"] == 0 else "Transitional"
    else:
        return "Symbolic"

df["expected_classification"] = df.apply(expected_tier, axis=1)
tier_mismatches = df[df["classification"] != df["expected_classification"]]

print(f"\nTier mapping mismatches (unexplained by the q1=0 override): {len(tier_mismatches)}")
if len(tier_mismatches) > 0:
    print(tier_mismatches[["passage_id", "firm", "year", "q1_score", "q2_score",
                            "q3_score", "classification_score", "classification",
                            "expected_classification"]].to_string(index=False))
    tier_mismatches.to_csv("results/tier_mismatches.csv", index=False)
else:
    print("None found. Every classification is fully explained by the scoring "
          "rule plus the documented q1=0 override.")
