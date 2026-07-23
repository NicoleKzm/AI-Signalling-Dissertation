"""
Chapter 4 cites two different zero-score-share figures (64.8% in Section 4.2.1, implied
63.8% baseline in Section 4.4.3); this script computes the true value directly from
data/signalling_scores.csv and prints it to resolve the discrepancy.
"""

import pandas as pd

df = pd.read_csv("data/signalling_scores.csv")

total_firm_years = len(df)
zero_score_count = (df["mean_signal_score"] == 0).sum()
zero_score_pct = round(zero_score_count / total_firm_years * 100, 1)

print(f"Total firm-years: {total_firm_years}")
print(f"Firm-years with mean_signal_score exactly 0: {zero_score_count}")
print(f"Percentage: {zero_score_pct}%")

print("\nComparison:")
print(f"  Chapter 4.2.1 claims: 64.8%")
print(f"  Chapter 4.4.3 implies baseline: 63.8%")
print(f"  Actual computed value: {zero_score_pct}%")

if zero_score_pct == 64.8:
    print("\n-> 4.2.1 is correct. 4.4.3's '63.8%' baseline reference needs correcting to 64.8%.")
elif zero_score_pct == 63.8:
    print("\n-> 4.4.3 is correct. 4.2.1's '64.8%' needs correcting to 63.8%.")
else:
    print(f"\n-> NEITHER figure matches. The true value is {zero_score_pct}%. "
          f"Both 4.2.1 and 4.4.3 need correcting, and the '65.2% after exclusion' "
          f"figure in 4.4.3 should also be re-checked against the fragmented-passage-excluded dataset.")

# Show the actual zero-score rows for a sanity check
print("\nFirm-years with a zero score:")
print(df[df["mean_signal_score"] == 0][["firm", "year", "total_passages", "modal_classification"]].to_string(index=False))
