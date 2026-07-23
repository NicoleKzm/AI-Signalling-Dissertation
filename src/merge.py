import pandas as pd

# Load both datasets
signalling = pd.read_csv('data/signalling_scores.csv')
financial = pd.read_csv('data/final_dataset.csv')

# Check what we're working with
print("Signalling firms:", sorted(signalling['firm'].unique()))
print("Financial firms:", sorted(financial['Firm'].unique()))

# Standardise column names for merge
financial = financial.rename(columns={'Firm': 'firm', 'Year': 'year'})

# Merge on firm and year
merged = pd.merge(signalling, financial, on=['firm', 'year'], how='inner')

print(f"\nSignalling rows: {len(signalling)}")
print(f"Financial rows: {len(financial)}")
print(f"Merged rows: {len(merged)}")
print(f"\nMissing after merge: {len(signalling) - len(merged)} rows lost")

# Save
merged.to_csv('data/panel_dataset.csv', index=False)
print("\nDone — data/panel_dataset.csv saved")
print(merged[['firm', 'year', 'mean_signal_score', 'modal_classification',
              'Revenue_Growth_%', 'Gross_Margin_%', 'Stock_Price_Movement_%']].to_string())