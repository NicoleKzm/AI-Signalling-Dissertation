import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, PooledOLS
from linearmodels.panel import compare
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────────────
df = pd.read_csv('panel_dataset.csv')

# ── Drop first year (NaN revenue growth) ──────────────────────────
df = df.dropna(subset=['Revenue_Growth_%'])

# ── Set Panel Structure ────────────────────────────────────────────
df = df.set_index(['firm', 'year'])

print(f"Panel dataset: {len(df)} observations")
print(f"Firms: {df.index.get_level_values('firm').nunique()}")
print(f"Years: {sorted(df.index.get_level_values('year').unique())}")
print(f"\nSignalling score summary:")
print(df['mean_signal_score'].describe())

# ── Sector Dummies ─────────────────────────────────────────────────
# Fashion & Apparel = baseline (omitted)
sector_map = {
    'ASOS': 'Fashion',
    'Boohoo': 'Fashion',
    'Zalando': 'Fashion',
    'Mytheresa': 'Fashion',
    'Boozt': 'Fashion',
    'About You': 'Fashion',
    'THG': 'Beauty_Multi',
    'HelloFresh': 'Grocery',
    'Westwing': 'Home',
    'Redcare Pharmacy': 'Health',
    'DocMorris': 'Health',
    'Moonpig': 'Gifts',
    'AO World': 'Electronics',
    'Allegro': 'Marketplace',
}

df['sector'] = df.index.get_level_values('firm').map(sector_map)
sector_dummies = pd.get_dummies(df['sector'], drop_first=False)
sector_dummies = sector_dummies.drop(columns=['Fashion'], errors='ignore')

# Add sector dummies to dataframe
for col in sector_dummies.columns:
    df[col] = sector_dummies[col].values

sector_cols = [c for c in sector_dummies.columns]

# ── Controls ───────────────────────────────────────────────────────
# Log market cap as firm size proxy (use revenue as alternative if missing)
df['log_revenue'] = np.log(df['Revenue'].replace(0, np.nan))

# Prior year performance (already lagged via Revenue_Growth_%)
# We'll use current gross margin as additional control
df['gross_margin'] = df['Gross_Margin_%']

# ── Define Models ──────────────────────────────────────────────────
dependent_vars = {
    'H1_Stock_Price': 'Stock_Price_Movement_%',
    'H2_Revenue_Growth': 'Revenue_Growth_%',
    'H3_Gross_Margin': 'Gross_Margin_%',
}

independent_var = 'mean_signal_score'
controls = ['log_revenue'] + sector_cols

print("\n" + "="*60)
print("PANEL REGRESSION RESULTS")
print("Fixed Effects: Firm + Year")
print("Primary IV: mean_signal_score")
print("="*60)

results_summary = []

for hypothesis, dv in dependent_vars.items():
    print(f"\n── {hypothesis}: DV = {dv} ──")

    # Drop missing values for this model
    model_df = df[[dv, independent_var] + controls].dropna()

    if len(model_df) < 20:
        print(f"  Insufficient observations: {len(model_df)}")
        continue

    # Build regressor matrix
    exog_vars = [independent_var] + controls
    exog = sm.add_constant(model_df[exog_vars])

    try:
        # Panel OLS with firm and year fixed effects
        model = PanelOLS(
            model_df[dv],
            exog,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )
        result = model.fit(cov_type='clustered', cluster_entity=True)

        print(f"  Observations: {result.nobs}")
        print(f"  R-squared (within): {result.rsquared:.4f}")
        print(f"\n  Coefficient on mean_signal_score:")
        coef = result.params[independent_var]
        se = result.std_errors[independent_var]
        pval = result.pvalues[independent_var]
        tstat = result.tstats[independent_var]

        print(f"    Coefficient: {coef:.4f}")
        print(f"    Std Error:   {se:.4f}")
        print(f"    T-stat:      {tstat:.4f}")
        print(f"    P-value:     {pval:.4f}")

        sig = ""
        if pval < 0.01:
            sig = "*** (p<0.01)"
        elif pval < 0.05:
            sig = "** (p<0.05)"
        elif pval < 0.10:
            sig = "* (p<0.10)"
        else:
            sig = "Not significant"

        print(f"    Significance: {sig}")

        # Hypothesis verdict
        if pval < 0.10 and coef > 0:
            verdict = "SUPPORTED"
        elif pval < 0.10 and coef < 0:
            verdict = "SUPPORTED (negative direction)"
        else:
            verdict = "NOT SUPPORTED"

        print(f"    Hypothesis verdict: {verdict}")

        results_summary.append({
            'Hypothesis': hypothesis,
            'DV': dv,
            'Coefficient': round(coef, 4),
            'Std_Error': round(se, 4),
            'T_stat': round(tstat, 4),
            'P_value': round(pval, 4),
            'Significance': sig,
            'Verdict': verdict,
            'N_obs': result.nobs,
            'R2_within': round(result.rsquared, 4)
        })

    except Exception as e:
        print(f"  Model error: {e}")

# ── Summary Table ──────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY TABLE")
print("="*60)
results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))
results_df.to_csv('regression_results.csv', index=False)
print("\nregression_results.csv saved")

# ── DIAGNOSTIC TESTS ──────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC TESTS")
print("="*60)

# VIF Test for multicollinearity
model_df = df[['Revenue_Growth_%', 'mean_signal_score',
               'log_revenue'] + sector_cols].dropna()

exog = model_df[['mean_signal_score', 'log_revenue'] + sector_cols]

# Convert to float to avoid type errors
exog = exog.astype(float)
exog_with_const = sm.add_constant(exog)

vif_data = pd.DataFrame()
vif_data['Variable'] = exog_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(
    exog_with_const.values.astype(float), i)
    for i in range(len(exog_with_const.columns))]

print("\nVariance Inflation Factors:")
print(vif_data.to_string(index=False))
print("\nVIF > 10 indicates problematic multicollinearity")

# ── ROBUSTNESS CHECK: Categorical Dummies ─────────────────────────
print("\n" + "="*60)
print("ROBUSTNESS CHECK: Categorical Dummies")
print("Baseline: Symbolic / No_AI_Disclosure")
print("="*60)

# Create dummies
df['is_transitional'] = (df['modal_classification'] == 'Transitional').astype(int)
df['is_substantive'] = (df['modal_classification'] == 'Substantive').astype(int)

for hypothesis, dv in dependent_vars.items():
    print(f"\n── {hypothesis}: DV = {dv} ──")
    model_df = df[[dv, 'is_transitional', 'is_substantive',
                   'log_revenue'] + sector_cols].dropna()

    exog = sm.add_constant(
        model_df[['is_transitional', 'is_substantive', 'log_revenue'] + sector_cols]
    )

    try:
        model = PanelOLS(
            model_df[dv],
            exog,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=False
        )
        result = model.fit(cov_type='clustered', cluster_entity=True)

        for var in ['is_transitional', 'is_substantive']:
            coef = result.params[var]
            pval = result.pvalues[var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else "ns"
            print(f"  {var}: coef={coef:.4f}, p={pval:.4f} {sig}")

    except Exception as e:
        print(f"  Error: {e}")

        # ── ROBUSTNESS CHECKS ─────────────────────────────────────────────
print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

robustness_results = []

checks = {
    "Excl. DocMorris 2023": df[~((df.index.get_level_values('firm') == 'DocMorris') & 
                                  (df.index.get_level_values('year') == 2023))],
    "Excl. Boohoo": df[df.index.get_level_values('firm') != 'Boohoo'],
    "Excl. DocMorris 2023 + Boohoo": df[~((df.index.get_level_values('firm') == 'DocMorris') & 
                                           (df.index.get_level_values('year') == 2023)) & 
                                        (df.index.get_level_values('firm') != 'Boohoo')],
}

for check_name, check_df in checks.items():
    print(f"\n── {check_name} (N={len(check_df)}) ──")
    for hypothesis, dv in dependent_vars.items():
        model_df = check_df[[dv, independent_var] + controls].dropna()
        if len(model_df) < 15:
            print(f"  {hypothesis}: insufficient observations")
            continue
        exog = sm.add_constant(model_df[exog_vars])
        try:
            model = PanelOLS(
                model_df[dv],
                exog,
                entity_effects=True,
                time_effects=True,
                drop_absorbed=True
            )
            result = model.fit(cov_type='clustered', cluster_entity=True)
            coef = result.params[independent_var]
            pval = result.pvalues[independent_var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else "ns"
            print(f"  {dv}: coef={coef:.4f}, p={pval:.4f} {sig}")
            robustness_results.append({
                'Check': check_name,
                'Hypothesis': hypothesis,
                'DV': dv,
                'Coefficient': round(coef, 4),
                'P_value': round(pval, 4),
                'Significance': sig,
                'N_obs': result.nobs
            })
        except Exception as e:
            print(f"  {dv}: Error — {e}")

pd.DataFrame(robustness_results).to_csv('robustness_results.csv', index=False)
print("\nrobustness_results.csv saved")