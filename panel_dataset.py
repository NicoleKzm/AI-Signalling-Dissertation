import pandas as pd, numpy as np
df = pd.read_csv("panel_dataset.csv").dropna(subset=["Revenue_Growth_%"])
df["log_revenue"] = np.log(df["Revenue"].replace(0, np.nan))

cols = ["mean_signal_score", "Stock_Price_Movement_%",
        "Revenue_Growth_%", "Gross_Margin_%", "log_revenue"]
summary = df[cols].agg(["mean", "std", "min", "max", "count"]).round(3)
print(summary.T)   # .T flips it so variables are rows, stats are columns