import yfinance as yf
import pandas as pd

# ── 14 firms with their Yahoo Finance tickers ──────────────────
firms = {
    "Zalando":           "ZAL.DE",
    "ASOS":              "ASC.L",
    "Boohoo":            "DEBS.L",
    "Mytheresa":         "LUXE",
    "Allegro":           "ALE.WA",
    "THG":               "THG.L",
    "HelloFresh":        "HFG.DE",
    "Westwing":          "WEW.DE",
    "Boozt":             "BOOZT.ST",
    "Redcare Pharmacy":  "RDC.DE",
    "Moonpig":           "MOON.L",
    "AO World":          "AO.L",
    "DocMorris":         "DOCM.SW",
    "About You":         "YOU",
}

years = [2021, 2022, 2023, 2024, 2025]
results = []

for firm, ticker in firms.items():
    print(f"Pulling data for {firm} ({ticker})...")
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        info = stock.info

        for year in years:
            row = {"Firm": firm, "Ticker": ticker, "Year": year}

            # ── Revenue Growth ─────────────────────────────────
            try:
                rev_cols = [c for c in financials.columns if c.year == year]
                prev_cols = [c for c in financials.columns if c.year == year - 1]
                if rev_cols and prev_cols:
                    rev = financials.loc["Total Revenue", rev_cols[0]]
                    prev_rev = financials.loc["Total Revenue", prev_cols[0]]
                    row["Revenue"] = rev
                    row["Revenue_Growth_%"] = round(((rev - prev_rev) / prev_rev) * 100, 2)
                else:
                    row["Revenue"] = None
                    row["Revenue_Growth_%"] = None
            except Exception as e:
                row["Revenue"] = None
                row["Revenue_Growth_%"] = None

            # ── Gross Margin ───────────────────────────────────
            try:
                if rev_cols:
                    gross_profit = financials.loc["Gross Profit", rev_cols[0]]
                    revenue = financials.loc["Total Revenue", rev_cols[0]]
                    row["Gross_Margin_%"] = round((gross_profit / revenue) * 100, 2)
                else:
                    row["Gross_Margin_%"] = None
            except Exception as e:
                row["Gross_Margin_%"] = None

            # ── Stock Price Movement ───────────────────────────
            try:
                hist = stock.history(
                    start=f"{year}-01-01",
                    end=f"{year}-12-31"
                )
                if not hist.empty:
                    price_start = hist["Close"].iloc[0]
                    price_end = hist["Close"].iloc[-1]
                    row["Stock_Price_Start"] = round(price_start, 2)
                    row["Stock_Price_End"] = round(price_end, 2)
                    row["Stock_Price_Movement_%"] = round(
                        ((price_end - price_start) / price_start) * 100, 2
                    )
                else:
                    row["Stock_Price_Start"] = None
                    row["Stock_Price_End"] = None
                    row["Stock_Price_Movement_%"] = None
            except Exception as e:
                row["Stock_Price_Start"] = None
                row["Stock_Price_End"] = None
                row["Stock_Price_Movement_%"] = None

            results.append(row)

    except Exception as e:
        print(f"  ERROR with {firm}: {e}")
        for year in years:
            results.append({
                "Firm": firm, "Ticker": ticker, "Year": year,
                "Revenue": None, "Revenue_Growth_%": None,
                "Gross_Margin_%": None, "Stock_Price_Start": None,
                "Stock_Price_End": None, "Stock_Price_Movement_%": None
            })

# ── Save to CSV ────────────────────────────────────────────────
df = pd.DataFrame(results)
df = df.sort_values(["Firm", "Year"]).reset_index(drop=True)
df.to_csv("financial_data.csv", index=False)

print("\nDone. financial_data.csv saved.")
print(df.to_string())