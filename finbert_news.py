import pandas as pd
import requests
import time
from transformers import pipeline
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────
NEWSAPI_KEY = "af80fbcec6e74348b832f06bcd43d5b2"

print("Loading FinBERT model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    max_length=512,
    truncation=True
)
print("FinBERT loaded.")

# ── Firms and search terms ─────────────────────────────────────────
FIRMS = {
    "Zalando": "Zalando AI",
    "ASOS": "ASOS AI technology",
    "Boohoo": "Boohoo technology digital",
    "Mytheresa": "Mytheresa AI technology",
    "Allegro": "Allegro AI technology",
    "THG": "THG AI technology",
    "HelloFresh": "HelloFresh AI technology",
    "Westwing": "Westwing AI technology",
    "Boozt": "Boozt AI technology",
    "Redcare Pharmacy": "Redcare Pharmacy AI digital",
    "Moonpig": "Moonpig AI technology",
    "AO World": "AO World AI technology",
    "DocMorris": "DocMorris AI digital",
    "About You": "About You AI fashion",
}

YEARS = [2021, 2022, 2023, 2024, 2025]

# ── NewsAPI fetch ──────────────────────────────────────────────────
def get_headlines(firm, year):
    """Fetch headlines from NewsAPI for a firm in a given year."""
    headlines = []
    
    # NewsAPI free tier only goes back 1 month
    # For historical data we use everything endpoint
    url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": FIRMS[firm],
        "from": f"{year}-01-01",
        "to": f"{year}-12-31",
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": NEWSAPI_KEY,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                if title and len(title) > 20:
                    headlines.append(title)
                if description and len(description) > 20:
                    headlines.append(description)
        else:
            print(f"  API error: {data.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"  Request error: {e}")
    
    return headlines[:20]


# ── FinBERT Sentiment ──────────────────────────────────────────────
def analyse_sentiment(headlines):
    if not headlines:
        return None, None, 0
    
    results = []
    for headline in headlines:
        try:
            result = sentiment_pipeline(headline[:512])[0]
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                numeric = score
            elif label == 'negative':
                numeric = -score
            else:
                numeric = 0
            
            results.append({
                'headline': headline,
                'label': label,
                'score': score,
                'numeric_score': numeric
            })
        except Exception:
            continue
    
    if not results:
        return None, None, 0
    
    mean_sentiment = sum(r['numeric_score'] for r in results) / len(results)
    modal_sentiment = max(
        ['positive', 'neutral', 'negative'],
        key=lambda x: sum(1 for r in results if r['label'] == x)
    )
    
    return round(mean_sentiment, 4), modal_sentiment, len(results)


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    all_headlines = []
    
    for firm in FIRMS:
        for year in YEARS:
            if firm == "About You" and year == 2025:
                continue
            
            print(f"\nProcessing {firm} {year}...")
            headlines = get_headlines(firm, year)
            print(f"  Found {len(headlines)} headlines")
            
            mean_sent, modal_sent, n = analyse_sentiment(headlines)
            
            results.append({
                'firm': firm,
                'year': year,
                'n_headlines': n,
                'mean_sentiment': mean_sent,
                'modal_sentiment': modal_sent,
            })
            
            for h in headlines:
                all_headlines.append({
                    'firm': firm,
                    'year': year,
                    'headline': h
                })
            
            time.sleep(1)
    
    # Save
    df_sentiment = pd.DataFrame(results)
    df_sentiment.to_csv('sentiment_scores.csv', index=False)
    
    df_headlines = pd.DataFrame(all_headlines)
    df_headlines.to_csv('all_headlines.csv', index=False)
    
    print("\nDone!")
    print(df_sentiment[['firm', 'year', 'n_headlines',
                         'mean_sentiment', 'modal_sentiment']].to_string())