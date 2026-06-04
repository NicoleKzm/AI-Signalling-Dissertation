import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from transformers import pipeline
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────
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
    "Zalando": "Zalando AI artificial intelligence",
    "ASOS": "ASOS AI artificial intelligence technology",
    "Boohoo": "Boohoo AI technology digital",
    "Mytheresa": "Mytheresa AI technology luxury",
    "Allegro": "Allegro AI technology ecommerce",
    "THG": "THG AI artificial intelligence technology",
    "HelloFresh": "HelloFresh AI technology machine learning",
    "Westwing": "Westwing AI technology digital",
    "Boozt": "Boozt AI technology Nordic",
    "Redcare Pharmacy": "Redcare Pharmacy AI digital health",
    "Moonpig": "Moonpig AI technology personalisation",
    "AO World": "AO World AI technology ecommerce",
    "DocMorris": "DocMorris AI digital pharmacy",
    "About You": "About You AI fashion technology",
}

YEARS = [2021, 2022, 2023, 2024, 2025]

# ── News scraping via DuckDuckGo ───────────────────────────────────
def get_headlines(firm, year):
    """Fetch news headlines for a firm in a given year."""
    query = f"{FIRMS[firm]} {year}"
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    headlines = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract result snippets
        results = soup.find_all('a', {'class': 'result__a'})
        snippets = soup.find_all('a', {'class': 'result__snippet'})
        
        for r in results[:10]:
            text = r.get_text(strip=True)
            if len(text) > 20:
                headlines.append(text)
        
        for s in snippets[:10]:
            text = s.get_text(strip=True)
            if len(text) > 20:
                headlines.append(text)
                
    except Exception as e:
        print(f"  Scraping error for {firm} {year}: {e}")
    
    return headlines[:15]  # Cap at 15 headlines per firm-year


# ── FinBERT Sentiment Analysis ─────────────────────────────────────
def analyse_sentiment(headlines):
    """Run FinBERT on a list of headlines."""
    if not headlines:
        return None, None, 0
    
    results = []
    for headline in headlines:
        try:
            result = sentiment_pipeline(headline[:512])[0]
            score = result['score']
            label = result['label']
            
            # Convert to numeric: positive=1, neutral=0, negative=-1
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
        except Exception as e:
            continue
    
    if not results:
        return None, None, 0
    
    # Aggregate
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
            # Skip About You 2025
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
            
            time.sleep(2)  # Be polite to servers
    
    # Save
    df_sentiment = pd.DataFrame(results)
    df_sentiment.to_csv('sentiment_scores.csv', index=False)
    
    df_headlines = pd.DataFrame(all_headlines)
    df_headlines.to_csv('all_headlines.csv', index=False)
    
    print("\nDone!")
    print(df_sentiment[['firm', 'year', 'n_headlines', 
                         'mean_sentiment', 'modal_sentiment']].to_string())