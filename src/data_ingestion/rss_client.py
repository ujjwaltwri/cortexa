import feedparser
import pandas as pd
import re
import yaml
import yfinance as yf
from datetime import datetime
from pathlib import Path

def clean_html(raw_html):
    if raw_html:
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext
    return ""

def fetch_specific_ticker_news(tickers):
    """
    Explicitly fetches news for each ticker using yfinance API.
    """
    print(f"--- 🎯 Hunting specific news for {len(tickers)} tickers ---")
    all_articles = []
    
    for ticker in tickers:
        clean_ticker = ticker.replace('^', '')
        print(f"  - Fetching news for: {clean_ticker}...")
        
        try:
            stock = yf.Ticker(clean_ticker)
            news_items = stock.news
            
            if not news_items:
                print(f"    (No news found for {clean_ticker})")
                continue
                
            for item in news_items:
                pub_time = item.get('providerPublishTime', 0)
                pub_date = datetime.fromtimestamp(pub_time).isoformat() if pub_time else datetime.now().isoformat()
                
                article = {
                    'source': f"YahooFinance-{clean_ticker}",
                    'title': item.get('title', 'N/A'),
                    'link': item.get('link', '#'),
                    'published': pub_date,
                    'summary': item.get('relatedTickers', [clean_ticker])[0] if 'relatedTickers' in item else item.get('title'),
                    
                    # --- CRITICAL FIX: Add the Ticker Tag ---
                    'ticker': clean_ticker 
                    # ----------------------------------------
                }
                all_articles.append(article)
                
        except Exception as e:
            print(f"    Error fetching {clean_ticker}: {e}")
            
    return all_articles

def fetch_rss_news(config_path="config.yaml") -> pd.DataFrame | None:
    print("--- Starting Comprehensive News Fetch ---")
    
    config = yaml.safe_load(open(config_path))
    rss_feeds = config.get('rss_feeds', {})
    tickers = config.get('tickers', []) 
    raw_data_path = Path(config['data_paths']['raw'])
    
    all_articles = []

    # 1. General RSS Feeds
    if rss_feeds:
        for feed_name, url in rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                # print(f"  - Parsing RSS: {feed_name} ({len(feed.entries)} entries)")
                for entry in feed.entries:
                    article = {
                        'source': feed_name,
                        'title': entry.get('title', 'N/A'),
                        'link': entry.get('link', 'N/A'),
                        'published': entry.get('published', datetime.now().isoformat()),
                        'summary': clean_html(entry.get('summary', '')),
                        'ticker': None # General news has no specific ticker
                    }
                    all_articles.append(article)
            except Exception as e:
                print(f"Error parsing RSS feed {feed_name}: {e}")

    # 2. Targeted Ticker News
    ticker_articles = fetch_specific_ticker_news(tickers)
    all_articles.extend(ticker_articles)
    
    if not all_articles:
        print("\nNo articles fetched.")
        return None
        
    df = pd.DataFrame(all_articles)
    df = df.drop_duplicates(subset=['title'])
    
    filename = f"raw_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_file = raw_data_path / filename
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully fetched {len(df)} total articles.")
    print(f"Saved raw news to: {output_file}")
    print("--- RSS News Fetch Complete ---")
    
    return df

if __name__ == "__main__":
    fetch_rss_news(config_path="config.yaml")