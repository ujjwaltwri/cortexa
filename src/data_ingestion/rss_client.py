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
    Fetches news using the Google News RSS feed that we verified works.
    """
    print(f"--- 🎯 Hunting specific news for {len(tickers)} tickers ---")
    all_articles = []
    
    for ticker in tickers:
        clean_ticker = ticker.replace('^', '')
        print(f"  - Fetching news for: {clean_ticker}...")
        
        # --- THE WORKING URL FROM YOUR TEST ---
        # We removed "+outlook" to ensure we get the full firehose of data
        google_news_url = (
            f"https://news.google.com/rss/search?q={clean_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        )
        
        try:
            feed = feedparser.parse(google_news_url)
            
            if not feed.entries:
                print(f"    Warning: No entries found for {clean_ticker} in Google RSS.")
            
            for entry in feed.entries:
                summary = clean_html(entry.get('summary', '')) if entry.get('summary') else entry.get('title')
                all_articles.append({
                    'source': f"GoogleNews-{clean_ticker}",
                    'title': entry.get('title', 'N/A'),
                    'link': entry.get('link', '#'),
                    'published': entry.get('published', datetime.now().isoformat()),
                    'summary': summary,
                    'ticker': clean_ticker # Vital tag for retrieval
                })
            
        except Exception as e:
            print(f"    Error fetching Google News for {clean_ticker}: {e}")

        # Failover to Yahoo just in case
        try:
            stock = yf.Ticker(clean_ticker)
            news_items = stock.news
            if news_items:
                for item in news_items:
                    pub_time = item.get('providerPublishTime', 0)
                    pub_date = datetime.fromtimestamp(pub_time).isoformat() if pub_time else datetime.now().isoformat()
                    all_articles.append({
                        'source': f"YahooFinance-{clean_ticker}",
                        'title': item.get('title', 'N/A'),
                        'link': item.get('link', '#'),
                        'published': pub_date,
                        'summary': item.get('title'),
                        'ticker': clean_ticker
                    })
        except Exception:
            pass
            
    return all_articles

def fetch_rss_news(config_path="config.yaml") -> pd.DataFrame | None:
    print("--- Starting Comprehensive News Fetch ---")
    
    config = yaml.safe_load(open(config_path))
    rss_feeds = config.get('rss_feeds', {})
    tickers = config.get('tickers', []) 
    raw_data_path = Path(config['data_paths']['raw'])
    
    all_articles = []

    # 1. Targeted Ticker News (PRIORITY)
    ticker_articles = fetch_specific_ticker_news(tickers)
    all_articles.extend(ticker_articles)

    # 2. General RSS Feeds
    if rss_feeds:
        for feed_name, url in rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    all_articles.append({
                        'source': feed_name,
                        'title': entry.get('title', 'N/A'),
                        'link': entry.get('link', 'N/A'),
                        'published': entry.get('published', datetime.now().isoformat()),
                        'summary': clean_html(entry.get('summary', '')),
                        'ticker': None 
                    })
            except Exception:
                pass
    
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