import feedparser
import pandas as pd
import re
import yaml
from datetime import datetime
from pathlib import Path

def clean_html(raw_html):
    """Utility to remove HTML tags from RSS feed summaries."""
    if raw_html:
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext
    return ""

def fetch_rss_news(config_path="config.yaml") -> pd.DataFrame | None:
    """
    Fetches news from all RSS feeds listed in the config file
    and returns them as a DataFrame.
    """
    print("--- Starting RSS News Fetch ---")
    
    config = yaml.safe_load(open(config_path))
    rss_feeds = config.get('rss_feeds', {})
    raw_data_path = Path(config['data_paths']['raw'])
    
    if not rss_feeds:
        print("No RSS feeds found in config.yaml. Skipping.")
        return None

    all_articles = []
    for feed_name, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url)
            print(f"  - Parsing feed: {feed_name} ({len(feed.entries)} entries)")
            for entry in feed.entries:
                article = {
                    'source': feed_name,
                    'title': entry.get('title', 'N/A'),
                    'link': entry.get('link', 'N/A'),
                    'published': entry.get('published', datetime.now().isoformat()),
                    'summary': clean_html(entry.get('summary', ''))
                }
                all_articles.append(article)
        except Exception as e:
            print(f"Error parsing RSS feed {feed_name}: {e}")
    
    if not all_articles:
        print("\nNo articles fetched.")
        return None
        
    df = pd.DataFrame(all_articles)
    
    # --- CHANGE ---
    # We still save a raw copy, but now we also return the df
    # for the next task in the pipeline.
    filename = f"raw_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_file = raw_data_path / filename
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully fetched {len(df)} total articles.")
    print(f"Saved raw news to: {output_file}")
    print("--- RSS News Fetch Complete ---")
    
    return df

if __name__ == "__main__":
    fetch_rss_news(config_path="config.yaml")