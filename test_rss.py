import feedparser

def test_google_rss(ticker):
    print(f"\n--- Testing Google News Feed for {ticker} ---")
    
    # This is the specific, high-quality RSS format for Google Finance/News
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    
    print(f"1. Hitting URL: {url}")
    feed = feedparser.parse(url)
    
    # Check if it worked
    status = getattr(feed, 'status', 'Unknown')
    print(f"2. Status Code: {status}")
    
    if hasattr(feed, 'bozo') and feed.bozo:
        print(f"3. Error: {feed.bozo_exception}")
        
    count = len(feed.entries)
    print(f"4. Articles Found: {count}")
    
    if count > 0:
        print(f"✅ SUCCESS! First Headline: {feed.entries[0].title}")
    else:
        print("❌ FAILURE. Google returned 0 articles. (Likely blocked)")

if __name__ == "__main__":
    # Test a known active stock
    test_google_rss("MSFT")
    test_google_rss("GOOGL")