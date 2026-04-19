# FILE: backend/ingestion.py
# Run this script once to populate Qdrant with historical news + stock reactions.
# Then run periodically (daily cron) to keep it fresh.

import feedparser
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
import uuid
import time
import re
from config import (
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME,
    VECTOR_SIZE, ALL_TICKERS, TICKER_NAMES
)

# ==========================================
# INIT
# ==========================================
print("🔧 Initializing ingestion pipeline...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ==========================================
# QDRANT COLLECTION SETUP
# ==========================================
def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"✅ Collection already exists: {COLLECTION_NAME}")

# ==========================================
# NEWS FETCHING
# ==========================================
def get_company_name(ticker: str) -> str:
    return TICKER_NAMES.get(ticker, ticker.replace(".NS", ""))

def fetch_news_for_ticker(ticker: str) -> list[dict]:
    """Fetch news from Google News RSS for a given company."""
    company = get_company_name(ticker)
    # Search both company name and ticker
    queries = [
        f"{company} stock NSE",
        f"{company} BSE share price",
        f"{company.split()[0]} India earnings",
    ]

    articles = []
    seen_titles = set()

    for query in queries:
        encoded = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                title = entry.get("title", "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                # Parse date
                published = entry.get("published", "")
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                except Exception:
                    pub_date = datetime.now()

                articles.append({
                    "title": title,
                    "summary": entry.get("summary", title)[:500],
                    "published": pub_date,
                    "link": entry.get("link", ""),
                    "ticker": ticker,
                    "company": company,
                })
        except Exception as e:
            print(f"⚠️ News fetch error for {ticker}: {e}")

        time.sleep(0.3)  # be polite to Google

    return articles

# ==========================================
# STOCK REACTION CALCULATION
# ==========================================
def get_stock_reaction(ticker: str, news_date: datetime) -> dict:
    """
    Fetch stock price around the news date and calculate
    1d, 3d, 7d returns after the news.
    """
    try:
        start = news_date - timedelta(days=2)
        end = news_date + timedelta(days=10)

        stock = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if stock.empty or len(stock) < 2:
            return {"return_1d": None, "return_3d": None, "return_7d": None}

        # Get price on or after news date
        stock.index = pd.to_datetime(stock.index)
        after = stock[stock.index >= pd.Timestamp(news_date.date())]

        if after.empty:
            return {"return_1d": None, "return_3d": None, "return_7d": None}

        base_price = float(after["Close"].iloc[0])

        def pct(days):
            if len(after) > days:
                future = float(after["Close"].iloc[days])
                return round((future - base_price) / base_price * 100, 2)
            return None

        return {
            "return_1d": pct(1),
            "return_3d": pct(3),
            "return_7d": pct(7),
            "base_price": round(base_price, 2),
        }

    except Exception as e:
        print(f"⚠️ Stock reaction error for {ticker}: {e}")
        return {"return_1d": None, "return_3d": None, "return_7d": None}

# ==========================================
# EMBED & STORE
# ==========================================
def store_articles(articles: list[dict]):
    """Embed articles and upsert into Qdrant."""
    if not articles:
        return

    texts = [f"{a['title']}. {a['summary']}" for a in articles]
    vectors = embedder.encode(texts, show_progress_bar=False).tolist()

    points = []
    for i, (article, vector) in enumerate(zip(articles, vectors)):
        reaction = article.get("reaction", {})
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "title": article["title"],
                    "summary": article["summary"],
                    "published": article["published"].isoformat(),
                    "link": article["link"],
                    "ticker": article["ticker"],
                    "company": article["company"],
                    "return_1d": reaction.get("return_1d"),
                    "return_3d": reaction.get("return_3d"),
                    "return_7d": reaction.get("return_7d"),
                    "base_price": reaction.get("base_price"),
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  ✅ Stored {len(points)} articles")

# ==========================================
# MAIN INGESTION LOOP
# ==========================================
def run_ingestion(tickers: list[str] = None):
    """Run full ingestion for all or selected tickers."""
    ensure_collection()
    tickers = tickers or ALL_TICKERS

    print(f"\n📰 Starting ingestion for {len(tickers)} tickers...\n")

    for ticker in tickers:
        company = get_company_name(ticker)
        print(f"→ Processing {company} ({ticker})")

        # 1. Fetch news
        articles = fetch_news_for_ticker(ticker)
        print(f"  📰 Found {len(articles)} articles")

        if not articles:
            continue

        # 2. Get stock reactions for each article
        for article in articles:
            article["reaction"] = get_stock_reaction(ticker, article["published"])

        # 3. Store in Qdrant
        store_articles(articles)
        time.sleep(1)  # rate limit yfinance

    print("\n✅ Ingestion complete!")

# ==========================================
# QUERY FUNCTION (used by main.py)
# ==========================================
def search_similar_news(query: str, ticker: str, top_k: int = 5) -> list[dict]:
    """Search Qdrant for historically similar news for a ticker."""
    vector = embedder.encode([query])[0].tolist()

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=Filter(
            must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
        ),
        limit=top_k,
        with_payload=True,
    )

    hits = []
    for r in results:
        p = r.payload
        hits.append({
            "title": p.get("title"),
            "summary": p.get("summary"),
            "published": p.get("published"),
            "link": p.get("link"),
            "return_1d": p.get("return_1d"),
            "return_3d": p.get("return_3d"),
            "return_7d": p.get("return_7d"),
            "base_price": p.get("base_price"),
            "score": round(r.score, 3),
        })

    return hits


if __name__ == "__main__":
    run_ingestion()