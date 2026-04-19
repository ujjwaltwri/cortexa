# FILE: backend/main.py

import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from config import TICKER_MAP, TICKER_NAMES
from ingestion import search_similar_news, fetch_news_for_ticker, get_stock_reaction, store_articles
from analyzer import fetch_latest_news, fetch_fundamentals, generate_analysis

app = FastAPI(title="Cortexa India API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MODELS
# ==========================================
class QueryRequest(BaseModel):
    query: str

# ==========================================
# TICKER DETECTION
# ==========================================
def detect_ticker(text: str) -> tuple[str, str] | tuple[None, None]:
    """Extract company/ticker from user query."""
    text_lower = text.lower()

    # Direct match
    for key, ticker in TICKER_MAP.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            return ticker, TICKER_NAMES.get(ticker, ticker)

    # Partial match fallback
    for key, ticker in TICKER_MAP.items():
        if key in text_lower:
            return ticker, TICKER_NAMES.get(ticker, ticker)

    return None, None

# ==========================================
# ROUTES
# ==========================================
@app.get("/")
def root():
    return {"status": "Cortexa India API is running", "version": "2.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/tickers")
def get_tickers():
    """Return all supported companies."""
    return {
        "companies": [
            {"ticker": ticker, "name": name}
            for ticker, name in TICKER_NAMES.items()
            if ticker not in ["^NSEI", "^BSESN"]
        ]
    }

@app.post("/analyze")
def analyze(request: QueryRequest):
    """
    Main endpoint — takes user query, detects company,
    fetches news + fundamentals + historical context,
    returns Gemini analysis.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Detect ticker
    ticker, company = detect_ticker(query)
    if not ticker:
        return {
            "error": True,
            "message": (
                "I couldn't identify a company in your query. "
                "Try asking about specific companies like 'Should I invest in Reliance?' "
                "or 'What do you think about TCS stock?'"
            )
        }

    print(f"📊 Analyzing {company} ({ticker}) for query: '{query}'")

    # 2. Fetch latest live news
    latest_news = fetch_latest_news(ticker, company)
    print(f"  📰 Got {len(latest_news)} latest news articles")

    # 3. Search historical similar news from Qdrant
    search_query = f"{company} {query}"
    similar_past = search_similar_news(search_query, ticker, top_k=5)
    print(f"  🔍 Found {len(similar_past)} similar historical articles")

    # 4. If no historical data exists for this ticker, ingest live news on the fly
    if not similar_past:
        print(f"  ⚡ No historical data for {ticker}, ingesting now...")
        try:
            articles = fetch_news_for_ticker(ticker)
            for article in articles:
                article["reaction"] = get_stock_reaction(ticker, article["published"])
            store_articles(articles)
            similar_past = search_similar_news(search_query, ticker, top_k=5)
        except Exception as e:
            print(f"  ⚠️ On-the-fly ingestion failed: {e}")

    # 5. Fetch fundamentals
    fundamentals = fetch_fundamentals(ticker)
    print(f"  💹 Got fundamentals: price=₹{fundamentals.get('current_price', 'N/A')}")

    # 6. Generate Gemini analysis
    result = generate_analysis(
        user_query=query,
        ticker=ticker,
        company=company,
        fundamentals=fundamentals,
        latest_news=latest_news,
        similar_past=similar_past,
    )

    return {
        "error": False,
        "ticker": ticker,
        "company": company,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "analysis": result["analysis"],
        "current_price": result.get("current_price"),
        "day_change": result.get("day_change"),
        "latest_news": latest_news[:3],
        "similar_past": similar_past[:3],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)