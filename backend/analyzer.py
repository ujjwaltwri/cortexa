# FILE: backend/analyzer.py

import yfinance as yf
import google.generativeai as genai
import feedparser
import time
from datetime import datetime
from config import GEMINI_API_KEY, TICKER_NAMES

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ==========================================
# FETCH LIVE NEWS (for current situation)
# ==========================================
def fetch_latest_news(ticker: str, company: str, limit: int = 6) -> list[dict]:
    """Fetch the most recent news for the company right now."""
    queries = [f"{company} NSE stock", f"{company} India business"]
    articles = []
    seen = set()

    for query in queries:
        encoded = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                articles.append({
                    "title": title,
                    "summary": entry.get("summary", title)[:300],
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            print(f"⚠️ Live news error: {e}")
        time.sleep(0.2)

    return articles[:limit]


# ==========================================
# FETCH STOCK FUNDAMENTALS
# ==========================================
def fetch_fundamentals(ticker: str) -> dict:
    """Get key fundamentals from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Current price
        hist = stock.history(period="5d")
        current_price = round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else None
        prev_price = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else None
        day_change = None
        if current_price and prev_price:
            day_change = round((current_price - prev_price) / prev_price * 100, 2)

        # 52 week
        week_52_high = info.get("fiftyTwoWeekHigh")
        week_52_low = info.get("fiftyTwoWeekLow")

        return {
            "current_price": current_price,
            "day_change_pct": day_change,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "dividend_yield": info.get("dividendYield"),
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
            "beta": info.get("beta"),
            "analyst_rating": info.get("recommendationKey", "").upper(),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }
    except Exception as e:
        print(f"⚠️ Fundamentals error for {ticker}: {e}")
        return {}


# ==========================================
# FORMAT HELPERS
# ==========================================
def format_market_cap(mc):
    if not mc:
        return "N/A"
    if mc >= 1e12:
        return f"₹{mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"₹{mc/1e9:.2f}B"
    if mc >= 1e7:
        return f"₹{mc/1e7:.2f}Cr"
    return f"₹{mc:,.0f}"

def format_revenue(rev):
    if not rev:
        return "N/A"
    if rev >= 1e12:
        return f"₹{rev/1e12:.2f}T"
    if rev >= 1e9:
        return f"₹{rev/1e9:.2f}B"
    if rev >= 1e7:
        return f"₹{rev/1e7:.2f}Cr"
    return f"₹{rev:,.0f}"


# ==========================================
# GEMINI ANALYSIS
# ==========================================
def generate_analysis(
    user_query: str,
    ticker: str,
    company: str,
    fundamentals: dict,
    latest_news: list[dict],
    similar_past: list[dict],
) -> dict:
    """Send everything to Gemini and get a structured investment analysis."""

    # Format similar past news
    past_news_text = ""
    if similar_past:
        past_news_text = "\n".join([
            f"- [{h['published'][:10] if h['published'] else 'N/A'}] {h['title']}\n"
            f"  Stock reaction → 1d: {h['return_1d']}%, 3d: {h['return_3d']}%, 7d: {h['return_7d']}%"
            for h in similar_past if h.get("return_1d") is not None
        ])
    if not past_news_text:
        past_news_text = "No closely matching historical news found."

    # Format latest news
    latest_news_text = "\n".join([
        f"- {n['title']}" for n in latest_news
    ]) or "No recent news available."

    # Format fundamentals
    f = fundamentals
    fund_text = f"""
Current Price: ₹{f.get('current_price', 'N/A')} ({f.get('day_change_pct', 'N/A')}% today)
Market Cap: {format_market_cap(f.get('market_cap'))}
P/E Ratio: {f.get('pe_ratio', 'N/A')}
P/B Ratio: {f.get('pb_ratio', 'N/A')}
Revenue: {format_revenue(f.get('revenue'))}
Profit Margin: {round(f.get('profit_margin', 0) * 100, 2) if f.get('profit_margin') else 'N/A'}%
Debt/Equity: {f.get('debt_to_equity', 'N/A')}
ROE: {round(f.get('roe', 0) * 100, 2) if f.get('roe') else 'N/A'}%
52W High: ₹{f.get('week_52_high', 'N/A')} | 52W Low: ₹{f.get('week_52_low', 'N/A')}
Beta: {f.get('beta', 'N/A')}
Analyst Rating: {f.get('analyst_rating', 'N/A')}
Sector: {f.get('sector', 'N/A')}
""".strip()

    prompt = f"""
You are Cortexa, an AI investment analyst specializing in the Indian stock market (NSE/BSE).
A user has asked: "{user_query}"

You are analyzing: {company} ({ticker})

---
CURRENT FUNDAMENTALS:
{fund_text}

---
LATEST NEWS (happening right now):
{latest_news_text}

---
HISTORICALLY SIMILAR NEWS & HOW THE STOCK REACTED:
{past_news_text}

---
YOUR TASK:
Provide a detailed, honest investment analysis. Structure your response EXACTLY as follows:

VERDICT: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
CONFIDENCE: [0-100]%

CURRENT SITUATION:
[2-3 sentences summarizing what is happening with this company right now based on the latest news and fundamentals]

HISTORICAL PRECEDENT:
[2-3 sentences explaining what happened in the past when similar news/conditions occurred for this stock, with specific return figures if available]

FUNDAMENTAL ANALYSIS:
[2-3 sentences on valuation, financial health, and key ratios — is it cheap or expensive? strong or weak balance sheet?]

RISK FACTORS:
[2-3 key risks as bullet points]

OPPORTUNITY FACTORS:
[2-3 key opportunities as bullet points]

FINAL RECOMMENDATION:
[2-3 sentences of your honest final take — what should the user actually do?]

DISCLAIMER: This is AI-generated analysis for educational purposes only. Not financial advice. Always consult a SEBI-registered advisor before investing.

---
Be honest. If the outlook is negative, say so. Do not be blindly optimistic.
If data is insufficient, say so clearly. Keep language clear and direct — no jargon overload.
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Parse confidence and verdict
        verdict = "HOLD"
        confidence = 50

        for line in raw.split("\n"):
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip()
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = int(line.replace("CONFIDENCE:", "").replace("%", "").strip())
                except Exception:
                    pass

        return {
            "verdict": verdict,
            "confidence": confidence,
            "analysis": raw,
            "ticker": ticker,
            "company": company,
            "current_price": f.get("current_price"),
            "day_change": f.get("day_change_pct"),
        }

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return {
            "verdict": "ERROR",
            "confidence": 0,
            "analysis": f"Analysis failed: {str(e)}",
            "ticker": ticker,
            "company": company,
        }