# Cortexa — Indian Market Intelligence

An AI-powered investment research tool for NSE/BSE listed companies. Ask a question about any major Indian stock and get a structured analysis combining live news, historical precedents, fundamental data, and an LLM-generated verdict with confidence score.

**Live demo:** [cortexa-amber.vercel.app](https://cortexa-amber.vercel.app)  
**API:** [cortexa-747288447158.asia-south1.run.app/docs](https://cortexa-747288447158.asia-south1.run.app/docs)

---

## What It Does

When a user asks "Should I invest in Reliance?" the system:

1. Detects the company and maps it to its NSE/BSE ticker
2. Fetches the latest news about the company from Google News
3. Searches a vector database for historically similar news and how the stock reacted after each
4. Pulls current fundamentals — price, P/E, P/B, revenue, debt/equity, ROE, analyst rating
5. Sends everything to Gemini and generates a structured verdict with confidence score

The output covers current situation, historical precedent, fundamental analysis, risk factors, opportunities, and a final recommendation.

---

## Tech Stack

- **Backend:** FastAPI, Python — deployed on Google Cloud Run
- **AI/LLM:** Google Gemini 2.5 Flash
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database:** Qdrant Cloud
- **Market Data:** yfinance (NSE/BSE via `.NS` suffix)
- **News:** Google News RSS via feedparser
- **Frontend:** Vanilla HTML/CSS/JS — deployed on Vercel
- **Scheduler:** Google Cloud Scheduler — daily ingestion at 2:00 AM IST

---

## Project Structure

```
cortexa/
├── backend/
│   ├── main.py          # FastAPI server — /analyze endpoint
│   ├── ingestion.py     # News fetching, stock reaction calculation, Qdrant population
│   ├── analyzer.py      # Fundamentals fetching + Gemini reasoning
│   └── config.py        # Ticker map, company names, constants
│   ├── requirements.txt
├── frontend/
│   └── index.html       # Chat-style single-page frontend
├── .env
├── .gitignore
└── README.md
```

---

## Supported Companies

**IT:** TCS, Infosys, Wipro, HCL Technologies, Tech Mahindra, LTIMindtree

**Banking and Finance:** HDFC Bank, ICICI Bank, SBI, Kotak Mahindra Bank, Axis Bank, Bajaj Finance

**Energy and Industrial:** Reliance Industries, ONGC, Adani Ports, NTPC, Power Grid Corporation

**Consumer:** Hindustan Unilever, ITC, Asian Paints, Maruti Suzuki, Tata Motors, Zomato, Paytm, Nykaa

**Pharma:** Sun Pharma, Dr. Reddy's Laboratories, Cipla

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ujjwaltwri/cortexa.git
cd cortexa
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file inside the `backend/` directory:

```
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
```

- Get a Gemini API key from [Google AI Studio](https://aistudio.google.com)
- Get a free Qdrant Cloud cluster at [cloud.qdrant.io](https://cloud.qdrant.io)

### 4. Run ingestion

Fetches news for all supported companies, calculates historical stock reactions, and stores everything in Qdrant.

```bash
cd backend
python ingestion.py
```

Takes approximately 5–10 minutes for all supported companies.

### 5. Start the server

```bash
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

---

## API

Base URL: `https://cortexa-747288447158.asia-south1.run.app`

Interactive docs: [/docs](https://cortexa-747288447158.asia-south1.run.app/docs)

### POST /analyze

Takes a natural language query and returns a full investment analysis.

**Request:**
```json
{
  "query": "Should I invest in Reliance Industries?"
}
```

**Response:**
```json
{
  "error": false,
  "ticker": "RELIANCE.NS",
  "company": "Reliance Industries",
  "verdict": "BUY",
  "confidence": 72,
  "analysis": "...",
  "current_price": 2847.50,
  "day_change": 1.23,
  "latest_news": [...],
  "similar_past": [...]
}
```

### POST /ingest

Triggers a full ingestion run — fetches fresh news for all companies and updates Qdrant. This endpoint is called automatically every night at 2:00 AM IST by Cloud Scheduler.

### GET /tickers

Returns all supported companies and their tickers.

### GET /health

Health check.

---

## How the Vector Search Works

During ingestion, each news article is embedded using `all-MiniLM-L6-v2` and stored in Qdrant along with:

- The stock's price reaction 1 day, 3 days, and 7 days after the news
- Company ticker, publication date, headline, and summary

When a user asks about a company, the current query is embedded and searched against historical articles for that ticker. The most semantically similar past news is retrieved along with the actual stock movement that followed — giving Gemini concrete historical evidence to reason from.

---

## Automated Daily Ingestion

The Qdrant vector database is refreshed automatically every night at **2:00 AM IST** via a Google Cloud Scheduler job (`cortexa-daily-ingestion`) that hits the `/ingest` endpoint on the Cloud Run backend. This keeps news data and stock price reactions current without any manual intervention.

```
Schedule:  0 2 * * *  (Asia/Calcutta)
Target:    POST https://cortexa-747288447158.asia-south1.run.app/ingest
Region:    us-central1
```

---

## Deployment

### Backend — Google Cloud Run

```bash
cd backend
gcloud run deploy cortexa \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated
```

Set environment variables (`GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`) in the Cloud Run console under Variables & Secrets.

### Frontend — Vercel

The `frontend/` directory is deployed to Vercel. `API_BASE` in `index.html` points to the Cloud Run backend URL. Connect the GitHub repo to Vercel for automatic deploys on push.

### Scheduler — Google Cloud Scheduler

```bash
gcloud scheduler jobs create http cortexa-daily-ingestion \
  --schedule="0 2 * * *" \
  --uri="https://cortexa-747288447158.asia-south1.run.app/ingest" \
  --http-method=POST \
  --time-zone="Asia/Calcutta" \
  --location=us-central1
```

---

## Limitations

- Analysis quality depends on available news and historical data in Qdrant. Historical price reactions are only available for articles old enough for 1d/3d/7d prices to have settled.
- yfinance data can occasionally be delayed or incomplete for certain NSE tickers.
- This is a demo deployment — no rate limiting or auth is applied to the public API.
- For research and educational purposes only. Not financial advice.

---

## Disclaimer

This project is for educational and portfolio purposes. The analysis generated is AI-produced and should not be used as the sole basis for any investment decision. Always consult a SEBI-registered investment advisor before investing.

---

## Author

Ujjwal Kumar Tiwari  
[GitHub](https://github.com/ujjwaltwri) · [LinkedIn](https://www.linkedin.com/in/ujjwal-tiwari-020a21293/)