# FILE: backend/config.py

import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INGEST_SECRET = os.getenv("INGEST_SECRET", "change-me")

# ==========================================
# QDRANT SETTINGS
# ==========================================
COLLECTION_NAME = "cortexa_india_news"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output size

# ==========================================
# INDIAN MARKET TICKERS
# ==========================================
TICKER_MAP = {
    # IT
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "wipro": "WIPRO.NS",
    "hcl": "HCLTECH.NS",
    "hcltech": "HCLTECH.NS",
    "hcl tech": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS",
    "techm": "TECHM.NS",
    "ltimindtree": "LTIM.NS",
    "lti": "LTIM.NS",
    "mindtree": "LTIM.NS",

    # Banking & Finance
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "axis bank": "AXISBANK.NS",
    "axis": "AXISBANK.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj": "BAJFINANCE.NS",

    # Energy & Industrial
    "reliance": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "ongc": "ONGC.NS",
    "adani ports": "ADANIPORTS.NS",
    "adani": "ADANIPORTS.NS",
    "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS",

    # Consumer & Auto
    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
    "itc": "ITC.NS",
    "asian paints": "ASIANPAINT.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",

    # Tata Motors demerged Oct 2025 into two listed entities:
    # TMPV.NS = Tata Motors Passenger Vehicles (JLR, EVs, passenger cars)
    # TMCV.NS = Tata Motors Commercial Vehicles (trucks, buses) — renamed "Tata Motors Ltd"
    "tata motors": "TMPV.NS",
    "tata motors passenger": "TMPV.NS",
    "tmpv": "TMPV.NS",
    "tata motors commercial": "TMCV.NS",
    "tmcv": "TMCV.NS",
    "tata cv": "TMCV.NS",

    # Zomato rebranded to Eternal Limited (Oct 2025)
    "zomato": "ETERNAL.NS",
    "eternal": "ETERNAL.NS",
    "eternal limited": "ETERNAL.NS",

    "paytm": "PAYTM.NS",
    "one97": "PAYTM.NS",
    "nykaa": "NYKAA.NS",
    "fsn": "NYKAA.NS",

    # Pharma
    "sun pharma": "SUNPHARMA.NS",
    "sun pharmaceutical": "SUNPHARMA.NS",
    "dr reddy": "DRREDDY.NS",
    "dr. reddy": "DRREDDY.NS",
    "drreddys": "DRREDDY.NS",
    "cipla": "CIPLA.NS",

    # Indices
    "nifty": "^NSEI",
    "sensex": "^BSESN",
    "nifty 50": "^NSEI",
    "market": "^NSEI",
}

# All supported tickers for ingestion (excludes indices)
ALL_TICKERS = list(set(
    v for v in TICKER_MAP.values()
    if not v.startswith("^")
))

# Company display names
TICKER_NAMES = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "TECHM.NS": "Tech Mahindra",
    "LTIM.NS": "LTIMindtree",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "RELIANCE.NS": "Reliance Industries",
    "ONGC.NS": "ONGC",
    "ADANIPORTS.NS": "Adani Ports",
    "NTPC.NS": "NTPC",
    "POWERGRID.NS": "Power Grid Corporation",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TMPV.NS": "Tata Motors (Passenger Vehicles)",   # demerged Oct 2025
    "TMCV.NS": "Tata Motors (Commercial Vehicles)",  # demerged Oct 2025
    "ETERNAL.NS": "Zomato (Eternal Ltd)",            # rebranded Oct 2025
    "PAYTM.NS": "Paytm",
    "NYKAA.NS": "Nykaa",
    "SUNPHARMA.NS": "Sun Pharma",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "CIPLA.NS": "Cipla",
    "^NSEI": "Nifty 50",
    "^BSESN": "BSE Sensex",
}