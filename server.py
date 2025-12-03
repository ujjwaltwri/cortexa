from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import re

# Import your project's core components
from src.rag.retrieval import query_vector_db
from src.reasoning.agent import get_llm_reasoning
from src.signals.rag_signal import RAGSignalEngine 
from src.utils import get_current_regime, get_latest_features

# --- Ticker Mapping (For Qualitative Search) ---
TICKER_MAP = {
    "apple": "AAPL", "aapl": "AAPL", "appl": "AAPL",
    "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL", "goog": "GOOGL",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA",
    "sp500": "GSPC", "market": "GSPC", "s&p": "GSPC"
}

# --- Global State ---
cortexa_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- 🚀 Server starting up... ---")
    
    # 1. Load RAG Signal Engine
    try:
        cortexa_models["rag_engine"] = RAGSignalEngine.from_artifacts("config.yaml")
        print("✅ RAGSignalEngine loaded.")
    except Exception as e:
        print(f"❌ Failed to load RAGSignalEngine: {e}")

    # 2. Load Current Regime
    try:
        cortexa_models["current_regime"] = get_current_regime("config.yaml")
        print(f"✅ Current Regime loaded: {cortexa_models['current_regime']}")
    except Exception as e:
        print(f"❌ Failed to load Regime: {e}")
        
    print("--- Server ready. ---")
    yield
    cortexa_models.clear()
    print("--- 🔌 Server shutting down. ---")

app = FastAPI(title="Cortexa API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str

def detect_ticker_from_text(text):
    """Simple helper to find a ticker in the user's query."""
    text = text.lower()
    for key, ticker in TICKER_MAP.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text):
            return ticker
    return None

@app.get("/")
def read_root():
    return {"status": "Cortexa API is running"}

@app.post("/query")
def run_cortexa_pipeline(request: QueryRequest):
    """Qualitative RAG Endpoint (News & Context)"""
    print(f"API received RAG query: {request.text}")
    
    # 1. Detect Ticker
    detected_ticker = detect_ticker_from_text(request.text)
    if detected_ticker:
        print(f"Detected ticker in query: {detected_ticker}")
    
    try:
        # 2. Retrieval (Pass Ticker for Strict Filtering)
        context = query_vector_db(request.text, ticker=detected_ticker, config_path="config.yaml")
        
        if not context or not context.get('documents') or not context['documents'][0]:
             return {"answer": "I could not find any relevant data in my knowledge base.", "sources": []}

        # 3. Synthesis
        answer = get_llm_reasoning(request.text, context)
        
        # 4. Format Sources
        sources = []
        if 'metadatas' in context and context['metadatas']:
            for i, meta in enumerate(context['metadatas'][0]):
                sources.append({
                    "title": meta.get('title', 'N/A'),
                    "source": meta.get('source', 'N/A'),
                    "date": meta.get('published', 'N/A'),
                    "link": meta.get('link', '#')
                })

        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{ticker}")
def get_prediction_endpoint(ticker: str):
    """Quantitative ML Endpoint (Signals)"""
    print(f"API received prediction request for: {ticker}")
    
    engine = cortexa_models.get("rag_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="RAGSignalEngine is not loaded.")
        
    # 1. Get Features
    latest_row = get_latest_features(ticker)
    if latest_row is None:
        return {"signal": "No Data", "detail": f"No feature data found for {ticker}"}
        
    # 2. Get Regime
    current_regime = cortexa_models.get("current_regime", 0)
    latest_row['market_regime'] = current_regime
    
    try:
        # 3. Score
        signal_output = engine.score(latest_row)
        
        # --- LABEL LOGIC FIX ---
        # If decision is 1 -> BUY
        # If decision is 0 -> HOLD (Neutral), not Sell
        if signal_output.decision == 1:
            signal_text = "BUY (UP)"
        else:
            signal_text = "NEUTRAL / HOLD"

        return {
            "signal": signal_text,
            "ml_probability": f"{signal_output.proba_ml * 100:.2f}%",
            "rag_probability": f"{signal_output.proba_rag * 100:.2f}%",
            "combined_score": f"{signal_output.final_score * 100:.2f}%",
            "rag_contexts_found": len(signal_output.context['hits']),
            "filter_used": signal_output.context['filters']
        }
        
    except Exception as e:
        print(f"Error during signal scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Error scoring signal: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)