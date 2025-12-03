from __future__ import annotations
import os
import json
import requests
import re
from typing import Dict, Any, List
import yaml

# -------------------- Config --------------------

def _read_cfg(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

# -------------------- Context formatting --------------------

def _format_context_for_prompt(ctx: Dict[str, Any], max_docs: int = 6, max_chars: int = 4000) -> str:
    docs: List[str] = (ctx or {}).get("documents", [[]])[0]
    if not docs:
        return "No context documents."
    
    docs = docs[:max_docs]
    lines: List[str] = []
    size = 0
    for i, d in enumerate(docs, 1):
        if not isinstance(d, str):
            d = json.dumps(d, ensure_ascii=False)
        line = f"[{i}] {d}".strip()
        if size + len(line) > max_chars:
            break
        lines.append(line)
        size += len(line)
    return "\n".join(lines) if lines else "No context documents."

def _extractive_fallback(question: str, ctx: Dict[str, Any], bullets: int = 4) -> str:
    docs: List[str] = (ctx or {}).get("documents", [[]])[0]
    if not docs:
        return "I don't have context to answer that."
    
    clean_docs = []
    for d in docs:
        if d and isinstance(d, str):
            clean_docs.append(d.replace("\n", " ").strip())
            
    bullets = max(1, min(8, bullets))
    snippets = clean_docs[:bullets]
    
    if not snippets:
        return "No relevant context found."
        
    bullets_md = "\n".join(f"- {s[:200]}..." for s in snippets)
    return f"**Context-based summary (no LLM):**\n{bullets_md}"

# -------------------- LLM backends --------------------

def _ollama_generate(base_url: str, model: str, prompt: str, temperature: float = 0.2, timeout: int = 120) -> str:
    url = f"{base_url.rstrip('/')}/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("message") or str(data)
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

def _gemini_generate(api_key: str, model: str, prompt: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("google-generativeai library not installed. Run `pip install google-generativeai`") from e

    try:
        genai.configure(api_key=api_key)
        safe_model = model if model else "gemini-2.5-flash"
        gmodel = genai.GenerativeModel(safe_model)
        
        out = gmodel.generate_content(prompt)
        
        if hasattr(out, "text") and out.text:
            return out.text
        
        if hasattr(out, "candidates") and out.candidates:
            return out.candidates[0].content.parts[0].text
            
        return "Gemini returned no text."
        
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

# -------------------- Helpers --------------------

def _detect_ticker_locally(text: str) -> str | None:
    """
    Local detection to avoid circular import with server.py
    """
    # Basic mapping for common queries
    TICKER_MAP = {
        "apple": "AAPL", "aapl": "AAPL", "appl": "AAPL",
        "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL",
        "microsoft": "MSFT", "msft": "MSFT",
        "nvidia": "NVDA", "nvda": "NVDA",
        "tesla": "TSLA", "tsla": "TSLA",
        "gspc": "GSPC", "market": "GSPC"
    }
    text = text.lower()
    for key, ticker in TICKER_MAP.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text):
            return ticker
    return None

# -------------------- Public entrypoint --------------------

def get_llm_reasoning(question: str, context: Dict[str, Any], config_path: str = "config.yaml") -> str:
    
    docs: List[str] = (context or {}).get("documents", [[]])[0]
    if not docs:
        return "The provided context is empty. Please ensure recent news is available."

    cfg = _read_cfg(config_path)
    llm_cfg = cfg.get("llm", {}) or {}
    google_cfg = cfg.get("google_api", {}) or {}

    # --- FIX: Local detection, no default "GOOGL" ---
    ticker_hint = _detect_ticker_locally(question)
    
    context_text = _format_context_for_prompt(context, max_docs=8, max_chars=8000)
    
    # Fail-safe: If we know the ticker, but it's NOT in the context, warn the user.
    if ticker_hint and ticker_hint.upper() not in context_text.upper():
        return f"The provided context does not contain specific information regarding {ticker_hint}. Please try fetching fresh news."

    prompt = (
        "You are a careful financial analyst. Your task is to provide an outlook on the specific ticker requested. "
        "You must use ONLY the provided context. Do NOT use outside knowledge. "
        "Synthesize a balanced summary of bullish and bearish points relevant to the question. "
        "If you cannot find specific evidence about the stock's outlook in the context, you MUST state that the context is insufficient.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer concisely in 4-8 bullet points with dates and tickers where relevant."
    )

    # 1. Try Gemini
    api_key = (os.getenv("GOOGLE_API_KEY") or google_cfg.get("api_key") or "").strip()
    
    if api_key and "PASTE_YOUR" not in api_key:
        gemini_model = llm_cfg.get("gemini_model", "gemini-2.5-flash")
        try:
            return _gemini_generate(api_key, gemini_model, prompt)
        except Exception as e:
            print(f"[Reasoning] Gemini call failed: {e}")

    # 2. Try Ollama
    base_url = llm_cfg.get("base_url", "http://localhost:11434/api")
    model_cfg = (llm_cfg.get("model") or "").strip()
    
    use_ollama = bool(model_cfg) and ("ollama" in model_cfg.lower() or llm_cfg.get("engine") == "ollama")
    
    if use_ollama:
        try:
            model_name = model_cfg.replace("ollama/", "")
            return _ollama_generate(base_url, model_name, prompt)
        except Exception as e:
            print(f"[Reasoning] Ollama call failed: {e}")

    return _extractive_fallback(question, context)