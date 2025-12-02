# src/reasoning/agent.py
from __future__ import annotations
import os
import json
import requests
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
    """
    Turn retrieved docs into a compact prompt chunk.
    """
    docs: List[str] = (ctx or {}).get("documents", [[]])[0]
    # Handle case where docs might be empty or None
    if not docs:
        return "No context documents."
    
    # Slice to max_docs
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
    """
    Free fallback: surface top snippets as a quick 'summary'.
    """
    docs: List[str] = (ctx or {}).get("documents", [[]])[0]
    if not docs:
        return "I don't have context to answer that."
    
    # Clean up snippets
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
    """
    Call a local Ollama server.
    """
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
    """
    Call Google Gemini.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("google-generativeai library not installed. Run `pip install google-generativeai`") from e

    try:
        genai.configure(api_key=api_key)
        # Use a safe default if model is weird
        safe_model = model if model else "gemini-2.5-flash"
        gmodel = genai.GenerativeModel(safe_model)
        
        out = gmodel.generate_content(prompt)
        
        if hasattr(out, "text") and out.text:
            return out.text
        
        # Fallback parsing for safety
        if hasattr(out, "candidates") and out.candidates:
            return out.candidates[0].content.parts[0].text
            
        return "Gemini returned no text."
        
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

# -------------------- Public entrypoint --------------------

def get_llm_reasoning(question: str, context: Dict[str, Any], config_path: str = "config.yaml") -> str:
    """
    Returns an answer string. Tries:
      1) Gemini (if API key present & valid)
      2) Ollama (if configured)
      3) Free extractive fallback (no LLM)
    Uses a hyper-restrictive prompt to avoid synthesizing noise.
    """
    #Force no-LLM mode if requested
    if os.getenv("FORCE_FALLBACK", "").strip() == "1":
        return _extractive_fallback(question, context)

    cfg = _read_cfg(config_path)
    llm_cfg = cfg.get("llm", {}) or {}
    google_cfg = cfg.get("google_api", {}) or {}

    # --------------------------------------------------------------------------
    # CRITICAL FIX: Ensure Gemini self-censors irrelevant data
    # We check the question for the ticker and inject it into the prompt.
    # We assume the query is about the first ticker detected.
    
    ticker_hint = "GOOGL" # Default value to avoid breaking the prompt
    try:
        from src.server import detect_ticker_from_text # Lazy import from the server helper
        ticker_hint = detect_ticker_from_text(question) or "MARKET"
    except:
        pass
    
    # Check if the primary ticker is present in the context
    context_text = _format_context_for_prompt(context, max_docs=8, max_chars=8000)
    
    # This is the fail condition: If the specific ticker isn't even in the text, reject it.
    if ticker_hint and ticker_hint != "MARKET" and ticker_hint.upper() not in context_text.upper():
        # This is the cleanest failure: The retrieval failed to find the specific ticker.
        return f"The provided context does not contain any specific information regarding the outlook on {ticker_hint}. Please ensure recent news is available for this ticker."
    # --------------------------------------------------------------------------

    prompt = (
        "You are a careful financial analyst. Your task is to provide an outlook on the specific ticker requested. "
        "You must use ONLY the provided context. Do NOT use outside knowledge. "
        "If you cannot find specific evidence about the stock's outlook in the context, you MUST state that the context is insufficient.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer concisely in 4-8 bullet points with dates and tickers where relevant. If context is insufficient, state only that."
    )

    api_key = (os.getenv("GOOGLE_API_KEY") or google_cfg.get("api_key") or "").strip()
    
    # 1. Try Gemini
    if api_key and "PASTE_YOUR" not in api_key:
        gemini_model = llm_cfg.get("gemini_model", "gemini-2.5-flash")
        try:
            return _gemini_generate(api_key, gemini_model, prompt)
        except Exception as e:
            print(f"[Reasoning] Gemini call failed: {e}")

    # 2. Try Ollama (Simplified)
    base_url = llm_cfg.get("base_url", "http://localhost:11434/api")
    model_cfg = (llm_cfg.get("model") or "").strip()
    use_ollama = bool(model_cfg) and ("ollama" in model_cfg.lower() or llm_cfg.get("engine") == "ollama")
    
    if use_ollama:
        try:
            model_name = model_cfg.replace("ollama/", "")
            return _ollama_generate(base_url, model_name, prompt)
        except Exception as e:
            print(f"[Reasoning] Ollama call failed: {e}")

    # 3. Fallback (Extractive summary of what was found, which is usually noise)
    return _extractive_fallback(question, context)