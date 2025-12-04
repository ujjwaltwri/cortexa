from __future__ import annotations
import os
import json
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

def _format_context_for_prompt(ctx: Dict[str, Any], max_docs: int = 8, max_chars: int = 6000) -> str:
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
    """
    Very simple non-LLM fallback that just surfaces a few snippets.
    We also make sure it NEVER says 'insufficient context'.
    """
    docs: List[str] = (ctx or {}).get("documents", [[]])[0]
    if not docs:
        return (
            "Based on the available data, I can't see any relevant recent headlines. "
            "This usually means the knowledge base has not ingested fresh articles yet."
        )
    
    clean_docs = []
    for d in docs:
        if d and isinstance(d, str):
            clean_docs.append(d.replace("\n", " ").strip())
            
    bullets = max(1, min(8, bullets))
    snippets = clean_docs[:bullets]
    
    if not snippets:
        return (
            "The indexed context does not contain clear headlines to summarize. "
            "You may want to refresh or rebuild the news database."
        )
        
    bullets_md = "\n".join(f"- {s[:200]}..." for s in snippets)
    return f"**Context-based summary (no LLM):**\n{bullets_md}"

# -------------------- LLM backend: Gemini only --------------------

def _gemini_generate(api_key: str, model: str, prompt: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("google-generativeai library is not installed.") from e

    try:
        genai.configure(api_key=api_key)
        safe_model = model if model else "gemini-2.5-flash"
        gmodel = genai.GenerativeModel(safe_model)
        out = gmodel.generate_content(prompt)
        
        if hasattr(out, "text") and out.text:
            return out.text
        if hasattr(out, "candidates") and out.candidates:
            # defensive: try to extract first candidate text
            cand = out.candidates[0]
            if hasattr(cand, "content") and cand.content and cand.content.parts:
                return cand.content.parts[0].text
        return "The model returned no text, but you can still rely on the raw headlines for sentiment."
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

# -------------------- Public entrypoint --------------------

def get_llm_reasoning(question: str, context: Dict[str, Any], config_path: str = "config.yaml") -> str:
    """
    Summarize the outlook for a ticker using ONLY the provided news context.

    Changes vs old version:
    - Uses ONLY Gemini (gemini-2.5-flash by default).
    - Explicitly forbids phrases like 'insufficient context', 'context is insufficient',
      'insufficient information', 'lack of data', etc.
    - Always returns a best-effort outlook; missing details can be mentioned as a *final* bullet,
      not as 'context insufficient'.
    """
    docs: List[str] = (context or {}).get("documents", [[]])[0]
    if not docs:
        # Even here, avoid 'insufficient context'
        return (
            "I don't see any indexed headlines or summaries for this ticker in the current "
            "knowledge base snapshot. You may need to refresh the news ingestion pipeline."
        )

    cfg = _read_cfg(config_path)
    llm_cfg = cfg.get("llm", {}) or {}
    google_cfg = cfg.get("google_api", {}) or {}

    context_text = _format_context_for_prompt(context, max_docs=10, max_chars=8000)

    # --- STRICT PROMPT TO AVOID 'INSUFFICIENT CONTEXT' LANGUAGE ---

    prompt = (
        "You are a concise, neutral financial analyst.\n"
        "Your job is to summarize the STOCK OUTLOOK for the requested ticker "
        "using ONLY the news headlines and snippets provided in the context.\n\n"
        "VERY IMPORTANT RULES (DO NOT VIOLATE):\n"
        "1. You MUST NOT say 'insufficient context', 'insufficient information', "
        "'context is insufficient', 'lack of data', or similar phrases.\n"
        "2. Even if specific details like price targets, exact analyst ratings, or detailed risks "
        "are missing, you MUST still provide a best-effort outlook based on what IS present.\n"
        "3. Each bullet MUST start with one of these labels exactly: 'Bullish:', 'Bearish:', or 'Neutral:'.\n"
        "   - Use 'Bullish:' when the point clearly supports upside, strength, growth, or positive sentiment.\n"
        "   - Use 'Bearish:' when the point clearly supports downside, risk, pressure, or negative sentiment.\n"
        "   - Use 'Neutral:' when the point is mixed, descriptive, or not clearly positive/negative.\n"
        "4. Focus on:\n"
        "   - Bullish factors (growth drivers, product launches, strong earnings, AI/cloud momentum, etc.).\n"
        "   - Bearish factors (regulation, competitive pressure, layoffs, margin compression, etc.).\n"
        "   - Any mention of analyst sentiment (upgrades, downgrades, 'top pick', etc.).\n"
        "5. Always produce 4–6 bullet points. Keep each bullet short and fact-based.\n"
        "6. At the end, add a final line in this exact format:\n"
        "   'Overall Outlook: <Bullish / Bearish / Neutral>'\n"
        "   Choose ONE word (Bullish, Bearish, or Neutral) based on the balance of the bullets.\n"
        "7. Do NOT hallucinate company fundamentals beyond what can be reasonably inferred "
        "from the headlines. When in doubt, stay general but useful.\n\n"
        f"User Question:\n{question}\n\n"
        f"News Context (headlines and summaries):\n{context_text}\n\n"
        "Now respond with 4–6 labeled bullet points and a final 'Overall Outlook:' line, using the rules above."
    )


    # ---- GEMINI ONLY ----
    api_key = (os.getenv("GOOGLE_API_KEY") or google_cfg.get("api_key") or "").strip()
    if api_key and "PASTE_YOUR" not in api_key:
        # default to gemini-2.5-flash unless overridden in config
        gemini_model = llm_cfg.get("gemini_model", "gemini-2.5-flash")
        try:
            return _gemini_generate(api_key, gemini_model, prompt)
        except Exception as e:
            print(f"[Reasoning] Gemini call failed: {e}")

    # If no API key or Gemini fails, fall back to simple extractive summary
    return _extractive_fallback(question, context)
