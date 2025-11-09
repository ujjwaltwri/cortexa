# src/reasoning/agent.py
from __future__ import annotations
import os
import json
import requests
from typing import Dict, Any, List
import yaml

# -------------------- Config --------------------

def _read_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------- Context formatting --------------------

def _format_context_for_prompt(ctx: Dict[str, Any], max_docs: int = 6, max_chars: int = 4000) -> str:
    """
    Turn retrieved docs into a compact prompt chunk.
    Caps number of docs and total characters to avoid overly long prompts.
    """
    docs: List[str] = (ctx or {}).get("documents", [[]])[0][:max_docs]
    if not docs:
        return "No context documents."
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
    bullets = max(1, min(8, bullets))
    snippets = [str(x).strip() for x in docs[:bullets]]
    bullets_md = "\n".join(f"- {s}" for s in snippets if s)
    return f"**Context-based summary (no LLM):**\n{bullets_md}"

# -------------------- LLM backends --------------------

def _ollama_generate(base_url: str, model: str, prompt: str, temperature: float = 0.2, timeout: int = 120) -> str:
    """
    Call a local Ollama server. No streaming for simplicity.
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
        # Most Ollama builds return {'response': '...'}
        return data.get("response") or data.get("message") or str(data)
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

def _gemini_generate(api_key: str, model: str, prompt: str) -> str:
    """
    Call Google Gemini only if a valid API key is present.
    Import is inside the function to avoid hard dependency when key is empty.
    """
    try:
        import google.generativeai as genai  # lazy import
    except Exception as e:
        raise RuntimeError("google-generativeai not installed") from e

    try:
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model)
        out = gmodel.generate_content(prompt)
        if hasattr(out, "text") and out.text:
            return out.text
        # Fallback parsing
        return (
            getattr(out, "candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

# -------------------- Public entrypoint --------------------

def get_llm_reasoning(question: str, context: Dict[str, Any], config_path: str = "config.yaml") -> str:
    """"
    Returns an answer string. Tries:
      1)Gemini (if api key present & valid)
      2)Ollama (if configured)
      3)Free extractive fallback (no LLM)
    You can force fallback by setting env FORCE_FALLBACK=1.
    """
    #Force no-LLM mode if requested
    if os.getenv("FORCE_FALLBACK", "").strip() == "1":
        return _extractive_fallback(question, context)

    cfg = _read_cfg(config_path)
    llm_cfg = cfg.get("llm", {}) or {}
    google_cfg = cfg.get("google_api", {}) or {}

    prompt = (
        "You are a careful financial analyst. Use ONLY the provided context. "
        "If the context is insufficient, say so briefly.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{_format_context_for_prompt(context)}\n\n"
        "Answer concisely in 4-8 bullet points with dates and tickers where relevant."
    )

    api_key = (os.getenv("GOOGLE_API_KEY") or google_cfg.get("api_key") or "").strip()
    gemini_model = llm_cfg.get("gemini_model", "gemini-2.5-flash")
    if api_key:
        try:
            return _gemini_generate(api_key, gemini_model, prompt)
        except Exception as e:
            print(f"[Reasoning] Gemini call failed: {e}")


    base_url = llm_cfg.get("base_url", "http://localhost:11434/api")
    model_cfg = (llm_cfg.get("model") or "").strip()
    use_ollama = bool(model_cfg) and ("ollama" in model_cfg.lower() or llm_cfg.get("engine") == "ollama")
    if use_ollama:
        try:
            # Model name may be "ollama/llama3" → strip the prefix
            model_name = model_cfg.replace("ollama/", "")
            temperature = float(llm_cfg.get("temperature", 0.2))
            return _ollama_generate(base_url, model_name, prompt, temperature=temperature)
        except Exception as e:
            print(f"[Reasoning] Ollama call failed: {e}")

    return _extractive_fallback(question, context)
