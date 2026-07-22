import platform
import re
import psutil
import ollama
import requests
from flask import Blueprint, request, jsonify

cookbook_bp = Blueprint("cookbook", __name__)

OLLAMA_LIBRARY_URL = "https://ollama.com/library"
OLLAMA_SEARCH_URL  = "https://ollama.com/search?q=&o=popular"

# ── Fetch models from Ollama library ─────────────────────────────────────────

def fetch_ollama_models():
    """
    Scrape the Ollama library page for available models.
    Returns a list of dicts: {name, desc, pulls, tags}
    Falls back to a small hardcoded list if the request fails.
    """
    try:
        resp = requests.get(OLLAMA_SEARCH_URL, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text

        # extract model names + descriptions from the HTML
        # Ollama search page uses <h2> for name and <p> for description
        names = re.findall(r'href="/library/([a-zA-Z0-9_\-\.]+)"', html)
        descs = re.findall(r'<p[^>]*class="[^"]*break-words[^"]*"[^>]*>(.*?)</p>', html, re.DOTALL)

        # clean html tags from descriptions
        clean = lambda s: re.sub(r"<[^>]+>", "", s).strip()
        descs = [clean(d) for d in descs]

        # deduplicate names while preserving order
        seen  = set()
        models = []
        for i, name in enumerate(names):
            if name in seen:
                continue
            seen.add(name)
            models.append({
                "name": name,
                "desc": descs[i] if i < len(descs) else "",
            })
        return models if models else _fallback_models()
    except Exception:
        return _fallback_models()


# def _fallback_models():
#     """Minimal hardcoded fallback if Ollama website is unreachable."""
#     return [
#         {"name": "qwen2.5-coder:7b",    "desc": "Code-focused model, great for data analysis"},
#         {"name": "llama3.1:8b",          "desc": "Strong reasoning and instruction following"},
#         {"name": "mistral:7b",           "desc": "Fast, efficient general purpose model"},
#         {"name": "phi3:mini",            "desc": "Lightweight model for simple tasks"},
#         {"name": "llama3.2:3b",          "desc": "Compact and fast, good for simple queries"},
#         {"name": "deepseek-coder:6.7b",  "desc": "Excellent code generation"},
#         {"name": "gemma2:9b",            "desc": "Google model, strong reasoning"},
#         {"name": "llama3.1:13b",         "desc": "Large model for complex analysis"},
#         {"name": "codellama:7b",         "desc": "Meta's code-focused model"},
#         {"name": "dolphin-mistral:7b",   "desc": "Fine-tuned Mistral for instruction following"},
#     ]

def ensure_tag(name):
    """Add :latest tag if no tag present."""
    return name if ":" in name else name + ":latest"


def get_pulled_models():
    """Return list of model names already pulled in Ollama."""
    try:
        return [m["name"] for m in ollama.list()["models"]]
    except Exception:
        return []


# ── Hardware detection ────────────────────────────────────────────────────────

def get_hardware():
    ram_gb  = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    cpu     = platform.processor() or platform.machine() or "Unknown CPU"
    # try to detect GPU VRAM (rough, works on most systems)
    vram_gb = 0
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        )
        vram_gb = round(int(out.strip().split("\n")[0]) / 1024, 1)
    except Exception:
        pass
    return {"ram_gb": ram_gb, "cpu": cpu, "vram_gb": vram_gb}


# ── Data complexity scorer ────────────────────────────────────────────────────

def score_data_complexity(df):
    import pandas as pd
    score   = 1
    reasons = []

    if len(df) > 10000:
        score += 1; reasons.append(f"large dataset ({len(df):,} rows)")
    if len(df.columns) > 10:
        score += 1; reasons.append(f"many columns ({len(df.columns)})")

    has_datetime = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
    has_text     = any(df[c].dtype == object for c in df.columns)
    num_cols     = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)

    if has_datetime: score += 1;   reasons.append("time series data")
    if has_text:     score += 0.5; reasons.append("text/categorical columns")
    if num_cols > 5: score += 0.5; reasons.append(f"{num_cols} numeric columns")

    score = min(round(score), 3)
    label = {1: "Simple", 2: "Medium", 3: "Complex"}[score]
    return score, label, reasons


# ── Model scorer ──────────────────────────────────────────────────────────────

def score_models(all_models, pulled, hw, complexity):
    """
    Score each model based on:
    - Whether it fits in RAM (param size heuristic from model name)
    - How well it matches data complexity
    - Whether it's already pulled (bonus)
    """
    ram_gb = hw["ram_gb"]

    # param size heuristic from model name  e.g. "7b" → 7
    def param_size(name):
        m = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
        return float(m.group(1)) if m else 7.0   # default guess 7B

    # rough RAM needed: ~2x params in GB for Q4 quantised
    def ram_needed(name):
        return param_size(name) * 2

    results = []
    for model in all_models:
        name   = model["name"]
        needed = ram_needed(name)
        if needed > ram_gb:
            continue   # won't fit

        score = 0

        # RAM headroom — more headroom = comfortable
        headroom = ram_gb - needed
        score += min(headroom * 4, 25)

        # complexity matching
        params = param_size(name)
        is_coder = any(k in name.lower() for k in ("coder", "code", "deepseek", "starcoder", "qwen"))

        if complexity >= 3:
            if is_coder:          score += 40
            if 6 <= params <= 9:  score += 20
        elif complexity == 2:
            if is_coder:          score += 20
            if 5 <= params <= 9:  score += 15
        else:
            if params <= 4:       score += 30   # prefer small/fast for simple data

        # already pulled → small bonus (convenience)
        if name in pulled or any(name in p for p in pulled):
            score += 10
            model["pulled"] = True
        else:
            model["pulled"] = False

        model["name"] = ensure_tag(model["name"])
        model["fit_score"] = round(score, 1)
        model["params"]    = f"{param_size(name):.1f}B"
        model["ram_need"]  = f"~{needed:.0f} GB"
        results.append(model)

    results.sort(key=lambda x: x["fit_score"], reverse=True)
    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@cookbook_bp.route("/cookbook", methods=["POST"])
def cookbook(sessions=None):
    # sessions is injected via the route wrapper in app.py
    from flask import current_app
    sessions = current_app.config.get("SESSIONS", {})

    body = request.json or {}
    sid  = body.get("session_id")

    hw      = get_hardware()
    pulled  = get_pulled_models()
    models  = fetch_ollama_models()

    if sid and sid in sessions:
        df = sessions[sid]["df"]
        complexity, complexity_label, reasons = score_data_complexity(df)
        has_data = True
    else:
        complexity       = 2
        complexity_label = None
        reasons          = []
        has_data         = False

    scored = score_models(models, pulled, hw, complexity)

    return jsonify({
        "hardware":   hw,
        "complexity": complexity_label,
        "reasons":    reasons,
        "has_data":   has_data,
        "models":     scored[:8],          # top 8
        "total_available": len(scored),
    })


@cookbook_bp.route("/cookbook/models", methods=["GET"])
def list_models():
    """Return the full Ollama library list + which ones are pulled."""
    pulled = get_pulled_models()
    models = fetch_ollama_models()
    for m in models:
        m["pulled"] = m["name"] in pulled or any(m["name"] in p for p in pulled)
    return jsonify({"models": models, "pulled": pulled})
