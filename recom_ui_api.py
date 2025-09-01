# recom_ui_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from wordllama import WordLlama
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# ── Load environment ─────────────────────────────────────────────
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "westside_mens").strip()
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "").strip()
EMBED_DIM = int(os.getenv("EMBED_DIM", "64"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
DIVKIT_JSON_PATH = os.getenv("DIVKIT_JSON_PATH", "./divkit_card.json").strip()

# ── Validate env ────────────────────────────────────────────────
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Missing Qdrant config. Set QDRANT_URL and QDRANT_API_KEY in .env.")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing Gemini key. Set GEMINI_API_KEY in .env.")

# ── Init clients ────────────────────────────────────────────────
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
wl = WordLlama.load(trunc_dim=EMBED_DIM)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel(GEMINI_MODEL)

# ── Load divkit JSON ────────────────────────────────────────────
""" 
def _load_divkit_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8when") as f:
            return json.load(f)
    except Exception:
        return {
            "card": {
                "log_id": "product_grid",
                "states": [{"state_id": 0, "div": {"type": "container", "items": []}}],
                "variables": [{"type": "array", "name": "products", "value": []}],
            }
        }
"""        
def _load_divkit_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

DIVKIT_JSON = _load_divkit_json(DIVKIT_JSON_PATH)

# ── Helpers (same as before) ───────────────────────────────────
def _embed(text: str) -> List[float]:
    vec = wl.embed([text])[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)

def _build_filter(product_type: Optional[str], price_min: Optional[float], price_max: Optional[float]):
    conditions: List[Any] = []
    if product_type:
        conditions.append(FieldCondition(key="product_type", match=MatchValue(value=product_type)))
    if (price_min is not None) or (price_max is not None):
        rng: Dict[str, float] = {}
        if price_min is not None: rng["gte"] = float(price_min)
        if price_max is not None: rng["lte"] = float(price_max)
        conditions.append(FieldCondition(key="price", range=Range(**rng)))
    return Filter(must=conditions) if conditions else None

_ALLOWED_KEYS = ["id", "description", "product_type", "price", "size", "image_url", "product_url"]

def _coerce_price(v) -> Optional[float]:
    try:
        s = str(v).strip()
        return float(s) if s != "" and s.lower() != "none" else None
    except Exception:
        return None

def _sanitise_product(hit: Any) -> Dict[str, Any]:
    p = (getattr(hit, "payload", None) or {})
    img = p.get("image_url") or p.get("image") or p.get("featured_image") or None
    url = p.get("product_url") or p.get("url") or None
    out = {
        "id": str(getattr(hit, "id", "")) if getattr(hit, "id", "") is not None else "",
        "description": (p.get("description") or "").strip(),
        "product_type": p.get("product_type") or None,
        "price": _coerce_price(p.get("price")),
        "size": p.get("size") or None,
        "image_url": img,
        "product_url": url,
    }
    return {k: out.get(k, None) for k in _ALLOWED_KEYS}

def _validate_products(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = []
    for r in rows:
        if (r.get("description")) and (r.get("image_url")) and (r.get("product_url")):
            valid.append(r)
    return valid

def _llm_reply(user_query: str, products: List[Dict[str, Any]]) -> str:
    bullets = []
    for r in products[:5]:
        name = r.get("product_type") or "Item"
        price = r.get("price")
        desc = (r.get("description") or "")[:120]
        purl = r.get("product_url") or ""
        price_txt = f"₹{price}" if price is not None else ""
        bullets.append(f"- {name}: {price_txt} — {desc}... {purl}")

    prompt = (
        "You are a helpful fashion shopping assistant. "
        "Reply in clear Indian English, one short paragraph, practical tone. "
        "Reference 2–3 items by type/price. Avoid flowery language.\n\n"
        f"User request: {user_query}\n\n"
        "Top matches:\n" + "\n".join(bullets)
    )
    try:
        resp = gemini.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(LLM reply unavailable: {type(e).__name__}: {e})"

# ── Core function ───────────────────────────────────────────────
def recommend_products(
    query: str,
    top_k: int = 8,
    product_type: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    strict_only: bool = False,
) -> Dict[str, Any]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    qvec = _embed(query.strip())
    qfilter = _build_filter(product_type, price_min, price_max)

    kwargs: Dict[str, Any] = dict(
        collection_name=QDRANT_COLLECTION,
        limit=max(1, min(int(top_k), 50)),
        query_filter=qfilter,
        with_payload=True,
        with_vectors=False,
    )
    if QDRANT_VECTOR_NAME:
        kwargs["query_vector"] = {QDRANT_VECTOR_NAME: qvec}
    else:
        kwargs["query_vector"] = qvec

    hits = qdrant.search(**kwargs)

    rows = [_sanitise_product(h) for h in hits]
    if strict_only:
        rows = _validate_products(rows)

    message = _llm_reply(query, rows)

    return {
        "products": rows,
        "divkitJSON": DIVKIT_JSON,
        "message": message,
    }

app = FastAPI(title="Fashion RAG Chatbot")

# ── Add CORS middleware ──────────────────────────────
origins = [
    "http://localhost:3000",  # allow your frontend
    "https://fashion-assistant-virid.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow GET, POST, etc.
    allow_headers=["*"],  # allow headers like Content-Type
)

class RecommendQuerySimple(BaseModel):
    query: str

@app.post("/recommend", response_model=Dict)
def recommend_endpoint(
    req: RecommendQuerySimple
):
    result = recommend_products(query=req.query)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recom_ui_api:app", host="127.0.0.1", port=3000, reload=True)
