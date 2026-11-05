"""Match an uploaded product photo to catalog items of the same kind."""

from __future__ import annotations

import re

from database.models import get_db
from recommenders import deals

_STEMS = ("watch", "shoe", "boot", "bag", "backpack", "pant", "shirt", "dress", "lamp", "nail", "sandal", "slipper")
_STOP = {
    "the", "and", "for", "with", "this", "that", "photo", "picture", "image",
    "product", "item", "style", "color", "mens", "men", "womens", "women",
    "man", "woman", "new", "set", "pack",
}


def caption_product_image(image_bytes: bytes, mime: str | None) -> str:
    """Ask Gemini what the product is. Returns a short search phrase or ''."""
    if not image_bytes:
        return ""
    try:
        from google.genai import types
        from recommenders.vertex_client import _genai_client, chat_location, chat_model, fallback_model

        client = _genai_client(chat_location())
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime or "image/jpeg")
        prompt = (
            "Name only the product type in this photo as a 1 to 3 word shopping search. "
            "Examples: wristwatch, backpack, running shoe, nail kit. No sentence."
        )
        for model in (chat_model(), fallback_model()):
            try:
                response = client.models.generate_content(model=model, contents=[part, prompt])
                text = (getattr(response, "text", None) or "").strip().strip('"').strip(".")
                if text:
                    return text.split("\n")[0][:80]
            except Exception as e:
                print(f"⚠️ photo caption {model} failed: {e}")
    except Exception as e:
        print(f"⚠️ photo caption skipped: {e}")
    return ""


def search_tokens(caption: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (caption or "").lower())
    tokens = []
    for word in words:
        if len(word) < 3 or word in _STOP:
            continue
        tokens.append(word)
        for stem in _STEMS:
            if stem in word and stem != word:
                tokens.append(stem)
        if word.endswith("es") and len(word) > 4:
            tokens.append(word[:-2])
        elif word.endswith("s") and len(word) > 4:
            tokens.append(word[:-1])
    # Keep order, drop dupes.
    seen = set()
    out = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out[:6]


def _blob(product: dict) -> str:
    return f"{product.get('title') or ''} {product.get('category') or ''}".lower()


def is_same_kind(product: dict, tokens: list[str]) -> bool:
    blob = _blob(product)
    return any(token in blob for token in tokens)


def _keyword_products(tokens: list[str], limit: int = 8) -> list[dict]:
    if not tokens:
        return []
    clauses = []
    params = []
    for token in tokens[:4]:
        clauses.append("(LOWER(title) LIKE ? OR LOWER(category) LIKE ?)")
        params.extend([f"%{token}%", f"%{token}%"])
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT product_id FROM products WHERE {' OR '.join(clauses)} ORDER BY rating_count DESC LIMIT ?",
            params,
        ).fetchall()
    return deals.get_products([row["product_id"] for row in rows])


def products_for_caption(caption: str, limit: int = 5, retriever=None) -> list[dict]:
    tokens = search_tokens(caption)
    if not tokens:
        return []
    found = deals.search_products(caption, limit=12, retriever=retriever)
    if len(found) < limit:
        found = found + deals.find_best_deals(caption, limit=12, retriever=retriever)
    same = []
    seen = set()
    for product in found + _keyword_products(tokens, limit=12):
        pid = product.get("product_id")
        if pid in seen or not is_same_kind(product, tokens):
            continue
        seen.add(pid)
        same.append(product)
        if len(same) >= limit:
            break
    return same
