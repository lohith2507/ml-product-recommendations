"""
Product retrieval and best-deal ranking used by the shopping agent.

Retrieval is semantic (FAISS / Vector Search) with a keyword fallback.
Prices and stock are re-hydrated from live product_facts (Pub/Sub), never
trusted from the vector index alone.
"""

import math

from database.models import get_db, dict_from_row
from recommenders.facts import apply_live_facts

RELEVANCE_WEIGHT = 0.45
SAVINGS_WEIGHT = 0.35
RATING_WEIGHT = 0.20
# A 50%+ discount earns the full savings component.
SAVINGS_CAP = 0.5
# Ratings backed by this many reviews get full confidence.
RATING_COUNT_FOR_FULL_CONFIDENCE = 500


def deal_metrics(product):
    price = product.get("price") or 0.0
    list_price = product.get("list_price") or 0.0
    savings = list_price - price if list_price > price > 0 else 0.0
    return {
        "savings": round(savings, 2),
        "savings_pct": round(savings / list_price, 4) if savings else 0.0,
    }


def deal_score(product, relevance):
    savings_pct = deal_metrics(product)["savings_pct"]
    rating = (product.get("avg_rating") or 0.0) / 5.0
    confidence = min(
        1.0,
        math.log1p(product.get("rating_count") or 0) / math.log1p(RATING_COUNT_FOR_FULL_CONFIDENCE),
    )
    return round(
        RELEVANCE_WEIGHT * relevance
        + SAVINGS_WEIGHT * min(savings_pct / SAVINGS_CAP, 1.0)
        + RATING_WEIGHT * rating * confidence,
        4,
    )


def _semantic_retriever(query, top_k):
    from recommenders.semantic import semantic_engine

    return semantic_engine.search(query, top_k=top_k)


def _keyword_candidates(conn, query, top_k):
    terms = [t for t in query.lower().split() if len(t) > 2][:6]
    if not terms:
        return []
    clause = " OR ".join(["LOWER(title) LIKE ?"] * len(terms))
    rows = conn.execute(
        f"SELECT product_id FROM products WHERE {clause} ORDER BY rating_count DESC LIMIT ?",
        [f"%{t}%" for t in terms] + [top_k],
    ).fetchall()
    return [{"product_id": r["product_id"], "similarity": 0.5} for r in rows]


def _candidates(query, category, max_price, min_rating, top_k, retriever):
    """Return product dicts matching the filters, each with relevance and deal fields."""
    hits = []
    if query:
        try:
            hits = (retriever or _semantic_retriever)(query, top_k)
        except Exception as e:
            print(f"⚠️ Semantic retrieval failed, using keyword search: {e}")

    with get_db() as conn:
        if query and not hits:
            hits = _keyword_candidates(conn, query, top_k)

        if hits:
            relevance = {h["product_id"]: max(0.0, min(1.0, float(h["similarity"]))) for h in hits}
            placeholders = ",".join("?" * len(relevance))
            rows = conn.execute(
                f"SELECT * FROM products WHERE product_id IN ({placeholders})",
                list(relevance),
            ).fetchall()
        else:
            relevance = {}
            rows = conn.execute(
                "SELECT * FROM products ORDER BY (list_price - price) / NULLIF(list_price, 0) DESC LIMIT ?",
                (top_k,),
            ).fetchall()

    products = []
    for row in rows:
        p = dict_from_row(row)
        if category and (p.get("category") or "").lower() != category.lower():
            continue
        if min_rating is not None and (p.get("avg_rating") or 0) < min_rating:
            continue
        p["relevance"] = round(relevance.get(p["product_id"], 0.0), 4)
        products.append(p)

    # Live price/stock from streaming facts (Pub/Sub → product_facts).
    products = apply_live_facts(products)
    filtered = []
    for p in products:
        if max_price is not None and (p.get("price") or 0) > max_price:
            continue
        stock = p.get("stock_qty")
        if stock is not None and stock <= 0:
            continue
        p.update(deal_metrics(p))
        p["deal_score"] = deal_score(p, p["relevance"])
        filtered.append(p)
    return filtered


def search_products(query, category=None, max_price=None, min_rating=None, limit=5, retriever=None):
    filtered = category or max_price is not None or min_rating is not None
    products = _candidates(query, category, max_price, min_rating, 100 if filtered else 40, retriever)
    products.sort(key=lambda p: p["relevance"], reverse=True)
    return products[:limit]


def find_best_deals(query, category=None, max_price=None, min_rating=None, limit=5, retriever=None):
    products = _candidates(query, category, max_price, min_rating, 100, retriever)
    products.sort(key=lambda p: p["deal_score"], reverse=True)
    return products[:limit]


def get_products(product_ids):
    ids = [int(i) for i in product_ids][:10]
    if not ids:
        return []
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM products WHERE product_id IN ({','.join('?' * len(ids))})", ids
        ).fetchall()
    by_id = {}
    for row in rows:
        p = dict_from_row(row)
        by_id[p["product_id"]] = p
    ordered = [by_id[i] for i in ids if i in by_id]
    ordered = apply_live_facts(ordered)
    for p in ordered:
        p.update(deal_metrics(p))
    return ordered


def compact(product):
    """Small, LLM-friendly view of a product."""
    title = product.get("title") or ""
    return {
        "product_id": product["product_id"],
        "title": title[:90],
        "category": product.get("category"),
        "price": product.get("price"),
        "list_price": product.get("list_price") or None,
        "savings": product.get("savings", 0.0),
        "savings_pct": round(product.get("savings_pct", 0.0) * 100, 1),
        "avg_rating": product.get("avg_rating"),
        "rating_count": product.get("rating_count"),
    }


def find_cheaper_match(product_id=None, query=None, limit=5, retriever=None):
    """Find cheaper substitutes for a product (or free-text query from an image caption)."""
    anchor = None
    if product_id is not None:
        products = get_products([int(product_id)])
        if products:
            anchor = products[0]
    search_q = (query or "").strip()
    if not search_q and anchor:
        search_q = " ".join((anchor.get("title") or "").split()[:8])
    if not search_q:
        return {"error": "Provide product_id or query", "products": []}

    max_price = None
    if anchor and anchor.get("price"):
        max_price = float(anchor["price"]) - 0.01

    category = (anchor or {}).get("category")
    candidates = find_best_deals(
        search_q,
        category=category,
        max_price=max_price,
        limit=max(limit * 3, 12),
        retriever=retriever,
    )
    if anchor:
        candidates = [c for c in candidates if c["product_id"] != anchor["product_id"]]
    # Prefer same category first.
    if category:
        same = [c for c in candidates if (c.get("category") or "").lower() == category.lower()]
        other = [c for c in candidates if c not in same]
        candidates = same + other
    out = candidates[:limit]
    receipt = None
    if anchor and out:
        save = round(float(anchor["price"]) - float(out[0]["price"]), 2)
        receipt = {
            "anchor_product_id": anchor["product_id"],
            "anchor_price": anchor.get("price"),
            "cheaper_product_id": out[0]["product_id"],
            "cheaper_price": out[0].get("price"),
            "you_save_vs_anchor": save,
            "why": (
                f"${out[0].get('price')} vs ${anchor.get('price')} for similar "
                f"{category or 'item'}; save ${save}."
            ),
        }
    return {
        "anchor": compact(anchor) if anchor else None,
        "query": search_q,
        "products": [compact(p) for p in out],
        "receipt": receipt,
    }


def explain_substitute(product_id: int, alternative_id: int) -> dict:
    """Receipt explaining why an alternative is a better/worse deal than the original."""
    products = get_products([int(product_id), int(alternative_id)])
    by_id = {p["product_id"]: p for p in products}
    a = by_id.get(int(product_id))
    b = by_id.get(int(alternative_id))
    if not a or not b:
        return {"error": "Unknown product id(s)"}
    price_delta = round(float(a.get("price") or 0) - float(b.get("price") or 0), 2)
    a_m = deal_metrics(a)
    b_m = deal_metrics(b)
    why_parts = []
    if price_delta > 0:
        why_parts.append(f"Alternative is ${price_delta} cheaper.")
    elif price_delta < 0:
        why_parts.append(f"Alternative costs ${-price_delta} more.")
    else:
        why_parts.append("Same price.")
    if b_m["savings"] > a_m["savings"]:
        why_parts.append(
            f"Bigger discount off list (${b_m['savings']} vs ${a_m['savings']})."
        )
    if (b.get("avg_rating") or 0) >= (a.get("avg_rating") or 0):
        why_parts.append(
            f"Rating {b.get('avg_rating')} ({b.get('rating_count')} reviews) "
            f"vs {a.get('avg_rating')} ({a.get('rating_count')})."
        )
    else:
        why_parts.append(
            f"Original has a higher rating ({a.get('avg_rating')} vs {b.get('avg_rating')})."
        )
    return {
        "original": compact(a),
        "alternative": compact(b),
        "price_delta": price_delta,
        "why": " ".join(why_parts),
        "receipt": {
            "original_price": a.get("price"),
            "alternative_price": b.get("price"),
            "original_list": a.get("list_price"),
            "alternative_list": b.get("list_price"),
            "original_savings": a_m["savings"],
            "alternative_savings": b_m["savings"],
        },
    }
