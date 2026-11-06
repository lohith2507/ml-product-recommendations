"""
Live product facts (price + stock) updated by streaming events.

Hot path uses Firestore in production (Bigtable stand-in); SQLite for local/tests.
Deals/search re-hydrate products from facts when a row exists.
"""

from __future__ import annotations

from recommenders.store import get_facts_store


def upsert_price(product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
    """Write or refresh the live price (and optional list price) for a product."""
    get_facts_store().upsert_price(product_id, price, list_price, warehouse_id)


def upsert_stock(product_id: int, quantity: int, warehouse_id: str = "default"):
    """Write or refresh the live stock quantity for a product."""
    get_facts_store().upsert_stock(product_id, quantity, warehouse_id)


def get_facts(product_ids: list[int], warehouse_id: str = "default") -> dict[int, dict]:
    """Return a map of product_id -> {price, list_price, stock_qty, updated_at} for the given ids.

    Missing ids are omitted. Quantities and prices come from the active facts store
    (Firestore when enabled, otherwise the local SQLite stand-in).
    """
    return get_facts_store().get_facts([int(i) for i in product_ids], warehouse_id=warehouse_id)


def apply_live_facts(products: list[dict], warehouse_id: str = "default") -> list[dict]:
    """Overlay live price/stock onto product dicts; compute savings when list_price > price."""
    facts = get_facts([p["product_id"] for p in products], warehouse_id=warehouse_id)
    enriched = []
    for product in products:
        item = dict(product)
        fact = facts.get(item["product_id"])
        if fact:
            if fact.get("price") is not None:
                item["price"] = fact["price"]
            if fact.get("list_price") is not None:
                item["list_price"] = fact["list_price"]
            item["stock_qty"] = fact.get("stock_qty")
            item["facts_updated_at"] = fact.get("updated_at")
            list_price = item.get("list_price") or 0
            price = item.get("price") or 0
            savings = list_price - price if list_price > price > 0 else 0.0
            item["savings"] = round(savings, 2)
            item["savings_pct"] = round(savings / list_price, 4) if savings else 0.0
        enriched.append(item)
    return enriched
