"""
Live product facts (price + stock) updated by streaming events.

Hot path uses Firestore in production (Bigtable stand-in); SQLite for local/tests.
Deals/search re-hydrate products from facts when a row exists.
"""

from __future__ import annotations

from recommenders.store import get_facts_store


def upsert_price(product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
    get_facts_store().upsert_price(product_id, price, list_price, warehouse_id)


def upsert_stock(product_id: int, quantity: int, warehouse_id: str = "default"):
    get_facts_store().upsert_stock(product_id, quantity, warehouse_id)


def get_facts(product_ids: list[int], warehouse_id: str = "default") -> dict[int, dict]:
    return get_facts_store().get_facts([int(i) for i in product_ids], warehouse_id=warehouse_id)


def apply_live_facts(products: list[dict], warehouse_id: str = "default") -> list[dict]:
    """Overlay live price/stock onto product dicts."""
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
