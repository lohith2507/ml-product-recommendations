"""Checkout handoff: deep links to Amazon (ASIN) or a generic store cart URL."""

from __future__ import annotations

import os
from urllib.parse import quote

from recommenders import cart as cart_service
from recommenders.store import product_snapshot


def _affiliate_tag() -> str:
    return os.environ.get("AMAZON_AFFILIATE_TAG", "shoppingagent-20")


def product_checkout_url(product: dict) -> str:
    asin = (product.get("asin") or "").strip()
    tag = _affiliate_tag()
    if asin:
        return f"https://www.amazon.com/dp/{asin}?tag={quote(tag)}"
    title = quote((product.get("title") or "product")[:80])
    return f"https://www.amazon.com/s?k={title}&tag={quote(tag)}"


def cart_checkout(user_id: int) -> dict:
    cart = cart_service.get_cart(user_id)
    lines = []
    for item in cart["items"]:
        try:
            snap = product_snapshot(item["product_id"])
        except LookupError:
            snap = item
        url = product_checkout_url(snap)
        lines.append(
            {
                "product_id": item["product_id"],
                "title": (item.get("title") or "")[:90],
                "quantity": item["quantity"],
                "price": item.get("price"),
                "asin": snap.get("asin"),
                "checkout_url": url,
            }
        )

    # Amazon multi-ASIN deep link when every line has an ASIN.
    asins = [l["asin"] for l in lines if l.get("asin")]
    multi = None
    if asins and len(asins) == len(lines):
        # Open first ASIN; list the rest as related links (Amazon cart APIs need Partner API).
        multi = lines[0]["checkout_url"] if lines else None

    return {
        "user_id": user_id,
        "item_count": cart["item_count"],
        "subtotal": cart["subtotal"],
        "savings": cart["savings"],
        "primary_url": multi or (lines[0]["checkout_url"] if lines else None),
        "items": lines,
        "note": "Opens Amazon product pages with your affiliate tag. Multi-item cart APIs require Amazon Partner credentials.",
    }
