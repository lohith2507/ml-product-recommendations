"""Allowlisted website actions the shopping agent can run for the member."""

from __future__ import annotations

SECTIONS = {"recommendations", "products", "analytics", "abtest", "admin", "incidents"}
ALGOS = {"collaborative", "content", "hybrid"}
UI_TYPES = {
    "navigate",
    "search",
    "filter_category",
    "switch_algo",
    "open_cart",
    "close_cart",
    "checkout",
    "show_product",
    "summarize_reviews",
    "refresh_cart",
}
SERVER_TYPES = {"add_to_cart", "remove_from_cart", "set_cart_qty"}


def _clean_str(value, limit=120) -> str:
    return str(value or "").strip()[:limit]


def normalize_actions(raw) -> list[dict]:
    """Turn model arguments into a short list of safe actions."""
    if isinstance(raw, dict):
        if isinstance(raw.get("actions"), list):
            items = raw["actions"]
        elif raw.get("type") or raw.get("action"):
            items = [raw]
        else:
            items = []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    out = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        kind = _clean_str(item.get("type") or item.get("action"), 40).lower()
        action = {"type": kind}
        if kind == "navigate":
            section = _clean_str(item.get("section"), 40).lower()
            if section not in SECTIONS:
                continue
            action["section"] = section
        elif kind == "search":
            action["query"] = _clean_str(item.get("query"), 160)
            category = _clean_str(item.get("category"), 80)
            if category and category.lower() != "all":
                action["category"] = category
            if not action["query"] and "category" not in action:
                continue
        elif kind == "filter_category":
            category = _clean_str(item.get("category") or "all", 80)
            action["category"] = category or "all"
        elif kind == "switch_algo":
            algo = _clean_str(item.get("algo") or item.get("algorithm"), 40).lower()
            if algo not in ALGOS:
                continue
            action["algo"] = algo
        elif kind in ("show_product", "summarize_reviews", "add_to_cart", "remove_from_cart", "set_cart_qty"):
            try:
                action["product_id"] = int(item.get("product_id"))
            except (TypeError, ValueError):
                continue
            if kind == "set_cart_qty":
                try:
                    qty = int(item.get("quantity"))
                except (TypeError, ValueError):
                    continue
                action["quantity"] = max(0, min(qty, 99))
            elif kind == "add_to_cart":
                try:
                    qty = int(item.get("quantity") or 1)
                except (TypeError, ValueError):
                    qty = 1
                action["quantity"] = max(1, min(qty, 99))
        elif kind in ("open_cart", "close_cart", "checkout", "refresh_cart"):
            pass
        else:
            continue
        out.append(action)
    return out


def apply_server_actions(user_id: int, actions: list[dict]) -> list[str]:
    """Apply cart mutations. Navigation stays on the client."""
    from recommenders import cart as cart_service

    notes = []
    for action in actions:
        kind = action["type"]
        try:
            if kind == "add_to_cart":
                cart_service.update_cart(user_id, action["product_id"], "add", action.get("quantity") or 1)
                notes.append(f"Added product {action['product_id']} to the cart.")
            elif kind == "remove_from_cart":
                cart_service.update_cart(user_id, action["product_id"], "remove", 1)
                notes.append(f"Removed product {action['product_id']} from the cart.")
            elif kind == "set_cart_qty":
                cart_service.update_cart(user_id, action["product_id"], "set", action["quantity"])
                notes.append(f"Set product {action['product_id']} quantity to {action['quantity']}.")
        except Exception as e:
            notes.append(f"Could not {kind.replace('_', ' ')}: {str(e)[:120]}")
    if any(a["type"] in SERVER_TYPES for a in actions):
        if not any(a["type"] == "refresh_cart" for a in actions):
            actions.append({"type": "refresh_cart"})
        if not any(a["type"] == "open_cart" for a in actions):
            actions.append({"type": "open_cart"})
    return notes


def public_actions(actions: list[dict]) -> list[dict]:
    """Actions the browser should run. Cart writes already happened on the server."""
    return [a for a in actions if a["type"] in UI_TYPES or a["type"] == "refresh_cart" or a["type"] == "open_cart"]
