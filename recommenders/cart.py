"""
Server-side shopping cart.

Members change the cart directly from the UI. The shopping agent can only
*propose* changes (stored in cart_actions); a proposal is applied once,
when the member confirms it.

Hot path is backed by Firestore in production (see recommenders.store);
SQLite remains the default for local/tests.
"""

from recommenders.store import get_cart_store, product_snapshot, require_user

MAX_QUANTITY = 99
ACTIONS = ("add", "set", "remove")


class CartError(ValueError):
    """Invalid cart request (unknown user/product, bad quantity, stale action)."""


def _validate_quantity(action, quantity):
    if not isinstance(quantity, int) or isinstance(quantity, bool):
        raise CartError("Quantity must be an integer")
    low = 0 if action == "set" else 1
    if action != "remove" and not (low <= quantity <= MAX_QUANTITY):
        raise CartError(f"Quantity must be between {low} and {MAX_QUANTITY}")


def _require_user(user_id):
    if not require_user(user_id):
        raise CartError(f"Unknown user {user_id}")


def _require_product(product_id):
    try:
        return product_snapshot(product_id)
    except LookupError as e:
        raise CartError(str(e)) from e


def _read_cart(user_id):
    from recommenders.facts import apply_live_facts

    raw_items = get_cart_store().get_cart_items(user_id)
    items, subtotal, savings = [], 0.0, 0.0
    for row in raw_items:
        try:
            product = product_snapshot(row["product_id"])
        except LookupError:
            continue
        product["quantity"] = row["quantity"]
        items.append(product)

    items = apply_live_facts(items)
    for item in items:
        price = item.get("price") or 0.0
        list_price = item.get("list_price") or 0.0
        subtotal += price * item["quantity"]
        if list_price > price > 0:
            savings += (list_price - price) * item["quantity"]

    return {
        "user_id": user_id,
        "items": items,
        "item_count": sum(i["quantity"] for i in items),
        "subtotal": round(subtotal, 2),
        "savings": round(savings, 2),
    }


def get_cart(user_id):
    _require_user(user_id)
    return _read_cart(user_id)


def update_cart(user_id, product_id, action, quantity=1):
    """Apply a member-initiated cart change and return the updated cart."""
    if action not in ACTIONS:
        raise CartError(f"Unknown cart action '{action}'")
    _validate_quantity(action, quantity)
    _require_user(user_id)
    _require_product(product_id)
    get_cart_store().apply_item(user_id, product_id, action, quantity, MAX_QUANTITY)
    cart = _read_cart(user_id)
    try:
        from recommenders.events import get_publisher
        get_publisher().cart_event(user_id, action, product_id, quantity)
    except Exception as e:
        print(f"⚠️ cart event publish skipped: {e}")
    return cart


def propose_action(user_id, product_id, action, quantity=1, reason=""):
    """Record an agent-proposed cart change. Nothing in the cart changes until confirm_action."""
    if action not in ACTIONS:
        raise CartError(f"Unknown cart action '{action}'")
    _validate_quantity(action, quantity)
    _require_user(user_id)
    product = _require_product(product_id)
    action_id = get_cart_store().propose_action(user_id, product_id, action, quantity, reason or "")
    return {
        "action_id": action_id,
        "action": action,
        "quantity": quantity,
        "reason": reason or "",
        "product": product,
    }


def confirm_action(action_id, user_id):
    """
    Apply a proposed action exactly once. Confirming an already-applied action
    returns the current cart without changing it again.
    """
    store = get_cart_store()
    claimed = store.claim_action(action_id, user_id)
    row = store.get_action(action_id, user_id)
    if not row:
        raise CartError("Cart action not found")
    if row["status"] == "rejected":
        raise CartError("Cart action was dismissed")

    if claimed:
        store.apply_item(user_id, row["product_id"], row["action"], row["quantity"], MAX_QUANTITY)
        try:
            from recommenders.events import get_publisher
            get_publisher().cart_event(
                user_id, row["action"], row["product_id"], row["quantity"], action_id=action_id, source="agent_confirm"
            )
        except Exception as e:
            print(f"⚠️ cart event publish skipped: {e}")
    return _read_cart(user_id)


def reject_action(action_id, user_id):
    status = get_cart_store().reject_action(action_id, user_id)
    if status is None:
        raise CartError("Cart action not found")
    if status == "applied":
        raise CartError("Cart action was already applied")
    return {"action_id": action_id, "status": status}
