"""
Server-side shopping cart.

Members change the cart directly from the UI. The shopping agent can only
*propose* changes (stored in agent_cart_actions); a proposal is applied once,
when the member confirms it.
"""

import uuid

from database.models import get_db, dict_from_row

MAX_QUANTITY = 99
ACTIONS = ("add", "set", "remove")


class CartError(ValueError):
    """Invalid cart request (unknown user/product, bad quantity, stale action)."""


def _require_user(conn, user_id):
    if not conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone():
        raise CartError(f"Unknown user {user_id}")


def _require_product(conn, product_id):
    row = conn.execute(
        "SELECT product_id, title, price, list_price, image_url, category FROM products WHERE product_id = ?",
        (product_id,),
    ).fetchone()
    if not row:
        raise CartError(f"Unknown product {product_id}")
    return dict_from_row(row)


def _validate_quantity(action, quantity):
    if not isinstance(quantity, int) or isinstance(quantity, bool):
        raise CartError("Quantity must be an integer")
    low = 0 if action == "set" else 1
    if action != "remove" and not (low <= quantity <= MAX_QUANTITY):
        raise CartError(f"Quantity must be between {low} and {MAX_QUANTITY}")


def _apply(conn, user_id, product_id, action, quantity):
    if action == "add":
        conn.execute(
            """
            INSERT INTO cart_items (user_id, product_id, quantity) VALUES (?, ?, ?)
            ON CONFLICT(user_id, product_id) DO UPDATE SET
                quantity = MIN(quantity + excluded.quantity, ?),
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, product_id, quantity, MAX_QUANTITY),
        )
    elif action == "set" and quantity > 0:
        conn.execute(
            """
            INSERT INTO cart_items (user_id, product_id, quantity) VALUES (?, ?, ?)
            ON CONFLICT(user_id, product_id) DO UPDATE SET
                quantity = excluded.quantity,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, product_id, quantity),
        )
    else:
        conn.execute(
            "DELETE FROM cart_items WHERE user_id = ? AND product_id = ?",
            (user_id, product_id),
        )


def _read_cart(conn, user_id):
    rows = conn.execute(
        """
        SELECT c.product_id, c.quantity, p.title, p.category, p.price, p.list_price, p.image_url
        FROM cart_items c
        JOIN products p ON p.product_id = c.product_id
        WHERE c.user_id = ?
        ORDER BY c.updated_at DESC, c.product_id
        """,
        (user_id,),
    ).fetchall()

    items, subtotal, savings = [], 0.0, 0.0
    for r in rows:
        item = dict_from_row(r)
        price = item["price"] or 0.0
        list_price = item["list_price"] or 0.0
        subtotal += price * item["quantity"]
        if list_price > price > 0:
            savings += (list_price - price) * item["quantity"]
        items.append(item)

    return {
        "user_id": user_id,
        "items": items,
        "item_count": sum(i["quantity"] for i in items),
        "subtotal": round(subtotal, 2),
        "savings": round(savings, 2),
    }


def get_cart(user_id):
    with get_db() as conn:
        _require_user(conn, user_id)
        return _read_cart(conn, user_id)


def update_cart(user_id, product_id, action, quantity=1):
    """Apply a member-initiated cart change and return the updated cart."""
    if action not in ACTIONS:
        raise CartError(f"Unknown cart action '{action}'")
    _validate_quantity(action, quantity)
    with get_db() as conn:
        _require_user(conn, user_id)
        _require_product(conn, product_id)
        _apply(conn, user_id, product_id, action, quantity)
        return _read_cart(conn, user_id)


def propose_action(user_id, product_id, action, quantity=1, reason=""):
    """Record an agent-proposed cart change. Nothing in the cart changes until confirm_action."""
    if action not in ACTIONS:
        raise CartError(f"Unknown cart action '{action}'")
    _validate_quantity(action, quantity)
    action_id = uuid.uuid4().hex
    with get_db() as conn:
        _require_user(conn, user_id)
        product = _require_product(conn, product_id)
        conn.execute(
            """
            INSERT INTO agent_cart_actions (action_id, user_id, product_id, action, quantity, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (action_id, user_id, product_id, action, quantity, (reason or "")[:300]),
        )
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
    with get_db() as conn:
        claimed = conn.execute(
            """
            UPDATE agent_cart_actions
            SET status = 'applied', resolved_at = CURRENT_TIMESTAMP
            WHERE action_id = ? AND user_id = ? AND status = 'proposed'
            """,
            (action_id, user_id),
        ).rowcount

        row = conn.execute(
            "SELECT * FROM agent_cart_actions WHERE action_id = ? AND user_id = ?",
            (action_id, user_id),
        ).fetchone()
        if not row:
            raise CartError("Cart action not found")
        if row["status"] == "rejected":
            raise CartError("Cart action was dismissed")

        if claimed:
            _apply(conn, user_id, row["product_id"], row["action"], row["quantity"])
        return _read_cart(conn, user_id)


def reject_action(action_id, user_id):
    with get_db() as conn:
        conn.execute(
            """
            UPDATE agent_cart_actions
            SET status = 'rejected', resolved_at = CURRENT_TIMESTAMP
            WHERE action_id = ? AND user_id = ? AND status = 'proposed'
            """,
            (action_id, user_id),
        )
        row = conn.execute(
            "SELECT status FROM agent_cart_actions WHERE action_id = ? AND user_id = ?",
            (action_id, user_id),
        ).fetchone()
    if not row:
        raise CartError("Cart action not found")
    if row["status"] == "applied":
        raise CartError("Cart action was already applied")
    return {"action_id": action_id, "status": row["status"]}
