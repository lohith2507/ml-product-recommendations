"""
Hot-path stores for cart, live facts, and agent thread state.

Backends:
  - sqlite: local / tests (default when FIRESTORE_ENABLED=0)
  - firestore: Cloud Run high-traffic path
  - memory: unit tests without a DB
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from database.models import get_db, dict_from_row
from recommenders.firestore_client import firestore_enabled

FACTS_CACHE_TTL_SEC = float(os.environ.get("FACTS_CACHE_TTL_SEC", "8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _product_snapshot(product_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT product_id, title, category, price, list_price, image_url FROM products WHERE product_id = ?",
            (product_id,),
        ).fetchone()
    if not row:
        raise LookupError(f"Unknown product {product_id}")
    return dict_from_row(row)


def _user_exists(user_id: int) -> bool:
    with get_db() as conn:
        return bool(conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone())


def _catalog_price(product_id: int) -> tuple[float, float]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT price, list_price FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
    if not row:
        return 0.0, 0.0
    return float(row["price"] or 0), float(row["list_price"] or 0)


# ---------- Facts ----------


class FactsStore(ABC):
    @abstractmethod
    def upsert_price(self, product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
        ...

    @abstractmethod
    def upsert_stock(self, product_id: int, quantity: int, warehouse_id: str = "default"):
        ...

    @abstractmethod
    def get_facts(self, product_ids: list[int], warehouse_id: str = "default") -> dict[int, dict]:
        ...


class SqliteFactsStore(FactsStore):
    def upsert_price(self, product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
        with get_db() as conn:
            existing = conn.execute(
                "SELECT stock_qty, list_price FROM product_facts WHERE product_id=? AND warehouse_id=?",
                (product_id, warehouse_id),
            ).fetchone()
            stock = existing["stock_qty"] if existing else 100
            lp = list_price if list_price is not None else (existing["list_price"] if existing else None)
            if lp is None:
                _, catalog_lp = _catalog_price(product_id)
                lp = catalog_lp
            conn.execute(
                """
                INSERT INTO product_facts (product_id, warehouse_id, price, list_price, stock_qty, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(product_id, warehouse_id) DO UPDATE SET
                    price = excluded.price,
                    list_price = excluded.list_price,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (product_id, warehouse_id, price, lp, stock),
            )

    def upsert_stock(self, product_id: int, quantity: int, warehouse_id: str = "default"):
        with get_db() as conn:
            existing = conn.execute(
                "SELECT price, list_price FROM product_facts WHERE product_id=? AND warehouse_id=?",
                (product_id, warehouse_id),
            ).fetchone()
            if existing:
                price, list_price = existing["price"], existing["list_price"]
            else:
                price, list_price = _catalog_price(product_id)
            conn.execute(
                """
                INSERT INTO product_facts (product_id, warehouse_id, price, list_price, stock_qty, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(product_id, warehouse_id) DO UPDATE SET
                    stock_qty = excluded.stock_qty,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (product_id, warehouse_id, price, list_price, quantity),
            )

    def get_facts(self, product_ids: list[int], warehouse_id: str = "default") -> dict[int, dict]:
        ids = [int(i) for i in product_ids]
        if not ids:
            return {}
        with get_db() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM product_facts
                WHERE warehouse_id = ? AND product_id IN ({",".join("?" * len(ids))})
                """,
                [warehouse_id, *ids],
            ).fetchall()
        return {r["product_id"]: dict_from_row(r) for r in rows}


class FirestoreFactsStore(FactsStore):
    def _doc_id(self, warehouse_id: str, product_id: int) -> str:
        return f"{warehouse_id}_{product_id}"

    def upsert_price(self, product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
        from recommenders.firestore_client import collection

        ref = collection("product_facts").document(self._doc_id(warehouse_id, product_id))
        snap = ref.get()
        data = snap.to_dict() if snap.exists else {}
        stock = data.get("stock_qty", 100)
        lp = list_price if list_price is not None else data.get("list_price")
        if lp is None:
            _, lp = _catalog_price(product_id)
        ref.set(
            {
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "price": float(price),
                "list_price": float(lp or 0),
                "stock_qty": int(stock),
                "updated_at": _now(),
            },
            merge=True,
        )
        _facts_cache_invalidate(warehouse_id, product_id)

    def upsert_stock(self, product_id: int, quantity: int, warehouse_id: str = "default"):
        from recommenders.firestore_client import collection

        ref = collection("product_facts").document(self._doc_id(warehouse_id, product_id))
        snap = ref.get()
        data = snap.to_dict() if snap.exists else {}
        if data:
            price, list_price = data.get("price"), data.get("list_price")
        else:
            price, list_price = _catalog_price(product_id)
        ref.set(
            {
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "price": float(price or 0),
                "list_price": float(list_price or 0),
                "stock_qty": int(quantity),
                "updated_at": _now(),
            },
            merge=True,
        )
        _facts_cache_invalidate(warehouse_id, product_id)

    def get_facts(self, product_ids: list[int], warehouse_id: str = "default") -> dict[int, dict]:
        from recommenders.firestore_client import collection, get_firestore_client

        ids = [int(i) for i in product_ids]
        if not ids:
            return {}
        refs = [collection("product_facts").document(self._doc_id(warehouse_id, pid)) for pid in ids]
        docs = get_firestore_client().get_all(refs)
        out = {}
        for doc in docs:
            if not doc.exists:
                continue
            data = doc.to_dict() or {}
            pid = int(data.get("product_id") or doc.id.rsplit("_", 1)[-1])
            out[pid] = data
        return out


class MemoryFactsStore(FactsStore):
    def __init__(self):
        self._data: dict[tuple[str, int], dict] = {}

    def upsert_price(self, product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
        key = (warehouse_id, product_id)
        prev = self._data.get(key, {})
        lp = list_price if list_price is not None else prev.get("list_price")
        if lp is None:
            _, lp = _catalog_price(product_id)
        self._data[key] = {
            "product_id": product_id,
            "warehouse_id": warehouse_id,
            "price": float(price),
            "list_price": float(lp or 0),
            "stock_qty": prev.get("stock_qty", 100),
            "updated_at": _now(),
        }

    def upsert_stock(self, product_id: int, quantity: int, warehouse_id: str = "default"):
        key = (warehouse_id, product_id)
        prev = self._data.get(key)
        if prev:
            price, list_price = prev["price"], prev["list_price"]
        else:
            price, list_price = _catalog_price(product_id)
        self._data[key] = {
            "product_id": product_id,
            "warehouse_id": warehouse_id,
            "price": float(price or 0),
            "list_price": float(list_price or 0),
            "stock_qty": int(quantity),
            "updated_at": _now(),
        }

    def get_facts(self, product_ids: list[int], warehouse_id: str = "default") -> dict[int, dict]:
        return {
            pid: dict(self._data[(warehouse_id, pid)])
            for pid in product_ids
            if (warehouse_id, pid) in self._data
        }


# TTL cache wrapping any FactsStore
_facts_cache: dict[tuple[str, int], tuple[float, dict]] = {}
_facts_cache_lock = threading.Lock()


def _facts_cache_invalidate(warehouse_id: str, product_id: int):
    with _facts_cache_lock:
        _facts_cache.pop((warehouse_id, product_id), None)


class CachedFactsStore(FactsStore):
    def __init__(self, inner: FactsStore, ttl: float = FACTS_CACHE_TTL_SEC):
        self.inner = inner
        self.ttl = ttl

    def upsert_price(self, product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
        self.inner.upsert_price(product_id, price, list_price, warehouse_id)
        _facts_cache_invalidate(warehouse_id, product_id)

    def upsert_stock(self, product_id: int, quantity: int, warehouse_id: str = "default"):
        self.inner.upsert_stock(product_id, quantity, warehouse_id)
        _facts_cache_invalidate(warehouse_id, product_id)

    def get_facts(self, product_ids: list[int], warehouse_id: str = "default") -> dict[int, dict]:
        now = time.time()
        result: dict[int, dict] = {}
        missing: list[int] = []
        with _facts_cache_lock:
            for pid in product_ids:
                hit = _facts_cache.get((warehouse_id, pid))
                if hit and now - hit[0] < self.ttl:
                    result[pid] = hit[1]
                else:
                    missing.append(pid)
        if missing:
            fetched = self.inner.get_facts(missing, warehouse_id=warehouse_id)
            with _facts_cache_lock:
                for pid, fact in fetched.items():
                    _facts_cache[(warehouse_id, pid)] = (now, fact)
                    result[pid] = fact
        return result


# ---------- Cart ----------


class CartStore(ABC):
    @abstractmethod
    def get_cart_items(self, user_id: int) -> list[dict]:
        """Return [{product_id, quantity}] ordered newest-first."""

    @abstractmethod
    def apply_item(self, user_id: int, product_id: int, action: str, quantity: int, max_quantity: int = 99):
        ...

    @abstractmethod
    def propose_action(self, user_id: int, product_id: int, action: str, quantity: int, reason: str) -> str:
        """Returns action_id."""

    @abstractmethod
    def claim_action(self, action_id: str, user_id: int) -> Optional[dict]:
        """Atomically claim proposed -> applied. Returns action row or None if already claimed."""

    @abstractmethod
    def get_action(self, action_id: str, user_id: int) -> Optional[dict]:
        ...

    @abstractmethod
    def reject_action(self, action_id: str, user_id: int) -> Optional[str]:
        """Returns new status or None if missing."""


class SqliteCartStore(CartStore):
    def get_cart_items(self, user_id: int) -> list[dict]:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT product_id, quantity FROM cart_items
                WHERE user_id = ? ORDER BY updated_at DESC, product_id
                """,
                (user_id,),
            ).fetchall()
        return [dict_from_row(r) for r in rows]

    def apply_item(self, user_id: int, product_id: int, action: str, quantity: int, max_quantity: int = 99):
        with get_db() as conn:
            if action == "add":
                conn.execute(
                    """
                    INSERT INTO cart_items (user_id, product_id, quantity) VALUES (?, ?, ?)
                    ON CONFLICT(user_id, product_id) DO UPDATE SET
                        quantity = MIN(quantity + excluded.quantity, ?),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, product_id, quantity, max_quantity),
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

    def propose_action(self, user_id: int, product_id: int, action: str, quantity: int, reason: str) -> str:
        action_id = uuid.uuid4().hex
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO agent_cart_actions (action_id, user_id, product_id, action, quantity, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (action_id, user_id, product_id, action, quantity, (reason or "")[:300]),
            )
        return action_id

    def claim_action(self, action_id: str, user_id: int) -> Optional[dict]:
        with get_db() as conn:
            claimed = conn.execute(
                """
                UPDATE agent_cart_actions
                SET status = 'applied', resolved_at = CURRENT_TIMESTAMP
                WHERE action_id = ? AND user_id = ? AND status = 'proposed'
                """,
                (action_id, user_id),
            ).rowcount
            if not claimed:
                return None
            row = conn.execute(
                "SELECT * FROM agent_cart_actions WHERE action_id = ? AND user_id = ?",
                (action_id, user_id),
            ).fetchone()
        return dict_from_row(row) if row else None

    def get_action(self, action_id: str, user_id: int) -> Optional[dict]:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM agent_cart_actions WHERE action_id = ? AND user_id = ?",
                (action_id, user_id),
            ).fetchone()
        return dict_from_row(row) if row else None

    def reject_action(self, action_id: str, user_id: int) -> Optional[str]:
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
        return row["status"] if row else None


class FirestoreCartStore(CartStore):
    def get_cart_items(self, user_id: int) -> list[dict]:
        from recommenders.firestore_client import collection

        snap = collection("carts").document(str(user_id)).get()
        if not snap.exists:
            return []
        items = (snap.to_dict() or {}).get("items") or {}
        rows = []
        for pid, meta in items.items():
            rows.append(
                {
                    "product_id": int(meta.get("product_id", pid)),
                    "quantity": int(meta.get("quantity") or 0),
                    "updated_at": meta.get("updated_at") or "",
                }
            )
        rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return [r for r in rows if r["quantity"] > 0]

    def apply_item(self, user_id: int, product_id: int, action: str, quantity: int, max_quantity: int = 99):
        from recommenders.firestore_client import collection, get_firestore_client
        from google.cloud import firestore as fs

        db = get_firestore_client()
        ref = collection("carts").document(str(user_id))
        transaction = db.transaction()

        @fs.transactional
        def _txn(txn):
            snap = ref.get(transaction=txn)
            data = snap.to_dict() if snap.exists else {"items": {}}
            items = dict(data.get("items") or {})
            key = str(product_id)
            current = int((items.get(key) or {}).get("quantity") or 0)
            if action == "add":
                new_qty = min(current + quantity, max_quantity)
            elif action == "set" and quantity > 0:
                new_qty = min(quantity, max_quantity)
            else:
                new_qty = 0
            if new_qty <= 0:
                items.pop(key, None)
            else:
                items[key] = {"product_id": int(product_id), "quantity": new_qty, "updated_at": _now()}
            # Full document write so removed item keys are not left behind by merge.
            txn.set(ref, {"items": items, "updated_at": _now()})

        _txn(transaction)

    def propose_action(self, user_id: int, product_id: int, action: str, quantity: int, reason: str) -> str:
        from recommenders.firestore_client import collection

        action_id = uuid.uuid4().hex
        collection("cart_actions").document(action_id).set(
            {
                "action_id": action_id,
                "user_id": user_id,
                "product_id": product_id,
                "action": action,
                "quantity": quantity,
                "reason": (reason or "")[:300],
                "status": "proposed",
                "created_at": _now(),
                "resolved_at": None,
            }
        )
        return action_id

    def claim_action(self, action_id: str, user_id: int) -> Optional[dict]:
        from recommenders.firestore_client import collection, get_firestore_client
        from google.cloud import firestore as fs

        db = get_firestore_client()
        ref = collection("cart_actions").document(action_id)
        transaction = db.transaction()

        @fs.transactional
        def _txn(txn):
            snap = ref.get(transaction=txn)
            if not snap.exists:
                return None
            data = snap.to_dict() or {}
            if int(data.get("user_id")) != int(user_id):
                return None
            if data.get("status") != "proposed":
                return None
            data["status"] = "applied"
            data["resolved_at"] = _now()
            txn.set(ref, data)
            return data

        return _txn(transaction)

    def get_action(self, action_id: str, user_id: int) -> Optional[dict]:
        from recommenders.firestore_client import collection

        snap = collection("cart_actions").document(action_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        if int(data.get("user_id")) != int(user_id):
            return None
        return data

    def reject_action(self, action_id: str, user_id: int) -> Optional[str]:
        from recommenders.firestore_client import collection, get_firestore_client
        from google.cloud import firestore as fs

        db = get_firestore_client()
        ref = collection("cart_actions").document(action_id)
        transaction = db.transaction()

        @fs.transactional
        def _txn(txn):
            snap = ref.get(transaction=txn)
            if not snap.exists:
                return None
            data = snap.to_dict() or {}
            if int(data.get("user_id")) != int(user_id):
                return None
            if data.get("status") == "proposed":
                data["status"] = "rejected"
                data["resolved_at"] = _now()
                txn.set(ref, data)
            return data.get("status")

        return _txn(transaction)


class MemoryCartStore(CartStore):
    def __init__(self):
        self.items: dict[int, dict[int, int]] = {}
        self.actions: dict[str, dict] = {}

    def get_cart_items(self, user_id: int) -> list[dict]:
        return [
            {"product_id": pid, "quantity": qty}
            for pid, qty in sorted(self.items.get(user_id, {}).items())
        ]

    def apply_item(self, user_id: int, product_id: int, action: str, quantity: int, max_quantity: int = 99):
        bag = self.items.setdefault(user_id, {})
        current = bag.get(product_id, 0)
        if action == "add":
            bag[product_id] = min(current + quantity, max_quantity)
        elif action == "set" and quantity > 0:
            bag[product_id] = min(quantity, max_quantity)
        else:
            bag.pop(product_id, None)

    def propose_action(self, user_id: int, product_id: int, action: str, quantity: int, reason: str) -> str:
        action_id = uuid.uuid4().hex
        self.actions[action_id] = {
            "action_id": action_id,
            "user_id": user_id,
            "product_id": product_id,
            "action": action,
            "quantity": quantity,
            "reason": reason,
            "status": "proposed",
        }
        return action_id

    def claim_action(self, action_id: str, user_id: int) -> Optional[dict]:
        row = self.actions.get(action_id)
        if not row or row["user_id"] != user_id or row["status"] != "proposed":
            return None
        row["status"] = "applied"
        return dict(row)

    def get_action(self, action_id: str, user_id: int) -> Optional[dict]:
        row = self.actions.get(action_id)
        if not row or row["user_id"] != user_id:
            return None
        return dict(row)

    def reject_action(self, action_id: str, user_id: int) -> Optional[str]:
        row = self.actions.get(action_id)
        if not row or row["user_id"] != user_id:
            return None
        if row["status"] == "proposed":
            row["status"] = "rejected"
        return row["status"]


# ---------- Agent threads ----------


class ThreadStore(ABC):
    @abstractmethod
    def load_messages(self, thread_id: str) -> list[dict]:
        ...

    @abstractmethod
    def save_messages(self, thread_id: str, messages: list[dict], max_keep: int = 20):
        ...


class SqliteThreadStore(ThreadStore):
    """Best-effort local fallback: no durable threads (stateless)."""

    def load_messages(self, thread_id: str) -> list[dict]:
        return []

    def save_messages(self, thread_id: str, messages: list[dict], max_keep: int = 20):
        return


class FirestoreThreadStore(ThreadStore):
    def load_messages(self, thread_id: str) -> list[dict]:
        from recommenders.firestore_client import collection

        snap = collection("agent_threads").document(thread_id).get()
        if not snap.exists:
            return []
        return list((snap.to_dict() or {}).get("messages") or [])

    def save_messages(self, thread_id: str, messages: list[dict], max_keep: int = 20):
        from recommenders.firestore_client import collection

        trimmed = messages[-max_keep:]
        collection("agent_threads").document(thread_id).set(
            {"thread_id": thread_id, "messages": trimmed, "updated_at": _now()},
            merge=True,
        )


class MemoryThreadStore(ThreadStore):
    def __init__(self):
        self._threads: dict[str, list[dict]] = {}

    def load_messages(self, thread_id: str) -> list[dict]:
        return list(self._threads.get(thread_id) or [])

    def save_messages(self, thread_id: str, messages: list[dict], max_keep: int = 20):
        self._threads[thread_id] = messages[-max_keep:]


# ---------- Factory ----------

_facts_store: Optional[FactsStore] = None
_cart_store: Optional[CartStore] = None
_thread_store: Optional[ThreadStore] = None
_store_lock = threading.Lock()


def reset_stores():
    global _facts_store, _cart_store, _thread_store
    with _store_lock:
        _facts_store = None
        _cart_store = None
        _thread_store = None
        _facts_cache.clear()


def get_facts_store() -> FactsStore:
    global _facts_store
    if _facts_store is None:
        with _store_lock:
            if _facts_store is None:
                backend = FirestoreFactsStore() if firestore_enabled() else SqliteFactsStore()
                _facts_store = CachedFactsStore(backend)
    return _facts_store


def get_cart_store() -> CartStore:
    global _cart_store
    if _cart_store is None:
        with _store_lock:
            if _cart_store is None:
                _cart_store = FirestoreCartStore() if firestore_enabled() else SqliteCartStore()
    return _cart_store


def get_thread_store() -> ThreadStore:
    global _thread_store
    if _thread_store is None:
        with _store_lock:
            if _thread_store is None:
                _thread_store = FirestoreThreadStore() if firestore_enabled() else SqliteThreadStore()
    return _thread_store


# Re-export helpers used by cart service
require_user = _user_exists
product_snapshot = _product_snapshot
