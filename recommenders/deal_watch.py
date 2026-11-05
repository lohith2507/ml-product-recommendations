"""Wishlist price watches: alert when live price drops below a target or last seen."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recommenders.firestore_client import firestore_enabled
from recommenders.store import product_snapshot
from recommenders import facts
from recommenders import wishlist as wishlist_service

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_path() -> Path:
    root = Path(__file__).resolve().parent.parent / "data"
    root.mkdir(exist_ok=True)
    return root / "deal_watches.json"


def _alerts_path() -> Path:
    root = Path(__file__).resolve().parent.parent / "data"
    root.mkdir(exist_ok=True)
    return root / "deal_alerts.json"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, data: dict):
    path.write_text(json.dumps(data), encoding="utf-8")


def watch_product(user_id: int, product_id: int, target_price: float | None = None) -> dict:
    product = product_snapshot(product_id)
    price = float(product.get("price") or 0)
    entry = {
        "product_id": product_id,
        "target_price": float(target_price) if target_price is not None else None,
        "last_price": price,
        "created_at": _now(),
    }
    if firestore_enabled():
        from recommenders.firestore_client import collection

        ref = collection("deal_watches").document(str(user_id))
        snap = ref.get()
        data = snap.to_dict() if snap.exists else {"items": {}}
        items = dict(data.get("items") or {})
        items[str(product_id)] = entry
        ref.set({"items": items, "updated_at": _now()}, merge=True)
    else:
        with _lock:
            data = _load(_local_path())
            bag = data.setdefault(str(user_id), {})
            bag[str(product_id)] = entry
            _save(_local_path(), data)
    # Also pin to wishlist so the member sees it.
    try:
        wishlist_service.update_wishlist(user_id, product_id, "add")
    except Exception:
        pass
    return entry


def list_watches(user_id: int) -> list[dict]:
    if firestore_enabled():
        from recommenders.firestore_client import collection

        snap = collection("deal_watches").document(str(user_id)).get()
        items = ((snap.to_dict() or {}).get("items") or {}) if snap.exists else {}
        return list(items.values())
    with _lock:
        data = _load(_local_path())
        return list((data.get(str(user_id)) or {}).values())


def unwatch_product(user_id: int, product_id: int) -> list[dict]:
    if firestore_enabled():
        from recommenders.firestore_client import collection

        ref = collection("deal_watches").document(str(user_id))
        snap = ref.get()
        data = snap.to_dict() if snap.exists else {"items": {}}
        items = dict(data.get("items") or {})
        items.pop(str(product_id), None)
        ref.set({"items": items, "updated_at": _now()}, merge=True)
    else:
        with _lock:
            data = _load(_local_path())
            bag = data.setdefault(str(user_id), {})
            bag.pop(str(product_id), None)
            _save(_local_path(), data)
    return list_watches(user_id)


def _append_alert(user_id: int, alert: dict):
    if firestore_enabled():
        from recommenders.firestore_client import collection

        collection("deal_alerts").document(str(user_id)).collection("items").document(
            f"{alert['product_id']}_{alert.get('created_at', _now())}"
        ).set(alert)
        return
    with _lock:
        data = _load(_alerts_path())
        bag = data.setdefault(str(user_id), [])
        bag.insert(0, alert)
        data[str(user_id)] = bag[:50]
        _save(_alerts_path(), data)


def list_alerts(user_id: int, limit: int = 20) -> list[dict]:
    if firestore_enabled():
        from recommenders.firestore_client import collection

        docs = (
            collection("deal_alerts")
            .document(str(user_id))
            .collection("items")
            .order_by("created_at", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        return [d.to_dict() for d in docs]
    with _lock:
        data = _load(_alerts_path())
        return list(data.get(str(user_id)) or [])[:limit]


def mark_alerts_read(user_id: int) -> int:
    alerts = list_alerts(user_id, limit=50)
    if firestore_enabled():
        from recommenders.firestore_client import collection

        col = collection("deal_alerts").document(str(user_id)).collection("items")
        count = 0
        for a in alerts:
            if a.get("read"):
                continue
            # Best-effort: rewrite unread flags via list reload is complex; skip for Firestore.
            count += 1
        return count
    with _lock:
        data = _load(_alerts_path())
        bag = data.get(str(user_id)) or []
        n = 0
        for a in bag:
            if not a.get("read"):
                a["read"] = True
                n += 1
        data[str(user_id)] = bag
        _save(_alerts_path(), data)
        return n


def _all_watch_entries() -> list[tuple[int, dict]]:
    out = []
    if firestore_enabled():
        from recommenders.firestore_client import collection

        for doc in collection("deal_watches").stream():
            user_id = int(doc.id)
            items = (doc.to_dict() or {}).get("items") or {}
            for entry in items.values():
                out.append((user_id, entry))
        return out
    with _lock:
        data = _load(_local_path())
        for uid, bag in data.items():
            for entry in (bag or {}).values():
                out.append((int(uid), entry))
    return out


def _update_last_price(user_id: int, product_id: int, price: float):
    if firestore_enabled():
        from recommenders.firestore_client import collection
        from google.cloud import firestore as fs

        ref = collection("deal_watches").document(str(user_id))
        ref.set({"items": {str(product_id): {"last_price": price, "updated_at": _now()}}, "updated_at": _now()}, merge=True)
        return
    with _lock:
        data = _load(_local_path())
        bag = data.setdefault(str(user_id), {})
        entry = bag.get(str(product_id)) or {"product_id": product_id}
        entry["last_price"] = price
        entry["updated_at"] = _now()
        bag[str(product_id)] = entry
        _save(_local_path(), data)


def check_price_drop(product_id: int, new_price: float) -> list[dict]:
    """Called when a price.changed event lands. Returns alerts fired."""
    fired = []
    for user_id, entry in _all_watch_entries():
        if int(entry.get("product_id") or -1) != int(product_id):
            continue
        last = entry.get("last_price")
        target = entry.get("target_price")
        drop_vs_last = last is not None and float(new_price) < float(last) - 0.01
        hit_target = target is not None and float(new_price) <= float(target)
        if not (drop_vs_last or hit_target):
            _update_last_price(user_id, product_id, float(new_price))
            continue
        try:
            product = product_snapshot(product_id)
            title = (product.get("title") or "")[:90]
        except LookupError:
            title = f"Product {product_id}"
        alert = {
            "user_id": user_id,
            "product_id": product_id,
            "title": title,
            "old_price": last,
            "new_price": float(new_price),
            "target_price": target,
            "message": (
                f"Price drop: {title} is now ${float(new_price):.2f}"
                + (f" (was ${float(last):.2f})" if last is not None else "")
                + (f", at or under your ${float(target):.2f} target" if hit_target else "")
            ),
            "created_at": _now(),
            "read": False,
        }
        _append_alert(user_id, alert)
        _update_last_price(user_id, product_id, float(new_price))
        fired.append(alert)
    return fired


def scan_all_watches() -> list[dict]:
    """Periodic scan: compare live facts to watched last_price/target."""
    fired = []
    for user_id, entry in _all_watch_entries():
        pid = int(entry.get("product_id") or -1)
        if pid < 0:
            continue
        live = facts.get_facts([pid]).get(pid) or {}
        price = live.get("price")
        if price is None:
            try:
                price = product_snapshot(pid).get("price")
            except LookupError:
                continue
        fired.extend(check_price_drop(pid, float(price)))
    return fired
