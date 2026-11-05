"""Member wishlist stored in Firestore (or local JSON when Firestore is off)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from datetime import datetime, timezone

from recommenders.firestore_client import firestore_enabled
from recommenders.store import product_snapshot

_lock = threading.Lock()
_memory: dict[int, dict[int, dict]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_path() -> Path:
    root = Path(__file__).resolve().parent.parent / "data"
    root.mkdir(exist_ok=True)
    return root / "wishlists.json"


def _load_local() -> dict:
    path = _local_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_local(data: dict):
    _local_path().write_text(json.dumps(data), encoding="utf-8")


def get_wishlist(user_id: int) -> list[dict]:
    if firestore_enabled():
        from recommenders.firestore_client import collection

        snap = collection("wishlists").document(str(user_id)).get()
        items = ((snap.to_dict() or {}).get("items") or {}) if snap.exists else {}
        product_ids = [int(k) for k in items.keys()]
    else:
        with _lock:
            data = _load_local()
            product_ids = [int(k) for k in (data.get(str(user_id)) or {}).keys()]

    out = []
    for pid in product_ids:
        try:
            out.append(product_snapshot(pid))
        except LookupError:
            continue
    return out


def update_wishlist(user_id: int, product_id: int, action: str = "add") -> list[dict]:
    action = (action or "add").lower()
    if firestore_enabled():
        from recommenders.firestore_client import collection

        ref = collection("wishlists").document(str(user_id))
        snap = ref.get()
        data = snap.to_dict() if snap.exists else {"items": {}}
        items = dict(data.get("items") or {})
        key = str(product_id)
        if action == "remove":
            items.pop(key, None)
        else:
            items[key] = {"product_id": product_id, "added_at": _now()}
        ref.set({"items": items, "updated_at": _now()}, merge=True)
    else:
        with _lock:
            data = _load_local()
            bag = data.setdefault(str(user_id), {})
            key = str(product_id)
            if action == "remove":
                bag.pop(key, None)
            else:
                bag[key] = {"product_id": product_id}
            _save_local(data)
    return get_wishlist(user_id)


def reset_wishlist_memory():
    global _memory
    with _lock:
        _memory = {}
