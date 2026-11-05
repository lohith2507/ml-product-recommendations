"""Structured per-member shopping preferences (sizes, brands, budget)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recommenders.firestore_client import firestore_enabled

_lock = threading.Lock()
_DEFAULT = {
    "preferred_brands": [],
    "avoid_brands": [],
    "sizes": {},
    "max_budget": None,
    "notes": "",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_path() -> Path:
    root = Path(__file__).resolve().parent.parent / "data"
    root.mkdir(exist_ok=True)
    return root / "member_prefs.json"


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


def _normalize(raw: dict | None) -> dict:
    base = dict(_DEFAULT)
    if not raw:
        return base
    for key in base:
        if key in raw and raw[key] is not None:
            base[key] = raw[key]
    base["preferred_brands"] = [str(b).strip() for b in (base.get("preferred_brands") or []) if str(b).strip()][:20]
    base["avoid_brands"] = [str(b).strip() for b in (base.get("avoid_brands") or []) if str(b).strip()][:20]
    sizes = base.get("sizes") or {}
    if not isinstance(sizes, dict):
        sizes = {}
    base["sizes"] = {str(k)[:40]: str(v)[:40] for k, v in sizes.items()}
    if base.get("max_budget") is not None:
        try:
            base["max_budget"] = float(base["max_budget"])
        except (TypeError, ValueError):
            base["max_budget"] = None
    base["notes"] = str(base.get("notes") or "")[:500]
    return base


def get_prefs(user_id: int) -> dict:
    if firestore_enabled():
        from recommenders.firestore_client import collection

        snap = collection("member_prefs").document(str(user_id)).get()
        return _normalize(snap.to_dict() if snap.exists else None)
    with _lock:
        data = _load_local()
        return _normalize(data.get(str(user_id)))


def update_prefs(user_id: int, patch: dict[str, Any]) -> dict:
    current = get_prefs(user_id)
    merged = dict(current)
    for key in _DEFAULT:
        if key in patch and patch[key] is not None:
            merged[key] = patch[key]
    merged = _normalize(merged)
    merged["updated_at"] = _now()

    if firestore_enabled():
        from recommenders.firestore_client import collection

        collection("member_prefs").document(str(user_id)).set(merged, merge=True)
    else:
        with _lock:
            data = _load_local()
            data[str(user_id)] = merged
            _save_local(data)
    return get_prefs(user_id)


def format_prefs_for_prompt(prefs: dict) -> str:
    parts = []
    if prefs.get("preferred_brands"):
        parts.append("prefer brands: " + ", ".join(prefs["preferred_brands"]))
    if prefs.get("avoid_brands"):
        parts.append("avoid brands: " + ", ".join(prefs["avoid_brands"]))
    if prefs.get("sizes"):
        parts.append("sizes: " + ", ".join(f"{k}={v}" for k, v in prefs["sizes"].items()))
    if prefs.get("max_budget") is not None:
        parts.append(f"max budget ${prefs['max_budget']:g}")
    if prefs.get("notes"):
        parts.append(prefs["notes"])
    return "; ".join(parts) if parts else "none set"


def product_matches_prefs(product: dict, prefs: dict) -> tuple[bool, str]:
    """Return (ok, reason). Soft filter: avoid brands hard-fail; preferred brands boost via reason."""
    title = (product.get("title") or "").lower()
    for brand in prefs.get("avoid_brands") or []:
        if brand.lower() in title:
            return False, f"avoid brand {brand}"
    return True, ""
