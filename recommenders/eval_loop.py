"""Offline golden chats + online thumbs feedback for the shopping agent."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recommenders.firestore_client import firestore_enabled

_lock = threading.Lock()

# Small golden set: each case has a user message and required tool names / keywords.
GOLDEN_CHATS = [
    {
        "id": "deals_backpack",
        "message": "best backpack deals under 40",
        "expect_tools_any": ["find_best_deals", "search_products"],
        "expect_response_keywords": ["$"],
    },
    {
        "id": "cart_savings",
        "message": "how much am I saving in my cart?",
        "expect_tools_any": ["cart_savings_summary", "get_cart"],
        "expect_response_keywords": [],
    },
    {
        "id": "explain",
        "message": "why is product 1 a good deal?",
        "expect_tools_any": ["explain_deal"],
        "expect_response_keywords": [],
    },
    {
        "id": "budget_kit",
        "message": "build a $100 back to school kit with backpack and lunch box",
        "expect_tools_any": ["plan_budget_kit"],
        "expect_response_keywords": [],
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _feedback_path() -> Path:
    root = Path(__file__).resolve().parent.parent / "data"
    root.mkdir(exist_ok=True)
    return root / "chat_feedback.json"


def _load() -> list:
    path = _feedback_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(rows: list):
    _feedback_path().write_text(json.dumps(rows, indent=2), encoding="utf-8")


def record_feedback(
    user_id: int,
    rating: str,
    message: str = "",
    response: str = "",
    tools_used: list | None = None,
    thread_id: str | None = None,
) -> dict:
    rating = (rating or "").lower()
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    row = {
        "feedback_id": uuid.uuid4().hex,
        "user_id": int(user_id),
        "rating": rating,
        "message": (message or "")[:500],
        "response": (response or "")[:1000],
        "tools_used": list(tools_used or [])[:20],
        "thread_id": thread_id,
        "created_at": _now(),
    }
    if firestore_enabled():
        from recommenders.firestore_client import collection

        collection("chat_feedback").document(row["feedback_id"]).set(row)
    else:
        with _lock:
            rows = _load()
            rows.insert(0, row)
            _save(rows[:500])
    return row


def feedback_stats(limit: int = 200) -> dict:
    if firestore_enabled():
        from recommenders.firestore_client import collection

        rows = [d.to_dict() for d in collection("chat_feedback").limit(limit).stream()]
    else:
        with _lock:
            rows = _load()[:limit]
    ups = sum(1 for r in rows if r.get("rating") == "up")
    downs = sum(1 for r in rows if r.get("rating") == "down")
    total = ups + downs
    return {
        "total": total,
        "up": ups,
        "down": downs,
        "up_rate": round(ups / total, 3) if total else None,
        "recent": rows[:20],
    }


def score_turn(case: dict, tools_used: list[str], response: str) -> dict[str, Any]:
    tools = set(tools_used or [])
    expect_any = set(case.get("expect_tools_any") or [])
    tool_ok = (not expect_any) or bool(tools & expect_any)
    keywords = case.get("expect_response_keywords") or []
    text = (response or "").lower()
    kw_ok = all(k.lower() in text for k in keywords)
    return {
        "id": case["id"],
        "pass": bool(tool_ok and kw_ok),
        "tool_ok": tool_ok,
        "keyword_ok": kw_ok,
        "tools_used": list(tools),
    }


def run_golden(agent, user_id: int = 1) -> dict:
    """Run GOLDEN_CHATS against a ShoppingAgent (legacy or langgraph)."""
    results = []
    for case in GOLDEN_CHATS:
        try:
            out = agent.chat(user_id=user_id, message=case["message"], conversation_history=[])
            scored = score_turn(case, out.get("tools_used") or [], out.get("response") or "")
            scored["response_preview"] = (out.get("response") or "")[:160]
        except Exception as e:
            scored = {"id": case["id"], "pass": False, "error": str(e)[:200]}
        results.append(scored)
    passed = sum(1 for r in results if r.get("pass"))
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "results": results,
    }
