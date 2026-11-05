"""Budget / trip kit planner: greedily pack best deals under a dollar cap."""

from __future__ import annotations

from recommenders import deals
from recommenders import member_prefs


def plan_kit(
    user_id: int,
    budget: float,
    needs: list[str],
    category: str | None = None,
    retriever=None,
) -> dict:
    """
    Build a kit under `budget` USD from free-text need slots.
    Each need gets at most one product (best deal that still fits remaining budget).
    """
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        raise ValueError("Budget must be a number")
    if budget <= 0 or budget > 5000:
        raise ValueError("Budget must be between 0 and 5000")

    needs = [str(n).strip() for n in (needs or []) if str(n).strip()][:8]
    if not needs:
        raise ValueError("Provide at least one need, e.g. backpack, lunchbox")

    prefs = member_prefs.get_prefs(user_id)
    remaining = budget
    picks = []
    skipped = []

    for need in needs:
        candidates = deals.find_best_deals(
            need,
            category=category,
            max_price=remaining,
            limit=12,
            retriever=retriever,
        )
        chosen = None
        for p in candidates:
            ok, reason = member_prefs.product_matches_prefs(p, prefs)
            if not ok:
                continue
            price = float(p.get("price") or 0)
            if price <= 0 or price > remaining:
                continue
            chosen = p
            break
        if not chosen:
            skipped.append({"need": need, "reason": "no item fits remaining budget"})
            continue
        price = float(chosen["price"])
        remaining = round(remaining - price, 2)
        picks.append(
            {
                **deals.compact(chosen),
                "need": need,
                "reason": f"Best value for '{need}' within remaining ${remaining + price:g}",
            }
        )

    return {
        "budget": budget,
        "spent": round(budget - remaining, 2),
        "remaining": remaining,
        "items": picks,
        "skipped": skipped,
        "prefs_applied": member_prefs.format_prefs_for_prompt(prefs),
    }
