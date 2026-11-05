"""Tests for budget kit, prefs, deal watch, cheaper match, checkout, eval."""

from recommenders import budget_kit
from recommenders import checkout as checkout_service
from recommenders import deal_watch
from recommenders import deals
from recommenders import eval_loop
from recommenders import member_prefs
from recommenders.shopping_agent import ShoppingAgent


def test_member_prefs_roundtrip():
    prefs = member_prefs.update_prefs(
        1,
        {
            "preferred_brands": ["Patagonia"],
            "avoid_brands": ["NoName"],
            "sizes": {"shirt": "M"},
            "max_budget": 150,
        },
    )
    assert "Patagonia" in prefs["preferred_brands"]
    assert prefs["sizes"]["shirt"] == "M"
    summary = member_prefs.format_prefs_for_prompt(prefs)
    assert "Patagonia" in summary


def test_budget_kit_respects_budget(retriever):
    kit = budget_kit.plan_kit(1, 80, ["backpack", "tee"], retriever=retriever)
    assert kit["spent"] <= kit["budget"] + 0.01
    assert kit["remaining"] >= -0.01
    assert isinstance(kit["items"], list)


def test_find_cheaper_match_and_substitute(retriever):
    cheaper = deals.find_cheaper_match(product_id=1, limit=3, retriever=retriever)
    assert "products" in cheaper
    products = deals.get_products([1, 2])
    if len(products) >= 2:
        receipt = deals.explain_substitute(1, 2)
        assert "why" in receipt
        assert "receipt" in receipt


def test_deal_watch_fires_on_drop():
    deal_watch.watch_product(1, 1, target_price=9999.0)
    # Force last_price high then drop
    deal_watch._update_last_price(1, 1, 50.0)
    alerts = deal_watch.check_price_drop(1, 10.0)
    assert any(a["product_id"] == 1 for a in alerts)
    listed = deal_watch.list_alerts(1, limit=5)
    assert listed


def test_checkout_links_for_cart():
    from recommenders import cart as cart_service

    cart_service.update_cart(1, 1, "add", 1)
    handoff = checkout_service.cart_checkout(1)
    assert handoff["item_count"] >= 1
    assert handoff["items"]
    assert handoff["items"][0]["checkout_url"].startswith("http")
    cart_service.update_cart(1, 1, "remove", 1)


def test_eval_feedback_and_score():
    row = eval_loop.record_feedback(1, "up", message="hi", response="hello", tools_used=["find_best_deals"])
    assert row["rating"] == "up"
    stats = eval_loop.feedback_stats()
    assert stats["total"] >= 1
    scored = eval_loop.score_turn(
        eval_loop.GOLDEN_CHATS[0],
        tools_used=["find_best_deals"],
        response="A deal for $19",
    )
    assert scored["pass"] is True


def test_agent_new_power_tools(retriever):
    agent = ShoppingAgent(clients=[], retriever=retriever, use_langgraph=False)
    turn = {"shown": [], "pending": [], "context": set(), "tools": []}
    kit = agent._run_tool("plan_budget_kit", {"budget": 100, "needs": ["backpack"]}, 1, turn)
    assert "items" in kit
    prefs = agent._run_tool("update_member_prefs", {"avoid_brands": ["XBrand"]}, 1, turn)
    assert "prefs" in prefs
    cheaper = agent._run_tool("find_cheaper_match", {"product_id": 1}, 1, turn)
    assert "products" in cheaper
    checkout = agent._run_tool("get_checkout_links", {}, 1, turn)
    assert "items" in checkout or "primary_url" in checkout or checkout.get("item_count") == 0
