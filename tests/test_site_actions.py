"""Site control actions and chat payload safety."""

from recommenders.shopping_agent import ShoppingAgent
from recommenders.site_actions import apply_server_actions, normalize_actions, public_actions


def test_normalize_site_actions_drops_unknown_types():
    actions = normalize_actions({
        "actions": [
            {"type": "navigate", "section": "analytics"},
            {"type": "search", "query": "backpack", "category": "Backpacks"},
            {"type": "switch_algo", "algo": "hybrid"},
            {"type": "navigate", "section": "billing"},
            {"type": "delete_database"},
        ]
    })
    kinds = [a["type"] for a in actions]
    assert kinds == ["navigate", "search", "switch_algo"]
    assert public_actions(actions)[0]["section"] == "analytics"


def test_explicit_cart_action_applies_and_refreshes():
    actions = normalize_actions({"actions": [{"type": "add_to_cart", "product_id": 1, "quantity": 1}]})
    notes = apply_server_actions(1, actions)
    assert notes and "Added" in notes[0]
    assert any(a["type"] == "refresh_cart" for a in actions)
    assert any(a["type"] == "open_cart" for a in public_actions(actions))
    from recommenders import cart as cart_service
    cart = cart_service.get_cart(1)
    assert any(i["product_id"] == 1 for i in cart["items"])
    cart_service.update_cart(1, 1, "remove", 1)


def test_agent_control_site_and_safe_payload(retriever):
    agent = ShoppingAgent(clients=[], retriever=retriever, use_langgraph=False)
    turn = {"shown": [], "pending": [], "context": set(), "tools": [], "ui_actions": []}
    result = agent._run_tool(
        "control_site",
        {"actions": [{"type": "navigate", "section": "products"}, {"type": "filter_category", "category": "all"}]},
        1,
        turn,
    )
    assert result["status"] == "done"
    assert turn["ui_actions"][0]["section"] == "products"
    safe = agent._safe_out({"response": "", "products": [{"product_id": 1, "price": 1.5}], "ui_actions": turn["ui_actions"]})
    assert safe["response"]
    assert safe["products"][0]["product_id"] == 1
    assert safe["ui_actions"]
