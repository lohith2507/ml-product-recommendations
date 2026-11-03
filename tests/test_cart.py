import pytest

from recommenders import cart


def test_add_accumulates_and_reports_savings():
    cart.update_cart(1, 1, "add", 1)
    result = cart.update_cart(1, 1, "add", 2)

    assert result["item_count"] == 3
    assert result["subtotal"] == 150.0
    assert result["savings"] == 150.0


def test_set_zero_and_remove_delete_the_item():
    cart.update_cart(1, 1, "add", 2)
    assert cart.update_cart(1, 1, "set", 0)["items"] == []

    cart.update_cart(1, 4, "add", 1)
    assert cart.update_cart(1, 4, "remove")["items"] == []


@pytest.mark.parametrize("action,quantity", [("add", 0), ("add", 100), ("set", -1), ("add", "2")])
def test_rejects_invalid_quantities(action, quantity):
    with pytest.raises(cart.CartError):
        cart.update_cart(1, 1, action, quantity)


def test_rejects_unknown_product_and_user():
    with pytest.raises(cart.CartError):
        cart.update_cart(1, 999, "add", 1)
    with pytest.raises(cart.CartError):
        cart.update_cart(999, 1, "add", 1)


def test_proposal_does_not_change_cart_until_confirmed():
    action = cart.propose_action(1, 4, "add", 2, "cheap backpack")
    assert cart.get_cart(1)["items"] == []

    result = cart.confirm_action(action["action_id"], 1)
    assert [(i["product_id"], i["quantity"]) for i in result["items"]] == [(4, 2)]


def test_confirm_is_applied_exactly_once():
    action = cart.propose_action(1, 4, "add", 1)
    cart.confirm_action(action["action_id"], 1)
    result = cart.confirm_action(action["action_id"], 1)

    assert result["item_count"] == 1


def test_other_members_cannot_confirm_an_action():
    action = cart.propose_action(1, 4, "add", 1)
    with pytest.raises(cart.CartError):
        cart.confirm_action(action["action_id"], 2)
    assert cart.get_cart(1)["items"] == []


def test_rejected_action_cannot_be_confirmed():
    action = cart.propose_action(1, 4, "add", 1)
    cart.reject_action(action["action_id"], 1)
    with pytest.raises(cart.CartError):
        cart.confirm_action(action["action_id"], 1)
    assert cart.get_cart(1)["items"] == []
