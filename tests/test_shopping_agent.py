import json
from types import SimpleNamespace

from recommenders import cart
from recommenders.shopping_agent import ShoppingAgent


def tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content="", tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class ScriptedClient:
    """OpenAI-compatible fake that returns canned replies and records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


class FailingClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        raise RuntimeError("model retired")


def test_agent_finds_deals_and_proposes_cart_change_without_applying_it(retriever):
    client = ScriptedClient([
        reply(tool_calls=[tool_call("c1", "find_best_deals", {"query": "running shoes", "max_price": 55})]),
        reply(tool_calls=[tool_call("c2", "propose_cart_change", {"product_id": 1, "action": "add", "quantity": 2, "reason": "50% off"})]),
        reply("I've prepared 2 pairs of Trail Running Shoes for you to confirm."),
    ])
    agent = ShoppingAgent([(client, "test-model")], retriever=retriever)

    result = agent.chat(1, "best running shoes under $55, add two")

    assert result["tools_used"] == ["find_best_deals", "propose_cart_change"]
    assert [p["product_id"] for p in result["products"]] == [1, 3]
    assert "Under $55" in result["active_context"]
    assert len(result["pending_actions"]) == 1
    assert cart.get_cart(1)["items"] == []

    tool_result = json.loads(client.requests[1]["messages"][-1]["content"])
    assert tool_result["products"][0]["savings"] == 50.0

    applied = cart.confirm_action(result["pending_actions"][0]["action_id"], 1)
    assert applied["item_count"] == 2


def test_invalid_tool_arguments_are_reported_back_to_the_model(retriever):
    client = ScriptedClient([
        reply(tool_calls=[tool_call("c1", "propose_cart_change", {"product_id": 999, "action": "add"})]),
        reply("Sorry, I couldn't find that product."),
    ])
    agent = ShoppingAgent([(client, "test-model")], retriever=retriever)

    result = agent.chat(1, "add product 999")

    assert result["pending_actions"] == []
    assert "Unknown product" in client.requests[1]["messages"][-1]["content"]


def test_falls_back_to_next_model_then_to_plain_deals(retriever):
    client = ScriptedClient([reply("Here you go.")])
    agent = ShoppingAgent([(FailingClient(), "retired"), (client, "backup")], retriever=retriever)
    assert agent.chat(1, "hello")["response"] == "Here you go."

    offline = ShoppingAgent([(FailingClient(), "retired")], retriever=retriever)
    result = offline.chat(1, "running shoes")
    assert result["tools_used"] == ["find_best_deals"]
    assert result["products"][0]["product_id"] == 1


def test_last_round_forces_a_text_answer(retriever):
    loop = [reply(tool_calls=[tool_call(f"c{i}", "get_cart", {})]) for i in range(5)]
    client = ScriptedClient(loop + [reply("Your cart is empty.")])
    agent = ShoppingAgent([(client, "test-model")], retriever=retriever)

    result = agent.chat(1, "what's in my cart?")

    assert result["response"] == "Your cart is empty."
    assert client.requests[-1]["tool_choice"] == "none"
