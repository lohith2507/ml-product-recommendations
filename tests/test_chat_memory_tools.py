from recommenders import chat_memory
from recommenders.shopping_agent import ShoppingAgent
from recommenders import wishlist as wishlist_service


def test_chat_memory_retrieve_ranks_similar_turns():
    chat_memory.remember_turn(1, "I need a backpack under 40", "Here is a hiking backpack deal.")
    chat_memory.remember_turn(1, "Show me kitchen knives", "Here are chef knives on sale.")
    hits = chat_memory.retrieve_memory(1, "backpack deals under forty", top_k=2)
    assert hits
    assert any("backpack" in h["text"].lower() for h in hits)


def test_memory_store_clear_user_drops_chunks():
    store = chat_memory.MemoryChatMemoryStore()
    store.add_chunk(7, "watch under 50", [1.0, 0.0])
    store.clear_user(7)
    assert store.list_recent(7) == []


def test_wishlist_add_and_remove():
    items = wishlist_service.update_wishlist(1, 4, "add")
    assert any(i["product_id"] == 4 for i in items)
    items = wishlist_service.update_wishlist(1, 4, "remove")
    assert all(i["product_id"] != 4 for i in items)


def test_agent_new_tools_via_legacy(retriever):
    agent = ShoppingAgent(clients=[], retriever=retriever, use_langgraph=False)
    turn = {"shown": [], "pending": [], "context": set(), "tools": []}
    explained = agent._run_tool("explain_deal", {"product_id": 1}, 1, turn)
    assert explained["product"]["product_id"] == 1
    assert "why" in explained

    summary = agent._run_tool("cart_savings_summary", {}, 1, turn)
    assert "subtotal" in summary

    suggestions = agent._run_tool("suggest_next_questions", {}, 1, turn)
    assert len(suggestions["suggestions"]) == 3

    recalled = agent._run_tool("recall_past_chats", {"query": "backpack", "limit": 2}, 1, turn)
    assert "memories" in recalled
