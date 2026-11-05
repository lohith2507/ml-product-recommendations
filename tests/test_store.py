from recommenders.store import MemoryCartStore, MemoryFactsStore, MemoryThreadStore, CachedFactsStore


def test_memory_cart_propose_confirm_flow():
    store = MemoryCartStore()
    action_id = store.propose_action(1, 1, "add", 2, "deal")
    claimed = store.claim_action(action_id, 1)
    assert claimed["status"] == "applied"
    store.apply_item(1, 1, "add", 2)
    items = store.get_cart_items(1)
    assert items == [{"product_id": 1, "quantity": 2}]
    assert store.claim_action(action_id, 1) is None


def test_memory_facts_and_cache(monkeypatch):
    inner = MemoryFactsStore()
    store = CachedFactsStore(inner, ttl=60)
    store.upsert_price(1, 9.99, list_price=20.0)
    store.upsert_stock(1, 3)
    first = store.get_facts([1])
    assert first[1]["price"] == 9.99
    assert first[1]["stock_qty"] == 3
    # Mutate inner without invalidating cache to prove TTL cache is used.
    inner.upsert_price(1, 1.0, list_price=20.0)
    cached = store.get_facts([1])
    assert cached[1]["price"] == 9.99


def test_thread_store_trims():
    store = MemoryThreadStore()
    msgs = [{"role": "user", "content": str(i)} for i in range(30)]
    store.save_messages("user_1", msgs, max_keep=5)
    assert len(store.load_messages("user_1")) == 5
    assert store.load_messages("user_1")[0]["content"] == "25"


def test_thread_store_clear_writes_empty():
    store = MemoryThreadStore()
    store.save_messages("user_9", [{"role": "user", "content": "shoes"}])
    store.save_messages("user_9", [])
    assert store.load_messages("user_9") == []
