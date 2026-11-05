import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import models  # noqa: E402

PRODUCTS = [
    # product_id, title, category, price, list_price, avg_rating, rating_count
    (1, "Trail Running Shoes", "Men's Shoes", 50.0, 100.0, 4.5, 800),
    (2, "Road Running Shoes", "Men's Shoes", 60.0, 60.0, 4.8, 900),
    (3, "Budget Running Shoes", "Men's Shoes", 20.0, 25.0, 3.0, 5),
    (4, "Hiking Backpack 40L", "Backpacks", 35.0, 70.0, 4.2, 300),
]


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("PUBSUB_ENABLED", "0")
    monkeypatch.setenv("BQ_ENABLED", "0")
    monkeypatch.setenv("FIRESTORE_ENABLED", "0")
    from recommenders.store import reset_stores
    from recommenders.chat_memory import reset_chat_memory, set_embed_fn, MemoryChatMemoryStore
    import recommenders.chat_memory as chat_memory_mod

    reset_stores()
    reset_chat_memory()
    # Deterministic tiny embeddings for RAG unit tests (no Vertex).
    def fake_embed(texts):
        out = []
        for t in texts:
            v = [0.0] * 32
            for token in t.lower().replace("\n", " ").split():
                h = sum(ord(c) for c in token)
                v[h % 32] += 1.0 + (len(token) % 5) * 0.1
            out.append(v)
        return out

    set_embed_fn(fake_embed)
    chat_memory_mod._store = MemoryChatMemoryStore()
    models.init_db()
    with models.get_db() as conn:
        conn.execute("INSERT INTO users (user_id, username) VALUES (1, 'alice'), (2, 'bob')")
        conn.executemany(
            "INSERT INTO products (product_id, title, category, price, list_price, avg_rating, rating_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            PRODUCTS,
        )
    yield
    reset_stores()
    reset_chat_memory()


@pytest.fixture
def retriever():
    """Deterministic stand-in for FAISS: running-shoe queries hit products 1-3."""
    def search(query, top_k):
        if "shoe" in query.lower():
            return [
                {"product_id": 2, "similarity": 0.9},
                {"product_id": 1, "similarity": 0.85},
                {"product_id": 3, "similarity": 0.8},
            ][:top_k]
        return [{"product_id": 4, "similarity": 0.7}]
    return search
