"""
Per-member chat RAG memory.

Stores embedded chat turns and retrieves the most similar past chunks for a
new message. Firestore in production; in-process memory for tests / local.
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from recommenders.firestore_client import firestore_enabled

EmbedFn = Callable[[list[str]], list[list[float]]]
MAX_CHUNKS_PER_USER = int(os.environ.get("CHAT_MEMORY_MAX_CHUNKS", "80"))
RETRIEVE_CANDIDATES = int(os.environ.get("CHAT_MEMORY_CANDIDATES", "40"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dot(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def cosine(a: list[float], b: list[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


class ChatMemoryStore:
    def add_chunk(self, user_id: int, text: str, embedding: list[float], *, role: str = "turn", product_ids: list[int] | None = None):
        raise NotImplementedError

    def list_recent(self, user_id: int, limit: int = RETRIEVE_CANDIDATES) -> list[dict]:
        raise NotImplementedError

    def clear_user(self, user_id: int):
        raise NotImplementedError


class MemoryChatMemoryStore(ChatMemoryStore):
    def __init__(self):
        self._by_user: dict[int, list[dict]] = {}

    def add_chunk(self, user_id: int, text: str, embedding: list[float], *, role: str = "turn", product_ids: list[int] | None = None):
        chunks = self._by_user.setdefault(user_id, [])
        chunks.append(
            {
                "chunk_id": uuid.uuid4().hex,
                "user_id": user_id,
                "text": text,
                "role": role,
                "embedding": list(embedding),
                "product_ids": product_ids or [],
                "created_at": _now(),
            }
        )
        if len(chunks) > MAX_CHUNKS_PER_USER:
            self._by_user[user_id] = chunks[-MAX_CHUNKS_PER_USER:]

    def list_recent(self, user_id: int, limit: int = RETRIEVE_CANDIDATES) -> list[dict]:
        return list(self._by_user.get(user_id, [])[-limit:])

    def clear_user(self, user_id: int):
        self._by_user.pop(user_id, None)


class FirestoreChatMemoryStore(ChatMemoryStore):
    def add_chunk(self, user_id: int, text: str, embedding: list[float], *, role: str = "turn", product_ids: list[int] | None = None):
        from recommenders.firestore_client import collection

        chunk_id = uuid.uuid4().hex
        collection("chat_memory").document(str(user_id)).collection("chunks").document(chunk_id).set(
            {
                "chunk_id": chunk_id,
                "user_id": user_id,
                "text": text[:2000],
                "role": role,
                "embedding": list(embedding),
                "product_ids": product_ids or [],
                "created_at": _now(),
            }
        )

    def list_recent(self, user_id: int, limit: int = RETRIEVE_CANDIDATES) -> list[dict]:
        from recommenders.firestore_client import collection

        docs = collection("chat_memory").document(str(user_id)).collection("chunks").stream()
        out = []
        for doc in docs:
            data = doc.to_dict() or {}
            out.append(data)
        out.sort(key=lambda c: c.get("created_at") or "", reverse=True)
        return out[:limit]

    def clear_user(self, user_id: int):
        from recommenders.firestore_client import collection

        parent = collection("chat_memory").document(str(user_id)).collection("chunks")
        for doc in parent.stream():
            doc.reference.delete()


class SqliteChatMemoryStore(ChatMemoryStore):
    """JSON file under local data dir — used when Firestore is off."""

    def __init__(self):
        import json
        from pathlib import Path

        self._json = json
        root = Path(__file__).resolve().parent.parent / "data"
        root.mkdir(exist_ok=True)
        self._path = root / "chat_memory.json"
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict:
        try:
            return self._json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: dict):
        self._path.write_text(self._json.dumps(data), encoding="utf-8")

    def add_chunk(self, user_id: int, text: str, embedding: list[float], *, role: str = "turn", product_ids: list[int] | None = None):
        data = self._load()
        key = str(user_id)
        chunks = data.setdefault(key, [])
        chunks.append(
            {
                "chunk_id": uuid.uuid4().hex,
                "user_id": user_id,
                "text": text[:2000],
                "role": role,
                "embedding": list(embedding),
                "product_ids": product_ids or [],
                "created_at": _now(),
            }
        )
        data[key] = chunks[-MAX_CHUNKS_PER_USER:]
        self._save(data)

    def list_recent(self, user_id: int, limit: int = RETRIEVE_CANDIDATES) -> list[dict]:
        chunks = self._load().get(str(user_id), [])
        return chunks[-limit:]

    def clear_user(self, user_id: int):
        data = self._load()
        data.pop(str(user_id), None)
        self._save(data)


_store: Optional[ChatMemoryStore] = None
_store_lock = threading.Lock()
_embed_fn: Optional[EmbedFn] = None


def reset_chat_memory():
    global _store, _embed_fn
    with _store_lock:
        _store = None
        _embed_fn = None


def set_embed_fn(fn: Optional[EmbedFn]):
    global _embed_fn
    _embed_fn = fn


def get_chat_memory_store() -> ChatMemoryStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                if firestore_enabled():
                    _store = FirestoreChatMemoryStore()
                else:
                    _store = SqliteChatMemoryStore()
    return _store


def _default_embed(texts: list[str]) -> list[list[float]]:
    if _embed_fn:
        return _embed_fn(texts)
    from recommenders.vertex_client import embed_texts

    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")


def remember_turn(user_id: int, user_message: str, assistant_message: str, product_ids: list[int] | None = None):
    text = f"Member: {user_message}\nAgent: {assistant_message}"
    try:
        vectors = _default_embed([text])
        if not vectors:
            return
        get_chat_memory_store().add_chunk(
            user_id, text, vectors[0], role="turn", product_ids=product_ids or []
        )
    except Exception as e:
        print(f"⚠️ chat memory write skipped: {e}")


def retrieve_memory(user_id: int, query: str, top_k: int = 3) -> list[dict]:
    """Return top_k similar past turns for this member."""
    chunks = get_chat_memory_store().list_recent(user_id)
    if not chunks or not (query or "").strip():
        return []
    try:
        qvec = _default_embed([query])[0]
    except Exception as e:
        print(f"⚠️ chat memory embed skipped: {e}")
        return chunks[-top_k:]

    scored = []
    for chunk in chunks:
        emb = chunk.get("embedding") or []
        if not emb:
            continue
        scored.append((cosine(qvec, emb), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"text": c["text"], "score": round(s, 4), "product_ids": c.get("product_ids") or [], "created_at": c.get("created_at")}
        for s, c in scored[:top_k]
        if s > 0.15
    ]


def format_memory_for_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return "(none)"
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"{i}. (sim={c.get('score', '?')}) {c.get('text', '')[:400]}")
    return "\n".join(lines)
