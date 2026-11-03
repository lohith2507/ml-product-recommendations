"""
Rebuild the FAISS product index using Vertex AI text embeddings.

Stores:
  products_vertex.index
  products_vertex_map.json
  products_vertex_meta.json  (model, dim, project)

Query-time encoding then uses Vertex too, so Cloud Run does not need torch.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

import faiss
import numpy as np

from recommenders.vertex_client import embed_model, embed_texts, location, project_id

DB_PATH = os.path.join(os.path.dirname(__file__), "recommendation.db")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "products_vertex.index")
MAP_PATH = os.path.join(os.path.dirname(__file__), "products_vertex_map.json")
META_PATH = os.path.join(os.path.dirname(__file__), "products_vertex_meta.json")
BATCH_SIZE = 32


def _product_text(row) -> str:
    title = row["title"] or ""
    desc = row["description"] or ""
    category = row["category"] or ""
    try:
        features = json.loads(row["features"]) if isinstance(row["features"], str) else (row["features"] or [])
        features_text = " ".join(features)
    except Exception:
        features_text = ""
    return f"{title}. {category}. {desc}. {features_text}"


def setup_vertex_index():
    print(f"Project={project_id()} location={location()} model={embed_model()}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    products = conn.execute(
        "SELECT product_id, title, description, category, features FROM products"
    ).fetchall()
    conn.close()

    if not products:
        raise SystemExit("No products found. Run setup_data.py first.")

    texts = [_product_text(row) for row in products]
    product_ids = [row["product_id"] for row in products]
    print(f"Embedding {len(texts)} products in batches of {BATCH_SIZE}...")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        for attempt in range(4):
            try:
                vectors.extend(embed_texts(batch, task_type="RETRIEVAL_DOCUMENT"))
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  batch {start}-{start + len(batch)} failed ({e}); retry in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed batch starting at {start}")
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    embeddings = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({i: pid for i, pid in enumerate(product_ids)}, f)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": embed_model(),
                "dim": int(embeddings.shape[1]),
                "count": len(product_ids),
                "project": project_id(),
                "location": location(),
            },
            f,
            indent=2,
        )

    print(f"[OK] Vertex FAISS index written: {INDEX_PATH} ({embeddings.shape})")


if __name__ == "__main__":
    setup_vertex_index()
