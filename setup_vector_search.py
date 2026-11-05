"""
Upload product embeddings to GCS and create a Vertex AI Vector Search index.

Serving still uses the local FAISS Vertex index until VECTOR_SEARCH_INDEX_ENDPOINT
and VECTOR_SEARCH_DEPLOYED_INDEX_ID are set. This script is the Phase-3 path to
managed Vector Search at Costco catalog scale.

Usage:
  python setup_vector_search.py              # upload JSONL only
  python setup_vector_search.py --create     # upload + create index (async, ~30–60 min)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

from recommenders.vertex_client import embed_model, embed_texts, location, project_id

DB_PATH = os.path.join(os.path.dirname(__file__), "recommendation.db")
BUCKET = os.environ.get("VECTOR_SEARCH_BUCKET", f"{project_id()}-vector-search")
GCS_PREFIX = os.environ.get("VECTOR_SEARCH_PREFIX", "product-embeddings")
INDEX_DISPLAY_NAME = os.environ.get("VECTOR_SEARCH_INDEX_NAME", "shopping-agent-products")
BATCH_SIZE = 32
REGION = os.environ.get("VERTEX_EMBED_LOCATION") or location() or "us-central1"


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


def _load_products():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT product_id, title, description, category, features FROM products"
    ).fetchall()
    conn.close()
    if not rows:
        raise SystemExit("No products found. Run setup_data.py first.")
    return rows


def _ensure_bucket(storage_client):
    from google.cloud.exceptions import NotFound

    try:
        storage_client.get_bucket(BUCKET)
    except NotFound:
        print(f"Creating bucket gs://{BUCKET} in {REGION}...")
        storage_client.create_bucket(BUCKET, location=REGION)


def _vectors_from_faiss():
    """Reuse products_vertex.index when present (avoids re-embedding 5k products)."""
    import faiss
    import numpy as np

    index_path = os.path.join(os.path.dirname(__file__), "products_vertex.index")
    map_path = os.path.join(os.path.dirname(__file__), "products_vertex_map.json")
    if not (os.path.exists(index_path) and os.path.exists(map_path)):
        return None, None
    index = faiss.read_index(index_path)
    with open(map_path, encoding="utf-8") as f:
        mapping = {int(k): int(v) for k, v in json.load(f).items()}
    # Reconstruct vectors in FAISS row order.
    vectors = [index.reconstruct(i).tolist() for i in range(index.ntotal)]
    product_ids = [mapping[i] for i in range(index.ntotal)]
    _ = np  # kept for type clarity / future batch reconstruct
    print(f"Reusing FAISS Vertex index ({index.ntotal} x {index.d})")
    return product_ids, vectors


def upload_embeddings(create_index: bool = False):
    from google.cloud import storage
    from google.cloud import aiplatform

    print(f"Project={project_id()} region={REGION} model={embed_model()} bucket=gs://{BUCKET}")
    product_ids, vectors = _vectors_from_faiss()
    if vectors is None:
        products = _load_products()
        texts = [_product_text(r) for r in products]
        product_ids = [r["product_id"] for r in products]
        print(f"Embedding {len(texts)} products...")
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            for attempt in range(4):
                try:
                    vectors.extend(embed_texts(batch, task_type="RETRIEVAL_DOCUMENT"))
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  batch {start} failed ({e}); retry in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to embed batch at {start}")
            print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    dim = len(vectors[0])
    local_jsonl = os.path.join(os.path.dirname(__file__), "vector_search_embeddings.json")
    with open(local_jsonl, "w", encoding="utf-8") as f:
        for pid, vec in zip(product_ids, vectors):
            f.write(json.dumps({"id": str(pid), "embedding": vec}) + "\n")

    storage_client = storage.Client(project=project_id())
    _ensure_bucket(storage_client)
    blob_path = f"{GCS_PREFIX}/embeddings.json"
    bucket = storage_client.bucket(BUCKET)
    bucket.blob(blob_path).upload_from_filename(local_jsonl)
    gcs_uri = f"gs://{BUCKET}/{GCS_PREFIX}/"
    print(f"[OK] Uploaded embeddings to {gcs_uri}embeddings.json (dim={dim}, n={len(product_ids)})")

    meta = {
        "model": embed_model(),
        "dim": dim,
        "count": len(product_ids),
        "gcs_uri": gcs_uri,
        "project": project_id(),
        "region": REGION,
    }
    meta_path = os.path.join(os.path.dirname(__file__), "vector_search_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not create_index:
        print("Skipping index create (pass --create). FAISS continues to serve queries.")
        return meta

    aiplatform.init(project=project_id(), location=REGION)
    print(f"Creating Vector Search index '{INDEX_DISPLAY_NAME}' (this can take 30–60+ minutes)...")
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=gcs_uri,
        dimensions=dim,
        approximate_neighbors_count=20,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=10,
        description="Shopping agent product embeddings",
        index_update_method="STREAM_UPDATE",
    )
    print(f"[OK] Index created: {index.resource_name}")

    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=f"{INDEX_DISPLAY_NAME}-endpoint",
        public_endpoint_enabled=True,
        description="Shopping agent Vector Search endpoint",
    )
    print(f"[OK] Endpoint created: {endpoint.resource_name}")

    deployed_id = "shopping_agent_products"
    endpoint = endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_id,
        display_name=INDEX_DISPLAY_NAME,
        min_replica_count=1,
        max_replica_count=1,
    )
    print(f"[OK] Index deployed as {deployed_id}")
    print("Set env vars:")
    print(f"  VECTOR_SEARCH_INDEX_ENDPOINT={endpoint.resource_name}")
    print(f"  VECTOR_SEARCH_DEPLOYED_INDEX_ID={deployed_id}")

    meta.update(
        {
            "index": index.resource_name,
            "endpoint": endpoint.resource_name,
            "deployed_index_id": deployed_id,
        }
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create and deploy Matching Engine index")
    args = parser.parse_args()
    upload_embeddings(create_index=args.create)
