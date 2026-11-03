"""
Semantic product search.

Prefers a Vertex AI embedding index (products_vertex.index) when present so
Cloud Run does not need torch / sentence-transformers. Falls back to the
local MiniLM FAISS index (products_text.index / products_vector_map.json)
built by setup_vector_db.py. That local index is an IndexFlatL2 over
L2-normalized vectors, so the "similarity" value from search() is a squared
L2 distance: lower means a closer match. Returns [] if the index files are
missing.
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np

DB_DIR = os.path.dirname(os.path.dirname(__file__))
LOCAL_INDEX_PATH = os.path.join(DB_DIR, "products_text.index")
LOCAL_MAP_PATH = os.path.join(DB_DIR, "products_vector_map.json")
VERTEX_INDEX_PATH = os.path.join(DB_DIR, "products_vertex.index")
VERTEX_MAP_PATH = os.path.join(DB_DIR, "products_vertex_map.json")


class SemanticSearch:
    def __init__(self):
        self.backend = None  # "vertex" | "local"
        self.model = None
        self.index = None
        self.mapping = None

    def _load_resources(self):
        if self.index is not None and self.mapping is not None:
            return

        if os.path.exists(VERTEX_INDEX_PATH) and os.path.exists(VERTEX_MAP_PATH):
            self.index = faiss.read_index(VERTEX_INDEX_PATH)
            with open(VERTEX_MAP_PATH, "r", encoding="utf-8") as f:
                self.mapping = {int(k): v for k, v in json.load(f).items()}
            self.backend = "vertex"
            return

        if os.path.exists(LOCAL_INDEX_PATH) and os.path.exists(LOCAL_MAP_PATH):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.index = faiss.read_index(LOCAL_INDEX_PATH)
            with open(LOCAL_MAP_PATH, "r", encoding="utf-8") as f:
                self.mapping = {int(k): v for k, v in json.load(f).items()}
            self.backend = "local"

    def _encode(self, query: str) -> np.ndarray:
        if self.backend == "vertex":
            from recommenders.vertex_client import embed_texts

            vectors = embed_texts([query], task_type="RETRIEVAL_QUERY")
            query_vector = np.asarray(vectors, dtype="float32")
        else:
            query_vector = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vector)
        return query_vector

    def search(self, query: str, top_k: int = 10):
        self._load_resources()
        if not self.index or not self.mapping:
            return []

        query_vector = self._encode(query)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.mapping:
                results.append({
                    "product_id": self.mapping[idx],
                    "similarity": float(distances[0][i]),
                })
        return results


semantic_engine = SemanticSearch()
