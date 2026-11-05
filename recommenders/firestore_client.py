"""
Firestore client for the shopping agent hot path.

Enabled when FIRESTORE_ENABLED is truthy. Catalog data stays in SQLite;
carts, facts, cart actions, and agent threads live here under traffic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from recommenders.vertex_client import project_id


def firestore_enabled() -> bool:
    return os.environ.get("FIRESTORE_ENABLED", "0").lower() not in ("0", "false", "no")


@lru_cache(maxsize=1)
def get_firestore_client():
    from google.cloud import firestore

    database = os.environ.get("FIRESTORE_DATABASE", "(default)")
    return firestore.Client(project=project_id(), database=database)


def collection(name: str):
    return get_firestore_client().collection(name)
