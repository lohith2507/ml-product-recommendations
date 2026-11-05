"""
Streaming event worker.

Consumes Pub/Sub push payloads (or pull messages) and:
  - updates live product_facts for price/stock events
  - appends cart/agent events to BigQuery (best-effort)
  - marks catalog changes for re-embedding (logged for Vector Search upserts)

This Cloud Run / local worker is the Phase-3 streaming path. A Dataflow job
can replace the BigQuery sink later without changing publishers.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from typing import Any

from recommenders import facts


def _decode_pubsub_envelope(body: dict) -> dict:
    """Decode a Pub/Sub push request body into our event envelope."""
    message = body.get("message") or body
    data = message.get("data")
    if data:
        raw = base64.b64decode(data).decode("utf-8")
        return json.loads(raw)
    if "event_type" in body:
        return body
    raise ValueError("Unrecognized Pub/Sub payload")


def handle_event(envelope: dict[str, Any]) -> dict[str, Any]:
    event_type = envelope.get("event_type") or ""
    payload = envelope.get("payload") or {}

    if event_type == "price.changed":
        facts.upsert_price(
            int(payload["product_id"]),
            float(payload["price"]),
            list_price=payload.get("list_price"),
            warehouse_id=str(payload.get("warehouse_id") or "default"),
        )
        alerts = []
        try:
            from recommenders import deal_watch

            alerts = deal_watch.check_price_drop(int(payload["product_id"]), float(payload["price"]))
        except Exception as e:
            print(f"⚠️ deal watch check skipped: {e}")
        return {
            "handled": "price",
            "product_id": payload["product_id"],
            "alerts_fired": len(alerts),
        }

    if event_type == "stock.changed":
        facts.upsert_stock(
            int(payload["product_id"]),
            int(payload["quantity"]),
            warehouse_id=str(payload.get("warehouse_id") or "default"),
        )
        return {"handled": "stock", "product_id": payload["product_id"]}

    if event_type == "catalog.changed":
        # Vector Search / FAISS upsert hook — full re-embed is async Phase-3+.
        print(f"📦 Catalog change for product {payload.get('product_id')}: {payload.get('change')}")
        _maybe_write_bq("catalog_events", envelope)
        return {"handled": "catalog", "product_id": payload.get("product_id")}

    if event_type in ("cart.changed", "agent.turn"):
        table = "cart_events" if event_type == "cart.changed" else "agent_turns"
        _maybe_write_bq(table, envelope)
        return {"handled": event_type}

    return {"handled": "ignored", "event_type": event_type}


def _maybe_write_bq(table: str, envelope: dict):
    if os.environ.get("BQ_ENABLED", "1").lower() in ("0", "false", "no"):
        return
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        dataset = os.environ.get("BQ_DATASET", "shopping_agent")
        table_id = f"{client.project}.{dataset}.{table}"
        row = {
            "event_id": envelope.get("event_id"),
            "event_type": envelope.get("event_type"),
            "timestamp": envelope.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "payload": json.dumps(envelope.get("payload") or {}),
        }
        errors = client.insert_rows_json(table_id, [row])
        if errors:
            print(f"⚠️ BigQuery insert errors on {table}: {errors}")
    except Exception as e:
        print(f"⚠️ BigQuery write skipped ({table}): {e}")


def handle_push_request(body: dict) -> dict:
    envelope = _decode_pubsub_envelope(body)
    return handle_event(envelope)
