"""
Pub/Sub event publisher for the shopping agent.

Publishes fire-and-forget JSON events. When Pub/Sub is unavailable (local
dev without ADC, tests), events are logged and dropped so the request path
never fails.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from recommenders.vertex_client import project_id

TOPICS = {
    "catalog": "catalog.product-changed",
    "pricing": "pricing.price-changed",
    "inventory": "inventory.stock-changed",
    "cart": "commerce.cart-event",
    "agent": "agent.turn-logged",
}


def events_enabled() -> bool:
    return os.environ.get("PUBSUB_ENABLED", "1").lower() not in ("0", "false", "no")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventPublisher:
    def __init__(self, project: Optional[str] = None):
        self.project = project or project_id()
        self._publisher = None
        self._lock = threading.Lock()
        self._topic_paths: dict[str, str] = {}

    def _client(self):
        if self._publisher is None:
            from google.cloud import pubsub_v1

            self._publisher = pubsub_v1.PublisherClient()
        return self._publisher

    def _topic_path(self, key: str) -> str:
        if key not in self._topic_paths:
            topic = TOPICS[key]
            self._topic_paths[key] = self._client().topic_path(self.project, topic)
        return self._topic_paths[key]

    def publish(self, key: str, event_type: str, payload: dict[str, Any], *, sync: bool = False) -> Optional[str]:
        """Publish an event. Returns message id when sync=True and publish succeeds."""
        if not events_enabled():
            return None

        envelope = {
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "timestamp": _now(),
            "source": "shopping-agent",
            "payload": payload,
        }
        data = json.dumps(envelope, default=str).encode("utf-8")

        try:
            future = self._client().publish(
                self._topic_path(key),
                data,
                event_type=event_type,
                source="shopping-agent",
            )
            if sync:
                return future.result(timeout=10)
            # Keep a callback so errors surface in logs without blocking the request.
            future.add_done_callback(lambda f: f.exception())
            return None
        except Exception as e:
            print(f"⚠️ Pub/Sub publish failed ({key}/{event_type}): {e}")
            return None

    def cart_event(self, user_id: int, action: str, product_id: int, quantity: int, **extra):
        return self.publish(
            "cart",
            "cart.changed",
            {"user_id": user_id, "action": action, "product_id": product_id, "quantity": quantity, **extra},
        )

    def agent_turn(self, user_id: int, message: str, tools_used: list, pending_count: int, orchestrator: str = ""):
        return self.publish(
            "agent",
            "agent.turn",
            {
                "user_id": user_id,
                "message": (message or "")[:500],
                "tools_used": tools_used,
                "pending_count": pending_count,
                "orchestrator": orchestrator,
            },
        )

    def price_changed(self, product_id: int, price: float, list_price: float | None = None, warehouse_id: str = "default"):
        return self.publish(
            "pricing",
            "price.changed",
            {"product_id": product_id, "price": price, "list_price": list_price, "warehouse_id": warehouse_id},
            sync=True,
        )

    def stock_changed(self, product_id: int, quantity: int, warehouse_id: str = "default"):
        return self.publish(
            "inventory",
            "stock.changed",
            {"product_id": product_id, "quantity": quantity, "warehouse_id": warehouse_id},
            sync=True,
        )

    def catalog_changed(self, product_id: int, change: str = "updated"):
        return self.publish(
            "catalog",
            "catalog.changed",
            {"product_id": product_id, "change": change},
            sync=True,
        )


_publisher: Optional[EventPublisher] = None


def get_publisher() -> EventPublisher:
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher
