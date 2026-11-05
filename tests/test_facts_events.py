import base64
import json
import os

from recommenders import deals, facts
from workers.event_worker import handle_event, handle_push_request


def test_live_price_overlays_deal_ranking(retriever, monkeypatch):
    monkeypatch.setenv("PUBSUB_ENABLED", "0")
    facts.upsert_price(1, price=10.0, list_price=100.0)

    results = deals.find_best_deals("running shoes", retriever=retriever)

    assert results[0]["product_id"] == 1
    assert results[0]["price"] == 10.0
    assert results[0]["savings"] == 90.0


def test_out_of_stock_products_are_filtered(retriever, monkeypatch):
    monkeypatch.setenv("PUBSUB_ENABLED", "0")
    facts.upsert_stock(1, 0)
    facts.upsert_stock(2, 5)
    facts.upsert_stock(3, 5)

    results = deals.find_best_deals("running shoes", retriever=retriever)

    assert 1 not in [p["product_id"] for p in results]
    assert {p["product_id"] for p in results} <= {2, 3}


def test_price_event_updates_facts(monkeypatch):
    monkeypatch.setenv("BQ_ENABLED", "0")
    result = handle_event(
        {
            "event_type": "price.changed",
            "payload": {"product_id": 4, "price": 19.99, "list_price": 40.0},
        }
    )
    assert result["handled"] == "price"
    loaded = facts.get_facts([4])
    assert loaded[4]["price"] == 19.99


def test_pubsub_push_decodes_base64(monkeypatch):
    monkeypatch.setenv("BQ_ENABLED", "0")
    envelope = {
        "event_id": "abc",
        "event_type": "stock.changed",
        "payload": {"product_id": 4, "quantity": 3},
    }
    body = {
        "message": {
            "data": base64.b64encode(json.dumps(envelope).encode()).decode(),
            "messageId": "1",
        }
    }
    result = handle_push_request(body)
    assert result["handled"] == "stock"
    assert facts.get_facts([4])[4]["stock_qty"] == 3


def test_cart_publish_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("PUBSUB_ENABLED", "0")
    from recommenders import cart as cart_service

    cart = cart_service.update_cart(1, 1, "add", 2)
    assert cart["item_count"] == 2
