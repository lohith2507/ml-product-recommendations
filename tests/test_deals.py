from recommenders import deals


def test_deal_metrics_ignore_prices_at_or_above_list():
    assert deals.deal_metrics({"price": 50.0, "list_price": 100.0}) == {"savings": 50.0, "savings_pct": 0.5}
    assert deals.deal_metrics({"price": 60.0, "list_price": 60.0}) == {"savings": 0.0, "savings_pct": 0.0}
    assert deals.deal_metrics({"price": 60.0, "list_price": 0.0}) == {"savings": 0.0, "savings_pct": 0.0}


def test_best_deals_prefers_a_well_rated_discount_over_slightly_more_relevant_full_price(retriever):
    results = deals.find_best_deals("running shoes", retriever=retriever)

    assert [p["product_id"] for p in results] == [1, 2, 3]
    assert results[0]["savings"] == 50.0


def test_search_ranks_by_relevance(retriever):
    results = deals.search_products("running shoes", retriever=retriever)

    assert [p["product_id"] for p in results] == [2, 1, 3]


def test_filters_apply(retriever):
    results = deals.find_best_deals("running shoes", max_price=55, min_rating=4.0, retriever=retriever)

    assert [p["product_id"] for p in results] == [1]


def test_falls_back_to_keyword_search_when_semantic_fails():
    def broken(query, top_k):
        raise RuntimeError("index missing")

    results = deals.search_products("backpack", retriever=broken)

    assert [p["product_id"] for p in results] == [4]
