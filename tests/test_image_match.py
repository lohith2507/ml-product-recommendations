from recommenders.image_match import is_same_kind, products_for_caption, search_tokens


def test_watch_caption_does_not_keep_shoes():
    tokens = search_tokens("men's silver wristwatch")
    assert "wristwatch" in tokens or "watch" in tokens
    watch = {"product_id": 1, "title": "Casio Men's Wrist Watch", "category": "Watches"}
    shoe = {"product_id": 2, "title": "Men's Encore Gust Slip-On Shoe", "category": "Shoes"}
    assert is_same_kind(watch, tokens)
    assert not is_same_kind(shoe, tokens)


def test_catalog_search_for_watch_stays_on_watches():
    products = products_for_caption("wristwatch", limit=5, retriever=lambda q, top_k: [])
    assert all(is_same_kind(p, search_tokens("wristwatch")) for p in products)
