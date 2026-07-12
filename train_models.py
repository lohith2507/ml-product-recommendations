"""
Model Training Orchestrator.

Trains both Collaborative Filtering (SVD) and Content-Based (TF-IDF)
models, saves them to disk, and runs A/B test simulation.
"""

import time
import sys

def main():
    print("=" * 60)
    print("🚀 AI Recommendation System — Model Training")
    print("=" * 60)

    # Step 1: Ensure database is seeded
    print("\n📋 Step 1: Checking database...")
    from database.models import get_db
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    if count == 0:
        print("   Database is empty. Running data setup...")
        from setup_data import seed_database
        seed_database()
    else:
        print(f"   ✅ Database has {count} reviews. Skipping seed.")

    # Step 2: Train Collaborative Filtering model
    print("\n📋 Step 2: Training Collaborative Filtering (SVD) model...")
    print("-" * 40)
    start = time.time()

    from recommenders.collaborative import CollaborativeFilteringRecommender
    cf = CollaborativeFilteringRecommender()
    cf_results = cf.train(tune_hyperparams=True)
    cf.save_model()

    cf_time = time.time() - start
    print(f"   ⏱️ CF training time: {cf_time:.1f}s")

    # Step 3: Train Content-Based model
    print("\n📋 Step 3: Training Content-Based (TF-IDF) model...")
    print("-" * 40)
    start = time.time()

    from recommenders.content_based import ContentBasedRecommender
    cb = ContentBasedRecommender()
    cb.train()
    cb.save_model()

    cb_time = time.time() - start
    print(f"   ⏱️ CB training time: {cb_time:.1f}s")

    # Step 4: Simulate A/B testing
    print("\n📋 Step 4: Running A/B test simulation...")
    print("-" * 40)

    from recommenders.ab_testing import ABTestingEngine
    ab = ABTestingEngine()
    ab.simulate_ab_test(num_sessions=500)

    results = ab.get_results()
    print("\n   📊 A/B Test Results Summary:")
    if "groups" in results:
        for g in results["groups"]:
            print(f"      {g['algorithm_group']:20s} — CTR: {g['avg_ctr']:.2f}% "
                  f"({g['num_sessions']} sessions, {g['total_clicks']} clicks)")
        print(f"\n   🏆 Winner: {results.get('winner', 'N/A')}")
        sig = results.get("significance", {})
        print(f"   📈 Significant: {sig.get('significant', 'N/A')} (p={sig.get('p_value', 'N/A')})")

    # Step 5: Quick validation
    print("\n📋 Step 5: Validation — Sample Recommendations")
    print("-" * 40)

    print("\n   🔗 Collaborative Filtering — User 1:")
    cf_recs = cf.recommend(user_id=1, n=3)
    for r in cf_recs:
        print(f"      ⭐ {r.get('predicted_rating', 'N/A'):.2f} — {r.get('title', 'Unknown')[:50]}")

    print("\n   📝 Content-Based — User 1:")
    cb_recs = cb.recommend(user_id=1, n=3)
    for r in cb_recs:
        print(f"      🔗 {r.get('similarity_score', 'N/A')} — {r.get('title', 'Unknown')[:50]}")

    print("\n   🛒 Also Bought — Product 1:")
    also_bought = cf.get_similar_items(product_id=1, n=3)
    for r in also_bought:
        print(f"      🤝 {r.get('co_purchase_count', 'N/A')} co-purchasers — {r.get('title', 'Unknown')[:50]}")

    print("\n" + "=" * 60)
    print("✅ All models trained and validated successfully!")
    print(f"   Total training time: {cf_time + cb_time:.1f}s")
    print("   Run 'uvicorn app:app --reload' to start the server")
    print("=" * 60)


if __name__ == "__main__":
    main()
