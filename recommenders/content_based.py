"""
Content-Based Filtering Recommender using TF-IDF.

Uses TF-IDF vectorization of product text (title + description + category)
and cosine similarity to find similar products and recommend items
based on a user's rating history.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database.models import get_db, dict_from_row

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cb_model.pkl")


class ContentBasedRecommender:
    """TF-IDF content-based filtering recommendation engine."""

    def __init__(self):
        self.tfidf_matrix = None
        self.vectorizer = None
        self.product_ids = []
        self.product_index_map = {}
        self.similarity_matrix = None

    def load_products(self):
        """Load all products from the database."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT product_id, title, category, description, features FROM products"
            ).fetchall()
        return [dict_from_row(r) for r in rows]

    def build_content_profile(self, product):
        """
        Build a text profile for a product by combining its textual attributes.
        Weights: category (3x), title (2x), description (1x), features (1x)
        """
        category = (product.get("category", "") + " ") * 3
        title = (product.get("title", "") + " ") * 2
        description = product.get("description", "")
        features = product.get("features", [])

        if isinstance(features, list):
            features_text = " ".join(features)
        else:
            features_text = str(features)

        return f"{category} {title} {description} {features_text}".strip()

    def train(self):
        """Build TF-IDF matrix and precompute similarity."""
        products = self.load_products()

        if not products:
            raise ValueError("No products found in the database. Run setup_data.py first.")

        print(f"📊 Building content profiles for {len(products)} products...")

        self.product_ids = [p["product_id"] for p in products]
        self.product_index_map = {pid: idx for idx, pid in enumerate(self.product_ids)}

        # Build text corpus
        corpus = [self.build_content_profile(p) for p in products]

        # TF-IDF Vectorization
        print("🔧 Computing TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Precompute cosine similarity matrix
        print("🔧 Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        print(f"✅ Content-based model built: TF-IDF matrix shape = {self.tfidf_matrix.shape}")

        return self.tfidf_matrix.shape

    def save_model(self):
        """Save the TF-IDF model and similarity matrix."""
        if self.tfidf_matrix is None:
            raise ValueError("No trained model. Call train() first.")

        model_data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "product_ids": self.product_ids,
            "product_index_map": self.product_index_map,
            "similarity_matrix": self.similarity_matrix,
        }

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)

        print(f"💾 Content-based model saved to {MODEL_PATH}")

    def load_model(self):
        """Load a pre-trained model from disk."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train() first.")

        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.product_ids = data["product_ids"]
        self.product_index_map = data["product_index_map"]
        self.similarity_matrix = data["similarity_matrix"]

        print(f"✅ Content-based model loaded from {MODEL_PATH}")

    def get_similar_products(self, product_id, n=10):
        """
        Find products most similar to a given product based on content.

        Args:
            product_id: The reference product ID.
            n: Number of similar products to return.

        Returns:
            List of dicts with product details and similarity scores.
        """
        if self.similarity_matrix is None:
            self.load_model()

        if product_id not in self.product_index_map:
            return []

        idx = self.product_index_map[product_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity, exclude self
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:n]

        # Get product IDs and similarity scores
        similar_pids = [(self.product_ids[i], round(float(s), 4)) for i, s in sim_scores]

        return self._enrich_results(similar_pids, algorithm="content_similar")

    def recommend(self, user_id, n=10):
        """
        Recommend products for a user based on their highly-rated items.

        Strategy:
        1. Find all products the user rated >= 4.0
        2. For each liked product, find similar items
        3. Aggregate similarity scores across all liked products
        4. Return top-N items the user hasn't seen

        Args:
            user_id: Target user ID.
            n: Number of recommendations.

        Returns:
            List of recommendation dicts with product details.
        """
        if self.similarity_matrix is None:
            self.load_model()

        # Get user's highly-rated products
        with get_db() as conn:
            liked = conn.execute(
                "SELECT product_id, rating FROM reviews WHERE user_id = ? AND rating >= 4.0 ORDER BY rating DESC",
                (user_id,)
            ).fetchall()

            all_rated = conn.execute(
                "SELECT product_id FROM reviews WHERE user_id = ?",
                (user_id,)
            ).fetchall()

        rated_ids = {r["product_id"] for r in all_rated}

        if not liked:
            # Fallback: recommend popular products from diverse categories
            return self._get_diverse_popular(n)

        # Aggregate similarity scores across all liked products
        candidate_scores = {}

        for item in liked:
            pid = item["product_id"]
            rating = item["rating"]

            if pid not in self.product_index_map:
                continue

            idx = self.product_index_map[pid]
            sim_scores = self.similarity_matrix[idx]

            for j, score in enumerate(sim_scores):
                candidate_pid = self.product_ids[j]
                if candidate_pid in rated_ids or candidate_pid == pid:
                    continue

                # Weight by user's rating and content similarity
                weighted_score = float(score) * (rating / 5.0)

                if candidate_pid not in candidate_scores:
                    candidate_scores[candidate_pid] = 0.0
                candidate_scores[candidate_pid] += weighted_score

        # Sort by aggregate score
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )[:n]

        results = [(pid, round(score, 4)) for pid, score in sorted_candidates]
        return self._enrich_results(results, algorithm="content_based")

    def _get_diverse_popular(self, n=10):
        """Fallback: popular products from diverse categories."""
        with get_db() as conn:
            rows = conn.execute("""
                SELECT p.*, AVG(r.rating) as computed_rating
                FROM products p
                JOIN reviews r ON p.product_id = r.product_id
                GROUP BY p.product_id
                HAVING COUNT(r.review_id) >= 3
                ORDER BY computed_rating DESC
                LIMIT ?
            """, (n,)).fetchall()

        return [{
            **dict_from_row(r),
            "similarity_score": round(r["computed_rating"] / 5.0, 4),
            "predicted_rating": round(r["computed_rating"], 2),
            "algorithm": "content_popular",
        } for r in rows]

    def _enrich_results(self, pid_score_pairs, algorithm="content_based"):
        """Add product details to results."""
        if not pid_score_pairs:
            return []

        pids = [p[0] for p in pid_score_pairs]
        score_map = {p[0]: p[1] for p in pid_score_pairs}
        placeholders = ",".join(["?" for _ in pids])

        with get_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM products WHERE product_id IN ({placeholders})",
                pids
            ).fetchall()

        product_map = {dict_from_row(r)["product_id"]: dict_from_row(r) for r in rows}

        enriched = []
        for pid, score in pid_score_pairs:
            product = product_map.get(pid, {})
            enriched.append({
                **product,
                "similarity_score": score,
                "predicted_rating": round(score * 5.0, 2) if score <= 1.0 else score,
                "algorithm": algorithm,
            })

        return enriched


if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    shape = recommender.train()
    recommender.save_model()

    # Test similar products
    print("\n🔗 Products similar to Product 1:")
    similar = recommender.get_similar_products(product_id=1, n=5)
    for s in similar:
        print(f"   [{s.get('similarity_score', 'N/A')}] {s.get('title', 'Unknown')}")

    # Test user recommendations
    print("\n🎯 Content-based recommendations for User 1:")
    recs = recommender.recommend(user_id=1, n=5)
    for r in recs:
        print(f"   [{r.get('similarity_score', 'N/A')}] {r.get('title', 'Unknown')}")
