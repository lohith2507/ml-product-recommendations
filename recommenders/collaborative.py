"""
Collaborative Filtering Recommender using Surprise SVD.

Uses Singular Value Decomposition (SVD) for matrix factorization
to predict user-item ratings and generate top-N recommendations.
"""

import os
import pickle
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, SVDpp, KNNBaseline
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import accuracy

from database.models import get_db, dict_from_row

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cf_model.pkl")
TRAINSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cf_trainset.pkl")


class CollaborativeFilteringRecommender:
    """SVD-based collaborative filtering recommendation engine."""

    def __init__(self):
        self.model = None
        self.trainset = None
        self.all_product_ids = set()
        self.popular_products = []

    def load_data(self):
        """Load user-item-rating triplets from the database."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT user_id, product_id, rating FROM reviews"
            ).fetchall()

            # Get all product IDs for recommendation coverage
            products = conn.execute("SELECT product_id FROM products").fetchall()
            self.all_product_ids = {r["product_id"] for r in products}

            # Get popular products (fallback for cold-start)
            popular = conn.execute("""
                SELECT product_id, AVG(rating) as avg_rating, COUNT(*) as num_ratings
                FROM reviews
                GROUP BY product_id
                HAVING num_ratings >= 3
                ORDER BY avg_rating DESC, num_ratings DESC
                LIMIT 50
            """).fetchall()
            self.popular_products = [dict_from_row(r) for r in popular]

        # Create DataFrame
        df = pd.DataFrame([dict(r) for r in rows], columns=["user_id", "product_id", "rating"])

        print(f"📊 Loaded {len(df)} ratings from {df['user_id'].nunique()} users "
              f"on {df['product_id'].nunique()} products")

        return df

    def train(self, tune_hyperparams=True):
        """
        Train the SVD model with optional hyperparameter tuning.

        Args:
            tune_hyperparams: If True, uses GridSearchCV for optimal params.
        """
        df = self.load_data()

        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(df[["user_id", "product_id", "rating"]], reader)

        if tune_hyperparams:
            print("\n🔍 Running hyperparameter tuning with GridSearchCV...")
            param_grid = {
                "n_factors": [50, 100],
                "n_epochs": [20, 30],
                "lr_all": [0.005, 0.01],
                "reg_all": [0.02, 0.1],
            }

            gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
            gs.fit(data)

            print(f"   Best RMSE: {gs.best_score['rmse']:.4f}")
            print(f"   Best MAE:  {gs.best_score['mae']:.4f}")
            print(f"   Best params: {gs.best_params['rmse']}")

            self.model = gs.best_estimator["rmse"]
        else:
            print("\n🏗️ Training SVD model with default parameters...")
            self.model = SVD(n_factors=100, n_epochs=30, lr_all=0.005, reg_all=0.02)

        # Train on the full dataset
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

        # Cross-validate for final metrics
        print("\n📈 Cross-validation results:")
        cv_results = cross_validate(self.model, data, measures=["rmse", "mae"], cv=5, verbose=False)
        print(f"   RMSE: {cv_results['test_rmse'].mean():.4f} (+/- {cv_results['test_rmse'].std():.4f})")
        print(f"   MAE:  {cv_results['test_mae'].mean():.4f} (+/- {cv_results['test_mae'].std():.4f})")

        return cv_results

    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save. Call train() first.")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(TRAINSET_PATH, "wb") as f:
            pickle.dump({
                "trainset": self.trainset,
                "all_product_ids": self.all_product_ids,
                "popular_products": self.popular_products,
            }, f)

        print(f"💾 Model saved to {MODEL_PATH}")

    def load_model(self):
        """Load a pre-trained model from disk."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train() first.")

        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(TRAINSET_PATH, "rb") as f:
            data = pickle.load(f)
            self.trainset = data["trainset"]
            self.all_product_ids = data["all_product_ids"]
            self.popular_products = data["popular_products"]

        print(f"✅ Model loaded from {MODEL_PATH}")

    def get_user_rated_products(self, user_id):
        """Get the set of product IDs already rated by the user."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT product_id FROM reviews WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {r["product_id"] for r in rows}

    def recommend(self, user_id, n=10):
        """
        Generate top-N recommendations for a user.

        Args:
            user_id: The target user ID.
            n: Number of recommendations to return.

        Returns:
            List of dicts with product_id, predicted_rating, and product details.
        """
        if self.model is None:
            self.load_model()

        # Check if user exists in training data
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            is_cold_start = False
        except ValueError:
            is_cold_start = True

        if is_cold_start:
            # Cold-start fallback: return popular products
            return self._get_popular_recommendations(n)

        # Get products already rated by the user
        rated_products = self.get_user_rated_products(user_id)

        # Predict ratings for all unrated products
        predictions = []
        for pid in self.all_product_ids:
            if pid not in rated_products:
                pred = self.model.predict(user_id, pid)
                predictions.append({
                    "product_id": pid,
                    "predicted_rating": round(pred.est, 3),
                })

        # Sort by predicted rating
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        top_n = predictions[:n]

        # Enrich with product details
        return self._enrich_recommendations(top_n, algorithm="collaborative")

    def _get_popular_recommendations(self, n=10):
        """Fallback recommendations for cold-start users."""
        recommendations = []
        for p in self.popular_products[:n]:
            recommendations.append({
                "product_id": p["product_id"],
                "predicted_rating": p["avg_rating"],
            })
        return self._enrich_recommendations(recommendations, algorithm="collaborative_popular")

    def _enrich_recommendations(self, predictions, algorithm="collaborative"):
        """Add product details to prediction results."""
        if not predictions:
            return []

        product_ids = [p["product_id"] for p in predictions]
        placeholders = ",".join(["?" for _ in product_ids])

        with get_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM products WHERE product_id IN ({placeholders})",
                product_ids
            ).fetchall()

        product_map = {dict_from_row(r)["product_id"]: dict_from_row(r) for r in rows}

        enriched = []
        for pred in predictions:
            product = product_map.get(pred["product_id"], {})
            enriched.append({
                **product,
                "predicted_rating": pred["predicted_rating"],
                "algorithm": algorithm,
            })

        return enriched

    def get_similar_items(self, product_id, n=10):
        """
        Find products similar to a given product based on user co-ratings.
        Uses the "also bought" pattern.
        """
        with get_db() as conn:
            # Find users who rated this product highly (>= 4.0)
            users = conn.execute(
                "SELECT user_id FROM reviews WHERE product_id = ? AND rating >= 4.0",
                (product_id,)
            ).fetchall()

            if not users:
                return self._get_popular_recommendations(n)

            user_ids = [u["user_id"] for u in users]
            placeholders = ",".join(["?" for _ in user_ids])

            # Find other products these users also rated highly
            rows = conn.execute(f"""
                SELECT r.product_id, AVG(r.rating) as avg_rating,
                       COUNT(DISTINCT r.user_id) as co_raters,
                       p.title, p.category, p.price, p.avg_rating as product_avg,
                       p.rating_count, p.image_url, p.description, p.features, p.asin
                FROM reviews r
                JOIN products p ON r.product_id = p.product_id
                WHERE r.user_id IN ({placeholders})
                  AND r.product_id != ?
                  AND r.rating >= 3.5
                GROUP BY r.product_id
                HAVING co_raters >= 2
                ORDER BY co_raters DESC, avg_rating DESC
                LIMIT ?
            """, (*user_ids, product_id, n)).fetchall()

        return [{
            **dict_from_row(r),
            "predicted_rating": round(r["avg_rating"], 2),
            "co_purchase_count": r["co_raters"],
            "algorithm": "also_bought",
        } for r in rows]


if __name__ == "__main__":
    recommender = CollaborativeFilteringRecommender()
    cv_results = recommender.train(tune_hyperparams=True)
    recommender.save_model()

    # Test recommendations
    print("\n🎯 Sample recommendations for User 1:")
    recs = recommender.recommend(user_id=1, n=5)
    for r in recs:
        print(f"   [{r.get('predicted_rating', 'N/A')}] {r.get('title', 'Unknown')}")
