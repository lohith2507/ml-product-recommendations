"""
A/B Testing Engine for Recommendation Algorithms.

Simulates A/B testing by randomly assigning users to algorithm groups,
tracking impressions and clicks, and computing CTR with statistical
significance testing.
"""

import random
import uuid
import json
from datetime import datetime, timedelta
from scipy import stats

from database.models import get_db, dict_from_row

# Algorithm groups for A/B testing
ALGORITHM_GROUPS = ["collaborative", "content_based", "hybrid", "popular"]


class ABTestingEngine:
    """A/B testing engine for comparing recommendation algorithms."""

    def __init__(self):
        self.experiment_name = "recommendation_algorithm_comparison"

    def assign_user_group(self, user_id):
        """
        Assign a user to an algorithm group deterministically.
        Uses hash-based assignment for consistent group membership.
        """
        # Deterministic assignment based on user_id
        group_index = user_id % len(ALGORITHM_GROUPS)
        return ALGORITHM_GROUPS[group_index]

    def record_impression(self, user_id, algorithm, num_items=10):
        """Record that recommendations were shown to a user."""
        session_id = str(uuid.uuid4())[:8]

        with get_db() as conn:
            conn.execute("""
                INSERT INTO ab_experiments (user_id, algorithm_group, session_id, impressions, clicks, ctr)
                VALUES (?, ?, ?, ?, 0, 0.0)
            """, (user_id, algorithm, session_id, num_items))

        return session_id

    def record_click(self, user_id, product_id, algorithm):
        """Record a click on a recommended product."""
        with get_db() as conn:
            # Update the recommendation record
            conn.execute("""
                UPDATE recommendations SET clicked = 1
                WHERE user_id = ? AND product_id = ? AND algorithm = ? AND clicked = 0
            """, (user_id, product_id, algorithm))

            # Update the latest A/B experiment entry
            conn.execute("""
                UPDATE ab_experiments
                SET clicks = clicks + 1,
                    ctr = CAST(clicks + 1 AS REAL) / CASE WHEN impressions > 0 THEN impressions ELSE 1 END
                WHERE experiment_id = (
                    SELECT experiment_id FROM ab_experiments
                    WHERE user_id = ? AND algorithm_group = ?
                    ORDER BY timestamp DESC LIMIT 1
                )
            """, (user_id, algorithm))

    def simulate_ab_test(self, num_sessions=500):
        """
        Simulate an A/B test with synthetic click data.

        Different algorithms have different simulated CTR:
        - collaborative: 12-18% CTR
        - content_based: 10-15% CTR
        - hybrid: 18-25% CTR (best - the LLM re-ranking helps)
        - popular: 5-10% CTR (baseline)
        """
        ctr_ranges = {
            "collaborative": (0.12, 0.18),
            "content_based": (0.10, 0.15),
            "hybrid": (0.18, 0.25),
            "popular": (0.05, 0.10),
        }

        with get_db() as conn:
            users = conn.execute("SELECT user_id FROM users").fetchall()

        if not users:
            print("⚠️ No users found. Run setup_data.py first.")
            return

        user_ids = [u["user_id"] for u in users]

        print(f"🧪 Simulating A/B test with {num_sessions} sessions...")

        with get_db() as conn:
            for i in range(num_sessions):
                user_id = random.choice(user_ids)
                algorithm = self.assign_user_group(user_id)
                impressions = random.randint(5, 15)

                # Simulate clicks based on expected CTR
                ctr_low, ctr_high = ctr_ranges[algorithm]
                expected_ctr = random.uniform(ctr_low, ctr_high)
                clicks = sum(1 for _ in range(impressions) if random.random() < expected_ctr)
                actual_ctr = clicks / impressions if impressions > 0 else 0.0

                session_id = str(uuid.uuid4())[:8]

                # Add some time variation
                days_ago = random.randint(0, 30)
                timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()

                conn.execute("""
                    INSERT INTO ab_experiments
                    (user_id, algorithm_group, session_id, impressions, clicks, ctr, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, algorithm, session_id, impressions, clicks, actual_ctr, timestamp))

        print(f"✅ Simulated {num_sessions} A/B test sessions.")

    def get_results(self):
        """
        Compute A/B test results with statistical significance.

        Returns:
            Dict with per-algorithm metrics and statistical test results.
        """
        with get_db() as conn:
            results = conn.execute("""
                SELECT
                    algorithm_group,
                    COUNT(*) as num_sessions,
                    SUM(impressions) as total_impressions,
                    SUM(clicks) as total_clicks,
                    AVG(ctr) as avg_ctr,
                    MIN(ctr) as min_ctr,
                    MAX(ctr) as max_ctr
                FROM ab_experiments
                GROUP BY algorithm_group
                ORDER BY avg_ctr DESC
            """).fetchall()

        if not results:
            return {"error": "No A/B test data found. Run simulate first."}

        groups = []
        for r in results:
            row = dict(r)
            row["avg_ctr"] = round(row["avg_ctr"] * 100, 2)
            row["min_ctr"] = round(row["min_ctr"] * 100, 2)
            row["max_ctr"] = round(row["max_ctr"] * 100, 2)
            row["total_ctr"] = round(
                (row["total_clicks"] / row["total_impressions"] * 100)
                if row["total_impressions"] > 0 else 0.0, 2
            )
            groups.append(row)

        # Statistical significance: chi-squared test between best and second-best
        significance = self._compute_significance()

        return {
            "groups": groups,
            "winner": groups[0]["algorithm_group"] if groups else None,
            "significance": significance,
        }

    def _compute_significance(self):
        """Run chi-squared test between algorithm groups."""
        with get_db() as conn:
            groups = conn.execute("""
                SELECT algorithm_group, SUM(clicks) as clicks,
                       SUM(impressions) - SUM(clicks) as non_clicks
                FROM ab_experiments
                GROUP BY algorithm_group
                HAVING SUM(impressions) > 0
            """).fetchall()

        if len(groups) < 2:
            return {"significant": False, "message": "Not enough data for significance testing."}

        # Build contingency table
        observed = []
        labels = []
        for g in groups:
            observed.append([g["clicks"], g["non_clicks"]])
            labels.append(g["algorithm_group"])

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)

            return {
                "significant": bool(p_value < 0.05),
                "p_value": round(float(p_value), 6),
                "chi2_statistic": round(float(chi2), 4),
                "degrees_of_freedom": int(dof),
                "message": (
                    f"Results are statistically significant (p={p_value:.4f} < 0.05). "
                    f"The differences between algorithms are unlikely due to chance."
                    if p_value < 0.05
                    else f"Results are NOT statistically significant (p={p_value:.4f} >= 0.05). "
                    f"More data may be needed to draw conclusions."
                ),
            }
        except Exception as e:
            return {"significant": False, "message": f"Error computing significance: {str(e)}"}

    def get_ctr_over_time(self, days=30):
        """Get CTR trends over time for each algorithm group."""
        with get_db() as conn:
            rows = conn.execute("""
                SELECT
                    algorithm_group,
                    DATE(timestamp) as date,
                    AVG(ctr) as avg_ctr,
                    SUM(impressions) as impressions,
                    SUM(clicks) as clicks
                FROM ab_experiments
                WHERE timestamp >= datetime('now', ?)
                GROUP BY algorithm_group, DATE(timestamp)
                ORDER BY date
            """, (f'-{days} days',)).fetchall()

        # Organize by algorithm
        timeline = {}
        for r in rows:
            algo = r["algorithm_group"]
            if algo not in timeline:
                timeline[algo] = []
            timeline[algo].append({
                "date": r["date"],
                "ctr": round(r["avg_ctr"] * 100, 2),
                "impressions": r["impressions"],
                "clicks": r["clicks"],
            })

        return timeline


if __name__ == "__main__":
    engine = ABTestingEngine()

    # Run simulation
    engine.simulate_ab_test(num_sessions=500)

    # Get results
    results = engine.get_results()
    print("\n📊 A/B Test Results:")
    print(json.dumps(results, indent=2))

    # Get timeline
    timeline = engine.get_ctr_over_time()
    print(f"\n📈 Timeline data for {len(timeline)} algorithm groups")
