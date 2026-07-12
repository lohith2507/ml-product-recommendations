"""
Amazon Dataset Loader & Database Seeder.

Loads real Amazon product data from the dataset/ folder,
selects a strategic subset, generates synthetic users and reviews,
and seeds the SQLite database.
"""

import os
import random
import json
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
PRODUCTS_CSV = os.path.join(DATASET_DIR, "amazon_products.csv")
CATEGORIES_CSV = os.path.join(DATASET_DIR, "amazon_categories.csv")

# Number of products to load per category (top categories)
PRODUCTS_PER_CATEGORY = 250
TOP_N_CATEGORIES = 20
NUM_USERS = 200
AVG_REVIEWS_PER_USER = 25

# Review text templates
POSITIVE_REVIEWS = [
    "Absolutely love this product! It exceeded all my expectations. The quality is outstanding.",
    "Best purchase I've made this year. Incredibly well-made and performs flawlessly.",
    "Five stars without hesitation! Exactly what I was looking for. Highly recommend.",
    "Impressed by the quality and attention to detail. Even better than described.",
    "Outstanding value for the price. Has become an essential part of my daily routine.",
    "Can't say enough good things. Beautifully designed and works like a charm.",
    "This is a game-changer! I've tried many alternatives, but nothing comes close.",
    "Perfect in every way. Well-built, looks great, and performs exceptionally well.",
]

NEUTRAL_REVIEWS = [
    "Decent for the price. Does what it's supposed to do, though nothing exceptional.",
    "Good product overall. Meets expectations but there's room for improvement.",
    "It's okay. Works as advertised but I expected a bit more for this price point.",
    "Solid product with some nice features but also a few minor drawbacks.",
    "Average quality. Functional but doesn't stand out compared to similar products.",
]

NEGATIVE_REVIEWS = [
    "Disappointed. The quality doesn't match the description and it feels cheaply made.",
    "Not what I expected. Has some serious design flaws that affect everyday usage.",
    "Would not recommend. Stopped working after just a few weeks of normal use.",
    "Below average. Looks nothing like the pictures and the performance is underwhelming.",
]

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Skyler", "Dakota", "Reese", "Finley", "Harper", "Emerson", "Sage", "Blake",
    "Cameron", "Drew", "Jamie", "Kendall", "Logan", "Parker", "Rowan", "Hayden",
    "Liam", "Emma", "Noah", "Olivia", "James", "Sophia", "Lucas", "Isabella",
    "Mason", "Mia", "Ethan", "Charlotte", "Aiden", "Amelia", "Oliver", "Ella",
    "Elijah", "Ava", "Benjamin", "Luna", "Jack", "Chloe", "Henry", "Lily",
    "Phoenix", "Remy", "Shiloh", "Tatum", "Val", "Wren", "Zion", "Kit",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen",
    "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Green", "Adams",
    "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Chen",
]


def load_amazon_data():
    """Load and prepare Amazon product data from CSV files."""
    print("[1/6] Loading Amazon dataset...")

    # Load categories
    categories_df = pd.read_csv(CATEGORIES_CSV)
    cat_map = dict(zip(categories_df['id'], categories_df['category_name']))
    print(f"  Loaded {len(cat_map)} categories")

    # Load products in chunks (file is ~376MB)
    print("  Reading products CSV (this may take a moment)...")
    chunks = []
    for chunk in pd.read_csv(PRODUCTS_CSV, chunksize=200000):
        # Filter: must have price > 0 and a valid title
        chunk = chunk[(chunk['price'] > 0) & (chunk['title'].str.len() > 5)]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} products with valid prices")

    # Map category IDs to names
    df['category'] = df['category_id'].map(cat_map)
    df = df.dropna(subset=['category'])

    return df, cat_map


def select_product_subset(df):
    """Select a strategic subset of products across top categories."""
    print("[2/6] Selecting product subset...")

    # Find top categories by product count
    cat_counts = df['category'].value_counts()
    top_cats = cat_counts.head(TOP_N_CATEGORIES).index.tolist()
    print(f"  Top {TOP_N_CATEGORIES} categories: {', '.join(top_cats[:5])}...")

    selected = []
    for cat in top_cats:
        cat_df = df[df['category'] == cat].copy()

        # Prioritize: best sellers first, then by bought_last_month, then by reviews
        cat_df['priority'] = (
            cat_df['isBestSeller'].astype(int) * 10000 +
            cat_df['boughtInLastMonth'].clip(upper=10000) +
            cat_df['reviews'].clip(upper=5000)
        )
        cat_df = cat_df.sort_values('priority', ascending=False)

        # Take top N from this category
        sample = cat_df.head(PRODUCTS_PER_CATEGORY)
        selected.append(sample)

    result = pd.concat(selected, ignore_index=True)

    # Remove duplicates by ASIN
    result = result.drop_duplicates(subset='asin')

    print(f"  Selected {len(result):,} products across {result['category'].nunique()} categories")
    print(f"  Best sellers: {result['isBestSeller'].sum()}")
    print(f"  Price range: ${result['price'].min():.2f} - ${result['price'].max():.2f}")

    return result


def generate_users(categories):
    """Generate synthetic users with preferences for real categories."""
    print("[3/6] Generating user profiles...")

    users = []
    used_names = set()
    cat_list = list(categories)

    for i in range(NUM_USERS):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        username = f"{first.lower()}_{last.lower()}"

        attempt = 0
        while username in used_names and attempt < 20:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            username = f"{first.lower()}_{last.lower()}"
            attempt += 1
        if username in used_names:
            username += f"_{i}"
        used_names.add(username)

        # Assign 2-4 preferred categories
        num_prefs = random.randint(2, 4)
        preferred = random.sample(cat_list, min(num_prefs, len(cat_list)))
        rating_bias = random.uniform(-0.2, 0.4)

        users.append({
            "username": username,
            "email": f"{username}@example.com",
            "preferences": json.dumps({
                "preferred_categories": preferred,
                "rating_bias": round(rating_bias, 2),
            }),
        })

    print(f"  Generated {len(users)} users")
    return users


def generate_reviews(users, products_df, avg_per_user=25):
    """Generate synthetic reviews based on real product ratings."""
    print("[4/6] Generating reviews...")

    reviews = []
    products_by_category = {}
    for idx, row in products_df.iterrows():
        cat = row['category']
        if cat not in products_by_category:
            products_by_category[cat] = []
        products_by_category[cat].append(idx)

    all_cats = list(products_by_category.keys())
    product_review_counts = {}

    for user_idx, user in enumerate(users):
        prefs = json.loads(user["preferences"])
        preferred_cats = prefs["preferred_categories"]
        rating_bias = prefs["rating_bias"]

        num_reviews = random.randint(max(5, avg_per_user - 15), avg_per_user + 15)
        reviewed = set()

        for _ in range(num_reviews):
            # 70% preferred category, 30% random
            if random.random() < 0.7 and preferred_cats:
                cat = random.choice(preferred_cats)
                if cat not in products_by_category:
                    cat = random.choice(all_cats)
            else:
                cat = random.choice(all_cats)

            if not products_by_category.get(cat):
                continue

            product_idx = random.choice(products_by_category[cat])
            if product_idx in reviewed:
                continue
            reviewed.add(product_idx)

            product = products_df.loc[product_idx]

            # Base rating near the product's actual stars rating
            base = product.get('stars', 4.0)
            noise = random.gauss(0, 0.5)
            rating = min(5.0, max(1.0, round(base + noise + rating_bias, 1)))
            # Snap to nearest 0.5
            rating = round(rating * 2) / 2
            rating = min(5.0, max(1.0, rating))

            # Review text
            if rating >= 4.0:
                text = random.choice(POSITIVE_REVIEWS)
            elif rating >= 3.0:
                text = random.choice(NEUTRAL_REVIEWS)
            else:
                text = random.choice(NEGATIVE_REVIEWS)

            days_ago = random.randint(0, 730)
            timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()

            reviews.append({
                "user_id": user_idx + 1,
                "product_id": product_idx,  # Will be remapped after DB insert
                "rating": rating,
                "review_text": text,
                "helpful_votes": random.randint(0, 50),
                "verified_purchase": random.choices([1, 0], weights=[0.8, 0.2], k=1)[0],
                "timestamp": timestamp,
                "_product_asin": product['asin'],
            })

            product_review_counts[product_idx] = product_review_counts.get(product_idx, 0) + 1

    print(f"  Generated {len(reviews)} reviews")
    return reviews


def seed_database():
    """Load real Amazon data and seed the database."""
    from database.models import init_db, get_db, DB_PATH

    # Step 1: Load Amazon data
    df, cat_map = load_amazon_data()

    # Step 2: Select subset
    products_df = select_product_subset(df)

    # Step 3: Generate users
    categories = products_df['category'].unique().tolist()
    users = generate_users(categories)

    # Step 4: Generate reviews
    reviews = generate_reviews(users, products_df, avg_per_user=AVG_REVIEWS_PER_USER)

    # Step 5: Initialize database
    print("[5/6] Seeding database...")
    init_db()

    with get_db() as conn:
        # Insert users
        conn.executemany(
            "INSERT OR IGNORE INTO users (username, email, preferences) VALUES (?, ?, ?)",
            [(u["username"], u["email"], u["preferences"]) for u in users]
        )

        # Insert products
        asin_to_dbid = {}
        for _, row in products_df.iterrows():
            # Clean image URL - use higher resolution
            img_url = str(row.get('imgUrl', ''))
            if img_url and '_AC_UL320_' in img_url:
                img_url = img_url.replace('_AC_UL320_', '_AC_UL400_')

            cursor = conn.execute(
                """INSERT OR IGNORE INTO products
                   (asin, title, category, description, price, list_price, avg_rating,
                    rating_count, image_url, product_url, is_best_seller, bought_last_month)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row['asin'],
                    str(row['title'])[:200],
                    row['category'],
                    str(row['title']),  # Use title as description since dataset doesn't have one
                    float(row.get('price', 0)),
                    float(row.get('listPrice', 0)),
                    float(row.get('stars', 0)),
                    int(row.get('reviews', 0)),
                    img_url,
                    str(row.get('productURL', '')),
                    1 if row.get('isBestSeller', False) else 0,
                    int(row.get('boughtInLastMonth', 0)),
                )
            )
            if cursor.lastrowid:
                asin_to_dbid[row['asin']] = cursor.lastrowid

        # Build ASIN to DB ID mapping for all products
        rows = conn.execute("SELECT product_id, asin FROM products").fetchall()
        for r in rows:
            asin_to_dbid[r['asin']] = r['product_id']

        # Insert reviews (remap product IDs)
        review_inserts = []
        for r in reviews:
            db_pid = asin_to_dbid.get(r["_product_asin"])
            if db_pid:
                review_inserts.append((
                    r["user_id"], db_pid, r["rating"], r["review_text"],
                    r["helpful_votes"], r["verified_purchase"], r["timestamp"]
                ))

        conn.executemany(
            """INSERT OR IGNORE INTO reviews
               (user_id, product_id, rating, review_text, helpful_votes, verified_purchase, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            review_inserts
        )

    # Step 6: Verify
    print("[6/6] Verifying...")
    with get_db() as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        review_count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        cat_count = conn.execute("SELECT COUNT(DISTINCT category) FROM products").fetchone()[0]
        bestseller_count = conn.execute("SELECT COUNT(*) FROM products WHERE is_best_seller = 1").fetchone()[0]

    print(f"\n[OK] Database seeded successfully!")
    print(f"  Database: {DB_PATH}")
    print(f"  Users: {user_count}")
    print(f"  Products: {product_count}")
    print(f"  Categories: {cat_count}")
    print(f"  Best Sellers: {bestseller_count}")
    print(f"  Reviews: {review_count}")
    print(f"  Avg reviews/user: {review_count / max(user_count, 1):.1f}")


if __name__ == "__main__":
    seed_database()
