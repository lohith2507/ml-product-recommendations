"""
SQLite Database Models and Connection Manager.

Defines the schema for users, products, reviews, recommendations,
and A/B testing experiments. Updated for real Amazon dataset.
"""

import sqlite3
import os
import json
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recommendation.db")


def get_connection():
    """Get a new database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT,
                preferences TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                asin TEXT UNIQUE,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                price REAL DEFAULT 0.0,
                list_price REAL DEFAULT 0.0,
                avg_rating REAL DEFAULT 0.0,
                rating_count INTEGER DEFAULT 0,
                image_url TEXT,
                product_url TEXT,
                features TEXT DEFAULT '[]',
                is_best_seller INTEGER DEFAULT 0,
                bought_last_month INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                rating REAL NOT NULL CHECK(rating >= 1.0 AND rating <= 5.0),
                review_text TEXT,
                helpful_votes INTEGER DEFAULT 0,
                verified_purchase INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id),
                UNIQUE(user_id, product_id)
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                score REAL DEFAULT 0.0,
                explanation TEXT,
                clicked INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            );

            CREATE TABLE IF NOT EXISTS ab_experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                algorithm_group TEXT NOT NULL,
                session_id TEXT,
                impressions INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                ctr REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- Indexes for fast lookups
            CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(user_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
            CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id);
            CREATE INDEX IF NOT EXISTS idx_recommendations_algo ON recommendations(algorithm);
            CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
            CREATE INDEX IF NOT EXISTS idx_products_bestseller ON products(is_best_seller);
            CREATE INDEX IF NOT EXISTS idx_ab_experiments_user ON ab_experiments(user_id);
            CREATE INDEX IF NOT EXISTS idx_ab_experiments_algo ON ab_experiments(algorithm_group);
        """)
    print("[OK] Database schema initialized successfully.")


def dict_from_row(row):
    """Convert a sqlite3.Row to a dictionary."""
    if row is None:
        return None
    d = dict(row)
    # Parse JSON fields
    for key in ['preferences', 'features']:
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


if __name__ == "__main__":
    init_db()
