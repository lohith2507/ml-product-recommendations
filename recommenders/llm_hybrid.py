"""
LLM-Powered Hybrid Recommender using Groq API.

Combines candidates from Collaborative Filtering and Content-Based engines,
then uses Groq's Llama 3.3 model to re-rank, personalize, and generate
natural language explanations for recommendations.

Also powers the AI chatbot for conversational product discovery.
"""

import os
import json
import time
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
import httpx

from database.models import get_db, dict_from_row

load_dotenv()


class LLMHybridRecommender:
    """LLM-powered hybrid recommendation engine with chatbot."""

    def __init__(self):
        nvidia_key = os.environ.get("NVIDIA_API_KEY")
        if nvidia_key:
            self.nvidia_client = OpenAI(
                api_key=nvidia_key,
                base_url="https://integrate.api.nvidia.com/v1",
                http_client=httpx.Client(verify=False, timeout=15.0)
            )
        else:
            self.nvidia_client = None

        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key, timeout=15.0)
        else:
            self.groq_client = None

        if not self.nvidia_client and not self.groq_client:
            print("⚠️  No API keys found. LLM features will be disabled.")
            
        self.nvidia_model = "meta/llama-3.3-70b-instruct"
        self.groq_model = "llama-3.1-8b-instant"
        self.max_retries = 2

    @property
    def client(self):
        """Get the active LLM client (Groq or NVIDIA)."""
        return self.groq_client if self.groq_client else self.nvidia_client

    @property
    def model_name(self):
        """Get the active model name corresponding to the active client."""
        return self.groq_model if self.groq_client else self.nvidia_model

    def _call_llm(self, messages, temperature=0.7, max_tokens=2000):
        """
        Make an API call to NVIDIA NIM or Groq with retry logic.

        Args:
            messages: List of message dicts (role, content).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            The response text, or None on failure.
        """
        if not self.nvidia_client and not self.groq_client:
            return None

        for attempt in range(self.max_retries):
            # Try Groq first for maximum speed
            if self.groq_client:
                try:
                    response = self.groq_client.chat.completions.create(
                        messages=messages,
                        model=self.groq_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    error_msg = str(e)
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        wait_time = (2 ** attempt) * 2
                        print(f"⏳ Groq rate limited. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"⚠️ Groq API error: {error_msg}. Falling back to NVIDIA...")

            # Fallback to NVIDIA
            if self.nvidia_client:
                try:
                    response = self.nvidia_client.chat.completions.create(
                        messages=messages,
                        model=self.nvidia_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"❌ NVIDIA API error: {e}")
                    if attempt == self.max_retries - 1:
                        return None
                    time.sleep(1)

        return None

    def get_user_profile(self, user_id):
        """Build a text description of the user's preferences."""
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()

            reviews = conn.execute("""
                SELECT r.rating, r.review_text, p.title, p.category
                FROM reviews r
                JOIN products p ON r.product_id = p.product_id
                WHERE r.user_id = ?
                ORDER BY r.rating DESC
                LIMIT 15
            """, (user_id,)).fetchall()

        if not user:
            return "Unknown user with no purchase history."

        user_data = dict_from_row(user)
        prefs = user_data.get("preferences", {})

        # Build profile summary
        liked = [dict(r) for r in reviews if r["rating"] >= 4.0]
        disliked = [dict(r) for r in reviews if r["rating"] <= 2.0]

        categories = {}
        for r in reviews:
            cat = r["category"]
            categories[cat] = categories.get(cat, 0) + 1

        profile = {
            "username": user_data.get("username", "unknown"),
            "persona": prefs.get("persona", "general"),
            "top_categories": sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3],
            "recent_likes": [{"title": r["title"], "rating": r["rating"]} for r in liked[:5]],
            "recent_dislikes": [{"title": r["title"], "rating": r["rating"]} for r in disliked[:3]],
            "total_reviews": len(reviews),
        }

        return json.dumps(profile, indent=2)

    def recommend(self, user_id, cf_candidates, cb_candidates, n=10):
        """
        Generate hybrid recommendations by having the LLM re-rank and
        explain candidates from CF and CB engines.

        Args:
            user_id: Target user ID.
            cf_candidates: Top candidates from collaborative filtering.
            cb_candidates: Top candidates from content-based filtering.
            n: Number of final recommendations.

        Returns:
            List of recommendation dicts with LLM explanations.
        """
        if self.client is None:
            return self._fallback_merge(cf_candidates, cb_candidates, n)

        # Build user profile
        user_profile = self.get_user_profile(user_id)

        # Format candidates for the LLM
        all_candidates = []
        seen_ids = set()

        for item in cf_candidates + cb_candidates:
            pid = item.get("product_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_candidates.append({
                    "product_id": pid,
                    "title": item.get("title", "Unknown"),
                    "category": item.get("category", "Unknown"),
                    "price": item.get("price", 0),
                    "avg_rating": item.get("avg_rating", 0),
                    "cf_score": item.get("predicted_rating", 0) if item in cf_candidates else None,
                    "cb_score": item.get("similarity_score", 0) if item in cb_candidates else None,
                })

        if not all_candidates:
            return []

        # Construct the LLM prompt
        messages = [
            {
                "role": "system",
                "content": """You are an AI recommendation engine for an e-commerce platform.
Your job is to re-rank product candidates and provide personalized explanations.

You MUST respond with a valid JSON object in this exact format:
{
  "recommendations": [
    {
      "product_id": <int>,
      "rank": <int>,
      "confidence": <float 0-1>,
      "explanation": "<1-2 sentence personalized explanation>"
    }
  ]
}

Rules:
- Re-rank based on the user profile, considering their preferences, liked categories, and purchase patterns.
- Provide diverse recommendations across categories when possible.
- Explanations should feel personal and mention specific user preferences.
- Return exactly the number requested or fewer if not enough good candidates."""
            },
            {
                "role": "user",
                "content": f"""Re-rank these product candidates for the following user and provide personalized explanations.

USER PROFILE:
{user_profile}

CANDIDATES (from collaborative and content-based filtering):
{json.dumps(all_candidates[:20], indent=2)}

Return the top {n} recommendations as JSON."""
            }
        ]

        response = self._call_llm(messages, temperature=0.3)

        if response is None:
            return self._fallback_merge(cf_candidates, cb_candidates, n)

        try:
            result = json.loads(response)
            recs = result.get("recommendations", [])

            # Enrich with full product details
            return self._enrich_llm_results(recs)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Failed to parse LLM response: {e}")
            return self._fallback_merge(cf_candidates, cb_candidates, n)

    def chat(self, user_id, message, conversation_history=None):
        """
        AI chatbot that recommends products based on natural language queries.

        Args:
            user_id: The user making the request.
            message: The user's chat message.
            conversation_history: Previous messages in the conversation.

        Returns:
            Dict with response text and optional product recommendations.
        """
        if not self.nvidia_client and not self.groq_client:
            return {
                "response": "I'm sorry, the AI assistant is currently unavailable. Please try the recommendation tabs instead!",
                "products": []
            }

        # Get user profile for context
        user_profile = self.get_user_profile(user_id)

        # Get product catalog summary
        with get_db() as conn:
            categories = conn.execute(
                "SELECT DISTINCT category FROM products"
            ).fetchall()
            cat_list = [r["category"] for r in categories]

            # Sample products for context
            sample_products = conn.execute("""
                SELECT product_id, title, category, price, avg_rating
                FROM products
                WHERE avg_rating >= 3.5
                ORDER BY rating_count DESC
                LIMIT 30
            """).fetchall()

        product_catalog = [dict(r) for r in sample_products]

        # Build conversation
        messages = [
            {
                "role": "system",
                "content": f"""You are a friendly and knowledgeable AI shopping assistant for an e-commerce platform.

You help users discover products based on their needs and preferences.

Available categories: {', '.join(cat_list)}

You MUST respond with a valid JSON object:
{{
  "response": "<friendly, helpful response text addressing the user's query>",
  "products": [
    {{
      "product_id": <int>,
      "reason": "<brief reason for recommending>"
    }}
  ],
  "active_context": ["<short tag representing a user constraint, e.g. 'Under $100', 'Electronics', 'Red color'>"]
}}

Rules:
- Be conversational, friendly, and helpful
- Reference the user's known preferences when relevant
- Recommend 3-5 products when the user asks for suggestions
- If the user asks about something not product-related, be helpful but gently redirect
- Only recommend products from the catalog provided
- If unsure what the user wants, ask clarifying questions

USER PROFILE:
{user_profile}

PRODUCT CATALOG SAMPLE:
{json.dumps(product_catalog, indent=2)}"""
            }
        ]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:  # Keep last 6 messages for context
                messages.append(msg)

        messages.append({"role": "user", "content": message})

        # Call the LLM (non-JSON response format for natural conversation)
        response = self._call_llm(messages, temperature=0.7, max_tokens=1500)

        if response is None:
            return {
                "response": "I'm having trouble connecting right now. Please try again in a moment!",
                "products": []
            }

        try:
            result = json.loads(response)
            chat_response = result.get("response", "I'd be happy to help you find the perfect product!")
            product_recs = result.get("products", [])

            # Enrich product recommendations
            enriched_products = []
            if product_recs:
                pids = [p["product_id"] for p in product_recs if "product_id" in p]
                if pids:
                    placeholders = ",".join(["?" for _ in pids])
                    with get_db() as conn:
                        rows = conn.execute(
                            f"SELECT * FROM products WHERE product_id IN ({placeholders})",
                            pids
                        ).fetchall()

                    product_map = {dict_from_row(r)["product_id"]: dict_from_row(r) for r in rows}

                    for rec in product_recs:
                        pid = rec.get("product_id")
                        if pid in product_map:
                            enriched_products.append({
                                **product_map[pid],
                                "reason": rec.get("reason", ""),
                                "algorithm": "chatbot",
                            })

            return {
                "response": chat_response,
                "products": enriched_products,
                "active_context": result.get("active_context", []),
            }

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            print(f"⚠️ Failed to parse LLM chat response: {e}")
            return {
                "response": "I found some options, but had trouble formatting them. Please try again!",
                "products": [],
                "active_context": [],
            }

    def _fallback_merge(self, cf_candidates, cb_candidates, n=10):
        """
        Fallback when LLM is unavailable: interleave CF and CB candidates.
        Uses round-robin merging for diversity.
        """
        merged = []
        seen = set()

        cf_iter = iter(cf_candidates)
        cb_iter = iter(cb_candidates)

        while len(merged) < n:
            # Alternate between CF and CB
            try:
                item = next(cf_iter)
                pid = item.get("product_id")
                if pid not in seen:
                    seen.add(pid)
                    item["algorithm"] = "hybrid_cf"
                    item["explanation"] = "Recommended based on similar users' preferences."
                    merged.append(item)
            except StopIteration:
                pass

            if len(merged) >= n:
                break

            try:
                item = next(cb_iter)
                pid = item.get("product_id")
                if pid not in seen:
                    seen.add(pid)
                    item["algorithm"] = "hybrid_cb"
                    item["explanation"] = "Recommended based on products you've enjoyed."
                    merged.append(item)
            except StopIteration:
                pass

            if not cf_candidates and not cb_candidates:
                break

        return merged[:n]

    def _enrich_llm_results(self, llm_recs):
        """Enrich LLM recommendation results with full product details."""
        if not llm_recs:
            return []

        pids = [r["product_id"] for r in llm_recs if "product_id" in r]
        if not pids:
            return []

        placeholders = ",".join(["?" for _ in pids])

        with get_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM products WHERE product_id IN ({placeholders})",
                pids
            ).fetchall()

        product_map = {dict_from_row(r)["product_id"]: dict_from_row(r) for r in rows}

        enriched = []
        for rec in llm_recs:
            pid = rec.get("product_id")
            if pid in product_map:
                enriched.append({
                    **product_map[pid],
                    "confidence": rec.get("confidence", 0.5),
                    "explanation": rec.get("explanation", "Recommended for you."),
                    "predicted_rating": round(rec.get("confidence", 0.5) * 5, 2),
                    "algorithm": "llm_hybrid",
                })

        return enriched


if __name__ == "__main__":
    recommender = LLMHybridRecommender()
    profile = recommender.get_user_profile(1)
    print("User Profile:")
    print(profile)
