"""
API Routes for the Recommendation System.

Defines all REST endpoints for products, users, recommendations,
chatbot, and analytics.
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from typing import Optional, List

from database.models import get_db, dict_from_row
from api.schemas import (
    ChatRequest, ClickRequest, ChatResponse,
    ABTestResponse, AnalyticsResponse, CartRequest,
)
from recommenders.semantic import semantic_engine
import json

router = APIRouter(prefix="/api", tags=["API"])

# Global recommender instances (set during app startup)
cf_recommender = None
cb_recommender = None
llm_recommender = None
ab_engine = None


def set_recommenders(cf, cb, llm, ab):
    """Set global recommender instances. Called from app.py on startup."""
    global cf_recommender, cb_recommender, llm_recommender, ab_engine
    cf_recommender = cf
    cb_recommender = cb
    llm_recommender = llm
    ab_engine = ab


# ========== User Endpoints ==========

@router.get("/users")
async def list_users(limit: int = Query(50, le=200)):
    """List all users with their review counts."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT u.*, COUNT(r.review_id) as review_count
            FROM users u
            LEFT JOIN reviews r ON u.user_id = r.user_id
            GROUP BY u.user_id
            ORDER BY review_count DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict_from_row(r) for r in rows]


@router.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user profile with purchase history summary."""
    with get_db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        reviews = conn.execute("""
            SELECT r.*, p.title, p.category, p.price, p.image_url
            FROM reviews r
            JOIN products p ON r.product_id = p.product_id
            WHERE r.user_id = ?
            ORDER BY r.rating DESC
            LIMIT 20
        """, (user_id,)).fetchall()

        # Category distribution
        categories = conn.execute("""
            SELECT p.category, COUNT(*) as count, AVG(r.rating) as avg_rating
            FROM reviews r
            JOIN products p ON r.product_id = p.product_id
            WHERE r.user_id = ?
            GROUP BY p.category
            ORDER BY count DESC
        """, (user_id,)).fetchall()

    user_data = dict_from_row(user)
    user_data["reviews"] = [dict_from_row(r) for r in reviews]
    user_data["category_distribution"] = [dict_from_row(c) for c in categories]
    user_data["review_count"] = len([dict_from_row(r) for r in reviews])

    return user_data


# ========== Product Endpoints ==========

@router.get("/products")
async def list_products(
    category: Optional[str] = None,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, le=100),
):
    """Browse products with optional filtering."""
    offset = (page - 1) * limit

    with get_db() as conn:
        query = "SELECT * FROM products WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        if search:
            query += " AND (title LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        # Get total count
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = conn.execute(count_query, params).fetchone()[0]

        query += " ORDER BY avg_rating DESC, rating_count DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()

    return {
        "products": [dict_from_row(r) for r in rows],
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,
    }


@router.get("/search/semantic")
async def semantic_search(q: str = Query(..., min_length=2), limit: int = Query(10, le=50)):
    """Perform a vector-based semantic search across products."""
    results = semantic_engine.search(q, top_k=limit)
    if not results:
        return {"products": [], "total": 0}
        
    product_ids = [r["product_id"] for r in results]
    
    with get_db() as conn:
        placeholders = ",".join(["?"] * len(product_ids))
        query = f"SELECT * FROM products WHERE product_id IN ({placeholders})"
        rows = conn.execute(query, product_ids).fetchall()
        
        # Sort rows based on the semantic search order
        product_map = {r["product_id"]: dict_from_row(r) for r in rows}
        sorted_products = []
        for r in results:
            pid = r["product_id"]
            if pid in product_map:
                prod = product_map[pid]
                prod["semantic_score"] = r["similarity"]
                sorted_products.append(prod)
                
    return {"products": sorted_products, "total": len(sorted_products)}

import base64

@router.post("/search/visual")
async def visual_search(file: UploadFile = File(...), limit: int = Query(10, le=50)):
    """Perform visual search by converting image to text description via LLM, then semantic search."""
    if llm_recommender is None or not llm_recommender.client:
        raise HTTPException(status_code=503, detail="LLM not available")
        
    try:
        # Read and encode image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Use Groq Vision to describe the product
        prompt = "Describe this product image in 1-2 short sentences, focusing on the item type, color, style, and key features. This will be used as a search query."
        
        response = llm_recommender.client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        description = response.choices[0].message.content.strip()
        print(f"📷 Vision Model Description: {description}")
        
        # Now run semantic search using the generated description
        return await semantic_search(q=description, limit=limit)
        
    except Exception as e:
        print(f"Visual search error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image")

@router.get("/products/{product_id}")
async def get_product(product_id: int):
    """Get product details with reviews."""
    with get_db() as conn:
        product = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()

        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        reviews = conn.execute("""
            SELECT r.*, u.username
            FROM reviews r
            JOIN users u ON r.user_id = u.user_id
            WHERE r.product_id = ?
            ORDER BY r.helpful_votes DESC
            LIMIT 10
        """, (product_id,)).fetchall()

    product_data = dict_from_row(product)
    product_data["reviews"] = [dict_from_row(r) for r in reviews]

    return product_data


@router.get("/products/{product_id}/summary")
async def get_product_review_summary(product_id: int):
    """Use the LLM to summarize all reviews for a product."""
    if llm_recommender is None or not llm_recommender.client:
        raise HTTPException(status_code=503, detail="LLM not available")
        
    with get_db() as conn:
        product = conn.execute("SELECT title FROM products WHERE product_id = ?", (product_id,)).fetchone()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
            
        reviews = conn.execute("SELECT review_text, rating FROM reviews WHERE product_id = ?", (product_id,)).fetchall()
        
    if not reviews:
        return {"summary": "No reviews available for this product yet.", "pros": [], "cons": []}
        
    reviews_text = "\n".join([f"- Rating: {r['rating']}/5. Review: {r['review_text']}" for r in reviews])
    
    prompt = f"""You are an AI shopping assistant. Please analyze the following reviews for the product "{product['title']}".
Provide a short overview, a list of Pros, and a list of Cons based ONLY on the provided reviews.

Reviews:
{reviews_text}

Respond strictly in valid JSON format:
{{
  "summary": "2-3 sentences overview of what customers think",
  "pros": ["pro 1", "pro 2", ...],
  "cons": ["con 1", "con 2", ...]
}}
"""
    try:
        response = llm_recommender.client.chat.completions.create(
            model=llm_recommender.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Failed to summarize reviews: {e}")
        return {"summary": "Failed to generate summary.", "pros": [], "cons": []}



@router.get("/products/categories/list")
async def list_categories():
    """Get all product categories with counts."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT category, COUNT(*) as count, AVG(avg_rating) as avg_rating
            FROM products
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()
    return [dict_from_row(r) for r in rows]


# ========== Recommendation Endpoints ==========

@router.get("/recommend/collaborative/{user_id}")
async def recommend_collaborative(user_id: int, n: int = Query(10, le=50)):
    """Get collaborative filtering recommendations for a user."""
    if cf_recommender is None:
        raise HTTPException(status_code=503, detail="Collaborative filtering model not loaded")

    try:
        recs = cf_recommender.recommend(user_id=user_id, n=n)
        # Record impressions for A/B testing
        if ab_engine:
            ab_engine.record_impression(user_id, "collaborative", len(recs))
        return {"recommendations": recs, "algorithm": "collaborative", "count": len(recs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommend/content/{user_id}")
async def recommend_content(user_id: int, n: int = Query(10, le=50)):
    """Get content-based recommendations for a user."""
    if cb_recommender is None:
        raise HTTPException(status_code=503, detail="Content-based model not loaded")

    try:
        recs = cb_recommender.recommend(user_id=user_id, n=n)
        if ab_engine:
            ab_engine.record_impression(user_id, "content_based", len(recs))
        return {"recommendations": recs, "algorithm": "content_based", "count": len(recs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommend/hybrid/{user_id}")
async def recommend_hybrid(user_id: int, n: int = Query(10, le=50)):
    """Get LLM-powered hybrid recommendations for a user."""
    if cf_recommender is None or cb_recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Get candidates from both engines
        cf_candidates = cf_recommender.recommend(user_id=user_id, n=15)
        cb_candidates = cb_recommender.recommend(user_id=user_id, n=15)

        if llm_recommender:
            recs = llm_recommender.recommend(user_id, cf_candidates, cb_candidates, n=n)
        else:
            # Fallback: interleave results
            recs = _merge_candidates(cf_candidates, cb_candidates, n)

        if ab_engine:
            ab_engine.record_impression(user_id, "hybrid", len(recs))

        return {"recommendations": recs, "algorithm": "hybrid", "count": len(recs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend/cart")
async def recommend_cart(request: CartRequest, n: int = Query(10, le=50)):
    """Get frequently bought together recommendations for a shopping cart."""
    if cf_recommender is None:
        raise HTTPException(status_code=503, detail="CF Model not loaded")

    try:
        # Aggregate similar items for all products in cart
        all_recs = []
        for pid in request.product_ids:
            recs = cf_recommender.get_similar_items(product_id=pid, n=n)
            all_recs.extend(recs)
            
        # Deduplicate and sort by co_purchase_count / score
        unique_recs = {}
        for rec in all_recs:
            pid = rec["product_id"]
            if pid in request.product_ids:
                continue # Don't recommend what's already in the cart
            if pid not in unique_recs:
                unique_recs[pid] = rec
            else:
                # Boost score if recommended by multiple cart items
                if "co_purchase_count" in rec:
                    unique_recs[pid]["co_purchase_count"] = unique_recs[pid].get("co_purchase_count", 0) + rec["co_purchase_count"]
                if "predicted_rating" in rec:
                    unique_recs[pid]["predicted_rating"] = max(unique_recs[pid].get("predicted_rating", 0), rec["predicted_rating"])
                    
        # Sort by co_purchase_count (desc)
        sorted_recs = sorted(unique_recs.values(), key=lambda x: x.get("co_purchase_count", 0), reverse=True)[:n]
        
        # Override algorithm label
        for rec in sorted_recs:
            rec["algorithm"] = "also_bought"
            
        return {"recommendations": sorted_recs, "count": len(sorted_recs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend/also-bought/{product_id}")
async def recommend_also_bought(product_id: int, n: int = Query(10, le=50)):
    """Get 'customers also bought' recommendations."""
    if cf_recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        recs = cf_recommender.get_similar_items(product_id=product_id, n=n)
        return {"recommendations": recs, "algorithm": "also_bought", "count": len(recs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Chat Endpoint ==========

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """AI chatbot endpoint for conversational product discovery."""
    if llm_recommender is None:
        return ChatResponse(
            response="The AI assistant is currently unavailable. Please try the recommendation tabs!",
            products=[]
        )

    try:
        result = llm_recommender.chat(
            user_id=request.user_id,
            message=request.message,
            conversation_history=request.conversation_history
        )
        return ChatResponse(**result)
    except Exception as e:
        return ChatResponse(
            response=f"Sorry, I encountered an error. Please try again! ({str(e)[:100]})",
            products=[]
        )


@router.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(user_id: int = Form(...), conversation_history: str = Form("[]"), file: UploadFile = File(...)):
    """AI chatbot endpoint that accepts voice audio, transcribes it via Whisper, and returns product recommendations."""
    if llm_recommender is None or not llm_recommender.groq_client:
        return ChatResponse(
            response="The AI voice assistant is currently unavailable.",
            products=[]
        )

    try:
        # Transcribe audio using Groq Whisper API
        audio_content = await file.read()
        transcription = llm_recommender.groq_client.audio.transcriptions.create(
            file=(file.filename, audio_content),
            model="whisper-large-v3",
            response_format="json",
        )
        
        user_message = transcription.text
        if not user_message or user_message.strip() == "":
            return ChatResponse(
                response="I didn't quite catch that. Could you try speaking again?",
                products=[]
            )

        history = json.loads(conversation_history)
        
        result = llm_recommender.chat(
            user_id=user_id,
            message=user_message,
            conversation_history=history
        )
        
        # Inject the transcribed text into the response so the UI knows what was said
        result["transcription"] = user_message
        return ChatResponse(**result)
        
    except Exception as e:
        return ChatResponse(
            response=f"Sorry, I couldn't process your voice message. ({str(e)[:100]})",
            products=[]
        )


@router.get("/admin/generate-email")
async def generate_marketing_email(user_id: int):
    """Generate a personalized marketing email for a user."""
    if llm_recommender is None or not llm_recommender.client:
        raise HTTPException(status_code=503, detail="LLM not available")

    profile = llm_recommender.get_user_profile(user_id)
    
    prompt = f"""You are an expert e-commerce marketing copywriter.
Based on the following user profile, write a highly personalized, engaging, and short marketing email (under 150 words).
Include a catchy subject line. Format the response as Markdown.

User Profile:
{json.dumps(profile, indent=2)}

Output format:
**Subject:** [Your Subject Line]

[Email Body]
"""

    try:
        response = llm_recommender.client.chat.completions.create(
            model=llm_recommender.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        email_content = response.choices[0].message.content
        return {"email": email_content}
    except Exception as e:
        print(f"Failed to generate email: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate email")

# ========== A/B Testing Endpoints ==========

@router.get("/admin/stats")
async def get_admin_stats():
    """Calculate revenue, clicks, and impressions by algorithm."""
    with get_db() as conn:
        stats = conn.execute("""
            SELECT 
                r.algorithm,
                COUNT(r.rec_id) as total_impressions,
                SUM(r.clicked) as total_clicks,
                SUM(CASE WHEN r.clicked = 1 THEN p.price ELSE 0 END) as total_revenue
            FROM recommendations r
            JOIN products p ON r.product_id = p.product_id
            GROUP BY r.algorithm
            ORDER BY total_revenue DESC
        """).fetchall()
        
    results = []
    for row in stats:
        d = dict(row)
        impressions = d["total_impressions"]
        clicks = d["total_clicks"]
        d["ctr_pct"] = round((clicks / impressions * 100) if impressions > 0 else 0, 2)
        d["total_revenue"] = round(d["total_revenue"], 2)
        results.append(d)
        
    return results

@router.get("/analytics/ctr")
async def get_ctr_analytics():
    """Get CTR analytics data for the dashboard."""
    with get_db() as conn:
        # Per-algorithm CTR
        algo_stats = conn.execute("""
            SELECT
                algorithm_group,
                COUNT(*) as sessions,
                SUM(impressions) as total_impressions,
                SUM(clicks) as total_clicks,
                AVG(ctr) * 100 as avg_ctr_pct
            FROM ab_experiments
            GROUP BY algorithm_group
        """).fetchall()

        # Overall stats
        overall = conn.execute("""
            SELECT
                COUNT(*) as total_sessions,
                SUM(impressions) as total_impressions,
                SUM(clicks) as total_clicks
            FROM ab_experiments
        """).fetchone()

    algo_data = {}
    for row in algo_stats:
        r = dict(row)
        algo_data[r["algorithm_group"]] = {
            "sessions": r["sessions"],
            "impressions": r["total_impressions"],
            "clicks": r["total_clicks"],
            "ctr": round(r["avg_ctr_pct"], 2),
        }

    overall_data = dict(overall)
    overall_ctr = (
        (overall_data["total_clicks"] / overall_data["total_impressions"] * 100)
        if overall_data["total_impressions"] > 0 else 0.0
    )

    return {
        "ctr_by_algorithm": algo_data,
        "total_sessions": overall_data["total_sessions"],
        "total_impressions": overall_data["total_impressions"],
        "total_clicks": overall_data["total_clicks"],
        "overall_ctr": round(overall_ctr, 2),
    }


@router.get("/analytics/ab-test")
async def get_ab_test_results():
    """Get A/B test results with statistical significance."""
    if ab_engine is None:
        raise HTTPException(status_code=503, detail="A/B testing engine not initialized")

    return ab_engine.get_results()


@router.get("/analytics/ctr-timeline")
async def get_ctr_timeline(days: int = Query(30, le=90)):
    """Get CTR trends over time."""
    if ab_engine is None:
        raise HTTPException(status_code=503, detail="A/B testing engine not initialized")

    return ab_engine.get_ctr_over_time(days=days)


# ========== Interaction Tracking ==========

@router.post("/simulate/click")
async def simulate_click(request: ClickRequest):
    """Record a simulated user click on a recommendation and update real-time profile."""
    if ab_engine:
        ab_engine.record_click(request.user_id, request.product_id, request.algorithm)

    with get_db() as conn:
        # Record click
        conn.execute("""
            INSERT INTO recommendations (user_id, product_id, algorithm, clicked)
            VALUES (?, ?, ?, 1)
        """, (request.user_id, request.product_id, request.algorithm))
        
        # Real-Time Profile Update: Add category to liked_categories
        product = conn.execute("SELECT category FROM products WHERE product_id = ?", (request.product_id,)).fetchone()
        if product:
            category = product["category"]
            user = conn.execute("SELECT preferences FROM users WHERE user_id = ?", (request.user_id,)).fetchone()
            if user:
                try:
                    prefs = json.loads(user["preferences"]) if isinstance(user["preferences"], str) else {}
                except json.JSONDecodeError:
                    prefs = {}
                    
                liked = prefs.get("liked_categories", [])
                if category not in liked:
                    liked.append(category)
                    # Keep only last 10 to avoid bloated profiles
                    prefs["liked_categories"] = liked[-10:]
                    
                    conn.execute("UPDATE users SET preferences = ? WHERE user_id = ?", 
                               (json.dumps(prefs), request.user_id))

    return {"status": "ok", "message": "Click recorded and profile updated"}


# ========== Helper Functions ==========

def _merge_candidates(cf_list, cb_list, n=10):
    """Round-robin merge of CF and CB candidates for diversity."""
    merged = []
    seen = set()
    cf_idx, cb_idx = 0, 0

    while len(merged) < n:
        if cf_idx < len(cf_list):
            item = cf_list[cf_idx]
            cf_idx += 1
            pid = item.get("product_id")
            if pid not in seen:
                seen.add(pid)
                item["algorithm"] = "hybrid_cf"
                item["explanation"] = "Based on users with similar taste."
                merged.append(item)

        if len(merged) >= n:
            break

        if cb_idx < len(cb_list):
            item = cb_list[cb_idx]
            cb_idx += 1
            pid = item.get("product_id")
            if pid not in seen:
                seen.add(pid)
                item["algorithm"] = "hybrid_cb"
                item["explanation"] = "Based on products you've liked."
                merged.append(item)

        if cf_idx >= len(cf_list) and cb_idx >= len(cb_list):
            break

    return merged
