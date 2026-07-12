"""
FastAPI Application — AI-Based Product Recommendation System.

Main entry point that initializes the application, loads trained models,
and serves both the API and the frontend dashboard.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Import routes and recommenders
from api.routes import router, set_recommenders
from database.models import init_db

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    print("\n🚀 Starting AI Recommendation System...")

    # Initialize database
    init_db()

    # Load trained models
    try:
        from recommenders.collaborative import CollaborativeFilteringRecommender
        cf = CollaborativeFilteringRecommender()
        cf.load_model()
        print("✅ Collaborative Filtering model loaded")
    except FileNotFoundError:
        print("⚠️  CF model not found. Run 'python train_models.py' first.")
        cf = None

    try:
        from recommenders.content_based import ContentBasedRecommender
        cb = ContentBasedRecommender()
        cb.load_model()
        print("✅ Content-Based model loaded")
    except FileNotFoundError:
        print("⚠️  CB model not found. Run 'python train_models.py' first.")
        cb = None

    try:
        from recommenders.llm_hybrid import LLMHybridRecommender
        llm = LLMHybridRecommender()
        print("✅ LLM Hybrid recommender initialized")
    except Exception as e:
        print(f"⚠️  LLM recommender failed to init: {e}")
        llm = None

    from recommenders.ab_testing import ABTestingEngine
    ab = ABTestingEngine()
    print("✅ A/B Testing engine initialized")

    # Set recommenders in routes module
    set_recommenders(cf, cb, llm, ab)

    print("\n🎯 System ready! Visit http://localhost:8000")
    print("📚 API docs at http://localhost:8000/docs\n")

    yield

    # Cleanup
    print("\n🛑 Shutting down...")


# Create the FastAPI app
app = FastAPI(
    title="AI Product Recommendation System",
    description="E-commerce recommendation engine with Collaborative Filtering, Content-Based, and LLM-Hybrid approaches.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include API routes
app.include_router(router)


# ========== Frontend Route ==========

@app.get("/")
async def dashboard(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
