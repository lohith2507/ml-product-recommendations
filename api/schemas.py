"""
Pydantic Schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# --- Request Schemas ---

class ChatRequest(BaseModel):
    user_id: int
    message: str
    conversation_history: Optional[List[dict]] = None


class ClickRequest(BaseModel):
    user_id: int
    product_id: int
    algorithm: str


class CartRequest(BaseModel):
    product_ids: List[int]


class CartItemRequest(BaseModel):
    product_id: int
    quantity: int = Field(1, ge=1, le=99)


class CartQuantityRequest(BaseModel):
    quantity: int = Field(..., ge=0, le=99)


class CartActionRequest(BaseModel):
    user_id: int


class PriceUpdateRequest(BaseModel):
    product_id: int
    price: float = Field(..., gt=0)
    list_price: Optional[float] = Field(None, gt=0)
    warehouse_id: str = "default"


class StockUpdateRequest(BaseModel):
    product_id: int
    quantity: int = Field(..., ge=0)
    warehouse_id: str = "default"


# --- Response Schemas ---

class ProductResponse(BaseModel):
    product_id: int
    asin: Optional[str] = None
    title: str
    category: str
    description: Optional[str] = None
    price: float = 0.0
    avg_rating: float = 0.0
    rating_count: int = 0
    image_url: Optional[str] = None
    features: Any = []

    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    product_id: int
    title: str
    category: str
    price: float = 0.0
    avg_rating: float = 0.0
    rating_count: int = 0
    image_url: Optional[str] = None
    predicted_rating: Optional[float] = None
    similarity_score: Optional[float] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    algorithm: str = "unknown"

    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    user_id: int
    username: str
    email: Optional[str] = None
    preferences: Any = {}
    review_count: int = 0

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    response: str
    products: List[dict] = []
    transcription: Optional[str] = None
    active_context: List[str] = []
    pending_actions: List[dict] = []
    tools_used: List[str] = []
    orchestrator: Optional[str] = None
    thread_id: Optional[str] = None
    ui_actions: List[dict] = []


class ABTestGroupResult(BaseModel):
    algorithm_group: str
    num_sessions: int
    total_impressions: int
    total_clicks: int
    avg_ctr: float
    total_ctr: float


class ABTestResponse(BaseModel):
    groups: List[ABTestGroupResult] = []
    winner: Optional[str] = None
    significance: dict = {}


class CTRDataPoint(BaseModel):
    date: str
    ctr: float
    impressions: int
    clicks: int


class AnalyticsResponse(BaseModel):
    ctr_by_algorithm: dict = {}
    total_recommendations: int = 0
    total_clicks: int = 0
    overall_ctr: float = 0.0
