"""
Response Schemas for API Endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class AIInsights(BaseModel):
    """AI-generated insights about predictions using OpenAI"""
    executive_summary: str = Field(..., description="High-level summary of predictions in business language")
    key_drivers: List[str] = Field(..., description="Main factors driving the predictions")
    recommendations: List[str] = Field(..., description="Actionable recommendations for the studio")
    risks: List[str] = Field(..., description="Potential risks or concerns to be aware of")
    confidence_explanation: str = Field(..., description="Explanation of confidence levels in plain language")


class MonthlyPrediction(BaseModel):
    """Prediction for a single month (v2.2.0 format)"""
    month: int = Field(..., description="Month number (1, 2, 3, etc.)")
    revenue: float = Field(..., description="Predicted revenue")
    member_count: int = Field(..., description="Predicted member count")
    retention_rate: Optional[float] = Field(None, description="Predicted retention rate")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")


# v2.3.0 Multi-Horizon Prediction Schemas

class DailyPredictionItem(BaseModel):
    """Single daily prediction item (v2.3.0)"""
    target: str = Field(..., description="Target name (e.g., 'revenue_day_1', 'attendance_day_7')")
    date: str = Field(..., description="Prediction date (ISO format)")
    value: float = Field(..., description="Predicted value")
    unit: str = Field(..., description="Unit of measurement (USD, visits, etc.)")


class WeeklyPredictionItem(BaseModel):
    """Single weekly prediction item (v2.3.0)"""
    target: str = Field(..., description="Target name (e.g., 'revenue_week_1')")
    week: int = Field(..., description="Week number (1, 2, 4)")
    date_range: Dict[str, str] = Field(..., description="Date range with 'start' and 'end' dates")
    value: float = Field(..., description="Predicted value")
    unit: str = Field(..., description="Unit of measurement (USD, visits, etc.)")


class MonthlyPredictionItem(BaseModel):
    """Single monthly prediction item (v2.3.0)"""
    target: str = Field(..., description="Target name (e.g., 'revenue_month_1', 'members_month_3')")
    month: int = Field(..., description="Month number (1, 2, 3)")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range with 'start' and 'end' dates")
    value: float = Field(..., description="Predicted value")
    unit: str = Field(..., description="Unit of measurement (USD, members, rate, etc.)")


class DailyPredictions(BaseModel):
    """Container for daily predictions (v2.3.0)"""
    horizon: str = Field(default="daily", description="Prediction horizon")
    predictions: List[DailyPredictionItem] = Field(..., description="List of daily predictions")
    summary: Dict = Field(..., description="Summary statistics for daily predictions")


class WeeklyPredictions(BaseModel):
    """Container for weekly predictions (v2.3.0)"""
    horizon: str = Field(default="weekly", description="Prediction horizon")
    predictions: List[WeeklyPredictionItem] = Field(..., description="List of weekly predictions")
    summary: Dict = Field(..., description="Summary statistics for weekly predictions")


class MonthlyPredictions(BaseModel):
    """Container for monthly predictions (v2.3.0)"""
    horizon: str = Field(default="monthly", description="Prediction horizon")
    predictions: List[MonthlyPredictionItem] = Field(..., description="List of monthly predictions")
    summary: Dict = Field(..., description="Summary statistics for monthly predictions")


class ForwardPredictionResponse(BaseModel):
    """
    Response for forward prediction

    For v2.2.0 models:
    - Returns 'predictions' (list of MonthlyPrediction)
    - Returns 'total_projected_revenue' and 'average_confidence'

    For v2.3.0 models:
    - Returns 'daily', 'weekly', 'monthly' (horizon-specific predictions)
    - Returns 'model_info' with additional metadata
    - Does not include 'total_projected_revenue' or 'average_confidence'
    """
    model_config = {"protected_namespaces": ()}

    scenario_id: str = Field(..., description="Unique scenario identifier")
    studio_id: str = Field(..., description="Studio identifier")

    # v2.2.0 fields (for backward compatibility)
    predictions: Optional[List[MonthlyPrediction]] = Field(None, description="Monthly predictions (v2.2.0 only)")
    total_projected_revenue: Optional[float] = Field(None, description="Sum of projected revenue (v2.2.0 only)")
    average_confidence: Optional[float] = Field(None, description="Average confidence score (v2.2.0 only)")

    # v2.3.0 multi-horizon fields
    daily: Optional[DailyPredictions] = Field(None, description="Daily predictions (v2.3.0 when 'daily' in horizons)")
    weekly: Optional[WeeklyPredictions] = Field(None, description="Weekly predictions (v2.3.0 when 'weekly' in horizons)")
    monthly: Optional[MonthlyPredictions] = Field(None, description="Monthly predictions (v2.3.0, always included)")

    # Common fields
    model_info: Optional[Dict] = Field(None, description="Model metadata (v2.3.0) or None (v2.2.0)")
    model_version: Optional[str] = Field(None, description="Model version used (v2.2.0 only, in model_info for v2.3.0)")
    business_levers: Optional[Dict] = Field(None, description="Input business levers (v2.3.0 only)")
    timestamp: str = Field(..., description="Prediction timestamp")

    # Optional enrichments
    explanation: Optional[Dict] = Field(None, description="SHAP-based explanation of prediction drivers")
    explainability: Optional[Dict] = Field(None, description="SHAP explainability details (v2.3.0 format)")
    quick_wins: Optional[List[Dict]] = Field(None, description="Quick win recommendations for improving revenue")
    ai_insights: Optional[AIInsights] = Field(None, description="AI-generated business insights about predictions")


class LeverChange(BaseModel):
    """Details of a lever change recommendation"""
    lever_name: str = Field(..., description="Name of the lever")
    current_value: float = Field(..., description="Current value")
    recommended_value: float = Field(..., description="Recommended value")
    change_absolute: float = Field(..., description="Absolute change")
    change_percentage: float = Field(..., description="Percentage change")
    priority: int = Field(..., ge=1, le=10, description="Priority (1=highest, 10=lowest)")


class ActionItem(BaseModel):
    """Recommended action item"""
    priority: int = Field(..., ge=1, description="Priority ranking")
    lever: str = Field(..., description="Lever to adjust")
    action: str = Field(..., description="Specific action to take")
    expected_impact: float = Field(..., description="Expected revenue impact ($)")
    timeline_weeks: int = Field(..., description="Estimated timeline in weeks")


class InversePredictionResponse(BaseModel):
    """Response for inverse prediction"""
    model_config = {"protected_namespaces": ()}

    optimization_id: str = Field(..., description="Unique optimization identifier")
    studio_id: str = Field(..., description="Studio identifier")
    target_revenue: float = Field(..., description="Target revenue requested")
    achievable_revenue: float = Field(..., description="Achievable revenue with constraints")
    achievement_rate: float = Field(..., ge=0.0, le=1.0, description="Target achievement rate (0-1)")
    recommended_levers: Dict[str, float] = Field(..., description="Optimal lever values")
    lever_changes: List[LeverChange] = Field(..., description="Detailed lever changes")
    action_plan: List[ActionItem] = Field(..., description="Prioritized action items")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Optimization confidence")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Optimization timestamp")
    ai_insights: Optional[AIInsights] = Field(None, description="AI-generated business insights about optimization")


class PredictedLever(BaseModel):
    """Predicted lever value with confidence"""
    lever_name: str = Field(..., description="Name of the lever")
    predicted_value: float = Field(..., description="Predicted value")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    value_range: Optional[List[float]] = Field(None, description="[min, max] range")


class PartialPredictionResponse(BaseModel):
    """Response for partial lever prediction"""
    model_config = {"protected_namespaces": ()}

    prediction_id: str = Field(..., description="Unique prediction identifier")
    studio_id: str = Field(..., description="Studio identifier")
    input_levers: Dict[str, float] = Field(..., description="Input lever values provided")
    predicted_levers: List[PredictedLever] = Field(..., description="Predicted lever values")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall prediction confidence")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    notes: Optional[str] = Field(None, description="Additional notes or warnings")
    ai_insights: Optional[AIInsights] = Field(None, description="AI-generated business insights about predicted levers")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Default model version")
    model_type: str = Field(..., description="Default model type")
    n_features: int = Field(..., description="Number of features in default model")
    available_models: int = Field(..., description="Number of available models")
    cached_models: int = Field(..., description="Number of currently cached models")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    timestamp: str = Field(..., description="Current timestamp")

