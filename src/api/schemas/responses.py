"""
Response Schemas for API Endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class MonthlyPrediction(BaseModel):
    """Prediction for a single month"""
    month: int = Field(..., description="Month number (1, 2, 3, etc.)")
    revenue: float = Field(..., description="Predicted revenue")
    member_count: int = Field(..., description="Predicted member count")
    retention_rate: Optional[float] = Field(None, description="Predicted retention rate")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")


class ForwardPredictionResponse(BaseModel):
    """Response for forward prediction"""
    model_config = {"protected_namespaces": ()}
    
    scenario_id: str = Field(..., description="Unique scenario identifier")
    studio_id: str = Field(..., description="Studio identifier")
    predictions: List[MonthlyPrediction] = Field(..., description="Monthly predictions")
    total_projected_revenue: float = Field(..., description="Sum of projected revenue")
    average_confidence: float = Field(..., description="Average confidence score")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


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


class HealthCheckResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Loaded model version")
    model_type: str = Field(..., description="Model type")
    n_features: int = Field(..., description="Number of features")
    available_versions: List[str] = Field(..., description="All available model versions")
    timestamp: str = Field(..., description="Current timestamp")

