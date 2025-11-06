"""
Request Schemas for API Endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import date


class LeverInputs(BaseModel):
    """Input levers for forward prediction"""
    retention_rate: float = Field(..., ge=0.5, le=1.0, description="Member retention rate (0.5-1.0)")
    avg_ticket_price: float = Field(..., ge=50.0, le=500.0, description="Average monthly ticket price ($50-$500)")
    class_attendance_rate: float = Field(..., ge=0.4, le=1.0, description="Class attendance rate (0.4-1.0)")
    new_members: int = Field(..., ge=0, le=100, description="New members per month (0-100)")
    staff_utilization_rate: float = Field(..., ge=0.6, le=1.0, description="Staff utilization rate (0.6-1.0)")
    upsell_rate: float = Field(..., ge=0.0, le=0.5, description="Upsell rate (0.0-0.5)")
    total_classes_held: int = Field(..., ge=50, le=500, description="Total classes per month (50-500)")
    total_members: int = Field(..., ge=50, description="Current total member count (minimum 50)")


class ForwardPredictionRequest(BaseModel):
    """Request for forward prediction (levers → revenue)"""
    studio_id: str = Field(..., description="Unique studio identifier")
    levers: LeverInputs
    projection_months: int = Field(default=3, ge=1, le=12, description="Number of months to project (1-12)")
    model_type: Optional[str] = Field(default=None, description="Model type: 'ridge', 'xgboost', 'lightgbm', 'neural_network' (default: 'ridge')")
    model_version: Optional[str] = Field(default=None, description="Model version: '2.2.0', '2.3.0' (default: '2.2.0')")
    horizons: Optional[List[str]] = Field(
        default=['monthly'],
        description="Prediction horizons for v2.3.0 models: ['daily', 'weekly', 'monthly']. For v2.2.0, only 'monthly' is supported."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "studio_id": "STU001",
                "levers": {
                    "retention_rate": 0.75,
                    "avg_ticket_price": 150.0,
                    "class_attendance_rate": 0.70,
                    "new_members": 25,
                    "staff_utilization_rate": 0.85,
                    "upsell_rate": 0.25,
                    "total_classes_held": 120,
                    "total_members": 200
                },
                "projection_months": 3,
                "model_type": "xgboost",
                "model_version": "2.3.0",
                "horizons": ["daily", "weekly", "monthly"]
            }
        }


class OptimizationConstraints(BaseModel):
    """Constraints for inverse optimization"""
    max_retention_increase: Optional[float] = Field(0.05, description="Maximum retention rate increase")
    max_ticket_increase: Optional[float] = Field(20.0, description="Maximum ticket price increase ($)")
    max_new_members_increase: Optional[int] = Field(10, description="Maximum new member increase")
    min_retention: Optional[float] = Field(0.5, description="Minimum retention rate")
    max_retention: Optional[float] = Field(1.0, description="Maximum retention rate")
    min_ticket_price: Optional[float] = Field(50.0, description="Minimum ticket price")
    max_ticket_price: Optional[float] = Field(500.0, description="Maximum ticket price")


class InversePredictionRequest(BaseModel):
    """Request for inverse prediction (target revenue → optimal levers)"""
    studio_id: str = Field(..., description="Unique studio identifier")
    target_revenue: float = Field(..., gt=0, description="Target revenue to achieve")
    current_state: LeverInputs = Field(..., description="Current state of all levers")
    constraints: Optional[OptimizationConstraints] = Field(None, description="Optimization constraints")
    target_months: int = Field(default=3, ge=1, le=12, description="Time horizon for target (1-12 months)")
    model_type: Optional[str] = Field(default=None, description="Model type: 'ridge', 'xgboost', 'lightgbm', 'neural_network' (default: 'ridge')")
    model_version: Optional[str] = Field(default=None, description="Model version: '2.2.0', '2.3.0' (default: '2.2.0')")

    class Config:
        json_schema_extra = {
            "example": {
                "studio_id": "STU001",
                "target_revenue": 35000.0,
                "current_state": {
                    "retention_rate": 0.70,
                    "avg_ticket_price": 140.0,
                    "class_attendance_rate": 0.65,
                    "new_members": 20,
                    "staff_utilization_rate": 0.80,
                    "upsell_rate": 0.20,
                    "total_classes_held": 100,
                    "total_members": 180
                },
                "constraints": {
                    "max_retention_increase": 0.05,
                    "max_ticket_increase": 20.0,
                    "max_new_members_increase": 10
                },
                "target_months": 3,
                "model_type": "ridge",
                "model_version": "2.2.0"
            }
        }


class PartialPredictionRequest(BaseModel):
    """Request for partial lever prediction (subset of levers → remaining levers)"""
    studio_id: str = Field(..., description="Unique studio identifier")
    input_levers: Dict[str, float] = Field(..., description="Known lever values")
    output_levers: List[str] = Field(..., description="Lever names to predict")
    model_type: Optional[str] = Field(default=None, description="Model type: 'ridge', 'xgboost', 'lightgbm', 'neural_network' (default: 'ridge')")
    model_version: Optional[str] = Field(default=None, description="Model version: '2.2.0', '2.3.0' (default: '2.2.0')")

    class Config:
        json_schema_extra = {
            "example": {
                "studio_id": "STU001",
                "input_levers": {
                    "retention_rate": 0.75,
                    "avg_ticket_price": 150.0,
                    "total_members": 200
                },
                "output_levers": [
                    "class_attendance_rate",
                    "new_members",
                    "total_revenue"
                ],
                "model_type": "ridge",
                "model_version": "2.2.0"
            }
        }

