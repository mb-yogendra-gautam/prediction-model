"""
Model Metadata Schemas

Schemas for endpoints that return model features and targets information.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class FeatureMetadata(BaseModel):
    """Metadata for a single feature"""
    name: str = Field(..., description="Feature name")
    category: str = Field(..., description="Feature category (core, temporal, rolling, interaction, etc.)")
    description: Optional[str] = Field(None, description="Human-readable description")
    data_type: str = Field(..., description="Data type (float, int, etc.)")


class TargetMetadata(BaseModel):
    """Metadata for a single prediction target"""
    name: str = Field(..., description="Target name")
    horizon: str = Field(..., description="Prediction horizon (daily, weekly, monthly)")
    description: str = Field(..., description="Human-readable description")
    unit: str = Field(..., description="Unit of measurement (USD, members, rate, visits, etc.)")
    index: int = Field(..., description="Index in model output array")


class ModelFeaturesResponse(BaseModel):
    """Response containing model features information"""
    model_type: str = Field(..., description="Model type (ridge, xgboost, etc.)")
    model_version: str = Field(..., description="Model version (2.2.0, 2.3.0, etc.)")
    n_features: int = Field(..., description="Total number of features")
    features: List[FeatureMetadata] = Field(..., description="List of all features")
    feature_names: List[str] = Field(..., description="List of feature names only")
    granularity: str = Field(..., description="Data granularity (monthly, daily)")


class ModelTargetsResponse(BaseModel):
    """Response containing model prediction targets information"""
    model_type: str = Field(..., description="Model type (ridge, xgboost, etc.)")
    model_version: str = Field(..., description="Model version (2.2.0, 2.3.0, etc.)")
    n_targets: int = Field(..., description="Total number of prediction targets")
    targets: List[TargetMetadata] = Field(..., description="List of all prediction targets")
    target_names: List[str] = Field(..., description="List of target names only")
    horizons: List[str] = Field(..., description="Available prediction horizons")
    granularity: str = Field(..., description="Prediction granularity (monthly, daily)")


class AvailableModelsResponse(BaseModel):
    """Response containing list of available models"""
    models: List[dict] = Field(..., description="List of available models with metadata")
    count: int = Field(..., description="Total number of available models")
