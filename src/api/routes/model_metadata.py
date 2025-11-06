"""
API Routes for Model Metadata

Provides endpoints to retrieve model features and targets information.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.api.schemas.model_metadata import (
    ModelFeaturesResponse,
    ModelTargetsResponse,
    AvailableModelsResponse,
    FeatureMetadata,
    TargetMetadata
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for model cache manager (will be injected)
_model_cache_manager = None


def set_model_cache_manager(cache_manager):
    """Set the model cache manager instance"""
    global _model_cache_manager
    _model_cache_manager = cache_manager


def get_model_cache_manager():
    """Dependency to get model cache manager"""
    if _model_cache_manager is None:
        raise HTTPException(status_code=500, detail="Model cache manager not initialized")
    return _model_cache_manager


# Feature categorization for v2.3.0 daily models
V2_3_FEATURE_CATEGORIES = {
    # Temporal features (15)
    'day_of_week': 'temporal', 'is_monday': 'temporal', 'is_friday': 'temporal',
    'is_saturday': 'temporal', 'is_sunday': 'temporal', 'is_weekend': 'temporal',
    'month': 'temporal', 'day_of_month': 'temporal', 'week_of_year': 'temporal',
    'is_january': 'temporal', 'is_summer_prep': 'temporal', 'is_holiday': 'temporal',
    'is_holiday_week': 'temporal', 'day_of_year': 'temporal', 'quarter': 'temporal',

    # Core daily metrics (8)
    'daily_revenue': 'core', 'daily_attendance': 'core', 'total_members': 'core',
    'retention_rate': 'core', 'avg_ticket_price': 'core', 'class_attendance_rate': 'core',
    'new_members': 'core', 'staff_utilization_rate': 'core',

    # Rolling features (10)
    'revenue_7d_avg': 'rolling', 'revenue_14d_avg': 'rolling', 'revenue_30d_avg': 'rolling',
    'attendance_7d_avg': 'rolling', 'attendance_14d_avg': 'rolling', 'attendance_30d_avg': 'rolling',
    'retention_7d_avg': 'rolling', 'retention_14d_avg': 'rolling', 'retention_30d_avg': 'rolling',
    'members_30d_avg': 'rolling',

    # Growth/momentum features (4)
    'revenue_growth_7d': 'momentum', 'revenue_growth_30d': 'momentum',
    'member_growth_30d': 'momentum', 'retention_trend_30d': 'momentum',

    # Interaction features (8)
    'retention_x_ticket': 'interaction', 'attendance_x_classes': 'interaction',
    'members_x_retention': 'interaction', 'revenue_per_member': 'interaction',
    'revenue_per_class': 'interaction', 'members_per_class': 'interaction',
    'attendance_per_member': 'interaction', 'ticket_x_attendance': 'interaction',

    # Location/tier features (9)
    'studio_location_encoded': 'categorical', 'studio_size_tier_encoded': 'categorical',
    'studio_price_tier_encoded': 'categorical', 'location_urban': 'categorical',
    'location_suburban': 'categorical', 'location_rural': 'categorical',
    'size_small': 'categorical', 'size_medium': 'categorical', 'size_large': 'categorical',

    # Engineered features (21)
    'total_classes_held': 'engineered', 'upsell_rate': 'engineered',
    'avg_members_per_class': 'engineered', 'revenue_per_attendance': 'engineered',
    'new_member_rate': 'engineered', 'churn_rate': 'engineered',
    'capacity_utilization': 'engineered', 'revenue_momentum': 'engineered',
    'member_engagement': 'engineered', 'pricing_power': 'engineered',
    'growth_rate': 'engineered', 'retention_strength': 'engineered',
    'class_efficiency': 'engineered', 'member_lifetime_value': 'engineered',
    'acquisition_cost_ratio': 'engineered', 'month_sin': 'cyclical',
    'month_cos': 'cyclical', 'day_sin': 'cyclical', 'day_cos': 'cyclical',
    'revenue_volatility_30d': 'statistical', 'days_since_last_promo': 'engineered'
}

# Feature categorization for v2.2.0 monthly models
V2_2_FEATURE_CATEGORIES = {
    'retention_rate': 'core', 'avg_ticket_price': 'core', 'class_attendance_rate': 'core',
    'new_members': 'core', 'staff_utilization_rate': 'core', 'upsell_rate': 'core',
    'total_classes_held': 'core', 'total_members': 'core',
    'retention_x_ticket': 'interaction', 'attendance_x_classes': 'interaction',
    'members_x_retention': 'interaction', 'revenue_per_member': 'interaction',
    'revenue_per_class': 'interaction', 'members_per_class': 'interaction',
    'ticket_x_attendance': 'interaction'
}


@router.get("/features", response_model=ModelFeaturesResponse)
async def get_model_features(
    model_type: Optional[str] = Query(None, description="Model type: 'ridge', 'xgboost', 'lightgbm', 'neural_network'"),
    model_version: Optional[str] = Query(None, description="Model version: '2.2.0', '2.3.0'")
):
    """
    Get list of features for a specific model

    **Parameters:**
    - model_type: Model type (default: 'ridge')
    - model_version: Model version (default: '2.2.0')

    **Returns:**
    - List of all features with metadata
    - Feature categories and descriptions
    - Total feature count
    - Data granularity (daily/monthly)
    """
    try:
        cache_manager = get_model_cache_manager()

        # Validate and get defaults
        model_type, model_version = cache_manager.validate_model_request(model_type, model_version)

        # Load model artifacts to get feature list
        artifacts = cache_manager.get_or_load_model(model_type, model_version)
        feature_names = artifacts['selected_features']

        # Categorize features
        if model_version == "2.3.0":
            categories = V2_3_FEATURE_CATEGORIES
            granularity = "daily"
        else:
            categories = V2_2_FEATURE_CATEGORIES
            granularity = "monthly"

        # Build feature metadata list
        features = []
        for feature_name in feature_names:
            category = categories.get(feature_name, 'unknown')

            # Generate description based on category and name
            description = _generate_feature_description(feature_name, category)
            data_type = _infer_data_type(feature_name)

            features.append(FeatureMetadata(
                name=feature_name,
                category=category,
                description=description,
                data_type=data_type
            ))

        return ModelFeaturesResponse(
            model_type=model_type,
            model_version=model_version,
            n_features=len(feature_names),
            features=features,
            feature_names=feature_names,
            granularity=granularity
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving model features: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model features: {str(e)}")


@router.get("/targets", response_model=ModelTargetsResponse)
async def get_model_targets(
    model_type: Optional[str] = Query(None, description="Model type: 'ridge', 'xgboost', 'lightgbm', 'neural_network'"),
    model_version: Optional[str] = Query(None, description="Model version: '2.2.0', '2.3.0'")
):
    """
    Get list of prediction targets for a specific model

    **Parameters:**
    - model_type: Model type (default: 'ridge')
    - model_version: Model version (default: '2.2.0')

    **Returns:**
    - List of all prediction targets with metadata
    - Target horizons (daily, weekly, monthly)
    - Total target count
    - Prediction granularity
    """
    try:
        cache_manager = get_model_cache_manager()

        # Validate and get defaults
        model_type, model_version = cache_manager.validate_model_request(model_type, model_version)

        # Define targets based on version
        if model_version == "2.3.0":
            # v2.3.0: 14 targets (4 daily + 4 weekly + 6 monthly)
            targets = [
                # Daily targets (indices 0-3)
                TargetMetadata(name="revenue_day_1", horizon="daily", description="Revenue prediction for day 1", unit="USD", index=0),
                TargetMetadata(name="revenue_day_3", horizon="daily", description="Revenue prediction for day 3", unit="USD", index=1),
                TargetMetadata(name="revenue_day_7", horizon="daily", description="Revenue prediction for day 7", unit="USD", index=2),
                TargetMetadata(name="attendance_day_7", horizon="daily", description="Attendance prediction for day 7", unit="visits", index=3),

                # Weekly targets (indices 4-7)
                TargetMetadata(name="revenue_week_1", horizon="weekly", description="Revenue prediction for week 1 (days 1-7)", unit="USD", index=4),
                TargetMetadata(name="revenue_week_2", horizon="weekly", description="Revenue prediction for week 2 (days 8-14)", unit="USD", index=5),
                TargetMetadata(name="revenue_week_4", horizon="weekly", description="Revenue prediction for week 4 (days 22-28)", unit="USD", index=6),
                TargetMetadata(name="attendance_week_1", horizon="weekly", description="Attendance prediction for week 1", unit="visits", index=7),

                # Monthly targets (indices 8-13)
                TargetMetadata(name="revenue_month_1", horizon="monthly", description="Revenue prediction for month 1 (30 days)", unit="USD", index=8),
                TargetMetadata(name="revenue_month_2", horizon="monthly", description="Revenue prediction for month 2 (60 days)", unit="USD", index=9),
                TargetMetadata(name="revenue_month_3", horizon="monthly", description="Revenue prediction for month 3 (90 days)", unit="USD", index=10),
                TargetMetadata(name="members_month_1", horizon="monthly", description="Member count prediction for month 1", unit="members", index=11),
                TargetMetadata(name="members_month_3", horizon="monthly", description="Member count prediction for month 3", unit="members", index=12),
                TargetMetadata(name="retention_month_3", horizon="monthly", description="Retention rate prediction for month 3", unit="rate", index=13)
            ]
            horizons = ["daily", "weekly", "monthly"]
            granularity = "daily"
        else:
            # v2.2.0: 5 targets (monthly only)
            targets = [
                TargetMetadata(name="revenue_month_1", horizon="monthly", description="Revenue prediction for month 1", unit="USD", index=0),
                TargetMetadata(name="revenue_month_2", horizon="monthly", description="Revenue prediction for month 2", unit="USD", index=1),
                TargetMetadata(name="revenue_month_3", horizon="monthly", description="Revenue prediction for month 3", unit="USD", index=2),
                TargetMetadata(name="members_month_1", horizon="monthly", description="Member count prediction for month 1", unit="members", index=3),
                TargetMetadata(name="retention_month_3", horizon="monthly", description="Retention rate prediction for month 3", unit="rate", index=4)
            ]
            horizons = ["monthly"]
            granularity = "monthly"

        target_names = [t.name for t in targets]

        return ModelTargetsResponse(
            model_type=model_type,
            model_version=model_version,
            n_targets=len(targets),
            targets=targets,
            target_names=target_names,
            horizons=horizons,
            granularity=granularity
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving model targets: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model targets: {str(e)}")


@router.get("/available", response_model=AvailableModelsResponse)
async def get_available_models():
    """
    Get list of all available models

    **Returns:**
    - List of available models with metadata
    - Model types, versions, and feature counts
    - Total count of available models
    """
    try:
        cache_manager = get_model_cache_manager()
        available_models = cache_manager.list_available_models()

        return AvailableModelsResponse(
            models=available_models,
            count=len(available_models)
        )

    except Exception as e:
        logger.error(f"Error retrieving available models: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving available models: {str(e)}")


def _generate_feature_description(feature_name: str, category: str) -> str:
    """Generate human-readable description for a feature"""

    descriptions = {
        # Temporal
        'day_of_week': 'Day of the week (1-7, Monday=1)',
        'is_monday': 'Binary flag for Monday',
        'is_friday': 'Binary flag for Friday',
        'is_saturday': 'Binary flag for Saturday',
        'is_sunday': 'Binary flag for Sunday',
        'is_weekend': 'Binary flag for weekend (Saturday/Sunday)',
        'month': 'Month of the year (1-12)',
        'day_of_month': 'Day of the month (1-31)',
        'week_of_year': 'Week number in the year (1-52)',
        'is_january': 'Binary flag for January (New Year resolution period)',
        'is_summer_prep': 'Binary flag for summer preparation months (May-June)',
        'is_holiday': 'Binary flag for holiday',
        'is_holiday_week': 'Binary flag for holiday week',

        # Core
        'retention_rate': 'Member retention rate (0-1)',
        'avg_ticket_price': 'Average monthly ticket price (USD)',
        'class_attendance_rate': 'Class attendance rate (0-1)',
        'new_members': 'Number of new members',
        'staff_utilization_rate': 'Staff utilization rate (0-1)',
        'upsell_rate': 'Rate of successful upsells (0-1)',
        'total_classes_held': 'Total number of classes held',
        'total_members': 'Total member count',
        'daily_revenue': 'Daily revenue (USD)',
        'daily_attendance': 'Daily class attendance',

        # Rolling averages
        'revenue_7d_avg': '7-day rolling average revenue',
        'revenue_14d_avg': '14-day rolling average revenue',
        'revenue_30d_avg': '30-day rolling average revenue',
        'attendance_7d_avg': '7-day rolling average attendance',
        'attendance_14d_avg': '14-day rolling average attendance',
        'attendance_30d_avg': '30-day rolling average attendance',
        'retention_7d_avg': '7-day rolling average retention rate',
        'retention_14d_avg': '14-day rolling average retention rate',
        'retention_30d_avg': '30-day rolling average retention rate',
        'members_30d_avg': '30-day rolling average member count',

        # Momentum
        'revenue_growth_7d': '7-day revenue growth rate',
        'revenue_growth_30d': '30-day revenue growth rate',
        'member_growth_30d': '30-day member growth rate',
        'retention_trend_30d': '30-day retention trend',

        # Interactions
        'retention_x_ticket': 'Interaction: retention × ticket price',
        'attendance_x_classes': 'Interaction: attendance × classes',
        'members_x_retention': 'Interaction: members × retention',
        'revenue_per_member': 'Revenue per member',
        'revenue_per_class': 'Revenue per class',
        'members_per_class': 'Members per class ratio',
        'attendance_per_member': 'Attendance per member',
        'ticket_x_attendance': 'Interaction: ticket × attendance',
    }

    return descriptions.get(feature_name, f"{category.capitalize()} feature: {feature_name}")


def _infer_data_type(feature_name: str) -> str:
    """Infer data type from feature name"""

    # Binary flags
    if feature_name.startswith('is_') or feature_name.startswith('location_') or feature_name.startswith('size_'):
        return 'int'

    # Counts/integers
    if any(word in feature_name for word in ['members', 'classes', 'attendance', 'new_', 'total_']):
        if 'rate' not in feature_name and 'per' not in feature_name and 'avg' not in feature_name:
            return 'int'

    # Everything else is float
    return 'float'
