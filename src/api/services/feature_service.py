"""
Feature Service for Engineering Features from Lever Inputs

Supports both v2.2.0 (15 monthly features) and v2.3.0 (79 daily features)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .historical_data_service_daily import DailyHistoricalDataService

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for transforming lever inputs into engineered features"""

    def __init__(
        self,
        selected_features: List[str],
        daily_historical_service: Optional[DailyHistoricalDataService] = None,
        model_version: str = "2.2.0"
    ):
        """
        Initialize feature service

        Args:
            selected_features: List of feature names selected by the model
            daily_historical_service: Service for loading daily historical data (v2.3.0 only)
            model_version: Model version ("2.2.0" or "2.3.0")
        """
        self.selected_features = selected_features
        self.model_version = model_version
        self.daily_historical_service = daily_historical_service

        # Initialize daily service if needed for v2.3.0
        if model_version == "2.3.0" and daily_historical_service is None:
            logger.info("Initializing daily historical service for v2.3.0 features")
            self.daily_historical_service = DailyHistoricalDataService()

        logger.info(f"Feature service initialized for v{model_version} with {len(selected_features)} features")

    def engineer_features_from_levers(
        self,
        levers: Dict[str, float],
        historical_data: pd.DataFrame,
        studio_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Transform lever inputs into engineered features

        Args:
            levers: Dictionary of lever values
            historical_data: Recent historical data for context
            studio_id: Studio ID (required for v2.3.0)

        Returns:
            Numpy array of engineered features in correct order
        """
        logger.debug(f"Engineering features from lever inputs for v{self.model_version}")

        # Route to appropriate feature generation method
        if self.model_version == "2.3.0":
            return self._engineer_v2_3_features(levers, studio_id or "UNKNOWN")
        else:
            return self._engineer_v2_2_features(levers, historical_data)

    def _engineer_v2_2_features(
        self,
        levers: Dict[str, float],
        historical_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate v2.2.0 features (15 monthly features)

        Args:
            levers: Dictionary of lever values
            historical_data: Recent historical data for context

        Returns:
            Numpy array of engineered features
        """
        logger.debug("Generating v2.2.0 features (monthly, 15 features)")
        
        # Start with base features from levers
        features = {}
        
        # Direct lever features
        features['total_members'] = levers.get('total_members', 0)
        features['new_members'] = levers.get('new_members', 0)
        features['retention_rate'] = levers.get('retention_rate', 0.75)
        features['avg_ticket_price'] = levers.get('avg_ticket_price', 150.0)
        features['class_attendance_rate'] = levers.get('class_attendance_rate', 0.70)
        features['total_classes_held'] = levers.get('total_classes_held', 100)
        features['staff_utilization_rate'] = levers.get('staff_utilization_rate', 0.80)
        features['upsell_rate'] = levers.get('upsell_rate', 0.25)
        
        # Calculate derived features
        features['churned_members'] = int(features['total_members'] * (1 - features['retention_rate']))
        
        # Calculate revenue components (for current state)
        features['membership_revenue'] = features['total_members'] * features['avg_ticket_price']
        features['class_pack_revenue'] = features['total_classes_held'] * 20 * features['class_attendance_rate'] * 0.2 * 15
        features['retail_revenue'] = features['total_members'] * 10 * 0.3
        features['total_revenue'] = (
            features['membership_revenue'] + 
            features['class_pack_revenue'] + 
            features['retail_revenue']
        )
        
        # Add staff count (estimated from members)
        features['staff_count'] = max(5, int(features['total_members'] / 25))
        
        # Calculate total attendance
        features['total_class_attendance'] = int(features['total_classes_held'] * 20 * features['class_attendance_rate'])
        
        # Derived business metrics
        features['revenue_per_member'] = features['total_revenue'] / max(features['total_members'], 1)
        features['churn_rate'] = 1 - features['retention_rate']
        features['class_utilization'] = features['class_attendance_rate']
        features['staff_per_member'] = features['staff_count'] / max(features['total_members'], 1)
        features['estimated_ltv'] = features['avg_ticket_price'] * features['retention_rate'] * 12
        features['membership_revenue_pct'] = features['membership_revenue'] / max(features['total_revenue'], 1)
        features['class_pack_revenue_pct'] = features['class_pack_revenue'] / max(features['total_revenue'], 1)
        
        # Lagged features from historical data
        if historical_data is not None and len(historical_data) > 0:
            features['prev_month_revenue'] = float(historical_data['total_revenue'].iloc[-1])
            features['prev_month_members'] = int(historical_data['total_members'].iloc[-1])
            
            # Rolling features (last 3 months)
            if len(historical_data) >= 3:
                features['3m_avg_retention'] = float(historical_data['retention_rate'].tail(3).mean())
                features['3m_avg_revenue'] = float(historical_data['total_revenue'].tail(3).mean())
                features['3m_avg_attendance'] = float(historical_data['class_attendance_rate'].tail(3).mean())
                features['3m_std_revenue'] = float(historical_data['total_revenue'].tail(3).std())
                features['revenue_momentum'] = float(historical_data['total_revenue'].ewm(span=3).mean().iloc[-1])
            else:
                # Use single values if less than 3 months
                features['3m_avg_retention'] = features['retention_rate']
                features['3m_avg_revenue'] = features['prev_month_revenue']
                features['3m_avg_attendance'] = features['class_attendance_rate']
                features['3m_std_revenue'] = 0.0
                features['revenue_momentum'] = features['prev_month_revenue']
            
            # Month-over-month growth
            features['mom_revenue_growth'] = (
                (features['total_revenue'] - features['prev_month_revenue']) / 
                max(features['prev_month_revenue'], 1)
            )
            features['mom_member_growth'] = (
                (features['total_members'] - features['prev_month_members']) / 
                max(features['prev_month_members'], 1)
            )
        else:
            # Fallback values if no historical data
            features['prev_month_revenue'] = features['total_revenue']
            features['prev_month_members'] = features['total_members']
            features['3m_avg_retention'] = features['retention_rate']
            features['3m_avg_revenue'] = features['total_revenue']
            features['3m_avg_attendance'] = features['class_attendance_rate']
            features['3m_std_revenue'] = 0.0
            features['revenue_momentum'] = features['total_revenue']
            features['mom_revenue_growth'] = 0.0
            features['mom_member_growth'] = 0.0
        
        # Interaction features
        features['retention_x_ticket'] = features['retention_rate'] * features['avg_ticket_price']
        features['attendance_x_classes'] = features['class_attendance_rate'] * features['total_classes_held']
        features['upsell_x_members'] = features['upsell_rate'] * features['total_members']
        features['staff_util_x_members'] = features['staff_utilization_rate'] * features['total_members']
        
        # Cyclical features (if we have month info from historical data)
        if historical_data is not None and len(historical_data) > 0 and 'month_year' in historical_data.columns:
            last_date = pd.to_datetime(historical_data['month_year'].iloc[-1])
            month = last_date.month
        else:
            # Use current month as fallback
            month = pd.Timestamp.now().month
        
        features['month_index'] = month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        features['is_january'] = 1 if month == 1 else 0
        features['is_summer'] = 1 if month in [6, 7, 8] else 0
        features['is_fall'] = 1 if month == 9 else 0
        
        # Extract only selected features in correct order
        feature_vector = []
        for feature_name in self.selected_features:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                logger.warning(f"Feature '{feature_name}' not found, using 0.0")
                feature_vector.append(0.0)
        
        logger.debug(f"Engineered {len(feature_vector)} features")
        return np.array(feature_vector).reshape(1, -1)

    def _engineer_v2_3_features(
        self,
        levers: Dict[str, float],
        studio_id: str
    ) -> np.ndarray:
        """
        Generate v2.3.0 features (79 daily features)

        Args:
            levers: Dictionary of lever values
            studio_id: Studio identifier for historical data lookup

        Returns:
            Numpy array of 79 engineered features
        """
        logger.debug(f"Generating v2.3.0 features (daily, 79 features) for studio {studio_id}")

        features = {}

        # Get daily historical context
        if self.daily_historical_service:
            rolling_7d = self.daily_historical_service.get_rolling_metrics(studio_id, window=7)
            rolling_14d = self.daily_historical_service.get_rolling_metrics(studio_id, window=14)
            rolling_30d = self.daily_historical_service.get_rolling_metrics(studio_id, window=30)
            day_context = self.daily_historical_service.get_day_of_week_context()
            seasonal_context = self.daily_historical_service.get_seasonal_context()
            latest_state = self.daily_historical_service.get_latest_studio_state(studio_id)
        else:
            # Fallback if service not available
            rolling_7d = {'rolling_revenue': 2000.0, 'rolling_attendance': 85.0, 'rolling_retention': 0.75}
            rolling_14d = {'rolling_revenue': 2000.0, 'rolling_attendance': 85.0}
            rolling_30d = {'rolling_revenue': 2000.0, 'rolling_attendance': 85.0, 'rolling_retention': 0.75}
            day_context = {'day_of_week': 1, 'is_monday': 1, 'is_weekend': 0, 'month': 1}
            seasonal_context = {'is_january': 0, 'is_summer_prep': 0, 'is_holiday': 0}
            latest_state = {'studio_location': 'urban', 'studio_size_tier': 'medium', 'studio_price_tier': 'medium'}

        # 1. Temporal Features (15 features)
        features['day_of_week'] = day_context.get('day_of_week', 1)
        features['is_weekend'] = day_context.get('is_weekend', 0)
        features['is_monday'] = day_context.get('is_monday', 0)
        features['is_friday'] = day_context.get('is_friday', 0)
        features['is_saturday'] = day_context.get('is_saturday', 0)
        features['is_sunday'] = day_context.get('is_sunday', 0)
        features['is_holiday'] = seasonal_context.get('is_holiday', 0)
        features['is_holiday_week'] = 0  # Simplified
        features['is_january'] = seasonal_context.get('is_january', 0)
        features['is_summer_prep'] = seasonal_context.get('is_summer_prep', 0)
        features['month'] = day_context.get('month', 1)
        features['day_of_month'] = day_context.get('day_of_month', 1)
        features['week_of_year'] = day_context.get('week_of_year', 1)

        # 2. Core Daily Metrics (8 features) - derived from levers
        total_members = levers.get('total_members', 200)
        new_members = levers.get('new_members', 5)
        retention_rate = levers.get('retention_rate', 0.75)
        avg_ticket_price = levers.get('avg_ticket_price', 150.0)
        class_attendance_rate = levers.get('class_attendance_rate', 0.70)
        total_classes_held = levers.get('total_classes_held', 8)  # Daily classes
        staff_utilization_rate = levers.get('staff_utilization_rate', 0.80)
        upsell_rate = levers.get('upsell_rate', 0.25)

        # Convert monthly levers to daily estimates
        daily_attendance = int(total_members * class_attendance_rate * 0.4)  # ~40% of members attend daily
        daily_membership_revenue = (avg_ticket_price / 30) * total_members  # Monthly price / 30 days
        daily_class_pack_revenue = daily_attendance * 15 * 0.15  # 15% drop-ins at $15
        daily_retail_revenue = total_members * 0.10  # Small daily retail
        daily_revenue = daily_membership_revenue + daily_class_pack_revenue + daily_retail_revenue

        features['daily_attendance'] = daily_attendance
        features['daily_revenue'] = daily_revenue
        features['daily_membership_revenue'] = daily_membership_revenue
        features['daily_class_pack_revenue'] = daily_class_pack_revenue
        features['daily_retail_revenue'] = daily_retail_revenue

        # 3. Member Metrics (5 features)
        features['total_members'] = total_members
        features['new_members'] = new_members
        features['churned_members'] = int(total_members * (1 - retention_rate) / 30)  # Daily churn
        features['retention_rate'] = retention_rate
        features['avg_ticket_price'] = avg_ticket_price

        # 4. Class Metrics (4 features)
        features['total_classes_held'] = total_classes_held
        features['total_class_attendance'] = daily_attendance
        features['class_attendance_rate'] = class_attendance_rate
        features['staff_count'] = max(3, int(total_members / 30))  # Rough staff estimate
        features['staff_utilization_rate'] = staff_utilization_rate
        features['upsell_rate'] = upsell_rate

        # 5. Studio Characteristics (7 features)
        location = latest_state.get('studio_location', 'urban') if latest_state else 'urban'
        size = latest_state.get('studio_size_tier', 'medium') if latest_state else 'medium'
        price_tier = latest_state.get('studio_price_tier', 'medium') if latest_state else 'medium'

        features['studio_location'] = location
        features['studio_size_tier'] = size
        features['studio_price_tier'] = price_tier
        # Encode as binary features
        features['location_urban'] = 1 if location == 'urban' else 0
        features['size_large'] = 1 if size == 'large' else 0
        features['size_medium'] = 1 if size == 'medium' else 0
        features['size_small'] = 1 if size == 'small' else 0
        features['price_high'] = 1 if price_tier == 'high' else 0
        features['price_medium'] = 1 if price_tier == 'medium' else 0
        features['price_low'] = 1 if price_tier == 'low' else 0

        # 6. Rolling Features (10 features)
        features['rolling_7d_revenue'] = rolling_7d.get('rolling_revenue', daily_revenue * 7)
        features['rolling_7d_attendance'] = rolling_7d.get('rolling_attendance', daily_attendance * 7)
        features['rolling_7d_retention'] = rolling_7d.get('rolling_retention', retention_rate)
        features['rolling_14d_revenue'] = rolling_14d.get('rolling_revenue', daily_revenue * 14)
        features['rolling_14d_attendance'] = rolling_14d.get('rolling_attendance', daily_attendance * 14)
        features['rolling_30d_revenue'] = rolling_30d.get('rolling_revenue', daily_revenue * 30)
        features['rolling_30d_attendance'] = rolling_30d.get('rolling_attendance', daily_attendance * 30)
        features['rolling_30d_retention'] = rolling_30d.get('rolling_retention', retention_rate)

        # 7. Momentum Features (4 features)
        features['day_momentum_revenue'] = features['rolling_7d_revenue'] / 7  # EMA approximation
        features['day_momentum_attendance'] = features['rolling_7d_attendance'] / 7
        features['weekday_vs_weekend_ratio'] = 1.2 if features['is_weekend'] == 0 else 0.8
        features['revenue_vs_7d_avg'] = daily_revenue / (features['rolling_7d_revenue'] / 7 + 1)

        # 8. Growth Indicators (6 features)
        features['dod_revenue_growth'] = rolling_7d.get('revenue_growth', 0.02) / 7  # Daily growth
        features['wow_revenue_growth'] = rolling_7d.get('revenue_growth', 0.02)
        features['dod_attendance_growth'] = 0.01  # Estimate
        features['attendance_vs_7d_avg'] = daily_attendance / (features['rolling_7d_attendance'] / 7 + 1)
        features['revenue_trend_momentum'] = features['day_momentum_revenue']

        # 9. Derived Business Metrics (8 features)
        features['revenue_per_member'] = daily_revenue / max(total_members, 1)
        features['churn_rate'] = 1 - retention_rate
        features['class_utilization'] = class_attendance_rate
        features['staff_per_member'] = features['staff_count'] / max(total_members, 1)
        features['estimated_ltv'] = avg_ticket_price * retention_rate * 12
        features['membership_revenue_pct'] = daily_membership_revenue / max(daily_revenue, 1)
        features['class_pack_revenue_pct'] = daily_class_pack_revenue / max(daily_revenue, 1)
        features['retail_revenue_pct'] = daily_retail_revenue / max(daily_revenue, 1)

        # 10. Interaction Features (8 features)
        features['retention_x_ticket'] = retention_rate * avg_ticket_price
        features['attendance_x_classes'] = class_attendance_rate * total_classes_held
        features['upsell_x_members'] = upsell_rate * total_members
        features['staff_util_x_members'] = staff_utilization_rate * total_members
        features['revenue_x_retention'] = daily_revenue * retention_rate
        features['members_x_classes'] = total_members * total_classes_held
        features['attendance_x_price'] = daily_attendance * avg_ticket_price
        features['utilization_x_classes'] = class_attendance_rate * total_classes_held

        # 11. Cyclical Encodings (4 features)
        month = features['month']
        day_of_week = features['day_of_week']
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

        # Extract features in the correct order
        feature_vector = []
        for feature_name in self.selected_features:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                logger.warning(f"Feature '{feature_name}' not found in v2.3.0 features, using 0.0")
                feature_vector.append(0.0)

        logger.debug(f"Generated {len(feature_vector)} v2.3.0 features")
        return np.array(feature_vector).reshape(1, -1)

    def get_feature_names(self) -> List[str]:
        """Get list of selected feature names"""
        return self.selected_features

