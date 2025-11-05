"""
Feature Service for Engineering Features from Lever Inputs
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for transforming lever inputs into engineered features"""

    def __init__(self, selected_features: List[str]):
        """
        Initialize feature service
        
        Args:
            selected_features: List of feature names selected by the model
        """
        self.selected_features = selected_features
        logger.info(f"Feature service initialized with {len(selected_features)} selected features")

    def engineer_features_from_levers(
        self, 
        levers: Dict[str, float], 
        historical_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Transform lever inputs into engineered features
        
        Args:
            levers: Dictionary of lever values
            historical_data: Recent historical data for context
            
        Returns:
            Numpy array of engineered features in correct order
        """
        logger.info("Engineering features from lever inputs")
        
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
        
        logger.info(f"Engineered {len(feature_vector)} features")
        return np.array(feature_vector).reshape(1, -1)

    def get_feature_names(self) -> List[str]:
        """Get list of selected feature names"""
        return self.selected_features

