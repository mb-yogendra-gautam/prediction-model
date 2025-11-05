"""
Feature Engineering Module for Studio Revenue Simulator

Transforms raw features into model-ready features including:
- Derived business metrics
- Lagged features for historical context
- Rolling statistics
- Interaction features
- Cyclical encodings for seasonality
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transform raw features into model-ready features"""

    def __init__(self):
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame,
                         is_training: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering transformations

        Args:
            df: Raw data with lever values and historical context
            is_training: Whether this is for training (includes targets)

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")

        df = df.copy()

        # 1. Direct lever features (already in data)
        direct_features = [
            'retention_rate',
            'avg_ticket_price',
            'class_attendance_rate',
            'new_members',
            'staff_utilization_rate',
            'upsell_rate',
            'total_classes_held',
            'total_members'
        ]

        # 2. Temporal features
        df = self._add_temporal_features(df)

        # 3. Derived business metrics
        df = self._add_derived_features(df)

        # 4. Lagged features (historical context)
        df = self._add_lagged_features(df)

        # 5. Rolling statistics
        df = self._add_rolling_features(df)

        # 6. Interaction features
        df = self._add_interaction_features(df)

        # 7. Cyclical encoding for seasonality
        df = self._add_cyclical_features(df)

        # Remove rows with NaN from rolling/lagged features
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows due to NaN in engineered features")

        # Store feature names
        if is_training:
            target_cols = [
                'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
                'member_count_month_3', 'retention_rate_month_3'
            ]
            self.feature_names = [col for col in df.columns
                                 if col not in target_cols + ['month_year', 'split']]

        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add month and year index features"""
        # Already have month_index and year_index from data generation
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived business metrics"""
        logger.info("Adding derived features")

        # Revenue per member
        df['revenue_per_member'] = (df['total_revenue'] / df['total_members']).round(2)

        # Member churn rate
        df['churn_rate'] = (1 - df['retention_rate']).round(2)

        # Class utilization (assuming 20 person capacity per class)
        df['class_utilization'] = (df['total_class_attendance'] / (df['total_classes_held'] * 20)).round(2)

        # Staff per member ratio
        df['staff_per_member'] = (df['staff_count'] / df['total_members']).round(2)

        # LTV estimate
        df['estimated_ltv'] = (df['avg_ticket_price'] * df['retention_rate'] * 12).round(2)

        # Revenue mix percentages
        df['membership_revenue_pct'] = (df['membership_revenue'] / df['total_revenue']).round(2)
        df['class_pack_revenue_pct'] = (df['class_pack_revenue'] / df['total_revenue']).round(2)

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for historical context"""
        logger.info("Adding lagged features")

        # Previous month revenue
        df['prev_month_revenue'] = df['total_revenue'].shift(1).round(2)

        # Previous month members
        df['prev_month_members'] = df['total_members'].shift(1)

        # Month-over-month growth
        df['mom_revenue_growth'] = (
            (df['total_revenue'] - df['prev_month_revenue']) / df['prev_month_revenue']
        ).round(2)

        df['mom_member_growth'] = (
            (df['total_members'] - df['prev_month_members']) / df['prev_month_members']
        ).round(2)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics"""
        logger.info("Adding rolling features")

        # 3-month rolling averages
        df['3m_avg_retention'] = df['retention_rate'].rolling(window=3).mean().round(2)
        df['3m_avg_revenue'] = df['total_revenue'].rolling(window=3).mean().round(2)
        df['3m_avg_attendance'] = df['class_attendance_rate'].rolling(window=3).mean().round(2)

        # 3-month rolling std (volatility)
        df['3m_std_revenue'] = df['total_revenue'].rolling(window=3).std().round(2)

        # Revenue momentum (EMA)
        df['revenue_momentum'] = df['total_revenue'].ewm(span=3).mean().round(2)

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        logger.info("Adding interaction features")

        # Retention × Ticket Price
        df['retention_x_ticket'] = (df['retention_rate'] * df['avg_ticket_price']).round(2)

        # Attendance × Classes
        df['attendance_x_classes'] = (df['class_attendance_rate'] * df['total_classes_held']).round(2)

        # Upsell × Members
        df['upsell_x_members'] = (df['upsell_rate'] * df['total_members']).round(2)

        # Staff utilization × Members
        df['staff_util_x_members'] = (df['staff_utilization_rate'] * df['total_members']).round(2)

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for month seasonality"""
        logger.info("Adding cyclical features")

        # Sin/cos encoding for month (captures seasonality)
        df['month_sin'] = np.sin(2 * np.pi * df['month_index'] / 12).round(2)
        df['month_cos'] = np.cos(2 * np.pi * df['month_index'] / 12).round(2)

        # Binary flags for key months
        df['is_january'] = (df['month_index'] == 1).astype(int)
        df['is_summer'] = df['month_index'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = (df['month_index'] == 9).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names

    def transform_for_prediction(self, lever_inputs: Dict[str, float],
                                 historical_data: pd.DataFrame) -> np.ndarray:
        """
        Transform lever inputs into feature vector for prediction

        Args:
            lever_inputs: Dictionary of lever values
            historical_data: Recent historical data for context

        Returns:
            Feature vector ready for model prediction
        """
        # Create a single-row dataframe
        feature_dict = lever_inputs.copy()

        # Add historical context
        feature_dict['prev_month_revenue'] = historical_data.iloc[-1]['total_revenue']
        feature_dict['3m_avg_retention'] = historical_data.tail(3)['retention_rate'].mean()
        feature_dict['3m_avg_revenue'] = historical_data.tail(3)['total_revenue'].mean()
        feature_dict['revenue_momentum'] = historical_data['total_revenue'].ewm(span=3).mean().iloc[-1]

        # Add derived features
        feature_dict['revenue_per_member'] = (
            lever_inputs['avg_ticket_price'] * lever_inputs['retention_rate']
        )
        feature_dict['churn_rate'] = 1 - lever_inputs['retention_rate']

        # Add interaction features
        feature_dict['retention_x_ticket'] = (
            lever_inputs['retention_rate'] * lever_inputs['avg_ticket_price']
        )

        # Convert to array in correct order
        feature_vector = np.array([feature_dict[name] for name in self.feature_names])

        return feature_vector.reshape(1, -1)
