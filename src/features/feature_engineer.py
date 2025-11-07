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

    def __init__(self, multi_studio=False):
        self.feature_names = []
        self.multi_studio = multi_studio

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
        
        # Detect if this is multi-studio data
        if 'studio_id' in df.columns:
            self.multi_studio = True
            logger.info("Multi-studio mode detected")

        # 1. Direct lever features (already in data)
        # Note: staff_utilization_rate is optional - will use staff_count as fallback
        direct_features = [
            'retention_rate',
            'avg_ticket_price',
            'class_attendance_rate',
            'new_members',
            'staff_utilization_rate',  # Optional
            'upsell_rate',
            'total_classes_held',
            'total_members'
        ]

        # 2. Studio-level features (if multi-studio)
        if self.multi_studio:
            df = self._add_studio_features(df)

        # 3. Temporal features
        df = self._add_temporal_features(df)

        # 4. Derived business metrics
        df = self._add_derived_features(df)

        # 5. Lagged features (historical context, per studio if multi-studio)
        df = self._add_lagged_features(df)

        # 6. Rolling statistics (per studio if multi-studio)
        df = self._add_rolling_features(df)

        # 7. Interaction features
        df = self._add_interaction_features(df)

        # 8. Cyclical encoding for seasonality
        df = self._add_cyclical_features(df)
        
        # 9. Add target variables (for training only)
        if is_training:
            df = self._add_target_variables(df)

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

    def _add_studio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add studio-level features for multi-studio data"""
        logger.info("Adding studio-level features")
        
        # Studio age (months since first record for each studio)
        if 'month_year' in df.columns:
            df['month_year_dt'] = pd.to_datetime(df['month_year'])
            df['studio_age'] = df.groupby('studio_id')['month_year_dt'].rank(method='dense').astype(int)
            df = df.drop('month_year_dt', axis=1)
        
        # Encode studio categorical features
        if 'studio_location' in df.columns:
            df['location_urban'] = (df['studio_location'] == 'urban').astype(int)
        
        if 'studio_size_tier' in df.columns:
            df['size_small'] = (df['studio_size_tier'] == 'small').astype(int)
            df['size_medium'] = (df['studio_size_tier'] == 'medium').astype(int)
            df['size_large'] = (df['studio_size_tier'] == 'large').astype(int)
        
        if 'studio_price_tier' in df.columns:
            df['price_low'] = (df['studio_price_tier'] == 'low').astype(int)
            df['price_medium'] = (df['studio_price_tier'] == 'medium').astype(int)
            df['price_high'] = (df['studio_price_tier'] == 'high').astype(int)
        
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

        if self.multi_studio:
            # Apply shifts per studio to avoid leakage across studios
            df['prev_month_revenue'] = df.groupby('studio_id')['total_revenue'].shift(1).round(2)
            df['prev_month_members'] = df.groupby('studio_id')['total_members'].shift(1)
        else:
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

        if self.multi_studio:
            # Apply rolling features per studio to avoid leakage across studios
            # 3-month rolling averages (EXCLUDING current month to prevent leakage)
            df['3m_avg_retention'] = df.groupby('studio_id')['retention_rate'].apply(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            ).reset_index(level=0, drop=True).round(2)
            df['3m_avg_revenue'] = df.groupby('studio_id')['total_revenue'].apply(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            ).reset_index(level=0, drop=True).round(2)
            df['3m_avg_attendance'] = df.groupby('studio_id')['class_attendance_rate'].apply(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            ).reset_index(level=0, drop=True).round(2)
            
            # 3-month rolling std (volatility) - EXCLUDING current month
            df['3m_std_revenue'] = df.groupby('studio_id')['total_revenue'].apply(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
            ).reset_index(level=0, drop=True).round(2)
            
            # Revenue momentum (EMA) - EXCLUDING current month
            df['revenue_momentum'] = df.groupby('studio_id')['total_revenue'].apply(
                lambda x: x.shift(1).ewm(span=3).mean()
            ).reset_index(level=0, drop=True).round(2)
        else:
            # 3-month rolling averages (EXCLUDING current month to prevent leakage)
            df['3m_avg_retention'] = df['retention_rate'].shift(1).rolling(window=3).mean().round(2)
            df['3m_avg_revenue'] = df['total_revenue'].shift(1).rolling(window=3).mean().round(2)
            df['3m_avg_attendance'] = df['class_attendance_rate'].shift(1).rolling(window=3).mean().round(2)

            # 3-month rolling std (volatility) - EXCLUDING current month
            df['3m_std_revenue'] = df['total_revenue'].shift(1).rolling(window=3).std().round(2)

            # Revenue momentum (EMA) - EXCLUDING current month
            df['revenue_momentum'] = df['total_revenue'].shift(1).ewm(span=3).mean().round(2)

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

        # Staff utilization × Members (only if staff_utilization_rate exists)
        if 'staff_utilization_rate' in df.columns:
            df['staff_util_x_members'] = (df['staff_utilization_rate'] * df['total_members']).round(2)
        elif 'staff_count' in df.columns:
            # Use staff_count as alternative if available
            df['staff_util_x_members'] = (df['staff_count'] * df['total_members']).round(2)
        else:
            # Set to a default value if neither is available
            df['staff_util_x_members'] = 0

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for month seasonality"""
        logger.info("Adding cyclical features")

        # Create month_index from month_year if it doesn't exist
        if 'month_index' not in df.columns:
            if 'month_year' in df.columns:
                df['month_index'] = pd.to_datetime(df['month_year']).dt.month
            else:
                logger.warning("No 'month_year' or 'month_index' column found, skipping cyclical features")
                return df

        # Sin/cos encoding for month (captures seasonality)
        df['month_sin'] = np.sin(2 * np.pi * df['month_index'] / 12).round(2)
        df['month_cos'] = np.cos(2 * np.pi * df['month_index'] / 12).round(2)

        # Binary flags for key months
        df['is_january'] = (df['month_index'] == 1).astype(int)
        df['is_summer'] = df['month_index'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = (df['month_index'] == 9).astype(int)

        return df
    
    def _add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target variables by shifting future values
        Targets: revenue_month_1, revenue_month_2, revenue_month_3, 
                 member_count_month_3, retention_rate_month_3
        """
        logger.info("Adding target variables")
        
        if self.multi_studio:
            # For multi-studio data, shift within each studio group
            df = df.sort_values(['studio_id', 'month_year']).reset_index(drop=True)
            
            # Create target columns by shifting future values
            df['revenue_month_1'] = df.groupby('studio_id')['total_revenue'].shift(-1)
            df['revenue_month_2'] = df.groupby('studio_id')['total_revenue'].shift(-2)
            df['revenue_month_3'] = df.groupby('studio_id')['total_revenue'].shift(-3)
            df['member_count_month_3'] = df.groupby('studio_id')['total_members'].shift(-3)
            df['retention_rate_month_3'] = df.groupby('studio_id')['retention_rate'].shift(-3)
        else:
            # For single studio, simple shift
            df = df.sort_values('month_year').reset_index(drop=True)
            df['revenue_month_1'] = df['total_revenue'].shift(-1)
            df['revenue_month_2'] = df['total_revenue'].shift(-2)
            df['revenue_month_3'] = df['total_revenue'].shift(-3)
            df['member_count_month_3'] = df['total_members'].shift(-3)
            df['retention_rate_month_3'] = df['retention_rate'].shift(-3)
        
        logger.info("Target variables added successfully")
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
