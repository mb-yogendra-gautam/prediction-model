"""
Daily Data Feature Engineering Runner

Processes the daily studio data and adds additional engineered features
(most rolling features already calculated in data generator).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyFeatureEngineer:
    """Engineer features for daily studio data"""

    def __init__(self):
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering for daily data

        Args:
            df: Daily data with existing rolling features

        Returns:
            DataFrame with additional engineered features
        """
        logger.info(f"Starting feature engineering for {len(df)} daily records...")

        df = df.copy()

        # 1. Studio-level features
        df = self._add_studio_features(df)

        # 2. Additional derived features
        df = self._add_derived_features(df)

        # 3. Interaction features
        df = self._add_interaction_features(df)

        # 4. Cyclical encoding for month
        df = self._add_cyclical_features(df)

        # 5. Day-level momentum features (already have some rolling)
        df = self._add_additional_temporal_features(df)

        # Remove rows with NaN (from targets at the end)
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        logger.info(f"Removed {removed_rows} rows due to NaN in future targets")

        # Define feature columns (exclude targets and metadata)
        target_cols = [
            # Daily targets
            'revenue_day_1', 'revenue_day_3', 'revenue_day_7', 'attendance_day_7',
            # Weekly targets
            'revenue_week_1', 'revenue_week_2', 'revenue_week_4', 'attendance_week_1',
            # Monthly targets
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_1', 'member_count_month_3', 'retention_rate_month_3'
        ]

        exclude_cols = target_cols + ['studio_id', 'date', 'split',
                                      'studio_location', 'studio_size_tier', 'studio_price_tier']

        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")

        return df

    def _add_studio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add studio-level features"""
        logger.info("Adding studio-level features...")

        # Studio age (days since first record for each studio)
        df['date_dt'] = pd.to_datetime(df['date'])
        df['studio_age_days'] = df.groupby('studio_id')['date_dt'].rank(method='dense').astype(int)
        df = df.drop('date_dt', axis=1)

        # Encode studio categorical features
        df['location_urban'] = (df['studio_location'] == 'urban').astype(int)

        df['size_small'] = (df['studio_size_tier'] == 'small').astype(int)
        df['size_medium'] = (df['studio_size_tier'] == 'medium').astype(int)
        df['size_large'] = (df['studio_size_tier'] == 'large').astype(int)

        df['price_low'] = (df['studio_price_tier'] == 'low').astype(int)
        df['price_medium'] = (df['studio_price_tier'] == 'medium').astype(int)
        df['price_high'] = (df['studio_price_tier'] == 'high').astype(int)

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived business metrics for daily data"""
        logger.info("Adding derived features...")

        # Revenue per member (daily average)
        df['revenue_per_member'] = (df['daily_revenue'] / (df['total_members'] + 1)).round(2)

        # Member churn rate
        df['churn_rate'] = (1 - df['retention_rate']).round(4)

        # Class utilization (daily)
        df['class_utilization'] = (
            df['total_class_attendance'] / (df['total_classes_held'] * 20 + 1)
        ).round(4)

        # Staff per member ratio
        df['staff_per_member'] = (df['staff_count'] / (df['total_members'] + 1)).round(4)

        # Estimated LTV (monthly ticket × retention × 12 months)
        df['estimated_ltv'] = (df['avg_ticket_price'] * df['retention_rate'] * 12).round(2)

        # Revenue mix percentages
        df['membership_revenue_pct'] = (
            df['daily_membership_revenue'] / (df['daily_revenue'] + 1)
        ).round(4)
        df['class_pack_revenue_pct'] = (
            df['daily_class_pack_revenue'] / (df['daily_revenue'] + 1)
        ).round(4)
        df['retail_revenue_pct'] = (
            df['daily_retail_revenue'] / (df['daily_revenue'] + 1)
        ).round(4)

        # Revenue per attendance (average revenue per attendee)
        df['revenue_per_attendance'] = (
            df['daily_revenue'] / (df['daily_attendance'] + 1)
        ).round(2)

        # Classes per attendee
        df['classes_per_attendee'] = (
            df['total_classes_held'] / (df['daily_attendance'] + 1)
        ).round(2)

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        logger.info("Adding interaction features...")

        # Retention × Ticket (high-value stable customers)
        df['retention_x_ticket'] = (
            df['retention_rate'] * df['avg_ticket_price']
        ).round(2)

        # Attendance × Classes (class program effectiveness)
        df['attendance_x_classes'] = (
            df['class_attendance_rate'] * df['total_classes_held']
        ).round(2)

        # Upsell × Members (upsell revenue potential)
        df['upsell_x_members'] = (
            df['upsell_rate'] * df['total_members']
        ).round(2)

        # Staff utilization × Members (operational efficiency)
        df['staff_util_x_members'] = (
            df['staff_utilization_rate'] * df['total_members']
        ).round(2)

        # Weekend × Revenue (weekend revenue impact)
        df['weekend_revenue_impact'] = (
            df['is_weekend'] * df['daily_revenue']
        ).round(2)

        # Monday boost × Revenue
        df['monday_revenue_impact'] = (
            df['is_monday'] * df['daily_revenue']
        ).round(2)

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for temporal features"""
        logger.info("Adding cyclical features...")

        # Month cyclical encoding (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).round(4)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).round(4)

        # Day of month cyclical encoding (1-31)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31).round(4)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31).round(4)

        # Week of year cyclical encoding (1-52)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52).round(4)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52).round(4)

        # Day of week cyclical encoding (1-7)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).round(4)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).round(4)

        return df

    def _add_additional_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional temporal features per studio"""
        logger.info("Adding additional temporal features...")

        # Day-over-day growth (per studio)
        df['dod_revenue_growth'] = df.groupby('studio_id')['daily_revenue'].pct_change().round(4)
        df['dod_attendance_growth'] = df.groupby('studio_id')['daily_attendance'].pct_change().round(4)

        # Week-over-week growth (7 days ago)
        df['wow_revenue_growth'] = (
            df.groupby('studio_id')['daily_revenue'].pct_change(periods=7).round(4)
        )
        df['wow_attendance_growth'] = (
            df.groupby('studio_id')['daily_attendance'].pct_change(periods=7).round(4)
        )

        # Ratio of current to 7-day average
        df['revenue_vs_7d_avg'] = (
            df['daily_revenue'] / (df['rolling_7d_revenue'] + 1)
        ).round(4)
        df['attendance_vs_7d_avg'] = (
            df['daily_attendance'] / (df['rolling_7d_attendance'] + 1)
        ).round(4)

        # Trend momentum (30-day average vs 7-day average)
        df['revenue_trend_momentum'] = (
            df['rolling_7d_revenue'] / (df['rolling_30d_revenue'] + 1)
        ).round(4)

        return df


def main():
    """Run feature engineering on daily data"""

    print("\n" + "="*80)
    print("DAILY DATA FEATURE ENGINEERING")
    print("="*80 + "\n")

    # Load daily data
    input_path = Path('data/processed/multi_studio_daily_data.csv')

    if not input_path.exists():
        logger.error(f"Daily data file not found: {input_path}")
        logger.error("Please run: python src/data/generate_multi_studio_daily_data.py")
        return

    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} daily records")

    # Initialize feature engineer
    engineer = DailyFeatureEngineer()

    # Engineer features
    df_engineered = engineer.engineer_features(df)

    # Save engineered data
    output_path = Path('data/processed/multi_studio_daily_data_engineered.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_engineered.to_csv(output_path, index=False)
    logger.info(f"Engineered data saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"\nTotal records: {len(df_engineered)}")
    print(f"Total features: {len(engineer.feature_names)}")
    print(f"\nData splits:")
    print(f"  Train: {sum(df_engineered['split'] == 'train')} ({sum(df_engineered['split'] == 'train')/len(df_engineered)*100:.1f}%)")
    print(f"  Validation: {sum(df_engineered['split'] == 'validation')} ({sum(df_engineered['split'] == 'validation')/len(df_engineered)*100:.1f}%)")
    print(f"  Test: {sum(df_engineered['split'] == 'test')} ({sum(df_engineered['split'] == 'test')/len(df_engineered)*100:.1f}%)")

    print(f"\nFeature categories:")

    # Count feature types
    feature_categories = {
        'Studio attributes': [f for f in engineer.feature_names if any(x in f for x in ['location_', 'size_', 'price_', 'studio_age'])],
        'Day-of-week': [f for f in engineer.feature_names if any(x in f for x in ['day_of_week', 'is_monday', 'is_friday', 'is_saturday', 'is_sunday', 'is_weekend'])],
        'Holiday': [f for f in engineer.feature_names if 'holiday' in f or 'january' in f or 'summer_prep' in f],
        'Rolling features': [f for f in engineer.feature_names if 'rolling' in f or 'momentum' in f],
        'Derived metrics': [f for f in engineer.feature_names if any(x in f for x in ['_per_', '_pct', '_rate', 'churn', 'utilization', 'ltv'])],
        'Interaction features': [f for f in engineer.feature_names if '_x_' in f or 'impact' in f],
        'Cyclical features': [f for f in engineer.feature_names if '_sin' in f or '_cos' in f],
        'Growth features': [f for f in engineer.feature_names if 'growth' in f or '_vs_' in f],
    }

    for category, features in feature_categories.items():
        print(f"  {category}: {len(features)} features")

    print(f"\nSample feature names (first 20):")
    for i, feature in enumerate(engineer.feature_names[:20]):
        print(f"    {i+1}. {feature}")

    if len(engineer.feature_names) > 20:
        print(f"    ... and {len(engineer.feature_names) - 20} more")

    print(f"\nData saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Train Neural Network: python training/train_model_v2.3_neural_network.py")
    print(f"  2. Train XGBoost: python training/train_model_v2.3_xgboost.py")
    print(f"  3. Train LightGBM: python training/train_model_v2.3_lightgbm.py")
    print(f"  4. Train Random Forest: python training/train_model_v2.3_random_forest.py")
    print(f"  5. Train Stacking Ensemble: python training/train_model_v2.3_stacking_ensemble.py")
    print(f"  6. Compare all models: python training/compare_all_models_v2.3.py")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
