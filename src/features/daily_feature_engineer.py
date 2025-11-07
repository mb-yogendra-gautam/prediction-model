"""
Daily Feature Engineering Pipeline

Applies feature engineering to daily data with:
- Cyclical encodings for daily patterns (day_of_week, day_of_month)
- Daily temporal features (rolling averages, DoD/WoW growth)
- Interaction features with seasonality
- Daily revenue distribution patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyFeatureEngineer:
    """Feature engineering for daily time series data"""
    
    def __init__(self, multi_studio=True):
        """
        Initialize daily feature engineer
        
        Args:
            multi_studio: Whether working with multi-studio data (maintains studio boundaries)
        """
        self.multi_studio = multi_studio
        
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical encodings for temporal features
        
        Cyclical features ensure continuity (e.g., day 6 -> day 0, day 31 -> day 1)
        Uses sine/cosine transformations
        
        Args:
            df: DataFrame with temporal columns
            
        Returns:
            DataFrame with cyclical features added
        """
        logger.info("Adding cyclical features...")
        
        # Day of week (0-6)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month (1-31)
        df['day_of_month_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)
        
        # Month of year (1-12)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        logger.info(f"  ✓ Added 6 cyclical features")
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window features for temporal patterns
        
        Args:
            df: DataFrame with daily data
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Adding rolling features...")
        
        # Ensure data is sorted by studio and date
        df = df.sort_values(['studio_id', 'date']).reset_index(drop=True)
        
        rolling_features = []
        
        if self.multi_studio:
            # Calculate rolling features per studio
            for studio_id in df['studio_id'].unique():
                studio_mask = df['studio_id'] == studio_id
                studio_df = df[studio_mask].copy()
                
                # 7-day rolling average (previous week)
                studio_df['prev_7d_avg_revenue'] = studio_df['total_revenue'].rolling(
                    window=7, min_periods=1
                ).mean().shift(1)
                
                # 30-day rolling average (previous month)
                studio_df['prev_30d_avg_revenue'] = studio_df['total_revenue'].rolling(
                    window=30, min_periods=1
                ).mean().shift(1)
                
                # 7-day rolling std (volatility)
                studio_df['prev_7d_std_revenue'] = studio_df['total_revenue'].rolling(
                    window=7, min_periods=1
                ).std().shift(1)
                
                # Day-over-day growth
                studio_df['dod_revenue_growth'] = studio_df['total_revenue'].pct_change(periods=1)
                
                # Week-over-week growth (compare to same day last week)
                studio_df['wow_revenue_growth'] = studio_df['total_revenue'].pct_change(periods=7)
                
                rolling_features.append(studio_df)
            
            df = pd.concat(rolling_features, ignore_index=True)
        else:
            # Single studio - simpler calculation
            df['prev_7d_avg_revenue'] = df['total_revenue'].rolling(
                window=7, min_periods=1
            ).mean().shift(1)
            df['prev_30d_avg_revenue'] = df['total_revenue'].rolling(
                window=30, min_periods=1
            ).mean().shift(1)
            df['prev_7d_std_revenue'] = df['total_revenue'].rolling(
                window=7, min_periods=1
            ).std().shift(1)
            df['dod_revenue_growth'] = df['total_revenue'].pct_change(periods=1)
            df['wow_revenue_growth'] = df['total_revenue'].pct_change(periods=7)
        
        # Fill initial NaN values with 0
        rolling_cols = [
            'prev_7d_avg_revenue', 'prev_30d_avg_revenue', 'prev_7d_std_revenue',
            'dod_revenue_growth', 'wow_revenue_growth'
        ]
        for col in rolling_cols:
            df[col] = df[col].fillna(0)
        
        logger.info(f"  ✓ Added {len(rolling_cols)} rolling features")
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features for temporal dependencies
        
        Args:
            df: DataFrame with daily data
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Adding lag features...")
        
        # Ensure data is sorted
        df = df.sort_values(['studio_id', 'date']).reset_index(drop=True)
        
        lag_features = []
        
        if self.multi_studio:
            # Calculate lags per studio
            for studio_id in df['studio_id'].unique():
                studio_mask = df['studio_id'] == studio_id
                studio_df = df[studio_mask].copy()
                
                # Previous day revenue
                studio_df['prev_1d_revenue'] = studio_df['total_revenue'].shift(1)
                
                # Same day last week revenue
                studio_df['prev_7d_revenue'] = studio_df['total_revenue'].shift(7)
                
                # Previous day member count
                studio_df['prev_1d_members'] = studio_df['total_members'].shift(1)
                
                lag_features.append(studio_df)
            
            df = pd.concat(lag_features, ignore_index=True)
        else:
            df['prev_1d_revenue'] = df['total_revenue'].shift(1)
            df['prev_7d_revenue'] = df['total_revenue'].shift(7)
            df['prev_1d_members'] = df['total_members'].shift(1)
        
        # Fill NaN with 0
        lag_cols = ['prev_1d_revenue', 'prev_7d_revenue', 'prev_1d_members']
        for col in lag_cols:
            df[col] = df[col].fillna(0)
        
        logger.info(f"  ✓ Added {len(lag_cols)} lag features")
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between seasonality and business metrics
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Adding interaction features...")
        
        interactions = []
        
        # Weekend interactions
        if 'is_weekend' in df.columns and 'retention_rate' in df.columns:
            df['is_weekend_x_retention'] = df['is_weekend'] * df['retention_rate']
            interactions.append('is_weekend_x_retention')
        
        if 'is_weekend' in df.columns and 'total_members' in df.columns:
            df['is_weekend_x_members'] = df['is_weekend'] * df['total_members']
            interactions.append('is_weekend_x_members')
        
        # Holiday interactions
        if 'is_holiday' in df.columns and 'total_members' in df.columns:
            df['is_holiday_x_members'] = df['is_holiday'] * df['total_members']
            interactions.append('is_holiday_x_members')
        
        # Payday interactions
        if 'days_since_payday' in df.columns and 'upsell_rate' in df.columns:
            df['payday_x_upsell'] = (df['days_since_payday'] <= 2).astype(int) * df['upsell_rate']
            interactions.append('payday_x_upsell')
        
        # Month-end interactions
        if 'is_month_end' in df.columns and 'new_members' in df.columns:
            df['month_end_x_new_members'] = df['is_month_end'] * df['new_members']
            interactions.append('month_end_x_new_members')
        
        # Day of week x member count
        if 'day_of_week' in df.columns and 'total_members' in df.columns:
            df['dow_x_members'] = df['day_of_week'] * df['total_members'] / 1000  # Scaled
            interactions.append('dow_x_members')
        
        logger.info(f"  ✓ Added {len(interactions)} interaction features")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived business metrics
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Adding derived features...")
        
        derived = []
        
        # Revenue per member (daily average)
        if 'total_revenue' in df.columns and 'total_members' in df.columns:
            df['revenue_per_member'] = df['total_revenue'] / (df['total_members'] + 1)
            derived.append('revenue_per_member')
        
        # Class attendance per member
        if 'total_class_attendance' in df.columns and 'total_members' in df.columns:
            df['attendance_per_member'] = df['total_class_attendance'] / (df['total_members'] + 1)
            derived.append('attendance_per_member')
        
        # Revenue volatility score (recent std / recent mean)
        if 'prev_7d_std_revenue' in df.columns and 'prev_7d_avg_revenue' in df.columns:
            df['revenue_volatility'] = df['prev_7d_std_revenue'] / (df['prev_7d_avg_revenue'] + 1)
            derived.append('revenue_volatility')
        
        # Momentum indicator (short-term avg vs long-term avg)
        if 'prev_7d_avg_revenue' in df.columns and 'prev_30d_avg_revenue' in df.columns:
            df['revenue_momentum'] = (df['prev_7d_avg_revenue'] - df['prev_30d_avg_revenue']) / (df['prev_30d_avg_revenue'] + 1)
            derived.append('revenue_momentum')
        
        logger.info(f"  ✓ Added {len(derived)} derived features")
        
        return df
    
    def add_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target variables for model training
        
        For daily model, we predict:
        - Actual daily values (revenue, members, etc.)
        - Daily percentage of monthly total (distribution pattern)
        
        Args:
            df: DataFrame with daily data
            
        Returns:
            DataFrame with target features
        """
        logger.info("Adding target features...")
        
        # Daily targets (already exist)
        df['target_daily_revenue'] = df['total_revenue']
        df['target_daily_members'] = df['total_members']
        df['target_daily_retention'] = df['retention_rate']
        
        # Calculate daily percentage of monthly revenue (for distribution pattern learning)
        monthly_totals = df.groupby(['studio_id', 'month_year'])['total_revenue'].transform('sum')
        df['target_daily_pct_of_monthly'] = df['total_revenue'] / (monthly_totals + 1)
        
        logger.info(f"  ✓ Added 4 target features")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline
        
        Args:
            df: Raw daily DataFrame
            is_training: Whether this is training data (affects certain features)
            
        Returns:
            Engineered DataFrame ready for modeling
        """
        logger.info(f"Starting daily feature engineering (training={is_training})...")
        logger.info(f"Input: {len(df)} rows × {len(df.columns)} columns")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 1. Cyclical features
        df = self.add_cyclical_features(df)
        
        # 2. Rolling features
        df = self.add_rolling_features(df)
        
        # 3. Lag features
        df = self.add_lag_features(df)
        
        # 4. Interaction features
        df = self.add_interaction_features(df)
        
        # 5. Derived features
        df = self.add_derived_features(df)
        
        # 6. Target features (for training only)
        if is_training:
            df = self.add_target_features(df)
        
        # Remove rows with NaN values (from initial rolling windows)
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with NaN values ({removed/initial_len*100:.1f}%)")
        
        logger.info(f"Output: {len(df)} rows × {len(df.columns)} columns")
        logger.info("✓ Daily feature engineering complete")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names (excluding targets and metadata)
        
        Args:
            df: Engineered DataFrame
            
        Returns:
            List of feature column names
        """
        exclude_cols = [
            'studio_id', 'month_year', 'date', 'year', 'month', 'day', 'split',
            'studio_location', 'studio_size_tier', 'studio_price_tier',
            'target_daily_revenue', 'target_daily_members', 'target_daily_retention',
            'target_daily_pct_of_monthly'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("DAILY FEATURE ENGINEERING - Add Seasonality and Temporal Features")
    print("="*80 + "\n")
    
    # Load daily data
    input_path = Path('data/processed/multi_studio_data_daily.csv')
    logger.info(f"Loading daily data from {input_path}...")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run: python src/data/daily_data_generator.py")
        return
    
    df = pd.read_csv(input_path)
    logger.info(f"✓ Loaded {len(df)} daily records")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Studios: {df['studio_id'].nunique()}")
    
    # Initialize engineer
    engineer = DailyFeatureEngineer(multi_studio=True)
    
    # Apply feature engineering
    print("\nApplying feature engineering pipeline...")
    print("-" * 80)
    engineered_df = engineer.engineer_features(df, is_training=True)
    print("-" * 80)
    
    # Get feature names
    feature_cols = engineer.get_feature_names(engineered_df)
    
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    
    # Categorize features
    cyclical = [c for c in feature_cols if '_sin' in c or '_cos' in c]
    rolling = [c for c in feature_cols if 'prev_' in c and 'avg' in c or 'std' in c]
    lag = [c for c in feature_cols if 'prev_' in c and c not in rolling]
    growth = [c for c in feature_cols if 'growth' in c or 'momentum' in c]
    interaction = [c for c in feature_cols if '_x_' in c]
    seasonality = [c for c in feature_cols if any(x in c for x in ['is_weekend', 'is_holiday', 'week_of', 'days_since', 'is_month'])]
    derived = [c for c in feature_cols if 'per_member' in c or 'volatility' in c]
    base = [c for c in feature_cols if c not in cyclical + rolling + lag + growth + interaction + seasonality + derived]
    
    print(f"\n1. CYCLICAL FEATURES ({len(cyclical)}):")
    for feat in cyclical:
        print(f"   - {feat}")
    
    print(f"\n2. SEASONALITY FEATURES ({len(seasonality)}):")
    for feat in seasonality:
        print(f"   - {feat}")
    
    print(f"\n3. ROLLING FEATURES ({len(rolling)}):")
    for feat in rolling:
        print(f"   - {feat}")
    
    print(f"\n4. LAG FEATURES ({len(lag)}):")
    for feat in lag:
        print(f"   - {feat}")
    
    print(f"\n5. GROWTH FEATURES ({len(growth)}):")
    for feat in growth:
        print(f"   - {feat}")
    
    print(f"\n6. INTERACTION FEATURES ({len(interaction)}):")
    for feat in interaction:
        print(f"   - {feat}")
    
    print(f"\n7. DERIVED FEATURES ({len(derived)}):")
    for feat in derived:
        print(f"   - {feat}")
    
    print(f"\n8. BASE FEATURES ({len(base)}):")
    for feat in base[:10]:  # Show first 10
        print(f"   - {feat}")
    if len(base) > 10:
        print(f"   ... and {len(base) - 10} more")
    
    print(f"\nTOTAL FEATURES: {len(feature_cols)}")
    
    # Save engineered data
    output_path = Path('data/processed/multi_studio_data_daily_engineered.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving engineered data to {output_path}...")
    engineered_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(engineered_df)} daily records with {len(engineered_df.columns)} columns")
    
    # Show sample statistics
    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)
    
    print(f"\nDaily revenue distribution:")
    print(f"  Mean: ${engineered_df['total_revenue'].mean():.2f}")
    print(f"  Std:  ${engineered_df['total_revenue'].std():.2f}")
    print(f"  Min:  ${engineered_df['total_revenue'].min():.2f}")
    print(f"  Max:  ${engineered_df['total_revenue'].max():.2f}")
    
    if 'split' in engineered_df.columns:
        split_counts = engineered_df['split'].value_counts()
        print(f"\nData split:")
        for split_name, count in split_counts.items():
            print(f"  {split_name}: {count:,} records ({count/len(engineered_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("DAILY FEATURE ENGINEERING COMPLETE!")
    print("="*80)
    print(f"\nNext step: Train daily model v2.3.0")
    print(f"  python training/train_daily_model_v2.3.0.py")
    print("\n")


if __name__ == "__main__":
    main()

