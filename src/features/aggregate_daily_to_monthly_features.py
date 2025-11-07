"""
Aggregate Daily Data to Monthly with Seasonality Pattern Features

Takes daily time series data and aggregates to monthly level while preserving
daily pattern insights as features (weekend %, holiday impact, payday boost, etc.)

This approach combines:
- Monthly prediction targets (same as v2.2.0)
- Daily seasonality intelligence (from v2.3.0 daily data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyToMonthlyAggregator:
    """Aggregates daily data to monthly with daily pattern features"""
    
    def __init__(self):
        self.pattern_features = []
        
    def calculate_daily_pattern_features(self, daily_df: pd.DataFrame) -> Dict:
        """
        Calculate daily pattern features from a month of daily data
        
        Args:
            daily_df: DataFrame with daily records for one studio-month
            
        Returns:
            Dictionary of daily pattern features
        """
        features = {}
        
        # Total metrics (will be monthly aggregates)
        total_revenue = daily_df['total_revenue'].sum()
        
        if total_revenue == 0:
            # Avoid division by zero
            return self._get_default_features()
        
        # Weekend revenue pattern
        weekend_revenue = daily_df[daily_df['is_weekend'] == 1]['total_revenue'].sum()
        features['weekend_revenue_pct'] = weekend_revenue / total_revenue
        
        weekday_revenue = daily_df[daily_df['is_weekend'] == 0]['total_revenue'].sum()
        weekday_days = (daily_df['is_weekend'] == 0).sum()
        weekend_days = (daily_df['is_weekend'] == 1).sum()
        
        avg_weekday_revenue = weekday_revenue / weekday_days if weekday_days > 0 else 0
        avg_weekend_revenue = weekend_revenue / weekend_days if weekend_days > 0 else 0
        
        features['weekend_weekday_ratio'] = (avg_weekend_revenue / avg_weekday_revenue) if avg_weekday_revenue > 0 else 1.0
        
        # Holiday impact
        holiday_days = daily_df[daily_df['is_holiday'] == 1]
        if len(holiday_days) > 0:
            holiday_revenue = holiday_days['total_revenue'].sum()
            avg_holiday_revenue = holiday_revenue / len(holiday_days)
            avg_non_holiday_revenue = daily_df[daily_df['is_holiday'] == 0]['total_revenue'].mean()
            features['holiday_impact_score'] = (avg_holiday_revenue / avg_non_holiday_revenue) if avg_non_holiday_revenue > 0 else 0.5
        else:
            features['holiday_impact_score'] = 1.0  # No holidays = no impact
        
        # Payday boost (days 14-16 and 29-31, plus day 1-2)
        payday_mask = daily_df['day_of_month'].isin([1, 2, 14, 15, 16, 29, 30, 31])
        payday_revenue = daily_df[payday_mask]['total_revenue'].sum()
        payday_days = payday_mask.sum()
        non_payday_revenue = daily_df[~payday_mask]['total_revenue'].sum()
        non_payday_days = (~payday_mask).sum()
        
        avg_payday_revenue = payday_revenue / payday_days if payday_days > 0 else 0
        avg_non_payday_revenue = non_payday_revenue / non_payday_days if non_payday_days > 0 else 0
        
        features['payday_boost_factor'] = (avg_payday_revenue / avg_non_payday_revenue) if avg_non_payday_revenue > 0 else 1.0
        
        # Month start/end patterns
        month_start_revenue = daily_df[daily_df['is_month_start'] == 1]['total_revenue'].sum()
        month_end_revenue = daily_df[daily_df['is_month_end'] == 1]['total_revenue'].sum()
        
        features['month_start_revenue_pct'] = month_start_revenue / total_revenue
        features['month_end_revenue_pct'] = month_end_revenue / total_revenue
        
        # Daily revenue volatility (consistency measure)
        daily_revenues = daily_df['total_revenue'].values
        features['daily_revenue_std'] = np.std(daily_revenues)
        features['daily_revenue_cv'] = features['daily_revenue_std'] / np.mean(daily_revenues) if np.mean(daily_revenues) > 0 else 0
        
        # Week-by-week consistency
        daily_df_copy = daily_df.copy()
        daily_df_copy['week'] = (daily_df_copy['day_of_month'] - 1) // 7
        weekly_revenues = daily_df_copy.groupby('week')['total_revenue'].sum()
        features['weekly_revenue_std'] = weekly_revenues.std() if len(weekly_revenues) > 1 else 0
        
        # Weekday consistency (Mon-Fri pattern stability)
        weekday_data = daily_df[daily_df['is_weekend'] == 0]
        if len(weekday_data) > 1:
            features['weekday_consistency'] = weekday_data['total_revenue'].std()
        else:
            features['weekday_consistency'] = 0
        
        # Class attendance patterns
        weekend_attendance = daily_df[daily_df['is_weekend'] == 1]['total_class_attendance'].sum()
        weekday_attendance = daily_df[daily_df['is_weekend'] == 0]['total_class_attendance'].sum()
        total_attendance = weekend_attendance + weekday_attendance
        
        features['weekend_attendance_pct'] = weekend_attendance / total_attendance if total_attendance > 0 else 0.3
        
        # New member patterns
        total_new_members = daily_df['new_members'].sum()
        month_end_new_members = daily_df[daily_df['is_month_end'] == 1]['new_members'].sum()
        
        features['month_end_signup_pct'] = month_end_new_members / total_new_members if total_new_members > 0 else 0.2
        
        return features
    
    def _get_default_features(self) -> Dict:
        """Return default feature values for edge cases"""
        return {
            'weekend_revenue_pct': 0.28,
            'weekend_weekday_ratio': 0.70,
            'holiday_impact_score': 0.50,
            'payday_boost_factor': 1.15,
            'month_start_revenue_pct': 0.20,
            'month_end_revenue_pct': 0.20,
            'daily_revenue_std': 0.0,
            'daily_revenue_cv': 0.15,
            'weekly_revenue_std': 0.0,
            'weekday_consistency': 0.0,
            'weekend_attendance_pct': 0.28,
            'month_end_signup_pct': 0.25
        }
    
    def aggregate_to_monthly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily data to monthly with pattern features
        
        Args:
            daily_df: DataFrame with daily records
            
        Returns:
            DataFrame with monthly records + daily pattern features
        """
        logger.info(f"Aggregating {len(daily_df)} daily records to monthly...")
        
        monthly_records = []
        
        # Group by studio and month
        grouped = daily_df.groupby(['studio_id', 'month_year'])
        
        for (studio_id, month_year), month_data in grouped:
            # Base monthly aggregates (sum for revenues/counts, avg for rates)
            monthly_record = {
                'studio_id': studio_id,
                'month_year': month_year,
                
                # Aggregate revenue columns (sum)
                'total_revenue': month_data['total_revenue'].sum(),
                'membership_revenue': month_data['membership_revenue'].sum(),
                'class_pack_revenue': month_data['class_pack_revenue'].sum(),
                'retail_revenue': month_data['retail_revenue'].sum(),
                'basic_membership_revenue': month_data['basic_membership_revenue'].sum(),
                'premium_membership_revenue': month_data['premium_membership_revenue'].sum(),
                'family_membership_revenue': month_data['family_membership_revenue'].sum(),
                'drop_in_class_revenue': month_data['drop_in_class_revenue'].sum(),
                'class_pack_10_revenue': month_data['class_pack_10_revenue'].sum(),
                'class_pack_20_revenue': month_data['class_pack_20_revenue'].sum(),
                'unlimited_class_revenue': month_data['unlimited_class_revenue'].sum(),
                'apparel_revenue': month_data['apparel_revenue'].sum(),
                'supplements_revenue': month_data['supplements_revenue'].sum(),
                'equipment_revenue': month_data['equipment_revenue'].sum(),
                'personal_training_revenue': month_data['personal_training_revenue'].sum(),
                'nutrition_coaching_revenue': month_data['nutrition_coaching_revenue'].sum(),
                'wellness_services_revenue': month_data['wellness_services_revenue'].sum(),
                
                # Aggregate count columns (sum)
                'new_members': month_data['new_members'].sum(),
                'churned_members': month_data['churned_members'].sum(),
                'total_classes_held': month_data['total_classes_held'].sum(),
                'total_class_attendance': month_data['total_class_attendance'].sum(),
                'basic_membership_count': month_data['basic_membership_count'].sum(),
                'premium_membership_count': month_data['premium_membership_count'].sum(),
                'family_membership_count': month_data['family_membership_count'].sum(),
                'drop_in_class_count': month_data['drop_in_class_count'].sum(),
                'class_pack_10_count': month_data['class_pack_10_count'].sum(),
                'class_pack_20_count': month_data['class_pack_20_count'].sum(),
                'unlimited_class_count': month_data['unlimited_class_count'].sum(),
                'apparel_sales_count': month_data['apparel_sales_count'].sum(),
                'supplements_sales_count': month_data['supplements_sales_count'].sum(),
                'equipment_sales_count': month_data['equipment_sales_count'].sum(),
                'personal_training_count': month_data['personal_training_count'].sum(),
                'nutrition_coaching_count': month_data['nutrition_coaching_count'].sum(),
                'wellness_services_count': month_data['wellness_services_count'].sum(),
                
                # Average for rates and static values
                'total_members': month_data['total_members'].mean(),
                'retention_rate': month_data['retention_rate'].mean(),
                'avg_ticket_price': month_data['avg_ticket_price'].mean(),
                'class_attendance_rate': month_data['class_attendance_rate'].mean(),
                'staff_count': month_data['staff_count'].mode()[0] if len(month_data['staff_count'].mode()) > 0 else month_data['staff_count'].mean(),
                'avg_classes_per_member': month_data['avg_classes_per_member'].mean(),
                'upsell_rate': month_data['upsell_rate'].mean(),
                
                # Static columns (take first value)
                'studio_location': month_data['studio_location'].iloc[0],
                'studio_size_tier': month_data['studio_size_tier'].iloc[0],
                'studio_price_tier': month_data['studio_price_tier'].iloc[0],
                'split': month_data['split'].iloc[0],
            }
            
            # Calculate daily pattern features
            pattern_features = self.calculate_daily_pattern_features(month_data)
            monthly_record.update(pattern_features)
            
            monthly_records.append(monthly_record)
        
        monthly_df = pd.DataFrame(monthly_records)
        
        logger.info(f"✓ Created {len(monthly_df)} monthly records with daily pattern features")
        logger.info(f"  Studios: {monthly_df['studio_id'].nunique()}")
        logger.info(f"  Date range: {monthly_df['month_year'].min()} to {monthly_df['month_year'].max()}")
        
        return monthly_df


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("AGGREGATE DAILY TO MONTHLY WITH PATTERN FEATURES")
    print("="*80 + "\n")
    
    # Load daily data
    input_path = Path('data/processed/multi_studio_data_daily.csv')
    logger.info(f"Loading daily data from {input_path}...")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run: python src/data/daily_data_generator.py")
        return
    
    daily_df = pd.read_csv(input_path)
    logger.info(f"✓ Loaded {len(daily_df)} daily records")
    logger.info(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    logger.info(f"  Studios: {daily_df['studio_id'].nunique()}")
    
    # Initialize aggregator
    aggregator = DailyToMonthlyAggregator()
    
    # Aggregate to monthly
    print("\nAggregating daily data to monthly with pattern features...")
    print("-" * 80)
    monthly_df = aggregator.aggregate_to_monthly(daily_df)
    print("-" * 80)
    
    # Show sample of new pattern features
    pattern_feature_cols = [
        'weekend_revenue_pct', 'weekend_weekday_ratio', 'holiday_impact_score',
        'payday_boost_factor', 'month_start_revenue_pct', 'month_end_revenue_pct',
        'daily_revenue_std', 'daily_revenue_cv', 'weekend_attendance_pct'
    ]
    
    print("\n" + "="*80)
    print("DAILY PATTERN FEATURES SUMMARY")
    print("="*80)
    
    for feat in pattern_feature_cols:
        if feat in monthly_df.columns:
            print(f"\n{feat}:")
            print(f"  Mean: {monthly_df[feat].mean():.4f}")
            print(f"  Std:  {monthly_df[feat].std():.4f}")
            print(f"  Min:  {monthly_df[feat].min():.4f}")
            print(f"  Max:  {monthly_df[feat].max():.4f}")
    
    # Save output
    output_path = Path('data/processed/multi_studio_data_monthly_with_daily_patterns.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving monthly data to {output_path}...")
    monthly_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(monthly_df)} monthly records")
    
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE!")
    print("="*80)
    print(f"\nInput:  {len(daily_df):,} daily records")
    print(f"Output: {len(monthly_df):,} monthly records with {len(pattern_feature_cols)} daily pattern features")
    print(f"\nNext step: Run feature engineering and model training")
    print(f"  python src/features/run_feature_engineering_multi_studio.py --input {output_path}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

