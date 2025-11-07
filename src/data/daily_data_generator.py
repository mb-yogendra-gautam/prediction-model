"""
Daily Data Generator - Distributes Monthly Data Across Days with Seasonality Patterns

This module generates synthetic daily data from monthly aggregated data by:
- Distributing monthly totals across days with realistic patterns
- Adding seasonality effects (weekends, holidays, pay periods)
- Maintaining consistency (daily values sum to monthly totals)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import calendar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyDataGenerator:
    """Generates daily data from monthly aggregates with seasonality patterns"""
    
    # US Federal Holidays (static dates and month-based)
    HOLIDAYS = {
        (1, 1): "New Year's Day",
        (7, 4): "Independence Day",
        (12, 25): "Christmas Day",
        (12, 31): "New Year's Eve"
    }
    
    # Additional holidays (approximate - 3rd Monday patterns)
    FLOATING_HOLIDAYS_BY_MONTH = {
        1: [15],  # MLK Day (3rd Monday, approx day 15)
        2: [14, 15],  # Valentine's Day, Presidents Day
        5: [27],  # Memorial Day (last Monday, approx day 27)
        9: [3],   # Labor Day (1st Monday)
        11: [11, 22, 23],  # Veterans Day, Thanksgiving (4th Thu + Fri)
    }
    
    # Pay periods (1st and 15th of each month)
    PAY_DAYS = [1, 15]
    
    def __init__(self, seed=42):
        """
        Initialize daily data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
    def is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday"""
        # Static holidays
        if (date.month, date.day) in self.HOLIDAYS:
            return True
        
        # Floating holidays by month
        if date.month in self.FLOATING_HOLIDAYS_BY_MONTH:
            if date.day in self.FLOATING_HOLIDAYS_BY_MONTH[date.month]:
                return True
        
        return False
    
    def get_day_of_week(self, date: datetime) -> int:
        """Get day of week (0=Monday, 6=Sunday)"""
        return date.weekday()
    
    def is_weekend(self, date: datetime) -> bool:
        """Check if date is weekend (Saturday=5, Sunday=6)"""
        return date.weekday() >= 5
    
    def get_week_of_month(self, date: datetime) -> int:
        """Get week of month (1-5)"""
        first_day = date.replace(day=1)
        dom = date.day
        adjusted_dom = dom + first_day.weekday()
        return (adjusted_dom - 1) // 7 + 1
    
    def days_since_payday(self, date: datetime) -> int:
        """Calculate days since last payday (1st or 15th)"""
        day = date.day
        if day >= 15:
            return day - 15
        else:
            return day - 1
    
    def is_month_start(self, date: datetime) -> bool:
        """Check if in first 5 days of month"""
        return date.day <= 5
    
    def is_month_end(self, date: datetime) -> bool:
        """Check if in last 5 days of month"""
        last_day = calendar.monthrange(date.year, date.month)[1]
        return date.day > last_day - 5
    
    def calculate_daily_weight(self, date: datetime) -> float:
        """
        Calculate daily weight factor for revenue distribution
        
        Factors:
        - Weekends: 0.7x (30% reduction)
        - Weekdays: 1.0x baseline
        - Mid-week (Tue-Thu): 1.1x (10% boost)
        - Holidays: 0.5x (50% reduction)
        - Pay days (1st, 15th, +/- 2 days): 1.2x (20% boost)
        - Month end (last week): 1.15x for new signups
        
        Args:
            date: Date to calculate weight for
            
        Returns:
            Weight factor (relative to average day)
        """
        weight = 1.0
        
        # Holiday effect (strongest)
        if self.is_holiday(date):
            return 0.5
        
        # Weekend effect
        if self.is_weekend(date):
            weight *= 0.7
        else:
            # Mid-week boost (Tue=1, Wed=2, Thu=3)
            dow = date.weekday()
            if 1 <= dow <= 3:
                weight *= 1.1
        
        # Payday effect (boost 2 days before to 2 days after)
        days_since_pay = self.days_since_payday(date)
        if days_since_pay <= 2:  # Within 2 days after payday
            weight *= 1.2
        elif date.day in [14, 13]:  # 1-2 days before 15th
            weight *= 1.15
        elif date.day in [29, 30, 31]:  # Days before 1st of next month
            weight *= 1.15
        
        # Month-end boost for signups
        if self.is_month_end(date):
            weight *= 1.05
        
        return weight
    
    def distribute_monthly_value(
        self, 
        monthly_value: float, 
        year: int, 
        month: int,
        value_type: str = 'revenue'
    ) -> Dict[int, float]:
        """
        Distribute monthly total across days based on weights
        
        Args:
            monthly_value: Total value for the month
            year: Year
            month: Month (1-12)
            value_type: Type of value ('revenue', 'count', 'rate')
            
        Returns:
            Dictionary mapping day (1-31) to daily value
        """
        # Get number of days in month
        num_days = calendar.monthrange(year, month)[1]
        
        # Calculate weights for each day
        weights = {}
        total_weight = 0.0
        
        for day in range(1, num_days + 1):
            date = datetime(year, month, day)
            weight = self.calculate_daily_weight(date)
            weights[day] = weight
            total_weight += weight
        
        # Normalize weights and distribute value
        daily_values = {}
        distributed_sum = 0.0
        
        for day in range(1, num_days + 1):
            # Calculate proportional value
            proportion = weights[day] / total_weight
            daily_value = monthly_value * proportion
            
            # Add small random variation (±5%) for realism
            if value_type in ['revenue', 'count']:
                variation = np.random.uniform(0.95, 1.05)
                daily_value *= variation
            
            # Round to 2 decimal places for cleaner data
            daily_value = round(daily_value, 2)
            
            daily_values[day] = daily_value
            distributed_sum += daily_value
        
        # Adjust to ensure exact sum (correct for rounding and variation)
        if value_type in ['revenue', 'count'] and monthly_value > 0:
            adjustment_factor = monthly_value / distributed_sum
            for day in daily_values:
                daily_values[day] = round(daily_values[day] * adjustment_factor, 2)
        
        return daily_values
    
    def generate_daily_features(self, date: datetime) -> Dict:
        """
        Generate daily time features for a specific date
        
        Args:
            date: Date to generate features for
            
        Returns:
            Dictionary of daily features
        """
        return {
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'day_of_week': self.get_day_of_week(date),
            'is_weekend': int(self.is_weekend(date)),
            'week_of_month': self.get_week_of_month(date),
            'day_of_month': date.day,
            'is_holiday': int(self.is_holiday(date)),
            'days_since_payday': self.days_since_payday(date),
            'is_month_start': int(self.is_month_start(date)),
            'is_month_end': int(self.is_month_end(date)),
        }
    
    def expand_monthly_to_daily(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand monthly data to daily data with seasonality patterns
        
        Args:
            monthly_df: DataFrame with monthly aggregated data
            
        Returns:
            DataFrame with daily data
        """
        logger.info(f"Expanding {len(monthly_df)} monthly records to daily data...")
        
        daily_records = []
        
        # Columns to distribute
        revenue_cols = [
            'total_revenue', 'membership_revenue', 'class_pack_revenue', 
            'retail_revenue', 'basic_membership_revenue', 'premium_membership_revenue',
            'family_membership_revenue', 'drop_in_class_revenue', 'class_pack_10_revenue',
            'class_pack_20_revenue', 'unlimited_class_revenue', 'apparel_revenue',
            'supplements_revenue', 'equipment_revenue', 'personal_training_revenue',
            'nutrition_coaching_revenue', 'wellness_services_revenue'
        ]
        
        count_cols = [
            'new_members', 'churned_members', 'total_classes_held', 'total_class_attendance',
            'basic_membership_count', 'premium_membership_count', 'family_membership_count',
            'drop_in_class_count', 'class_pack_10_count', 'class_pack_20_count',
            'unlimited_class_count', 'apparel_sales_count', 'supplements_sales_count',
            'equipment_sales_count', 'personal_training_count', 'nutrition_coaching_count',
            'wellness_services_count'
        ]
        
        # Columns to replicate (rates, averages, static values)
        static_cols = [
            'total_members', 'retention_rate', 'avg_ticket_price', 
            'class_attendance_rate', 'staff_count', 'avg_classes_per_member',
            'upsell_rate', 'studio_location', 'studio_size_tier', 'studio_price_tier'
        ]
        
        # Process each monthly record
        for idx, row in monthly_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"  Processed {idx}/{len(monthly_df)} records...")
            
            # Parse month_year (format: YYYY-MM)
            year, month = map(int, row['month_year'].split('-'))
            num_days = calendar.monthrange(year, month)[1]
            
            # Distribute revenue columns
            daily_distributions = {}
            for col in revenue_cols:
                if col in row and pd.notna(row[col]):
                    daily_distributions[col] = self.distribute_monthly_value(
                        row[col], year, month, value_type='revenue'
                    )
            
            # Distribute count columns
            for col in count_cols:
                if col in row and pd.notna(row[col]):
                    daily_distributions[col] = self.distribute_monthly_value(
                        row[col], year, month, value_type='count'
                    )
            
            # Create daily records
            for day in range(1, num_days + 1):
                date = datetime(year, month, day)
                
                # Start with daily features
                daily_record = self.generate_daily_features(date)
                
                # Add studio ID
                daily_record['studio_id'] = row['studio_id']
                daily_record['month_year'] = row['month_year']
                
                # Add distributed values (rounded to 2 decimal places)
                for col in revenue_cols + count_cols:
                    if col in daily_distributions:
                        daily_record[col] = round(daily_distributions[col][day], 2)
                    else:
                        daily_record[col] = 0.0
                
                # Add static values (replicated across all days, rounded if numeric)
                for col in static_cols:
                    if col in row:
                        value = row[col]
                        # Round numeric values to 2 decimal places
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if col not in ['studio_location', 'studio_size_tier', 'studio_price_tier']:
                                daily_record[col] = round(float(value), 2)
                            else:
                                daily_record[col] = value
                        else:
                            daily_record[col] = value
                
                # Add split if present
                if 'split' in row:
                    daily_record['split'] = row['split']
                
                daily_records.append(daily_record)
        
        # Create DataFrame
        daily_df = pd.DataFrame(daily_records)
        
        logger.info(f"✓ Generated {len(daily_df)} daily records from {len(monthly_df)} monthly records")
        logger.info(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        
        # Validate: Check that daily sums match monthly totals (sample check)
        self._validate_daily_aggregation(daily_df, monthly_df, revenue_cols[:3])
        
        return daily_df
    
    def _validate_daily_aggregation(
        self, 
        daily_df: pd.DataFrame, 
        monthly_df: pd.DataFrame,
        cols_to_check: List[str]
    ):
        """Validate that daily values sum to monthly totals"""
        logger.info("Validating daily aggregation...")
        
        # Group daily data by studio and month
        daily_grouped = daily_df.groupby(['studio_id', 'month_year'])[cols_to_check].sum()
        
        # Compare with monthly data
        for col in cols_to_check:
            monthly_values = monthly_df.set_index(['studio_id', 'month_year'])[col]
            daily_sums = daily_grouped[col]
            
            # Calculate difference
            diff = (daily_sums - monthly_values).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            # Allow small rounding errors (< 0.01%)
            max_pct_diff = (max_diff / monthly_values.mean()) * 100 if monthly_values.mean() > 0 else 0
            
            if max_pct_diff < 0.01:
                logger.info(f"  ✓ {col}: Max diff = ${max_diff:.2f} ({max_pct_diff:.4f}%)")
            else:
                logger.warning(f"  ⚠ {col}: Max diff = ${max_diff:.2f} ({max_pct_diff:.4f}%)")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("DAILY DATA GENERATOR - Transform Monthly to Daily Data")
    print("="*80 + "\n")
    
    # Load monthly data
    input_path = Path('data/processed/multi_studio_data.csv')
    logger.info(f"Loading monthly data from {input_path}...")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please ensure multi_studio_data.csv exists")
        return
    
    monthly_df = pd.read_csv(input_path)
    logger.info(f"✓ Loaded {len(monthly_df)} monthly records")
    logger.info(f"  Studios: {monthly_df['studio_id'].nunique()}")
    logger.info(f"  Date range: {monthly_df['month_year'].min()} to {monthly_df['month_year'].max()}")
    
    # Initialize generator
    generator = DailyDataGenerator(seed=42)
    
    # Generate daily data
    print("\nGenerating daily data with seasonality patterns...")
    print("-" * 80)
    daily_df = generator.expand_monthly_to_daily(monthly_df)
    print("-" * 80)
    
    # Save output
    output_path = Path('data/processed/multi_studio_data_daily.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving daily data to {output_path}...")
    daily_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(daily_df)} daily records")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DAILY DATA SUMMARY")
    print("="*80)
    print(f"\nTotal daily records: {len(daily_df):,}")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Studios: {daily_df['studio_id'].nunique()}")
    
    print("\nDaily features added:")
    print(f"  - day_of_week: {daily_df['day_of_week'].min()} to {daily_df['day_of_week'].max()}")
    print(f"  - is_weekend: {daily_df['is_weekend'].sum():,} weekend days ({daily_df['is_weekend'].mean()*100:.1f}%)")
    print(f"  - is_holiday: {daily_df['is_holiday'].sum():,} holidays ({daily_df['is_holiday'].mean()*100:.1f}%)")
    print(f"  - week_of_month: {daily_df['week_of_month'].min()} to {daily_df['week_of_month'].max()}")
    
    print("\nRevenue distribution patterns:")
    weekday_revenue = daily_df[daily_df['is_weekend'] == 0]['total_revenue'].mean()
    weekend_revenue = daily_df[daily_df['is_weekend'] == 1]['total_revenue'].mean()
    holiday_revenue = daily_df[daily_df['is_holiday'] == 1]['total_revenue'].mean()
    
    print(f"  - Avg weekday revenue: ${weekday_revenue:.2f}")
    print(f"  - Avg weekend revenue: ${weekend_revenue:.2f} ({(weekend_revenue/weekday_revenue-1)*100:+.1f}%)")
    if daily_df['is_holiday'].sum() > 0:
        print(f"  - Avg holiday revenue: ${holiday_revenue:.2f} ({(holiday_revenue/weekday_revenue-1)*100:+.1f}%)")
    
    print("\n" + "="*80)
    print("DAILY DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"\nNext step: Run daily feature engineering")
    print(f"  python src/features/daily_feature_engineer.py")
    print("\n")


if __name__ == "__main__":
    main()

