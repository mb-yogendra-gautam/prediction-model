"""
Multi-Studio Daily Data Generator

Generates synthetic DAILY data for 12 studios with realistic daily patterns:
- Day-of-week effects (Monday peak, Sunday low)
- Weekend vs weekday patterns
- Holiday effects
- Rolling metrics (7-day, 14-day, 30-day averages)
- Multi-horizon targets (1 day, 7 days, 30 days, 90 days)

Each studio generates 5 years = ~1,825 daily records
Total: 12 studios × 1,825 days = 21,900 records
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiStudioDailyDataGenerator:
    """Generate synthetic daily data for multiple fitness studios"""

    def __init__(self, n_studios=12, years=5, seed=42):
        self.n_studios = n_studios
        self.years = years
        self.days = years * 365  # ~1,825 days
        self.seed = seed
        np.random.seed(seed)

        # US Federal Holidays (major ones affecting fitness)
        self.holidays = self._define_holidays()

        # Define studio profiles
        self.studio_profiles = self._create_studio_profiles()

    def _define_holidays(self):
        """Define major US holidays that impact fitness studio attendance"""
        # Generate holidays for 2019-2024 (5 years)
        holidays = []

        for year in range(2019, 2025):
            # Fixed date holidays
            holidays.append(datetime(year, 1, 1))   # New Year's Day
            holidays.append(datetime(year, 7, 4))   # Independence Day
            holidays.append(datetime(year, 12, 25)) # Christmas Day
            holidays.append(datetime(year, 12, 31)) # New Year's Eve

            # Thanksgiving (4th Thursday of November)
            nov_1 = datetime(year, 11, 1)
            days_to_thursday = (3 - nov_1.weekday()) % 7
            first_thursday = nov_1 + timedelta(days=days_to_thursday)
            thanksgiving = first_thursday + timedelta(days=21)  # 4th Thursday
            holidays.append(thanksgiving)
            holidays.append(thanksgiving + timedelta(days=1))  # Black Friday

            # Memorial Day (last Monday of May)
            may_31 = datetime(year, 5, 31)
            days_to_monday = (0 - may_31.weekday()) % 7
            if days_to_monday == 0:
                memorial_day = may_31
            else:
                memorial_day = may_31 - timedelta(days=7 - days_to_monday)
            holidays.append(memorial_day)

            # Labor Day (1st Monday of September)
            sep_1 = datetime(year, 9, 1)
            days_to_monday = (0 - sep_1.weekday()) % 7
            labor_day = sep_1 + timedelta(days=days_to_monday)
            holidays.append(labor_day)

        return set([h.date() for h in holidays])

    def _create_studio_profiles(self):
        """Define characteristics for each studio (same as monthly generator)"""
        profiles = [
            # Large Urban Studios (Growing)
            {
                'studio_id': 'STU001',
                'name': 'Urban Elite Fitness',
                'location_type': 'urban',
                'size_tier': 'large',
                'base_daily_attendance': 85,  # Daily instead of monthly members
                'growth_rate': 0.0005,  # Daily growth rate (~0.015 monthly)
                'retention_baseline': 0.75,
                'price_tier': 'high',
                'base_price': 180,
                'seasonality_strength': 0.08,
                'class_capacity': 25,
                'base_staff': 7,
                'daily_classes': 8,  # Classes per day
                'monday_boost': 1.20,  # Monday 20% above average
                'weekend_factor': 1.15,  # Weekend boost for this studio
            },
            {
                'studio_id': 'STU002',
                'name': 'Downtown Power Studio',
                'location_type': 'urban',
                'size_tier': 'large',
                'base_daily_attendance': 82,
                'growth_rate': 0.0004,
                'retention_baseline': 0.73,
                'price_tier': 'high',
                'base_price': 175,
                'seasonality_strength': 0.10,
                'class_capacity': 24,
                'base_staff': 7,
                'daily_classes': 8,
                'monday_boost': 1.18,
                'weekend_factor': 1.10,
            },
            # Medium Urban Studios (Stable)
            {
                'studio_id': 'STU003',
                'name': 'City Center Fitness',
                'location_type': 'urban',
                'size_tier': 'medium',
                'base_daily_attendance': 70,
                'growth_rate': 0.00017,
                'retention_baseline': 0.72,
                'price_tier': 'medium',
                'base_price': 155,
                'seasonality_strength': 0.12,
                'class_capacity': 22,
                'base_staff': 6,
                'daily_classes': 7,
                'monday_boost': 1.22,
                'weekend_factor': 1.05,
            },
            {
                'studio_id': 'STU004',
                'name': 'Metro Wellness Hub',
                'location_type': 'urban',
                'size_tier': 'medium',
                'base_daily_attendance': 68,
                'growth_rate': 0.00027,
                'retention_baseline': 0.70,
                'price_tier': 'medium',
                'base_price': 150,
                'seasonality_strength': 0.11,
                'class_capacity': 20,
                'base_staff': 5,
                'daily_classes': 7,
                'monday_boost': 1.20,
                'weekend_factor': 1.08,
            },
            # Small Urban Studios (Growing)
            {
                'studio_id': 'STU005',
                'name': 'Urban Boutique Fitness',
                'location_type': 'urban',
                'size_tier': 'small',
                'base_daily_attendance': 55,
                'growth_rate': 0.00067,
                'retention_baseline': 0.68,
                'price_tier': 'medium',
                'base_price': 145,
                'seasonality_strength': 0.15,
                'class_capacity': 18,
                'base_staff': 4,
                'daily_classes': 6,
                'monday_boost': 1.25,
                'weekend_factor': 1.12,
            },
            {
                'studio_id': 'STU006',
                'name': 'City Studio Express',
                'location_type': 'urban',
                'size_tier': 'small',
                'base_daily_attendance': 52,
                'growth_rate': 0.0006,
                'retention_baseline': 0.67,
                'price_tier': 'low',
                'base_price': 125,
                'seasonality_strength': 0.14,
                'class_capacity': 16,
                'base_staff': 4,
                'daily_classes': 6,
                'monday_boost': 1.23,
                'weekend_factor': 1.10,
            },
            # Large Suburban Studios (Stable)
            {
                'studio_id': 'STU007',
                'name': 'Suburban Family Fitness',
                'location_type': 'suburban',
                'size_tier': 'large',
                'base_daily_attendance': 88,
                'growth_rate': 0.00023,
                'retention_baseline': 0.77,
                'price_tier': 'medium',
                'base_price': 145,
                'seasonality_strength': 0.18,
                'class_capacity': 26,
                'base_staff': 7,
                'daily_classes': 8,
                'monday_boost': 1.15,
                'weekend_factor': 1.25,  # Suburban: higher weekend
            },
            {
                'studio_id': 'STU008',
                'name': 'Suburban Wellness Center',
                'location_type': 'suburban',
                'size_tier': 'large',
                'base_daily_attendance': 90,
                'growth_rate': 0.0002,
                'retention_baseline': 0.78,
                'price_tier': 'medium',
                'base_price': 140,
                'seasonality_strength': 0.20,
                'class_capacity': 25,
                'base_staff': 7,
                'daily_classes': 8,
                'monday_boost': 1.12,
                'weekend_factor': 1.28,
            },
            # Medium Suburban Studios (Growing)
            {
                'studio_id': 'STU009',
                'name': 'Neighborhood Fitness',
                'location_type': 'suburban',
                'size_tier': 'medium',
                'base_daily_attendance': 65,
                'growth_rate': 0.00033,
                'retention_baseline': 0.74,
                'price_tier': 'low',
                'base_price': 130,
                'seasonality_strength': 0.16,
                'class_capacity': 20,
                'base_staff': 5,
                'daily_classes': 7,
                'monday_boost': 1.18,
                'weekend_factor': 1.22,
            },
            {
                'studio_id': 'STU010',
                'name': 'Community Health Studio',
                'location_type': 'suburban',
                'size_tier': 'medium',
                'base_daily_attendance': 68,
                'growth_rate': 0.0003,
                'retention_baseline': 0.73,
                'price_tier': 'low',
                'base_price': 128,
                'seasonality_strength': 0.17,
                'class_capacity': 22,
                'base_staff': 6,
                'daily_classes': 7,
                'monday_boost': 1.17,
                'weekend_factor': 1.20,
            },
            # Small Suburban Studios (Declining/Stable)
            {
                'studio_id': 'STU011',
                'name': 'Local Fitness Corner',
                'location_type': 'suburban',
                'size_tier': 'small',
                'base_daily_attendance': 50,
                'growth_rate': -0.000067,  # Slight decline
                'retention_baseline': 0.66,
                'price_tier': 'low',
                'base_price': 120,
                'seasonality_strength': 0.22,
                'class_capacity': 15,
                'base_staff': 4,
                'daily_classes': 6,
                'monday_boost': 1.20,
                'weekend_factor': 1.15,
            },
            {
                'studio_id': 'STU012',
                'name': 'Suburban Fit Studio',
                'location_type': 'suburban',
                'size_tier': 'small',
                'base_daily_attendance': 52,
                'growth_rate': 0.0001,
                'retention_baseline': 0.68,
                'price_tier': 'low',
                'base_price': 122,
                'seasonality_strength': 0.19,
                'class_capacity': 16,
                'base_staff': 4,
                'daily_classes': 6,
                'monday_boost': 1.19,
                'weekend_factor': 1.18,
            }
        ]

        return profiles

    def generate_data(self):
        """Generate daily data for all studios"""
        logger.info(f"Generating daily data for {self.n_studios} studios over {self.years} years...")
        logger.info(f"Total days per studio: {self.days}")

        all_data = []

        for profile in self.studio_profiles:
            logger.info(f"Generating daily data for {profile['studio_id']} - {profile['name']}")
            studio_data = self._generate_studio_daily_data(profile)
            all_data.append(studio_data)

        # Combine all studios
        df = pd.concat(all_data, ignore_index=True)

        # Sort by studio and date
        df = df.sort_values(['studio_id', 'date']).reset_index(drop=True)

        logger.info(f"Generated {len(df)} total studio-days")

        # Add rolling features (must be done after all data is generated)
        logger.info("Adding rolling features...")
        df = self._add_rolling_features(df)

        # Add future targets for multi-horizon predictions
        logger.info("Adding future prediction targets...")
        df = self._add_future_targets(df)

        # Save raw data
        output_path = Path('data/raw/multi_studio_daily_data_raw.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")

        return df

    def _generate_studio_daily_data(self, profile):
        """Generate daily time series data for a single studio"""

        # Create date range
        start_date = datetime(2019, 3, 1)  # Start on Friday to have clean weeks
        dates = [start_date + timedelta(days=i) for i in range(self.days)]

        data = []

        # Initialize cumulative member count
        cumulative_members = int(profile['base_daily_attendance'] * 4)  # Monthly members = 4x daily

        for day_idx, date in enumerate(dates):

            # Day of week (0=Monday, 6=Sunday)
            day_of_week = date.weekday()

            # Calculate trend component (gradual growth/decline)
            trend_factor = 1 + (profile['growth_rate'] * day_idx)

            # Calculate seasonal component (monthly seasonality)
            month = date.month
            seasonal_factor = self._get_seasonal_factor(month, profile['seasonality_strength'])

            # Day-of-week effect
            dow_factor = self._get_day_of_week_factor(day_of_week, profile)

            # Holiday effect
            is_holiday = date.date() in self.holidays
            is_holiday_week = any((date + timedelta(days=i)).date() in self.holidays
                                 for i in range(-3, 4))

            holiday_factor = 0.5 if is_holiday else (0.8 if is_holiday_week else 1.0)

            # Post-New Year surge (January 2-31)
            post_new_year_boost = 1.4 if (date.month == 1 and date.day > 1) else 1.0

            # Pre-summer ramp (May-June)
            pre_summer_boost = 1.15 if date.month in [5, 6] else 1.0

            # Add random daily noise (weather, local events)
            daily_noise = np.random.normal(0, 0.12)  # 12% daily variability

            # Calculate daily attendance
            base_attendance = profile['base_daily_attendance']
            daily_attendance = int(
                base_attendance *
                trend_factor *
                seasonal_factor *
                dow_factor *
                holiday_factor *
                post_new_year_boost *
                pre_summer_boost *
                (1 + daily_noise)
            )
            daily_attendance = max(10, daily_attendance)  # Minimum 10 per day

            # Calculate daily metrics

            # Classes held (varies by day)
            classes_held = profile['daily_classes']
            if day_of_week == 6:  # Sunday: fewer classes
                classes_held = max(3, int(classes_held * 0.6))
            elif day_of_week == 5:  # Saturday: more classes
                classes_held = int(classes_held * 1.2)

            # Class attendance rate
            class_attendance_rate = 0.70 * dow_factor * (1 + np.random.normal(0, 0.08))
            class_attendance_rate = np.clip(class_attendance_rate, 0.50, 0.90)

            total_class_attendance = daily_attendance  # Daily attendance = class attendance

            # Member churn and acquisition (daily level)
            daily_churn_prob = (1 - profile['retention_baseline']) / 30  # Convert monthly to daily
            churned_today = np.random.binomial(cumulative_members, daily_churn_prob)

            new_members_today = int(daily_attendance * 0.02 + np.random.normal(0, 0.5))  # ~2% are new
            new_members_today = max(0, new_members_today)

            # Update cumulative members
            cumulative_members = cumulative_members - churned_today + new_members_today
            cumulative_members = max(50, cumulative_members)  # Minimum member base

            # Pricing (monthly membership amortized daily)
            avg_ticket_monthly = profile['base_price'] * (1 + np.random.normal(0, 0.02))

            # Revenue calculations (daily)
            # Daily revenue = (monthly membership / 30) * active members + drop-ins
            daily_membership_revenue = (avg_ticket_monthly / 30) * cumulative_members
            daily_class_pack_revenue = daily_attendance * 15 * 0.15  # 15% are drop-ins at $15
            daily_retail_revenue = cumulative_members * 0.10  # Small daily retail
            daily_revenue = daily_membership_revenue + daily_class_pack_revenue + daily_retail_revenue

            # Staff metrics (relatively stable)
            staff_count = profile['base_staff'] + int((cumulative_members - profile['base_daily_attendance'] * 4) / 30)
            staff_count = max(3, staff_count)

            staff_utilization = 0.80 + np.random.normal(0, 0.05)
            staff_utilization = np.clip(staff_utilization, 0.65, 0.95)

            # Upsell rate (weekly/monthly metric, but tracked daily)
            upsell_rate = 0.15 + np.random.normal(0, 0.03)
            upsell_rate = np.clip(upsell_rate, 0.08, 0.25)

            # Calculate retention rate (rolling monthly)
            monthly_retention = profile['retention_baseline'] * (1 + np.random.normal(0, 0.02))
            monthly_retention = np.clip(monthly_retention, 0.60, 0.85)

            # Create daily record
            record = {
                'studio_id': profile['studio_id'],
                'date': date.strftime('%Y-%m-%d'),
                'day_of_week': day_of_week + 1,  # 1-7 (Monday=1, Sunday=7)
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'is_monday': 1 if day_of_week == 0 else 0,
                'is_friday': 1 if day_of_week == 4 else 0,
                'is_saturday': 1 if day_of_week == 5 else 0,
                'is_sunday': 1 if day_of_week == 6 else 0,
                'is_holiday': 1 if is_holiday else 0,
                'is_holiday_week': 1 if is_holiday_week else 0,
                'is_january': 1 if date.month == 1 else 0,
                'is_summer_prep': 1 if date.month in [5, 6] else 0,
                'month': date.month,
                'day_of_month': date.day,
                'week_of_year': date.isocalendar()[1],

                # Core daily metrics
                'daily_attendance': daily_attendance,
                'daily_revenue': round(daily_revenue, 2),
                'daily_membership_revenue': round(daily_membership_revenue, 2),
                'daily_class_pack_revenue': round(daily_class_pack_revenue, 2),
                'daily_retail_revenue': round(daily_retail_revenue, 2),

                # Member metrics
                'total_members': cumulative_members,
                'new_members': new_members_today,
                'churned_members': churned_today,
                'retention_rate': round(monthly_retention, 4),
                'avg_ticket_price': round(avg_ticket_monthly, 2),

                # Class metrics
                'total_classes_held': classes_held,
                'total_class_attendance': total_class_attendance,
                'class_attendance_rate': round(class_attendance_rate, 4),

                # Staff metrics
                'staff_count': staff_count,
                'staff_utilization_rate': round(staff_utilization, 4),

                # Other metrics
                'upsell_rate': round(upsell_rate, 4),

                # Studio attributes
                'studio_location': profile['location_type'],
                'studio_size_tier': profile['size_tier'],
                'studio_price_tier': profile['price_tier'],
            }

            data.append(record)

        return pd.DataFrame(data)

    def _get_seasonal_factor(self, month, strength):
        """Calculate seasonal adjustment factor (same as monthly)"""
        seasonal_pattern = {
            1: 1.15,   # January - New Year peak
            2: 1.08,   # February
            3: 1.02,   # March
            4: 0.98,   # April
            5: 0.95,   # May
            6: 0.88,   # June - Summer dip starts
            7: 0.85,   # July - Summer low
            8: 0.87,   # August
            9: 1.05,   # September - Back to routine
            10: 1.03,  # October
            11: 0.97,  # November
            12: 0.90   # December - Holiday season
        }

        base_factor = seasonal_pattern[month]
        adjustment = (base_factor - 1) * strength

        return 1 + adjustment

    def _get_day_of_week_factor(self, day_of_week, profile):
        """
        Calculate day-of-week attendance factor
        0=Monday, 1=Tuesday, ..., 6=Sunday
        """
        # Base pattern (urban studios)
        dow_pattern = {
            0: profile['monday_boost'],  # Monday: high (post-weekend motivation)
            1: 1.05,   # Tuesday: slightly above average
            2: 1.00,   # Wednesday: average
            3: 1.03,   # Thursday: slightly above average
            4: 1.08,   # Friday: above average (TGIF energy)
            5: profile['weekend_factor'],  # Saturday: varies by studio
            6: 0.70,   # Sunday: low (rest day)
        }

        return dow_pattern[day_of_week]

    def _add_rolling_features(self, df):
        """Add rolling average features (7-day, 14-day, 30-day)"""

        # Group by studio to calculate rolling features per studio
        rolling_features = []

        for studio_id in df['studio_id'].unique():
            studio_df = df[df['studio_id'] == studio_id].copy()
            studio_df = studio_df.sort_values('date').reset_index(drop=True)

            # 7-day rolling averages
            studio_df['rolling_7d_revenue'] = studio_df['daily_revenue'].rolling(window=7, min_periods=1).mean()
            studio_df['rolling_7d_attendance'] = studio_df['daily_attendance'].rolling(window=7, min_periods=1).mean()
            studio_df['rolling_7d_retention'] = studio_df['retention_rate'].rolling(window=7, min_periods=1).mean()

            # 14-day rolling averages
            studio_df['rolling_14d_revenue'] = studio_df['daily_revenue'].rolling(window=14, min_periods=1).mean()
            studio_df['rolling_14d_attendance'] = studio_df['daily_attendance'].rolling(window=14, min_periods=1).mean()

            # 30-day rolling averages
            studio_df['rolling_30d_revenue'] = studio_df['daily_revenue'].rolling(window=30, min_periods=1).mean()
            studio_df['rolling_30d_attendance'] = studio_df['daily_attendance'].rolling(window=30, min_periods=1).mean()
            studio_df['rolling_30d_retention'] = studio_df['retention_rate'].rolling(window=30, min_periods=1).mean()

            # Day momentum (3-day EMA)
            studio_df['day_momentum_revenue'] = studio_df['daily_revenue'].ewm(span=3, adjust=False).mean()
            studio_df['day_momentum_attendance'] = studio_df['daily_attendance'].ewm(span=3, adjust=False).mean()

            # Weekday vs weekend ratio (rolling 14-day)
            studio_df['weekday_attendance'] = studio_df['daily_attendance'] * (1 - studio_df['is_weekend'])
            studio_df['weekend_attendance'] = studio_df['daily_attendance'] * studio_df['is_weekend']

            weekday_roll = studio_df['weekday_attendance'].rolling(window=14, min_periods=1).sum()
            weekend_roll = studio_df['weekend_attendance'].rolling(window=14, min_periods=1).sum()
            studio_df['weekday_vs_weekend_ratio'] = np.where(
                weekend_roll > 0,
                weekday_roll / (weekend_roll + 1),  # Avoid division by zero
                1.0
            )

            studio_df = studio_df.drop(['weekday_attendance', 'weekend_attendance'], axis=1)

            rolling_features.append(studio_df)

        df_with_rolling = pd.concat(rolling_features, ignore_index=True)
        df_with_rolling = df_with_rolling.sort_values(['studio_id', 'date']).reset_index(drop=True)

        return df_with_rolling

    def _add_future_targets(self, df):
        """
        Add multi-horizon prediction targets:
        - Daily: 1, 3, 7 days ahead
        - Weekly: 7, 14, 28 days ahead (weekly aggregates)
        - Monthly: 30, 60, 90 days ahead (monthly aggregates)
        """

        target_features = []

        for studio_id in df['studio_id'].unique():
            studio_df = df[df['studio_id'] == studio_id].copy()
            studio_df = studio_df.sort_values('date').reset_index(drop=True)

            # Daily targets (shift by N days)
            studio_df['revenue_day_1'] = studio_df['daily_revenue'].shift(-1)
            studio_df['revenue_day_3'] = studio_df['daily_revenue'].shift(-3)
            studio_df['revenue_day_7'] = studio_df['daily_revenue'].shift(-7)
            studio_df['attendance_day_7'] = studio_df['daily_attendance'].shift(-7)

            # Weekly targets (7-day rolling sum, shifted)
            studio_df['weekly_revenue_temp'] = studio_df['daily_revenue'].rolling(window=7, min_periods=7).sum()
            studio_df['weekly_attendance_temp'] = studio_df['daily_attendance'].rolling(window=7, min_periods=7).sum()

            studio_df['revenue_week_1'] = studio_df['weekly_revenue_temp'].shift(-7)
            studio_df['revenue_week_2'] = studio_df['weekly_revenue_temp'].shift(-14)
            studio_df['revenue_week_4'] = studio_df['weekly_revenue_temp'].shift(-28)
            studio_df['attendance_week_1'] = studio_df['weekly_attendance_temp'].shift(-7)

            # Monthly targets (30-day rolling sum, shifted)
            studio_df['monthly_revenue_temp'] = studio_df['daily_revenue'].rolling(window=30, min_periods=30).sum()
            studio_df['monthly_attendance_temp'] = studio_df['daily_attendance'].rolling(window=30, min_periods=30).sum()
            studio_df['monthly_members_temp'] = studio_df['total_members'].rolling(window=30, min_periods=30).mean()
            studio_df['monthly_retention_temp'] = studio_df['retention_rate'].rolling(window=30, min_periods=30).mean()

            studio_df['revenue_month_1'] = studio_df['monthly_revenue_temp'].shift(-30)
            studio_df['revenue_month_2'] = studio_df['monthly_revenue_temp'].shift(-60)
            studio_df['revenue_month_3'] = studio_df['monthly_revenue_temp'].shift(-90)
            studio_df['member_count_month_1'] = studio_df['monthly_members_temp'].shift(-30)
            studio_df['member_count_month_3'] = studio_df['monthly_members_temp'].shift(-90)
            studio_df['retention_rate_month_3'] = studio_df['monthly_retention_temp'].shift(-90)

            # Drop temporary columns
            studio_df = studio_df.drop([
                'weekly_revenue_temp', 'weekly_attendance_temp',
                'monthly_revenue_temp', 'monthly_attendance_temp',
                'monthly_members_temp', 'monthly_retention_temp'
            ], axis=1)

            target_features.append(studio_df)

        df_with_targets = pd.concat(target_features, ignore_index=True)
        df_with_targets = df_with_targets.sort_values(['studio_id', 'date']).reset_index(drop=True)

        return df_with_targets

    def add_data_splits(self, df):
        """Add train/validation/test splits stratified by studio (temporal)"""
        logger.info("Adding data splits...")

        df = df.sort_values(['studio_id', 'date']).reset_index(drop=True)

        splits = []

        for studio_id in df['studio_id'].unique():
            studio_df = df[df['studio_id'] == studio_id].copy()
            n = len(studio_df)

            # 70% train, 15% validation, 15% test (temporal split)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            studio_splits = ['train'] * train_end + \
                          ['validation'] * (val_end - train_end) + \
                          ['test'] * (n - val_end)

            splits.extend(studio_splits)

        df['split'] = splits

        logger.info(f"Train: {sum(df['split'] == 'train')} samples")
        logger.info(f"Validation: {sum(df['split'] == 'validation')} samples")
        logger.info(f"Test: {sum(df['split'] == 'test')} samples")

        return df

    def save_data(self, df, output_path='data/processed/multi_studio_daily_data.csv'):
        """Save generated data"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")

        return output_path


def main():
    """Generate multi-studio daily dataset"""

    print("\n" + "="*80)
    print("MULTI-STUDIO DAILY DATA GENERATION")
    print("="*80 + "\n")

    # Initialize generator
    generator = MultiStudioDailyDataGenerator(n_studios=12, years=5, seed=42)

    # Generate data
    print("Generating daily data (this may take a few minutes)...")
    df = generator.generate_data()

    # Add splits
    df = generator.add_data_splits(df)

    # Save data
    output_path = generator.save_data(df)

    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"\nTotal studios: {df['studio_id'].nunique()}")
    print(f"Total studio-days: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nData splits:")
    print(f"  Train: {sum(df['split'] == 'train')} ({sum(df['split'] == 'train')/len(df)*100:.1f}%)")
    print(f"  Validation: {sum(df['split'] == 'validation')} ({sum(df['split'] == 'validation')/len(df)*100:.1f}%)")
    print(f"  Test: {sum(df['split'] == 'test')} ({sum(df['split'] == 'test')/len(df)*100:.1f}%)")

    print(f"\nStudio distribution:")
    for location in df['studio_location'].unique():
        for size in df['studio_size_tier'].unique():
            count = len(df[(df['studio_location'] == location) & (df['studio_size_tier'] == size)]['studio_id'].unique())
            if count > 0:
                print(f"  {location.capitalize()} {size.capitalize()}: {count} studios")

    print(f"\nDaily statistics:")
    print(f"  Daily attendance: {df['daily_attendance'].min():.0f} - {df['daily_attendance'].max():.0f} (mean: {df['daily_attendance'].mean():.0f})")
    print(f"  Daily revenue: ${df['daily_revenue'].min():.0f} - ${df['daily_revenue'].max():.0f} (mean: ${df['daily_revenue'].mean():.0f})")
    print(f"  Monthly revenue (30d roll): ${df['revenue_month_1'].min():.0f} - ${df['revenue_month_1'].max():.0f} (mean: ${df['revenue_month_1'].mean():.0f})")
    print(f"  Retention: {df['retention_rate'].min():.2f} - {df['retention_rate'].max():.2f} (mean: {df['retention_rate'].mean():.2f})")

    print(f"\nDay-of-week distribution:")
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for dow in range(1, 8):
        count = sum(df['day_of_week'] == dow)
        avg_attendance = df[df['day_of_week'] == dow]['daily_attendance'].mean()
        print(f"  {dow_names[dow-1]}: {count} days (avg attendance: {avg_attendance:.0f})")

    print(f"\nHoliday impact:")
    holiday_days = sum(df['is_holiday'] == 1)
    non_holiday_avg = df[df['is_holiday'] == 0]['daily_attendance'].mean()
    holiday_avg = df[df['is_holiday'] == 1]['daily_attendance'].mean() if holiday_days > 0 else 0
    print(f"  Holiday days: {holiday_days}")
    print(f"  Non-holiday avg attendance: {non_holiday_avg:.0f}")
    print(f"  Holiday avg attendance: {holiday_avg:.0f} ({(holiday_avg/non_holiday_avg-1)*100:.1f}% impact)")

    print(f"\nPrediction targets available:")
    print(f"  Daily targets: 1, 3, 7 days ahead")
    print(f"  Weekly targets: 1, 2, 4 weeks ahead")
    print(f"  Monthly targets: 1, 2, 3 months ahead")
    print(f"  Total target columns: 11")

    print(f"\nData saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Run feature engineering: python src/features/run_feature_engineering_daily.py")
    print("  2. Train models: python training/train_model_v2.3_*.py")
    print("  3. Compare: python training/compare_all_models_v2.3.py")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
