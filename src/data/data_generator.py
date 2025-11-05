"""
Data Generation Module for Fitness Studio Revenue Simulator

Generates realistic synthetic data for training ML models with:
- Seasonality patterns (January surge, summer dip, fall recovery)
- Business growth phases (rapid, stabilization, mature)
- Realistic noise and outliers
- Revenue calculations with multiple components
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class StudioDataGenerator:
    """Generate realistic synthetic fitness studio data"""

    def __init__(self, num_months: int = None, num_years: int = None,
                 start_date: str = '2015-01-01', end_date: str = None,
                 studio_type: str = 'Yoga', seed: int = 42):
        """
        Initialize data generator

        Args:
            num_months: Number of months to generate (overrides num_years)
            num_years: Number of years of data to generate
            start_date: Start date (YYYY-MM-DD format) - DEFAULT: 2015-01-01 for more data
            end_date: End date (YYYY-MM-DD format, overrides num_months/num_years)
            studio_type: Type of studio (Yoga, Pilates, CrossFit, etc.)
            seed: Random seed for reproducibility
        """
        self.start_date = start_date
        self.end_date = end_date

        # Calculate number of months
        if end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            self.num_months = ((end.year - start.year) * 12 + end.month - start.month) + 1
        elif num_months:
            self.num_months = num_months
        elif num_years:
            self.num_months = num_years * 12
        else:
            self.num_months = 120  # Default 10 years (increased from 5 to address overfitting)

        self.studio_type = studio_type
        self.seed = seed
        np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        """Generate complete dataset with all metrics"""
        logger.info(f"Generating {self.num_months} months of data for {self.studio_type} studio")

        months = pd.date_range(self.start_date, periods=self.num_months, freq='MS')

        # Base values (starting point)
        base_members = 200
        base_retention = 0.75
        base_ticket = 150.0
        base_classes = 120
        base_staff = 10

        data = []
        current_members = base_members

        for i, month in enumerate(months):
            month_idx = month.month
            year_idx = i // 12

            # Apply seasonality factors
            seasonality = self._get_seasonality_factor(month_idx)

            # Apply business growth phase
            growth_factor = self._get_growth_factor(year_idx)

            # Calculate metrics with realistic noise
            retention_rate = self._calculate_retention(
                base_retention, seasonality, month_idx
            )

            new_members = self._calculate_new_members(
                seasonality, growth_factor, month_idx
            )

            # Member churn and count updates
            churned_members = int(current_members * (1 - retention_rate))
            current_members = max(
                current_members - churned_members + new_members,
                100  # Floor to maintain minimum members
            )

            # Pricing with gradual increase over time
            avg_ticket = base_ticket * (1.005 ** i) * (1 + np.random.normal(0, 0.02))

            # Class attendance with seasonality
            class_attendance_rate = self._calculate_attendance_rate(seasonality)

            # Class scheduling
            total_classes = int(
                base_classes * growth_factor * (1 + np.random.normal(0, 0.05))
            )

            # Total attendance (assuming 20 person capacity per class)
            total_attendance = int(total_classes * 20 * class_attendance_rate)

            # Staff scaling with member count
            staff_count = max(5, int(base_staff * (current_members / base_members)))
            staff_utilization = np.clip(
                0.80 * (1 + np.random.normal(0, 0.05)),
                0.65, 0.95
            )

            # Upsell rate (add-ons, personal training, etc.)
            upsell_rate = np.clip(
                0.25 * (1 + np.random.normal(0, 0.1)),
                0.1, 0.45
            )

            # Calculate revenue components
            membership_revenue = current_members * avg_ticket
            class_pack_revenue = total_attendance * 0.2 * 15  # 20% are drop-ins at $15
            retail_revenue = current_members * 10 * 0.3  # 30% buy retail, avg $10
            upsell_revenue = current_members * upsell_rate * 50  # Avg $50 upsell

            total_revenue = (
                membership_revenue +
                class_pack_revenue +
                retail_revenue +
                upsell_revenue
            )

            # Add realistic outliers (COVID impact, facility closure, major promotion)
            # March 2020 = month 63 from Jan 2015 (5 years * 12 + 3 months = 63)
            if i == 63:  # COVID impact - March 2020
                total_revenue *= 0.6
                current_members = int(current_members * 0.85)
                retention_rate *= 0.8
                logger.info(f"Applied COVID impact at month {i} (March 2020): Revenue dropped 40%")

            data.append({
                'month_year': month,
                'month_index': month_idx,
                'year_index': year_idx,
                'total_members': current_members,
                'new_members': new_members,
                'churned_members': churned_members,
                'retention_rate': round(retention_rate, 2),
                'avg_ticket_price': round(avg_ticket, 2),
                'total_classes_held': total_classes,
                'total_class_attendance': total_attendance,
                'class_attendance_rate': round(class_attendance_rate, 2),
                'staff_count': staff_count,
                'staff_utilization_rate': round(staff_utilization, 2),
                'total_revenue': round(total_revenue, 2),
                'membership_revenue': round(membership_revenue, 2),
                'class_pack_revenue': round(class_pack_revenue, 2),
                'retail_revenue': round(retail_revenue, 2),
                'upsell_rate': round(upsell_rate, 2),
            })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} rows of data")

        # Log summary statistics
        logger.info(f"Revenue range: ${df['total_revenue'].min():.2f} - ${df['total_revenue'].max():.2f}")
        logger.info(f"Member range: {df['total_members'].min()} - {df['total_members'].max()}")
        logger.info(f"Avg retention: {df['retention_rate'].mean():.2%}")

        return df

    def _get_seasonality_factor(self, month_idx: int) -> dict:
        """
        Get seasonality multipliers for given month

        Patterns:
        - January: New Year resolutions surge
        - Summer: Vacation dip
        - Fall: Back-to-routine recovery
        """
        return {
            'january_boost': 1.25 if month_idx == 1 else 1.0,
            'summer_dip': 0.90 if month_idx in [6, 7, 8] else 1.0,
            'fall_recovery': 1.10 if month_idx == 9 else 1.0,
        }

    def _get_growth_factor(self, year_idx: int) -> float:
        """
        Get growth factor based on business maturity phase

        Phase 1 (Year 0-1): Rapid growth
        Phase 2 (Year 2-3): Stabilization
        Phase 3 (Year 4-5): Mature business
        Phase 4 (Year 6+): Stable/Declining
        """
        if year_idx <= 1:
            return 1.012  # 1.2% monthly growth (15% annual)
        elif year_idx <= 3:
            return 1.005  # 0.5% monthly growth (6% annual)
        elif year_idx <= 5:
            return 1.002  # 0.2% monthly growth (2.4% annual)
        else:
            return 1.000  # 0% growth (stable)

    def _calculate_retention(self, base_retention: float,
                            seasonality: dict, month_idx: int) -> float:
        """Calculate retention rate with seasonality and noise"""
        retention = base_retention * seasonality['summer_dip'] * \
                   (1 + np.random.normal(0, 0.03))
        return np.clip(retention, 0.6, 0.95)

    def _calculate_new_members(self, seasonality: dict,
                              growth_factor: float, month_idx: int) -> int:
        """Calculate new members with seasonality"""
        base_new = 25
        return int(
            base_new *
            seasonality['january_boost'] *
            seasonality['fall_recovery'] *
            growth_factor *
            (1 + np.random.normal(0, 0.15))
        )

    def _calculate_attendance_rate(self, seasonality: dict) -> float:
        """Calculate class attendance rate"""
        base_rate = 0.70
        rate = base_rate * seasonality['summer_dip'] * \
               (1 + np.random.normal(0, 0.05))
        return np.clip(rate, 0.5, 0.9)


class DataPreprocessor:
    """Prepare data for model training"""

    @staticmethod
    def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add future revenue targets for supervised learning

        Creates targets for predicting:
        - Revenue 1, 2, 3 months ahead
        - Member count 3 months ahead
        - Retention rate 3 months ahead
        """
        logger.info("Adding target variables (future revenue predictions)")

        df = df.copy()

        # Create future targets by shifting data backwards
        df['revenue_month_1'] = df['total_revenue'].shift(-1).round(2)
        df['revenue_month_2'] = df['total_revenue'].shift(-2).round(2)
        df['revenue_month_3'] = df['total_revenue'].shift(-3).round(2)
        df['member_count_month_3'] = df['total_members'].shift(-3)
        df['retention_rate_month_3'] = df['retention_rate'].shift(-3).round(2)

        # Remove rows without targets (last 3 months)
        df = df[:-3].copy()

        logger.info(f"Dataset size after adding targets: {len(df)} rows")
        return df

    @staticmethod
    def add_data_splits(df: pd.DataFrame,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1) -> pd.DataFrame:
        """
        Add train/validation/test split labels

        Uses time-based splitting (no shuffling) to maintain temporal order
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        df = df.copy()
        df['split'] = 'train'
        df.loc[train_end:val_end, 'split'] = 'validation'
        df.loc[val_end:, 'split'] = 'test'

        # Log split information
        train_count = len(df[df['split'] == 'train'])
        val_count = len(df[df['split'] == 'validation'])
        test_count = len(df[df['split'] == 'test'])

        logger.info(f"Data splits created:")
        logger.info(f"  Train: {train_count} rows ({train_count/n*100:.1f}%)")
        logger.info(f"  Validation: {val_count} rows ({val_count/n*100:.1f}%)")
        logger.info(f"  Test: {test_count} rows ({test_count/n*100:.1f}%)")

        return df

    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate data quality and return issues

        Checks:
        - Missing values
        - Negative values in columns that should be positive
        - Revenue calculation consistency
        - Retention rate bounds
        - Realistic value ranges
        """
        issues = []

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            missing_cols = missing[missing > 0]
            issues.append(f"Missing values found: {missing_cols.to_dict()}")

        # Check for negative values where they shouldn't exist
        positive_cols = [
            'total_members', 'new_members', 'churned_members',
            'avg_ticket_price', 'total_classes_held', 'total_class_attendance',
            'staff_count', 'total_revenue', 'membership_revenue',
            'class_pack_revenue', 'retail_revenue'
        ]

        for col in positive_cols:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")

        # Check revenue consistency (within tolerance)
        if 'total_revenue' in df.columns:
            calculated_revenue = (
                df['membership_revenue'] +
                df['class_pack_revenue'] +
                df['retail_revenue']
            )
            revenue_diff = np.abs(df['total_revenue'] - calculated_revenue)
            inconsistent_count = (revenue_diff > 1000).sum()
            if inconsistent_count > 0:
                issues.append(
                    f"Revenue calculation inconsistency in {inconsistent_count} rows "
                    f"(diff > $1000)"
                )

        # Check retention rate bounds
        if 'retention_rate' in df.columns:
            out_of_bounds = ((df['retention_rate'] < 0.5) | (df['retention_rate'] > 1.0)).sum()
            if out_of_bounds > 0:
                issues.append(f"Retention rate out of bounds in {out_of_bounds} rows")

        # Check for realistic ranges
        if 'avg_ticket_price' in df.columns:
            if (df['avg_ticket_price'] < 50).any() or (df['avg_ticket_price'] > 500).any():
                issues.append("Unrealistic avg_ticket_price values detected")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("[OK] Data quality validation passed")
        else:
            logger.warning(f"[FAIL] Data quality issues found: {len(issues)}")
            for issue in issues:
                logger.warning(f"  - {issue}")

        return is_valid, issues


# Main execution script
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("Studio Revenue Simulator - Data Generation")
    print("="*60 + "\n")

    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic studio data...")
    generator = StudioDataGenerator(
        start_date='2015-01-01',  # Extended to 2015 for more training data
        end_date='2026-01-01',    # Generate data till Jan 2026 (11 years = 132 months)
        studio_type='Yoga'
    )
    raw_data = generator.generate()

    print(f"\nGenerated {len(raw_data)} months of data")
    print(f"Date range: {raw_data['month_year'].min()} to {raw_data['month_year'].max()}")

    # Step 2: Add target variables
    print("\nStep 2: Adding target variables...")
    preprocessor = DataPreprocessor()
    data = preprocessor.add_target_variables(raw_data)

    # Step 3: Create train/val/test splits
    print("\nStep 3: Creating train/validation/test splits...")
    data = preprocessor.add_data_splits(data)

    # Step 4: Validate data quality
    print("\nStep 4: Validating data quality...")
    is_valid, issues = preprocessor.validate_data_quality(data)

    if not is_valid:
        print("\n[WARNING] Data quality issues detected!")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] All data quality checks passed")

    # Step 5: Display summary statistics
    print("\n" + "="*60)
    print("Data Summary Statistics")
    print("="*60)
    print(f"\nTotal rows: {len(data)}")
    print(f"\nRevenue Statistics:")
    print(f"  Mean: ${data['total_revenue'].mean():,.2f}")
    print(f"  Median: ${data['total_revenue'].median():,.2f}")
    print(f"  Std: ${data['total_revenue'].std():,.2f}")
    print(f"  Min: ${data['total_revenue'].min():,.2f}")
    print(f"  Max: ${data['total_revenue'].max():,.2f}")

    print(f"\nMember Statistics:")
    print(f"  Mean: {data['total_members'].mean():.1f}")
    print(f"  Min: {data['total_members'].min()}")
    print(f"  Max: {data['total_members'].max()}")

    print(f"\nRetention Rate:")
    print(f"  Mean: {data['retention_rate'].mean():.2%}")
    print(f"  Min: {data['retention_rate'].min():.2%}")
    print(f"  Max: {data['retention_rate'].max():.2%}")

    # Step 6: Save processed data
    print("\nStep 6: Saving processed data...")
    output_path = 'data/processed/studio_data_2015_2025.csv'
    data.to_csv(output_path, index=False)
    print(f"[OK] Data saved to {output_path}")
    print(f"[INFO] This dataset has MORE samples to reduce overfitting")

    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60 + "\n")
