"""
Feature Engineering Execution Script

Loads generated studio data and applies feature engineering transformations
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_generated_data(filepath: str) -> pd.DataFrame:
    """Load the generated studio data"""
    logger.info(f"Loading data from {filepath}")

    df = pd.read_csv(filepath)

    # Convert month_year to datetime
    df['month_year'] = pd.to_datetime(df['month_year'])

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    logger.info(f"Date range: {df['month_year'].min()} to {df['month_year'].max()}")

    return df


def validate_engineered_data(df: pd.DataFrame) -> bool:
    """Validate the engineered dataset"""
    logger.info("\n" + "="*70)
    logger.info("VALIDATION CHECKS")
    logger.info("="*70)

    issues = []

    # Check 1: No NaN values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        issues.append(f"NaN values found in columns: {null_cols.to_dict()}")
    else:
        logger.info("[OK] No NaN values found")

    # Check 2: No infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)

    if inf_cols:
        issues.append(f"Infinite values found in columns: {inf_cols}")
    else:
        logger.info("[OK] No infinite values found")

    # Check 3: Split column preserved
    if 'split' not in df.columns:
        issues.append("Split column missing")
    else:
        split_counts = df['split'].value_counts()
        logger.info(f"[OK] Split column preserved: {split_counts.to_dict()}")

    # Check 4: Feature count
    target_cols = [
        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
        'member_count_month_3', 'retention_rate_month_3'
    ]
    exclude_cols = target_cols + ['month_year', 'split']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"[OK] Total features created: {len(feature_cols)}")

    # Check 5: Value ranges
    suspicious_cols = []
    for col in numeric_cols:
        if col not in ['month_year']:
            if df[col].std() == 0:
                suspicious_cols.append(f"{col} (zero variance)")
            elif df[col].min() == df[col].max():
                suspicious_cols.append(f"{col} (constant value)")

    if suspicious_cols:
        issues.append(f"Suspicious columns with no variance: {suspicious_cols}")
    else:
        logger.info("[OK] All features have variance")

    # Report validation results
    logger.info("\n" + "="*70)
    if len(issues) == 0:
        logger.info("VALIDATION PASSED: Dataset is ready for model training")
        logger.info("="*70 + "\n")
        return True
    else:
        logger.warning("VALIDATION FAILED: Issues detected")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.info("="*70 + "\n")
        return False


def display_feature_summary(df: pd.DataFrame, feature_names: list):
    """Display summary of engineered features"""
    logger.info("\n" + "="*70)
    logger.info("ENGINEERED FEATURES SUMMARY")
    logger.info("="*70)

    # Group features by type
    derived_features = [
        'revenue_per_member', 'churn_rate', 'class_utilization',
        'staff_per_member', 'estimated_ltv', 'membership_revenue_pct',
        'class_pack_revenue_pct'
    ]

    lagged_features = [
        'prev_month_revenue', 'prev_month_members',
        'mom_revenue_growth', 'mom_member_growth'
    ]

    rolling_features = [
        '3m_avg_retention', '3m_avg_revenue', '3m_avg_attendance',
        '3m_std_revenue', 'revenue_momentum'
    ]

    interaction_features = [
        'retention_x_ticket', 'attendance_x_classes',
        'upsell_x_members', 'staff_util_x_members'
    ]

    cyclical_features = [
        'month_sin', 'month_cos', 'is_january', 'is_summer', 'is_fall'
    ]

    logger.info(f"\n1. DERIVED BUSINESS METRICS ({len(derived_features)} features):")
    for feat in derived_features:
        if feat in df.columns:
            logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")

    logger.info(f"\n2. LAGGED FEATURES ({len(lagged_features)} features):")
    for feat in lagged_features:
        if feat in df.columns:
            logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")

    logger.info(f"\n3. ROLLING STATISTICS ({len(rolling_features)} features):")
    for feat in rolling_features:
        if feat in df.columns:
            logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")

    logger.info(f"\n4. INTERACTION FEATURES ({len(interaction_features)} features):")
    for feat in interaction_features:
        if feat in df.columns:
            logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")

    logger.info(f"\n5. CYCLICAL FEATURES ({len(cyclical_features)} features):")
    for feat in cyclical_features:
        if feat in df.columns:
            if df[feat].dtype == 'int64':
                logger.info(f"   - {feat}: binary (0/1)")
            else:
                logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")

    logger.info(f"\nTotal engineered features: {len(feature_names)}")
    logger.info("="*70 + "\n")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("Studio Revenue Simulator - Feature Engineering")
    print("="*70 + "\n")

    # Step 1: Load generated data
    print("Step 1: Loading generated data...")
    input_path = 'data/processed/studio_data_2019_2025.csv'
    df = load_generated_data(input_path)

    print(f"\nInput dataset: {len(df)} rows × {len(df.columns)} columns")

    # Step 2: Initialize feature engineer
    print("\nStep 2: Initializing feature engineer...")
    engineer = FeatureEngineer()

    # Step 3: Apply feature engineering
    print("\nStep 3: Applying feature engineering transformations...")
    engineered_df = engineer.engineer_features(df, is_training=True)

    print(f"\nEngineered dataset: {len(engineered_df)} rows × {len(engineered_df.columns)} columns")
    print(f"Rows removed due to NaN: {len(df) - len(engineered_df)}")

    # Step 4: Display feature summary
    print("\nStep 4: Analyzing engineered features...")
    feature_names = engineer.get_feature_names()
    display_feature_summary(engineered_df, feature_names)

    # Step 5: Validate data quality
    print("Step 5: Validating data quality...")
    is_valid = validate_engineered_data(engineered_df)

    if not is_valid:
        print("\n[FAIL] Data validation failed. Please review issues above.")
        return

    # Step 6: Display data splits
    print("Step 6: Verifying data splits...")
    print("\nData Splits:")
    for split_name in ['train', 'validation', 'test']:
        split_data = engineered_df[engineered_df['split'] == split_name]
        pct = (len(split_data) / len(engineered_df)) * 100
        print(f"  {split_name.capitalize()}: {len(split_data)} rows ({pct:.1f}%)")

    # Step 7: Save engineered data
    print("\nStep 7: Saving engineered dataset...")
    output_path = 'data/processed/studio_data_engineered.csv'
    engineered_df.to_csv(output_path, index=False)
    print(f"[OK] Engineered data saved to {output_path}")

    # Step 8: Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input: {len(df)} rows -> Output: {len(engineered_df)} rows")
    print(f"Features: {len(df.columns)} -> {len(engineered_df.columns)} columns")
    print(f"Engineered features: {len(feature_names)}")
    print(f"Date range: {engineered_df['month_year'].min()} to {engineered_df['month_year'].max()}")
    print(f"Ready for model training: {'YES' if is_valid else 'NO'}")
    print("="*70 + "\n")

    print("\n[SUCCESS] Feature engineering complete!\n")


if __name__ == "__main__":
    main()
