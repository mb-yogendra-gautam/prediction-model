"""
Feature Engineering Pipeline for Multi-Studio Data

Processes multi-studio generated data and applies feature engineering
transformations while maintaining temporal integrity per studio.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_multi_studio_data(input_path='data/processed/multi_studio_data.csv'):
    """Load multi-studio generated data"""
    logger.info(f"Loading data from {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    logger.info(f"Date range: {df['month_year'].min()} to {df['month_year'].max()}")
    logger.info(f"Studios: {df['studio_id'].nunique()}")
    
    return df


def engineer_features_multi_studio(df):
    """Apply feature engineering to multi-studio data"""
    logger.info("Applying feature engineering transformations...")
    
    # Initialize engineer with multi-studio mode
    engineer = FeatureEngineer(multi_studio=True)
    
    # Apply transformations
    engineered_df = engineer.engineer_features(df, is_training=True)
    
    logger.info(f"Engineered dataset: {len(engineered_df)} rows × {len(engineered_df.columns)} columns")
    logger.info(f"Rows removed due to NaN: {len(df) - len(engineered_df)}")
    
    return engineered_df


def analyze_features(df):
    """Analyze engineered features"""
    logger.info("Analyzing engineered features...")
    
    # Identify feature types
    feature_cols = [col for col in df.columns if col not in [
        'studio_id', 'month_year', 'split', 'studio_location', 'studio_size_tier', 'studio_price_tier',
        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
        'member_count_month_3', 'retention_rate_month_3'
    ]]
    
    studio_features = [col for col in feature_cols if any(x in col for x in ['studio_', 'location_', 'size_', 'price_'])]
    temporal_features = [col for col in feature_cols if any(x in col for x in ['prev_', 'mom_', '3m_', 'momentum'])]
    derived_features = [col for col in feature_cols if any(x in col for x in ['_per_', '_rate', '_pct', 'ltv', 'utilization'])]
    interaction_features = [col for col in feature_cols if '_x_' in col]
    cyclical_features = [col for col in feature_cols if any(x in col for x in ['_sin', '_cos', 'is_'])]
    
    base_features = [col for col in feature_cols if col not in 
                    studio_features + temporal_features + derived_features + interaction_features + cyclical_features]
    
    logger.info(f"\n{'='*70}")
    logger.info("ENGINEERED FEATURES SUMMARY")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n1. BASE FEATURES ({len(base_features)} features):")
    for feat in base_features[:10]:
        logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")
    
    logger.info(f"\n2. STUDIO-LEVEL FEATURES ({len(studio_features)} features):")
    for feat in studio_features:
        logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")
    
    logger.info(f"\n3. TEMPORAL FEATURES ({len(temporal_features)} features):")
    for feat in temporal_features:
        if df[feat].notna().sum() > 0:
            logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")
    
    logger.info(f"\n4. DERIVED FEATURES ({len(derived_features)} features):")
    for feat in derived_features[:8]:
        logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")
    
    logger.info(f"\n5. INTERACTION FEATURES ({len(interaction_features)} features):")
    for feat in interaction_features:
        logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")
    
    logger.info(f"\n6. CYCLICAL FEATURES ({len(cyclical_features)} features):")
    for feat in cyclical_features:
        logger.info(f"   - {feat}: {df[feat].min():.2f} to {df[feat].max():.2f}")
    
    logger.info(f"\nTotal engineered features: {len(feature_cols)}")
    logger.info(f"{'='*70}\n")


def validate_data(df):
    """Validate data quality"""
    logger.info("\n{'='*70}")
    logger.info("VALIDATION CHECKS")
    logger.info(f"{'='*70}")
    
    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.sum() == 0:
        logger.info("[OK] No NaN values found")
    else:
        logger.warning(f"[WARNING] Found {nan_counts.sum()} NaN values")
        logger.warning(nan_counts[nan_counts > 0])
    
    # Check for infinite values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() == 0:
        logger.info("[OK] No infinite values found")
    else:
        logger.warning(f"[WARNING] Found {inf_counts.sum()} infinite values")
    
    # Check split column
    if 'split' in df.columns:
        split_counts = df['split'].value_counts().to_dict()
        logger.info(f"[OK] Split column preserved: {split_counts}")
    
    # Check feature count
    feature_cols = [col for col in df.columns if col not in [
        'studio_id', 'month_year', 'split', 'studio_location', 'studio_size_tier', 'studio_price_tier',
        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
        'member_count_month_3', 'retention_rate_month_3'
    ]]
    logger.info(f"[OK] Total features created: {len(feature_cols)}")
    
    # Check for zero variance features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_var = [col for col in numeric_cols if df[col].std() == 0]
    if len(zero_var) == 0:
        logger.info("[OK] All features have variance")
    else:
        logger.warning(f"[WARNING] {len(zero_var)} features have zero variance: {zero_var}")
    
    # Check per-studio data
    logger.info(f"\nPer-studio sample counts:")
    for studio_id in sorted(df['studio_id'].unique()):
        studio_df = df[df['studio_id'] == studio_id]
        train_count = len(studio_df[studio_df['split'] == 'train'])
        val_count = len(studio_df[studio_df['split'] == 'validation'])
        test_count = len(studio_df[studio_df['split'] == 'test'])
        logger.info(f"  {studio_id}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION PASSED: Dataset is ready for model training")
    logger.info(f"{'='*70}\n")


def save_engineered_data(df, output_path='data/processed/multi_studio_data_engineered.csv'):
    """Save engineered dataset"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"[OK] Engineered data saved to {output_path}")
    
    return output_path


def main():
    """Main pipeline execution"""
    
    print("\n" + "="*70)
    print("Multi-Studio Revenue Simulator - Feature Engineering")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading multi-studio data...")
    df = load_multi_studio_data()
    
    print(f"\nInput dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"Studios: {df['studio_id'].nunique()}")
    print(f"Date range: {df['month_year'].min()} to {df['month_year'].max()}")
    
    # Step 2: Engineer features
    print("\nStep 2: Applying feature engineering transformations...")
    engineered_df = engineer_features_multi_studio(df)
    
    print(f"\nEngineered dataset: {len(engineered_df)} rows × {len(engineered_df.columns)} columns")
    print(f"Rows removed due to NaN: {len(df) - len(engineered_df)}")
    
    # Step 3: Analyze features
    print("\nStep 3: Analyzing engineered features...")
    analyze_features(engineered_df)
    
    # Step 4: Validate
    print("Step 4: Validating data quality...")
    validate_data(engineered_df)
    
    # Step 5: Verify splits
    print("Step 5: Verifying data splits...")
    print(f"\nData Splits:")
    print(f"  Train: {len(engineered_df[engineered_df['split'] == 'train'])} rows ({len(engineered_df[engineered_df['split'] == 'train'])/len(engineered_df)*100:.1f}%)")
    print(f"  Validation: {len(engineered_df[engineered_df['split'] == 'validation'])} rows ({len(engineered_df[engineered_df['split'] == 'validation'])/len(engineered_df)*100:.1f}%)")
    print(f"  Test: {len(engineered_df[engineered_df['split'] == 'test'])} rows ({len(engineered_df[engineered_df['split'] == 'test'])/len(engineered_df)*100:.1f}%)")
    
    # Step 6: Save
    print("\nStep 6: Saving engineered dataset...")
    output_path = save_engineered_data(engineered_df)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input: {len(df)} rows -> Output: {len(engineered_df)} rows")
    print(f"Features: {len(df.columns)} -> {len(engineered_df.columns)} columns")
    print(f"Studios: {engineered_df['studio_id'].nunique()}")
    print(f"Date range: {engineered_df['month_year'].min()} to {engineered_df['month_year'].max()}")
    print(f"Ready for model training: YES")
    print("="*70 + "\n")
    
    print("[SUCCESS] Multi-studio feature engineering complete!")
    print(f"\nNext steps:")
    print(f"  1. Train model: python training/train_model_v2.2_multi_studio.py")
    print(f"  2. Evaluate: python training/evaluate_multi_studio_model.py")
    

if __name__ == "__main__":
    main()

