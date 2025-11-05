"""
Data Leakage Detection Script

Checks for temporal integrity in feature engineering to ensure
future information doesn't leak into predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLeakageDetector:
    """Detect potential data leakage in features"""
    
    def __init__(self, data_path='data/processed/studio_data_engineered.csv'):
        self.df = pd.read_csv(data_path)
        self.suspicious_features = []
        self.leakage_report = {}
        
    def check_feature_timing(self):
        """Check if features only use past information"""
        logger.info("Checking feature temporal integrity...")
        
        suspicious = {
            'revenue_momentum': 'May include current/future revenue',
            'estimated_ltv': 'Lifetime value should only use past data',
            '3m_avg_revenue': 'Rolling average might include current month',
            '3m_avg_retention': 'Rolling average might include current period',
            '3m_avg_attendance': 'Rolling average might include current period',
            'prev_month_revenue': 'Verify this is truly previous month',
            'mom_revenue_growth': 'Growth calculation timing',
            'mom_member_growth': 'Growth calculation timing'
        }
        
        available_features = set(self.df.columns)
        
        for feature, concern in suspicious.items():
            if feature in available_features:
                self.suspicious_features.append(feature)
                logger.warning(f"⚠️  {feature}: {concern}")
        
        return suspicious
    
    def check_target_correlation(self):
        """Check if features are too highly correlated with targets"""
        logger.info("Checking feature-target correlations...")
        
        target_cols = [
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_3', 'retention_rate_month_3'
        ]
        
        feature_cols = [col for col in self.df.columns 
                       if col not in target_cols + ['month_year', 'split', 'year_index']]
        
        high_corr_features = {}
        
        for target in target_cols:
            if target in self.df.columns:
                correlations = self.df[feature_cols].corrwith(self.df[target]).abs()
                
                # Flag features with correlation > 0.95
                high_corr = correlations[correlations > 0.95].sort_values(ascending=False)
                
                if len(high_corr) > 0:
                    high_corr_features[target] = high_corr.to_dict()
                    logger.warning(f"\n⚠️  High correlation with {target}:")
                    for feat, corr in high_corr.items():
                        logger.warning(f"     {feat}: {corr:.4f}")
        
        return high_corr_features
    
    def check_rolling_features(self):
        """Verify rolling features don't include current period"""
        logger.info("Checking rolling window features...")
        
        rolling_features = [col for col in self.df.columns if '3m_avg' in col or '3m_std' in col]
        
        issues = {}
        
        for feature in rolling_features:
            # Check if rolling average is properly lagged
            # Compare feature value at time t with target at time t
            # They should not be identical (which would indicate leakage)
            
            if feature in self.df.columns:
                # Check variance - if too low, might be using current data
                variance = self.df[feature].var()
                mean = self.df[feature].mean()
                cv = variance / (mean ** 2) if mean != 0 else 0
                
                issues[feature] = {
                    'variance': variance,
                    'mean': mean,
                    'coefficient_of_variation': cv
                }
                
                logger.info(f"  {feature}: mean={mean:.2f}, var={variance:.2f}, cv={cv:.4f}")
        
        return issues
    
    def check_future_information(self):
        """Check if any feature contains information from the future"""
        logger.info("Checking for future information leakage...")
        
        # Sort by date
        self.df['month_year'] = pd.to_datetime(self.df['month_year'])
        self.df = self.df.sort_values('month_year')
        
        issues = []
        
        # For each row, check if any feature value depends on future rows
        # This is a simplified check - in practice, review feature engineering code
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['month_year', 'split', 'year_index'] 
                       and 'month_' not in col]
        
        # Check for sudden jumps that might indicate future leakage
        for col in feature_cols[:10]:  # Sample first 10 features
            if self.df[col].dtype in [np.float64, np.int64]:
                # Calculate diff
                diffs = self.df[col].diff().abs()
                
                # Check for anomalous jumps
                if diffs.std() > 0:
                    z_scores = (diffs - diffs.mean()) / diffs.std()
                    outliers = z_scores > 3
                    
                    if outliers.sum() > 0:
                        issues.append({
                            'feature': col,
                            'outlier_count': int(outliers.sum()),
                            'max_jump': float(diffs.max())
                        })
        
        if issues:
            logger.warning(f"\n⚠️  Found {len(issues)} features with anomalous jumps:")
            for issue in issues[:5]:  # Show first 5
                logger.warning(f"     {issue}")
        
        return issues
    
    def check_train_test_contamination(self):
        """Check if test data information leaked into training"""
        logger.info("Checking train/test data contamination...")
        
        if 'split' not in self.df.columns:
            logger.warning("No 'split' column found - cannot check contamination")
            return None
        
        train_df = self.df[self.df['split'] == 'train']
        test_df = self.df[self.df['split'] == 'test']
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['month_year', 'split', 'year_index'] 
                       and 'month_' not in col and 'retention_rate_' not in col]
        
        # Check if train and test distributions are similar (good)
        # or if train has info from test (bad)
        
        contamination_scores = {}
        
        for col in feature_cols[:20]:  # Sample features
            if train_df[col].dtype in [np.float64, np.int64]:
                train_mean = train_df[col].mean()
                test_mean = test_df[col].mean()
                train_std = train_df[col].std()
                test_std = test_df[col].std()
                
                # Calculate difference
                mean_diff = abs(train_mean - test_mean) / (train_mean + 1e-10)
                std_diff = abs(train_std - test_std) / (train_std + 1e-10)
                
                if mean_diff > 0.5 or std_diff > 0.5:
                    contamination_scores[col] = {
                        'mean_diff_pct': mean_diff * 100,
                        'std_diff_pct': std_diff * 100
                    }
        
        if contamination_scores:
            logger.warning(f"\n⚠️  Train/Test distribution differences found:")
            for feat, scores in list(contamination_scores.items())[:5]:
                logger.warning(f"     {feat}: mean_diff={scores['mean_diff_pct']:.1f}%, "
                             f"std_diff={scores['std_diff_pct']:.1f}%")
        else:
            logger.info("✓ Train/Test distributions look similar")
        
        return contamination_scores
    
    def generate_report(self, output_path='reports/audit/data_leakage_report.txt'):
        """Generate comprehensive leakage detection report"""
        logger.info("Generating leakage detection report...")
        
        # Run all checks
        timing_issues = self.check_feature_timing()
        correlation_issues = self.check_target_correlation()
        rolling_issues = self.check_rolling_features()
        future_issues = self.check_future_information()
        contamination_issues = self.check_train_test_contamination()
        
        # Create report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DATA LEAKAGE DETECTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. TEMPORAL INTEGRITY CHECK\n")
            f.write("-" * 80 + "\n")
            if timing_issues:
                f.write("Suspicious Features (Manual Review Required):\n\n")
                for feature, concern in timing_issues.items():
                    f.write(f"  ⚠️  {feature}\n")
                    f.write(f"      Concern: {concern}\n\n")
            else:
                f.write("  ✓ No obvious timing issues detected\n\n")
            
            f.write("\n2. HIGH TARGET CORRELATION CHECK\n")
            f.write("-" * 80 + "\n")
            if correlation_issues:
                f.write("Features with very high correlation to targets (> 0.95):\n\n")
                for target, corrs in correlation_issues.items():
                    f.write(f"  Target: {target}\n")
                    for feat, corr in corrs.items():
                        f.write(f"    - {feat}: {corr:.4f}\n")
                    f.write("\n")
            else:
                f.write("  ✓ No suspiciously high correlations found\n\n")
            
            f.write("\n3. ROLLING FEATURE CHECK\n")
            f.write("-" * 80 + "\n")
            if rolling_issues:
                f.write("Rolling window feature statistics:\n\n")
                for feat, stats in rolling_issues.items():
                    f.write(f"  {feat}:\n")
                    f.write(f"    Mean: {stats['mean']:.2f}\n")
                    f.write(f"    Variance: {stats['variance']:.2f}\n")
                    f.write(f"    CV: {stats['coefficient_of_variation']:.4f}\n\n")
            else:
                f.write("  ✓ No rolling features found\n\n")
            
            f.write("\n4. FUTURE INFORMATION CHECK\n")
            f.write("-" * 80 + "\n")
            if future_issues:
                f.write(f"Found {len(future_issues)} features with anomalous jumps:\n\n")
                for issue in future_issues[:10]:
                    f.write(f"  - {issue['feature']}: {issue['outlier_count']} outliers, "
                           f"max jump = {issue['max_jump']:.2f}\n")
            else:
                f.write("  ✓ No obvious future information leakage detected\n\n")
            
            f.write("\n5. TRAIN/TEST CONTAMINATION CHECK\n")
            f.write("-" * 80 + "\n")
            if contamination_issues:
                f.write("Train/Test distribution differences:\n\n")
                for feat, diffs in list(contamination_issues.items())[:10]:
                    f.write(f"  - {feat}:\n")
                    f.write(f"      Mean diff: {diffs['mean_diff_pct']:.1f}%\n")
                    f.write(f"      Std diff: {diffs['std_diff_pct']:.1f}%\n\n")
            elif contamination_issues is None:
                f.write("  ⚠️  Cannot check - no split column\n\n")
            else:
                f.write("  ✓ Train/Test distributions are similar\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            f.write("Priority Actions:\n\n")
            f.write("1. MANUAL CODE REVIEW (CRITICAL)\n")
            f.write("   Review src/features/feature_engineer.py for:\n")
            f.write("   - Rolling window calculations (should exclude current period)\n")
            f.write("   - Lagged features (should use proper time offsets)\n")
            f.write("   - Derived features (should only use past information)\n\n")
            
            f.write("2. FEATURE ENGINEERING FIXES\n")
            f.write("   For each suspicious feature, verify:\n")
            f.write("   - Calculation uses only data available at prediction time\n")
            f.write("   - Proper lagging is applied\n")
            f.write("   - No target information leaks\n\n")
            
            f.write("3. TESTING\n")
            f.write("   - Implement forward-validation (walk-forward)\n")
            f.write("   - Test on truly unseen future data\n")
            f.write("   - Monitor for performance drop in production\n\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"Report saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("DATA LEAKAGE DETECTION SUMMARY")
        print("="*80)
        print(f"\nSuspicious Features Found: {len(timing_issues)}")
        print(f"High Correlations Found: {len(correlation_issues)}")
        print(f"Anomalous Jumps Found: {len(future_issues) if future_issues else 0}")
        print(f"\n✓ Detailed report saved to: {output_path}")
        print("\n⚠️  IMPORTANT: Manual code review of feature_engineer.py is REQUIRED")
        print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("DATA LEAKAGE DETECTION - Temporal Integrity Check")
    print("="*80 + "\n")
    
    detector = DataLeakageDetector()
    detector.generate_report()
    
    print("\nNext Steps:")
    print("  1. Review the generated report carefully")
    print("  2. Manually inspect src/features/feature_engineer.py")
    print("  3. Fix any identified temporal leakage issues")
    print("  4. Re-train model with corrected features")
    print("  5. Use walk-forward validation for final testing")


if __name__ == "__main__":
    main()

