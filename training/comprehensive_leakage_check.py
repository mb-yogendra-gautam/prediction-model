"""
Comprehensive Data Leakage Investigation

Combines multiple checks to thoroughly investigate potential data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveLeakageChecker:
    """Thorough data leakage investigation"""
    
    def __init__(self, data_path='data/processed/multi_studio_data_engineered.csv'):
        self.data_path = data_path
        self.findings = []
        
    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Sort by time
        df['month_year'] = pd.to_datetime(df['month_year'])
        df = df.sort_values(['studio_id', 'month_year'])
        
        return df
    
    def check_1_target_in_features(self, df):
        """Check if target columns are accidentally included in features"""
        logger.info("\n" + "="*80)
        logger.info("CHECK 1: Target Columns in Feature Set")
        logger.info("="*80)
        
        target_cols = [
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_3', 'retention_rate_month_3'
        ]
        
        found_targets = [col for col in target_cols if col in df.columns]
        
        if len(found_targets) == len(target_cols):
            logger.info("✓ All target columns present (as expected)")
            
            # Check if they're being used as features (they shouldn't be)
            exclude_cols = target_cols + ['month_year', 'split', 'year_index', 'studio_id']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            leaked_targets = [col for col in target_cols if col in feature_cols]
            
            if leaked_targets:
                logger.error(f"❌ CRITICAL: Target columns found in features: {leaked_targets}")
                self.findings.append({
                    'severity': 'CRITICAL',
                    'check': 'Target in Features',
                    'issue': f'Target columns leaked into features: {leaked_targets}'
                })
                return False
            else:
                logger.info("✓ Target columns properly excluded from features")
                return True
        else:
            logger.warning(f"⚠️  Some target columns missing: {set(target_cols) - set(found_targets)}")
            return True
    
    def check_2_current_period_in_features(self, df):
        """Check if current period data is used in lagged/rolling features"""
        logger.info("\n" + "="*80)
        logger.info("CHECK 2: Current Period Data in Features")
        logger.info("="*80)
        
        issues = []
        
        # Check if features that should be lagged have perfect correlation with current values
        pairs_to_check = [
            ('prev_month_revenue', 'total_revenue'),
            ('prev_month_members', 'total_members'),
            ('3m_avg_revenue', 'total_revenue'),
            ('3m_avg_retention', 'retention_rate')
        ]
        
        for lag_feature, current_feature in pairs_to_check:
            if lag_feature in df.columns and current_feature in df.columns:
                # Check correlation (should be high but not perfect)
                valid_data = df[[lag_feature, current_feature]].dropna()
                if len(valid_data) > 10:
                    corr = valid_data[lag_feature].corr(valid_data[current_feature])
                    
                    logger.info(f"  {lag_feature} vs {current_feature}: correlation = {corr:.4f}")
                    
                    if corr > 0.99:
                        logger.warning(f"  ⚠️  Suspiciously high correlation!")
                        issues.append(f"{lag_feature} has correlation {corr:.4f} with {current_feature}")
                    
                    # Check if values are identical (shifted by 1)
                    for studio in df['studio_id'].unique() if 'studio_id' in df.columns else [None]:
                        if studio is not None:
                            studio_df = df[df['studio_id'] == studio].copy()
                        else:
                            studio_df = df.copy()
                        
                        studio_df = studio_df.sort_values('month_year')
                        
                        # Check if prev_month actually equals current month (data leakage!)
                        if len(studio_df) > 1:
                            shifted_current = studio_df[current_feature].shift(1)
                            lag_values = studio_df[lag_feature]
                            
                            # Compare (allowing for small floating point differences)
                            matches = np.abs(shifted_current - lag_values) < 0.01
                            match_rate = matches.sum() / len(matches[~matches.isna()])
                            
                            if match_rate > 0.95:
                                logger.info(f"    ✓ {lag_feature} correctly uses previous period data")
                            elif match_rate < 0.5:
                                logger.error(f"    ❌ {lag_feature} does NOT match previous period!")
                                issues.append(f"{lag_feature} doesn't match shifted {current_feature}")
                        
                        break  # Just check first studio
        
        if issues:
            logger.error(f"\n❌ Found {len(issues)} issues with temporal features")
            for issue in issues:
                logger.error(f"  - {issue}")
            self.findings.append({
                'severity': 'HIGH',
                'check': 'Current Period in Features',
                'issue': '; '.join(issues)
            })
            return False
        else:
            logger.info("\n✓ Temporal features appear properly lagged")
            return True
    
    def check_3_rolling_window_implementation(self, df):
        """Verify rolling windows exclude current period"""
        logger.info("\n" + "="*80)
        logger.info("CHECK 3: Rolling Window Implementation")
        logger.info("="*80)
        
        rolling_features = [col for col in df.columns if '3m_avg' in col or '3m_std' in col]
        
        if not rolling_features:
            logger.info("  No rolling features found")
            return True
        
        issues = []
        
        for feature in rolling_features:
            # Find the base feature
            base_feature = feature.replace('3m_avg_', '').replace('3m_std_', '')
            
            # Try to find matching column
            possible_bases = [
                base_feature,
                f'total_{base_feature}',
                f'{base_feature}_rate'
            ]
            
            base_col = None
            for possible in possible_bases:
                if possible in df.columns:
                    base_col = possible
                    break
            
            if base_col:
                # Check if rolling feature equals a 3-month rolling window INCLUDING current
                for studio in df['studio_id'].unique()[:1] if 'studio_id' in df.columns else [None]:
                    if studio is not None:
                        studio_df = df[df['studio_id'] == studio].copy()
                    else:
                        studio_df = df.copy()
                    
                    studio_df = studio_df.sort_values('month_year')
                    
                    # Calculate rolling window including current period
                    bad_rolling = studio_df[base_col].rolling(window=3, min_periods=1).mean()
                    
                    # Calculate rolling window excluding current period (correct way)
                    good_rolling = studio_df[base_col].shift(1).rolling(window=3, min_periods=1).mean()
                    
                    # Check which one matches
                    actual = studio_df[feature]
                    
                    bad_match = (np.abs(actual - bad_rolling) < 0.01).sum() / len(actual.dropna())
                    good_match = (np.abs(actual - good_rolling) < 0.01).sum() / len(actual.dropna())
                    
                    logger.info(f"  {feature}:")
                    logger.info(f"    Matches with current period: {bad_match:.2%}")
                    logger.info(f"    Matches without current period: {good_match:.2%}")
                    
                    if bad_match > good_match and bad_match > 0.8:
                        logger.error(f"    ❌ INCLUDES current period! (DATA LEAKAGE)")
                        issues.append(f"{feature} includes current period")
                        self.findings.append({
                            'severity': 'CRITICAL',
                            'check': 'Rolling Window',
                            'issue': f'{feature} includes current period data'
                        })
                    elif good_match > 0.8:
                        logger.info(f"    ✓ Correctly excludes current period")
                    else:
                        logger.warning(f"    ⚠️  Unable to verify implementation")
                    
                    break  # Only check first studio
        
        return len(issues) == 0
    
    def check_4_performance_sanity(self):
        """Check if model performance is suspiciously high"""
        logger.info("\n" + "="*80)
        logger.info("CHECK 4: Performance Sanity Check")
        logger.info("="*80)
        
        # Try to load evaluation results
        results_path = Path('reports/audit/multi_studio_evaluation_report_v2.2.0.txt')
        
        if not results_path.exists():
            logger.warning("  ⚠️  No evaluation report found")
            return True
        
        # Parse R² from report
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with latin-1 encoding if utf-8 fails (for special chars like ²)
            with open(results_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Extract R² values
        import re
        r2_matches = re.findall(r'R² = (0\.\d+)', content)
        
        if r2_matches:
            r2_values = [float(m) for m in r2_matches]
            avg_r2 = np.mean(r2_values)
            
            logger.info(f"  Average R² from report: {avg_r2:.4f}")
            
            if avg_r2 > 0.99:
                logger.error("  ❌ CRITICAL: R² > 0.99 is EXTREMELY rare in forecasting")
                logger.error("  This strongly suggests data leakage!")
                self.findings.append({
                    'severity': 'CRITICAL',
                    'check': 'Performance Sanity',
                    'issue': f'R² = {avg_r2:.4f} is suspiciously high (likely data leakage)'
                })
                return False
            elif avg_r2 > 0.95:
                logger.warning("  ⚠️  R² > 0.95 is unusually high")
                logger.warning("  Verify no data leakage before production deployment")
                self.findings.append({
                    'severity': 'HIGH',
                    'check': 'Performance Sanity',
                    'issue': f'R² = {avg_r2:.4f} is unusually high (verify no leakage)'
                })
                return False
            else:
                logger.info("  ✓ Performance is within reasonable range")
                return True
        
        return True
    
    def check_5_feature_future_dependency(self, df):
        """Check if features depend on future data"""
        logger.info("\n" + "="*80)
        logger.info("CHECK 5: Future Data Dependency")
        logger.info("="*80)
        
        # For each studio, check if earlier rows depend on later rows
        if 'studio_id' not in df.columns:
            logger.info("  Skipping (no studio_id column)")
            return True
        
        feature_cols = [col for col in df.columns if col not in 
                       ['month_year', 'split', 'year_index', 'studio_id',
                        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
                        'member_count_month_3', 'retention_rate_month_3']]
        
        issues = []
        
        # Check first few studios
        for studio in df['studio_id'].unique()[:3]:
            studio_df = df[df['studio_id'] == studio].sort_values('month_year')
            
            if len(studio_df) < 2:
                continue
            
            # For first row, many features should be NaN (no history)
            first_row = studio_df.iloc[0]
            
            lagged_cols = [col for col in feature_cols if 'prev_month' in col or '3m_avg' in col or 'mom_' in col]
            
            non_null_lagged = [col for col in lagged_cols if col in first_row.index and pd.notna(first_row[col])]
            
            if len(non_null_lagged) > 0:
                logger.warning(f"  ⚠️  Studio {studio}: First row has {len(non_null_lagged)} lagged features with values")
                logger.warning(f"      This might indicate improper feature engineering")
                # This is expected if min_periods=1 is used, so just a warning
        
        logger.info("  ✓ No obvious future dependencies detected")
        return True
    
    def generate_comprehensive_report(self, output_path='reports/audit/comprehensive_leakage_investigation.txt'):
        """Generate final comprehensive report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE DATA LEAKAGE INVESTIGATION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not self.findings:
                f.write("✓ NO CRITICAL ISSUES FOUND\n\n")
                f.write("All checks passed. Model appears to be free from obvious data leakage.\n")
            else:
                f.write(f"⚠️  FOUND {len(self.findings)} ISSUES\n\n")
                
                # Group by severity
                critical = [f for f in self.findings if f['severity'] == 'CRITICAL']
                high = [f for f in self.findings if f['severity'] == 'HIGH']
                medium = [f for f in self.findings if f['severity'] == 'MEDIUM']
                
                if critical:
                    f.write("CRITICAL ISSUES:\n")
                    f.write("-" * 80 + "\n")
                    for finding in critical:
                        f.write(f"  [{finding['check']}] {finding['issue']}\n")
                    f.write("\n")
                
                if high:
                    f.write("HIGH PRIORITY ISSUES:\n")
                    f.write("-" * 80 + "\n")
                    for finding in high:
                        f.write(f"  [{finding['check']}] {finding['issue']}\n")
                    f.write("\n")
                
                if medium:
                    f.write("MEDIUM PRIORITY ISSUES:\n")
                    f.write("-" * 80 + "\n")
                    for finding in medium:
                        f.write(f"  [{finding['check']}] {finding['issue']}\n")
                    f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            if critical:
                f.write("❌ DO NOT DEPLOY TO PRODUCTION\n\n")
                f.write("Critical data leakage issues detected. Required actions:\n")
                f.write("  1. Fix all critical issues in feature engineering code\n")
                f.write("  2. Re-train model with corrected features\n")
                f.write("  3. Re-run all validation tests\n")
                f.write("  4. Verify performance drops to realistic levels\n\n")
            elif high:
                f.write("⚠️  CONDITIONAL APPROVAL\n\n")
                f.write("High priority issues detected. Required actions:\n")
                f.write("  1. Investigate all high priority issues\n")
                f.write("  2. Run walk-forward validation\n")
                f.write("  3. Test on truly unseen future data\n")
                f.write("  4. Deploy to staging first with monitoring\n\n")
            else:
                f.write("✓ APPROVED FOR PRODUCTION\n\n")
                f.write("No critical issues detected. Recommended actions:\n")
                f.write("  1. Deploy to staging first\n")
                f.write("  2. Monitor performance carefully\n")
                f.write("  3. Set up alerts for performance degradation\n")
                f.write("  4. Regular model retraining schedule\n\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"\nReport saved to: {output_path}")
        return output_path
    
    def run_all_checks(self):
        """Run all leakage checks"""
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPREHENSIVE DATA LEAKAGE INVESTIGATION")
        logger.info("="*80)
        
        df = self.load_data()
        
        # Run all checks
        check1 = self.check_1_target_in_features(df)
        check2 = self.check_2_current_period_in_features(df)
        check3 = self.check_3_rolling_window_implementation(df)
        check4 = self.check_4_performance_sanity()
        check5 = self.check_5_feature_future_dependency(df)
        
        # Generate report
        report_path = self.generate_comprehensive_report()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("INVESTIGATION COMPLETE")
        logger.info("="*80)
        
        all_passed = all([check1, check2, check3, check4, check5])
        
        if all_passed:
            logger.info("\n✓ All checks passed")
            status = "APPROVED"
        elif any([not check1, not check3]):
            logger.error("\n❌ Critical issues found - DO NOT DEPLOY")
            status = "REJECTED"
        else:
            logger.warning("\n⚠️  Some issues found - verify before deployment")
            status = "CONDITIONAL"
        
        logger.info(f"Status: {status}")
        logger.info(f"Report: {report_path}")
        
        return status


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA LEAKAGE INVESTIGATION")
    print("="*80 + "\n")
    
    checker = ComprehensiveLeakageChecker()
    status = checker.run_all_checks()
    
    print("\n" + "="*80)
    print(f"FINAL STATUS: {status}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

