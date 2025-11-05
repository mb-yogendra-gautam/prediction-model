"""
Data Validation Module for Studio Revenue Simulator

Provides comprehensive data quality checks and validation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for studio metrics"""

    def __init__(self):
        self.validation_rules = self._define_validation_rules()

    def _define_validation_rules(self) -> Dict:
        """Define validation rules for each metric"""
        return {
            'total_members': {'min': 50, 'max': 1000, 'type': 'int'},
            'new_members': {'min': 0, 'max': 100, 'type': 'int'},
            'churned_members': {'min': 0, 'max': 200, 'type': 'int'},
            'retention_rate': {'min': 0.5, 'max': 1.0, 'type': 'float'},
            'avg_ticket_price': {'min': 50.0, 'max': 500.0, 'type': 'float'},
            'total_classes_held': {'min': 50, 'max': 500, 'type': 'int'},
            'total_class_attendance': {'min': 0, 'max': 10000, 'type': 'int'},
            'class_attendance_rate': {'min': 0.3, 'max': 1.0, 'type': 'float'},
            'staff_count': {'min': 3, 'max': 50, 'type': 'int'},
            'staff_utilization_rate': {'min': 0.5, 'max': 1.0, 'type': 'float'},
            'upsell_rate': {'min': 0.0, 'max': 0.6, 'type': 'float'},
            'total_revenue': {'min': 5000.0, 'max': 200000.0, 'type': 'float'},
        }

    def validate_full_dataset(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Run complete validation suite on dataset

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Running comprehensive data validation...")

        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'checks_passed': [],
            'checks_failed': [],
            'warnings': [],
            'errors': []
        }

        # Run all validation checks
        checks = [
            ('Schema Validation', self._validate_schema(df)),
            ('Missing Values', self._validate_missing_values(df)),
            ('Data Types', self._validate_data_types(df)),
            ('Value Ranges', self._validate_value_ranges(df)),
            ('Business Logic', self._validate_business_logic(df)),
            ('Temporal Consistency', self._validate_temporal_consistency(df)),
            ('Statistical Outliers', self._validate_statistical_outliers(df)),
            ('Correlations', self._validate_correlations(df)),
        ]

        all_passed = True
        for check_name, (passed, issues) in checks:
            if passed:
                report['checks_passed'].append(check_name)
                logger.info(f"✓ {check_name}: PASSED")
            else:
                report['checks_failed'].append(check_name)
                report['errors'].extend(issues)
                all_passed = False
                logger.warning(f"✗ {check_name}: FAILED")
                for issue in issues:
                    logger.warning(f"  - {issue}")

        report['is_valid'] = all_passed
        return all_passed, report

    def _validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate dataset schema"""
        required_columns = [
            'month_year', 'month_index', 'year_index',
            'total_members', 'new_members', 'churned_members',
            'retention_rate', 'avg_ticket_price',
            'total_classes_held', 'total_class_attendance', 'class_attendance_rate',
            'staff_count', 'staff_utilization_rate',
            'total_revenue', 'membership_revenue', 'class_pack_revenue', 'retail_revenue',
            'upsell_rate'
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            return False, [f"Missing required columns: {missing_cols}"]

        return True, []

    def _validate_missing_values(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]

        if len(missing_cols) > 0:
            issues = []
            for col, count in missing_cols.items():
                pct = (count / len(df)) * 100
                issues.append(f"{col}: {count} missing ({pct:.1f}%)")
            return False, issues

        return True, []

    def _validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data types match expectations"""
        issues = []

        # Check date column
        if 'month_year' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['month_year']):
                issues.append("month_year should be datetime type")

        # Check numeric columns
        numeric_cols = [
            'total_members', 'new_members', 'churned_members',
            'retention_rate', 'avg_ticket_price', 'total_revenue'
        ]

        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"{col} should be numeric type")

        return len(issues) == 0, issues

    def _validate_value_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate values are within acceptable ranges"""
        issues = []

        for col, rules in self.validation_rules.items():
            if col not in df.columns:
                continue

            # Check min/max bounds
            out_of_range = (df[col] < rules['min']) | (df[col] > rules['max'])
            if out_of_range.any():
                count = out_of_range.sum()
                min_val = df[col].min()
                max_val = df[col].max()
                issues.append(
                    f"{col}: {count} values out of range "
                    f"[{rules['min']}, {rules['max']}]. "
                    f"Actual: [{min_val:.2f}, {max_val:.2f}]"
                )

        return len(issues) == 0, issues

    def _validate_business_logic(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate business logic constraints"""
        issues = []

        # Revenue components should sum to total revenue (within tolerance)
        if all(col in df.columns for col in ['total_revenue', 'membership_revenue',
                                              'class_pack_revenue', 'retail_revenue']):
            calculated = (
                df['membership_revenue'] +
                df['class_pack_revenue'] +
                df['retail_revenue']
            )
            diff = np.abs(df['total_revenue'] - calculated)
            tolerance = 1000  # $1000 tolerance
            inconsistent = diff > tolerance

            if inconsistent.any():
                count = inconsistent.sum()
                max_diff = diff.max()
                issues.append(
                    f"Revenue inconsistency in {count} rows "
                    f"(max diff: ${max_diff:.2f})"
                )

        # Members: new - churned should roughly equal change
        if all(col in df.columns for col in ['total_members', 'new_members', 'churned_members']):
            member_change = df['total_members'].diff()
            expected_change = df['new_members'] - df['churned_members']
            diff = np.abs(member_change - expected_change)

            # Allow some tolerance for rounding
            inconsistent = (diff > 5) & (diff.notna())
            if inconsistent.any():
                count = inconsistent.sum()
                issues.append(
                    f"Member count inconsistency in {count} rows"
                )

        # Retention rate consistency
        if all(col in df.columns for col in ['retention_rate', 'churned_members', 'total_members']):
            # Churned should be roughly (1 - retention) * total_members
            prev_members = df['total_members'].shift(1)
            expected_churned = prev_members * (1 - df['retention_rate'])
            diff = np.abs(df['churned_members'] - expected_churned)

            # Allow 10% tolerance
            tolerance = expected_churned * 0.1
            inconsistent = (diff > tolerance) & (diff.notna())

            if inconsistent.any():
                count = inconsistent.sum()
                issues.append(
                    f"Retention rate inconsistency in {count} rows"
                )

        return len(issues) == 0, issues

    def _validate_temporal_consistency(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate temporal sequence and consistency"""
        issues = []

        # Check date sequence
        if 'month_year' in df.columns:
            # Dates should be sorted
            if not df['month_year'].is_monotonic_increasing:
                issues.append("Dates are not in chronological order")

            # Check for gaps
            date_diff = df['month_year'].diff()
            expected_diff = pd.Timedelta(days=28)  # Roughly 1 month

            # Allow 25-35 day range for monthly data
            gaps = ((date_diff < pd.Timedelta(days=25)) |
                   (date_diff > pd.Timedelta(days=35))) & (date_diff.notna())

            if gaps.any():
                count = gaps.sum()
                issues.append(f"Date gaps or inconsistencies in {count} places")

        return len(issues) == 0, issues

    def _validate_statistical_outliers(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Detect statistical outliers (warnings only)"""
        warnings = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['month_index', 'year_index']:
                continue

            # Use IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))

            if outliers.any():
                count = outliers.sum()
                pct = (count / len(df)) * 100
                if pct > 2:  # Only warn if > 2% are outliers
                    warnings.append(
                        f"{col}: {count} outliers ({pct:.1f}%) "
                        f"outside [{lower_bound:.2f}, {upper_bound:.2f}]"
                    )

        # Outliers are warnings, not failures
        return True, warnings

    def _validate_correlations(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate expected correlations between features"""
        warnings = []

        # Check expected positive correlations
        expected_positive = [
            ('total_members', 'total_revenue'),
            ('avg_ticket_price', 'total_revenue'),
            ('retention_rate', 'total_revenue'),
            ('total_classes_held', 'total_revenue'),
        ]

        for col1, col2 in expected_positive:
            if col1 in df.columns and col2 in df.columns:
                corr = df[col1].corr(df[col2])
                if corr < 0.3:  # Expect at least moderate positive correlation
                    warnings.append(
                        f"Low correlation between {col1} and {col2}: {corr:.2f} "
                        f"(expected > 0.3)"
                    )

        # Correlations are warnings, not failures
        return True, warnings

    def generate_validation_report(self, df: pd.DataFrame) -> str:
        """Generate formatted validation report"""
        is_valid, report = self.validate_full_dataset(df)

        report_str = "\n" + "="*70 + "\n"
        report_str += "DATA VALIDATION REPORT\n"
        report_str += "="*70 + "\n\n"

        report_str += f"Dataset: {report['total_rows']} rows × {report['total_columns']} columns\n\n"

        report_str += "CHECKS PASSED:\n"
        for check in report['checks_passed']:
            report_str += f"  ✓ {check}\n"

        if report['checks_failed']:
            report_str += "\nCHECKS FAILED:\n"
            for check in report['checks_failed']:
                report_str += f"  ✗ {check}\n"

        if report['errors']:
            report_str += "\nERRORS:\n"
            for error in report['errors']:
                report_str += f"  - {error}\n"

        if report['warnings']:
            report_str += "\nWARNINGS:\n"
            for warning in report['warnings']:
                report_str += f"  ! {warning}\n"

        report_str += "\n" + "="*70 + "\n"
        report_str += f"OVERALL STATUS: {'PASSED' if is_valid else 'FAILED'}\n"
        report_str += "="*70 + "\n"

        return report_str


if __name__ == "__main__":
    # Test validator
    logging.basicConfig(level=logging.INFO)

    print("Data Validator Module")
    print("Testing validation rules...")

    validator = DataValidator()
    print(f"Loaded {len(validator.validation_rules)} validation rules")
    print("Validation rules configured for:")
    for col in validator.validation_rules.keys():
        print(f"  - {col}")
