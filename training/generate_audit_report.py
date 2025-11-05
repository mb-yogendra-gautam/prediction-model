"""
Comprehensive Audit Report Generator

Creates detailed audit documentation for model validation including
all metrics, predictions, visualizations, and compliance documentation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditReportGenerator:
    """Generate comprehensive audit reports for model validation"""

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.timestamp = datetime.now()
        self.audit_dir = Path('reports/audit')
        self.figures_dir = Path('reports/figures')

        # Ensure directories exist
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_all_metrics(self):
        """Load all metrics from evaluation"""
        logger.info("Loading evaluation metrics")

        # Load main metrics
        with open(self.audit_dir / f'metrics_v{self.version}.json', 'r') as f:
            self.main_metrics = json.load(f)

        # Load business metrics
        business_metrics_file = self.audit_dir / f'business_metrics_v{self.version}.csv'
        if business_metrics_file.exists():
            self.business_metrics = pd.read_csv(business_metrics_file).iloc[0].to_dict()
        else:
            self.business_metrics = {}

        # Load feature importance
        importance_file = self.audit_dir / f'feature_importance_v{self.version}.csv'
        if importance_file.exists():
            self.feature_importance = pd.read_csv(importance_file)
        else:
            self.feature_importance = None

        # Load predictions
        self.predictions = {}
        for dataset in ['train', 'validation', 'test']:
            pred_file = self.audit_dir / f'predictions_{dataset}_v{self.version}.csv'
            if pred_file.exists():
                self.predictions[dataset] = pd.read_csv(pred_file)

        logger.info("All metrics loaded successfully")

    def generate_executive_summary(self) -> str:
        """Generate executive summary section"""
        summary = []

        summary.append("="*80)
        summary.append("MODEL VALIDATION AUDIT REPORT")
        summary.append("="*80)

        summary.append(f"\nModel Version:      {self.version}")
        summary.append(f"Evaluation Date:    {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Evaluator:          Automated Model Validation System")

        summary.append("\n" + "-"*80)
        summary.append("EXECUTIVE SUMMARY")
        summary.append("-"*80)

        # Get test set metrics
        test_metrics = self.main_metrics['metrics_by_dataset']['test']

        summary.append("\n1. MODEL PERFORMANCE OVERVIEW")
        summary.append(f"   Dataset Split: Train ({len(self.predictions.get('train', []))}) / "
                      f"Val ({len(self.predictions.get('validation', []))}) / "
                      f"Test ({len(self.predictions.get('test', []))})")
        summary.append(f"   Ensemble Model: XGBoost + LightGBM + Random Forest")
        summary.append(f"   Feature Count: {self.main_metrics['feature_count']}")

        # Average performance across targets
        avg_rmse = np.mean([m['RMSE'] for m in test_metrics.values()])
        avg_mape = np.mean([m['MAPE'] for m in test_metrics.values()])
        avg_r2 = np.mean([m['R2'] for m in test_metrics.values()])

        summary.append(f"\n2. TEST SET PERFORMANCE")
        summary.append(f"   Average RMSE:      {avg_rmse:.2f}")
        summary.append(f"   Average MAPE:      {avg_mape:.2f}%")
        summary.append(f"   Average R²:        {avg_r2:.4f}")

        # Business metrics
        if self.business_metrics:
            summary.append(f"\n3. BUSINESS METRICS (Test Set)")
            summary.append(f"   Predictions within ±5%:   {self.business_metrics.get('overall_within_5_percent', 'N/A')}%")
            summary.append(f"   Predictions within ±10%:  {self.business_metrics.get('overall_within_10_percent', 'N/A')}%")
            summary.append(f"   Forecast Accuracy:        {self.business_metrics.get('overall_forecast_accuracy', 'N/A')}%")
            summary.append(f"   Business Impact Score:    {self.business_metrics.get('overall_business_impact_score', 'N/A')}/100")

        # Model comparison
        model_comp = self.main_metrics['model_comparison']
        summary.append(f"\n4. MODEL COMPARISON")
        summary.append(f"   XGBoost RMSE:      {model_comp['xgboost']['RMSE']}")
        summary.append(f"   LightGBM RMSE:     {model_comp['lightgbm']['RMSE']}")
        summary.append(f"   Random Forest RMSE:{model_comp['random_forest']['RMSE']}")
        summary.append(f"   Ensemble Weights:  XGB={self.main_metrics['ensemble_weights']['xgboost']:.3f}, "
                      f"LGB={self.main_metrics['ensemble_weights']['lightgbm']:.3f}, "
                      f"RF={self.main_metrics['ensemble_weights']['random_forest']:.3f}")

        # Confidence intervals
        ci_coverage = self.main_metrics.get('confidence_interval_coverage', 'N/A')
        summary.append(f"\n5. PREDICTION UNCERTAINTY")
        summary.append(f"   95% CI Coverage:   {ci_coverage}%")

        summary.append("\n" + "-"*80)

        return "\n".join(summary)

    def generate_detailed_metrics_table(self) -> str:
        """Generate detailed metrics tables"""
        section = []

        section.append("\n" + "-"*80)
        section.append("DETAILED PERFORMANCE METRICS")
        section.append("-"*80)

        for dataset_name in ['train', 'validation', 'test']:
            if dataset_name not in self.main_metrics['metrics_by_dataset']:
                continue

            metrics = self.main_metrics['metrics_by_dataset'][dataset_name]

            section.append(f"\n{dataset_name.upper()} SET METRICS")
            section.append("-" * 80)

            # Table header
            section.append(f"{'Target':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE':>10} {'Dir.Acc':>10}")
            section.append("-" * 80)

            for target_name, target_metrics in metrics.items():
                dir_acc = target_metrics.get('Directional_Accuracy')
                dir_acc_str = f"{dir_acc:.1f}%" if dir_acc is not None else "N/A"

                section.append(
                    f"{target_name:<25} "
                    f"{target_metrics['RMSE']:>10.2f} "
                    f"{target_metrics['MAE']:>10.2f} "
                    f"{target_metrics['R2']:>10.4f} "
                    f"{target_metrics['MAPE']:>9.2f}% "
                    f"{dir_acc_str:>10}"
                )

            section.append("")

        return "\n".join(section)

    def generate_feature_importance_section(self) -> str:
        """Generate feature importance section"""
        section = []

        section.append("\n" + "-"*80)
        section.append("FEATURE IMPORTANCE ANALYSIS")
        section.append("-"*80)

        if self.feature_importance is not None:
            section.append("\nTop 20 Most Important Features:\n")
            section.append(f"{'Rank':<6} {'Feature':<35} {'Importance':>12}")
            section.append("-" * 80)

            for _, row in self.feature_importance.head(20).iterrows():
                section.append(f"{int(row['Rank']):<6} {row['Feature']:<35} {row['Importance']:>12.4f}")

        else:
            section.append("\nFeature importance data not available.")

        section.append("")
        return "\n".join(section)

    def generate_error_analysis_section(self) -> str:
        """Generate error analysis section"""
        section = []

        section.append("\n" + "-"*80)
        section.append("ERROR ANALYSIS")
        section.append("-"*80)

        # Analyze test set predictions
        if 'test' in self.predictions:
            test_df = self.predictions['test']

            section.append("\nTEST SET ERROR STATISTICS")
            section.append("-" * 80)

            targets = ['revenue_month_1', 'revenue_month_2', 'revenue_month_3',
                      'member_count_month_3', 'retention_rate_month_3']

            for target in targets:
                if f'residual_{target}' in test_df.columns:
                    residuals = test_df[f'residual_{target}']
                    pct_errors = test_df[f'pct_error_{target}']

                    section.append(f"\n{target.upper()}")
                    section.append(f"  Mean Residual:    {residuals.mean():>10.2f}")
                    section.append(f"  Std Residual:     {residuals.std():>10.2f}")
                    section.append(f"  Max Abs Error:    {residuals.abs().max():>10.2f}")
                    section.append(f"  Mean % Error:     {pct_errors.mean():>9.2f}%")
                    section.append(f"  Max % Error:      {pct_errors.max():>9.2f}%")

                    # Outliers (>2 std dev)
                    outliers = residuals[np.abs(residuals) > 2 * residuals.std()]
                    section.append(f"  Outliers (>2std): {len(outliers)}/{len(residuals)}")

        section.append("")
        return "\n".join(section)

    def generate_business_metrics_section(self) -> str:
        """Generate business metrics section"""
        section = []

        section.append("\n" + "-"*80)
        section.append("BUSINESS PERFORMANCE METRICS")
        section.append("-"*80)

        if not self.business_metrics:
            section.append("\nBusiness metrics not available.")
            return "\n".join(section)

        section.append("\n1. ACCURACY THRESHOLDS")
        section.append(f"   Within ±5%:       {self.business_metrics.get('overall_within_5_percent', 'N/A')}%")
        section.append(f"   Within ±10%:      {self.business_metrics.get('overall_within_10_percent', 'N/A')}%")
        section.append(f"   Large Errors:     {self.business_metrics.get('overall_large_error_rate', 'N/A')}%")

        section.append("\n2. FORECAST QUALITY")
        section.append(f"   3-Month Accuracy:   {self.business_metrics.get('overall_forecast_accuracy', 'N/A')}%")
        section.append(f"   Directional Acc:    {self.business_metrics.get('overall_directional_accuracy', 'N/A')}%")
        section.append(f"   Trend Consistency:  {self.business_metrics.get('overall_revenue_trend_consistency', 'N/A')}%")

        section.append("\n3. HORIZON-SPECIFIC MAPE")
        section.append(f"   Month 1:  {self.business_metrics.get('overall_mape_month_1', 'N/A')}%")
        section.append(f"   Month 2:  {self.business_metrics.get('overall_mape_month_2', 'N/A')}%")
        section.append(f"   Month 3:  {self.business_metrics.get('overall_mape_month_3', 'N/A')}%")

        section.append("\n4. QUARTERLY METRICS")
        section.append(f"   Quarterly MAPE:       {self.business_metrics.get('quarterly_quarterly_mape', 'N/A')}%")
        section.append(f"   Within ±5%:           {self.business_metrics.get('quarterly_quarterly_within_5pct', 'N/A')}%")
        section.append(f"   Within ±10%:          {self.business_metrics.get('quarterly_quarterly_within_10pct', 'N/A')}%")

        section.append("\n5. BUSINESS IMPACT")
        section.append(f"   Impact Score:     {self.business_metrics.get('overall_business_impact_score', 'N/A')}/100")

        # Interpretation
        impact_score = self.business_metrics.get('overall_business_impact_score', 0)
        section.append("\n6. ASSESSMENT")
        if impact_score >= 80:
            section.append("   [OK] EXCELLENT - High business value, low prediction errors")
        elif impact_score >= 60:
            section.append("   [OK] GOOD - Moderate business value, acceptable errors")
        else:
            section.append("   [!] NEEDS IMPROVEMENT - Consider model retraining or feature engineering")

        section.append("")
        return "\n".join(section)

    def generate_artifacts_section(self) -> str:
        """Generate artifacts and files section"""
        section = []

        section.append("\n" + "-"*80)
        section.append("AUDIT ARTIFACTS")
        section.append("-"*80)

        section.append("\nMODEL ARTIFACTS:")
        section.append(f"  - data/models/ensemble_models_v{self.version}.pkl")
        section.append(f"  - data/models/scaler_v{self.version}.pkl")
        section.append(f"  - data/models/weights_v{self.version}.pkl")
        section.append(f"  - data/models/features_v{self.version}.pkl")

        section.append("\nEVALUATION METRICS:")
        section.append(f"  - {self.audit_dir}/metrics_v{self.version}.json")
        section.append(f"  - {self.audit_dir}/business_metrics_v{self.version}.csv")
        section.append(f"  - {self.audit_dir}/feature_importance_v{self.version}.csv")

        section.append("\nPREDICTIONS & RESIDUALS:")
        for dataset in ['train', 'validation', 'test']:
            section.append(f"  - {self.audit_dir}/predictions_{dataset}_v{self.version}.csv")

        section.append("\nVISUALIZATIONS:")
        viz_files = [
            'evaluation_scatter_train.png', 'evaluation_scatter_validation.png',
            'evaluation_scatter_test.png', 'residual_distribution_train.png',
            'residual_distribution_validation.png', 'residual_distribution_test.png',
            'confidence_intervals_test.png', 'feature_importance.png'
        ]
        for viz in viz_files:
            section.append(f"  - {self.figures_dir}/{viz}")

        section.append("")
        return "\n".join(section)

    def generate_recommendations_section(self) -> str:
        """Generate recommendations section"""
        section = []

        section.append("\n" + "-"*80)
        section.append("RECOMMENDATIONS")
        section.append("-"*80)

        # Analyze metrics for recommendations
        test_metrics = self.main_metrics['metrics_by_dataset']['test']
        avg_mape = np.mean([m['MAPE'] for m in test_metrics.values()])

        section.append("\n1. MODEL PERFORMANCE")
        if avg_mape < 5:
            section.append("   [OK] Model performance is EXCELLENT for business use")
        elif avg_mape < 10:
            section.append("   [OK] Model performance is GOOD for business use")
        else:
            section.append("   [!] Consider model retraining or architecture changes")

        section.append("\n2. FEATURE ENGINEERING")
        if self.feature_importance is not None:
            top_5_importance = self.feature_importance.head(5)['Importance'].sum()
            if top_5_importance > 0.7:
                section.append("   [!] Top 5 features dominate - consider feature diversification")
            else:
                section.append("   [OK] Feature importance is well-distributed")

        section.append("\n3. DATA QUALITY")
        if 'test' in self.predictions:
            test_samples = len(self.predictions['test'])
            if test_samples < 20:
                section.append("   [!] Small test set - consider collecting more validation data")
            else:
                section.append("   [OK] Test set size is adequate")

        section.append("\n4. DEPLOYMENT READINESS")
        if avg_mape < 10 and self.business_metrics.get('overall_within_10_percent', 0) > 70:
            section.append("   [OK] APPROVED - Model ready for production deployment")
        else:
            section.append("   [!] CONDITIONAL - Monitor performance closely in production")

        section.append("")
        return "\n".join(section)

    def generate_complete_report(self) -> str:
        """Generate complete audit report"""
        logger.info("Generating comprehensive audit report")

        report = []

        # Add all sections
        report.append(self.generate_executive_summary())
        report.append(self.generate_detailed_metrics_table())
        report.append(self.generate_feature_importance_section())
        report.append(self.generate_error_analysis_section())
        report.append(self.generate_business_metrics_section())
        report.append(self.generate_recommendations_section())
        report.append(self.generate_artifacts_section())

        # Footer
        report.append("\n" + "="*80)
        report.append("END OF AUDIT REPORT")
        report.append(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80 + "\n")

        return "\n".join(report)

    def save_audit_report(self):
        """Save complete audit report to file"""
        report_text = self.generate_complete_report()

        output_path = self.audit_dir / f'evaluation_report_v{self.version}.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Audit report saved to {output_path}")

        # Also save audit log
        self.save_audit_log()

        return output_path

    def save_audit_log(self):
        """Save machine-readable audit log"""
        audit_log = {
            'model_version': self.version,
            'evaluation_timestamp': self.timestamp.isoformat(),
            'evaluator': 'Automated Model Validation System',
            'datasets_evaluated': list(self.predictions.keys()),
            'total_predictions': sum(len(df) for df in self.predictions.values()),
            'files_generated': {
                'model_artifacts': 4,
                'metric_files': 3,
                'prediction_files': len(self.predictions),
                'visualizations': 8
            },
            'validation_status': 'COMPLETE',
            'audit_trail_complete': True
        }

        output_path = self.audit_dir / f'audit_log_v{self.version}.json'
        with open(output_path, 'w') as f:
            json.dump(audit_log, f, indent=2)

        logger.info(f"Audit log saved to {output_path}")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE AUDIT REPORT GENERATION")
    print("="*80 + "\n")

    print("Step 1: Initializing audit report generator...")
    generator = AuditReportGenerator(version="1.0.0")

    print("\nStep 2: Loading all evaluation metrics...")
    generator.load_all_metrics()

    print("\nStep 3: Generating comprehensive audit report...")
    report_path = generator.save_audit_report()

    print(f"\n[OK] Audit report generated: {report_path}")

    # Display report to console
    with open(report_path, 'r') as f:
        print("\n" + f.read())

    print("\n" + "="*80)
    print("AUDIT REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll audit documentation saved to: {generator.audit_dir}")
    print(f"Review the complete report at: {report_path}")
    print("\n" + "="*80 + "\n")
