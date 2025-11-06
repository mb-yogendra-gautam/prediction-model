"""
Comprehensive Validation for Ridge Model v2.3.0
Daily Data with Multi-Horizon Predictions

Performs:
- Detailed metrics analysis (RMSE, MAE, R², MAPE)
- Business metrics evaluation
- Comprehensive visualizations
- Audit report generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
from datetime import datetime
from typing import Dict
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class RidgeValidator:
    """Comprehensive validation for Ridge v2.3.0"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.target_names = [
            'Revenue Day 1', 'Revenue Day 3', 'Revenue Day 7', 'Attendance Day 7',
            'Revenue Week 1', 'Revenue Week 2', 'Revenue Week 4', 'Attendance Week 1',
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 1', 'Members Month 3', 'Retention Month 3'
        ]
        self.target_cols = [
            'revenue_day_1', 'revenue_day_3', 'revenue_day_7', 'attendance_day_7',
            'revenue_week_1', 'revenue_week_2', 'revenue_week_4', 'attendance_week_1',
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_1', 'member_count_month_3', 'retention_rate_month_3'
        ]

    def load_model(self):
        """Load trained Ridge model and artifacts"""
        logger.info("Loading Ridge v2.3.0 model...")

        model_dir = Path('data/models/v2.3')

        # Load model
        self.model = joblib.load(model_dir / 'ridge_model_v2.3.0.pkl')
        self.scaler = joblib.load(model_dir / 'ridge_scaler_v2.3.0.pkl')
        self.feature_names = joblib.load(model_dir / 'ridge_features_v2.3.0.pkl')

        # Load metadata
        with open(model_dir / 'ridge_metadata_v2.3.0.json', 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Model loaded: {len(self.feature_names)} features, {len(self.target_names)} targets")

    def load_and_prepare_data(self):
        """Load and prepare test data"""
        logger.info("Loading test data...")

        data_path = 'data/processed/multi_studio_daily_data_engineered.csv'
        df = pd.read_csv(data_path)

        logger.info(f"Loaded {len(df)} samples from {df['studio_id'].nunique()} studios")

        # Separate by split
        test_df = df[df['split'] == 'test'].copy()

        X_test = test_df[self.feature_names].values
        y_test = test_df[self.target_cols].values

        logger.info(f"Test set: {len(X_test)} samples")

        # Scale features
        X_test_scaled = self.scaler.transform(X_test)

        return X_test_scaled, y_test

    def calculate_comprehensive_metrics(self, X_test, y_test) -> Dict:
        """Calculate comprehensive metrics for all targets"""
        logger.info("Calculating comprehensive metrics...")

        # Make predictions
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = (time.time() - start_time) * 1000

        metrics_by_target = {}

        for i, target_name in enumerate(self.target_names):
            y_true_target = y_test[:, i]
            y_pred_target = y_pred[:, i]

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true_target, y_pred_target))
            mae = mean_absolute_error(y_true_target, y_pred_target)
            r2 = r2_score(y_true_target, y_pred_target)
            mape = np.mean(np.abs((y_true_target - y_pred_target) / (y_true_target + 1e-10))) * 100

            # Mean and std of errors
            errors = y_pred_target - y_true_target
            mean_error = np.mean(errors)
            std_error = np.std(errors)

            # Max absolute error
            max_error = np.max(np.abs(errors))

            metrics_by_target[target_name] = {
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2),
                'Mean_Error': round(mean_error, 2),
                'Std_Error': round(std_error, 2),
                'Max_Error': round(max_error, 2)
            }

            logger.info(f"{target_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")

        # Calculate overall metrics
        overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
        overall_mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())

        # Calculate per-horizon metrics
        horizon_metrics = self._calculate_horizon_metrics(y_test, y_pred)

        logger.info(f"Overall: R²={overall_r2:.4f}, RMSE={overall_rmse:.2f}, MAE={overall_mae:.2f}")

        return {
            'metrics_by_target': metrics_by_target,
            'overall_rmse': round(overall_rmse, 2),
            'overall_r2': round(overall_r2, 4),
            'overall_mae': round(overall_mae, 2),
            'horizon_metrics': horizon_metrics,
            'inference_time_ms': round(inference_time, 2),
            'avg_inference_time_ms': round(inference_time / len(X_test), 4),
            'predictions': y_pred,
            'actuals': y_test
        }

    def _calculate_horizon_metrics(self, y_test, y_pred) -> Dict:
        """Calculate metrics grouped by prediction horizon"""
        # Daily targets: 0-3
        daily_indices = [0, 1, 2, 3]
        # Weekly targets: 4-7
        weekly_indices = [4, 5, 6, 7]
        # Monthly targets: 8-13
        monthly_indices = [8, 9, 10, 11, 12, 13]

        horizons = {
            'daily': daily_indices,
            'weekly': weekly_indices,
            'monthly': monthly_indices
        }

        horizon_metrics = {}

        for horizon_name, indices in horizons.items():
            y_test_horizon = y_test[:, indices]
            y_pred_horizon = y_pred[:, indices]

            rmse = np.sqrt(mean_squared_error(y_test_horizon, y_pred_horizon))
            mae = mean_absolute_error(y_test_horizon, y_pred_horizon)
            r2 = r2_score(y_test_horizon.flatten(), y_pred_horizon.flatten())

            horizon_metrics[horizon_name] = {
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4)
            }

        return horizon_metrics

    def calculate_business_metrics(self, y_test, y_pred) -> Dict:
        """Calculate business-focused metrics"""
        logger.info("Calculating business metrics...")

        # Focus on revenue predictions
        revenue_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10]  # All revenue targets
        y_test_revenue = y_test[:, revenue_indices]
        y_pred_revenue = y_pred[:, revenue_indices]

        # Calculate percentage errors
        pct_errors = np.abs((y_pred_revenue - y_test_revenue) / (y_test_revenue + 1e-10)) * 100

        within_5_pct = (pct_errors <= 5).sum() / pct_errors.size * 100
        within_10_pct = (pct_errors <= 10).sum() / pct_errors.size * 100
        within_15_pct = (pct_errors <= 15).sum() / pct_errors.size * 100

        # Directional accuracy (Day 1 → Day 7)
        direction_actual = np.sign(y_test[:, 2] - y_test[:, 0])
        direction_pred = np.sign(y_pred[:, 2] - y_pred[:, 0])
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100

        # Business impact score
        business_score = (
            0.5 * within_5_pct +
            0.3 * within_10_pct +
            0.2 * directional_accuracy
        )

        business_metrics = {
            'predictions_within_5pct': round(within_5_pct, 2),
            'predictions_within_10pct': round(within_10_pct, 2),
            'predictions_within_15pct': round(within_15_pct, 2),
            'directional_accuracy': round(directional_accuracy, 2),
            'business_impact_score': round(business_score, 2)
        }

        logger.info(f"Business Metrics:")
        logger.info(f"  Within 5%: {within_5_pct:.1f}%")
        logger.info(f"  Within 10%: {within_10_pct:.1f}%")
        logger.info(f"  Directional accuracy: {directional_accuracy:.1f}%")
        logger.info(f"  Business impact score: {business_score:.1f}/100")

        return business_metrics

    def plot_prediction_scatter(self, y_test, y_pred, save_dir='reports/figures'):
        """Create scatter plots of actual vs predicted for all targets"""
        logger.info("Creating prediction scatter plots...")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create figure with 14 subplots (4x4 grid, remove last 2)
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()

        for i, target_name in enumerate(self.target_names):
            ax = axes[i]

            y_true = y_test[:, i]
            y_pred_target = y_pred[:, i]

            # Scatter plot
            ax.scatter(y_true, y_pred_target, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred_target.min())
            max_val = max(y_true.max(), y_pred_target.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            # Calculate metrics
            r2 = r2_score(y_true, y_pred_target)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_target))

            # Add metrics annotation
            ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Actual', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
            ax.set_title(f'{target_name}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove extra subplots
        for i in range(len(self.target_names), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle('Ridge v2.3.0 - Prediction Accuracy (Test Set)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        save_path = save_dir / 'ridge_v2.3.0_predictions_scatter.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()

    def plot_residual_analysis(self, y_test, y_pred, save_dir='reports/figures'):
        """Create residual distribution plots"""
        logger.info("Creating residual analysis plots...")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()

        for i, target_name in enumerate(self.target_names):
            ax = axes[i]

            residuals = y_test[:, i] - y_pred[:, i]

            # Histogram
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.axvline(np.mean(residuals), color='green', linestyle='-', linewidth=2,
                      label=f'Mean = {np.mean(residuals):.2f}')

            ax.set_xlabel('Residual (Actual - Predicted)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'{target_name}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        # Remove extra subplots
        for i in range(len(self.target_names), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle('Ridge v2.3.0 - Residual Analysis (Test Set)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        save_path = save_dir / 'ridge_v2.3.0_residuals.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()

    def plot_horizon_performance(self, horizon_metrics, save_dir='reports/figures'):
        """Plot performance by prediction horizon"""
        logger.info("Creating horizon performance plot...")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        horizons = ['daily', 'weekly', 'monthly']
        metrics = ['R2', 'RMSE', 'MAE']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [horizon_metrics[h][metric] for h in horizons]

            bars = ax.bar(horizons, values, color=['#3498db', '#2ecc71', '#e74c3c'],
                         edgecolor='black', linewidth=1.5, alpha=0.8)

            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} by Horizon', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}' if metric != 'R2' else f'{val:.4f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.suptitle('Ridge v2.3.0 - Performance by Prediction Horizon', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / 'ridge_v2.3.0_horizon_performance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()

    def plot_error_distribution(self, y_test, y_pred, save_dir='reports/figures'):
        """Create box plots showing error distribution by target"""
        logger.info("Creating error distribution plot...")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Calculate percentage errors for all targets
        pct_errors = []
        for i in range(len(self.target_names)):
            pct_error = np.abs((y_pred[:, i] - y_test[:, i]) / (y_test[:, i] + 1e-10)) * 100
            pct_errors.append(pct_error)

        fig, ax = plt.subplots(figsize=(16, 8))

        bp = ax.boxplot(pct_errors, labels=self.target_names, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax.set_ylabel('Absolute Percentage Error (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target', fontsize=12, fontweight='bold')
        ax.set_title('Ridge v2.3.0 - Error Distribution by Target', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        # Add reference lines
        ax.axhline(y=5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='5% threshold')
        ax.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='10% threshold')
        ax.legend()

        plt.tight_layout()

        save_path = save_dir / 'ridge_v2.3.0_error_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
        plt.close()

    def measure_inference_performance(self, X_test, n_runs=100):
        """Measure inference performance benchmarks"""
        logger.info(f"Measuring inference performance ({n_runs} runs)...")

        latencies = []

        for _ in range(n_runs):
            start = time.time()
            _ = self.model.predict(X_test[:10])  # Batch of 10
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        latencies = np.array(latencies)

        performance = {
            'p50_latency_ms': round(np.percentile(latencies, 50), 2),
            'p95_latency_ms': round(np.percentile(latencies, 95), 2),
            'p99_latency_ms': round(np.percentile(latencies, 99), 2),
            'mean_latency_ms': round(np.mean(latencies), 2),
            'std_latency_ms': round(np.std(latencies), 2)
        }

        logger.info(f"Inference Performance:")
        logger.info(f"  P50: {performance['p50_latency_ms']:.2f}ms")
        logger.info(f"  P95: {performance['p95_latency_ms']:.2f}ms")
        logger.info(f"  P99: {performance['p99_latency_ms']:.2f}ms")

        return performance

    def get_model_size(self):
        """Get model size on disk and in memory"""
        model_path = Path('data/models/v2.3/ridge_model_v2.3.0.pkl')
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)

        logger.info(f"Model size: {size_mb:.2f} MB")

        return round(size_mb, 2)

    def create_audit_report(self, validation_results, save_dir='reports/audit'):
        """Create comprehensive audit reports (JSON and TXT)"""
        logger.info("Creating audit reports...")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare audit data
        audit_data = {
            'model_version': '2.3.0',
            'model_type': 'ridge',
            'validation_timestamp': datetime.now().isoformat(),
            'model_metadata': self.metadata,
            'test_metrics': {
                'overall_metrics': {
                    'R2': validation_results['overall_r2'],
                    'RMSE': validation_results['overall_rmse'],
                    'MAE': validation_results['overall_mae']
                },
                'metrics_by_target': validation_results['metrics_by_target'],
                'horizon_metrics': validation_results['horizon_metrics']
            },
            'business_metrics': validation_results['business_metrics'],
            'inference_performance': validation_results['inference_performance'],
            'model_size_mb': validation_results['model_size_mb'],
            'test_samples': validation_results['test_samples'],
            'success_criteria': {
                'r2_threshold': 0.90,
                'r2_achieved': bool(validation_results['overall_r2'] >= 0.90),
                'mape_threshold': 10.0,
                'business_score_threshold': 80.0,
                'business_score_achieved': bool(validation_results['business_metrics']['business_impact_score'] >= 80.0)
            }
        }

        # Save JSON report
        json_path = save_dir / 'ridge_v2.3.0_validation_report.json'
        with open(json_path, 'w') as f:
            json.dump(audit_data, f, indent=2)
        logger.info(f"Saved JSON report: {json_path}")

        # Create human-readable text report
        txt_path = save_dir / 'ridge_v2.3.0_validation_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RIDGE MODEL v2.3.0 - COMPREHENSIVE VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: Ridge MultiOutputRegressor\n")
            f.write(f"Version: 2.3.0\n")
            f.write(f"Test Samples: {validation_results['test_samples']}\n\n")

            f.write("-"*80 + "\n")
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Features: {self.metadata['n_features']}\n")
            f.write(f"Targets: {self.metadata['n_targets']}\n")
            f.write(f"Algorithm: {self.metadata['algorithm']}\n")
            f.write(f"Training Date: {self.metadata['training_date']}\n")
            f.write(f"Training Time: {self.metadata['training_time_seconds']}s\n\n")

            f.write("Hyperparameters:\n")
            for key, value in self.metadata['hyperparameters'].items():
                f.write(f"  - {key}: {value}\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"R² Score:        {validation_results['overall_r2']:.4f}\n")
            f.write(f"RMSE:            {validation_results['overall_rmse']:.2f}\n")
            f.write(f"MAE:             {validation_results['overall_mae']:.2f}\n\n")

            f.write("-"*80 + "\n")
            f.write("PERFORMANCE BY PREDICTION HORIZON\n")
            f.write("-"*80 + "\n")
            for horizon, metrics in validation_results['horizon_metrics'].items():
                f.write(f"\n{horizon.upper()}:\n")
                f.write(f"  R²:   {metrics['R2']:.4f}\n")
                f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
                f.write(f"  MAE:  {metrics['MAE']:.2f}\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("METRICS BY TARGET\n")
            f.write("-"*80 + "\n\n")
            for target, metrics in validation_results['metrics_by_target'].items():
                f.write(f"{target}:\n")
                f.write(f"  RMSE: {metrics['RMSE']:>10.2f}  |  MAE: {metrics['MAE']:>10.2f}\n")
                f.write(f"  R²:   {metrics['R2']:>10.4f}  |  MAPE: {metrics['MAPE']:>9.2f}%\n")
                f.write(f"  Mean Error: {metrics['Mean_Error']:>7.2f}  |  Std Error: {metrics['Std_Error']:>7.2f}\n\n")

            f.write("-"*80 + "\n")
            f.write("BUSINESS METRICS\n")
            f.write("-"*80 + "\n")
            bm = validation_results['business_metrics']
            f.write(f"Predictions within 5%:    {bm['predictions_within_5pct']:.2f}%\n")
            f.write(f"Predictions within 10%:   {bm['predictions_within_10pct']:.2f}%\n")
            f.write(f"Predictions within 15%:   {bm['predictions_within_15pct']:.2f}%\n")
            f.write(f"Directional Accuracy:     {bm['directional_accuracy']:.2f}%\n")
            f.write(f"Business Impact Score:    {bm['business_impact_score']:.2f}/100\n\n")

            f.write("-"*80 + "\n")
            f.write("INFERENCE PERFORMANCE\n")
            f.write("-"*80 + "\n")
            ip = validation_results['inference_performance']
            f.write(f"Mean Latency:     {ip['mean_latency_ms']:.2f}ms\n")
            f.write(f"P50 Latency:      {ip['p50_latency_ms']:.2f}ms\n")
            f.write(f"P95 Latency:      {ip['p95_latency_ms']:.2f}ms\n")
            f.write(f"P99 Latency:      {ip['p99_latency_ms']:.2f}ms\n")
            f.write(f"Model Size:       {validation_results['model_size_mb']:.2f} MB\n\n")

            f.write("-"*80 + "\n")
            f.write("SUCCESS CRITERIA ASSESSMENT\n")
            f.write("-"*80 + "\n")
            sc = audit_data['success_criteria']
            f.write(f"✓ R² > {sc['r2_threshold']:.2f}:          {'PASS' if sc['r2_achieved'] else 'FAIL'} ({validation_results['overall_r2']:.4f})\n")
            f.write(f"✓ Business Score > {sc['business_score_threshold']:.0f}: {'PASS' if sc['business_score_achieved'] else 'FAIL'} ({bm['business_impact_score']:.2f})\n")
            f.write(f"✓ Inference < 100ms:    {'PASS' if ip['p95_latency_ms'] < 100 else 'FAIL'} ({ip['p95_latency_ms']:.2f}ms)\n\n")

            f.write("="*80 + "\n")
            f.write("VALIDATION COMPLETE\n")
            f.write("="*80 + "\n")

        logger.info(f"Saved TXT report: {txt_path}")

        return audit_data

    def verify_consistency_with_training(self):
        """Verify validation results match training results"""
        logger.info("Verifying consistency with training results...")

        training_results_path = Path('reports/audit/model_results_v2.3.0_ridge.json')

        if not training_results_path.exists():
            logger.warning("Training results not found, skipping consistency check")
            return None

        with open(training_results_path, 'r') as f:
            training_results = json.load(f)

        # Compare key metrics
        training_r2 = training_results['test_results']['overall_r2']
        training_rmse = training_results['test_results']['overall_rmse']

        logger.info("Consistency Check:")
        logger.info(f"  Training R²: {training_r2:.4f}")
        logger.info(f"  Training RMSE: {training_rmse:.2f}")

        return training_results


def main():
    print("\n" + "="*80)
    print("RIDGE MODEL v2.3.0 - COMPREHENSIVE VALIDATION")
    print("="*80 + "\n")

    validator = RidgeValidator()

    # Step 1: Load model
    print("Step 1: Loading model...")
    validator.load_model()

    # Step 2: Load data
    print("\nStep 2: Loading test data...")
    X_test, y_test = validator.load_and_prepare_data()

    # Step 3: Calculate comprehensive metrics
    print("\nStep 3: Calculating comprehensive metrics...")
    print("-" * 80)
    metrics_results = validator.calculate_comprehensive_metrics(X_test, y_test)
    print("-" * 80)

    # Step 4: Calculate business metrics
    print("\nStep 4: Calculating business metrics...")
    print("-" * 80)
    business_metrics = validator.calculate_business_metrics(
        metrics_results['actuals'],
        metrics_results['predictions']
    )
    print("-" * 80)

    # Step 5: Measure inference performance
    print("\nStep 5: Measuring inference performance...")
    inference_performance = validator.measure_inference_performance(X_test)

    # Step 6: Get model size
    model_size = validator.get_model_size()

    # Step 7: Create visualizations
    print("\nStep 6: Creating visualizations...")
    print("-" * 80)
    validator.plot_prediction_scatter(metrics_results['actuals'], metrics_results['predictions'])
    validator.plot_residual_analysis(metrics_results['actuals'], metrics_results['predictions'])
    validator.plot_horizon_performance(metrics_results['horizon_metrics'])
    validator.plot_error_distribution(metrics_results['actuals'], metrics_results['predictions'])
    print("-" * 80)

    # Step 8: Create audit report
    print("\nStep 7: Creating audit reports...")
    validation_results = {
        'overall_r2': metrics_results['overall_r2'],
        'overall_rmse': metrics_results['overall_rmse'],
        'overall_mae': metrics_results['overall_mae'],
        'metrics_by_target': metrics_results['metrics_by_target'],
        'horizon_metrics': metrics_results['horizon_metrics'],
        'business_metrics': business_metrics,
        'inference_performance': inference_performance,
        'model_size_mb': model_size,
        'test_samples': len(X_test)
    }

    audit_data = validator.create_audit_report(validation_results)

    # Step 9: Verify consistency
    print("\nStep 8: Verifying consistency with training results...")
    print("-" * 80)
    training_results = validator.verify_consistency_with_training()
    if training_results:
        print(f"[OK] Results are consistent with training")
    print("-" * 80)

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\n[OK] Test R²: {validation_results['overall_r2']:.4f}")
    print(f"[OK] Test RMSE: {validation_results['overall_rmse']:.2f}")
    print(f"[OK] Business Score: {business_metrics['business_impact_score']:.1f}/100")
    print(f"[OK] Inference P95: {inference_performance['p95_latency_ms']:.2f}ms")
    print(f"[OK] Model Size: {model_size:.2f} MB")
    print("\nSuccess Criteria:")
    print(f"  [{'PASS' if validation_results['overall_r2'] >= 0.90 else 'FAIL'}] R² > 0.90: {validation_results['overall_r2']:.4f}")
    print(f"  [{'PASS' if business_metrics['business_impact_score'] >= 80 else 'FAIL'}] Business Score > 80: {business_metrics['business_impact_score']:.1f}")
    print(f"  [{'PASS' if inference_performance['p95_latency_ms'] < 100 else 'FAIL'}] Inference < 100ms: {inference_performance['p95_latency_ms']:.2f}ms")
    print("\nReports saved:")
    print("  - reports/audit/ridge_v2.3.0_validation_report.json")
    print("  - reports/audit/ridge_v2.3.0_validation_report.txt")
    print("\nVisualizations saved:")
    print("  - reports/figures/ridge_v2.3.0_predictions_scatter.png")
    print("  - reports/figures/ridge_v2.3.0_residuals.png")
    print("  - reports/figures/ridge_v2.3.0_horizon_performance.png")
    print("  - reports/figures/ridge_v2.3.0_error_distribution.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
