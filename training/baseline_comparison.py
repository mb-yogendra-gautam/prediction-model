"""
Baseline Model Comparison

Compare trained model against simple baseline models to validate that
the complex model actually adds value.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineComparison:
    """Compare model against baseline approaches"""
    
    def __init__(self, data_path='data/processed/multi_studio_data_engineered.csv',
                 model_path='models/ridge_model_v2.2.0.pkl',
                 scaler_path='models/scaler_v2.2.0.pkl'):
        self.data_path = data_path
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Define targets
        target_cols = [
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_3', 'retention_rate_month_3'
        ]
        
        # Split by the 'split' column
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        
        # Get features
        exclude_cols = target_cols + ['month_year', 'split', 'year_index', 'studio_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_cols].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_cols].values
        
        # Also keep the raw data for baseline calculations
        test_df_full = test_df.copy()
        
        logger.info(f"Train: {len(train_df)} samples")
        logger.info(f"Test: {len(test_df)} samples")
        
        return X_train, y_train, X_test, y_test, test_df_full, feature_cols, target_cols
    
    def baseline_last_value(self, test_df, target_cols):
        """Baseline: Use last known value (persistence model)"""
        logger.info("\nBaseline 1: Last Value (Persistence)")
        
        predictions = {}
        actuals = {}
        
        # For revenue predictions, use current month revenue
        if 'total_revenue' in test_df.columns:
            for i, target in enumerate(['revenue_month_1', 'revenue_month_2', 'revenue_month_3']):
                predictions[target] = test_df['total_revenue'].values
                actuals[target] = test_df[target].values
        
        # For member count, use current members
        if 'total_members' in test_df.columns and 'member_count_month_3' in target_cols:
            predictions['member_count_month_3'] = test_df['total_members'].values
            actuals['member_count_month_3'] = test_df['member_count_month_3'].values
        
        # For retention, use current retention
        if 'retention_rate' in test_df.columns and 'retention_rate_month_3' in target_cols:
            predictions['retention_rate_month_3'] = test_df['retention_rate'].values
            actuals['retention_rate_month_3'] = test_df['retention_rate_month_3'].values
        
        return predictions, actuals
    
    def baseline_mean(self, y_train, y_test, target_cols):
        """Baseline: Use training mean"""
        logger.info("\nBaseline 2: Training Mean")
        
        predictions = {}
        actuals = {}
        
        for i, target in enumerate(target_cols):
            mean_value = y_train[:, i].mean()
            predictions[target] = np.full(len(y_test), mean_value)
            actuals[target] = y_test[:, i]
        
        return predictions, actuals
    
    def baseline_linear_trend(self, test_df, target_cols):
        """Baseline: Simple linear trend from recent history"""
        logger.info("\nBaseline 3: Linear Trend")
        
        predictions = {}
        actuals = {}
        
        # Use prev_month values if available
        if 'prev_month_revenue' in test_df.columns and 'total_revenue' in test_df.columns:
            for i, target in enumerate(['revenue_month_1', 'revenue_month_2', 'revenue_month_3']):
                # Simple trend: current + (current - previous)
                trend = test_df['total_revenue'] - test_df['prev_month_revenue']
                predictions[target] = test_df['total_revenue'] + trend * (i + 1)
                actuals[target] = test_df[target].values
        
        # For members
        if 'prev_month_members' in test_df.columns and 'total_members' in test_df.columns:
            if 'member_count_month_3' in target_cols:
                trend = test_df['total_members'] - test_df['prev_month_members']
                predictions['member_count_month_3'] = test_df['total_members'] + trend * 3
                actuals['member_count_month_3'] = test_df['member_count_month_3'].values
        
        # For retention (assume constant)
        if 'retention_rate' in test_df.columns and 'retention_rate_month_3' in target_cols:
            predictions['retention_rate_month_3'] = test_df['retention_rate'].values
            actuals['retention_rate_month_3'] = test_df['retention_rate_month_3'].values
        
        return predictions, actuals
    
    def evaluate_model(self, X_train, y_train, X_test, y_test, target_cols):
        """Evaluate the trained model"""
        logger.info("\nTrained Model (Ridge)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and predict
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        predictions = {}
        actuals = {}
        
        for i, target in enumerate(target_cols):
            predictions[target] = y_pred[:, i]
            actuals[target] = y_test[:, i]
        
        return predictions, actuals
    
    def calculate_metrics(self, predictions, actuals):
        """Calculate metrics for a set of predictions"""
        metrics = {}
        
        for target in predictions.keys():
            y_true = actuals[target]
            y_pred = predictions[target]
            
            # Handle NaN values
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid_mask.sum() == 0:
                continue
            
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            
            metrics[target] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        
        return metrics
    
    def compare_all(self):
        """Run all comparisons"""
        # Load data
        X_train, y_train, X_test, y_test, test_df, feature_cols, target_cols = self.load_data()
        
        results = {}
        
        # Baseline 1: Last value
        preds1, actuals1 = self.baseline_last_value(test_df, target_cols)
        results['baseline_last_value'] = self.calculate_metrics(preds1, actuals1)
        
        # Baseline 2: Mean
        preds2, actuals2 = self.baseline_mean(y_train, y_test, target_cols)
        results['baseline_mean'] = self.calculate_metrics(preds2, actuals2)
        
        # Baseline 3: Linear trend
        preds3, actuals3 = self.baseline_linear_trend(test_df, target_cols)
        results['baseline_linear_trend'] = self.calculate_metrics(preds3, actuals3)
        
        # Trained model
        preds_model, actuals_model = self.evaluate_model(X_train, y_train, X_test, y_test, target_cols)
        results['trained_model'] = self.calculate_metrics(preds_model, actuals_model)
        
        return results
    
    def analyze_results(self, results):
        """Analyze comparison results"""
        logger.info("\n" + "="*80)
        logger.info("BASELINE COMPARISON ANALYSIS")
        logger.info("="*80)
        
        target_names = {
            'revenue_month_1': 'Revenue Month 1',
            'revenue_month_2': 'Revenue Month 2',
            'revenue_month_3': 'Revenue Month 3',
            'member_count_month_3': 'Members Month 3',
            'retention_rate_month_3': 'Retention Month 3'
        }
        
        for target, display_name in target_names.items():
            logger.info(f"\n{display_name}:")
            logger.info("-" * 80)
            
            # Get metrics for each model
            model_metrics = {}
            for model_name in results.keys():
                if target in results[model_name]:
                    model_metrics[model_name] = results[model_name][target]
            
            if not model_metrics:
                logger.info("  No data available")
                continue
            
            # Display in table format
            logger.info(f"{'Model':<25} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'MAPE':>10}")
            logger.info("-" * 80)
            
            for model_name, metrics in model_metrics.items():
                display_model = model_name.replace('_', ' ').title()
                logger.info(f"{display_model:<25} "
                          f"{metrics['r2']:>10.4f} "
                          f"{metrics['rmse']:>12.2f} "
                          f"{metrics['mae']:>12.2f} "
                          f"{metrics['mape']:>10.2f}%")
            
            # Check if trained model beats baselines
            if 'trained_model' in model_metrics:
                trained_r2 = model_metrics['trained_model']['r2']
                best_baseline_r2 = max(
                    [m['r2'] for k, m in model_metrics.items() if k != 'trained_model'],
                    default=0
                )
                
                improvement = trained_r2 - best_baseline_r2
                
                if improvement > 0.2:
                    logger.info(f"\n✓ Trained model beats baselines by {improvement:.4f} R² points")
                elif improvement > 0:
                    logger.info(f"\n⚠️  Trained model marginally better ({improvement:.4f} R² improvement)")
                else:
                    logger.warning(f"\n❌ Trained model WORSE than baselines ({improvement:.4f} R² difference)")
        
        # Overall assessment
        logger.info("\n" + "="*80)
        logger.info("OVERALL ASSESSMENT")
        logger.info("="*80 + "\n")
        
        # Calculate average R² for trained model vs best baseline
        trained_r2_values = [
            results['trained_model'][target]['r2']
            for target in results['trained_model'].keys()
        ]
        avg_trained_r2 = np.mean(trained_r2_values)
        
        logger.info(f"Trained Model Average R²: {avg_trained_r2:.4f}")
        
        if avg_trained_r2 > 0.99:
            logger.warning("\n⚠️  CRITICAL: R² > 0.99 is extremely rare in forecasting")
            logger.warning("This strongly suggests data leakage!")
            assessment = "LIKELY DATA LEAKAGE"
        elif avg_trained_r2 > 0.90:
            logger.info("\n⚠️  R² > 0.90: Very high performance")
            logger.info("Verify this is not due to data leakage")
            assessment = "VERIFY NO LEAKAGE"
        elif avg_trained_r2 > 0.70:
            logger.info("\n✓ R² > 0.70: Good forecasting performance")
            assessment = "ACCEPTABLE"
        else:
            logger.info("\n✓ R² < 0.70: Typical forecasting performance")
            assessment = "NORMAL"
        
        return {
            'results': results,
            'avg_trained_r2': avg_trained_r2,
            'assessment': assessment
        }
    
    def generate_report(self, analysis, output_path='reports/audit/baseline_comparison.txt'):
        """Generate detailed report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = analysis['results']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Models Compared:\n")
            f.write("  1. Baseline: Last Value (Persistence)\n")
            f.write("  2. Baseline: Training Mean\n")
            f.write("  3. Baseline: Linear Trend\n")
            f.write("  4. Trained Model: Ridge Regression\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESULTS BY TARGET\n")
            f.write("="*80 + "\n\n")
            
            target_names = {
                'revenue_month_1': 'Revenue Month 1',
                'revenue_month_2': 'Revenue Month 2',
                'revenue_month_3': 'Revenue Month 3',
                'member_count_month_3': 'Members Month 3',
                'retention_rate_month_3': 'Retention Month 3'
            }
            
            for target, display_name in target_names.items():
                f.write(f"{display_name}:\n")
                f.write("-" * 80 + "\n")
                
                # Get metrics
                model_metrics = {}
                for model_name in results.keys():
                    if target in results[model_name]:
                        model_metrics[model_name] = results[model_name][target]
                
                if not model_metrics:
                    f.write("  No data available\n\n")
                    continue
                
                f.write(f"{'Model':<25} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'MAPE':>10}\n")
                f.write("-" * 80 + "\n")
                
                for model_name, metrics in model_metrics.items():
                    display_model = model_name.replace('_', ' ').title()
                    f.write(f"{display_model:<25} "
                           f"{metrics['r2']:>10.4f} "
                           f"{metrics['rmse']:>12.2f} "
                           f"{metrics['mae']:>12.2f} "
                           f"{metrics['mape']:>10.2f}%\n")
                
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("ASSESSMENT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Trained Model Average R²: {analysis['avg_trained_r2']:.4f}\n")
            f.write(f"Status: {analysis['assessment']}\n\n")
            
            if analysis['assessment'] == "LIKELY DATA LEAKAGE":
                f.write("⚠️  CRITICAL WARNING:\n")
                f.write("The model performance is suspiciously high (R² > 0.99)\n")
                f.write("This is extremely rare in real-world forecasting and strongly suggests data leakage.\n\n")
                f.write("Recommended Actions:\n")
                f.write("  1. DO NOT deploy to production\n")
                f.write("  2. Review feature engineering code for temporal leakage\n")
                f.write("  3. Run walk-forward validation\n")
                f.write("  4. Check if target information is leaking into features\n")
            elif analysis['assessment'] == "VERIFY NO LEAKAGE":
                f.write("⚠️  WARNING:\n")
                f.write("The model performance is very high (R² > 0.90)\n")
                f.write("Verify that this is not due to data leakage before deployment.\n")
            else:
                f.write("✓ Model performance is within expected range for forecasting.\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"\nReport saved to: {output_path}")
        
        # Save JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"JSON results saved to: {json_path}")


def main():
    print("\n" + "="*80)
    print("BASELINE MODEL COMPARISON")
    print("="*80 + "\n")
    
    comparator = BaselineComparison()
    
    # Run comparisons
    print("Comparing trained model against baselines...\n")
    results = comparator.compare_all()
    
    # Analyze
    analysis = comparator.analyze_results(results)
    
    # Generate report
    comparator.generate_report(analysis)
    
    print("\n" + "="*80)
    print(f"ASSESSMENT: {analysis['assessment']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

