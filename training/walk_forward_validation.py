"""
Walk-Forward Validation Script

Tests model on truly unseen future data to detect data leakage.
This is the gold standard for time series validation.
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


class WalkForwardValidator:
    """Perform walk-forward validation on time series data"""
    
    def __init__(self, data_path='data/processed/multi_studio_data_engineered.csv',
                 model_path='data/models/best_model_v2.2.0.pkl',
                 scaler_path='data/models/scaler_v2.2.0.pkl'):
        self.data_path = data_path
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def load_and_prepare_data(self):
        """Load and sort data chronologically"""
        logger.info("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Convert to datetime and sort
        df['month_year'] = pd.to_datetime(df['month_year'])
        df = df.sort_values(['studio_id', 'month_year']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Date range: {df['month_year'].min()} to {df['month_year'].max()}")
        
        return df
    
    def walk_forward_validate(self, df, train_months=48, test_months=3, step=3):
        """
        Perform walk-forward validation
        
        Args:
            df: DataFrame with features and targets
            train_months: Number of months to use for training
            test_months: Number of months to predict
            step: How many months to move forward each iteration
        """
        logger.info("Starting walk-forward validation...")
        
        # Get unique dates
        unique_dates = sorted(df['month_year'].unique())
        
        # Define features and targets
        target_cols = [
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_3', 'retention_rate_month_3'
        ]
        exclude_cols = target_cols + ['month_year', 'split', 'year_index', 'studio_id', 
                                       'studio_location', 'studio_size_tier', 'studio_price_tier']
        # Only select numeric columns as features
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        results = []
        fold = 0
        
        # Walk forward through time
        for i in range(0, len(unique_dates) - train_months - test_months, step):
            fold += 1
            
            # Define train and test periods
            train_start = unique_dates[i]
            train_end = unique_dates[i + train_months - 1]
            test_start = unique_dates[i + train_months]
            test_end = unique_dates[min(i + train_months + test_months - 1, len(unique_dates) - 1)]
            
            # Split data
            train_mask = (df['month_year'] >= train_start) & (df['month_year'] <= train_end)
            test_mask = (df['month_year'] >= test_start) & (df['month_year'] <= test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(test_df) == 0:
                continue
            
            logger.info(f"\nFold {fold}:")
            logger.info(f"  Train: {train_start.date()} to {train_end.date()} ({len(train_df)} samples)")
            logger.info(f"  Test:  {test_start.date()} to {test_end.date()} ({len(test_df)} samples)")
            
            # Prepare features
            X_train = train_df[feature_cols].values
            y_train = train_df[target_cols].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_cols].values
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            fold_results = {
                'fold': fold,
                'train_start': str(train_start.date()),
                'train_end': str(train_end.date()),
                'test_start': str(test_start.date()),
                'test_end': str(test_end.date()),
                'train_samples': len(train_df),
                'test_samples': len(test_df)
            }
            
            # Per-target metrics
            target_names = [
                'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
                'Members Month 3', 'Retention Month 3'
            ]
            
            for i, target_name in enumerate(target_names):
                rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                r2 = r2_score(y_test[:, i], y_pred[:, i])
                mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + 1e-10))) * 100
                
                fold_results[f'{target_name}_rmse'] = rmse
                fold_results[f'{target_name}_mae'] = mae
                fold_results[f'{target_name}_r2'] = r2
                fold_results[f'{target_name}_mape'] = mape
            
            # Overall metrics
            overall_rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
            overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
            
            fold_results['overall_rmse'] = overall_rmse
            fold_results['overall_r2'] = overall_r2
            
            logger.info(f"  Overall R²: {overall_r2:.4f}")
            logger.info(f"  Overall RMSE: {overall_rmse:.2f}")
            
            results.append(fold_results)
        
        return results
    
    def analyze_results(self, results):
        """Analyze walk-forward validation results"""
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD VALIDATION ANALYSIS")
        logger.info("="*80)
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Overall statistics
        logger.info(f"\nTotal Folds: {len(results)}")
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Mean R²: {results_df['overall_r2'].mean():.4f} (+/- {results_df['overall_r2'].std():.4f})")
        logger.info(f"  Mean RMSE: {results_df['overall_rmse'].mean():.2f} (+/- {results_df['overall_rmse'].std():.2f})")
        
        # Per-target analysis
        target_names = [
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 3', 'Retention Month 3'
        ]
        
        logger.info("\nPer-Target Performance:")
        for target in target_names:
            r2_col = f'{target}_r2'
            rmse_col = f'{target}_rmse'
            mape_col = f'{target}_mape'
            
            logger.info(f"\n{target}:")
            logger.info(f"  R²:   {results_df[r2_col].mean():.4f} (+/- {results_df[r2_col].std():.4f})")
            logger.info(f"  RMSE: {results_df[rmse_col].mean():.2f} (+/- {results_df[rmse_col].std():.2f})")
            logger.info(f"  MAPE: {results_df[mape_col].mean():.2f}% (+/- {results_df[mape_col].std():.2f}%)")
        
        # Check for degradation over time
        logger.info("\n" + "-"*80)
        logger.info("TEMPORAL STABILITY CHECK")
        logger.info("-"*80)
        
        first_half = results_df.iloc[:len(results_df)//2]
        second_half = results_df.iloc[len(results_df)//2:]
        
        r2_degradation = first_half['overall_r2'].mean() - second_half['overall_r2'].mean()
        rmse_degradation = second_half['overall_rmse'].mean() - first_half['overall_rmse'].mean()
        
        logger.info(f"\nFirst Half vs Second Half:")
        logger.info(f"  R² change: {r2_degradation:+.4f} {'(degrading)' if r2_degradation > 0.05 else '(stable)'}")
        logger.info(f"  RMSE change: {rmse_degradation:+.2f} {'(worse)' if rmse_degradation > 100 else '(stable)'}")
        
        # Leakage detection
        logger.info("\n" + "-"*80)
        logger.info("DATA LEAKAGE ASSESSMENT")
        logger.info("-"*80)
        
        avg_r2 = results_df['overall_r2'].mean()
        
        if avg_r2 > 0.99:
            logger.warning("⚠️  CRITICAL: R² > 0.99 suggests SEVERE data leakage")
            leakage_status = "SEVERE LEAKAGE LIKELY"
        elif avg_r2 > 0.95:
            logger.warning("⚠️  WARNING: R² > 0.95 suggests possible data leakage")
            leakage_status = "POSSIBLE LEAKAGE"
        elif avg_r2 > 0.80:
            logger.info("✓ R² > 0.80: Good performance, leakage unlikely")
            leakage_status = "UNLIKELY"
        else:
            logger.info("✓ R² < 0.80: Normal forecasting performance")
            leakage_status = "NONE DETECTED"
        
        return {
            'results_df': results_df,
            'avg_r2': avg_r2,
            'avg_rmse': results_df['overall_rmse'].mean(),
            'r2_std': results_df['overall_r2'].std(),
            'rmse_std': results_df['overall_rmse'].std(),
            'r2_degradation': r2_degradation,
            'rmse_degradation': rmse_degradation,
            'leakage_status': leakage_status
        }
    
    def generate_report(self, results, analysis, output_path='reports/audit/walk_forward_validation.txt'):
        """Generate detailed report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("WALK-FORWARD VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Total Folds: {len(results)}\n")
            f.write(f"Average R²: {analysis['avg_r2']:.4f} (+/- {analysis['r2_std']:.4f})\n")
            f.write(f"Average RMSE: {analysis['avg_rmse']:.2f} (+/- {analysis['rmse_std']:.2f})\n\n")
            
            f.write("="*80 + "\n")
            f.write("DATA LEAKAGE ASSESSMENT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Status: {analysis['leakage_status']}\n\n")
            
            if analysis['leakage_status'] in ['SEVERE LEAKAGE LIKELY', 'POSSIBLE LEAKAGE']:
                f.write("⚠️  WARNING: Model performance is suspiciously high!\n\n")
                f.write("Recommended Actions:\n")
                f.write("  1. Review feature engineering code for temporal leakage\n")
                f.write("  2. Verify rolling windows exclude current period\n")
                f.write("  3. Check that no target information leaks into features\n")
                f.write("  4. Test on truly unseen data from future periods\n\n")
            else:
                f.write("✓ Model performance is within expected range for forecasting\n\n")
            
            f.write("="*80 + "\n")
            f.write("DETAILED FOLD RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for result in results:
                f.write(f"Fold {result['fold']}:\n")
                f.write(f"  Train: {result['train_start']} to {result['train_end']}\n")
                f.write(f"  Test:  {result['test_start']} to {result['test_end']}\n")
                f.write(f"  R²: {result['overall_r2']:.4f}\n")
                f.write(f"  RMSE: {result['overall_rmse']:.2f}\n\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"\nReport saved to: {output_path}")
        
        # Also save JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'results': results,
                'analysis': {k: v for k, v in analysis.items() if k != 'results_df'}
            }, f, indent=2, default=str)
        
        logger.info(f"JSON results saved to: {json_path}")


def main():
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION")
    print("="*80 + "\n")
    
    validator = WalkForwardValidator()
    
    # Load data
    df = validator.load_and_prepare_data()
    
    # Perform walk-forward validation
    print("\nPerforming walk-forward validation...")
    print("This may take several minutes...\n")
    
    results = validator.walk_forward_validate(df, train_months=36, test_months=3, step=3)
    
    # Analyze results
    analysis = validator.analyze_results(results)
    
    # Generate report
    validator.generate_report(results, analysis)
    
    print("\n" + "="*80)
    print(f"FINAL ASSESSMENT: {analysis['leakage_status']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
