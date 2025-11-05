"""
Model Evaluation Script with Comprehensive Metrics

Performs detailed evaluation of trained models with statistical metrics,
feature importance analysis, and confidence intervals for audit purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
from datetime import datetime
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')


class ModelEvaluator:
    """Comprehensive model evaluation with audit trail"""

    def __init__(self, model_dir: Path, version: str):
        self.model_dir = Path(model_dir)
        self.version = version
        self.target_names = [
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 3', 'Retention Month 3'
        ]
        self.load_model()

    def load_model(self):
        """Load trained model artifacts"""
        logger.info(f"Loading model version {self.version}")

        self.models = joblib.load(
            self.model_dir / f'ensemble_models_v{self.version}.pkl'
        )
        self.scaler = joblib.load(
            self.model_dir / f'scaler_v{self.version}.pkl'
        )
        self.weights = joblib.load(
            self.model_dir / f'weights_v{self.version}.pkl'
        )
        self.feature_names = joblib.load(
            self.model_dir / f'features_v{self.version}.pkl'
        )

        logger.info(f"Loaded {len(self.feature_names)} features")
        logger.info(f"Ensemble weights: {self.weights}")

    def predict_ensemble(self, X):
        """Make ensemble prediction"""
        predictions = []

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])

            weighted_pred = pred * self.weights[model_name]
            predictions.append(weighted_pred)

        return np.sum(predictions, axis=0)

    def predict_individual_models(self, X):
        """Get predictions from individual models"""
        predictions = {}

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])
            predictions[model_name] = pred

        return predictions

    def evaluate(self, X, y, dataset_name="Test") -> Tuple[Dict, np.ndarray]:
        """Evaluate model performance with comprehensive metrics"""
        logger.info(f"Evaluating on {dataset_name} set")

        # Make ensemble predictions
        y_pred = self.predict_ensemble(X)

        metrics = {}

        for i, name in enumerate(self.target_names):
            rmse = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y[:, i], y_pred[:, i])
            r2 = r2_score(y[:, i], y_pred[:, i])

            # MAPE
            mape = np.mean(np.abs((y[:, i] - y_pred[:, i]) / y[:, i])) * 100

            # Directional accuracy (for revenue targets only)
            if i < 3 and len(y) > 1:
                direction_actual = np.sign(np.diff(y[:, i]))
                direction_pred = np.sign(np.diff(y_pred[:, i]))
                dir_accuracy = np.mean(direction_actual == direction_pred) * 100 if len(direction_actual) > 0 else None
            else:
                dir_accuracy = None

            # Mean prediction error
            mean_error = np.mean(y_pred[:, i] - y[:, i])

            # Std of errors
            std_error = np.std(y_pred[:, i] - y[:, i])

            metrics[name] = {
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2),
                'Directional_Accuracy': round(dir_accuracy, 2) if dir_accuracy is not None else None,
                'Mean_Error': round(mean_error, 2),
                'Std_Error': round(std_error, 2)
            }

            logger.info(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, "
                       f"R2={r2:.4f}, MAPE={mape:.2f}%")

        return metrics, y_pred

    def evaluate_individual_models(self, X, y, dataset_name="Test"):
        """Evaluate each individual model separately"""
        logger.info(f"Evaluating individual models on {dataset_name} set")

        predictions = self.predict_individual_models(X)
        model_metrics = {}

        for model_name, y_pred in predictions.items():
            rmse = np.sqrt(np.mean((y_pred - y) ** 2))
            mae = np.mean(np.abs(y_pred - y))
            r2 = r2_score(y.flatten(), y_pred.flatten())

            model_metrics[model_name] = {
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4)
            }

            logger.info(f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        return model_metrics

    def plot_predictions(self, y_true, y_pred, dataset_name="Test", save_dir=None):
        """Plot actual vs predicted for audit visualization"""
        logger.info(f"Creating prediction plots for {dataset_name} set")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, name in enumerate(self.target_names):
            ax = axes[i]

            # Scatter plot
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            # Perfect prediction line
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            # Calculate R2 for annotation
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))

            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.2f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Actual', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_title(f'{name} - {dataset_name} Set', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / f'evaluation_scatter_{dataset_name.lower()}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")

        plt.close()

    def plot_residuals(self, y_true, y_pred, dataset_name="Test", save_dir=None):
        """Plot residual distributions for error analysis"""
        logger.info(f"Creating residual plots for {dataset_name} set")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, name in enumerate(self.target_names):
            ax = axes[i]
            residuals = y_true[:, i] - y_pred[:, i]

            ax.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.axvline(np.mean(residuals), color='green', linestyle='-', linewidth=2, label=f'Mean = {np.mean(residuals):.2f}')

            ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{name} - Residual Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / f'residual_distribution_{dataset_name.lower()}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")

        plt.close()

    def analyze_feature_importance(self, save_dir=None):
        """Analyze feature importance from XGBoost models"""
        logger.info("Analyzing feature importance")

        # Get feature importance from XGBoost (first model - revenue month 1)
        xgb_model = self.models['xgboost'][0]
        importance = xgb_model.feature_importances_

        # Sort features by importance
        indices = np.argsort(importance)[::-1][:20]  # Top 20

        # Create dataframe for audit
        importance_df = pd.DataFrame({
            'Feature': [self.feature_names[i] for i in indices],
            'Importance': [round(importance[i], 4) for i in indices],
            'Rank': range(1, 21)
        })

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))

        bars = ax.barh(range(20), importance[indices], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(20))
        ax.set_yticklabels([self.feature_names[i] for i in indices], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Important Features (XGBoost)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance[indices])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")

        plt.close()

        return importance_df

    def calculate_confidence_intervals(self, X, percentile=95):
        """Calculate prediction confidence intervals using ensemble variance"""
        logger.info(f"Calculating {percentile}% confidence intervals")

        # Get predictions from each model
        predictions = []

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])
            predictions.append(pred)

        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # (n_models, n_samples, n_targets)

        # Calculate mean and std
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)

        # Calculate confidence intervals
        z_score = {90: 1.645, 95: 1.96, 99: 2.576}[percentile]
        lower = pred_mean - z_score * pred_std
        upper = pred_mean + z_score * pred_std

        logger.info(f"Confidence intervals calculated with z={z_score}")

        return pred_mean, lower, upper, pred_std

    def plot_confidence_intervals(self, y_true, pred_mean, lower, upper, dataset_name="Test", save_dir=None):
        """Plot predictions with confidence intervals"""
        logger.info(f"Creating confidence interval plots for {dataset_name} set")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()

        x_indices = np.arange(len(y_true))

        for i, name in enumerate(self.target_names):
            ax = axes[i]

            # Plot actual values
            ax.plot(x_indices, y_true[:, i], 'o-', label='Actual', linewidth=2, markersize=6, color='black')

            # Plot predicted values
            ax.plot(x_indices, pred_mean[:, i], 's-', label='Predicted', linewidth=2, markersize=5, color='blue', alpha=0.7)

            # Plot confidence intervals
            ax.fill_between(x_indices, lower[:, i], upper[:, i], alpha=0.3, color='blue', label='95% CI')

            ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
            ax.set_ylabel(name.split()[0], fontsize=11, fontweight='bold')
            ax.set_title(f'{name} with 95% Confidence Intervals', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / f'confidence_intervals_{dataset_name.lower()}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")

        plt.close()


def save_predictions_to_csv(y_true, y_pred, dataset_name, save_path):
    """Save predictions and residuals to CSV for audit"""
    target_cols = [
        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
        'member_count_month_3', 'retention_rate_month_3'
    ]

    data = {}

    # Add actual values
    for i, col in enumerate(target_cols):
        data[f'actual_{col}'] = y_true[:, i]
        data[f'predicted_{col}'] = y_pred[:, i]
        data[f'residual_{col}'] = y_true[:, i] - y_pred[:, i]
        data[f'abs_error_{col}'] = np.abs(y_true[:, i] - y_pred[:, i])
        data[f'pct_error_{col}'] = np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i]) * 100

    df = pd.DataFrame(data)
    df['dataset'] = dataset_name
    df['sample_index'] = range(len(df))

    df.to_csv(save_path, index=False)
    logger.info(f"Saved predictions to {save_path}")

    return df


# Main evaluation execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Studio Revenue Simulator - Model Validation")
    print("="*70 + "\n")

    # Create output directories
    audit_dir = Path('reports/audit')
    figures_dir = Path('reports/figures')
    audit_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Step 1: Loading engineered data...")
    df = pd.read_csv('data/processed/studio_data_engineered.csv')
    print(f"Loaded {len(df)} rows")

    # Define columns
    target_cols = [
        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
        'member_count_month_3', 'retention_rate_month_3'
    ]

    # Load feature names
    feature_names = joblib.load('data/models/features_v1.0.0.pkl')
    scaler = joblib.load('data/models/scaler_v1.0.0.pkl')

    # Initialize evaluator
    print("\nStep 2: Loading trained model...")
    evaluator = ModelEvaluator(model_dir='data/models', version='1.0.0')

    # Prepare datasets
    print("\nStep 3: Preparing datasets...")
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']

    datasets = {
        'train': (train_df[feature_names].values, train_df[target_cols].values),
        'validation': (val_df[feature_names].values, val_df[target_cols].values),
        'test': (test_df[feature_names].values, test_df[target_cols].values)
    }

    # Scale features
    for name in datasets:
        X, y = datasets[name]
        X_scaled = scaler.transform(X)
        datasets[name] = (X_scaled, y)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Evaluate on all datasets
    print("\nStep 4: Evaluating model performance...")
    all_metrics = {}
    all_predictions = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n--- {dataset_name.upper()} SET ---")
        metrics, y_pred = evaluator.evaluate(X, y, dataset_name=dataset_name)
        all_metrics[dataset_name] = metrics
        all_predictions[dataset_name] = (y, y_pred)

        # Save predictions to CSV
        save_predictions_to_csv(
            y, y_pred, dataset_name,
            audit_dir / f'predictions_{dataset_name}_v1.0.0.csv'
        )

    # Evaluate individual models
    print("\nStep 5: Evaluating individual models on test set...")
    X_test, y_test = datasets['test']
    model_comparison = evaluator.evaluate_individual_models(X_test, y_test, dataset_name="Test")

    # Feature importance
    print("\nStep 6: Analyzing feature importance...")
    importance_df = evaluator.analyze_feature_importance(save_dir=figures_dir)
    importance_df.to_csv(audit_dir / 'feature_importance_v1.0.0.csv', index=False)

    # Confidence intervals
    print("\nStep 7: Calculating confidence intervals...")
    pred_mean, lower, upper, pred_std = evaluator.calculate_confidence_intervals(X_test, percentile=95)

    # Calculate coverage rate
    y_test_coverage = y_test
    coverage = np.mean((y_test_coverage >= lower) & (y_test_coverage <= upper)) * 100
    print(f"95% Confidence Interval Coverage: {coverage:.1f}%")

    # Generate visualizations
    print("\nStep 8: Generating visualizations...")
    for dataset_name, (y, y_pred) in all_predictions.items():
        evaluator.plot_predictions(y, y_pred, dataset_name=dataset_name, save_dir=figures_dir)
        evaluator.plot_residuals(y, y_pred, dataset_name=dataset_name, save_dir=figures_dir)

    evaluator.plot_confidence_intervals(y_test, pred_mean, lower, upper, dataset_name="Test", save_dir=figures_dir)

    # Save complete metrics to JSON
    print("\nStep 9: Saving audit metrics...")
    audit_data = {
        'model_version': '1.0.0',
        'evaluation_timestamp': datetime.now().isoformat(),
        'metrics_by_dataset': all_metrics,
        'model_comparison': model_comparison,
        'ensemble_weights': evaluator.weights,
        'confidence_interval_coverage': round(coverage, 2),
        'feature_count': len(feature_names),
        'top_10_features': importance_df.head(10).to_dict('records')
    }

    with open(audit_dir / 'metrics_v1.0.0.json', 'w') as f:
        json.dump(audit_data, f, indent=2)

    print(f"[OK] Metrics saved to {audit_dir / 'metrics_v1.0.0.json'}")

    print("\n" + "="*70)
    print("MODEL VALIDATION COMPLETE")
    print("="*70)
    print(f"\nAudit Files Saved:")
    print(f"  - {audit_dir / 'metrics_v1.0.0.json'}")
    print(f"  - {audit_dir / 'predictions_train_v1.0.0.csv'}")
    print(f"  - {audit_dir / 'predictions_validation_v1.0.0.csv'}")
    print(f"  - {audit_dir / 'predictions_test_v1.0.0.csv'}")
    print(f"  - {audit_dir / 'feature_importance_v1.0.0.csv'}")
    print(f"\nVisualization Files Saved:")
    print(f"  - {figures_dir / 'evaluation_scatter_*.png'} (3 files)")
    print(f"  - {figures_dir / 'residual_distribution_*.png'} (3 files)")
    print(f"  - {figures_dir / 'confidence_intervals_test.png'}")
    print(f"  - {figures_dir / 'feature_importance.png'}")
    print("\n" + "="*70 + "\n")
