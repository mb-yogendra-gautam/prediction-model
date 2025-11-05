"""
Forward Prediction Model Training Script

Trains ensemble ML model for predicting revenue and business metrics
from operational levers using XGBoost, LightGBM, and Random Forest.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Dict
import yaml
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class ForwardPredictionModel:
    """Ensemble model for forward prediction (levers -> revenue)"""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.ensemble_weights = {}
        self.target_names = [
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 3', 'Retention Month 3'
        ]

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare train/val/test splits"""
        logger.info("Preparing data splits")

        # Define feature columns
        target_cols = self.config['data']['target_columns']

        exclude_cols = target_cols + ['month_year', 'split', 'year_index']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"Using {len(self.feature_names)} features")

        # Split data
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'validation']
        test_df = df[df['split'] == 'test']

        # Extract features and targets
        X_train = train_df[self.feature_names].values
        y_train = train_df[target_cols].values

        X_val = val_df[self.feature_names].values
        y_val = val_df[target_cols].values

        X_test = test_df[self.feature_names].values
        y_test = test_df[target_cols].values

        # Scale features
        logger.info("Scaling features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Train shape: X={X_train_scaled.shape}, y={y_train.shape}")
        logger.info(f"Val shape: X={X_val_scaled.shape}, y={y_val.shape}")
        logger.info(f"Test shape: X={X_test_scaled.shape}, y={y_test.shape}")

        return (
            (X_train_scaled, y_train),
            (X_val_scaled, y_val),
            (X_test_scaled, y_test)
        )

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model for each target"""
        logger.info("Training XGBoost models")

        xgb_config = self.config['models']['xgboost']
        models = []

        for i in range(y_train.shape[1]):
            logger.info(f"Training XGBoost for target {i+1}: {self.target_names[i]}")

            model = xgb.XGBRegressor(
                n_estimators=xgb_config['n_estimators'],
                max_depth=xgb_config['max_depth'],
                learning_rate=xgb_config['learning_rate'],
                subsample=xgb_config['subsample'],
                colsample_bytree=xgb_config['colsample_bytree'],
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                early_stopping_rounds=xgb_config['early_stopping_rounds'],
                verbose=False
            )

            models.append(model)

        self.models['xgboost'] = models
        logger.info("XGBoost training complete")

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model for each target"""
        logger.info("Training LightGBM models")

        lgb_config = self.config['models']['lightgbm']
        models = []

        for i in range(y_train.shape[1]):
            logger.info(f"Training LightGBM for target {i+1}: {self.target_names[i]}")

            model = lgb.LGBMRegressor(
                n_estimators=lgb_config['n_estimators'],
                max_depth=lgb_config['max_depth'],
                learning_rate=lgb_config['learning_rate'],
                subsample=lgb_config['subsample'],
                colsample_bytree=lgb_config['colsample_bytree'],
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                callbacks=[lgb.early_stopping(stopping_rounds=lgb_config['early_stopping_rounds'], verbose=False)]
            )

            models.append(model)

        self.models['lightgbm'] = models
        logger.info("LightGBM training complete")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest")

        rf_config = self.config['models']['random_forest']

        rf = RandomForestRegressor(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )

        # Use MultiOutputRegressor for multi-target
        model = MultiOutputRegressor(rf)
        model.fit(X_train, y_train)

        self.models['random_forest'] = model
        logger.info("Random Forest training complete")

    def calculate_ensemble_weights(self, X_val, y_val):
        """Calculate optimal weights based on validation performance"""
        logger.info("Calculating ensemble weights")

        val_scores = {}

        for model_name, models in self.models.items():
            # Make predictions
            if model_name == 'random_forest':
                y_pred = models.predict(X_val)
            else:
                # For XGBoost/LightGBM (list of models)
                y_pred = np.column_stack([m.predict(X_val) for m in models])

            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            val_scores[model_name] = rmse

            logger.info(f"{model_name} validation RMSE: {rmse:.2f}")

        # Calculate weights (inverse of RMSE)
        inverse_rmse = {name: 1.0 / rmse for name, rmse in val_scores.items()}
        total_inverse = sum(inverse_rmse.values())

        self.ensemble_weights = {
            name: inv / total_inverse
            for name, inv in inverse_rmse.items()
        }

        logger.info(f"Ensemble weights: {self.ensemble_weights}")

        return val_scores

    def predict_ensemble(self, X):
        """Make ensemble prediction"""
        predictions = []

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])

            weighted_pred = pred * self.ensemble_weights[model_name]
            predictions.append(weighted_pred)

        return np.sum(predictions, axis=0)

    def train(self, df: pd.DataFrame):
        """Train complete ensemble pipeline"""
        logger.info("Starting model training")

        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(df)

        # Train models
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train)

        # Calculate ensemble weights
        val_scores = self.calculate_ensemble_weights(X_val, y_val)

        logger.info("Model training complete")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), val_scores

    def save(self, version: str, output_dir: Path):
        """Save all model artifacts"""
        logger.info(f"Saving model version {version}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        joblib.dump(self.models, output_dir / f'ensemble_models_v{version}.pkl')
        joblib.dump(self.scaler, output_dir / f'scaler_v{version}.pkl')
        joblib.dump(self.ensemble_weights, output_dir / f'weights_v{version}.pkl')
        joblib.dump(self.feature_names, output_dir / f'features_v{version}.pkl')

        logger.info(f"Models saved to {output_dir}")

    def load(self, version: str, model_dir: Path):
        """Load model artifacts"""
        logger.info(f"Loading model version {version}")

        model_dir = Path(model_dir)

        self.models = joblib.load(model_dir / f'ensemble_models_v{version}.pkl')
        self.scaler = joblib.load(model_dir / f'scaler_v{version}.pkl')
        self.ensemble_weights = joblib.load(model_dir / f'weights_v{version}.pkl')
        self.feature_names = joblib.load(model_dir / f'features_v{version}.pkl')

        logger.info(f"Models loaded from {model_dir}")


def calculate_metrics(y_true, y_pred, target_names):
    """Calculate comprehensive metrics"""
    metrics = {}

    for i, name in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        # MAPE
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100

        metrics[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

    return metrics


def create_prediction_visualizations(model, X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    """Create comprehensive prediction visualizations"""
    logger.info("Creating prediction visualizations")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make predictions
    y_train_pred = model.predict_ensemble(X_train)
    y_val_pred = model.predict_ensemble(X_val)
    y_test_pred = model.predict_ensemble(X_test)

    # Combine all data for visualization
    y_true_all = np.vstack([y_train, y_val, y_test])
    y_pred_all = np.vstack([y_train_pred, y_val_pred, y_test_pred])

    # 1. Actual vs Predicted Scatter Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, name in enumerate(model.target_names):
        ax = axes[i]
        ax.scatter(y_true_all[:, i], y_pred_all[:, i], alpha=0.6, s=50)

        # Perfect prediction line
        min_val = min(y_true_all[:, i].min(), y_pred_all[:, i].min())
        max_val = max(y_true_all[:, i].max(), y_pred_all[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{name}\nActual vs Predicted', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'prediction_scatter.png'}")

    # 2. Revenue Predictions Comparison (Time Series)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for i in range(3):
        ax = axes[i]
        x_indices = np.arange(len(y_true_all))

        ax.plot(x_indices, y_true_all[:, i], 'o-', label='Actual', linewidth=2, markersize=4)
        ax.plot(x_indices, y_pred_all[:, i], 's-', label='Predicted', linewidth=2, markersize=4, alpha=0.7)

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title(f'{model.target_names[i]}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'revenue_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'revenue_predictions_comparison.png'}")

    # 3. Member and Retention Predictions
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, i in enumerate([3, 4]):
        ax = axes[idx]
        x_indices = np.arange(len(y_true_all))

        ax.plot(x_indices, y_true_all[:, i], 'o-', label='Actual', linewidth=2, markersize=4)
        ax.plot(x_indices, y_pred_all[:, i], 's-', label='Predicted', linewidth=2, markersize=4, alpha=0.7)

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel(model.target_names[i].split()[0], fontsize=12)
        ax.set_title(f'{model.target_names[i]}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'member_retention_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'member_retention_predictions.png'}")

    # 4. Residual Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, name in enumerate(model.target_names):
        ax = axes[i]
        residuals = y_true_all[:, i] - y_pred_all[:, i]

        ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')

        ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name}\nResidual Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'residual_analysis.png'}")

    # 5. Model Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate RMSE for each model type on test set
    model_rmses = {}
    for model_name, models in model.models.items():
        if model_name == 'random_forest':
            y_pred = models.predict(X_test)
        else:
            y_pred = np.column_stack([m.predict(X_test) for m in models])

        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        model_rmses[model_name] = rmse

    # Add ensemble
    ensemble_pred = model.predict_ensemble(X_test)
    model_rmses['ensemble'] = np.sqrt(np.mean((ensemble_pred - y_test) ** 2))

    models_list = list(model_rmses.keys())
    rmse_values = list(model_rmses.values())

    bars = ax.bar(models_list, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'model_performance_comparison.png'}")

    # 6. Training Summary Dashboard
    test_metrics = calculate_metrics(y_test, y_test_pred, model.target_names)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Metrics table
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.axis('tight')
    ax1.axis('off')

    table_data = []
    for target, metrics in test_metrics.items():
        table_data.append([
            target,
            f"{metrics['RMSE']:.2f}",
            f"{metrics['MAE']:.2f}",
            f"{metrics['R2']:.3f}",
            f"{metrics['MAPE']:.1f}%"
        ])

    table = ax1.table(cellText=table_data,
                     colLabels=['Target', 'RMSE', 'MAE', 'R²', 'MAPE'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax1.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold', pad=20)

    # Ensemble weights
    ax2 = fig.add_subplot(gs[0, 1])
    weights = list(model.ensemble_weights.values())
    labels = list(model.ensemble_weights.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    wedges, texts, autotexts = ax2.pie(weights, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title('Ensemble Weights', fontsize=12, fontweight='bold')

    # Data splits
    ax3 = fig.add_subplot(gs[1, 1])
    splits = ['Train', 'Val', 'Test']
    split_sizes = [len(X_train), len(X_val), len(X_test)]
    bars = ax3.bar(splits, split_sizes, color=['#2196F3', '#FF9800', '#4CAF50'])
    ax3.set_ylabel('Samples', fontsize=11)
    ax3.set_title('Dataset Splits', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, size in zip(bars, split_sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{size}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Model info
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    info_text = f"""
    Model Configuration:
    • Features: {len(model.feature_names)} engineered features
    • Targets: 5 multi-output predictions
    • Models: XGBoost ({model.config['models']['xgboost']['n_estimators']} trees), LightGBM ({model.config['models']['lightgbm']['n_estimators']} trees), Random Forest ({model.config['models']['random_forest']['n_estimators']} trees)
    • Ensemble: Weighted average based on validation RMSE
    • Version: {model.config['model']['version']}
    """

    ax4.text(0.5, 0.5, info_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(output_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'training_summary.png'}")

    logger.info("All visualizations created successfully")


# Main training script
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Studio Revenue Simulator - Model Training")
    print("="*70 + "\n")

    # Load config
    print("Step 1: Loading configuration...")
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load engineered data
    print("\nStep 2: Loading engineered data...")
    df = pd.read_csv('data/processed/studio_data_engineered.csv')
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Initialize MLflow (optional)
    mlflow_enabled = False
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        mlflow_enabled = True
        print("\nMLflow tracking enabled")
    except Exception as e:
        print(f"\nMLflow not available (continuing without tracking): {e}")

    # Train model
    print("\nStep 3: Training ensemble model...")
    print("-" * 70)

    model = ForwardPredictionModel(config)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), val_scores = model.train(df)

    print("\n" + "-" * 70)
    print("Training Results:")
    print(f"  Ensemble Weights: {model.ensemble_weights}")

    # Calculate test metrics
    print("\nStep 4: Evaluating on test set...")
    y_test_pred = model.predict_ensemble(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, model.target_names)

    print("\nTest Set Performance:")
    for target, metrics in test_metrics.items():
        print(f"\n  {target}:")
        print(f"    RMSE: {metrics['RMSE']:.2f}")
        print(f"    MAE:  {metrics['MAE']:.2f}")
        print(f"    R2:   {metrics['R2']:.3f}")
        print(f"    MAPE: {metrics['MAPE']:.1f}%")

    # Save model
    print("\nStep 5: Saving model artifacts...")
    model.save(
        version=config['model']['version'],
        output_dir=Path(config['model']['output_dir'])
    )
    print(f"[OK] Models saved to {config['model']['output_dir']}")

    # Create visualizations
    print("\nStep 6: Creating prediction visualizations...")
    create_prediction_visualizations(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        output_dir=Path('reports/figures')
    )
    print("[OK] Visualizations saved to reports/figures/")

    # MLflow logging
    if mlflow_enabled:
        print("\nStep 7: Logging to MLflow...")
        try:
            with mlflow.start_run(run_name=f"training_v{config['model']['version']}"):
                # Log params
                mlflow.log_params(config['models']['xgboost'])

                # Log validation metrics
                for model_name, rmse in val_scores.items():
                    mlflow.log_metric(f"val_rmse_{model_name}", rmse)

                # Log test metrics
                for target, metrics in test_metrics.items():
                    mlflow.log_metric(f"test_rmse_{target.replace(' ', '_')}", metrics['RMSE'])
                    mlflow.log_metric(f"test_r2_{target.replace(' ', '_')}", metrics['R2'])

                # Log artifacts
                mlflow.log_artifacts('reports/figures')

                print("[OK] MLflow logging complete")
        except Exception as e:
            print(f"[WARNING] MLflow logging failed: {e}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel Artifacts:")
    print(f"  - data/models/ensemble_models_v{config['model']['version']}.pkl")
    print(f"  - data/models/scaler_v{config['model']['version']}.pkl")
    print(f"  - data/models/weights_v{config['model']['version']}.pkl")
    print(f"  - data/models/features_v{config['model']['version']}.pkl")
    print(f"\nVisualizations:")
    print(f"  - reports/figures/prediction_scatter.png")
    print(f"  - reports/figures/revenue_predictions_comparison.png")
    print(f"  - reports/figures/member_retention_predictions.png")
    print(f"  - reports/figures/residual_analysis.png")
    print(f"  - reports/figures/model_performance_comparison.png")
    print(f"  - reports/figures/training_summary.png")
    print("\n" + "="*70 + "\n")
