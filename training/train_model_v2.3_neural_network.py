"""
Model Training v2.3.0 - Neural Network (Multi-Layer Perceptron)
Daily Data with Multi-Horizon Predictions

Architecture:
- Input Layer: 79 features
- Hidden Layer 1: 128 neurons, ReLU activation
- Hidden Layer 2: 64 neurons, ReLU activation
- Hidden Layer 3: 32 neurons, ReLU activation
- Hidden Layer 4: 16 neurons, ReLU activation
- Output Layer: 14 targets (4 daily + 4 weekly + 6 monthly)

Uses scikit-learn MLPRegressor with early stopping
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import logging
from datetime import datetime
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralNetworkTrainer:
    """Neural Network trainer for daily multi-horizon predictions"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.target_names = [
            # Daily targets (4)
            'Revenue Day 1', 'Revenue Day 3', 'Revenue Day 7', 'Attendance Day 7',
            # Weekly targets (4)
            'Revenue Week 1', 'Revenue Week 2', 'Revenue Week 4', 'Attendance Week 1',
            # Monthly targets (6)
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 1', 'Members Month 3', 'Retention Month 3'
        ]
        self.training_time = 0

    def load_and_prepare_data(self):
        """Load daily engineered data"""
        data_path = 'data/processed/multi_studio_daily_data_engineered.csv'
        logger.info(f"Loading data from {data_path}...")

        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} samples from {df['studio_id'].nunique()} studios")
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            logger.error("Please run: python src/features/run_feature_engineering_daily.py")
            raise

        # Define target columns
        target_cols = [
            # Daily targets
            'revenue_day_1', 'revenue_day_3', 'revenue_day_7', 'attendance_day_7',
            # Weekly targets
            'revenue_week_1', 'revenue_week_2', 'revenue_week_4', 'attendance_week_1',
            # Monthly targets
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_1', 'member_count_month_3', 'retention_rate_month_3'
        ]

        # Exclude non-feature columns
        exclude_cols = target_cols + ['studio_id', 'date', 'split',
                                      'studio_location', 'studio_size_tier', 'studio_price_tier']

        self.all_features = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Total available features: {len(self.all_features)}")

        # Split data (temporal)
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'validation'].copy()
        test_df = df[df['split'] == 'test'].copy()

        # Combine train and val for cross-validation
        train_val_df = pd.concat([train_df, val_df])

        # Extract features and targets
        X_train_val = train_val_df[self.all_features].values
        y_train_val = train_val_df[target_cols].values
        groups_train_val = train_val_df['studio_id'].values

        X_test = test_df[self.all_features].values
        y_test = test_df[target_cols].values
        groups_test = test_df['studio_id'].values

        logger.info(f"Train+Val: {len(X_train_val)} samples from {len(np.unique(groups_train_val))} studios")
        logger.info(f"Test: {len(X_test)} samples from {len(np.unique(groups_test))} studios")
        logger.info(f"Features: {len(self.all_features)}")
        logger.info(f"Targets: {len(target_cols)}")

        return X_train_val, y_train_val, groups_train_val, X_test, y_test, groups_test

    def scale_features(self, X_train, X_test):
        """Scale features"""
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train):
        """Train Neural Network (MLP)"""
        logger.info("Training Neural Network (MLPRegressor)...")
        logger.info("Architecture: 79 -> 128 -> 64 -> 32 -> 16 -> 14 outputs")

        start_time = time.time()

        # Create MLPRegressor with specified architecture
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4,
            verbose=True,
            random_state=42
        )

        logger.info(f"Training parameters:")
        logger.info(f"  Hidden layers: (128, 64, 32, 16)")
        logger.info(f"  Activation: relu")
        logger.info(f"  Optimizer: adam")
        logger.info(f"  Learning rate: 0.001 (adaptive)")
        logger.info(f"  Batch size: 64")
        logger.info(f"  Max iterations: 500")
        logger.info(f"  Early stopping: True (patience=20)")

        # Wrap in MultiOutputRegressor for multi-target regression
        self.model = MultiOutputRegressor(mlp)
        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")

    def cross_validate(self, X, y, groups, cv=5):
        """Perform grouped cross-validation (respecting studio boundaries)"""
        logger.info(f"Performing {cv}-fold grouped cross-validation...")

        # Use GroupKFold to ensure studios don't appear in both train and val
        gkf = GroupKFold(n_splits=cv)

        r2_scores = []
        rmse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"  Fold {fold_idx + 1}/{cv}...")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Create and train model for this fold
            mlp_fold = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32, 16),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=64,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=300,  # Reduced for CV
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                tol=1e-4,
                verbose=False,  # Quiet for CV
                random_state=42 + fold_idx
            )

            model_fold = MultiOutputRegressor(mlp_fold)
            model_fold.fit(X_train_fold, y_train_fold)

            # Predict and score
            y_pred = model_fold.predict(X_val_fold)
            r2 = r2_score(y_val_fold.flatten(), y_pred.flatten())
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))

            r2_scores.append(r2)
            rmse_scores.append(rmse)

            logger.info(f"    R² = {r2:.4f}, RMSE = {rmse:.2f}")

        cv_results = {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores
        }

        logger.info(f"CV Results:")
        logger.info(f"  R² = {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
        logger.info(f"  RMSE = {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")

        return cv_results

    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")

        # Measure inference time
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(f"Inference time: {inference_time:.2f}ms for {len(X_test)} samples")
        logger.info(f"Average: {inference_time/len(X_test):.4f}ms per sample")

        metrics = {}
        for i, target_name in enumerate(self.target_names):
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + 1e-10))) * 100

            metrics[target_name] = {
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2)
            }

        # Overall metrics (across all targets)
        overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
        overall_mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())

        test_results = {
            'metrics_by_target': metrics,
            'overall_rmse': round(overall_rmse, 2),
            'overall_r2': round(overall_r2, 4),
            'overall_mae': round(overall_mae, 2),
            'inference_time_ms': round(inference_time, 2),
            'avg_inference_time_ms': round(inference_time/len(X_test), 4)
        }

        logger.info(f"Overall Test Performance:")
        logger.info(f"  R² = {overall_r2:.4f}")
        logger.info(f"  RMSE = {overall_rmse:.2f}")
        logger.info(f"  MAE = {overall_mae:.2f}")

        # Log per-target performance
        logger.info(f"\nPer-Target Performance:")
        for target_name, target_metrics in metrics.items():
            logger.info(f"  {target_name}: R²={target_metrics['R2']:.4f}, RMSE={target_metrics['RMSE']:.2f}")

        return test_results

    def calculate_business_metrics(self, X_test, y_test):
        """Calculate business-focused metrics"""
        logger.info("Calculating business metrics...")

        y_pred = self.model.predict(X_test)

        # Focus on revenue predictions (indices 0-6: daily, weekly, monthly)
        revenue_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10]  # All revenue targets

        y_test_revenue = y_test[:, revenue_indices]
        y_pred_revenue = y_pred[:, revenue_indices]

        # Calculate percentage errors
        pct_errors = np.abs((y_pred_revenue - y_test_revenue) / (y_test_revenue + 1e-10)) * 100

        within_5_pct = (pct_errors <= 5).sum() / pct_errors.size * 100
        within_10_pct = (pct_errors <= 10).sum() / pct_errors.size * 100
        within_15_pct = (pct_errors <= 15).sum() / pct_errors.size * 100

        # Directional accuracy (for month-over-month)
        # Compare day 1 vs day 7 predictions
        direction_actual = np.sign(y_test[:, 2] - y_test[:, 0])  # Day 7 - Day 1
        direction_pred = np.sign(y_pred[:, 2] - y_pred[:, 0])
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100

        # Business impact score (weighted by tolerance)
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

    def save_artifacts(self, version='2.3.0'):
        """Save model artifacts"""
        logger.info(f"Saving model v{version}_neural_network...")

        output_dir = Path('data/models/v2.3')
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / 'neural_network_model_v2.3.0.pkl')
        joblib.dump(self.scaler, output_dir / 'neural_network_scaler_v2.3.0.pkl')
        joblib.dump(self.all_features, output_dir / 'neural_network_features_v2.3.0.pkl')

        metadata = {
            'version': '2.3.0',
            'model_type': 'neural_network',
            'architecture': '79 -> 128 -> 64 -> 32 -> 16 -> 14',
            'algorithm': 'MLPRegressor',
            'n_features': len(self.all_features),
            'feature_names': self.all_features,
            'n_targets': len(self.target_names),
            'target_names': self.target_names,
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': round(self.training_time, 2),
            'data_type': 'daily_multi_horizon',
            'phase': 'Phase 4 - Daily Multi-Horizon Data'
        }

        with open(output_dir / 'neural_network_metadata_v2.3.0.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Models saved to {output_dir}")


def main():
    print("\n" + "="*80)
    print("MODEL TRAINING v2.3.0 - NEURAL NETWORK (DAILY DATA)")
    print("="*80 + "\n")

    # Initialize trainer
    print("Step 1: Initializing Neural Network trainer...")
    trainer = NeuralNetworkTrainer()

    # Load data
    print("\nStep 2: Loading daily multi-horizon data...")
    X_train_val, y_train_val, groups_train_val, X_test, y_test, groups_test = trainer.load_and_prepare_data()

    # Scale features
    print("\nStep 3: Scaling features...")
    X_train_val_scaled, X_test_scaled = trainer.scale_features(X_train_val, X_test)

    # Train model
    print("\nStep 4: Training Neural Network...")
    print("-" * 80)
    trainer.train_model(X_train_val_scaled, y_train_val)
    print("-" * 80)

    # Cross-validation
    print("\nStep 5: Performing grouped cross-validation...")
    print("-" * 80)
    cv_results = trainer.cross_validate(X_train_val_scaled, y_train_val, groups_train_val, cv=5)
    print("-" * 80)

    # Test evaluation
    print("\nStep 6: Evaluating on test set...")
    print("-" * 80)
    test_results = trainer.evaluate_model(X_test_scaled, y_test)
    print("-" * 80)

    # Business metrics
    print("\nStep 7: Calculating business metrics...")
    print("-" * 80)
    business_metrics = trainer.calculate_business_metrics(X_test_scaled, y_test)
    print("-" * 80)

    # Save results
    print("\nStep 8: Saving results...")
    results = {
        'model_type': 'neural_network',
        'cv_results': cv_results,
        'test_results': test_results,
        'business_metrics': business_metrics,
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 4 - Daily Multi-Horizon Data',
        'training_samples': len(X_train_val),
        'test_samples': len(X_test),
        'n_studios_train': len(np.unique(groups_train_val)),
        'n_studios_test': len(np.unique(groups_test)),
        'n_features': len(trainer.all_features),
        'n_targets': len(trainer.target_names),
        'training_time_seconds': round(trainer.training_time, 2)
    }

    output_path = Path('reports/audit/model_results_v2.3.0_neural_network.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save artifacts
    print("\nStep 9: Saving model artifacts...")
    trainer.save_artifacts(version='2.3.0')

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - v2.3.0 NEURAL NETWORK MODEL")
    print("="*80)

    print(f"\nModel: Neural Network (MLPRegressor)")
    print(f"Architecture: 79 -> 128 -> 64 -> 32 -> 16 -> 14 outputs")
    print(f"Test R²: {test_results['overall_r2']:.4f}")
    print(f"Test RMSE: {test_results['overall_rmse']:.2f}")
    print(f"Training time: {trainer.training_time:.2f} seconds")
    print(f"Inference time: {test_results['avg_inference_time_ms']:.4f}ms per sample")

    print(f"\nBusiness Metrics:")
    print(f"  Predictions within 5%: {business_metrics['predictions_within_5pct']:.1f}%")
    print(f"  Predictions within 10%: {business_metrics['predictions_within_10pct']:.1f}%")
    print(f"  Business impact score: {business_metrics['business_impact_score']:.1f}/100")

    print(f"\nTraining samples: {len(X_train_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Studios: {len(np.unique(groups_train_val))}")
    print(f"Features: {len(trainer.all_features)}")
    print(f"Targets: {len(trainer.target_names)}")

    print("\nNext Steps:")
    print("  1. Train XGBoost: python training/train_model_v2.3_xgboost.py")
    print("  2. Train LightGBM: python training/train_model_v2.3_lightgbm.py")
    print("  3. Train Random Forest: python training/train_model_v2.3_random_forest.py")
    print("  4. Compare models: python training/compare_all_models_v2.3.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
