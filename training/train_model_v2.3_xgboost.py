"""
Model Training v2.3.0 - XGBoost
Daily Data with Multi-Horizon Predictions

XGBoost optimized for tabular data with:
- 500 trees
- Max depth: 6
- Learning rate: 0.03
- Regularization: alpha=0.5, lambda=1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
import logging
from datetime import datetime
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """XGBoost trainer for daily multi-horizon predictions"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.target_names = [
            'Revenue Day 1', 'Revenue Day 3', 'Revenue Day 7', 'Attendance Day 7',
            'Revenue Week 1', 'Revenue Week 2', 'Revenue Week 4', 'Attendance Week 1',
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 1', 'Members Month 3', 'Retention Month 3'
        ]
        self.training_time = 0

    def load_and_prepare_data(self):
        """Load daily engineered data"""
        data_path = 'data/processed/multi_studio_daily_data_engineered.csv'
        logger.info(f"Loading data from {data_path}...")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {df['studio_id'].nunique()} studios")

        target_cols = [
            'revenue_day_1', 'revenue_day_3', 'revenue_day_7', 'attendance_day_7',
            'revenue_week_1', 'revenue_week_2', 'revenue_week_4', 'attendance_week_1',
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_1', 'member_count_month_3', 'retention_rate_month_3'
        ]

        exclude_cols = target_cols + ['studio_id', 'date', 'split',
                                      'studio_location', 'studio_size_tier', 'studio_price_tier']

        self.all_features = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Total available features: {len(self.all_features)}")

        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'validation'].copy()
        test_df = df[df['split'] == 'test'].copy()

        train_val_df = pd.concat([train_df, val_df])

        X_train_val = train_val_df[self.all_features].values
        y_train_val = train_val_df[target_cols].values
        groups_train_val = train_val_df['studio_id'].values

        X_test = test_df[self.all_features].values
        y_test = test_df[target_cols].values
        groups_test = test_df['studio_id'].values

        logger.info(f"Train+Val: {len(X_train_val)} samples, Test: {len(X_test)} samples")

        return X_train_val, y_train_val, groups_train_val, X_test, y_test, groups_test

    def scale_features(self, X_train, X_test):
        """Scale features"""
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")

        start_time = time.time()

        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        logger.info(f"Parameters: n_estimators=500, max_depth=6, lr=0.03")

        self.model = MultiOutputRegressor(xgb_model)
        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")

    def cross_validate(self, X, y, groups, cv=5):
        """Perform grouped cross-validation"""
        logger.info(f"Performing {cv}-fold grouped cross-validation...")

        gkf = GroupKFold(n_splits=cv)

        r2_scores = []
        rmse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"  Fold {fold_idx + 1}/{cv}...")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            xgb_fold = xgb.XGBRegressor(
                n_estimators=300,  # Reduced for CV
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                objective='reg:squarederror',
                random_state=42 + fold_idx,
                n_jobs=-1
            )

            model_fold = MultiOutputRegressor(xgb_fold)
            model_fold.fit(X_train_fold, y_train_fold)

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

        logger.info(f"CV Results: R² = {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")

        return cv_results

    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")

        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = (time.time() - start_time) * 1000

        logger.info(f"Inference: {inference_time:.2f}ms total, {inference_time/len(X_test):.4f}ms per sample")

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

        logger.info(f"Test: R²={overall_r2:.4f}, RMSE={overall_rmse:.2f}, MAE={overall_mae:.2f}")

        return test_results

    def calculate_business_metrics(self, X_test, y_test):
        """Calculate business-focused metrics"""
        logger.info("Calculating business metrics...")

        y_pred = self.model.predict(X_test)

        revenue_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        y_test_revenue = y_test[:, revenue_indices]
        y_pred_revenue = y_pred[:, revenue_indices]

        pct_errors = np.abs((y_pred_revenue - y_test_revenue) / (y_test_revenue + 1e-10)) * 100

        within_5_pct = (pct_errors <= 5).sum() / pct_errors.size * 100
        within_10_pct = (pct_errors <= 10).sum() / pct_errors.size * 100

        direction_actual = np.sign(y_test[:, 2] - y_test[:, 0])
        direction_pred = np.sign(y_pred[:, 2] - y_pred[:, 0])
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100

        business_score = 0.5 * within_5_pct + 0.3 * within_10_pct + 0.2 * directional_accuracy

        business_metrics = {
            'predictions_within_5pct': round(within_5_pct, 2),
            'predictions_within_10pct': round(within_10_pct, 2),
            'directional_accuracy': round(directional_accuracy, 2),
            'business_impact_score': round(business_score, 2)
        }

        logger.info(f"Business: Within 5%={within_5_pct:.1f}%, Within 10%={within_10_pct:.1f}%, Score={business_score:.1f}")

        return business_metrics

    def save_artifacts(self, version='2.3.0'):
        """Save model artifacts"""
        logger.info(f"Saving XGBoost model v{version}...")

        output_dir = Path('data/models/v2.3')
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / 'xgboost_model_v2.3.0.pkl')
        joblib.dump(self.scaler, output_dir / 'xgboost_scaler_v2.3.0.pkl')
        joblib.dump(self.all_features, output_dir / 'xgboost_features_v2.3.0.pkl')

        metadata = {
            'version': '2.3.0',
            'model_type': 'xgboost',
            'algorithm': 'XGBRegressor',
            'hyperparameters': {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0
            },
            'n_features': len(self.all_features),
            'n_targets': len(self.target_names),
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': round(self.training_time, 2)
        }

        with open(output_dir / 'xgboost_metadata_v2.3.0.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved to {output_dir}")


def main():
    print("\n" + "="*80)
    print("MODEL TRAINING v2.3.0 - XGBOOST (DAILY DATA)")
    print("="*80 + "\n")

    trainer = XGBoostTrainer()

    print("Loading data...")
    X_train_val, y_train_val, groups_train_val, X_test, y_test, groups_test = trainer.load_and_prepare_data()

    print("\nScaling features...")
    X_train_val_scaled, X_test_scaled = trainer.scale_features(X_train_val, X_test)

    print("\nTraining XGBoost...")
    print("-" * 80)
    trainer.train_model(X_train_val_scaled, y_train_val)
    print("-" * 80)

    print("\nCross-validation...")
    print("-" * 80)
    cv_results = trainer.cross_validate(X_train_val_scaled, y_train_val, groups_train_val, cv=5)
    print("-" * 80)

    print("\nTesting...")
    print("-" * 80)
    test_results = trainer.evaluate_model(X_test_scaled, y_test)
    business_metrics = trainer.calculate_business_metrics(X_test_scaled, y_test)
    print("-" * 80)

    results = {
        'model_type': 'xgboost',
        'cv_results': cv_results,
        'test_results': test_results,
        'business_metrics': business_metrics,
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(X_train_val),
        'test_samples': len(X_test),
        'training_time_seconds': round(trainer.training_time, 2)
    }

    output_path = Path('reports/audit/model_results_v2.3.0_xgboost.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    trainer.save_artifacts()

    print("\n" + "="*80)
    print("XGBOOST TRAINING COMPLETE")
    print("="*80)
    print(f"Test R²: {test_results['overall_r2']:.4f}")
    print(f"Test RMSE: {test_results['overall_rmse']:.2f}")
    print(f"Training time: {trainer.training_time:.2f}s")
    print(f"Business score: {business_metrics['business_impact_score']:.1f}/100")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
