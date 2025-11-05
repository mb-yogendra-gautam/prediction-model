"""
Improved Model Training Script
Addresses overfitting issues with simplified, regularized models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import yaml
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


class ImprovedModelTrainer:
    """Simplified, regularized model trainer optimized for small datasets"""
    
    def __init__(self, config_path='config/model_config_v2.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = []
        self.target_names = [
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 3', 'Retention Month 3'
        ]
        
    def load_and_prepare_data(self, data_path='data/processed/studio_data_engineered.csv'):
        """Load data with proper temporal split"""
        logger.info("Loading data...")
        df = pd.read_csv(data_path)
        
        # Define targets and features
        target_cols = self.config['data']['target_columns']
        exclude_cols = target_cols + ['month_year', 'split', 'year_index']
        
        self.all_features = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Total available features: {len(self.all_features)}")
        
        # Split data
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'validation']
        test_df = df[df['split'] == 'test']
        
        # Combine train and val for cross-validation
        train_val_df = pd.concat([train_df, val_df])
        
        X_train_val = train_val_df[self.all_features].values
        y_train_val = train_val_df[target_cols].values
        
        X_test = test_df[self.all_features].values
        y_test = test_df[target_cols].values
        
        logger.info(f"Train+Val: {len(X_train_val)} samples")
        logger.info(f"Test: {len(X_test)} samples")
        
        return X_train_val, y_train_val, X_test, y_test
    
    def select_features(self, X, y, k=15):
        """Select top K most important features"""
        logger.info(f"Selecting top {k} features...")
        
        # Use SelectKBest with F-regression score
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        self.feature_selector.fit(X, y[:, 0])  # Use first target for selection
        
        # Get selected feature indices
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.all_features[i] for i in selected_indices]
        
        # Get feature scores
        scores = self.feature_selector.scores_
        feature_importance = pd.DataFrame({
            'Feature': self.all_features,
            'Score': scores
        }).sort_values('Score', ascending=False)
        
        logger.info(f"Selected features: {self.selected_features}")
        
        # Transform data
        X_selected = self.feature_selector.transform(X)
        
        return X_selected, feature_importance
    
    def scale_features(self, X_train, X_test):
        """Scale features"""
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train):
        """Train simplified, regularized models"""
        logger.info("Training models...")
        
        # Model 1: Ridge Regression (L2 regularization)
        logger.info("Training Ridge Regression...")
        ridge = Ridge(alpha=10.0, random_state=42)
        self.models['ridge'] = MultiOutputRegressor(ridge)
        self.models['ridge'].fit(X_train, y_train)
        
        # Model 2: Lasso Regression (L1 regularization + feature selection)
        logger.info("Training Lasso Regression...")
        lasso = Lasso(alpha=1.0, random_state=42, max_iter=5000)
        self.models['lasso'] = MultiOutputRegressor(lasso)
        self.models['lasso'].fit(X_train, y_train)
        
        # Model 3: Elastic Net (L1 + L2 regularization)
        logger.info("Training Elastic Net...")
        elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=5000)
        self.models['elastic_net'] = MultiOutputRegressor(elastic)
        self.models['elastic_net'].fit(X_train, y_train)
        
        # Model 4: Simplified Gradient Boosting (optional)
        logger.info("Training Simplified GBM...")
        gbm = GradientBoostingRegressor(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.7,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.models['gbm'] = MultiOutputRegressor(gbm)
        self.models['gbm'].fit(X_train, y_train)
        
        logger.info(f"Trained {len(self.models)} models")
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation for all models"""
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"CV for {model_name}...")
            
            # Calculate R² scores (using full multi-output)
            r2_scores = cross_val_score(model, X, y, cv=kfold, 
                                        scoring='r2', n_jobs=-1)
            
            # Calculate RMSE scores (using full multi-output)
            rmse_scores = -cross_val_score(model, X, y, cv=kfold,
                                           scoring='neg_root_mean_squared_error', 
                                           n_jobs=-1)
            
            cv_results[model_name] = {
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std()
            }
            
            logger.info(f"  R² = {r2_scores.mean():.3f} (+/- {r2_scores.std():.3f})")
            logger.info(f"  RMSE = {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
        
        return cv_results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        logger.info("Evaluating models on test set...")
        
        test_results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            metrics = {}
            for i, target_name in enumerate(self.target_names):
                rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                r2 = r2_score(y_test[:, i], y_pred[:, i])
                mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i])) * 100
                
                metrics[target_name] = {
                    'RMSE': round(rmse, 2),
                    'MAE': round(mae, 2),
                    'R2': round(r2, 4),
                    'MAPE': round(mape, 2)
                }
            
            # Overall metrics
            overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
            
            test_results[model_name] = {
                'metrics_by_target': metrics,
                'overall_rmse': round(overall_rmse, 2),
                'overall_r2': round(overall_r2, 4)
            }
            
            logger.info(f"{model_name}:")
            logger.info(f"  Overall R² = {overall_r2:.4f}")
            logger.info(f"  Overall RMSE = {overall_rmse:.2f}")
        
        return test_results
    
    def select_best_model(self, test_results):
        """Select best model based on test performance"""
        best_model_name = min(test_results.keys(), 
                             key=lambda k: test_results[k]['overall_rmse'])
        
        logger.info(f"Best model: {best_model_name}")
        return best_model_name
    
    def visualize_results(self, X_test, y_test, save_dir='reports/figures'):
        """Create comparison visualizations"""
        logger.info("Creating visualizations...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        model_names = list(self.models.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for target_idx in range(3):  # First 3 revenue targets
            ax = axes[target_idx]
            
            for i, (model_name, model) in enumerate(self.models.items()):
                y_pred = model.predict(X_test)
                
                ax.scatter(y_test[:, target_idx], y_pred[:, target_idx], 
                          alpha=0.6, label=model_name, color=colors[i], s=80)
            
            # Perfect prediction line
            min_val = y_test[:, target_idx].min()
            max_val = y_test[:, target_idx].max()
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Perfect')
            
            ax.set_xlabel('Actual', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.target_names[target_idx]}', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Model performance comparison
        ax = axes[3]
        test_rmses = []
        for model_name in model_names:
            y_pred = self.models[model_name].predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_rmses.append(rmse)
        
        bars = ax.bar(model_names, test_rmses, color=colors)
        ax.set_ylabel('Overall RMSE', fontsize=11, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, test_rmses):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison_v2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {save_dir / 'model_comparison_v2.png'}")
    
    def save_artifacts(self, best_model_name, version='2.0.0'):
        """Save model artifacts"""
        logger.info(f"Saving model v{version}...")
        
        output_dir = Path(self.config['model']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        best_model = self.models[best_model_name]
        joblib.dump(best_model, output_dir / f'best_model_v{version}.pkl')
        
        # Save all models
        joblib.dump(self.models, output_dir / f'all_models_v{version}.pkl')
        
        # Save preprocessing artifacts
        joblib.dump(self.scaler, output_dir / f'scaler_v{version}.pkl')
        joblib.dump(self.feature_selector, output_dir / f'feature_selector_v{version}.pkl')
        joblib.dump(self.selected_features, output_dir / f'selected_features_v{version}.pkl')
        
        # Save metadata
        metadata = {
            'version': version,
            'best_model': best_model_name,
            'n_features': len(self.selected_features),
            'selected_features': self.selected_features,
            'training_date': datetime.now().isoformat()
        }
        
        with open(output_dir / f'metadata_v{version}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {output_dir}")


def main():
    print("\n" + "="*80)
    print("IMPROVED MODEL TRAINING - Addressing Overfitting Issues")
    print("="*80 + "\n")
    
    # Initialize trainer
    print("Step 1: Initializing trainer...")
    trainer = ImprovedModelTrainer()
    
    # Load data
    print("\nStep 2: Loading and preparing data...")
    X_train_val, y_train_val, X_test, y_test = trainer.load_and_prepare_data()
    
    # Feature selection
    print("\nStep 3: Selecting most important features...")
    X_train_val_selected, feature_importance = trainer.select_features(
        X_train_val, y_train_val, k=15
    )
    X_test_selected = trainer.feature_selector.transform(X_test)
    
    # Save feature importance
    feature_importance.to_csv('reports/audit/feature_importance_v2.0.0.csv', index=False)
    print(f"[OK] Feature importance saved")
    
    # Scale features
    print("\nStep 4: Scaling features...")
    X_train_val_scaled, X_test_scaled = trainer.scale_features(
        X_train_val_selected, X_test_selected
    )
    
    # Train models
    print("\nStep 5: Training simplified models...")
    print("-" * 80)
    trainer.train_models(X_train_val_scaled, y_train_val)
    print("-" * 80)
    
    # Cross-validation
    print("\nStep 6: Performing cross-validation...")
    print("-" * 80)
    cv_results = trainer.cross_validate(X_train_val_scaled, y_train_val, cv=5)
    print("-" * 80)
    
    # Test evaluation
    print("\nStep 7: Evaluating on test set...")
    print("-" * 80)
    test_results = trainer.evaluate_models(X_test_scaled, y_test)
    print("-" * 80)
    
    # Select best model
    print("\nStep 8: Selecting best model...")
    best_model_name = trainer.select_best_model(test_results)
    
    # Save results
    print("\nStep 9: Saving results...")
    results = {
        'cv_results': cv_results,
        'test_results': test_results,
        'best_model': best_model_name,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('reports/audit/improved_model_results_v2.0.0.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    print("\nStep 10: Creating visualizations...")
    trainer.visualize_results(X_test_scaled, y_test)
    
    # Save artifacts
    print("\nStep 11: Saving model artifacts...")
    trainer.save_artifacts(best_model_name, version='2.0.0')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nKey Improvements Over v1.0.0:")
    print("  ✓ Simplified models (Ridge, Lasso, ElasticNet, GBM)")
    print("  ✓ Reduced features from 41 → 15")
    print("  ✓ 5-fold cross-validation for reliable evaluation")
    print("  ✓ Strong regularization to prevent overfitting")
    print(f"\nBest Model: {best_model_name}")
    print(f"Test R²: {test_results[best_model_name]['overall_r2']:.4f}")
    print(f"Test RMSE: {test_results[best_model_name]['overall_rmse']:.2f}")
    
    print("\nNext Steps:")
    print("  1. Compare with v1.0.0 performance")
    print("  2. If still overfitting, try data augmentation")
    print("  3. Collect more real data if possible")
    print("  4. Review feature engineering for data leakage")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

