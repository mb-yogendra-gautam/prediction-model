"""
Phase 2: Hyperparameter Optimization

Finds optimal hyperparameters for the augmented dataset.
Tests different alpha values and feature counts.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from pathlib import Path
import logging
import json
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterSearch:
    """Search for optimal hyperparameters"""
    
    def __init__(self):
        with open('config/model_config_v2.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.target_cols = self.config['data']['target_columns']
        self.best_params = {}
        self.results = []
        
    def load_data(self):
        """Load augmented data"""
        logger.info("Loading augmented data...")
        
        try:
            df = pd.read_csv('data/processed/studio_data_augmented.csv')
        except FileNotFoundError:
            logger.warning("Augmented data not found, using original")
            df = pd.read_csv('data/processed/studio_data_engineered.csv')
        
        exclude_cols = self.target_cols + ['month_year', 'split', 'year_index']
        self.all_features = [col for col in df.columns if col not in exclude_cols]
        
        # Combine train and val
        train_val_df = df[df['split'].isin(['train', 'validation'])]
        
        X = train_val_df[self.all_features].values
        y = train_val_df[self.target_cols].values
        
        logger.info(f"Loaded {len(X)} samples, {len(self.all_features)} features")
        
        return X, y
    
    def search_feature_count(self, X, y):
        """Find optimal number of features"""
        logger.info("\n" + "="*60)
        logger.info("SEARCHING FOR OPTIMAL FEATURE COUNT")
        logger.info("="*60)
        
        feature_counts = [8, 10, 12, 15, 18, 20]
        results = []
        
        for k in feature_counts:
            logger.info(f"\nTesting k={k} features...")
            
            # Select features
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y[:, 0])
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            # Train Ridge with default alpha
            ridge = Ridge(alpha=10.0, random_state=42)
            model = MultiOutputRegressor(ridge)
            
            # Cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = cross_val_score(model, X_scaled, y, cv=kfold, 
                                       scoring='r2', n_jobs=-1)
            rmse_scores = -cross_val_score(model, X_scaled, y, cv=kfold,
                                          scoring='neg_root_mean_squared_error',
                                          n_jobs=-1)
            
            result = {
                'k_features': k,
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std()
            }
            results.append(result)
            
            logger.info(f"  R² = {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
            logger.info(f"  RMSE = {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
        
        # Find best
        best = max(results, key=lambda x: x['r2_mean'])
        
        logger.info(f"\n✓ Best k_features: {best['k_features']}")
        logger.info(f"  R² = {best['r2_mean']:.4f} (+/- {best['r2_std']:.4f})")
        
        return best['k_features'], results
    
    def search_ridge_alpha(self, X, y, k_features):
        """Find optimal Ridge alpha"""
        logger.info("\n" + "="*60)
        logger.info("SEARCHING FOR OPTIMAL RIDGE ALPHA")
        logger.info("="*60)
        
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        results = []
        
        # Select features
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X, y[:, 0])
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        for alpha in alphas:
            logger.info(f"\nTesting alpha={alpha}...")
            
            ridge = Ridge(alpha=alpha, random_state=42)
            model = MultiOutputRegressor(ridge)
            
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = cross_val_score(model, X_scaled, y, cv=kfold,
                                       scoring='r2', n_jobs=-1)
            rmse_scores = -cross_val_score(model, X_scaled, y, cv=kfold,
                                          scoring='neg_root_mean_squared_error',
                                          n_jobs=-1)
            
            result = {
                'alpha': alpha,
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std()
            }
            results.append(result)
            
            logger.info(f"  R² = {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
            logger.info(f"  RMSE = {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
        
        # Find best
        best = max(results, key=lambda x: x['r2_mean'])
        
        logger.info(f"\n✓ Best alpha: {best['alpha']}")
        logger.info(f"  R² = {best['r2_mean']:.4f} (+/- {best['r2_std']:.4f})")
        
        return best['alpha'], results
    
    def search_elastic_net_params(self, X, y, k_features):
        """Find optimal ElasticNet alpha and l1_ratio"""
        logger.info("\n" + "="*60)
        logger.info("SEARCHING FOR OPTIMAL ELASTIC NET PARAMS")
        logger.info("="*60)
        
        alphas = [0.1, 0.5, 1.0, 5.0, 10.0]
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        # Select features
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X, y[:, 0])
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        best_r2 = -np.inf
        best_params_elastic = {}
        
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, 
                                    random_state=42, max_iter=5000)
                model = MultiOutputRegressor(elastic)
                
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                r2_scores = cross_val_score(model, X_scaled, y, cv=kfold,
                                           scoring='r2', n_jobs=-1)
                
                r2_mean = r2_scores.mean()
                
                result = {
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'r2_mean': r2_mean,
                    'r2_std': r2_scores.std()
                }
                results.append(result)
                
                if r2_mean > best_r2:
                    best_r2 = r2_mean
                    best_params_elastic = result
        
        logger.info(f"\n✓ Best ElasticNet params:")
        logger.info(f"  alpha = {best_params_elastic['alpha']}")
        logger.info(f"  l1_ratio = {best_params_elastic['l1_ratio']}")
        logger.info(f"  R² = {best_params_elastic['r2_mean']:.4f} (+/- {best_params_elastic['r2_std']:.4f})")
        
        return best_params_elastic, results
    
    def run_full_search(self):
        """Run complete hyperparameter search"""
        print("\n" + "="*80)
        print("PHASE 2: HYPERPARAMETER OPTIMIZATION")
        print("="*80 + "\n")
        
        # Load data
        X, y = self.load_data()
        
        # Search 1: Feature count
        best_k, feature_results = self.search_feature_count(X, y)
        
        # Search 2: Ridge alpha
        best_ridge_alpha, ridge_results = self.search_ridge_alpha(X, y, best_k)
        
        # Search 3: ElasticNet params
        best_elastic_params, elastic_results = self.search_elastic_net_params(X, y, best_k)
        
        # Compile results
        self.best_params = {
            'k_features': best_k,
            'ridge_alpha': best_ridge_alpha,
            'elastic_net_alpha': best_elastic_params['alpha'],
            'elastic_net_l1_ratio': best_elastic_params['l1_ratio']
        }
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'best_params': self.best_params,
            'feature_count_search': feature_results,
            'ridge_alpha_search': ridge_results,
            'elastic_net_search': elastic_results
        }
        
        # Save results
        output_path = Path('reports/audit/hyperparameter_search_v2.1.0.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("HYPERPARAMETER SEARCH COMPLETE")
        print("="*80)
        print("\nBest Hyperparameters Found:")
        print(f"  Feature count: {self.best_params['k_features']}")
        print(f"  Ridge alpha: {self.best_params['ridge_alpha']}")
        print(f"  ElasticNet alpha: {self.best_params['elastic_net_alpha']}")
        print(f"  ElasticNet l1_ratio: {self.best_params['elastic_net_l1_ratio']}")
        
        print("\nNext Steps:")
        print("  1. Update model_config_v2.yaml with these parameters")
        print("  2. Retrain model with optimal hyperparameters")
        print("  3. Compare with default hyperparameters")
        print("\n" + "="*80 + "\n")
        
        return self.best_params


def main():
    search = HyperparameterSearch()
    best_params = search.run_full_search()
    
    # Provide code to update config
    print("\nTo use these hyperparameters, update train_improved_model_v2.1.py:")
    print("\ntrainer.train_models(")
    print(f"    X_train_val_scaled, y_train_val,")
    print(f"    alpha_ridge={best_params['ridge_alpha']},")
    print(f"    alpha_elastic={best_params['elastic_net_alpha']}")
    print(")")
    print("\nAnd in select_features():")
    print(f"trainer.select_features(X_train_val, y_train_val, k={best_params['k_features']})")


if __name__ == "__main__":
    main()

