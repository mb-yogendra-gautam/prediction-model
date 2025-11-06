"""
Model Registry for Loading and Managing Trained Models
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Centralized model registry for loading model artifacts"""

    def __init__(self, base_dir: str = "data/models"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Model directory not found: {base_dir}")

    def load_model(self, version: str = "2.2.0") -> Dict[str, Any]:
        """
        Load model artifacts for a specific version

        Args:
            version: Model version to load (default: 2.2.0)

        Returns:
            Dictionary containing model artifacts:
                - model: The trained model
                - scaler: Feature scaler
                - selected_features: List of selected feature names
                - feature_selector: Feature selector object
                - shap_background: Background data for SHAP (optional)
                - metadata: Model metadata
        """
        logger.info(f"Loading model version {version}")

        try:
            # Load model artifacts
            model_path = self.base_dir / f'best_model_v{version}.pkl'
            scaler_path = self.base_dir / f'scaler_v{version}.pkl'
            features_path = self.base_dir / f'selected_features_v{version}.pkl'
            selector_path = self.base_dir / f'feature_selector_v{version}.pkl'
            metadata_path = self.base_dir / f'metadata_v{version}.json'
            shap_background_path = self.base_dir / f'shap_background_v{version}.pkl'

            # Check if all required files exist
            for path in [model_path, scaler_path, features_path, selector_path, metadata_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Missing model artifact: {path}")

            # Load artifacts
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            selected_features = joblib.load(features_path)
            feature_selector = joblib.load(selector_path)

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Load SHAP background data if available
            shap_background = None
            if shap_background_path.exists():
                shap_background = joblib.load(shap_background_path)
                logger.info(f"Loaded SHAP background data: {shap_background.shape}")
            else:
                logger.warning(f"SHAP background data not found at {shap_background_path}")

            logger.info(f"Successfully loaded model version {version}")
            logger.info(f"Model type: {metadata.get('best_model', 'unknown')}")
            logger.info(f"Number of features: {metadata.get('n_features', 'unknown')}")

            return {
                'model': model,
                'scaler': scaler,
                'selected_features': selected_features,
                'feature_selector': feature_selector,
                'shap_background': shap_background,
                'metadata': metadata,
                'version': version
            }

        except Exception as e:
            logger.error(f"Error loading model version {version}: {e}")
            raise

    def load_v2_3_model(self, model_type: str, version: str = "2.3.0") -> Dict[str, Any]:
        """
        Load v2.3.0 daily data models (xgboost, lightgbm, neural_network, ridge)

        Args:
            model_type: Model type ('xgboost', 'lightgbm', 'neural_network', 'ridge')
            version: Model version (default: 2.3.0)
        
        Returns:
            Dictionary containing model artifacts:
                - model: The trained model
                - scaler: Feature scaler
                - features: List of feature names  
                - metadata: Model metadata
                - model_type: Type of model
                - version: Model version
        """
        logger.info(f"Loading {model_type} model version {version}")
        
        try:
            # v2.3.0 models are in v2.3 subdirectory
            v2_3_dir = self.base_dir / 'v2.3'
            
            if not v2_3_dir.exists():
                raise FileNotFoundError(f"v2.3 model directory not found: {v2_3_dir}")
            
            # Construct file paths for v2.3 models
            model_path = v2_3_dir / f'{model_type}_model_v{version}.pkl'
            scaler_path = v2_3_dir / f'{model_type}_scaler_v{version}.pkl'
            features_path = v2_3_dir / f'{model_type}_features_v{version}.pkl'
            metadata_path = v2_3_dir / f'{model_type}_metadata_v{version}.json'
            
            # Check if all required files exist
            for path in [model_path, scaler_path, features_path, metadata_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Missing model artifact: {path}")
            
            # Load artifacts
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            features = joblib.load(features_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Successfully loaded {model_type} v{version}")
            logger.info(f"Model type: {metadata.get('model_type', 'unknown')}")
            logger.info(f"Number of features: {metadata.get('n_features', 'unknown')}")
            logger.info(f"Number of targets: {metadata.get('n_targets', 'unknown')}")
            
            return {
                'model': model,
                'scaler': scaler,
                'features': features,
                'selected_features': features,  # Compatibility with v2.2.0
                'metadata': metadata,
                'model_type': model_type,
                'version': version
            }
        
        except Exception as e:
            logger.error(f"Error loading {model_type} model version {version}: {e}")
            raise
    
    def get_model_key(self, model_type: str, version: str) -> str:
        """
        Generate unique cache key for model
        
        Args:
            model_type: Model type ('ridge', 'xgboost', 'lightgbm', 'neural_network')
            version: Model version (e.g., '2.2.0', '2.3.0')
        
        Returns:
            Unique key string
        """
        return f"{model_type}_v{version}"
    
    def list_available_models(self) -> list:
        """
        List all available models with their metadata
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        # Check for v2.2.0 models
        for metadata_file in self.base_dir.glob('metadata_v2.*.json'):
            version = metadata_file.stem.replace('metadata_v', '')
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                models.append({
                    'model_type': 'ridge',  # v2.2.0 is Ridge regression
                    'version': version,
                    'training_date': metadata.get('training_date'),
                    'best_model': metadata.get('best_model'),
                    'n_features': metadata.get('n_features'),
                    'n_targets': 5,  # v2.2.0 has 5 targets
                    'prediction_horizons': ['monthly']
                })
            except Exception as e:
                logger.warning(f"Error reading metadata from {metadata_file}: {e}")
        
        # Check for v2.3.0 models
        v2_3_dir = self.base_dir / 'v2.3'
        if v2_3_dir.exists():
            for model_type in ['xgboost', 'lightgbm', 'neural_network', 'ridge']:
                metadata_file = v2_3_dir / f'{model_type}_metadata_v2.3.0.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            'model_type': model_type,
                            'version': '2.3.0',
                            'training_date': metadata.get('training_date'),
                            'algorithm': metadata.get('algorithm'),
                            'n_features': metadata.get('n_features'),
                            'n_targets': metadata.get('n_targets', 14),
                            'prediction_horizons': ['daily', 'weekly', 'monthly']
                        })
                    except Exception as e:
                        logger.warning(f"Error reading metadata from {metadata_file}: {e}")
        
        return models
    
    def list_available_versions(self):
        """List all available model versions"""
        versions = []
        for metadata_file in self.base_dir.glob('metadata_v*.json'):
            version = metadata_file.stem.replace('metadata_v', '')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            versions.append({
                'version': version,
                'training_date': metadata.get('training_date'),
                'best_model': metadata.get('best_model'),
                'n_features': metadata.get('n_features')
            })
        return sorted(versions, key=lambda x: x['version'], reverse=True)

